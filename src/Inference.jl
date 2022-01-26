include("./src/ExperimentHelper.jl")
include("./src/Utils.jl")
include("./src/Datasets/Dataset.jl")
include("./src/Datasets/SequenceDataset.jl")
include("./src/Datasets/SyntheticDataset.jl")
include("./src/Models/EditCNN.jl")
include("./src/Utils/Faiss.jl")

# include("./args.jl")
# include("./Utils.jl")
# include("./Datasets/SyntheticDataset.jl")
# include("./Model.jl")

using BenchmarkTools
using Random
using Statistics

using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy, params, update!, Chain
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using Distances
using LinearAlgebra
using Zygote
using Zygote: gradient, @ignore
using Parameters

using JLD2: @load
using PythonCall

using ..Dataset
using ..ExperimentHelper: ExperimentParams
using ..EditCNN
using ..Utils
using ..Faiss

try
    using CUDA
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end

np = pyimport("numpy")


ROOT_DIR = "/home/jon/JuliaProjects/NeuralOverlap"
# EXPERIMENT_DIR="/home/jon/JuliaProjects/NeuralOverlap/data/experiments/2022-01-20T08:03:13.264_newModelMeanNorm"
EXPERIMENT_DIR = "/home/jon/JuliaProjects/NeuralOverlap/data/experiments/2022-01-25T14:53:55.090"
INFERENCE_FASTQ = "/home/jon/JuliaProjects/NeuralOverlap/data_fetch/covid_test.fa_R1.fastq"
# INFERENCE_FASTQ = "/home/jon/JuliaProjects/NeuralOverlap/data_fetch/phix174_train.fa_R1.fastq"
MODEL_LOAD_PATH = "/home/jon/JuliaProjects/NeuralOverlap/data/experiments/2022-01-25T08:22:12.709/saved_models/epoch_5_mean_abs_error_2.6985315187913312.jld2"


function inference_experiment(args, embedding_model, calibration_model; use_pipe_cleaner::Bool=false, quantize_faiss_index::Bool=false, max_inference_samples::Int64=Int(1e4), use_gpu::Bool=false)

    function get_inference_sequences()::Array
        # Eval Dataset
        if args.USE_SYNTHETIC_DATA == true
            test_sequences = SyntheticDataset.generate_synethetic_sequences(
                max_inference_samples, args.MAX_STRING_LENGTH,
                args.MAX_STRING_LENGTH, args.ALPHABET,ratio_of_random=args.RATIO_OF_RANDOM_SAMPLES,
                similarity_min=args.SIMILARITY_MIN, similarity_max=args.SIMILARITY_MAX
                )
        elseif args.USE_SEQUENCE_DATA == true
            test_sequences = SequenceDataset.read_sequence_data(max_inference_samples, args.MAX_STRING_LENGTH, fastq_filepath=INFERENCE_FASTQ)
        else
            throw("Must provide type of dataset")
        end
        # Create a map if sequence ID to sequence
        return test_sequences
    end


    function batch_sequence_producer(args, chnl::Channel, sequences::Array{String})
        i = 1
        j = args.BSIZE
        while j < length(sequences)
            batch = sequences[i:j]
            i += args.BSIZE
            j += args.BSIZE
            put!(chnl, batch)
        end
        # Last batch
        put!(chnl, sequences[i:end])
    end


    function _update_predicted_nn_map(embeddings_array::Array, index; k::Int64=100)

        if args.DISTANCE_MATRIX_NORM_METHOD == "max"
            denorm_factor = args.MAX_STRING_LENGTH
        elseif args.DISTANCE_MATRIX_NORM_METHOD == "mean"
            # TODO: Use the mean here
            denorm_factor = args.MAX_STRING_LENGTH
        else
            throw("Invalid distance matrix norm factor")
        end

        py_batch_embeddings = np.concatenate(pylist(embeddings_array), axis=0)
        distances, nnIds = index.search(py_batch_embeddings, pyint(k))

        # Note: Distances in L2 space are squared
        jl_distances = convert(Array, PyArray(distances))
        # Note that the first index may be negative due to approximation/rounding errors
        jl_distances[jl_distances .< 0] .= 0
        D = sqrt.(jl_distances) * denorm_factor

        ids = convert(Array, PyArray(nnIds))
        return ids, D
    end

    function get_approximate_nn_overlap(faiss_index, n_sequences::Int64; k=1000)

        # TODO: Make this part of the model that's fit on the training dataset
        # (e.g. for mean distance)
        denorm_factor = args.MAX_STRING_LENGTH

        batch_array = []
        time_finding_approx_nns = @elapsed begin
            predicted_nn_map = Dict()
            reconstructVectorID = 0
            faiss_vector_id_pointer = 0
            for _ in 0:n_sequences - 1
                embedding = faiss_index.reconstruct(reconstructVectorID).reshape(1, -1)
                push!(batch_array, embedding)
                reconstructVectorID += 1

                if length(batch_array) == args.BSIZE
                    ids, distances = _update_predicted_nn_map(batch_array, faiss_index, k=k)

                    for idIdx in 1:size(ids)[1]
                        predicted_nn_map[faiss_vector_id_pointer] = Dict("topKNN" => ids[idIdx, 1:end], "distances" => distances[idIdx, 1:end])
                        faiss_vector_id_pointer += 1
                    end
                    batch_array = []
                end
            end

            if length(batch_array) > 0
                ids, distances = _update_predicted_nn_map(batch_array, faiss_index, k=k)
                for _ in 1:length(batch_array)
                    predicted_nn_map[faiss_vector_id_pointer] = Dict("topKNN" => ids, "distances" => distances)
                    faiss_vector_id_pointer += 1
                end
                batch_array = []
            end
        end
        @info("Time to find approximate NNs is %s", time_finding_approx_nns)
        return predicted_nn_map
    end


    # trainedfaiss_clusterer, faiss_indexWithEmbeddings = create_knn_index(args)


    function get_true_nn_overlap(inference_sequences::Array{String};k=1000)
        # Get the pairwise sequence distance array
        timeFindingTrueNNs = @elapsed begin
            true_nn_map = Dict()
            _, truepairwise_distances = Utils.pairwise_hamming_distance(inference_sequences)

            for i in 1:size(truepairwise_distances)[1]
                nns = sortperm(truepairwise_distances[i, 1:end])[1:k]
                distances = truepairwise_distances[i, 1:end][nns]

                # Use FAISS 0-based indexing
                nns = nns .- 1
                true_nn_map[i - 1] = Dict("topKNN" => nns, "distances" => distances)  # Index the same as FAISS indexed
            end
        end
        @info("Time to find true NNs is %s", timeFindingTrueNNs)
        return true_nn_map
    end

    function execute()
        embedding_model = embedding_model |> DEVICE

        trainmode!(embedding_model, false)

        inference_sequences = get_inference_sequences()
        batch_sequence_iterator = Channel( (channel) -> batch_sequence_producer(args, channel, inference_sequences), 1)
        faiss_index = Faiss.create_embedding_index(args, embedding_model, batch_sequence_iterator, quantize=quantize_faiss_index)

        # After training and adding vectors, make a direct map if the index allows for it
        if hasproperty(faiss_index, :make_direct_map)
            faiss_index.make_direct_map()
        end

        numNeighbours = 1000
        predicted_nn_map = get_approximate_nn_overlap(faiss_index, length(inference_sequences), k=numNeighbours)

        true_nn_map = get_true_nn_overlap(inference_sequences, k=numNeighbours)

        epoch_recall_dict = Utils.get_top_t_recall_at_k(
            ".", "test", length(true_nn_map), numNN=1000,
            kStart=1, kEnd=1001, kStep=100; startIndex=2, trueid_seq_data_map=true_nn_map, predicted_id_seq_data_map=predicted_nn_map
        )

        return predicted_nn_map, true_nn_map, epoch_recall_dict, faiss_index
    end
    () -> (execute;get_inference_sequences;batch_sequence_producer;_update_predicted_nn_map;get_approximate_nn_overlap;get_true_nn_overlap;faiss)
end



@load joinpath(EXPERIMENT_DIR, "args.jld2") args


@load joinpath(ROOT_DIR, Utils.get_best_model_path(args.MODEL_SAVE_DIR)) embedding_model distance_calibration_model


inference_exp = inference_experiment(args, embedding_model, distance_calibration_model)
predicted_nn_map, true_nn_map, epoch_recall_dict, faiss_index = inference_exp.execute()
