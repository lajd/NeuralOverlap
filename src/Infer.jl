include("./src/ExperimentHelper.jl")
include("./src/Utils.jl")
include("./src/Datasets/Dataset.jl")
include("./src/Datasets/SequenceDataset.jl")
include("./src/Datasets/SyntheticDataset.jl")
include("./src/Model.jl")


# include("./args.jl")
# include("./Utils.jl")
# include("./Datasets/SyntheticDataset.jl")
# include("./Model.jl")

using BenchmarkTools
using Random
using Statistics

using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy, params, update!
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using Distances
using LinearAlgebra
using Zygote
using Zygote: gradient, @ignore

using JLD2: @load
using PythonCall

using ..Dataset
using ..ExperimentHelper: ExperimentParams
using ..Model
using ..Utils

faiss = pyimport("faiss")
np = pyimport("numpy")


try
    using CUDA
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end

ExperimentParamsType = Union{ExperimentParams, ExperimentHelper.ExperimentParams}

# EXPERIMENT_DIR="/home/jon/JuliaProjects/NeuralOverlap/data/experiments/2022-01-20T08:03:13.264_newModelMeanNorm"
EXPERIMENT_DIR="/home/jon/JuliaProjects/NeuralOverlap/data/experiments/2022-01-23T21:26:14.086"
INFERENCE_FASTQ = "/home/jon/JuliaProjects/NeuralOverlap/data_fetch/covid_test.fa_R1.fastq"
# INFERENCE_FASTQ = "/home/jon/JuliaProjects/NeuralOverlap/data_fetch/phix174_train.fa_R1.fastq"


function get_inference_sequences(max_samples::Int64=Int(1e4))::Array
    # Eval Dataset
    if args.USE_SYNTHETIC_DATA == true
        test_sequences = SyntheticDataset.generate_synethetic_sequences(
            max_samples, args.MAX_STRING_LENGTH,
            args.MAX_STRING_LENGTH, args.ALPHABET,ratio_of_random=args.RATIO_OF_RANDOM_SAMPLES,
            similarity_min=args.SIMILARITY_MIN, similarity_max=args.SIMILARITY_MAX
            )
    elseif args.USE_SEQUENCE_DATA == true
        test_sequences = SequenceDataset.read_sequence_data(max_samples, args.MAX_STRING_LENGTH, fastq_filepath=INFERENCE_FASTQ)
    else
        throw("Must provide type of dataset")
    end
    # Create a map if sequence ID to sequence
    return test_sequences
end


function batch_sequence_producer(argsExperimentParamsType, chnl, sequences::Array{String})
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



function addEmbeddingsToKMeans!(argsExperimentParamsType, embeddings::Array, clusterer, index; do_train_index::Bool=false)
    # Size (128, 512*N)
    training_vector_matrix = cat(embeddings..., dims=2) |> Flux.cpu
    convert(Array, training_vector_matrix)
    @assert size(training_vector_matrix)[1] == args.EMBEDDING_DIM size(training_vector_matrix)[1]

    vs_ = Py(convert(AbstractMatrix{Float32}, training_vector_matrix)').__array__()

    nEmbeddings = size(training_vector_matrix)[2]
    nEmbeddings = size(training_vector_matrix)[2]

    if do_train_index == true
        @info("Training Faiss kmeans index on %s embeddings", nEmbeddings)
        timeSpentTrainingKMeans = @elapsed begin
            clusterer.train(vs_, index)
            clusterer.post_process_centroids()
            npCentroids = faiss.vector_float_to_array(clusterer.centroids)
            jlCentroids = pyconvert(Vector{Float32}, faiss.vector_float_to_array(clusterer.centroids))
            k = pyconvert(Int, clusterer.k)
            clusterer.postProcessedCentroids = reshape(jlCentroids, (k, args.EMBEDDING_DIM))
        end
        @info("Finished training Faiss kmeans index on %s embeddings in %ss", nEmbeddings, timeSpentTrainingKMeans)
    end
    index.add(vs_)
    @info("Added %s vectors to Faiss clusterer. Faiss index has %s embeddings total.", nEmbeddings, index.ntotal)
end


function get_faiss_index(argsExperimentParamsType; quantize::Bool=false)
    nlist = 100
    m = 8
    d = args.EMBEDDING_DIM
    if quantize == true
        quantizer = faiss.IndexFlatL2(d)  # this remains the same
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    else
        index = faiss.IndexFlatL2(d)
    end
    return index
end

function createKNNIndex(argsExperimentParamsType)

    # TODO: Get this from args
    faiss_train_size = 5000

    estimated_genome_length = 5000

    toleranceScaleFactor = 1.3

#     if DEVICE == Flux.gpu
#         faiss_index = faiss.GpuIndexFlatL2(128)
#     else
#         faiss_index = faiss.IndexFlatL2(128)
#     end
    faiss_index = faiss.IndexFlatL2(128)

    estimated_num_true_reads = Int(ceil(estimated_genome_length/args.MAX_STRING_LENGTH))
    nClusters = 20 #estimated_num_true_reads * toleranceScaleFactor
    nRedo = 10
    maxIterations = 50
    expected_coverage = 100

    faiss_clusterer = faiss.Clustering(args.EMBEDDING_DIM, nClusters)
    faiss_clusterer.niter = maxIterations
    faiss_clusterer.nredo = nRedo
    # otherwise the kmeans implementation sub-samples the training set
    faiss_clusterer.max_points_per_centroid = expected_coverage

    kmeans_centroids_extracted = false

    training_vectors = []

    add_batch_size = 200

    map_faiss_id_to_sequence_id = Dict()

    LOG_EVERY = 10000
    @info("Accumulating training vectors...")

    faiss_sequence_index = 0
    sequence_index = 1  # Julia sequence index

    for batch_sequence in batch_sequence_iterator

        n_sequences = length(batch_sequence)

        one_hot_batch = Dataset.one_hot_encode_sequence_batch(batch_sequence, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS)
        X = permutedims(one_hot_batch, (3, 2, 1))
        X = reshape(one_hot_batch, :, 1, args.BSIZE)
        X = X |> DEVICE

        sequence_embeddings = embedding_model(X)

        push!(training_vectors, sequence_embeddings)

        if (length(training_vectors)*args.BSIZE) > faiss_train_size && kmeans_centroids_extracted == false
            # perform the training
            addEmbeddingsToKMeans!(args, training_vectors, faiss_clusterer, faiss_index; do_train_index=true)

            training_vectors = []
            # Reset training vectors
            kmeans_centroids_extracted = true
        end

        if length(training_vectors)*args.BSIZE > add_batch_size && kmeans_centroids_extracted == true
            # Add embeddings to the trained index
            addEmbeddingsToKMeans!(args, training_vectors, faiss_clusterer, faiss_index; do_train_index=false)
            # Reset training vectors
            training_vectors = []
        end
    end

    if length(training_vectors)*args.BSIZE > 0
        # Add remaining embeddings
        @info("Adding remaining embeddings")
        addEmbeddingsToKMeans!(args, training_vectors, faiss_clusterer, faiss_index; do_train_index=false)
        training_vectors = []
    end

    return faiss_clusterer, faiss_index
end


function addEmbeddoingsToIndex!(argsExperimentParamsType, embeddings_array::Array, index; do_train_index::Bool=false)
    # Size (128, 512*N)
    training_vector_matrix = cat(embeddings_array..., dims=2) |> Flux.cpu
    convert(Array, training_vector_matrix)
    @assert size(training_vector_matrix)[1] == args.EMBEDDING_DIM size(training_vector_matrix)[1]
    vs_ = Py(convert(AbstractMatrix{Float32}, training_vector_matrix)').__array__()

    if do_train_index == true
        index.train(vs_)
    end
    index.add(vs_)
    @info("Added %s vectors to Faiss index", size(training_vector_matrix)[2])
end


function create_embedding_index(argsExperimentParamsType, batch_sequence_iterator::Channel{Any}; quantize::Bool=true)
    faiss_train_size = 5000

    faiss_index = get_faiss_index(args, quantize=quantize)
    training_vectors = []

    faiss_is_trained = false

    add_batch_size = 2000

    LOG_EVERY = 10000
    @info("Accumulating training vectors...")
    for batch_sequence in batch_sequence_iterator
        one_hot_batch = Dataset.one_hot_encode_sequence_batch(batch_sequence, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS;do_padding=true)
        X = permutedims(one_hot_batch, (3, 2, 1))
        X = reshape(X, :, 1, args.BSIZE)

        X = X |> DEVICE
        sequence_embeddings = embedding_model(X)

        push!(training_vectors, sequence_embeddings)

        numVectors = length(training_vectors)*args.BSIZE

        if mod(numVectors, LOG_EVERY) == 0
            @info("Training vectors size is %s", length(training_vectors))
        end

        if (length(training_vectors)*args.BSIZE) > faiss_train_size && faiss_is_trained == false
            # Size (128, 512*N)
            addEmbeddoingsToIndex!(args, training_vectors::Array, faiss_index, do_train_index=true)
            faiss_is_trained = true
            @info("Trained Faiss index on %s vectors", faiss_train_size)
            training_vectors = []
        end

        if length(training_vectors)*args.BSIZE > add_batch_size && faiss_is_trained == true
            addEmbeddoingsToIndex!(args, training_vectors::Array, faiss_index)
            training_vectors = []
        end
    end

    if length(training_vectors) > 0
        # Add remaining vectors to the index
        @info("Adding remaining embeddings")
        addEmbeddoingsToIndex!(args, training_vectors::Array, faiss_index)
        training_vectors = []
    end
    return faiss_index
end


function _update_predicted_nn_map(argsExperimentParamsType, embeddings_array::Array, index; k::Int64=100)

    if args.DISTANCE_MATRIX_NORM_METHOD == "max"
        denorm_factor = args.MAX_STRING_LENGTH
    elseif args.DISTANCE_MATRIX_NORM_METHOD == "mean"
        # TODO: Use the mean here
        denorm_factor = args.MAX_STRING_LENGTH
    else
        throw("Invalid distance matrix norm factor")
    end

    pyBatchEmbeddings = np.concatenate(pylist(embeddings_array), axis=0)
    distances, nnIds = index.search(pyBatchEmbeddings, pyint(k))

    # Note: Distances in L2 space are squared
    jlDistances = convert(Array, PyArray(distances))
    # Note that the first index may be negative due to approximation/rounding errors
    jlDistances[jlDistances .< 0] .= 0
    D = sqrt.(jlDistances) * denorm_factor

    IDs = convert(Array, PyArray(nnIds))
    return IDs, D
end

function get_approximate_nn_overlap(argsExperimentParamsType, faiss_index; k=1000)

    # TODO: Make this part of the model that's fit on the training dataset
    # (e.g. for mean distance)
    denorm_factor = args.MAX_STRING_LENGTH

    batch_array = []
    time_finding_approx_nns = @elapsed begin
        predicted_nn_map = Dict()
        reconstructVectorID = 0
        faiss_vector_id_pointer = 0
        for _ in 0:length(inferenceSequences) - 1
            embedding = faiss_index.reconstruct(reconstructVectorID).reshape(1, -1)
            push!(batch_array, embedding)
            reconstructVectorID += 1

            if length(batch_array) == args.BSIZE
                ids, distances = _update_predicted_nn_map(args, batch_array, faiss_index, k=k)

                for idIdx in 1:size(ids)[1]
                    predicted_nn_map[faiss_vector_id_pointer] = Dict("topKNN" => ids[idIdx, 1:end], "distances" => distances[idIdx, 1:end])
                    faiss_vector_id_pointer += 1
                end
                batch_array = []
            end
        end

        if length(batch_array) > 0
            ids, distances = _update_predicted_nn_map(args, batch_array, faiss_index, k=k)
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


# trainedfaiss_clusterer, faiss_indexWithEmbeddings = createKNNIndex(args)


function get_true_nn_overlap(argsExperimentParamsType; k=1000)
    # Get the pairwise sequence distance array
    timeFindingTrueNNs = @elapsed begin
        true_nn_map = Dict()
        _, truepairwise_distances = Utils.pairwise_hamming_distance(inferenceSequences)

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


@load joinpath(EXPERIMENT_DIR, "args.jld2") args
@load Utils.get_best_model_path(args.MODEL_SAVE_DIR) embedding_model distance_calibration_model

embedding_model = embedding_model |> DEVICE

trainmode!(embedding_model, false)


inferenceSequences = get_inference_sequences()

QUANTIZE=false

batch_sequence_iterator = Channel( (channel) -> batch_sequence_producer(args, channel, inferenceSequences), 1)

faiss_index = create_embedding_index(args, batch_sequence_iterator, quantize=QUANTIZE)

# Make direct map if the index allows for it
if hasproperty(faiss_index, :make_direct_map)
    faiss_index.make_direct_map()
end

numNeighbours = 1000
predicted_nn_map = get_approximate_nn_overlap(args, faiss_index, k=numNeighbours)

true_nn_map = get_true_nn_overlap(args, k=numNeighbours)

epcoh_recall_dict = Utils.get_top_t_recall_at_k(
    ".", "test", length(true_nn_map), numNN=1000,
    kStart=1, kEnd=1001, kStep=100; startIndex=2, trueid_seq_data_map=true_nn_map, predictedid_seq_data_map=predicted_nn_map
)
