include("experiment_helper.jl")

using BenchmarkTools
using Random
using Statistics

using Flux
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using Distances
using LinearAlgebra
using Zygote: gradient, @ignore
using Parameters
using ProgressBars

using JLD2: @load
using PythonCall

using .Models
using .Utils
using .Dataset

try
    using CUDA
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end

np = pyimport("numpy")


function get_inference_sequences(args, is_testing::Bool)::Array
    if is_testing == true
        max_sequences = args.NUM_TEST_EXAMPLES
    else
        max_sequences = args.MAX_INFERENCE_SAMPLES
    end
    # Eval
    all_sequences = Dataset.get_sequences(args, max_sequences)
    return all_sequences
end

function inferencehelper(args, embedding_model::Flux.Chain, calibration_model; use_pipe_cleaner::Bool=false,
     quantize_faiss_index::Bool=false, use_gpu::Bool=false, num_neighbours::Int64=1000, is_testing:: Bool=false)

     # TODO: Don't load all sequences into memory
     inference_sequences = get_inference_sequences(args, is_testing)

     # Number of batches in the queue
     num_inference_sequences = length(inference_sequences)

     function batch_sequence_producer(args, chnl::Channel)
        i = 1
        j = args.BSIZE
        while j < length(inference_sequences)
            batch = inference_sequences[i:j]
            i += args.BSIZE
            j += args.BSIZE
            put!(chnl, batch)
        end
        # Last batch
        put!(chnl, inference_sequences[i:end])
    end


    function _get_approx_ids_distances(embeddings_array::Array, index; k::Int64=100)

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

    function _predicted_nns(n_sequences::Int64, use_mmap_arrays::Bool = false; k_nns::Int64=1000)
        if use_mmap_arrays == true
            predicted_nn_map_top_knn = Utils.mmap_array(
                n_sequences,
                k_nns,
                path=joinpath(args.INFERENCE_MMAP_SAVE_DIR, "nn_ids.bin"),
                matrix_type=Int64,
                overwrite=true,
            )
            predicted_nn_map_distances = Utils.mmap_array(
                n_sequences,
                k_nns,
                path=joinpath(args.INFERENCE_MMAP_SAVE_DIR, "distances.bin"),
                matrix_type=Float64,
                overwrite=true
            )
            predicted_nn_map = nothing
        else
            predicted_nn_map = Dict()
            predicted_nn_map_top_knn = predicted_nn_map_distances = nothing
        end

        get_predicted_nn_arrays() = predicted_nn_map_top_knn, predicted_nn_map_distances
        get_predicted_nn_dict() = predicted_nn_map

        function update!(faiss_id::Int64, julia_id::Int64, ids::Array, distances::Array)::Nothing
            if use_mmap_arrays == true
                # Note: mmapped arrays index starts at 1
                predicted_nn_map_top_knn.mmarray[faiss_id + 1, 1:end] = ids[julia_id, 1:end]
                predicted_nn_map_distances.mmarray[faiss_id + 1, 1:end] = distances[julia_id, 1:end]
            else
                predicted_nn_map[faiss_id] = Dict("topKNN" => ids, "distances" => distances)
            end
            return nothing
        end

        () -> (get_predicted_nn_arrays;get_predicted_nn_dict;update!)
    end

    function get_approximate_nn_overlap(faiss_index, n_sequences::Int64;
        k::Int64=1000, use_mmap_arrays::Bool=false)

        @info("Starting to find aproximate NNs on $(n_sequences) sequences")
        # TODO: Make this part of the model that's fit on the training dataset
        # (e.g. for mean distance)
        denorm_factor = args.MAX_STRING_LENGTH

        batch_array = []

        approximate_nns = _predicted_nns(n_sequences, use_mmap_arrays)

        time_finding_approx_nns = @elapsed begin

            reconstructVectorID = 0
            faiss_vector_id_pointer = 0
            for _ in ProgressBar(0:n_sequences - 1)
                embedding = faiss_index.reconstruct(reconstructVectorID).reshape(1, -1)
                push!(batch_array, embedding)
                reconstructVectorID += 1

                if length(batch_array) == args.BSIZE
                    ids, distances = _get_approx_ids_distances(batch_array, faiss_index, k=k)
                    for idIdx in 1:size(ids)[1]
                        approximate_nns.update!(faiss_vector_id_pointer, idIdx, ids, distances)
                        faiss_vector_id_pointer += 1
                    end
                    batch_array = []
                end
            end

            if length(batch_array) > 0
                ids, distances = _get_approx_ids_distances(batch_array, faiss_index, k=k)
                for idIdx in 1:size(ids)[1]
                    approximate_nns.update!(faiss_vector_id_pointer, idIdx, ids, distances)
                    faiss_vector_id_pointer += 1
                end
                batch_array = []
            end
        end

        @info("Time to find approximate NNs is $(round(time_finding_approx_nns))s")

        if use_mmap_arrays == true
            ids_mmap, distances_mmap = approximate_nns.get_predicted_nn_arrays()
            ids_mmap.closeio()
            distances_mmap.closeio()

            return approximate_nns.get_predicted_nn_arrays()
        else
            return approximate_nns.get_predicted_nn_dict()
        end
    end


    function get_true_nn_overlap(;k::Int64=1000)
        @info("Starting to find true NNs")
        # Get the pairwise sequence distance array
        timeFindingTrueNNs = @elapsed begin
            true_nn_map = Dict()
            # TODO: This is inefficient and intended for small datasets
            _, truepairwise_distances = Utils.pairwise_hamming_distance(inference_sequences)

            for i in ProgressBar(1:size(truepairwise_distances)[1])
                nns = sortperm(truepairwise_distances[i, 1:end])[1:k]
                distances = truepairwise_distances[i, 1:end][nns]

                # Use FAISS 0-based indexing
                nns = nns .- 1
                true_nn_map[i - 1] = Dict("topKNN" => nns, "distances" => distances)  # Index the same as FAISS indexed
            end
        end
        @info("Time to find true NNs is $(round(timeFindingTrueNNs))s")
        return true_nn_map
    end

    function infer()
        embedding_model = embedding_model |> DEVICE

        trainmode!(embedding_model, false)

        batch_sequence_iterator = Channel( (channel) -> batch_sequence_producer(args, channel), 1)

        faiss_index = Utils.create_embedding_index(
            args, embedding_model, batch_sequence_iterator,
            num_inference_sequences, quantize=quantize_faiss_index
        )

        # After training and adding vectors, make a direct map if the index allows for it
        if hasproperty(faiss_index, :make_direct_map)
            faiss_index.make_direct_map()
        end

        predicted_nn_map = get_approximate_nn_overlap(
            faiss_index,
            length(inference_sequences),
            k=num_neighbours,
            use_mmap_arrays=~is_testing  # Mmap arrays when doing inference only
        )

        # Write the faiss index to disk
        @info("Writing the inference faiss index to dist...")
        if is_testing == true
            faiss_index_id = "testing"
        else
            faiss_index_id = "inference"
        end

        Utils.write_faiss_index(
            args,
            faiss_index,
            faiss_index_id
        )

        return predicted_nn_map, faiss_index
    end

    function test()
        # Test is inference with evaluation
        predicted_nn_map, faiss_index = infer()

        true_nn_map = get_true_nn_overlap(k=num_neighbours)

        epoch_recall_dict = Utils.get_top_t_recall_at_k(
            args.PLOTS_SAVE_DIR, "testing", length(true_nn_map), numNN=1000,
            kStart=1, kEnd=1001, kStep=100; nn_start_index=2,
            true_id_seq_data_map=true_nn_map,
            predicted_id_seq_data_map=predicted_nn_map, num_samples=1000
        )

        return predicted_nn_map, true_nn_map, epoch_recall_dict, faiss_index
    end
    () -> (test;infer;inference_sequences;batch_sequence_producer;_get_approx_ids_distances;get_approximate_nn_overlap;get_true_nn_overlap;faiss)
end
