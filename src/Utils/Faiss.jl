# include("./src/ExperimentHelper.jl")

include("../ExperimentHelper.jl")
include("../Datasets/Dataset.jl")

module Faiss
    using PythonCall
    using Flux
    using Parameters

    using ..ExperimentHelper
    using ..ExperimentHelper: ExperimentParams
    using ..Dataset

    faiss = pyimport("faiss")
    np = pyimport("numpy")

    try
        using CUDA
        global DEVICE = Flux.gpu
    catch e
        global DEVICE = Flux.cpu
    end

    function add_embeddings_to_kmeans!(args::ExperimentParams, embeddings::Array, clusterer, index; do_train_index::Bool=false)
        # Size (128, 512*N)
        training_vector_matrix = cat(embeddings..., dims=2) |> Flux.cpu
        convert(Array, training_vector_matrix)
        @assert size(training_vector_matrix)[1] == args.EMBEDDING_DIM size(training_vector_matrix)[1]

        vs_ = Py(convert(AbstractMatrix{Float32}, training_vector_matrix)').__array__()

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


    function get_faiss_index(args::ExperimentParams; quantize::Bool=false, d::Int64=128, nlist::Int64=100, m::Int64=8,bits::Int64=8, use_gpu::Bool=false)
        d = args.EMBEDDING_DIM

        if quantize == true
            quantizer = faiss.IndexFlatL2(d)  # build a flat (CPU) index
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
        else
            index = faiss.IndexFlatL2(d)  # build a flat (CPU) index
        end

        if use_gpu && DEVICE == Flux.gpu
            resource = faiss.StandardGpuResources()  # use a single GPU
            @info("Creating GPU FAISS index")
            index = faiss.index_cpu_to_gpu(resource, 0, index)
        end

        return index
    end

    function create_knn_index(args::ExperimentParams, model::Flux.Chain; quantize::Bool=false, use_gpu::Bool = false)

        # TODO: Get this from args
        faiss_train_size = 5000

        estimated_genome_length = 5000

        toleranceScaleFactor = 1.3

        faiss_index = get_faiss_index(args, quantize=quantize, use_gpu=use_gpu)

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

        model = model |> DEVICE

        for batch_sequence in batch_sequence_iterator

            n_sequences = length(batch_sequence)

            one_hot_batch = Dataset.one_hot_encode_sequence_batch(batch_sequence, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS)
            X = permutedims(one_hot_batch, (3, 2, 1))
            X = reshape(one_hot_batch, :, 1, args.BSIZE)
            X = X |> DEVICE

            sequence_embeddings = model(X)

            push!(training_vectors, sequence_embeddings)

            if (length(training_vectors)*args.BSIZE) > faiss_train_size && kmeans_centroids_extracted == false
                # perform the training
                add_embeddings_to_kmeans!(args, training_vectors, faiss_clusterer, faiss_index; do_train_index=true)

                training_vectors = []
                # Reset training vectors
                kmeans_centroids_extracted = true
            end

            if length(training_vectors)*args.BSIZE > add_batch_size && kmeans_centroids_extracted == true
                # Add embeddings to the trained index
                add_embeddings_to_kmeans!(training_vectors, faiss_clusterer, faiss_index; do_train_index=false)
                # Reset training vectors
                training_vectors = []
            end
        end

        if length(training_vectors)*args.BSIZE > 0
            # Add remaining embeddings
            @info("Adding remaining embeddings")
            add_embeddings_to_kmeans!(training_vectors, faiss_clusterer, faiss_index; do_train_index=false)
            training_vectors = []
        end

        return faiss_clusterer, faiss_index
    end


    function add_embeddings_to_index!(args::ExperimentParams, embeddings_array::Array, index; do_train_index::Bool=false)
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


    function create_embedding_index(args::ExperimentParams, model, batch_sequence_iterator::Channel{Any}; quantize::Bool=true, use_gpu::Bool=false)
        faiss_train_size = 5000

        faiss_index = get_faiss_index(args, quantize=quantize, use_gpu=use_gpu)
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
            sequence_embeddings = model(X)

            push!(training_vectors, sequence_embeddings)

            numVectors = length(training_vectors)*args.BSIZE

            if mod(numVectors, LOG_EVERY) == 0
                @info("Training vectors size is %s", length(training_vectors))
            end

            if (length(training_vectors)*args.BSIZE) > faiss_train_size && faiss_is_trained == false
                # Size (128, 512*N)
                add_embeddings_to_index!(args, training_vectors::Array, faiss_index, do_train_index=true)
                faiss_is_trained = true
                @info("Trained Faiss index on %s vectors", faiss_train_size)
                training_vectors = []
            end

            if length(training_vectors)*args.BSIZE > add_batch_size && faiss_is_trained == true
                add_embeddings_to_index!(args, training_vectors::Array, faiss_index)
                training_vectors = []
            end
        end

        if length(training_vectors) > 0
            # Add remaining vectors to the index
            @info("Adding remaining embeddings")
            add_embeddings_to_index!(args, training_vectors::Array, faiss_index)
            training_vectors = []
        end
        return faiss_index
    end
end
