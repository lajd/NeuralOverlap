include("./src/ExperimentParams.jl")
include("./src/Utils.jl")
include("./src/Datasets/Dataset.jl")
include("./src/Datasets/SyntheticDataset.jl")
include("./src/Model.jl")
include("./src/SimilaritySearch/VectorSearch.jl")
include("./src/SimilaritySearch/VectorSearch.jl")

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
using BSON: @load
using JLD2
using PythonCall

using .Dataset
using .ExperimentParams
using .Model
using .Utils
using .VectorSearch

using .VectorSearch

faiss = pyimport("faiss")


try
    using CUDA
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end


EXPERIMENT_DIR="/home/jon/JuliaProjects/NeuralOverlap/data/experiments/2022-01-17T21:46:46.347"
INFERENCE_FASTQ = "/home/jon/JuliaProjects/NeuralOverlap/data_fetch/covid_test.fa_R1.fastq"

args = JLD2.load(joinpath(EXPERIMENT_DIR, "args.jld2"))["args"]
models = JLD2.load(Utils.getBestModelPath(args.MODEL_SAVE_DIR)) |> DEVICE
embeddingModel = models["embedding_model"] |> DEVICE
calibrationModel = models["distance_calibration_model"]
trainmode!(embeddingModel, false)

function getInfererenceSequences(maxSamples=Int(1e4))::Array
    # Eval Dataset
    if args.USE_SYNTHETIC_DATA == true
        testSequences = SyntheticDataset.generateSequences(
            maxSamples, args.MAX_STRING_LENGTH,
            args.MAX_STRING_LENGTH, args.ALPHABET,ratioOfRandom=args.RATIO_OF_RANDOM_SAMPLES,
            similarityMin=args.SIMILARITY_MIN, similarityMax=args.SIMILARITY_MAX
            )
    elseif args.USE_SEQUENCE_DATA == true
        testSequences = SequenceDataset.getReadSequenceData(maxSamples, args.MAX_STRING_LENGTH, fastqFilePath=INFERENCE_FASTQ)
    else
        throw("Must provide type of dataset")
    end
    # Create a map if sequence ID to sequence
    return testSequences
end


inferenceSequences = getInfererenceSequences()

bSize = 512
faissTrainSize = 1000


function batchSequencesProducer(chnl, sequences, bsize)
    i = 1
    j = bsize
    while j < length(sequences)
        batch = sequences[i:j]
        i += bsize
        j += bsize
        put!(chnl, batch)
    end
    # Last batch
    put!(chnl, sequences[i:end])
end



function addEmbeddingsToKMeans!(args, embeddings::Array, clusterer, index; doTrain=false)
    # Size (128, 512*N)
    trainingVectorMatrix = cat(embeddings..., dims=2) |> Flux.cpu
    convert(Array, trainingVectorMatrix)
    @assert size(trainingVectorMatrix)[1] == args.EMBEDDING_DIM size(trainingVectorMatrix)[1]

    vs_ = Py(convert(AbstractMatrix{Float32}, trainingVectorMatrix)').__array__()

    nEmbeddings = size(trainingVectorMatrix)[2]
    nEmbeddings = size(trainingVectorMatrix)[2]

    if doTrain == true
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


batchSequenceIterator = Channel( (channel) -> batchSequencesProducer(channel, inferenceSequences, bSize), 1)


function getFaissIndex(args; quantize=false)
    nlist = 100
    m = 8
    d = 128
    if quantize
        quantizer = faiss.IndexFlatL2(d)  # this remains the same
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    else
        index = faiss.IndexFlatL2(d)
    end
    return index
end

function createKNNIndex(args)
    estimatedGenomeLength = 5000

    toleranceScaleFactor = 1.3

#     if DEVICE == Flux.gpu
#         faissIndex = faiss.GpuIndexFlatL2(128)
#     else
#         faissIndex = faiss.IndexFlatL2(128)
#     end
    faissIndex = faiss.IndexFlatL2(128)

    estimatedNumTrueReads = Int(ceil(estimatedGenomeLength/args.MAX_STRING_LENGTH))
    nClusters = 20 #estimatedNumTrueReads * toleranceScaleFactor
    nRedo = 10
    maxIterations = 50
    anticipatedCoverage = 100

    faissClusterer = faiss.Clustering(args.EMBEDDING_DIM, nClusters)
    faissClusterer.niter = maxIterations
    faissClusterer.nredo = nRedo
    # otherwise the kmeans implementation sub-samples the training set
    faissClusterer.max_points_per_centroid = anticipatedCoverage

    kMeansCentroidsExtracted = false

    trainingVectors = []

    addBatchSize = 200

    mapFaissIDToSequenceID = Dict()

    LOG_EVERY = 10000
    @info("Accumulating training vectors...")

    faissSequenceIndex = 0
    sequenceIndex = 1  # Julia sequence index

    for batchSequence in batchSequenceIterator

        numSequences = length(batchSequence)

        oneHotBatch = Dataset.oneHotBatchSequences(batchSequence, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS)
        X = permutedims(oneHotBatch, (3, 2, 1))
        X = reshape(oneHotBatch, :, 1, bSize)
        X = X |> DEVICE

        sequenceEmbeddings = embeddingModel(X)

        push!(trainingVectors, sequenceEmbeddings)

        if (length(trainingVectors)*args.BSIZE) > faissTrainSize && kMeansCentroidsExtracted == false
            # perform the training
            addEmbeddingsToKMeans!(args, trainingVectors, faissClusterer, faissIndex; doTrain=true)

            trainingVectors = []
            # Reset training vectors
            kMeansCentroidsExtracted = true
        end

        if length(trainingVectors)*args.BSIZE > addBatchSize && kMeansCentroidsExtracted == true
            # Add embeddings to the trained index
            addEmbeddingsToKMeans!(args, trainingVectors, faissClusterer, faissIndex; doTrain=false)
            # Reset training vectors
            trainingVectors = []
        end
    end

    if length(trainingVectors)*args.BSIZE > 0
        # Add remaining embeddings
        @info("Adding remaining embeddings")
        addEmbeddingsToKMeans!(args, trainingVectors, faissClusterer, faissIndex; doTrain=false)
        trainingVectors = []
    end

    return faissClusterer, faissIndex
end


function addEmbeddoingsToIndex!(args, embeddingsArray::Array, index; doTrain=false)
    # Size (128, 512*N)
    trainingVectorMatrix = cat(embeddingsArray..., dims=2) |> Flux.cpu
    convert(Array, trainingVectorMatrix)
    @assert size(trainingVectorMatrix)[1] == args.EMBEDDING_DIM size(trainingVectorMatrix)[1]
    vs_ = Py(convert(AbstractMatrix{Float32}, trainingVectorMatrix)').__array__()

    if doTrain == true
        index.train(vs_)
    end
    index.add(vs_)
    @info("Added %s vectors to Faiss index", size(trainingVectorMatrix)[2])
end


function createEmbeddingIndex(args)
#     faissIndex = VectorSearch.FaissIndex(args.EMBEDDING_DIM)

    faissIndex = getFaissIndex(args)
    trainingVectors = []

    faissIsTrained = false

    addBatchSize = 2000

    LOG_EVERY = 10000
    @info("Accumulating training vectors...")
    for batchSequence in batchSequenceIterator
        oneHotBatch = Dataset.oneHotBatchSequences(batchSequence, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS;doPad=true)
        X = permutedims(oneHotBatch, (3, 2, 1))
        X = reshape(oneHotBatch, :, 1, bSize)

        X = X |> DEVICE
        sequenceEmbeddings = embeddingModel(X)
        e = sequenceEmbeddings

        push!(trainingVectors, sequenceEmbeddings)

        numVectors = length(trainingVectors)*args.BSIZE

        if mod(numVectors, LOG_EVERY) == 0
            @info("Training vectors size is %s", length(trainingVectors))
        end

        if (length(trainingVectors)*args.BSIZE) > faissTrainSize && faissIsTrained == false
            # Size (128, 512*N)
            addEmbeddoingsToIndex!(args, trainingVectors::Array, faissIndex, doTrain=true)
            faissIsTrained = true
            @info("Trained Faiss index on %s vectors", faissTrainSize)
            trainingVectors = []
        end

        if length(trainingVectors)*args.BSIZE > addBatchSize && faissIsTrained == true
            addEmbeddoingsToIndex!(args, trainingVectors::Array, faissIndex)
            trainingVectors = []
        end
    end

    if length(trainingVectors) > 0
        # Add remaining vectors to the index
        @info("Adding remaining embeddings")
        addEmbeddoingsToIndex!(args, trainingVectors::Array, faissIndex)
        trainingVectors = []
    end
    return faissIndex
end



faissIndex = createEmbeddingIndex(args)

# Make direct map

faissIndex.make_direct_map()


function getApproximateNNOverlap(args, faissIndex; k=20)
    timeFindingApproximateNNs = @elapsed begin
        predictedNNMap = Dict()
        for faissVectorID in 0:length(inferenceSequences)
            embedding = faissIndex.reconstruct(faissVectorID)
            distances, nnIds = faissIndex.search(embedding.reshape(1, -1), pyint(k))
            D = convert(Array, PyArray(distances)) * args.MAX_STRING_LENGTH
            IDs = convert(Array, PyArray(nnIds))
            predictedNNMap[faissVectorID] = Dict("nnIds" => IDs, "distances" => D)
        end
    end
    @info("Time to find approximate NNs is %s", timeFindingApproximateNNs)
    return predictedNNMap
end


# trainedFaissClusterer, faissIndexWithEmbeddings = createKNNIndex(args)


function getTrueNNOverlap(args; k=20)
    # Get the pairwise sequence distance array
    timeFindingTrueNNs = @elapsed begin
        trueNNMap = Dict()
        _, truePairwiseDistances = Utils.pairwiseHammingDistance(inferenceSequences)

        for i in 1:size(truePairwiseDistances)[1]
            nns = sortperm(truePairwiseDistances[i, 1:end])[1:k]
            distances = truePairwiseDistances[i, 1:end][nns]
            trueNNMap[i - 1] = Dict("nnIds" => nns, "distances" => distances)  # Index the same as FAISS indexed
        end
    end
    @info("Time to find true NNs is %s", timeFindingTrueNNs)
    return trueNNMap
end


predictedNNMap = getApproximateNNOverlap(args, faissIndex)
trueNNMap = getTrueNNOverlap(args)











































#
# function getSequence(faissIndex, inferenceSequences; nClusters=20)
#     if (faissIndex < nClusters) || (faissIndex > length(inferenceSequences) + nClusters)
#         throw("Faiss index out of range")
#     else
#         return inferenceSequences[faissIndex - nClusters + 1]
#     end
# end
#
# # Get the pairwise sequence distance array
# timeCreatingPairwiseDistance = @elapsed begin
#     _, truePairwiseDistances = Utils.pairwiseHammingDistance(inferenceSequences)
#     # Get the nearest neighbours for each vector
#
#     # Get nearest neighbours by overlap
#
#
# end
#
#
#
# @info("Time creating true pairwise distances is %s", timeCreatingPairwiseDistance)
# function getClusterMembers(clusterer, index, neighbourhoodRadius=0.5)
#     centroids = pyconvert(Array, trainedFaissClusterer.postProcessedCentroids)
#     distances, ids = faissIndex.search(
#         faiss.vector_float_to_array(trainedFaissClusterer.centroids).reshape(trainedFaissClusterer.k, trainedFaissClusterer.d),
#         pyint(50)
#     )
#
#     embeddingVectors = faiss.vector_float_to_array(faissIndex.xb).reshape(-1, 128).shape
#
#     # Convert to Julia
#     jlDistances = pyconvert(Array{Float32}, distances)
#     jlIds = pyconvert(Array{Int64}, ids)
#
#     vs_ = Py(convert(AbstractMatrix{Float32}, centroids)').__array__()
# end
#
#
# clusterMembers = getClusterMembers(trainedFaissClusterer, faissIndexWithEmbeddings)


#=

getClusterMembers(trainedFaissClusterer, faissIndexWithEmbeddings)
=#


#####################################################################################################
#####################################################################################################


# @info "Number of embedding vectors is %s ", faissIndex.index.py.ntotal


# Create overlap graph using KMEANS
# Use number of clusters estimated to be the number of true segments
# For example, assume the genome's true size is 5000 bases
# (this could be determined using an algorithm and comparing to similar sequence content, etc).
# Since the read length is 300, we can assume there are roughly 5000/300 = ~17 true reads.
# Let's round this to 20 true reads, and so we use 20 clusters for our Kmeans clustering.


#
# # Perform training
# clusterer.train(x, faissIndex)
#
# kmeans.train(x_train.astype(np.float32))
#
# e = time.time()
# print("Training time = {}".format(e - s))
#
# s = time.time()
# kmeans.index.search(x_train.astype(np.float32), 1)[1]
# e = time.time()
# print("Prediction time = {}".format((e - s) / len(y_test)))
#
#
#
# # # function Infer(s1, s2, embeddingModel)
# # #     sequenceBatch = [s1, s2]
# # #     oneHotBatch = Dataset.oneHotBatchSequences(sequenceBatch, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS)
# #
# #
# # #     X, = Model.formatOneHotSequences((oneHotBatch,))
# #
# #
# # #     X = X |> DEVICE
# # #     sequenceEmbeddings = embeddingModel(X)
# #
# # #     E1 = sequenceEmbeddings[1:end, 1]
# # #     E2 = sequenceEmbeddings[1:end, 2]
# #
# # #     D12 = Utils.EmbeddingDistance(E1, E2, args.DISTANCE_METHOD, dims=1) |> DEVICE  # 1D dist vector of size bsize
# #
# # #     predictedEditDistance = args.MAX_STRING_LENGTH * D12
# # #     return D12, predictedEditDistance
# # # end
# #
# # evalDatasetHelper = Dataset.DatasetHelper(
# #     evalSequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
# #     Utils.pairwiseHammingDistance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
# #     args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS
# # )
#
#
# meanAbsError, maxAbsError, minAbsError, totalAbsError, meanEstimationError, recallDict = Utils.evaluateModel(
#     evalDatasetHelper, embeddingModel, args.MAX_STRING_LENGTH,
#     method=args.DISTANCE_METHOD, plotsSavePath=args.PLOTS_SAVE_DIR, numNN=args.NUM_NNS,
#     identifier="inference", distanceMatrixNormMethod=args.DISTANCE_MATRIX_NORM_METHOD,
#     kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP
# )
#
#
#
# function createKNNIndex()
#     estimatedGenomeLength = 5000
#
#     toleranceScaleFactor = 2
#
#     faissIndex = faiss.IndexFlatL2(128)
#     nClusters = assumedCoverage * toleranceScaleFactor
#     nRedo= 10
#     maxIterations = 50
#     anticipatedCoverage = 100
#
#     faissClusterer = faiss.Clustering(args.EMBEDDING_DIM, nClusters)
#     faissClusterer.niter = maxIterations
#     faissClusterer.nredo = nRedo
#     # otherwise the kmeans implementation sub-samples the training set
#     faissClusterer.max_points_per_centroid = anticipatedCoverage
#
#     trainingVectors = []
#
#     faissIsTrained = false
#
#     addBatchSize = 2000
#
#     LOG_EVERY = 10000
#     @info("Accumulating training vectors...")
#     for batchSequence in batchSequenceIterator
#         s = batchSequence
#         oneHotBatch = Dataset.oneHotBatchSequences(batchSequence, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS)
#         o = oneHotBatch
#         X = permutedims(o, (3, 2, 1))
#         X = reshape(o, :, 1, bSize)
#
#         X = X |> DEVICE
#         sequenceEmbeddings = embeddingModel(X)
#         e = sequenceEmbeddings
#
#         push!(trainingVectors, sequenceEmbeddings)
#
#         numVectors = length(trainingVectors)*args.BSIZE
#
#         if mod(numVectors, LOG_EVERY) == 0
#             @info("Training vectors size is %s", length(trainingVectors))
#         end
#
#         if (numVectors) > faissTrainSize && faissIsTrained == false
#             break
#         end
#     end
#
#     # perform the training
#     trainingVectorMatrix = cat(trainingVectors..., dims=2) |> Flux.cpu
#     convert(Array, trainingVectorMatrix)
#     @assert size(trainingVectorMatrix)[1] == args.EMBEDDING_DIM size(trainingVectorMatrix)[1]
#
#     vs_ = Py(convert(AbstractMatrix{Float32}, trainingVectorMatrix)').__array__()
#
#     faissClusterer.train(vs_, faissIndex)
#
#     # Now the faiss index has centroids based on the KMEANS training above
#     faissIndex.add(vs_)
#
#     @info ("Faiss index has %s embeddings", faissIndex.ntotal)
#
#     # Now add the remaining
#
#
#
#
#     faissClusterer.add(vs_, faissIndex)
#
#
#
#
#         trainingVectors = []
#         @info("Added %s vectors to Faiss clusterer", size(trainingVectorMatrix)[2])
#
#         addEmbeddingsToClusterer!(args, trainingVectors::Array, faissClusterer, faissIndex, doTrain=true)
#         faissIsTrained = true
#         @info("Trained Faiss index on %s vectors", faissTrainSize)
#     end
#
#     if numVectors > addBatchSize && faissIsTrained == true
#         addEmbeddingsToClusterer!(args, trainingVectors::Array, faissClusterer, faissIndex)
#     end
# end
#
# # Add remaining vectors to the index
# addEmbeddingsToClusterer!(args, trainingVectors::Array, faissClusterer, faissIndex)
#
#
#
# function addEmbeddoingsToIndex!(args, embeddingsArray::Array, faissHelper; doTrain=false)
#     # Size (128, 512*N)
#     trainingVectorMatrix = cat(embeddingsArray..., dims=2) |> Flux.cpu
#     convert(Array, trainingVectorMatrix)
#     @assert size(trainingVectorMatrix)[1] == args.EMBEDDING_DIM size(trainingVectorMatrix)[1]
#     if doTrain == true
#         faissHelper.train!(trainingVectorMatrix)
#     end
#     faissHelper.add!(trainingVectorMatrix)
#     embeddingsArray = []
#     @info("Added %s vectors to Faiss index", size(trainingVectorMatrix)[2])
# end
#
#
# function addEmbeddingsToClusterer!(args, embeddingsArray::Array, faissClusterer, faissIndex; doTrain=false)
#     # Size (128, 512*N)
#     trainingVectorMatrix = cat(embeddingsArray..., dims=2) |> Flux.cpu
#     convert(Array, trainingVectorMatrix)
#     @assert size(trainingVectorMatrix)[1] == args.EMBEDDING_DIM size(trainingVectorMatrix)[1]
#     if doTrain == true
#         faissClusterer.train(trainingVectorMatrix, faissIndex)
#     end
#     faissClusterer.add(trainingVectorMatrix, faissIndex)
#     embeddingsArray = []
#     @info("Added %s vectors to Faiss clusterer", size(trainingVectorMatrix)[2])
# end
#
#
# function createEmbeddingIndex(args)
#     faissIndex = VectorSearch.FaissIndex(args.EMBEDDING_DIM)
#
#     trainingVectors = []
#     X = nothing
#     s = nothing
#     o = nothing
#     e = nothing
#
#     faissIsTrained = false
#
#     addBatchSize = 2000
#
#     LOG_EVERY = 10000
#     @info("Accumulating training vectors...")
#     for batchSequence in batchSequenceIterator
#         s = batchSequence
#         oneHotBatch = Dataset.oneHotBatchSequences(batchSequence, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS)
#         o = oneHotBatch
#         X = permutedims(o, (3, 2, 1))
#         X = reshape(o, :, 1, bSize)
#
#         X = X |> DEVICE
#         sequenceEmbeddings = embeddingModel(X)
#         e = sequenceEmbeddings
#
#         push!(trainingVectors, sequenceEmbeddings)
#
#         numVectors = length(trainingVectors)*args.BSIZE
#
#         if mod(numVectors, LOG_EVERY) == 0
#             @info("Training vectors size is %s", length(trainingVectors))
#         end
#
#         if (numVectors) > faissTrainSize && faissIsTrained == false
#             # Size (128, 512*N)
#             addEmbeddoingsToIndex!(args, trainingVectors::Array, faissIndex, doTrain=true)
#             faissIsTrained = true
#             @info("Trained Faiss index on %s vectors", faissTrainSize)
#         end
#
#         if numVectors > addBatchSize && faissIsTrained == true
#             addEmbeddoingsToIndex!(args, trainingVectors::Array, faissIndex)
#         end
#     end
#
#     # Add remaining vectors to the index
#     addEmbeddoingsToIndex!(args, trainingVectors::Array, faiss)
#
#     return faissIndex
# end
