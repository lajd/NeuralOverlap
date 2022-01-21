include("./src/ExperimentParams.jl")
include("./src/Utils.jl")
include("./src/Datasets/Dataset.jl")
include("./src/Datasets/SequenceDataset.jl")
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
np = pyimport("numpy")


try
    using CUDA
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end


EXPERIMENT_DIR="/home/jon/JuliaProjects/NeuralOverlap/data/experiments/2022-01-20T08:03:13.264_newModelMeanNorm"
INFERENCE_FASTQ = "/home/jon/JuliaProjects/NeuralOverlap/data_fetch/covid_test.fa_R1.fastq"
# INFERENCE_FASTQ = "/home/jon/JuliaProjects/NeuralOverlap/data_fetch/phix174_train.fa_R1.fastq"


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


function batchSequencesProducer(args, chnl, sequences)
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


function getFaissIndex(args; quantize=false)
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

function createKNNIndex(args)

    # TODO: Get this from args
    FAISS_TRAIN_SIZE = 5000

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
        X = reshape(oneHotBatch, :, 1, args.BSIZE)
        X = X |> DEVICE

        sequenceEmbeddings = embeddingModel(X)

        push!(trainingVectors, sequenceEmbeddings)

        if (length(trainingVectors)*args.BSIZE) > FAISS_TRAIN_SIZE && kMeansCentroidsExtracted == false
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


function createEmbeddingIndex(args, batchSequenceIterator;quantize=true)
#     faissIndex = VectorSearch.FaissIndex(args.EMBEDDING_DIM)

    FAISS_TRAIN_SIZE = 5000

    faissIndex = getFaissIndex(args, quantize=quantize)
    trainingVectors = []

    faissIsTrained = false

    addBatchSize = 2000

    LOG_EVERY = 10000
    @info("Accumulating training vectors...")
    for batchSequence in batchSequenceIterator
        oneHotBatch = Dataset.oneHotBatchSequences(batchSequence, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS;doPad=true)
        X = permutedims(oneHotBatch, (3, 2, 1))
        X = reshape(X, :, 1, args.BSIZE)

        X = X |> DEVICE
        sequenceEmbeddings = embeddingModel(X)

        push!(trainingVectors, sequenceEmbeddings)

        numVectors = length(trainingVectors)*args.BSIZE

        if mod(numVectors, LOG_EVERY) == 0
            @info("Training vectors size is %s", length(trainingVectors))
        end

        if (length(trainingVectors)*args.BSIZE) > FAISS_TRAIN_SIZE && faissIsTrained == false
            # Size (128, 512*N)
            addEmbeddoingsToIndex!(args, trainingVectors::Array, faissIndex, doTrain=true)
            faissIsTrained = true
            @info("Trained Faiss index on %s vectors", FAISS_TRAIN_SIZE)
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


function _updatePredictedNNmap(args, embeddingsArray, index; k=100)

    if args.DISTANCE_MATRIX_NORM_METHOD == "max"
        denormFactor = args.MAX_STRING_LENGTH
    elseif args.DISTANCE_MATRIX_NORM_METHOD == "mean"
        # TODO: Use the mean here
        denormFactor = args.MAX_STRING_LENGTH
    else
        throw("Invalid distance matrix norm factor")
    end

    pyBatchEmbeddings = np.concatenate(pylist(embeddingsArray), axis=0)
    distances, nnIds = index.search(pyBatchEmbeddings, pyint(k))

    # Note: Distances in L2 space are squared
    jlDistances = convert(Array, PyArray(distances))
    # Note that the first index may be negative due to approximation/rounding errors
    jlDistances[jlDistances .< 0] .= 0
    D = sqrt.(jlDistances) * denormFactor

    IDs = convert(Array, PyArray(nnIds))
    return IDs, D
end

function getApproximateNNOverlap(args, faissIndex; k=1000)

    # TODO: Make this part of the model that's fit on the training dataset
    # (e.g. for mean distance)
    denormFactor = args.MAX_STRING_LENGTH

    batchArray = []
    timeFindingApproximateNNs = @elapsed begin
        predictedNNMap = Dict()
        reconstructVectorID = 0
        faissVectorIDPointer = 0
        for _ in 0:length(inferenceSequences) - 1
            embedding = faissIndex.reconstruct(reconstructVectorID).reshape(1, -1)
            push!(batchArray, embedding)
            reconstructVectorID += 1

            if length(batchArray) == args.BSIZE
                ids, distances = _updatePredictedNNmap(args, batchArray, faissIndex, k=k)

                for idIdx in 1:size(ids)[1]
                    predictedNNMap[faissVectorIDPointer] = Dict("topKNN" => ids[idIdx, 1:end], "distances" => distances[idIdx, 1:end])
                    faissVectorIDPointer += 1
                end
                batchArray = []
            end
        end

        if length(batchArray) > 0
            ids, distances = _updatePredictedNNmap(args, batchArray, faissIndex, k=k)
            for _ in 1:length(batchArray)
                predictedNNMap[faissVectorIDPointer] = Dict("topKNN" => ids, "distances" => distances)
                faissVectorIDPointer += 1
            end
            batchArray = []
        end
    end
    @info("Time to find approximate NNs is %s", timeFindingApproximateNNs)
    return predictedNNMap
end


# trainedFaissClusterer, faissIndexWithEmbeddings = createKNNIndex(args)


function getTrueNNOverlap(args; k=1000)
    # Get the pairwise sequence distance array
    timeFindingTrueNNs = @elapsed begin
        trueNNMap = Dict()
        _, truePairwiseDistances = Utils.pairwiseHammingDistance(inferenceSequences)

        for i in 1:size(truePairwiseDistances)[1]
            nns = sortperm(truePairwiseDistances[i, 1:end])[1:k]
            distances = truePairwiseDistances[i, 1:end][nns]

            # Use FAISS 0-based indexing
            nns = nns .- 1
            trueNNMap[i - 1] = Dict("topKNN" => nns, "distances" => distances)  # Index the same as FAISS indexed
        end
    end
    @info("Time to find true NNs is %s", timeFindingTrueNNs)
    return trueNNMap
end


args = JLD2.load(joinpath(EXPERIMENT_DIR, "args.jld2"))["args"]

models = JLD2.load(Utils.getBestModelPath(args.MODEL_SAVE_DIR)) |> DEVICE
embeddingModel = models["embedding_model"] |> DEVICE
calibrationModel = models["distance_calibration_model"]
trainmode!(embeddingModel, false)


inferenceSequences = getInfererenceSequences()

QUANTIZE=false

batchSequenceIterator = Channel( (channel) -> batchSequencesProducer(args, channel, inferenceSequences), 1)

faissIndex = createEmbeddingIndex(args, batchSequenceIterator, quantize=QUANTIZE)

# Make direct map if the index allows for it
if hasproperty(faissIndex, :make_direct_map)
    faissIndex.make_direct_map()
end

numNeighbours = 1000
predictedNNMap = getApproximateNNOverlap(args, faissIndex, k=numNeighbours)

trueNNMap = getTrueNNOverlap(args, k=numNeighbours)

recallDict = Utils.getTopTRecallAtK(
    ".", "test", numNN=1000,
    kStart=1, kEnd=1001, kStep=100; startIndex=2, trueIDSeqDataMap=trueNNMap, predictedIDSeqDataMap=predictedNNMap
)
