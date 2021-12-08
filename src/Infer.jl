include("./src/Constants.jl")
include("./src/Utils.jl")
include("./src/Datasets/SyntheticDataset.jl")
include("./src/Model.jl")


# include("./Constants.jl")
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
using JLD

using .Dataset
using .Constants
using .Model
using .Utils

try
    using CUDA
    CUDA.allowscalar(Constants.ALLOW_SCALAR)
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end

function LoadModel(modelPath:: String)
    @info "Loading model"
    @load modelPath cpuModel
    model = cpuModel |> DEVICE
    return model
end


function Infer(s1, s2, embeddingModel)
    sequenceBatch = [s1, s2]
    oneHotBatch = Dataset.oneHotBatchSequences(sequenceBatch, Constants.MAX_STRING_LENGTH, Constants.BSIZE, Constants.ALPHABET_SYMBOLS)


    X, = Model.formatOneHotSequences((oneHotBatch,)) 


    X = X |> DEVICE
    sequenceEmbeddings = embeddingModel(X)

    E1 = sequenceEmbeddings[1:end, 1]
    E2 = sequenceEmbeddings[1:end, 2]

    D12 = Utils.EmbeddingDistance(E1, E2, dims=1, method=Constants.DISTANCE_METHOD) |> DEVICE  # 1D dist vector of size bsize

    predictedEditDistance = Constants.MAX_STRING_LENGTH * D12
    return D12, predictedEditDistance
end


embeddingModel = LoadModel(Utils.getBestModelPath(Constants.MODEL_SAVE_DIR, Constants.MODEL_SAVE_SUFFIX))
trainmode!(embeddingModel, false)

evalDatasetHelper = Dataset.TrainingDataset(Constants.NUM_EVAL_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
evalDataset = evalDatasetHelper.getTripletBatch(100)
evalDatasetHelper.shuffleTripletBatch!(evalDataset)
evalDatasetBatches = evalDatasetHelper.extractBatches(evalDataset, 1)

Utils.evaluateModel(evalDatasetBatches, embeddingModel, Constants.MAX_STRING_LENGTH, method=Constants.DISTANCE_METHOD)

# totalMSE, averageMSEPerTriplet, averageAbsError, maxAbsError, numTriplets = evaluateModel(evalDataset, embeddingModel)


# # dataset = Dataset.TrainingDataset(Int(Constants.NUM_TRAINING_EXAMPLES*0.3), Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
# # datasetBatches = [dataset.getTripletBatch(Constants.BSIZE) for _ in 1:50]


# s1 = Dataset.randstring(Constants.ALPHABET, Constants.MAX_STRING_LENGTH)

# s2 = s1


# function complement(char)
#     char = Symbol(char)
#     if char == Symbol("A")
#         return "T"
#     elseif char == Symbol("T")
#         return "A"
#     elseif char == Symbol("C")
#         return "G"
#     elseif char == Symbol("G")
#         return "C"
#     end
# end

# function reverseComplement(seq)
#     revComp = ""
#     for char in seq
#         revComp = string(revComp, complement(char))
#     end
#     return revComp
# end


# s3 = reverseComplement(s1)

# dist12 = Infer(s1, s2, embeddingModel)
# dist13 = Infer(s1, s3, embeddingModel)
# dist23 = Infer(s1, s3, embeddingModel)

# @info(dist12, dist13, dist23)

# # rawSequenceBatch = [R1, R2]

# # oneHotBatch = Dataset.oneHotBatchSequences(rawSequenceBatch, MAX_STRING_LENGTH, BSIZE)

# # sequenceEmbeddings = embeddingModel(oneHotBatch)[1:length(rawSequenceBatch), :]
