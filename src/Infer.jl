include("./src/ExperimentParams.jl")
include("./src/Utils.jl")
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
using BSON: @load
using JLD2

using .Dataset
using .ExperimentParams
using .Model
using .Utils

try
    using CUDA
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end


EXPERIMENT_DIR="/home/jon/JuliaProjects/NeuralOverlap/data/experiments/4_2_relu_identity_false_true_true_mean_2_8_3_128_l2_10000_2000_128_2021-12-10T22:56:20.800"

args = JLD2.load(joinpath(EXPERIMENT_DIR, "args.jld2"))["args"]


function LoadModel(modelPath:: String)
    @info "Loading model"
    @load modelPath cpuModel
    model = cpuModel |> DEVICE
    return model
end


function Infer(s1, s2, embeddingModel)
    sequenceBatch = [s1, s2]
    oneHotBatch = Dataset.oneHotBatchSequences(sequenceBatch, args.MAX_STRING_LENGTH, args.BSIZE, args.ALPHABET_SYMBOLS)


    X, = Model.formatOneHotSequences((oneHotBatch,)) 


    X = X |> DEVICE
    sequenceEmbeddings = embeddingModel(X)

    E1 = sequenceEmbeddings[1:end, 1]
    E2 = sequenceEmbeddings[1:end, 2]

    D12 = Utils.EmbeddingDistance(E1, E2, args.DISTANCE_METHOD, dims=1) |> DEVICE  # 1D dist vector of size bsize

    predictedEditDistance = args.MAX_STRING_LENGTH * D12
    return D12, predictedEditDistance
end


embeddingModel = LoadModel(Utils.getBestModelPath(args.MODEL_SAVE_DIR, args.MODEL_SAVE_SUFFIX)) |> DEVICE
trainmode!(embeddingModel, false)

evalDatasetHelper = Dataset.TrainingDataset(args.NUM_EVAL_EXAMPLES, args.MAX_STRING_LENGTH, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance, args.KNN_TRIPLET_SAMPLING_METHOD)

meanAbsError, maxAbsError, minAbsError, totalAbsError, recallDict = Utils.evaluateModel(evalDatasetHelper, embeddingModel, args.MAX_STRING_LENGTH, method=args.DISTANCE_METHOD, plotsSavePath=args.PLOTS_SAVE_DIR, identifier="inference")
