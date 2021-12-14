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


EXPERIMENT_DIR="/Users/jonathan/PycharmProjects/NeuralOverlap/data/experiments/2_1_relu_identity_false_false_false_mean_2_8_3_128_l2_1000_1000_5_2021-12-12T17:27:16.334"

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


# Eval Dataset
if args.USE_SYNTHETIC_DATA == true
    evalSequences = SyntheticDataset.generateSequences(
        args.NUM_EVAL_EXAMPLES, args.MAX_STRING_LENGTH,
        args.MAX_STRING_LENGTH, args.ALPHABET,ratioOfRandom=args.RATIO_OF_RANDOM_SAMPLES,
        similarityMin=args.SIMILARITY_MIN, similarityMax=args.SIMILARITY_MAX
        )
elseif args.USE_SEQUENCE_DATA == true
    evalSequences = SequenceDataset.getReadSequenceData(args.NUM_EVAL_EXAMPLES)
else
    throw("Must provide type of dataset")
end

evalDatasetHelper = Dataset.DatasetHelper(
    evalSequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
    Utils.pairwiseHammingDistance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
    args.DISTANCE_MATRIX_NORM_METHOD
)


meanAbsError, maxAbsError, minAbsError, totalAbsError, meanEstimationError, recallDict = Utils.evaluateModel(evalDatasetHelper, embeddingModel, args.MAX_STRING_LENGTH, method=args.DISTANCE_METHOD, plotsSavePath=args.PLOTS_SAVE_DIR, identifier="inference", distanceMatrixNormMethod=args.DISTANCE_MATRIX_NORM_METHOD)
