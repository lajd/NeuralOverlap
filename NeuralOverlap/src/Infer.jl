include("./Constants.jl")
include("./Utils.jl")
include("./Datasets/SyntheticDataset.jl")
include("./Model.jl")

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

using .Dataset
using .Constants
using .Model
using .Utils

try
    using CUDA
    CUDA.allowscalar(false)
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end

function LoadModel(modelPath:: String)
    @info "Loading model"
    @load modelPath model
    return model
end


function Infer(s1, s2, embeddingModel)
    sequenceBatch = [s1, s2]
    oneHotBatch = Dataset.oneHotBatchSequences(sequenceBatch, Constants.MAX_STRING_LENGTH, Constants.BSIZE)
    sequenceEmbeddings = embeddingModel(oneHotBatch)[1:length(sequenceBatch), :]
    dist = Utils.approxHammingDistance(sequenceEmbeddings[1, :], sequenceEmbeddings[2, :])
    return dist
end


embeddingModel = LoadModel(Utils.getBestModelPath(Constants.MODEL_SAVE_DIR, Constants.MODEL_SAVE_SUFFIX))



R1 = "ATCGCGATCGCGC"
R2 = "ATCGCGATCGCGC"

dist = Infer(R1, R2, embeddingModel)

@info("Embedding distance is: %s", dist)
@info("Hamming distance is: %s", hamming(R1, R2))

# rawSequenceBatch = [R1, R2]

# oneHotBatch = Dataset.oneHotBatchSequences(rawSequenceBatch, MAX_STRING_LENGTH, BSIZE)

# sequenceEmbeddings = embeddingModel(oneHotBatch)[1:length(rawSequenceBatch), :]
