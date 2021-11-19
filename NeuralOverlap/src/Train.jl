include("/home/jon/JuliaProjects/NeuralOverlap/NeuralOverlap/src/Constants.jl")
include("/home/jon/JuliaProjects/NeuralOverlap/NeuralOverlap/src/Utils.jl")
include("/home/jon/JuliaProjects/NeuralOverlap/NeuralOverlap/src/Dataset.jl")
include("/home/jon/JuliaProjects/NeuralOverlap/NeuralOverlap/src/Model.jl")

using BenchmarkTools
using Random
using Statistics

using CUDA
CUDA.allowscalar(false)
using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy, params, update!
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using Distances
using LinearAlgebra
using Zygote
using Zygote: gradient, @ignore


using .Utils
using .Dataset
using .Constants
using .Model

DEVICE = gpu


function trainingLoop!(loss, model, trainDataset, opt, norm; numEpochs=100)
    @info("Beginning training loop...")
    local training_loss
    ps = params(model)

    for epoch in 1:numEpochs
        for (batch_idx, batch) in enumerate(trainDataset)
            gs = gradient(ps) do
                training_loss = loss(batch..., embeddingModel=embeddingModel, norm=norm)
                return training_loss
            end
            @info("Epoch: %s, Batch: %s, Training loss: %s", epoch, batch_idx, training_loss)
            update!(opt, ps, gs)
        end
    end
end


N = 50

opt = ADAM(0.001, (0.9, 0.999))
oneHotTrainingDataset = Dataset.getoneHotTrainingDataset(N,  MAX_STRING_LENGTH, PROB_SAME, ALPHABET, BSIZE) .|> DEVICE

embeddingModel = Model.getModel(MAX_STRING_LENGTH, BSIZE, FLAT_SIZE, EMBEDDING_DIM) |> DEVICE

trainingLoop!(Model.tripletLoss, embeddingModel, oneHotTrainingDataset, opt, Utils.norm2)


function approxHammingDistance(x1, x2):: Float32
    return Utils.norm2(x1 - x2, dims=1)
end


function approximateStringDistance(s1, s2)
    rawSequenceBatch = [s1, s2]
    oneHotBatch = Dataset.oneHotBatchSequences(rawSequenceBatch, MAX_STRING_LENGTH, BSIZE)
    sequenceEmbeddings = embeddingModel(oneHotBatch)[1:length(rawSequenceBatch), :]
    dist = Utils.norm2(sequenceEmbeddings[1, :] - sequenceEmbeddings[2, :], dims=1)
    return dist
end


# R1 = "ATCGCGATCGCGC"
# R2 = "ATCGCGATCGCGC"

# rawSequenceBatch = [R1, R2]

# oneHotBatch = Dataset.oneHotBatchSequences(rawSequenceBatch, MAX_STRING_LENGTH, BSIZE)

# sequenceEmbeddings = embeddingModel(oneHotBatch)[1:length(rawSequenceBatch), :]
