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
using BSON: @save

using .Dataset
using .Constants
using .Model

try
    using CUDA
    CUDA.allowscalar(false)
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end


# Create the model save directory
mkpath(Constants.MODEL_SAVE_DIR)


function trainingLoop!(loss, model, trainDataset, opt, norm; numEpochs=100)
    @info("Beginning training loop...")
    local bestTrainingLoss
    local trainingLoss

    bestTrainingLoss = 1e6

    ps = params(model)

    for epoch in 1:numEpochs
        for (batch_idx, batch) in enumerate(trainDataset)
            gs = gradient(ps) do
                trainingLoss = loss(batch..., embeddingModel=embeddingModel, norm=norm)
                return trainingLoss
            end
            @info("Epoch: %s, Batch: %s, Training loss: %s", epoch, batch_idx, trainingLoss)

            update!(opt, ps, gs)
        end
        # Save the model
        if epoch > 1 & epoch % trainingLoss < bestTrainingLoss
            bestTrainingLoss = trainingLoss
            modelName = string("edit_cnn_epoch", "epoch_", epoch, "_", "training_loss_", bestTrainingLoss, Constants.MODEL_SAVE_SUFFIX)
            @info("Saving model %s", modelName)

            Utils.removeOldModels(Constants.MODEL_SAVE_DIR, Constants.MODEL_SAVE_SUFFIX)
            @save joinpath(Constants.MODEL_SAVE_DIR, modelName)  model
        end
    end
end


function approximateStringDistance(s1, s2)
    rawSequenceBatch = [s1, s2]
    oneHotBatch = Dataset.oneHotBatchSequences(rawSequenceBatch, Constants.MAX_STRING_LENGTH, Constants.BSIZE)
    sequenceEmbeddings = embeddingModel(oneHotBatch)[1:length(rawSequenceBatch), :]
    dist = Utils.approxHammingDistance(sequenceEmbeddings[1, :], sequenceEmbeddings[2, :])
    return dist
end


opt = ADAM(0.001, (0.9, 0.999))
oneHotTrainingDataset = Dataset.getoneHotTrainingDataset(Constants.NUM_BATCHES,  Constants.MAX_STRING_LENGTH, Constants.PROB_SAME, Constants.ALPHABET, Constants.BSIZE) .|> DEVICE

embeddingModel = Model.getModel(Constants.MAX_STRING_LENGTH, Constants.BSIZE, Constants.FLAT_SIZE, Constants.EMBEDDING_DIM) |> DEVICE

trainingLoop!(Model.tripletLoss, embeddingModel, oneHotTrainingDataset, opt, Utils.l2Norm)
