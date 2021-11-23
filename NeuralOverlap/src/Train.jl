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


function trainingLoop!(loss, model, trainDataset, opt, norm; numEpochs=100, logInterval=10)
    @info("Beginning training loop...")
    local bestTrainingLoss
    local trainingLoss

    bestAverageEpochLoss = 1e6

    ps = params(model)

    lReg::Float64 = 1.
    rReg::Float64 = 1.

    regularizationSteps = Model.getLossRegularizationSteps(numEpochs)

    for epoch in 1:numEpochs
        sumEpochLoss = 0
        
        batchNum = 0
        for (batch_idx, batch) in enumerate(trainDataset)

            lReg, rReg = Model.getRegularization(epoch, regularizationSteps, lReg, rReg)
            gs = gradient(ps) do
                trainingLoss = loss(batch..., embeddingModel=embeddingModel, norm=norm, lReg=lReg, rReg=rReg)
                return trainingLoss
            end

            update!(opt, ps, gs)

            sumEpochLoss = sumEpochLoss + trainingLoss
            batchNum += 1

            @info("Loss %s", trainingLoss)

            if batchNum % logInterval == 0
                @info("Processed %s / %s batches in epoch", batchNum, Constants.NUM_BATCHES)
            end

        end

        averageEpochLoss = sumEpochLoss / batchNum

        @info("Epoch: %s, Average loss: %s, lReg: %s, rReg: %s", epoch, averageEpochLoss, lReg, rReg)

        # Save the model, removing old ones
        if epoch > 1 && averageEpochLoss < bestAverageEpochLoss
            @info("made it here")
            bestAverageEpochLoss = averageEpochLoss
            modelName = string("edit_cnn_epoch", "epoch_", epoch, "_", "average_epoch_loss_", averageEpochLoss, Constants.MODEL_SAVE_SUFFIX)
            @info("Saving model %s", modelName)

            Utils.removeOldModels(Constants.MODEL_SAVE_DIR, Constants.MODEL_SAVE_SUFFIX)
            @save joinpath(Constants.MODEL_SAVE_DIR, modelName)  model
        end
    end
end


opt = ADAM(0.001, (0.9, 0.999))
oneHotTrainingDataset = Dataset.getoneHotTrainingDataset(Constants.NUM_BATCHES,  Constants.MIN_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.BSIZE) .|> DEVICE


embeddingModel = Model.getModel(Constants.MAX_STRING_LENGTH, Constants.BSIZE, Constants.FLAT_SIZE, Constants.EMBEDDING_DIM, numConvLayers=Constants.NUM_CONV_LAYERS, numFCLayers=Constants.NUM_FC_LAYERS) |> DEVICE

trainingLoop!(Model.tripletLoss, embeddingModel, oneHotTrainingDataset, opt, Utils.l2Norm, numEpochs=Constants.NUM_EPOCHS)

