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
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy, params, update!, trainmode!
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using Distances
using LinearAlgebra
using Zygote
using Zygote: gradient, @ignore
using BSON: @save
using JLD
using Debugger
using Plots

using Printf
using TimerOutputs

using .Utils
using .Dataset
using .Constants
using .Model

try
    using CUDA
    using CUDA: @allowscalar
    # TODO: Don't require allow scalar
    CUDA.allowscalar(Constants.ALLOW_SCALAR)
    global DEVICE = Flux.gpu
    global ArrayType = Union{Vector{Float32}, Matrix{Float32}, CuArray{Float32}}
catch e
    global DEVICE = Flux.cpu
    global ArrayType = Union{Vector{Float32}, Matrix{Float32}}
end


# Create the model save directory
mkpath(Constants.MODEL_SAVE_DIR)
mkpath(Constants.PLOTS_SAVE_DIR)

function trainingLoop!(model, trainDataHelper, evalDataHelper, opt; numEpochs=100, evalEvery=5)
    local trainingLossArray = []
    local rankLossArray = []
    local embeddingLossArray = []

    bestMeanAbsEvalError = 1e6

    ps = params(model)

    maxgsArray = [];  mingsArray = []; meangsArray = []

    @printf("Beginning training...\n")

    nbs = Constants.NUM_BATCHES
    # Get initial scaling for epoch
    lReg, rReg = Model.getLossScaling(0, Constants.LOSS_STEPS_DICT, Constants.L0rank, Constants.L0emb)
    for epoch in 1:numEpochs
        local epochRankLoss = epochEmbeddingLoss = epochTrainingLoss = 0
        local timeSpentFetchingData = timeSpentForward = timeSpentBackward = 0

        # Get loss scaling for epoch
        lReg, rReg = Model.getLossScaling(epoch, Constants.LOSS_STEPS_DICT, lReg, rReg)
        @time begin
            @info("Starting epoch %s...\n", epoch)
            # Set to train mode
            trainmode!(model, true)
            for _ in 1:nbs
                timeSpentFetchingData += @elapsed begin
                    batchDict = trainDataHelper.getTripletBatch(Constants.BSIZE)
                    batchTuple = trainDataHelper.batchToTuple(batchDict)
                    ids_and_reads = batchTuple[1:6]
                    tensorBatch = batchTuple[7:end] |> DEVICE
                end

                timeSpentForward += @elapsed begin
                    # lReg, rReg = Model.getLossScaling(epoch, regularizationSteps, lReg, rReg)
                    gs = gradient(ps) do
                        rankLoss, embeddingLoss, fullLoss = Model.tripletLoss(
                            tensorBatch..., embeddingModel=model, lReg=lReg, rReg=rReg
                        )
                        epochRankLoss += rankLoss; epochEmbeddingLoss += embeddingLoss;
                        epochTrainingLoss += fullLoss;
                        return fullLoss
                    end
                end

                timeSpentBackward += @elapsed begin
                    maxgs, mings, meangs = Utils.validateGradients(gs)
                    push!(maxgsArray, maxgs); push!(mingsArray, mings);  push!(meangsArray, meangs)
                    update!(opt, ps, gs)

                    # Terminate on NaN
                    if Utils.anynan(Flux.params(model))
                        @error("Model params NaN after update")
                        break
                    end
                end
            end

            @printf("-----Training dataset-----\n")
            @printf("Epoch %s stats:\n", epoch)
            @printf("Average loss: %s, Average Rank loss: %s, Average Embedding loss %s\n", epochTrainingLoss/nbs, epochRankLoss/nbs, epochEmbeddingLoss/nbs)
            @printf("lReg: %s, rReg: %s\n", lReg, rReg)
            @printf("DataFetchTime %s, TimeForward %s, TimeBackward %s\n", round(timeSpentFetchingData, digits=2), round(timeSpentForward, digits=2), round(timeSpentBackward, digits=2))
            @printf("maxGS: %s, minGS: %s, meanGS: %s\n", maximum(maxgsArray), minimum(mingsArray), mean(meangsArray))

            push!(trainingLossArray, round(epochTrainingLoss/nbs, digits=8)); push!(rankLossArray, round(epochEmbeddingLoss/nbs, digits=8)); push!(embeddingLossArray, round(epochRankLoss/nbs, digits=8))

            epochTrainingLoss = epochEmbeddingLoss = epochRankLoss = 0
            timeSpentFetchingData = timeSpentForward = timeSpentBackward = 0

            if mod(epoch, evalEvery) == 0
                evaluateTime = @elapsed begin
                    @printf("-----Evaluation dataset-----\n")
                    trainmode!(model, false)

                    @allowscalar begin
                        meanAbsEvalError, maxAbsEvalError, minAbsEvalError, totalAbsEvalError = Utils.evaluateModel(
                            evalDataHelper, model, Constants.MAX_STRING_LENGTH,
                            plotsSavePath=Constants.PLOTS_SAVE_DIR, identifier=epoch
                        )
                        if length(trainingLossArray) > 2
                            fig = plot(
                                1:length(trainingLossArray),
                                [trainingLossArray, rankLossArray, embeddingLossArray],
                                label=["training loss" "rank loss" "embedding loss"],
            #                     yaxis=:log,
                                xlabel="Epoch",
                                ylabel="Loss"
                            )
                            savefig(fig, joinpath(Constants.PLOTS_SAVE_DIR, "training_losses.png"))
                        end

                        # Save the model, removing old ones
                        if epoch > 1 && meanAbsEvalError < bestMeanAbsEvalError
                            SaveModelTime = @elapsed begin
                                @printf("Saving new model...\n")
                                bestMeanAbsEvalError = meanAbsEvalError
                                modelName = string("edit_cnn_epoch", "epoch_", epoch, "_", "mean_abs_error_", meanAbsEvalError, Constants.MODEL_SAVE_SUFFIX)
                                Utils.removeOldModels(Constants.MODEL_SAVE_DIR, Constants.MODEL_SAVE_SUFFIX)
                                cpuModel = model |> cpu
                                @save joinpath(Constants.MODEL_SAVE_DIR, modelName) cpuModel
                            end
                            @printf("Save moodel time %s\n", SaveModelTime)
                        end
                    end
                end
                @printf("Evaluation time %s\n", evaluateTime)
            end
        end
    end
end
    

# opt = Flux.Optimise.Optimiser(ClipValue(1e-3), ADAM(0.001, (0.9, 0.999)))
opt = ADAM(Constants.LR, (0.9, 0.999))

# Create the model
embeddingModel = Model.getModel(Constants.MAX_STRING_LENGTH, Constants.EMBEDDING_DIM, 
 numIntermediateConvLayers=Constants.NUM_INTERMEDIATE_CONV_LAYERS,
 numFCLayers=Constants.NUM_FC_LAYERS, FCAct=Constants.FC_ACTIVATION, ConvAct=Constants.CONV_ACTIVATION,
 withBatchnorm=Constants.WITH_BATCHNORM, withInputBatchnorm=Constants.WITH_INPUT_BATCHNORM,
 withDropout=Constants.WITH_DROPOUT, c=Constants.OUT_CHANNELS, k=Constants.KERNEL_SIZE,
 poolingMethod=Constants.POOLING_METHOD
 ) |> DEVICE

# Training dataset
trainDatasetHelper = Dataset.TrainingDataset(Constants.NUM_TRAINING_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
Dataset.plotSequenceDistances(trainDatasetHelper.getDistanceMatrix(), maxSamples=1000, plotsSavePath=Constants.PLOTS_SAVE_DIR)
Dataset.plotKNNDistances(trainDatasetHelper.getDistanceMatrix(), trainDatasetHelper.getIdSeqDataMap(), plotsSavePath=Constants.PLOTS_SAVE_DIR)

# Evaluation dataset
evalDatasetHelper = Dataset.TrainingDataset(Constants.NUM_EVAL_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)

trainingLoop!(embeddingModel, trainDatasetHelper, evalDatasetHelper, opt, numEpochs=Constants.NUM_EPOCHS, evalEvery=Constants.EVAL_EVERY)
