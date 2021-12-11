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
using JLD2
using FileIO

using .Utils
using .Dataset
using .ExperimentParams
using .Model

try
    using CUDA
    using CUDA: @allowscalar
    # TODO: Don't require allow scalar
    global DEVICE = Flux.gpu
    global ArrayType = Union{Vector{Float32}, Matrix{Float32}, CuArray{Float32}}
catch e
    global DEVICE = Flux.cpu
    global ArrayType = Union{Vector{Float32}, Matrix{Float32}}
end


function trainingLoop!(args, model, trainDataHelper, evalDataHelper, opt; numEpochs=100, evalEvery=5)
    local trainingLossArray = []
    local rankLossArray = []
    local embeddingLossArray = []

    local meanAbsEvalErrorArray = []
    local maxAbsEvalErrorArray = []
    local minAbsEvalErrorArray = []
    local totalAbsEvalErrorArray = []

    bestMeanAbsEvalError = 1e6

    ps = params(model)

    maxgsArray = [];  mingsArray = []; meangsArray = []

    @printf("Beginning training...\n")

    nbs = args.NUM_BATCHES
    # Get initial scaling for epoch
    lReg, rReg = Model.getLossScaling(0, args.LOSS_STEPS_DICT, args.L0rank, args.L0emb)

    maxChannelSize=1

    for epoch in 1:numEpochs
        local epochRankLoss = epochEmbeddingLoss = epochTrainingLoss = 0
        local timeSpentFetchingData = timeSpentForward = timeSpentBackward = 0

        # Get loss scaling for epoch
        lReg, rReg = Model.getLossScaling(epoch, args.LOSS_STEPS_DICT, lReg, rReg)
        @time begin
            @info("Starting epoch %s...\n", epoch)
            # Set to train mode
            trainmode!(model, true)
            epochBatchChannel = Channel( (channel) -> trainDataHelper.batchTuplesProducer(channel, nbs, args.BSIZE, DEVICE), maxChannelSize)
            for (ids_and_reads, tensorBatch) in epochBatchChannel
                timeSpentForward += @elapsed begin
                    gs = gradient(ps) do
                        rankLoss, embeddingLoss, fullLoss = Model.tripletLoss(
                            args, tensorBatch..., embeddingModel=model, lReg=lReg, rReg=rReg,
                        )
                        epochRankLoss += rankLoss; epochEmbeddingLoss += embeddingLoss;
                        epochTrainingLoss += fullLoss;
                        return fullLoss
                    end
                end

                timeSpentBackward += @elapsed begin
                    if args.DEBUG
                        maxgs, mings, meangs = Utils.validateGradients(gs)
                    end
                    push!(maxgsArray, maxgs); push!(mingsArray, mings);  push!(meangsArray, meangs)
                    update!(opt, ps, gs)

                    if args.DEBUG
                        # Terminate on NaN
                        if Utils.anynan(Flux.params(model))
                            @error("Model params NaN after update")
                            break
                        end
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
                            evalDataHelper, model, args.MAX_STRING_LENGTH,
                            plotsSavePath=args.PLOTS_SAVE_DIR, identifier=epoch
                        )

                        push!(meanAbsEvalErrorArray, meanAbsEvalError)
                        push!(maxAbsEvalErrorArray, maxAbsEvalError)
                        push!(minAbsEvalErrorArray, minAbsEvalError)
                        push!(totalAbsEvalErrorArray, totalAbsEvalError)

                        if length(trainingLossArray) > 2
                            fig = plot(
                                [i * evalEvery for i in 1:length(trainingLossArray)],
                                [trainingLossArray, rankLossArray, embeddingLossArray],
                                label=["training loss" "rank loss" "embedding loss"],
            #                     yaxis=:log,
                                xlabel="Epoch",
                                ylabel="Loss"
                            )
                            savefig(fig, joinpath(args.PLOTS_SAVE_DIR, "training_losses.png"))
                        end

                        if length(meanAbsEvalErrorArray) > 2
                            fig = plot(
                                [i * evalEvery for i in 1:length(meanAbsEvalErrorArray)],
                                [meanAbsEvalErrorArray, maxAbsEvalErrorArray, minAbsEvalErrorArray],
                                label=["Val Mean Abs Error" "Val Max Abs Error" "Val Min Abs Error"],
            #                     yaxis=:log,
                                xlabel="Epoch",
                                ylabel="Error"
                            )
                            savefig(fig, joinpath(args.PLOTS_SAVE_DIR, "validation_error.png"))
                        end

                        # Save the model, removing old ones
                        if epoch > 1 && meanAbsEvalError < bestMeanAbsEvalError
                            SaveModelTime = @elapsed begin
                                @printf("Saving new model...\n")
                                bestMeanAbsEvalError = meanAbsEvalError
                                modelName = string("edit_cnn_epoch", "epoch_", epoch, "_", "mean_abs_error_", meanAbsEvalError, args.MODEL_SAVE_SUFFIX)
                                Utils.removeOldModels(args.MODEL_SAVE_DIR, args.MODEL_SAVE_SUFFIX)
                                cpuModel = model |> cpu
                                @save joinpath(args.MODEL_SAVE_DIR, modelName) cpuModel
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


ExperimentArgs = [
    ExperimentParams.ExperimentArgs(
        NUM_EPOCHS=100,
        NUM_BATCHES=128,
        NUM_INTERMEDIATE_CONV_LAYERS=4,
        CONV_ACTIVATION=relu,
        WITH_INPUT_BATCHNORM=false,
        WITH_BATCHNORM=true,
        WITH_DROPOUT=true,
        NUM_FC_LAYERS=1,
        LR = 0.01,
        L0rank = 1.,
        L0emb = 0.1,
#         LOSS_STEPS_DICT = Dict(),
        NUM_TRAINING_EXAMPLES=1000,
        NUM_EVAL_EXAMPLES = 1000,
        KNN_TRIPLET_SAMPLING_METHOD="ranked"
    ),
#     ExperimentParams.ExperimentArgs(
#         NUM_EPOCHS=100,
#         NUM_BATCHES=128,
#         NUM_INTERMEDIATE_CONV_LAYERS=4,
#         CONV_ACTIVATION=identity,
#         WITH_INPUT_BATCHNORM=false,
#         WITH_BATCHNORM=true,
#         WITH_DROPOUT=true,
#         NUM_FC_LAYERS=2,
#         LR = 0.01,
#         L0rank = 1.,
#         L0emb = 0.1,
# #         LOSS_STEPS_DICT = Dict(),
#         NUM_TRAINING_EXAMPLES=10000,
#         NUM_EVAL_EXAMPLES = 2000,
#         KNN_TRIPLET_SAMPLING_METHOD="ranked"
#     ),
#     ExperimentParams.ExperimentArgs(
#         NUM_EPOCHS=100,
#         NUM_BATCHES=128,
#         NUM_INTERMEDIATE_CONV_LAYERS=2,
#         CONV_ACTIVATION=identity,
#         WITH_INPUT_BATCHNORM=false,
#         WITH_BATCHNORM=true,
#         WITH_DROPOUT=true,
#         NUM_FC_LAYERS=1,
#         LR = 0.01,
#         L0rank = 1.,
#         L0emb = 0.1,
# #         LOSS_STEPS_DICT = Dict(),
#         NUM_TRAINING_EXAMPLES=10000,
#         NUM_EVAL_EXAMPLES = 2000,
#         KNN_TRIPLET_SAMPLING_METHOD="ranked"
#     ),
]

for args in ExperimentArgs
    try
        # Create the model save directory
        mkpath(args.MODEL_SAVE_DIR)
        mkpath(args.PLOTS_SAVE_DIR)
        ExperimentParams.dumpArgs(args, joinpath(args.EXPERIMENT_DIR, "args.txt"))
        argsSaveFile = File(format"JLD2", joinpath(args.EXPERIMENT_DIR, "args.jld2"))
        JLD2.save(argsSaveFile, "args", args)

        # opt = Flux.Optimise.Optimiser(ClipValue(1e-3), ADAM(0.001, (0.9, 0.999)))
        opt = ADAM(args.LR, (0.9, 0.999))

        # Create the model
        embeddingModel = Model.getModel(args.MAX_STRING_LENGTH, 4, args.EMBEDDING_DIM,
         numIntermediateConvLayers=args.NUM_INTERMEDIATE_CONV_LAYERS,
         numFCLayers=args.NUM_FC_LAYERS, FCAct=args.FC_ACTIVATION, ConvAct=args.CONV_ACTIVATION,
         withBatchnorm=args.WITH_BATCHNORM, withInputBatchnorm=args.WITH_INPUT_BATCHNORM,
         withDropout=args.WITH_DROPOUT, c=args.OUT_CHANNELS, k=args.KERNEL_SIZE,
         poolingMethod=args.POOLING_METHOD
         ) |> DEVICE

        # Training dataset
        trainDatasetHelper = Dataset.TrainingDataset(args.NUM_TRAINING_EXAMPLES, args.MAX_STRING_LENGTH,
         args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
         Utils.pairwiseHammingDistance, args.KNN_TRIPLET_SAMPLING_METHOD)
        Dataset.plotSequenceDistances(trainDatasetHelper.getDistanceMatrix(), maxSamples=1000, plotsSavePath=args.PLOTS_SAVE_DIR)
        Dataset.plotKNNDistances(trainDatasetHelper.getDistanceMatrix(), trainDatasetHelper.getIdSeqDataMap(), plotsSavePath=args.PLOTS_SAVE_DIR)
        batchDict = trainDatasetHelper.getTripletBatch(args.BSIZE)
        Dataset.plotTripletBatchDistances(batchDict, args.PLOTS_SAVE_DIR)

        # Evaluation dataset
        evalDatasetHelper = Dataset.TrainingDataset(args.NUM_EVAL_EXAMPLES, args.MAX_STRING_LENGTH,
         args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
         Utils.pairwiseHammingDistance, args.KNN_TRIPLET_SAMPLING_METHOD)

        trainingLoop!(args, embeddingModel, trainDatasetHelper, evalDatasetHelper, opt, numEpochs=args.NUM_EPOCHS, evalEvery=args.EVAL_EVERY)
    catch e
        @warn("Skipping experiment: ", e)
    end
end
