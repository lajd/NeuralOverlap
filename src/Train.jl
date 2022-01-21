include("./src/ExperimentParams.jl")
include("./src/Utils.jl")
include("./src/Model.jl")

include("./src/Datasets/Dataset.jl")
include("./src/Datasets/SyntheticDataset.jl")
include("./src/Datasets/SequenceDataset.jl")

# include("./args.jl")
# include("./Utils.jl")
# include("./Datasets/SyntheticDataset.jl")
# include("./Model.jl")

using BenchmarkTools
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
using JLD2
using Debugger
using Plots

using Random

using Printf
using JLD2
using FileIO

using .Utils
using .ExperimentParams
using .Model

using .Dataset
using .SyntheticDataset
using .SequenceDataset


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

    maxgsArray = [];  mingsArray = []; meangsArray = []

    @printf("Beginning training...\n")

    nbs = args.NUM_BATCHES
    # Get initial scaling for epoch
    lReg, rReg = Model.getLossScaling(0, args.LOSS_STEPS_DICT, args.L0rank, args.L0emb)

    for epoch in 1:numEpochs
        local epochRankLoss = epochEmbeddingLoss = epochTrainingLoss = 0
        local timeSpentFetchingData = timeSpentForward = timeSpentBackward = 0

        # Get loss scaling for epoch
        lReg, rReg = Model.getLossScaling(epoch, args.LOSS_STEPS_DICT, lReg, rReg)
        @time begin
            @info("Starting epoch %s...\n", epoch)
            # Set to train mode
            trainmode!(model, true)

            timeSpentFetchingData += @elapsed begin
                epochBatchChannel = Channel( (channel) -> trainDataHelper.batchTuplesProducer(channel, nbs, args.BSIZE, DEVICE), 1)
            end
            for (ids_and_reads, tensorBatch) in epochBatchChannel
                timeSpentFetchingData += @elapsed begin
                    tensorBatch = tensorBatch |> DEVICE
                end
                timeSpentForward += @elapsed begin
                    gs = gradient(params(model)) do
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
                        push!(maxgsArray, maxgs); push!(mingsArray, mings);  push!(meangsArray, meangs)
                    end
                    update!(opt, params(model), gs)

                    if args.TERMINATE_ON_NAN
                        # Terminate on NaN
                        if Utils.anynan(Flux.params(model))
                            @error("Model params NaN after update")
                            break
                        end
                    end
                end
            end

            @printf("-----Training dataset-----\n")
            @printf("Experiment dir is: %s\n", args.EXPERIMENT_DIR)
            @printf("Epoch %s stats:\n", epoch)
            @printf("Average loss sum: %s, Average Rank loss: %s, Average Embedding loss %s\n", epochTrainingLoss/nbs, epochRankLoss/nbs, epochEmbeddingLoss/nbs)
            @printf("lReg: %s, rReg: %s\n", lReg, rReg)
            @printf("DataFetchTime %s, TimeForward %s, TimeBackward %s\n", round(timeSpentFetchingData, digits=2), round(timeSpentForward, digits=2), round(timeSpentBackward, digits=2))

            if args.DEBUG
                @printf("maxGS: %s, minGS: %s, meanGS: %s\n", maximum(maxgsArray), minimum(mingsArray), mean(meangsArray))
            end

            push!(trainingLossArray, round(epochTrainingLoss/nbs, digits=8)); push!(rankLossArray, round(epochEmbeddingLoss/nbs, digits=8)); push!(embeddingLossArray, round(epochRankLoss/nbs, digits=8))

            epochTrainingLoss = epochEmbeddingLoss = epochRankLoss = 0
            timeSpentFetchingData = timeSpentForward = timeSpentBackward = 0

            if mod(epoch, evalEvery) == 0
                evaluateTime = @elapsed begin
                    @printf("-----Evaluation dataset-----\n")
                    trainmode!(model, false)

                    @allowscalar begin

                        # Training dataset
                        Utils.evaluateModel(
                            trainDataHelper, model, args.MAX_STRING_LENGTH, numNN=args.NUM_NNS,
                            plotsSavePath=args.PLOTS_SAVE_DIR, identifier=string("training_epoch_", epoch),
                            distanceMatrixNormMethod=args.DISTANCE_MATRIX_NORM_METHOD,
                            kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                            estErrorN=args.EST_ERROR_N, bSize=args.BSIZE
                        )

                        # Evaluation dataset
                        meanAbsEvalError, maxAbsEvalError, minAbsEvalError,
                        totalAbsEvalError, meanEstimationError,
                        recallDict, linearEditDistanceModel = Utils.evaluateModel(
                            evalDataHelper, model, args.MAX_STRING_LENGTH, numNN=args.NUM_NNS,
                            plotsSavePath=args.PLOTS_SAVE_DIR, identifier=string("evaluation_epoch_", epoch),
                            distanceMatrixNormMethod=args.DISTANCE_MATRIX_NORM_METHOD,
                            kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                            estErrorN=args.EST_ERROR_N, bSize=args.BSIZE
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
                                # Save the embedding model
                                @printf("Saving new model...\n")
                                bestMeanAbsEvalError = meanAbsEvalError
                                Utils.removeOldModels(args.MODEL_SAVE_DIR)
                                emeddingModelCPU = model |> cpu

                                # Save the linear calibration model
                                modelName = string("epoch_", epoch, "_", "mean_abs_error_", meanAbsEvalError)

                               JLD2.save(
                                   joinpath(args.MODEL_SAVE_DIR, string(modelName, ".jld2")),
                                   Dict(
                                        "embedding_model" => emeddingModelCPU,
                                        "distance_calibration_model" => linearEditDistanceModel,
                                        "meanDistance", => trainDataHelper.meanDistance()
                                    )
                               )
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


function getDatasetSplit(args)
    totalSamples = args.NUM_TRAIN_EXAMPLES + args.NUM_EVAL_EXAMPLES + args.NUM_TEST_EXAMPLES
    if args.USE_SYNTHETIC_DATA == true
        allSequences = SyntheticDataset.generateSequences(
            totalSamples, args.MAX_STRING_LENGTH,
            args.MAX_STRING_LENGTH, args.ALPHABET, ratioOfRandom=args.RATIO_OF_RANDOM_SAMPLES,
            similarityMin=args.SIMILARITY_MIN, similarityMax=args.SIMILARITY_MAX
            )

    elseif args.USE_SEQUENCE_DATA == true
        allSequences = SequenceDataset.getReadSequenceData(totalSamples, args.MAX_STRING_LENGTH, fastqFilePath="/home/jon/JuliaProjects/NeuralOverlap/data_fetch/phix174_train.fa_R1.fastq")
    else
        throw("Must provide a valid dataset type")
    end

    # Randomly split into train/val/test
    shuffle!(allSequences)
    @info("Full train/validation set length is %s sequences", length(allSequences))

    trainSequences = allSequences[1: args.NUM_TRAIN_EXAMPLES]
    numTrainEval = args.NUM_TRAIN_EXAMPLES + args.NUM_EVAL_EXAMPLES
    valSequences = allSequences[args.NUM_TRAIN_EXAMPLES: numTrainEval]
    testSequences = allSequences[numTrainEval: end]
    return trainSequences, valSequences, testSequences

end


ExperimentArgs = [
        ExperimentParams.ExperimentArgs(
        NUM_EPOCHS=100,
        NUM_BATCHES=32,  #
        MAX_STRING_LENGTH=64,
        BSIZE=4096,
        NUM_INTERMEDIATE_CONV_LAYERS=4,
        CONV_ACTIVATION=relu,
        WITH_INPUT_BATCHNORM=false,
        WITH_BATCHNORM=true,
        WITH_DROPOUT=false,
        OUT_CHANNELS = 8,
        NUM_FC_LAYERS=1,
        CONV_ACTIVATION_MOD=1,
        LR=0.01,
        L0rank=1.,
        L0emb=0.1,
        LOSS_STEPS_DICT = Dict(),
#         _N_LOSS_STEPS = Int32(floor(50 / 5)),
#         LOSS_STEPS_DICT = Dict(
#             0 => (0., 1.),
#             10 => (10., 10.),
#             20 => (10., 1.),
#             30 => (5., 0.1),
#             40 => (1., 0.01),
#         ),
        K_START = 1,
        K_END = 1001,
        K_STEP = 100,
        NUM_NNS=1000,
        POOLING_METHOD="mean",
        DISTANCE_METHOD="l2",
        DISTANCE_MATRIX_NORM_METHOD="mean",
        GRADIENT_CLIP_VALUE=1,
        NUM_TRAIN_EXAMPLES=20000,
        NUM_EVAL_EXAMPLES=20000,
        NUM_TEST_EXAMPLES=200,
        KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD="uniform",
        # KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD="ranked",
        # KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD="inverseDistance",
        USE_SYNTHETIC_DATA=false,
        USE_SEQUENCE_DATA=true,
    ),
]


########
# Pairwise distances (L=32, N=20k -> 1min)
#
#
#
#

for args in ExperimentArgs
    try
        # Create the model save directory
        mkpath(args.MODEL_SAVE_DIR)
        mkpath(args.PLOTS_SAVE_DIR)
        ExperimentParams.dumpArgs(args, joinpath(args.EXPERIMENT_DIR, "args.txt"))
        argsSaveFile = File(format"JLD2", joinpath(args.EXPERIMENT_DIR, "args.jld2"))
        JLD2.save(argsSaveFile, "args", args)

#         LR = 0.1
#         baseOpt = AdaBelief(LR, (0.9, 0.999))
        baseOpt = ADAM(args.LR, (0.9, 0.999))

        opt = Flux.Optimise.Optimiser(
            baseOpt,
            ExpDecay(
                args.LR, args.EXP_DECAY_VALUE,
                args.EXP_DECAY_EVERY_N_EPOCHS * args.BSIZE, args.EXP_DECAY_CLIP
            )
        )

        if isnothing(args.GRADIENT_CLIP_VALUE)
            opt = baseOpt
        else
            opt = Flux.Optimise.Optimiser(ClipValue(args.GRADIENT_CLIP_VALUE), opt)
        end

        # Create the model
        embeddingModel = Model.getModel(
            args.MAX_STRING_LENGTH, args.ALPHABET_DIM, args.EMBEDDING_DIM,
            numIntermediateConvLayers=args.NUM_INTERMEDIATE_CONV_LAYERS,
            numFCLayers=args.NUM_FC_LAYERS, FCAct=args.FC_ACTIVATION, ConvAct=args.CONV_ACTIVATION,
            withBatchnorm=args.WITH_BATCHNORM, withInputBatchnorm=args.WITH_INPUT_BATCHNORM,
            withDropout=args.WITH_DROPOUT, c=args.OUT_CHANNELS, k=args.KERNEL_SIZE,
            poolingMethod=args.POOLING_METHOD, convActivationMod=args.CONV_ACTIVATION_MOD
        ) |> DEVICE

        @info(embeddingModel)

        # Get dataset split
        trainingSequences, evalSequences, testSequences =  getDatasetSplit(args)

        # Training dataset
        trainDatasetHelper = Dataset.DatasetHelper(
            trainingSequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
            Utils.pairwiseHammingDistance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
            args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS
        )

        Dataset.plotSequenceDistances(trainDatasetHelper.getDistanceMatrix(), maxSamples=1000, plotsSavePath=args.PLOTS_SAVE_DIR, identifier="training_dataset")
        Dataset.plotKNNDistances(trainDatasetHelper.getDistanceMatrix(), trainDatasetHelper.getIdSeqDataMap(), plotsSavePath=args.PLOTS_SAVE_DIR, identifier="training_dataset", sampledTopKNNs=args.SAMPLED_TOP_K_NEIGHBOURS)
        batchDict = trainDatasetHelper.getTripletBatch(args.BSIZE)
        Dataset.plotTripletBatchDistances(batchDict, args.PLOTS_SAVE_DIR)

        evalDatasetHelper = Dataset.DatasetHelper(
            evalSequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
            Utils.pairwiseHammingDistance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
            args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS
        )

        Dataset.plotSequenceDistances(trainDatasetHelper.getDistanceMatrix(), maxSamples=1000, plotsSavePath=args.PLOTS_SAVE_DIR, identifier="eval_dataset")
        Dataset.plotKNNDistances(trainDatasetHelper.getDistanceMatrix(), trainDatasetHelper.getIdSeqDataMap(), plotsSavePath=args.PLOTS_SAVE_DIR, identifier="eval_dataset", sampledTopKNNs=args.SAMPLED_TOP_K_NEIGHBOURS)

        trainingLoop!(args, embeddingModel, trainDatasetHelper, evalDatasetHelper, opt, numEpochs=args.NUM_EPOCHS, evalEvery=args.EVAL_EVERY)
    catch e
        display(stacktrace(catch_backtrace()))
        @warn("Skipping experiment: ", e)
    end
end
