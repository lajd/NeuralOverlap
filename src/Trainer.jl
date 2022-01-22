include("./src/ExperimentHelper.jl")
include("./src/Utils.jl")
include("./src/Model.jl")

include("./src/Datasets/Dataset.jl")
include("./src/Datasets/SyntheticDataset.jl")
include("./src/Datasets/SequenceDataset.jl")

# include("./args.jl")
# include("./Utils.jl")
# include("./Datasets/SyntheticDataset.jl")
# include("./Model.jl")

using Parameters
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

using .Utils: pairwiseHammingDistance
using .ExperimentHelper: ExperimentParams, ExperimentMeter, saveArgs!, createExperimentDirs!
using .Model

using .Dataset: DatasetHelper, plotSequenceDistances, plotKNNDistances, plotTripletBatchDistances
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

function getPipeCleanerArgs()
    return ExperimentParams(
        NUM_EPOCHS=10,
        NUM_BATCHES=4,  #
        MAX_STRING_LENGTH=64,
        BSIZE=64,
        N_INTERMEDIATE_CONV_LAYERS=2,
        CONV_ACTIVATION=relu,
        USE_INPUT_BATCHNORM=false,
        USE_INTERMEDIATE_BATCHNORM=true,
        USE_READOUT_DROPOUT=false,
        OUT_CHANNELS = 2,
        N_READOUT_LAYERS=1,
        CONV_ACTIVATION_LAYER_MOD=1,
        L0emb=0.1,
        LOSS_STEPS_DICT = Dict(),
        K_START = 1,
        K_END = 101,
        K_STEP = 1,
        NUM_NNS_EXTRACTED=100,
        NUM_NNS_SAMPLED_DURING_TRAINING=100,  # This must be less than NUM_NNS_EXTRACTED
        POOLING_METHOD="mean",
        DISTANCE_METHOD="l2",
        DISTANCE_MATRIX_NORM_METHOD="mean",
        GRADIENT_CLIP_VALUE=1,
        NUM_TRAIN_EXAMPLES=500,
        NUM_EVAL_EXAMPLES=500,
        NUM_TEST_EXAMPLES=500,
        USE_SYNTHETIC_DATA=false,
        USE_SEQUENCE_DATA=true,
    )
end



function TrainingExperiment(experimentParams::ExperimentParams)

    args = experimentParams

    function getDatasetSplit()
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

    function getModel()
        # Create the model
        embeddingModel = Model.getModel(
            args.MAX_STRING_LENGTH, args.ALPHABET_DIM, args.EMBEDDING_DIM,
            numIntermediateConvLayers=args.N_INTERMEDIATE_CONV_LAYERS,
            numFCLayers=args.N_READOUT_LAYERS, FCAct=args.READOUT_ACTIVATION, ConvAct=args.CONV_ACTIVATION,
            withBatchnorm=args.USE_INTERMEDIATE_BATCHNORM, withInputBatchnorm=args.USE_INPUT_BATCHNORM,
            withDropout=args.USE_READOUT_DROPOUT, c=args.OUT_CHANNELS, k=args.KERNEL_SIZE,
            poolingMethod=args.POOLING_METHOD, convActivationMod=args.CONV_ACTIVATION_LAYER_MOD, normalizeEmbeddings=args.L2_NORMALIZE_EMBEDDINGS
        ) |> DEVICE

        @info("---------Embedding model-------------")
        @info(embeddingModel)
        return embeddingModel
    end

    function evaluate!(trainDataHelper, evalDataHelper, model, meter, denormFactor, epoch, bestMeanAbsEvalError)
        @allowscalar begin
            # Training dataset
            Utils.evaluateModel(
                trainDataHelper, model, denormFactor, numNN=args.NUM_NNS_EXTRACTED,
                plotsSavePath=args.PLOTS_SAVE_DIR, identifier=string("training_epoch_", epoch),
                kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                estErrorN=args.EST_ERROR_N, bSize=args.BSIZE
            )

            # Evaluation dataset
            meanAbsEvalError, maxAbsEvalError, minAbsEvalError,
            totalAbsEvalError, meanEstimationError,
            recallDict, linearEditDistanceModel = Utils.evaluateModel(
                evalDataHelper, model, denormFactor, numNN=args.NUM_NNS_EXTRACTED,
                plotsSavePath=args.PLOTS_SAVE_DIR, identifier=string("evaluation_epoch_", epoch),
                kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                estErrorN=args.EST_ERROR_N, bSize=args.BSIZE
            )

            meter.addValidationError!(
                meanAbsEvalError, maxAbsEvalError, minAbsEvalError,
                totalAbsEvalError, meanEstimationError,
                recallDict, linearEditDistanceModel
            )

            experimentLossDict = meter.experimentLossDict["trainingLosses"]
            absValidationErrorDict = meter.errorDict["absValidationError"]
            n = length(experimentLossDict["totalLoss"])
            m = length(absValidationErrorDict["mean"])


            fig = plot(
                [i for i in 1:n],
                [experimentLossDict["totalLoss"], experimentLossDict["rankLoss"], experimentLossDict["embeddingLoss"]],
                label=["training loss" "rank loss" "embedding loss"],
#                     yaxis=:log,
                xlabel="Epoch",
                ylabel="Loss"
            )
            savefig(fig, joinpath(args.PLOTS_SAVE_DIR, "training_losses.png"))

            fig = plot(
                [i * args.EVAL_EVERY for i in 1:m],
                [absValidationErrorDict["mean"], absValidationErrorDict["max"], absValidationErrorDict["min"]],
                label=["Val Mean Abs Error" "Val Max Abs Error" "Val Min Abs Error"],
#                     yaxis=:log,
                xlabel="Epoch",
                ylabel="Error"
            )
            savefig(fig, joinpath(args.PLOTS_SAVE_DIR, "validation_error.png"))

            # Save the model, removing old ones
            if epoch > 1 && meanAbsEvalError < bestMeanAbsEvalError
                SaveModelTime = @elapsed begin
                    @printf("Saving new model...\n")
                    bestMeanAbsEvalError = Float64(meanAbsEvalError)
                    Utils.removeOldModels(args.MODEL_SAVE_DIR)
                    # Save the model
                    JLD2.save(
                       joinpath(args.MODEL_SAVE_DIR, string("epoch_", epoch, "_", "mean_abs_error_", meanAbsEvalError, ".jld2")),
                       Dict(
                            "embedding_model" => model |> cpu,
                            "distance_calibration_model" => linearEditDistanceModel,
                            "denormFactor" => denormFactor
                        )
                    )
                end
                @printf("Save moodel time %s\n", SaveModelTime)
            end
        end
    end


    function train(model, trainDataHelper, evalDataHelper, opt;)
        if args.DISTANCE_MATRIX_NORM_METHOD == "max"
            denormFactor = args.MAX_STRING_LENGTH
        elseif args.DISTANCE_MATRIX_NORM_METHOD == "mean"
            denormFactor = trainDataHelper.getMeanDistance()
        else
            throw("Invalid distance matrix norm method")
        end

        if args.DISTANCE_MATRIX_NORM_METHOD == "max"
            denormFactor = args.MAX_STRING_LENGTH
        elseif args.DISTANCE_MATRIX_NORM_METHOD == "mean"
            denormFactor = trainDataHelper.getMeanDistance()
        else
            throw("Invalid distance matrix norm method")
        end

        meter = ExperimentHelper.ExperimentMeter()
        bestMeanAbsEvalError = 1e6

        @printf("Beginning training...\n")

        nbs = args.NUM_BATCHES
        # Get initial scaling for epoch
        lReg, rReg = Model.getLossScaling(0, args.LOSS_STEPS_DICT, args.L0rank, args.L0emb)

        modelParams = params(model)

        for epoch in 1:args.NUM_EPOCHS

            meter.newEpoch!()

            # Get loss scaling for epoch
            lReg, rReg = Model.getLossScaling(epoch, args.LOSS_STEPS_DICT, lReg, rReg)


            @time begin
                @info("Starting epoch %s...\n", epoch)
                # Set to train mode
                trainmode!(model, true)

                meter.epochTimingDict["timeSpentFetchingData"] += @elapsed begin
                    epochBatchChannel = Channel( (channel) -> trainDataHelper.batchTuplesProducer(channel, nbs, args.BSIZE, DEVICE), 1)
                end

                for (ids_and_reads, tensorBatch) in epochBatchChannel
                    meter.newBatch!()

                    meter.epochTimingDict["timeSpentFetchingData"] += @elapsed begin
                        tensorBatch = tensorBatch |> DEVICE
                    end

                    meter.epochTimingDict["timeSpentForward"] += @elapsed begin
                        gs = gradient(modelParams) do
                            rankLoss, embeddingLoss, totalLoss = Model.tripletLoss(
                                args, tensorBatch..., embeddingModel=model, lReg=lReg, rReg=rReg,
                            )
                            meter.addBatchLoss!(totalLoss, embeddingLoss, rankLoss)
                            return totalLoss
                        end
                    end

                    meter.epochTimingDict["timeSpentBackward"] += @elapsed begin
                        if args.DEBUG
                            meter.addBatchGS!(Utils.validateGradients(gs)...)
                        end

                        update!(opt, modelParams, gs)

                        if args.TERMINATE_ON_NAN
                            # Terminate on NaN
                            if Utils.anynan(modelParams)
                                @error("Model params NaN after update")
                                break
                            end
                        end
                    end
                end

                @info("Finished epoch %s...\n", epoch)


                # End of epoch
                # Store training results and log
                meter.storeTrainingEpochLosses!()
                meter.logEpochResults!(args, epoch, lReg, rReg)

                if args.DEBUG
                    @printf("maxGS: %s, minGS: %s, meanGS: %s\n", meter.getEpochStatsGS())
                end


                if mod(epoch, args.EVAL_EVERY) == 0
                    evaluateTime = @elapsed begin
                        @printf("-----Evaluation dataset-----\n")
                        trainmode!(model, false)

                        # Run evaluation
                        evaluate!(trainDataHelper, evalDataHelper, model, meter, denormFactor, epoch, bestMeanAbsEvalError)

                    end
                    @printf("Evaluation time %s\n", evaluateTime)
                end
            end
        end
    end

    function getOptimizer()
        opt = ADAM(args.LR, (0.9, 0.999))


        if args.USE_EXP_DECAY == true
            opt = Flux.Optimise.Optimiser(
                opt,
                ExpDecay(
                    args.LR,
                    args.EXP_DECAY_VALUE,
                    args.EXP_DECAY_EVERY_N_EPOCHS * args.BSIZE, args.EXP_DECAY_CLIP
                )
            )
        end

        if args.GRADIENT_CLIP_VALUE != nothing
            opt = Flux.Optimise.Optimiser(ClipValue(args.GRADIENT_CLIP_VALUE), opt)
        end
        return opt
    end

    function execute()
        USE_PIPE_CLEANER = true

        if USE_PIPE_CLEANER == true
            args = getPipeCleanerArgs()
        end

        # Initiate experiment
        createExperimentDirs!(args)
        saveArgs!(args)

        # Create embedding model
        embeddingModel = getModel()

        # Get optimizer
        opt = getOptimizer()

        # Get dataset split
        trainingSequences, evalSequences, testSequences =  getDatasetSplit()

        # Training dataset
        trainDatasetHelper = DatasetHelper(
            trainingSequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
            pairwiseHammingDistance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
            args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS_EXTRACTED, args.NUM_NNS_SAMPLED_DURING_TRAINING
        )

        plotSequenceDistances(trainDatasetHelper.getDistanceMatrix(), maxSamples=1000, plotsSavePath=args.PLOTS_SAVE_DIR, identifier="training_dataset")
        plotKNNDistances(trainDatasetHelper.getDistanceMatrix(), trainDatasetHelper.getIdSeqDataMap(), plotsSavePath=args.PLOTS_SAVE_DIR, identifier="training_dataset", sampledTopKNNs=args.NUM_NNS_SAMPLED_DURING_TRAINING)
        batchDict = trainDatasetHelper.getTripletBatch(args.BSIZE)
        plotTripletBatchDistances(batchDict, args.PLOTS_SAVE_DIR)

        evalDatasetHelper = DatasetHelper(
            evalSequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
            pairwiseHammingDistance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
            args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS_EXTRACTED, args.NUM_NNS_SAMPLED_DURING_TRAINING
        )

        plotSequenceDistances(trainDatasetHelper.getDistanceMatrix(), maxSamples=1000, plotsSavePath=args.PLOTS_SAVE_DIR, identifier="eval_dataset")
        plotKNNDistances(trainDatasetHelper.getDistanceMatrix(), trainDatasetHelper.getIdSeqDataMap(), plotsSavePath=args.PLOTS_SAVE_DIR, identifier="eval_dataset", sampledTopKNNs=args.NUM_NNS_SAMPLED_DURING_TRAINING)

        train(embeddingModel, trainDatasetHelper, evalDatasetHelper, opt)

    end
    () -> (execute;getOptimizer;train;getDatasetSplit;evaluate)
end



function RunExperimentSet(argList::Array{ExperimentParams})
    # for args in argList
    try
        pass
    catch e
        display(stacktrace(catch_backtrace()))
        @warn("Skipping experiment: ", e)
    end
end





experimentArgs = ExperimentParams(
    NUM_EPOCHS=20,
    NUM_BATCHES=512,  #
    MAX_STRING_LENGTH=64,
    BSIZE=2048,
    N_INTERMEDIATE_CONV_LAYERS=2,
    CONV_ACTIVATION=relu,
    USE_INPUT_BATCHNORM=false,
    USE_INTERMEDIATE_BATCHNORM=true,
    USE_READOUT_DROPOUT=false,
    OUT_CHANNELS = 2,
    N_READOUT_LAYERS=1,
    CONV_ACTIVATION_LAYER_MOD=1,
    LR=0.01,
    L0rank=1.,
    L0emb=0.1,
    LOSS_STEPS_DICT = Dict(),
    K_START = 1,
    K_END = 15001,
    K_STEP = 100,
    NUM_NNS_EXTRACTED=15000,
    NUM_NNS_SAMPLED_DURING_TRAINING=200,
    POOLING_METHOD="mean",
    DISTANCE_METHOD="l2",
    DISTANCE_MATRIX_NORM_METHOD="mean",
    GRADIENT_CLIP_VALUE=1,
    NUM_TRAIN_EXAMPLES=20000,
    NUM_EVAL_EXAMPLES=20000,
    NUM_TEST_EXAMPLES=200,
    # KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD="uniform",
    KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD="ranked",
    # KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD="inverseDistance",
    USE_SYNTHETIC_DATA=false,
    USE_SEQUENCE_DATA=true,
)

experiment = TrainingExperiment(experimentArgs)

experiment.execute()
