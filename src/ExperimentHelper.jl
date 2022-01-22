module ExperimentHelper
    using Printf

    using Flux
    using Parameters
    using Dates

    using JLD2

    function stringFromParameters(parameters)
        s = []
        for p in parameters
            push!(s, string(p))
        end
        return join(s, "_")
    end


    function ExperimentMeter()

        getLossDict() = Dict(
            "totalLoss" => [],
            "rankLoss" => [],
            "embeddingLoss" => [],
        )

        getErrorDict() = Dict(
            "mean" => [],
            "max" => [],
            "min" => [],
            "total" => [],
        )

        errorDict = Dict(
            "absTrainingError" => getErrorDict(),
            "absValidationError" => getErrorDict(),
            "meanTrainingEstimationError" => [],
            "trainingRecallDicts" => [],
            "meanValidationEstimationError" => [],
            "validationRecallDicts" => []
        )

        experimentLossDict = Dict(
            "trainingLosses" => getLossDict(),
            "validationLosses" => getLossDict(),
        )

        epochNumber = 0
        evaluationNumber = 0
        epochBatchCount = 0

        epochLossDict = Dict(
            "epochRankLoss" => 0.,
            "epochEmbeddingLoss" => 0.,
            "epochTotalLoss" => 0.,
        )

        epochTimingDict = Dict(
            "timeSpentFetchingData" => 0.,
            "timeSpentForward" => 0.,
            "timeSpentBackward" => 0.
        )

        epochStatsGS = Dict(
            "min" => 0.,
            "max" => 0.,
            "mean" => 0.
        )


        function newEpoch!()
            for epochDict in (epochLossDict, epochTimingDict)
                for (k, _) in epochDict
                    epochDict[k] = 0.
                end
            end
            epochNumber += 1
        end

        function newBatch!()
            epochBatchCount += 1
        end

        function addBatchLoss!(batchTotalLoss::Float64, batchEmbeddingLoss::Float64, batchRankLoss::Float64)
            epochLossDict["epochRankLoss"] += batchRankLoss
            epochLossDict["epochEmbeddingLoss"] += batchEmbeddingLoss
            epochLossDict["epochTotalLoss"] += batchTotalLoss
        end

        function addBatchGS!(mings::Float64, maxgs::Float64, meangs::Float64)
            epochStatsGS["min"] += mings
            epochStatsGS["max"] += maxgs
            epochStatsGS["mean"] += meangs
        end


        function storeTrainingEpochLosses!()
            push!(experimentLossDict["trainingLosses"]["totalLoss"], epochLossDict["epochTotalLoss"]/epochBatchCount);
            push!(experimentLossDict["trainingLosses"]["rankLoss"], epochLossDict["epochRankLoss"]/epochBatchCount);
            push!(experimentLossDict["trainingLosses"]["embeddingLoss"], epochLossDict["epochEmbeddingLoss"]/epochBatchCount);
        end

        function getEpochLosses()::Array{Float64}
            n = epochBatchCount
            return [l/n for l in (epochLossDict["epochTotalLoss"], epochLossDict["epochRankLoss"], epochLossDict["epochEmbeddingLoss"])]
        end


        function getEpochTimeResults()::Array{Float64}
            n = epochBatchCount
            return [r/n for r in (epochTimingDict["timeSpentFetchingData"], epochTimingDict["timeSpentForward"], epochTimingDict["timeSpentBackward"])]
        end

        function getEpochStatsGS()::Array{Float64}
            n = epochBatchCount
            return [s/n for s in (epochStatsGS["max"], epochStatsGS["min"], epochStatsGS["mean"])]
        end

        function addTrainingError!(meanAbsError::Float64, maxAbsError::Float64, minAbsError::Float64, totalAbsError::Float64, meanEstimateionError::Float64, recallDict::Dict, calibrationModel)
            push!(errorDict["absTrainingError"]["mean"], meanAbsError)
            push!(errorDict["absTrainingError"]["max"], maxAbsError)
            push!(errorDict["absTrainingError"]["min"], minAbsError)
            push!(errorDict["absTrainingError"]["total"], totalAbsError)
            push!(errorDict["absTrainingError"]["total"], totalAbsError)
            push!(errorDict["meanTrainingEstimationError"], meanAbsError)
            push!(errorDict["validationRecallDicts"], meanAbsError)
        end

        function addValidationError!(meanAbsError::Float64, maxAbsError::Float64, minAbsError::Float64, totalAbsError::Float64, meanEstimateionError::Float64, recallDict::Dict, calibrationModel)
            push!(errorDict["absValidationError"]["mean"], meanAbsError)
            push!(errorDict["absValidationError"]["max"], maxAbsError)
            push!(errorDict["absValidationError"]["min"], minAbsError)
            push!(errorDict["absValidationError"]["total"], totalAbsError)
            push!(errorDict["absValidationError"]["total"], totalAbsError)
            push!(errorDict["meanValidationEstimationError"], meanAbsError)
            push!(errorDict["validationRecallDicts"], meanAbsError)
            evaluationNumber += 1
        end

        function logEpochResults!(args, epoch, lReg, rReg)
            @printf("-----Training dataset-----\n")
            @printf("Experiment dir is: %s\n", args.EXPERIMENT_DIR)
            @printf("Epoch %s stats:\n", epoch)
            @printf("Average loss sum: %s, Average Rank loss: %s, Average Embedding loss %s\n", getEpochLosses()...)
            @printf("lReg: %s, rReg: %s\n", lReg, rReg)
            @printf("DataFetchTime %s, TimeForward %s, TimeBackward %s\n", getEpochTimeResults()...)
        end

        () -> (newEpoch!;addBatchLoss!;storeTrainingEpochLosses!;getEpochLosses;getEpochTimeResults;addTrainingError!;addValidationError!;
            evaluationNumber;epochNumber;epochBatchCount;errorDict;experimentLossDict;epochLossDict;epochTimingDict;logEpochResults!;
            bestMeanAbsEvalError;addBatchGS!;getEpochStatsGS;newBatch!)

    end


    @with_kw struct ExperimentParams
        # Modes
        DEBUG::Bool = false  # Run in rebug mode

        ###############
        # Dataset
        ###############
        # Type of dataset to user (from file or synthetic)
        ## Synthetic dataset
        USE_SYNTHETIC_DATA::Bool = true
        ## Simulated sequence dataset
        USE_SEQUENCE_DATA::Bool = false

        # Sequences
        ## Alphabet for genomic data
        ALPHABET::Vector{Char} = ['A';'T';'C';'G']
        ALPHABET_SYMBOLS = [Symbol(i) for i in ALPHABET]
        ALPHABET_DIM::Int64 = length(ALPHABET_SYMBOLS)

        # Min/Max generated sequence length for synthetic data
        # Or the prefix/suffix length to use for real data
        MIN_STRING_LENGTH::Int64 = 128
        MAX_STRING_LENGTH::Int64 = 128

        # Dataset sampling method
        KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD::String = "ranked"  # ranked, uniform

        # Synthetic dataset args
        RATIO_OF_RANDOM_SAMPLES::Float64 = 0.015
        ## e.g. 10k examples -> 250 random samples, ~40 sequences of similarity 0.6-0.95 each random sample
        ## Note that average similarity for 4 char sequences is 25%, so we want min similarity > 0.25.
        ## There will be many examples with ~0.25 similarity
        SIMILARITY_MIN::Float64 = 0.3
        SIMILARITY_MAX::Float64 = 0.95

        ###############
        # Model Arch
        ###############
        N_READOUT_LAYERS::Int64 = 1
        READOUT_ACTIVATION = relu

        N_INTERMEDIATE_CONV_LAYERS::Int64 = 4
        CONV_ACTIVATION = identity

        USE_INPUT_BATCHNORM::Bool = false
        USE_INTERMEDIATE_BATCHNORM::Bool = false
        USE_READOUT_DROPOUT::Bool = false
        CONV_ACTIVATION_LAYER_MOD::Int64 = 1

        # Pooling
        POOLING_METHOD::String = "mean"
        POOL_KERNEL::Int64 = 2

        # Model
        OUT_CHANNELS::Int64 = 8
        KERNEL_SIZE::Int64 = 3
        EMBEDDING_DIM::Int64 = 128
        DISTANCE_METHOD::String ="l2"

        L2_NORMALIZE_EMBEDDINGS = true

        ###############
        # Training
        ################
        # Distance matrix normalization method
        DISTANCE_MATRIX_NORM_METHOD::String = "max"

        # Dataset size
        NUM_TRAIN_EXAMPLES::Int64 = 1000
        NUM_EVAL_EXAMPLES::Int64 = 1000
        NUM_TEST_EXAMPLES::Int64 = 1000

        # Number of samples to use for error estimation
        EST_ERROR_N::Int64 = 1000


        BSIZE::Int64 = 256
        NUM_EPOCHS::Int64 = 50
        NUM_NNS_EXTRACTED::Int64 = 1000
        K_START::Int64 = 1
        K_END::Int64 = 1001
        K_STEP::Int64 = 10
        NUM_NNS_SAMPLED_DURING_TRAINING::Int64 = 100

        FAISS_TRAIN_SIZE::Int64 = 5000


        @assert NUM_EPOCHS > 0
        LR::Float64 = 0.01
        GRADIENT_CLIP_VALUE = nothing
        # Evaluation
        EVAL_EVERY::Int64 = 5
        # Loss scaling
        L0rank::Float64 = 1.
        L0emb::Float64 = 0.1
        _N_LOSS_STEPS = Int32(floor(NUM_EPOCHS / 5))
        LOSS_STEPS_DICT = Dict(
            _N_LOSS_STEPS * 0 => (0., 10.),
            _N_LOSS_STEPS * 1 => (10., 10.),
            _N_LOSS_STEPS * 2 => (10., 1.),
            _N_LOSS_STEPS * 3 => (5., 0.1),
            _N_LOSS_STEPS * 4 => (1., 0.01),
        )

        # Compute num batches such that, on average, each sequence will be used once
        NUM_BATCHES::Int64 = 512

        USE_EXP_DECAY::Bool = true
        EXP_DECAY_EVERY_N_EPOCHS::Int64 = 5
        EXP_DECAY_VALUE::Float64 = 0.5
        EXP_DECAY_CLIP::Float64 = 1e-5

        TERMINATE_ON_NAN::Bool = true

        ################
        # Experiment
        ################
        EXPERIMENT_NAME::String = stringFromParameters([now()])
        EXPERIMENT_DIR::String = joinpath("data/experiments", EXPERIMENT_NAME)
        MODEL_SAVE_DIR::String = joinpath(EXPERIMENT_DIR, "saved_models")
        DATASET_SAVE_PATH::String = joinpath(EXPERIMENT_DIR, "dataset.jld")
        PLOTS_SAVE_DIR::String = joinpath(EXPERIMENT_DIR, "saved_plots")
        CONSTANTS_SAVE_PATH::String = joinpath(EXPERIMENT_DIR, "constants.txt")
    end

    # Create the directories for saving the model checkpoints and plots
    function createExperimentDirs!(args)
        mkpath(args.EXPERIMENT_DIR)
        mkpath(args.MODEL_SAVE_DIR)
        mkpath(args.PLOTS_SAVE_DIR)
    end

    function saveArgs!(args)
        savePath = joinpath(args.EXPERIMENT_DIR, "args.txt")
        open(savePath, "a") do f
            write(f, string(args))
        end
        argsSaveFile = joinpath(args.EXPERIMENT_DIR, "args.jld2")
        JLD2.save(argsSaveFile, Dict("args" => args))
    end
end
