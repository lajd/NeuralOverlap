module ExperimentHelper
    using Printf

    using Flux
    using Parameters
    using Dates

    using JLD2

    function string_from_parameters(parameters)
        s = []
        for p in parameters
            push!(s, string(p))
        end
        return join(s, "_")
    end


    function experiment_meter()

        get_loss_dict() = Dict(
            "totalLoss" => [],
            "rankLoss" => [],
            "embeddingLoss" => [],
        )

        get_error_dict() = Dict(
            "mean" => [],
            "max" => [],
            "min" => [],
            "total" => [],
        )

        error_dict = Dict(
            "absTrainingError" => get_error_dict(),
            "absValidationError" => get_error_dict(),
            "meanTrainingEstimationError" => [],
            "trainingRecallDicts" => [],
            "meanValidationEstimationError" => [],
            "validationRecallDicts" => []
        )

        experiment_loss_dict = Dict(
            "trainingLosses" => get_loss_dict(),
            "validationLosses" => get_loss_dict(),
        )

        epoch_number = 0
        evaluation_number = 0
        epoch_batch_counter = 0

        epoch_loss_dict = Dict(
            "epochRankLoss" => 0.,
            "epochEmbeddingLoss" => 0.,
            "epochTotalLoss" => 0.,
        )

        epoch_timing_dict = Dict(
            "timeSpentFetchingData" => 0.,
            "timeSpentForward" => 0.,
            "timeSpentBackward" => 0.
        )

        epoch_gs_stats = Dict(
            "min" => 0.,
            "max" => 0.,
            "mean" => 0.
        )


        function new_epoch!()
            for epochDict in (epoch_loss_dict, epoch_timing_dict)
                for (k, _) in epochDict
                    epochDict[k] = 0.
                end
            end
            epoch_number += 1
        end

        function new_batch!()
            epoch_batch_counter += 1
        end

        function add_batch_loss!(batchTotalLoss::Float64, batchEmbeddingLoss::Float64, batchRankLoss::Float64)
            epoch_loss_dict["epochRankLoss"] += batchRankLoss
            epoch_loss_dict["epochEmbeddingLoss"] += batchEmbeddingLoss
            epoch_loss_dict["epochTotalLoss"] += batchTotalLoss
        end

        function add_batch_gs!(mings::Float64, maxgs::Float64, meangs::Float64)
            epoch_gs_stats["min"] += mings
            epoch_gs_stats["max"] += maxgs
            epoch_gs_stats["mean"] += meangs
        end


        function store_epoch_training_losses!()
            push!(experiment_loss_dict["trainingLosses"]["totalLoss"], epoch_loss_dict["epochTotalLoss"]/epoch_batch_counter);
            push!(experiment_loss_dict["trainingLosses"]["rankLoss"], epoch_loss_dict["epochRankLoss"]/epoch_batch_counter);
            push!(experiment_loss_dict["trainingLosses"]["embeddingLoss"], epoch_loss_dict["epochEmbeddingLoss"]/epoch_batch_counter);
        end

        function get_experiment_losses()::Array{Float64}
            n = epoch_batch_counter
            return [l/n for l in (epoch_loss_dict["epochTotalLoss"], epoch_loss_dict["epochRankLoss"], epoch_loss_dict["epochEmbeddingLoss"])]
        end


        function get_epoch_timing_results()::Array{Float64}
            n = epoch_batch_counter
            return [r/n for r in (epoch_timing_dict["timeSpentFetchingData"], epoch_timing_dict["timeSpentForward"], epoch_timing_dict["timeSpentBackward"])]
        end

        function get_epoch_gs_stats()::Array{Float64}
            n = epoch_batch_counter
            return [s/n for s in (epoch_gs_stats["max"], epoch_gs_stats["min"], epoch_gs_stats["mean"])]
        end

        function add_experiment_training_error!(meanAbsError::Float64, maxAbsError::Float64, minAbsError::Float64, totalAbsError::Float64, meanEstimateionError::Float64, recallDict::Dict, calibrationModel)
            push!(error_dict["absTrainingError"]["mean"], meanAbsError)
            push!(error_dict["absTrainingError"]["max"], maxAbsError)
            push!(error_dict["absTrainingError"]["min"], minAbsError)
            push!(error_dict["absTrainingError"]["total"], totalAbsError)
            push!(error_dict["absTrainingError"]["total"], totalAbsError)
            push!(error_dict["meanTrainingEstimationError"], meanAbsError)
            push!(error_dict["validationRecallDicts"], meanAbsError)
        end

        function add_experiment_validation_error!(meanAbsError::Float64, maxAbsError::Float64, minAbsError::Float64, totalAbsError::Float64, meanEstimateionError::Float64, recallDict::Dict, calibrationModel)
            push!(error_dict["absValidationError"]["mean"], meanAbsError)
            push!(error_dict["absValidationError"]["max"], maxAbsError)
            push!(error_dict["absValidationError"]["min"], minAbsError)
            push!(error_dict["absValidationError"]["total"], totalAbsError)
            push!(error_dict["absValidationError"]["total"], totalAbsError)
            push!(error_dict["meanValidationEstimationError"], meanAbsError)
            push!(error_dict["validationRecallDicts"], meanAbsError)
            evaluation_number += 1
        end

        function log_epoch_results!(args, epoch, lReg, rReg)
            @printf("-----Training dataset-----\n")
            @printf("Experiment dir is: %s\n", args.EXPERIMENT_DIR)
            @printf("Epoch %s stats:\n", epoch)
            @printf("Average loss sum: %s, Average Rank loss: %s, Average Embedding loss %s\n", get_experiment_losses()...)
            @printf("lReg: %s, rReg: %s\n", lReg, rReg)
            @printf("DataFetchTime %s, TimeForward %s, TimeBackward %s\n", get_epoch_timing_results()...)
        end

        () -> (new_epoch!;add_batch_loss!;store_epoch_training_losses!;get_experiment_losses;get_epoch_timing_results;add_experiment_training_error!;add_experiment_validation_error!;
            evaluation_number;epoch_number;epoch_batch_counter;error_dict;experiment_loss_dict;epoch_loss_dict;epoch_timing_dict;log_epoch_results!;
            bestMeanAbsEvalError;add_batch_gs!;get_epoch_gs_stats;new_batch!)

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
        EXPERIMENT_NAME::String = string_from_parameters([now()])
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
