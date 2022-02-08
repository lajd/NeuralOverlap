include("utils/misc.jl")

using Printf
using Flux
using Parameters
using Dates
using JLD2: @save

using NumericIO

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
    TRAIN_EVAL_FASTQ_FILEPATH = "data/raw_data/simulated_reads/acanthamoeba_castellanii_mamavirus/train_R1.fastq"
    INFERENCE_FASTQ_FILEPATH = "data/raw_data/simulated_reads/acanthamoeba_castellanii_mamavirus/train_R1.fastq"#"data/raw_data/simulated_reads/megavirus_chiliensis/infer_R1.fastq"
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
    MAX_INFERENCE_SAMPLES::Int64 = 100000

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
    EVAL_EVERY_N_EPOCHS::Int64 = 5
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
    NUM_BATCHES_PER_EPOCH::Int64 = 512

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
    INFERENCE_MMAP_SAVE_DIR::String = joinpath(EXPERIMENT_DIR, "saved_plots")
end

# Create the directories for saving the model checkpoints and plots
function create_experiment_dirs!(args)
    mkpath(args.EXPERIMENT_DIR)
    mkpath(args.MODEL_SAVE_DIR)
    mkpath(args.PLOTS_SAVE_DIR)
end

function save_experiment_args!(args)
    save_path = joinpath(args.EXPERIMENT_DIR, "args.txt")
    open(save_path, "a") do f
        write(f, string(args))
    end
    args_save_file = joinpath(args.EXPERIMENT_DIR, "args.jld2")
    @save args_save_file args=args
end

function string_from_parameters(parameters::Array)
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
        "embedding_loss" => [],
    )

    get_error_dict() = Dict(
        "mean" => [],
        "max" => [],
        "min" => [],
        "total" => [],
    )

    error_dict = Dict(
        "abs_training_error" => get_error_dict(),
        "abs_validation_error" => get_error_dict(),
        "mean_training_estimation_error" => [],
        "training_epoch_recall_dicts" => [],
        "mean_validation_estimation_error" => [],
        "validation_epoch_recall_dicts" => []
    )

    experiment_loss_dict = Dict(
        "training_losses" => get_loss_dict(),
        "validation_losses" => get_loss_dict(),
    )

    epoch_number = 0
    evaluation_number = 0
    epoch_batch_counter = 0

    epoch_loss_dict = Dict(
        "epoch_rank_loss" => 0.,
        "epoch_embedding_loss" => 0.,
        "epoch_total_loss" => 0.,
    )

    epoch_timing_dict = Dict(
        "time_spent_fetching_data" => 0.,
        "time_spent_forward" => 0.,
        "time_spend_backward" => 0.
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

    function add_batch_loss!(total_batch_loss::Float64, batch_embedding_loss::Float64, batch_rank_loss::Float64)
        epoch_loss_dict["epoch_rank_loss"] += batch_rank_loss
        epoch_loss_dict["epoch_embedding_loss"] += batch_embedding_loss
        epoch_loss_dict["epoch_total_loss"] += total_batch_loss
    end

    function add_batch_gs!(mings::Float64, maxgs::Float64, meangs::Float64)
        epoch_gs_stats["min"] += mings
        epoch_gs_stats["max"] += maxgs
        epoch_gs_stats["mean"] += meangs
    end


    function store_epoch_training_losses!()
        push!(experiment_loss_dict["training_losses"]["totalLoss"], epoch_loss_dict["epoch_total_loss"]/epoch_batch_counter);
        push!(experiment_loss_dict["training_losses"]["rankLoss"], epoch_loss_dict["epoch_rank_loss"]/epoch_batch_counter);
        push!(experiment_loss_dict["training_losses"]["embedding_loss"], epoch_loss_dict["epoch_embedding_loss"]/epoch_batch_counter);
    end

    function get_experiment_losses()::Array{Float64}
        n = epoch_batch_counter
        return [l/n for l in (epoch_loss_dict["epoch_total_loss"], epoch_loss_dict["epoch_rank_loss"], epoch_loss_dict["epoch_embedding_loss"])]
    end


    function get_epoch_timing_results()::Array{Float64}
        n = epoch_batch_counter
        return [r/n for r in (epoch_timing_dict["time_spent_fetching_data"], epoch_timing_dict["time_spent_forward"], epoch_timing_dict["time_spend_backward"])]
    end

    function get_epoch_gs_stats()::Array{Float64}
        n = epoch_batch_counter
        return [s/n for s in (epoch_gs_stats["max"], epoch_gs_stats["min"], epoch_gs_stats["mean"])]
    end


    function _add_experiment_error!(epoch_mean_abs_error::Float64, epoch_max_abs_error::Float64, epoch_min_abs_error::Float64, epoch_total_abs_error::Float64, epoch_mean_estimation_error::Float64, epoch_recall_dict::Dict, epoch_calibration_model, type::String)
        if type == "validation"
            error_dict_key = "abs_validation_error"
            estimation_error_key = "mean_validation_estimation_error"
            recall_dict_key = "training_epoch_recall_dicts"
        elseif type == "training"
            error_dict_key = "abs_training_error"
            estimation_error_key = "mean_training_estimation_error"
            recall_dict_key = "validation_epoch_recall_dicts"
        else
            throw("Invalid type")
        end
        push!(error_dict[error_dict_key]["mean"], epoch_mean_abs_error)
        push!(error_dict[error_dict_key]["max"], epoch_max_abs_error)
        push!(error_dict[error_dict_key]["min"], epoch_min_abs_error)
        push!(error_dict[error_dict_key]["total"], epoch_total_abs_error)
        push!(error_dict[error_dict_key]["total"], epoch_total_abs_error)
        push!(error_dict[estimation_error_key], epoch_mean_abs_error)
        push!(error_dict[recall_dict_key], epoch_mean_abs_error)

    end

    function add_experiment_training_error!(epoch_mean_abs_error::Float64, epoch_max_abs_error::Float64, epoch_min_abs_error::Float64, epoch_total_abs_error::Float64, epoch_mean_estimation_error::Float64, epoch_recall_dict::Dict, epoch_calibration_model)
        _add_experiment_error!(epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error, epoch_total_abs_error, epoch_mean_estimation_error, epoch_recall_dict, epoch_calibration_model, "training")
    end

    function add_experiment_validation_error!(epoch_mean_abs_error::Float64, epoch_max_abs_error::Float64, epoch_min_abs_error::Float64, epoch_total_abs_error::Float64, epoch_mean_estimation_error::Float64, epoch_recall_dict::Dict, epoch_calibration_model)
        _add_experiment_error!(epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error, epoch_total_abs_error, epoch_mean_estimation_error, epoch_recall_dict, epoch_calibration_model, "validation")
        evaluation_number += 1
    end

    function log_epoch_results!(args, epoch, l_reg, r_reg)
        losses = [toscientific(x) for x in get_experiment_losses()]
        timing = [toscientific(x) for x in get_epoch_timing_results()]
        @info("Epoch $(epoch) training results | Experiment dir: $(args.EXPERIMENT_DIR) | Average loss sum: $(losses[1]), Average Rank loss: $(losses[2]), Average Embedding loss $(losses[3]) | l_reg: $(toscientific(l_reg)), r_reg: $(toscientific(r_reg))")
        @info("DataFetchTime $(timing[1]), TimeForward $(timing[2]), TimeBackward $(timing[3])")
    end

    () -> (new_epoch!;add_batch_loss!;store_epoch_training_losses!;get_experiment_losses;get_epoch_timing_results;add_experiment_training_error!;add_experiment_validation_error!;
        evaluation_number;epoch_number;epoch_batch_counter;error_dict;experiment_loss_dict;epoch_loss_dict;epoch_timing_dict;log_epoch_results!;
        best_mean_abs_eval_error;add_batch_gs!;get_epoch_gs_stats;new_batch!)

end

function get_pipe_cleaner_args()
    return ExperimentParams(
        NUM_EPOCHS=10,
        NUM_BATCHES_PER_EPOCH=4,  #
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
