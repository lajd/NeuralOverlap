include("./src/ExperimentHelper.jl")
include("./src/Utils.jl")
include("./src/Datasets/DatasetUtils.jl")

include("./src/Datasets/Dataset.jl")
include("./src/Datasets/SyntheticDataset.jl")
include("./src/Datasets/SequenceDataset.jl")
include("./src/Datasets/SequenceDataset.jl")
include("./src/Models/EditCNN.jl")

# include("./args.jl")
# include("./Utils.jl")
# include("./Datasets/SyntheticDataset.jl")
# include("./EditCNN.jl")

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
using JLD2: @load, @save
using Debugger
using Plots

using Random

using Printf
using JLD2
using FileIO

using ..Utils: pairwise_hamming_distance
using ..ExperimentHelper: ExperimentParams, experiment_meter, save_experiment_args!, create_experiment_dirs!
using ..EditCNN

using ..Dataset
using ..DatasetUtils

using ..SyntheticDataset
using ..SequenceDataset


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


function get_pipe_cleaner_args()
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



function training_experiment(experimentParams::ExperimentParams;use_pipe_cleaner::Bool=false)
    """ Helpers for training an experiment """

    args = experimentParams
    use_pipe_cleaner = use_pipe_cleaner

    function get_dataset_split()
        totalSamples = args.NUM_TRAIN_EXAMPLES + args.NUM_EVAL_EXAMPLES + args.NUM_TEST_EXAMPLES
        if args.USE_SYNTHETIC_DATA == true
            all_sequences = SyntheticDataset.generate_synethetic_sequences(
                totalSamples, args.MAX_STRING_LENGTH,
                args.MAX_STRING_LENGTH, args.ALPHABET, ratio_of_random=args.RATIO_OF_RANDOM_SAMPLES,
                similarity_min=args.SIMILARITY_MIN, similarity_max=args.SIMILARITY_MAX
                )

        elseif args.USE_SEQUENCE_DATA == true
            all_sequences = SequenceDataset.read_sequence_data(totalSamples, args.MAX_STRING_LENGTH, fastq_filepath="/home/jon/JuliaProjects/NeuralOverlap/data_fetch/phix174_train.fa_R1.fastq")
        else
            throw("Must provide a valid dataset type")
        end

        # Randomly split into train/val/test
        shuffle!(all_sequences)
        @info("Full train/validation set length is %s sequences", length(all_sequences))

        train_sequences = all_sequences[1: args.NUM_TRAIN_EXAMPLES]
        n_train_eval = args.NUM_TRAIN_EXAMPLES + args.NUM_EVAL_EXAMPLES
        val_sequences = all_sequences[args.NUM_TRAIN_EXAMPLES: n_train_eval]
        test_sequences = all_sequences[n_train_eval: end]
        return train_sequences, val_sequences, test_sequences
    end

    function get_embedding_model()
        # Create the model
        embedding_model = EditCNN.get_embedding_model(
            args.MAX_STRING_LENGTH, args.ALPHABET_DIM, args.EMBEDDING_DIM,
            n_intermediate_conv_layers=args.N_INTERMEDIATE_CONV_LAYERS,
            num_readout_layers=args.N_READOUT_LAYERS, readout_activation_fn=args.READOUT_ACTIVATION,
            conv_activation_fn=args.CONV_ACTIVATION, use_batchnorm=args.USE_INTERMEDIATE_BATCHNORM,
            use_input_batchnorm=args.USE_INPUT_BATCHNORM, use_dropout=args.USE_READOUT_DROPOUT,
            c=args.OUT_CHANNELS, k=args.KERNEL_SIZE, pooling_method=args.POOLING_METHOD,
            conv_activation_fn_mod=args.CONV_ACTIVATION_LAYER_MOD, normalize_embeddings=args.L2_NORMALIZE_EMBEDDINGS
        ) |> DEVICE

        @info("---------Embedding model-------------")
        @info(embedding_model)
        return embedding_model
    end

    function evaluate!(train_data_helper, eval_data_helper, model, meter, denorm_factor, epoch, best_mean_abs_eval_error)
        @allowscalar begin
            # Training dataset
            Utils.evaluate_model(
                train_data_helper, model, denorm_factor, numNN=args.NUM_NNS_EXTRACTED,
                plot_save_path=args.PLOTS_SAVE_DIR, identifier=string("training_epoch_", epoch),
                kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                est_error_n=args.EST_ERROR_N, bSize=args.BSIZE
            )

            # Evaluation dataset
            mean_abs_eval_error, max_abs_eval_error, min_abs_eval_error,
            total_abs_eval_error, mean_estimation_error,
            epcoh_recall_dict, linear_edit_distance_model = Utils.evaluate_model(
                eval_data_helper, model, denorm_factor, numNN=args.NUM_NNS_EXTRACTED,
                plot_save_path=args.PLOTS_SAVE_DIR, identifier=string("evaluation_epoch_", epoch),
                kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                est_error_n=args.EST_ERROR_N, bSize=args.BSIZE
            )

            meter.add_experiment_validation_error!(
                mean_abs_eval_error, max_abs_eval_error, min_abs_eval_error,
                total_abs_eval_error, mean_estimation_error,
                epcoh_recall_dict, linear_edit_distance_model
            )

            experiment_loss_dict = meter.experiment_loss_dict["training_losses"]
            abs_validation_error_dict = meter.error_dict["abs_validation_error"]
            n = length(experiment_loss_dict["totalLoss"])
            m = length(abs_validation_error_dict["mean"])


            fig = plot(
                [i for i in 1:n],
                [experiment_loss_dict["totalLoss"], experiment_loss_dict["rankLoss"], experiment_loss_dict["embedding_loss"]],
                label=["training loss" "rank loss" "embedding loss"],
#                     yaxis=:log,
                xlabel="Epoch",
                ylabel="Loss"
            )
            savefig(fig, joinpath(args.PLOTS_SAVE_DIR, "training_losses.png"))

            fig = plot(
                [i * args.EVAL_EVERY for i in 1:m],
                [abs_validation_error_dict["mean"], abs_validation_error_dict["max"], abs_validation_error_dict["min"]],
                label=["Val Mean Abs Error" "Val Max Abs Error" "Val Min Abs Error"],
#                     yaxis=:log,
                xlabel="Epoch",
                ylabel="Error"
            )
            savefig(fig, joinpath(args.PLOTS_SAVE_DIR, "validation_error.png"))

            # Save the model, removing old ones
            if epoch > 1 && mean_abs_eval_error < best_mean_abs_eval_error
                SaveModelTime = @elapsed begin
                    @printf("Saving new model...\n")
                    best_mean_abs_eval_error = Float64(mean_abs_eval_error)
                    Utils.remove_old_models(args.MODEL_SAVE_DIR)
                    # Save the model
                    save_path = joinpath(args.MODEL_SAVE_DIR, string("epoch_", epoch, "_", "mean_abs_error_", mean_abs_eval_error, ".jld2"))
                    cpu_model = model |> cpu
                    @save save_path embedding_model=cpu_model distance_calibration_model=linear_edit_distance_model denorm_factor=denorm_factor
                end
                @printf("Save moodel time %s\n", SaveModelTime)
            end
        end
    end


    function train(model, train_data_helper, eval_data_helper, opt;)

        if args.DISTANCE_MATRIX_NORM_METHOD == "max"
            denorm_factor = args.MAX_STRING_LENGTH
        elseif args.DISTANCE_MATRIX_NORM_METHOD == "mean"
            denorm_factor = train_data_helper.get_mean_distance()
        else
            throw("Invalid distance matrix norm method")
        end

        meter = ExperimentHelper.experiment_meter()
        best_mean_abs_eval_error = 1e6

        @printf("Beginning training...\n")

        nbs = args.NUM_BATCHES
        # Get initial scaling for epoch
        l_reg, r_reg = EditCNN.get_loss_scaling(0, args.LOSS_STEPS_DICT, args.L0rank, args.L0emb)

        modelParams = params(model)

        for epoch in 1:args.NUM_EPOCHS

            meter.new_epoch!()

            # Get loss scaling for epoch
            l_reg, r_reg = EditCNN.get_loss_scaling(epoch, args.LOSS_STEPS_DICT, l_reg, r_reg)


            @time begin
                @info("Starting epoch %s...\n", epoch)
                # Set to train mode
                trainmode!(model, true)

                meter.epoch_timing_dict["time_spent_fetching_data"] += @elapsed begin
                    epochBatchChannel = Channel( (channel) -> train_data_helper.get_batch_tuples_producer(channel, nbs, args.BSIZE, DEVICE), 1)
                end

                for (ids_and_reads, tensorBatch) in epochBatchChannel
                    meter.new_batch!()

                    meter.epoch_timing_dict["time_spent_fetching_data"] += @elapsed begin
                        tensorBatch = tensorBatch |> DEVICE
                    end

                    meter.epoch_timing_dict["time_spent_forward"] += @elapsed begin
                        gs = gradient(modelParams) do
                            rankLoss, embedding_loss, totalLoss = EditCNN.triplet_loss(
                                args, tensorBatch..., embedding_model=model, l_reg=l_reg, r_reg=r_reg,
                            )
                            meter.add_batch_loss!(totalLoss, embedding_loss, rankLoss)
                            return totalLoss
                        end
                    end

                    meter.epoch_timing_dict["time_spend_backward"] += @elapsed begin
                        if args.DEBUG
                            meter.add_batch_gs!(Utils.validate_gradients(gs)...)
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
                meter.store_epoch_training_losses!()
                meter.log_epoch_results!(args, epoch, l_reg, r_reg)

                if args.DEBUG
                    @printf("maxGS: %s, minGS: %s, meanGS: %s\n", meter.get_epoch_gs_stats())
                end


                if mod(epoch, args.EVAL_EVERY) == 0
                    evaluateTime = @elapsed begin
                        @printf("-----Evaluation dataset-----\n")
                        trainmode!(model, false)

                        # Run evaluation
                        evaluate!(train_data_helper, eval_data_helper, model, meter, denorm_factor, epoch, best_mean_abs_eval_error)

                    end
                    @printf("Evaluation time %s\n", evaluateTime)
                end
            end
        end
    end

    function get_optimizer()
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
        if use_pipe_cleaner == true
            args = get_pipe_cleaner_args()
        end

        # Initiate experiment
        create_experiment_dirs!(args)
        save_experiment_args!(args)

        # Create embedding model
        embedding_model = get_embedding_model()

        # Get optimizer
        opt = get_optimizer()

        # Get dataset split
        trainingSequences, eval_sequences, test_sequences =  get_dataset_split()

        # Training dataset
        traindataset_helper = Dataset.dataset_helper(
            trainingSequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
            pairwise_hamming_distance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
            args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS_EXTRACTED, args.NUM_NNS_SAMPLED_DURING_TRAINING
        )

        DatasetUtils.plot_sequence_distances(traindataset_helper.get_distance_matrix(), max_samples=1000, plot_save_path=args.PLOTS_SAVE_DIR, identifier="training_dataset")
        DatasetUtils.plot_knn_distances(traindataset_helper.get_distance_matrix(), traindataset_helper.get_id_seq_data_map(), plot_save_path=args.PLOTS_SAVE_DIR, identifier="training_dataset", sampled_top_k_nns=args.NUM_NNS_SAMPLED_DURING_TRAINING)
        batchDict = traindataset_helper.get_triplet_batch(args.BSIZE)
        DatasetUtils.plot_triplet_batch_distances(batchDict, args.PLOTS_SAVE_DIR)

        evaldataset_helper = Dataset.dataset_helper(
            eval_sequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
            pairwise_hamming_distance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
            args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS_EXTRACTED, args.NUM_NNS_SAMPLED_DURING_TRAINING
        )

        DatasetUtils.plot_sequence_distances(traindataset_helper.get_distance_matrix(), max_samples=1000, plot_save_path=args.PLOTS_SAVE_DIR, identifier="eval_dataset")
        DatasetUtils.plot_knn_distances(traindataset_helper.get_distance_matrix(), traindataset_helper.get_id_seq_data_map(), plot_save_path=args.PLOTS_SAVE_DIR, identifier="eval_dataset", sampled_top_k_nns=args.NUM_NNS_SAMPLED_DURING_TRAINING)

        train(embedding_model, traindataset_helper, evaldataset_helper, opt)

    end
    () -> (execute;get_optimizer;train;get_dataset_split;evaluate)
end



function run_experiment_set(argslist::Array{ExperimentParams})
    # for args in argList
    try
        pass
    catch e
        display(stacktrace(catch_backtrace()))
        @warn("Skipping experiment: ", e)
    end
end





experiment_args = ExperimentParams(
    NUM_EPOCHS=100,
    NUM_BATCHES=512,  #
    MAX_STRING_LENGTH=64,
    BSIZE=2048,
    N_INTERMEDIATE_CONV_LAYERS=3,
    CONV_ACTIVATION=relu,
    USE_INPUT_BATCHNORM=false,
    USE_INTERMEDIATE_BATCHNORM=true,
    USE_READOUT_DROPOUT=false,
    OUT_CHANNELS = 4,
    N_READOUT_LAYERS=1,
    CONV_ACTIVATION_LAYER_MOD=1,
    LR=0.01,
    L0rank=1.,
    L0emb=0.1,
    LOSS_STEPS_DICT = Dict(),
    K_START = 1,
    K_END = 501,
    K_STEP = 10,
    NUM_NNS_EXTRACTED=500,
    NUM_NNS_SAMPLED_DURING_TRAINING=100,
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
)

experiment = training_experiment(experiment_args, use_pipe_cleaner=true)

experiment.execute()
