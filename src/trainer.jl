include("experiment_helper.jl")

using Parameters
using BenchmarkTools
using Statistics

using Flux
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using Zygote: gradient
using JLD2: @load, @save
using ProgressBars
using Random
using Printf
using JLD2
using FileIO
using Plots

using .Models
using .Utils
using .Dataset

try
    using CUDA
    using CUDA: @allowscalar
    # TODO: Don't require allow scalar
    global DEVICE = Flux.gpu
catch e
    global DEVICE = Flux.cpu
end


function traininghelper(experimentParams)
    """ Helpers for training an experiment """

    args = experimentParams

    function get_dataset_split()
        totalSamples = args.NUM_TRAIN_EXAMPLES + args.NUM_EVAL_EXAMPLES

        all_sequences = Dataset.get_sequences(
            args, totalSamples, fastq_filepath=args.TRAIN_EVAL_FASTQ_FILEPATH
        )

        # Randomly split into train/val/test
        shuffle!(all_sequences)

        train_sequences = all_sequences[1: args.NUM_TRAIN_EXAMPLES]
        n_train_eval = args.NUM_TRAIN_EXAMPLES + args.NUM_EVAL_EXAMPLES
        val_sequences = all_sequences[args.NUM_TRAIN_EXAMPLES: end]
        @info("Train/validation set size is $(length(all_sequences)) sequences")
        return train_sequences, val_sequences
    end

    function evaluate!(train_data_helper, eval_data_helper, model::Flux.Chain, meter, denorm_factor::Union{Float64, Int64}, epoch::Int64, best_mean_abs_eval_error::Float64)
        denorm_factor = Float64(denorm_factor)

        @allowscalar begin
            # Training dataset
            Utils.evaluate_model(
                train_data_helper, model, denorm_factor, args.BSIZE, numNN=args.NUM_NNS_EXTRACTED,
                plot_save_path=args.PLOTS_SAVE_DIR, identifier=string("training_epoch_", epoch),
                kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                est_error_n=args.EST_ERROR_N,
            )

            # Evaluation dataset
            mean_abs_eval_error, max_abs_eval_error, min_abs_eval_error,
            total_abs_eval_error, mean_estimation_error,
            epoch_recall_dict, linear_edit_distance_model = Utils.evaluate_model(
                eval_data_helper, model, denorm_factor, args.BSIZE, numNN=args.NUM_NNS_EXTRACTED,
                plot_save_path=args.PLOTS_SAVE_DIR, identifier=string("evaluation_epoch_", epoch),
                kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                est_error_n=args.EST_ERROR_N,
            )

            meter.add_experiment_validation_error!(
                mean_abs_eval_error, max_abs_eval_error, min_abs_eval_error,
                total_abs_eval_error, mean_estimation_error,
                epoch_recall_dict, linear_edit_distance_model
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
                [i * args.EVAL_EVERY_N_EPOCHS for i in 1:m],
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
                    @debug("Saving new model...")
                    best_mean_abs_eval_error = Float64(mean_abs_eval_error)
                    Utils.remove_old_models(args.MODEL_SAVE_DIR)
                    # Save the model
                    save_path = joinpath(args.MODEL_SAVE_DIR, string("epoch_", epoch, "_", "mean_abs_error_", mean_abs_eval_error, ".jld2"))
                    cpu_model = model |> cpu
                    @save save_path embedding_model=cpu_model distance_calibration_model=linear_edit_distance_model denorm_factor=denorm_factor
                end
                @debug("Saved model in: $(round(SaveModelTime))s")
            end
        end
    end


    function train_model(model::Flux.Chain, train_data_helper, eval_data_helper, opt;)::Flux.Chain

        if args.DISTANCE_MATRIX_NORM_METHOD == "max"
            denorm_factor = args.MAX_STRING_LENGTH
        elseif args.DISTANCE_MATRIX_NORM_METHOD == "mean"
            denorm_factor = train_data_helper.get_mean_distance()
        else
            throw("Invalid distance matrix norm method")
        end

        meter = experiment_meter()
        best_mean_abs_eval_error = 1e6

        @info("Beginning training on $(args.NUM_EPOCHS) epochs...")

        nbs = args.NUM_BATCHES_PER_EPOCH
        # Get initial scaling for epoch
        l_reg, r_reg = Models.get_loss_scaling(0, args.LOSS_STEPS_DICT, args.L0rank, args.L0emb)

        modelParams = Flux.params(model)

        for epoch in ProgressBar(1:args.NUM_EPOCHS)

            meter.new_epoch!()

            # Get loss scaling for epoch
            l_reg, r_reg = Models.get_loss_scaling(epoch, args.LOSS_STEPS_DICT, l_reg, r_reg)

            epoch_total_time = @elapsed begin
                # Set to train mode
                Flux.trainmode!(model, true)

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
                            rankLoss, embedding_loss, totalLoss = Utils.triplet_loss(
                                args, tensorBatch..., embedding_model=model, l_reg=l_reg, r_reg=r_reg,
                            )
                            meter.add_batch_loss!(totalLoss, embedding_loss, rankLoss)
                            return totalLoss
                        end
                    end

                    meter.epoch_timing_dict["time_spend_backward"] += @elapsed begin
                        if args.DEBUG
                            meter.add_batch_gs!(validate_gradients(gs)...)
                        end

                        Flux.update!(opt, modelParams, gs)

                        if args.TERMINATE_ON_NAN
                            # Terminate on NaN
                            if Utils.anynan(modelParams)
                                @error("Model params NaN after update")
                                break
                            end
                        end
                    end
                end
                # End of epoch
                # Store training results and log
                meter.store_epoch_training_losses!()

                if args.DEBUG
                    @info("maxGS: %s, minGS: %s, meanGS: %s", meter.get_epoch_gs_stats())
                end

                if mod(epoch, args.EVAL_EVERY_N_EPOCHS) == 0
                    evaluateTime = @elapsed begin
                        @info("-----Evaluation dataset-----")
                        Flux.trainmode!(model, false)

                        # Run evaluation
                        evaluate!(train_data_helper, eval_data_helper, model, meter, denorm_factor, epoch, best_mean_abs_eval_error)

                    end
                    @info("Total Evaluation time: $(round(evaluateTime))s")
                end
            end
            meter.log_epoch_results!(args, epoch, l_reg, r_reg, epoch_total_time)
        end

        return model
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

    function run_training_experiment()
        # Initiate experiment
        create_experiment_dirs!(args)
        save_experiment_args!(args)

        # # Create embedding model
        # embedding_model = Models.get_embedding_model()

        # Create the model
        embedding_model = Models.get_embedding_model(
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

        # Get optimizer
        opt = get_optimizer()

        # Get dataset split
        trainingSequences, eval_sequences =  get_dataset_split()

        @info("---------Training/validation dataset split-------------")
        # Training dataset
        traindataset_helper = Dataset.dataset_helper(
            trainingSequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
            Utils.pairwise_hamming_distance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
            args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS_EXTRACTED, args.NUM_NNS_SAMPLED_DURING_TRAINING
        )

        Dataset.plot_sequence_distances(traindataset_helper.get_distance_matrix(), max_samples=1000, plot_save_path=args.PLOTS_SAVE_DIR, identifier="training_dataset")
        Dataset.plot_knn_distances(traindataset_helper.get_distance_matrix(), traindataset_helper.get_id_seq_data_map(), plot_save_path=args.PLOTS_SAVE_DIR, identifier="training_dataset", sampled_top_k_nns=args.NUM_NNS_SAMPLED_DURING_TRAINING)
        batchDict = traindataset_helper.get_triplet_batch(args.BSIZE)
        Dataset.plot_triplet_batch_distances(batchDict, args.PLOTS_SAVE_DIR)

        evaldataset_helper = Dataset.dataset_helper(
            eval_sequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
            Utils.pairwise_hamming_distance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
            args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS_EXTRACTED, args.NUM_NNS_SAMPLED_DURING_TRAINING
        )

        Dataset.plot_sequence_distances(traindataset_helper.get_distance_matrix(), max_samples=1000, plot_save_path=args.PLOTS_SAVE_DIR, identifier="eval_dataset")
        Dataset.plot_knn_distances(traindataset_helper.get_distance_matrix(), traindataset_helper.get_id_seq_data_map(), plot_save_path=args.PLOTS_SAVE_DIR, identifier="eval_dataset", sampled_top_k_nns=args.NUM_NNS_SAMPLED_DURING_TRAINING)

        @info("Training dataset size: $(length(trainingSequences))")
        @info("Evaluation dataset size: $(length(eval_sequences))")

        trained_model = train_model(embedding_model, traindataset_helper, evaldataset_helper, opt)

        return trained_model

    end
    () -> (execute;get_optimizer;run_training_experiment;train;get_dataset_split;evaluate)
end
