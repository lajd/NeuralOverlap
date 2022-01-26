include("./ExperimentHelper.jl")
include("./Datasets/DatasetUtils.jl")
include("./Datasets/Dataset.jl")
include("./Datasets/SyntheticDataset.jl")
include("./Datasets/SequenceDataset.jl")
include("./Datasets/SequenceDataset.jl")
include("./Models/EditCNN.jl")
include("./Utils/Misc.jl")
include("./Utils/Evaluation.jl")
include("./Utils/Loss.jl")
include("./Utils/Distance.jl")
include("./Datasets/SyntheticDataset.jl")

module Trainer

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

    using ..ExperimentHelper: ExperimentParams, experiment_meter, save_experiment_args!, create_experiment_dirs!, experiment_meter
    using ..EditCNN

    using ..Dataset
    using ..DatasetUtils

    using ..SyntheticDataset
    using ..SequenceDataset
    using ..MiscUtils
    using ..EvaluationUtils
    using ..LossUtils
    using ..DistanceUtils


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


    function trainer(experimentParams)
        """ Helpers for training an experiment """

        args = experimentParams

        function get_dataset_split()
            totalSamples = args.NUM_TRAIN_EXAMPLES + args.NUM_EVAL_EXAMPLES + args.NUM_TEST_EXAMPLES
            if args.USE_SYNTHETIC_DATA == true
                all_sequences = SyntheticDataset.generate_synethetic_sequences(
                    totalSamples, args.MAX_STRING_LENGTH,
                    args.MAX_STRING_LENGTH, args.ALPHABET, ratio_of_random=args.RATIO_OF_RANDOM_SAMPLES,
                    similarity_min=args.SIMILARITY_MIN, similarity_max=args.SIMILARITY_MAX
                    )

            elseif args.USE_SEQUENCE_DATA == true
                all_sequences = SequenceDataset.read_sequence_data(totalSamples, args.MAX_STRING_LENGTH, fastq_filepath=experimentParams.TRAIN_EVAL_FASTQ_FILEPATH)
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

        function evaluate!(train_data_helper, eval_data_helper, model::Flux.Chain, meter, denorm_factor::Float64, epoch::Int64, best_mean_abs_eval_error::Float64)
            @allowscalar begin
                # Training dataset
                EvaluationUtils.evaluate_model(
                    train_data_helper, model, denorm_factor, args.BSIZE, numNN=args.NUM_NNS_EXTRACTED,
                    plot_save_path=args.PLOTS_SAVE_DIR, identifier=string("training_epoch_", epoch),
                    kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                    est_error_n=args.EST_ERROR_N,
                )

                # Evaluation dataset
                mean_abs_eval_error, max_abs_eval_error, min_abs_eval_error,
                total_abs_eval_error, mean_estimation_error,
                epcoh_recall_dict, linear_edit_distance_model = EvaluationUtils.evaluate_model(
                    eval_data_helper, model, denorm_factor, args.BSIZE, numNN=args.NUM_NNS_EXTRACTED,
                    plot_save_path=args.PLOTS_SAVE_DIR, identifier=string("evaluation_epoch_", epoch),
                    kStart=args.K_START, kEnd=args.K_END, kStep=args.K_STEP,
                    est_error_n=args.EST_ERROR_N,
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
                        MiscUtils.remove_old_models(args.MODEL_SAVE_DIR)
                        # Save the model
                        save_path = joinpath(args.MODEL_SAVE_DIR, string("epoch_", epoch, "_", "mean_abs_error_", mean_abs_eval_error, ".jld2"))
                        cpu_model = model |> cpu
                        @save save_path embedding_model=cpu_model distance_calibration_model=linear_edit_distance_model denorm_factor=denorm_factor
                    end
                    @printf("Save moodel time %s\n", SaveModelTime)
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
                                rankLoss, embedding_loss, totalLoss = LossUtils.triplet_loss(
                                    args, tensorBatch..., embedding_model=model, l_reg=l_reg, r_reg=r_reg,
                                )
                                meter.add_batch_loss!(totalLoss, embedding_loss, rankLoss)
                                return totalLoss
                            end
                        end

                        meter.epoch_timing_dict["time_spend_backward"] += @elapsed begin
                            if args.DEBUG
                                meter.add_batch_gs!(MiscUtils.validate_gradients(gs)...)
                            end

                            update!(opt, modelParams, gs)

                            if args.TERMINATE_ON_NAN
                                # Terminate on NaN
                                if MiscUtils.anynan(modelParams)
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

            # Create embedding model
            embedding_model = get_embedding_model()

            # Get optimizer
            opt = get_optimizer()

            # Get dataset split
            trainingSequences, eval_sequences, test_sequences =  get_dataset_split()

            # Training dataset
            traindataset_helper = Dataset.dataset_helper(
                trainingSequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
                DistanceUtils.pairwise_hamming_distance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
                args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS_EXTRACTED, args.NUM_NNS_SAMPLED_DURING_TRAINING
            )

            DatasetUtils.plot_sequence_distances(traindataset_helper.get_distance_matrix(), max_samples=1000, plot_save_path=args.PLOTS_SAVE_DIR, identifier="training_dataset")
            DatasetUtils.plot_knn_distances(traindataset_helper.get_distance_matrix(), traindataset_helper.get_id_seq_data_map(), plot_save_path=args.PLOTS_SAVE_DIR, identifier="training_dataset", sampled_top_k_nns=args.NUM_NNS_SAMPLED_DURING_TRAINING)
            batchDict = traindataset_helper.get_triplet_batch(args.BSIZE)
            DatasetUtils.plot_triplet_batch_distances(batchDict, args.PLOTS_SAVE_DIR)

            evaldataset_helper = Dataset.dataset_helper(
                eval_sequences, args.MAX_STRING_LENGTH, args.ALPHABET, args.ALPHABET_SYMBOLS,
                DistanceUtils.pairwise_hamming_distance, args.KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD,
                args.DISTANCE_MATRIX_NORM_METHOD, args.NUM_NNS_EXTRACTED, args.NUM_NNS_SAMPLED_DURING_TRAINING
            )

            DatasetUtils.plot_sequence_distances(traindataset_helper.get_distance_matrix(), max_samples=1000, plot_save_path=args.PLOTS_SAVE_DIR, identifier="eval_dataset")
            DatasetUtils.plot_knn_distances(traindataset_helper.get_distance_matrix(), traindataset_helper.get_id_seq_data_map(), plot_save_path=args.PLOTS_SAVE_DIR, identifier="eval_dataset", sampled_top_k_nns=args.NUM_NNS_SAMPLED_DURING_TRAINING)

            trained_model = train_model(embedding_model, traindataset_helper, evaldataset_helper, opt)

            return trained_model

        end
        () -> (execute;get_optimizer;run_training_experiment;train;get_dataset_split;evaluate)
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
end
