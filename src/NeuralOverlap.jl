module NeuralOverlap
    include("experiment_helper.jl")
    include("utils/Utils.jl")
    include("models/Models.jl")
    include("dataset/Dataset.jl")

    include("trainer.jl")
    include("inference.jl")

    function train_eval_test(experiment_args)
        @info("Running experiment $(experiment_args.EXPERIMENT_NAME)")
        @info("Experiment args are $experiment_args")

        experiment_time = @elapsed begin
            train_helper = traininghelper(experiment_args)

            @info("Beginning experiment")
            trained_embedding_model = train_helper.run_training_experiment()
            distance_calibration_model = nothing

            inference_helper = inferencehelper(
                experiment_args, trained_embedding_model,
                distance_calibration_model, is_testing=true
            )
            predicted_nn_map, true_nn_map, epoch_recall_dict, faiss_index = inference_helper.test()

        end
        @info("Exerpiment completed in $(round(experiment_time))s")
        return trained_embedding_model, distance_calibration_model, predicted_nn_map, true_nn_map, epoch_recall_dict, faiss_index
    end

    function infer(experiment_args, embedding_model, calibration_model)
        @info("Running inference $(experiment_args.EXPERIMENT_NAME)")
        inference_time = @elapsed begin
            inference_helper = inferencehelper(
                experiment_args,
                embedding_model,
                calibration_model,
                is_testing=false  # Save as mmap
            )
            predicted_nn_map, faiss_index = inference_helper.infer()
        end
        @info("Inference completed in $(inference_time)s")
        return predicted_nn_map, faiss_index
    end

    function run_experiment(experiment_args)::Nothing
        # Archive the most recent experiment if it exists
        archive_experiment(experiment_args)

        # Train/eval/test
        trained_embedding_model, distance_calibration_model,
        predicted_nn_map, true_nn_map, epoch_recall_dict,
        faiss_index = train_eval_test(experiment_args)

        # Inference
        predicted_nn_map, faiss_index = infer(
            experiment_args,
            trained_embedding_model,
            distance_calibration_model,
        )

        @info("Experiment complete; See the experiment directory for details")
        @info(joinpath(pwd(), experiment_args.LATEST_EXPERIMENT_DIR))
        return nothing
    end
end
