include("./src/Trainer.jl")
include("./src/Inference.jl")
include("./src/ExperimentHelper.jl")


module NeuralOVerlap
    using JLD2: @load
    using Flux

    using ..Trainer: trainer
    using ..Inference: inferer
    using ..ExperimentHelper: ExperimentParams, get_pipe_cleaner_args

    function run_experiment(pipe_cleaner::Bool=true)
        if pipe_cleaner == true
            experiment_args = get_pipe_cleaner_args()
        else
            experiment_args = ExperimentParams()
        end

        @info("Running experiment %s\n", experiment_args.EXPERIMENT_NAME)

        @info("Experiment args are %s\n", experiment_args)

        train_helper = trainer(experiment_args)

        trained_embedding_model = train_helper.run_training_experiment()
        distance_calibration_model = nothing

        inference_helper = inferer(experiment_args, trained_embedding_model, distance_calibration_model)
        predicted_nn_map, true_nn_map, epoch_recall_dict, faiss_index = inference_helper.execute()
    end
end
