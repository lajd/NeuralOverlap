include("Trainer.jl")
include("Inference.jl")
include("ExperimentHelper.jl")


module NeuralOverlap
    using JLD2: @load
    using Flux

    using ..Trainer
    using ..Inference
    using ..ExperimentHelper

    function run_experiment(pipe_cleaner::Bool=true)
        if pipe_cleaner == true
            experiment_args = ExperimentHelper.get_pipe_cleaner_args()
        else
            experiment_args = ExperimentHelper.ExperimentParams()
        end

        @info("Running experiment %s\n", experiment_args.EXPERIMENT_NAME)

        @info("Experiment args are %s\n", experiment_args)

        embed_sequence_time = @elapsed begin
            train_helper = Trainer.traininghelper(experiment_args)

            trained_embedding_model = train_helper.run_training_experiment()
            distance_calibration_model = nothing

            inference_helper = Inference.inferencehelper(experiment_args, trained_embedding_model, distance_calibration_model)
            predicted_nn_map, true_nn_map, epoch_recall_dict, faiss_index = inference_helper.infer()
            @info("Exerpiment completed in %ss", embed_sequence_time)
        end

        return predicted_nn_map, true_nn_map, epoch_recall_dict, faiss_index
    end
end
