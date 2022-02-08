include("src/experiment_helper.jl")
include("src/utils/Utils.jl")
include("src/models/Models.jl")
include("src/dataset/Dataset.jl")

include("src/trainer.jl")
include("src/inference.jl")

include("src/NeuralOverlap.jl")
include("src/experiment_helper.jl")

using Flux

using .NeuralOverlap: train_eval_test, infer

function main()::Nothing
    experiment_args = ExperimentParams(
        # Dataset size
        NUM_TRAIN_EXAMPLES=10000,
        NUM_EVAL_EXAMPLES=5000,
        NUM_TEST_EXAMPLES=10000,
        MAX_INFERENCE_SAMPLES=50000,
        NUM_BATCHES_PER_EPOCH=128,
        NUM_EPOCHS=50,
        USE_SYNTHETIC_DATA = false,
        USE_SEQUENCE_DATA = true,
        # Models
        N_READOUT_LAYERS = 1,
        READOUT_ACTIVATION = relu,
        N_INTERMEDIATE_CONV_LAYERS = 3,
        CONV_ACTIVATION = identity,
        USE_INPUT_BATCHNORM = false,
        USE_INTERMEDIATE_BATCHNORM = true,
        USE_READOUT_DROPOUT = false,
        CONV_ACTIVATION_LAYER_MOD = 2,
        # Pooling
        POOLING_METHOD = "mean",
        POOL_KERNEL = 2,
        # Model
        OUT_CHANNELS = 8,
        KERNEL_SIZE = 3,
        EMBEDDING_DIM = 128,
        DISTANCE_METHOD ="l2",
    )

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
    @info(joinpath(pwd(), experiment_args.EXPERIMENT_DIR))
    return nothing
end

main()
