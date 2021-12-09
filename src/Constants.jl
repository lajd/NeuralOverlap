module Constants
    using Flux

    DEBUG = false

    # Datasets
    ALPHABET = ['A';'T';'C';'G']

    ALPHABET_SYMBOLS = [Symbol(i) for i in ALPHABET]
    ALPHABET_DIM = length(ALPHABET_SYMBOLS)

    MIN_STRING_LENGTH = 128
    MAX_STRING_LENGTH = 128

    NUM_INTERMEDIATE_CONV_LAYERS = 4
    NUM_FC_LAYERS = 1

    BSIZE = 64

    NUM_EPOCHS = 1000

    WITH_INPUT_BATCHNORM = false
    WITH_BATCHNORM = true
    WITH_DROPOUT = true

    FC_ACTIVATION = relu
    CONV_ACTIVATION = identity
    LR = 0.1

    # Loss scaling
    L0rank = 0.1
    L0emb = 1.
    _N_LOSS_STEPS = Int32(NUM_EPOCHS / 5)
    LOSS_STEPS_DICT = Dict(
        _N_LOSS_STEPS * 0 => (0., 10.),
        _N_LOSS_STEPS * 1 => (10., 10.),
        _N_LOSS_STEPS * 2 => (10., 1.),
        _N_LOSS_STEPS * 3 => (5., 0.1),
        _N_LOSS_STEPS * 4 => (1., 0.01),
    )

    # NUM_TRAINING_EXAMPLES = 1000
    # NUM_EVAL_EXAMPLES = 500
    NUM_TRAINING_EXAMPLES = 5000
    NUM_EVAL_EXAMPLES = 150
    NUM_BATCHES = 512

    # Pooling
    POOLING_METHOD= "mean"
    POOL_KERNEL= 2

    # Model
    OUT_CHANNELS = 8
    KERNEL_SIZE = 3
    EMBEDDING_DIM = 128
    DISTANCE_METHOD="l2"

    # Shared
    MODEL_SAVE_SUFFIX = "_synthetic.bson"

    # REMOVE ME
    ALLOW_SCALAR = true

    EXPERIMENT_NAME = "3_conv_2_fc_w_drop_w_bn_cosine"
    EXPERIMENT_DIR = joinpath("data/experiments", EXPERIMENT_NAME)
    MODEL_SAVE_DIR = joinpath(EXPERIMENT_DIR, "saved_models")
    DATASET_SAVE_PATH = joinpath(EXPERIMENT_DIR, "dataset.jld")
    PLOTS_SAVE_DIR = joinpath(EXPERIMENT_DIR, "saved_plots")
    CONSTANTS_SAVE_PATH = joinpath(EXPERIMENT_DIR, "constants.txt")
end
