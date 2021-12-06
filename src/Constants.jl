module Constants
    # Datasets
    ALPHABET = ['A';'T';'C';'G']

    ALPHABET_SYMBOLS = [Symbol(i) for i in ALPHABET]
    ALPHABET_DIM = length(ALPHABET_SYMBOLS)

    MIN_STRING_LENGTH = 256
    MAX_STRING_LENGTH = 256

    NUM_INTERMEDIATE_CONV_LAYERS = 3
    NUM_FC_LAYERS = 2

    BSIZE = 512

    NUM_EPOCHS = 1000

    WITH_BATCHNORM = true
    WITH_DROPOUT = true

    # NUM_TRAINING_EXAMPLES = 1000
    # NUM_EVAL_EXAMPLES = 500
    NUM_TRAINING_EXAMPLES = 10000
    NUM_EVAL_EXAMPLES = 5000
    EVAL_BSIZE = 1

    # Model
    OUT_CHANNELS = 16
    KERNEL_SIZE = 3
    EMBEDDING_DIM = 128

    # Shared
    NUM_BATCHES = 128
    MODEL_SAVE_SUFFIX = "_synthetic.bson"

    # REMOVE ME
    ALLOW_SCALAR = true

    EXPERIMENT_NAME = "3_conv_2_fc_w_drop_w_bn_cosine"
    EXPERIMENT_DIR = joinpath("data/experiments", EXPERIMENT_NAME)
    MODEL_SAVE_DIR = joinpath(EXPERIMENT_DIR, "saved_models")
    DATASET_SAVE_PATH = joinpath(EXPERIMENT_DIR, "dataset.jld")
end
