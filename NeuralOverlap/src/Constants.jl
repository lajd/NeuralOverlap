module Constants
    export ALPHABET, ALPHABET_SYMBOLS, ALPHABET_DIM, MAX_STRING_LENGTH,
     BSIZE, PROB_SAME, OUT_CHANNELS, EMBEDDING_DIM, FLAT_SIZE, NUM_BATCHES, 
     MODEL_SAVE_DIR, MODEL_SAVE_SUFFIX
    # Datasets
    ALPHABET = ['A';'T';'C';'G']

    ALPHABET_SYMBOLS = [Symbol(i) for i in ALPHABET]
    ALPHABET_DIM = length(ALPHABET_SYMBOLS)
    MAX_STRING_LENGTH = 256

    BSIZE = 512
    PROB_SAME = 0.5

    # Model
    OUT_CHANNELS = 8
    EMBEDDING_DIM = 128
    FLAT_SIZE = Int(MAX_STRING_LENGTH / 2 * ALPHABET_DIM * OUT_CHANNELS)


    # Shared
    NUM_BATCHES = 50
    MODEL_SAVE_DIR = "saved_models"
    MODEL_SAVE_SUFFIX = "synthetic.bson"
end