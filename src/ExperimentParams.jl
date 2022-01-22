module ExperimentParams
    using Flux
    using Parameters
    using Dates

    function stringFromParameters(parameters)
        s = []
        for p in parameters
            push!(s, string(p))
        end
        return join(s, "_")
    end

    @with_kw struct ExperimentArgs
        ###############
        # Dataset
        ###############
        DEBUG = false
        ALPHABET = ['A';'T';'C';'G']
        ALPHABET_SYMBOLS = [Symbol(i) for i in ALPHABET]
        ALPHABET_DIM = length(ALPHABET_SYMBOLS)
        MIN_STRING_LENGTH = 128
        MAX_STRING_LENGTH = 128
        KNN_TRIPLET_POS_EXAMPLE_SAMPLING_METHOD="ranked"  # ranked, uniform

        NUM_TRAIN_EXAMPLES = 10000
        NUM_EVAL_EXAMPLES = 1000
        NUM_TEST_EXAMPLES = 1000

        EST_ERROR_N = 1000

        # Synthetic dataset
        USE_SYNTHETIC_DATA=true
        RATIO_OF_RANDOM_SAMPLES=0.015
        # e.g. 10k examples -> 250 random samples, ~40 sequences of similarity 0.6-0.95 each random sample
        # Note that average similarity for 4 char sequences is 25%, so we want min similarity > 0.25.
        # There will be many examples with ~0.25 similarity
        SIMILARITY_MIN=0.3
        SIMILARITY_MAX=0.95

        # Simulated sequence dataset
        USE_SEQUENCE_DATA=false

        # Distance matrix normalization metho
        DISTANCE_MATRIX_NORM_METHOD="max"

        ###############
        # Model Arch
        ###############
        NUM_INTERMEDIATE_CONV_LAYERS = 4
        NUM_FC_LAYERS = 1

        FC_ACTIVATION = relu
        CONV_ACTIVATION = identity

        WITH_INPUT_BATCHNORM = false
        WITH_BATCHNORM = false
        WITH_DROPOUT = false
        CONV_ACTIVATION_MOD = 1

        # Pooling
        POOLING_METHOD= "mean"
        POOL_KERNEL= 2

        # Model
        OUT_CHANNELS = 8
        KERNEL_SIZE = 3
        EMBEDDING_DIM = 128
        DISTANCE_METHOD="l2"

        L2_NORMALIZE_EMBEDDINGS = true

        ###############
        # Training
        ################
        BSIZE = 256
        NUM_EPOCHS = 50
        NUM_NNS = 1000
        K_START = 1
        K_END = 1001
        K_STEP = 10
        SAMPLED_TOP_K_NEIGHBOURS=100

        FAISS_TRAIN_SIZE = 5000


        @assert NUM_EPOCHS > 0
        LR = 0.01
        GRADIENT_CLIP_VALUE = nothing
        # Evaluation
        EVAL_EVERY = 5
        # Loss scaling
        L0rank = 1.
        L0emb = 0.1
        _N_LOSS_STEPS = Int32(floor(NUM_EPOCHS / 5))
        LOSS_STEPS_DICT = Dict(
            _N_LOSS_STEPS * 0 => (0., 10.),
            _N_LOSS_STEPS * 1 => (10., 10.),
            _N_LOSS_STEPS * 2 => (10., 1.),
            _N_LOSS_STEPS * 3 => (5., 0.1),
            _N_LOSS_STEPS * 4 => (1., 0.01),
        )

        # Compute num batches such that, on average, each sequence will be used once
        NUM_BATCHES = 512

        USE_EXP_DECAY = true
        EXP_DECAY_EVERY_N_EPOCHS=5
        EXP_DECAY_VALUE=0.5
        EXP_DECAY_CLIP=1e-5

        TERMINATE_ON_NAN=true

        ################
        # Experiment
        ################
        EXPERIMENT_NAME = stringFromParameters([now()])
        EXPERIMENT_DIR = joinpath("data/experiments", EXPERIMENT_NAME)
        MODEL_SAVE_DIR = joinpath(EXPERIMENT_DIR, "saved_models")
        DATASET_SAVE_PATH = joinpath(EXPERIMENT_DIR, "dataset.jld")
        PLOTS_SAVE_DIR = joinpath(EXPERIMENT_DIR, "saved_plots")
        CONSTANTS_SAVE_PATH = joinpath(EXPERIMENT_DIR, "constants.txt")
    end


    function dumpArgs(args, savePath)
        open(savePath, "a") do f
            write(f, string(args))
        end
    end
end
