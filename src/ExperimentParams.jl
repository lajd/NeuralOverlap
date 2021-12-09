module ExperimentParams
    using Flux
    using Parameters

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

        ###############
        # Model Arch
        ###############
        NUM_INTERMEDIATE_CONV_LAYERS = 4
        NUM_FC_LAYERS = 1

        FC_ACTIVATION = relu
        CONV_ACTIVATION = relu

        WITH_INPUT_BATCHNORM = false
        WITH_BATCHNORM = true
        WITH_DROPOUT = true

        # Pooling
        POOLING_METHOD= "mean"
        POOL_KERNEL= 2

        # Model
        OUT_CHANNELS = 8
        KERNEL_SIZE = 3
        EMBEDDING_DIM = 128
        DISTANCE_METHOD="l2"

        ###############
        # Training
        ################
        BSIZE = 256
        NUM_EPOCHS = 50
        @assert NUM_EPOCHS > 0
        LR = 0.001
        # Evaluation
        EVAL_EVERY = 5
        # Loss scaling
        L0rank = 1.
        L0emb = 0.1
        _N_LOSS_STEPS = Int32(NUM_EPOCHS / 5)
        LOSS_STEPS_DICT = Dict(
            _N_LOSS_STEPS * 0 => (0., 10.),
            _N_LOSS_STEPS * 1 => (10., 10.),
            _N_LOSS_STEPS * 2 => (10., 1.),
            _N_LOSS_STEPS * 3 => (5., 0.1),
            _N_LOSS_STEPS * 4 => (1., 0.01),
        )

        NUM_TRAINING_EXAMPLES = 1000
        NUM_EVAL_EXAMPLES = 150
        NUM_BATCHES = 512

        ################
        # Experiment
        ################
        MODEL_SAVE_SUFFIX = "_synthetic.bson"
        EXPERIMENT_NAME = stringFromParameters(
            [NUM_INTERMEDIATE_CONV_LAYERS, NUM_FC_LAYERS, FC_ACTIVATION, CONV_ACTIVATION,
             WITH_INPUT_BATCHNORM, WITH_BATCHNORM, WITH_DROPOUT, POOLING_METHOD, POOL_KERNEL,
             OUT_CHANNELS, KERNEL_SIZE, EMBEDDING_DIM, DISTANCE_METHOD, ]
        )
        EXPERIMENT_DIR = joinpath("data/experiments", EXPERIMENT_NAME)
        MODEL_SAVE_DIR = joinpath(EXPERIMENT_DIR, "saved_models")
        DATASET_SAVE_PATH = joinpath(EXPERIMENT_DIR, "dataset.jld")
        PLOTS_SAVE_DIR = joinpath(EXPERIMENT_DIR, "saved_plots")
        CONSTANTS_SAVE_PATH = joinpath(EXPERIMENT_DIR, "constants.txt")
    end

end
