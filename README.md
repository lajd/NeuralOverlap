# NeuralOverlap

## Introduction



## Motivation



### Synthetic datasets and experiments

- In order to achieve high recall-item scores, it's important to have a dataset such that the KNNs of an example cover
  a (relatively) uniform range (similarity 0 -> 1).


## Prerequisites
- Linux with GPU (tested on Ubuntu 20.04, CUDA 11.4, nvidia driver 470.82.01)
- Julia (tested on Version 1.6.4)
- Python >= 3.6

## Installation


### Installation from source
```
git clone git@github.com:lajd/NeuralOverlap.git
cd NeuralOverlap
```


### Installing the package
Activate the environment and perform precompilation.

```shell script
julia
```

```julialang
using Pkg

Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()
```

## Preparing the dataset

### Install python dependencies
```
pip install -r requirements.txt
```

### Download viral sequence data for training/inference
Prepare the training data

```shell script
RELATIVE_RAW_GENOME_PATH=data/raw_data/full_genomes
RELATIVE_SIMULATED_READS=data/raw_data/simulated_reads

# Generate 20000 training reads samples from the Acanthamoeba castellanii mamavirus genome

GENOME_NAME=acanthamoeba_castellanii_mamavirus
mkdir -p $RELATIVE_SIMULATED_READS/$GENOME_NAME

iss generate \
  --genomes $RELATIVE_RAW_GENOME_PATH/$GENOME_NAME.fa \
  --model miseq \
  --output $RELATIVE_SIMULATED_READS/$GENOME_NAME/train \
  -n 20000

# Generate 50000 training reads sampled from the Megavirus chiliensis genome

GENOME_NAME=megavirus_chiliensis
mkdir -p $RELATIVE_SIMULATED_READS/$GENOME_NAME


iss generate \
  --genomes $RELATIVE_RAW_GENOME_PATH/$GENOME_NAME.fa \
  --model miseq \
  --output $RELATIVE_SIMULATED_READS/$GENOME_NAME/infer \
  -n 50000
```

## Run the example experiment

### Run an experiment with custom parameters

```shell script
include("src/NeuralOverlap.jl")
include("src/experiment_helper.jl")

using Flux

using .NeuralOverlap: train_eval_test, infer

function main()
    experiment_args = ExperimentParams(
        # Dataset size
        NUM_TRAIN_EXAMPLES=5000,
        NUM_EVAL_EXAMPLES=5000,
        NUM_TEST_EXAMPLES=5000,
        MAX_INFERENCE_SAMPLES=5000,
        NUM_BATCHES_PER_EPOCH=32,
        NUM_EPOCHS=1,
        USE_SYNTHETIC_DATA = false,
        USE_SEQUENCE_DATA = true,
        # Models
        N_READOUT_LAYERS = 1,
        READOUT_ACTIVATION = relu,
        N_INTERMEDIATE_CONV_LAYERS = 3,
        CONV_ACTIVATION = relu,
        USE_INPUT_BATCHNORM = false,
        USE_INTERMEDIATE_BATCHNORM = true,
        USE_READOUT_DROPOUT = false,
        CONV_ACTIVATION_LAYER_MOD = 2,
        # Pooling
        POOLING_METHOD = "mean",
        POOL_KERNEL = 2,
        # Model
        OUT_CHANNELS = 2,
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

end

main()

```
