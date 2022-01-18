# NeuralOverlap

## Installation

### (Optional) Installing Julia with Conda
```shell script
conda create -n NeuralOverlap python=3.9
conda activate NeuralOverlap
conda install -c conda-forge julia

# Python dependencies
pip install faiss-gpu

```

### Installing the package
```shell script
] activate .
] instantiate
] precompile
```


# Introduction

# Motivation

## Synthetic datasets and experiments

- In order to achieve high recall-item scores, it's important to have a dataset such that the KNNs of an example cover
  a (relatively) uniform range (similarity 0 -> 1).
