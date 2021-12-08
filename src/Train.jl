include("./src/Constants.jl")
include("./src/Utils.jl")
include("./src/Datasets/SyntheticDataset.jl")
include("./src/Model.jl")

# include("./Constants.jl")
# include("./Utils.jl")
# include("./Datasets/SyntheticDataset.jl")
# include("./Model.jl")

using BenchmarkTools
using Random
using Statistics

using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy, params, update!, trainmode!
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using Distances
using LinearAlgebra
using Zygote
using Zygote: gradient, @ignore
using BSON: @save
using JLD
using Debugger

using Printf
using TimerOutputs

using .Utils
using .Dataset
using .Constants
using .Model

try
    using CUDA
    # TODO: Don't require allow scalar
    CUDA.allowscalar(Constants.ALLOW_SCALAR)
    global DEVICE = Flux.gpu
    global ArrayType = Union{Vector{Float32}, Matrix{Float32}, CuArray{Float32}}
catch e
    global DEVICE = Flux.cpu
    global ArrayType = Union{Vector{Float32}, Matrix{Float32}}
end


# Create the model save directory
mkpath(Constants.MODEL_SAVE_DIR)


function trainingLoop!(model, trainDataHelper, evalBatches, opt; numEpochs=100, evalEvery=5)
    local trainingLoss
    local EpochRankLoss = 0
    local EpochEmbeddingLoss = 0

    bestAverageEpochLoss = 1e6

    ps = params(model)

    lReg::Float64 = 1.
    rReg::Float64 = 0.1

    regularizationSteps = Model.getLossRegularizationSteps(numEpochs)
    
    maxgsArray = []
    mingsArray = []
    meangsArray = []

    @printf("Beginning training...")
    for epoch in 1:numEpochs
        sumEpochLoss = 0

        batchNum = 0

        @time begin
            @info("Starting epoch %s...\n", epoch)
            # Set to train mode
            trainmode!(model, true)
            for _ in 1:Constants.NUM_BATCHES
                batchDict = trainDataHelper.getTripletBatch(Constants.BSIZE)
                batchTuple = trainDataHelper.batchToTuple(batchDict)

                reads = batchTuple[1:3]
                tensorBatch = batchTuple[4:end] |> DEVICE

                # lReg, rReg = Model.getRegularization(epoch, regularizationSteps, lReg, rReg)
                gs = gradient(ps) do
                    EpochRankLoss, EpochEmbeddingLoss, trainingLoss = Model.tripletLoss(tensorBatch..., embeddingModel=model, lReg=lReg, rReg=rReg)
                    EpochRankLoss += EpochRankLoss
                    EpochEmbeddingLoss += EpochEmbeddingLoss
                    return trainingLoss
                end
                
                maxgs, mings, meangs = Utils.validateGradients(gs)
                push!(maxgsArray, maxgs)
                push!(mingsArray, mings)
                push!(meangsArray, meangs)

                update!(opt, ps, gs)

                sumEpochLoss = sumEpochLoss + trainingLoss

                # Terminate on NaN
                if Utils.anynan(Flux.params(model))
                    @error("Model params NaN after update")
                    break
                end
                batchNum += 1
            end

            averageEpochLoss = sumEpochLoss / Constants.NUM_BATCHES
            @printf("-----Training dataset-----\n")
            @printf("Epoch %s stats:\n", epoch)
            @printf("Average loss: %s, lReg: %s, rReg: %s\n", averageEpochLoss, lReg, rReg)
            @printf("Average Rank loss: %s, Average Embedding loss %s\n", EpochRankLoss/Constants.NUM_BATCHES, EpochEmbeddingLoss/Constants.NUM_BATCHES)
            @printf("maxGS: %s, minGS: %s, meanGS: %s\n", maximum(maxgsArray), minimum(mingsArray), mean(meangsArray))
            EpochRankLoss = 0
            EpochEmbeddingLoss = 0


            if mod(epoch + 1, evalEvery) == 0 
                # Set to test mode
                @printf("-----Evaluation dataset-----")
                trainmode!(model, false)
                Utils.evaluateModel(evalBatches, model, Constants.MAX_STRING_LENGTH)
                # @printf("totalMSE: %s, \n averageMSEPerTriplet: %s, \n averageAbsError: %s, \nmaxAbsError: %s,\n numTriplets: %s\n",
                # totalMSE, averageMSEPerTriplet, averageAbsError, maxAbsError, numTriplets)
            end

            # Save the model, removing old ones
            if epoch > 1 && averageEpochLoss < bestAverageEpochLoss
                @printf("Saving new model...\n")
                bestAverageEpochLoss = averageEpochLoss
                modelName = string("edit_cnn_epoch", "epoch_", epoch, "_", "average_epoch_loss_", averageEpochLoss, Constants.MODEL_SAVE_SUFFIX)
                Utils.removeOldModels(Constants.MODEL_SAVE_DIR, Constants.MODEL_SAVE_SUFFIX)
                cpuModel = model |> cpu
                @save joinpath(Constants.MODEL_SAVE_DIR, modelName) cpuModel
            end
        end
    end
end
    

# opt = Flux.Optimise.Optimiser(ClipValue(1e-3), ADAM(0.001, (0.9, 0.999)))
opt = ADAM(0.001, (0.9, 0.999))
embeddingModel = Model.getModel(Constants.MAX_STRING_LENGTH, Constants.EMBEDDING_DIM, 
 numIntermediateConvLayers=Constants.NUM_INTERMEDIATE_CONV_LAYERS, numFCLayers=Constants.NUM_FC_LAYERS,
 withBatchnorm=Constants.WITH_BATCHNORM, withDropout=Constants.WITH_DROPOUT,c=Constants.OUT_CHANNELS,k=Constants.KERNEL_SIZE
 ) |> DEVICE

# if isfile(Constants.DATASET_SAVE_PATH)
#     trainingDatasetBatches = load(Constants.DATASET_SAVE_PATH,"trainingDatasetBatches",)
#     @info("Loaded %s unique triplet pairs", Constants.BSIZE * Constants.NUM_BATCHES)
# else
#     dataset = Dataset.TrainingDataset(Constants.NUM_TRAINING_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
#     trainingDatasetBatches = [dataset.getTripletBatch(Constants.BSIZE) for _ in 1:Constants.NUM_BATCHES]
#     save(Constants.DATASET_SAVE_PATH,"trainingDatasetBatches", trainingDatasetBatches)
#     @info("Number of triplet collisions encountered is: %s", dataset.numCollisions)
#     @info("Created %s unique triplet pairs", Constants.BSIZE * Constants.NUM_BATCHES)
# end

trainDatasetHelper = Dataset.TrainingDataset(Constants.NUM_TRAINING_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
Dataset.plotSequenceDistances(trainDatasetHelper.getDistanceMatrix(), maxSamples=1000)

# trainDatasetHelper = Dataset.TrainingDataset(Constants.NUM_TRAINING_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
# trainDataset = evalDataset.getTripletBatch(Constants.NUM_TRAINING_EXAMPLES * Constants.BSIZE)

# save(Constants.DATASET_SAVE_PATH,"trainingDataset", trainDatasetHelper)
@info("Number of triplet collisions encountered is: %s\n", trainDatasetHelper.numCollisions)
@info("Created %s unique triplet pairs\n", Constants.BSIZE * Constants.NUM_BATCHES)

evalDatasetHelper = Dataset.TrainingDataset(Constants.NUM_EVAL_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
evalDataset = evalDatasetHelper.getTripletBatch(Constants.NUM_EVAL_EXAMPLES)
evalDatasetHelper.shuffleTripletBatch!(evalDataset)
evalDatasetBatches = evalDatasetHelper.extractBatches(evalDataset, 1)

trainingLoop!(embeddingModel, trainDatasetHelper, evalDatasetBatches, opt, numEpochs=Constants.NUM_EPOCHS)
