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


function trainingLoop!(model, trainDataHelper, evalDataHelper, opt; numEpochs=100, evalEvery=5)
    local trainingLoss
    local EpochRankLoss = 0
    local EpochEmbeddingLoss = 0
    local timeSpentFetchingData = 0
    local timeSpentForward = 0
    local timeSpentBackward = 0

    bestAverageEpochLoss = 1e6

    ps = params(model)

    lReg::Float64 = 1.
    rReg::Float64 = 0.1

    regularizationSteps = Model.getLossRegularizationSteps(numEpochs)
    
    maxgsArray = []
    mingsArray = []
    meangsArray = []

    @printf("Beginning training...\n")

    nbs = Constants.NUM_BATCHES
    for epoch in 1:numEpochs
        sumEpochLoss = 0
        batchNum = 0

        @time begin
            @info("Starting epoch %s...\n", epoch)
            # Set to train mode
            trainmode!(model, true)
            for _ in 1:nbs
                timeSpentFetchingData += @elapsed begin
                    batchDict = trainDataHelper.getTripletBatch(Constants.BSIZE)
                    batchTuple = trainDataHelper.batchToTuple(batchDict)
                end

                ids_and_reads = batchTuple[1:6]
                tensorBatch = batchTuple[7:end] |> DEVICE

                timeSpentForward += @elapsed begin
                    # lReg, rReg = Model.getRegularization(epoch, regularizationSteps, lReg, rReg)
                    gs = gradient(ps) do
                        EpochRankLoss, EpochEmbeddingLoss, trainingLoss = Model.tripletLoss(tensorBatch..., embeddingModel=model, lReg=lReg, rReg=rReg)
                        EpochRankLoss += EpochRankLoss
                        EpochEmbeddingLoss += EpochEmbeddingLoss
                        return trainingLoss
                    end
                end

                timeSpentBackward += @elapsed begin
                    maxgs, mings, meangs = Utils.validateGradients(gs)
                    push!(maxgsArray, maxgs)
                    push!(mingsArray, mings)
                    push!(meangsArray, meangs)
                    update!(opt, ps, gs)

                    # Terminate on NaN
                    if Utils.anynan(Flux.params(model))
                        @error("Model params NaN after update")
                        break
                    end
                end

                sumEpochLoss = sumEpochLoss + trainingLoss
                batchNum += 1
            end

            averageEpochLoss = sumEpochLoss / nbs
            @printf("-----Training dataset-----\n")
            @printf("Epoch %s stats:\n", epoch)
            @printf("Average loss: %s, lReg: %s, rReg: %s\n", averageEpochLoss, lReg, rReg)
            @printf("Average Rank loss: %s, Average Embedding loss %s\n", round(EpochRankLoss/nbs, digits=8), round(EpochEmbeddingLoss/nbs, digits=8))
            @printf("DataFetchTime %s, TimeForward %s, TimeBackward %s\n", round(timeSpentFetchingData, digits=2), round(timeSpentForward, digits=2), round(timeSpentBackward, digits=2))
            @printf("Average Rank loss: %s, Average Embedding loss %s\n", EpochRankLoss/nbs, EpochEmbeddingLoss/nbs)
            @printf("maxGS: %s, minGS: %s, meanGS: %s\n", maximum(maxgsArray), minimum(mingsArray), mean(meangsArray))
            EpochRankLoss = 0
            EpochEmbeddingLoss = 0

            if mod(epoch, evalEvery) == 1
                evaluateTime = @elapsed begin
                    # Set to test mode
                    @printf("-----Evaluation dataset-----\n")
                    trainmode!(model, false)
                    Utils.evaluateModel(evalDataHelper, model, Constants.MAX_STRING_LENGTH)
                    # @printf("totalMSE: %s, \n averageMSEPerTriplet: %s, \n averageAbsError: %s, \nmaxAbsError: %s,\n numTriplets: %s\n",
                    # totalMSE, averageMSEPerTriplet, averageAbsError, maxAbsError, numTriplets)
                end
                @printf("Evaluation time %s", evaluateTime)
            end

            # Save the model, removing old ones
            if epoch > 1 && averageEpochLoss < bestAverageEpochLoss
                SaveModelTime = @elapsed begin
                    @printf("Saving new model...\n")
                    bestAverageEpochLoss = averageEpochLoss
                    modelName = string("edit_cnn_epoch", "epoch_", epoch, "_", "average_epoch_loss_", averageEpochLoss, Constants.MODEL_SAVE_SUFFIX)
                    Utils.removeOldModels(Constants.MODEL_SAVE_DIR, Constants.MODEL_SAVE_SUFFIX)
                    cpuModel = model |> cpu
                    @save joinpath(Constants.MODEL_SAVE_DIR, modelName) cpuModel
                end
                @printf("Save moodel time %s\n", SaveModelTime)
            end
        end
    end
end
    

# opt = Flux.Optimise.Optimiser(ClipValue(1e-3), ADAM(0.001, (0.9, 0.999)))
opt = ADAM(Constants.LR, (0.9, 0.999))
embeddingModel = Model.getModel(Constants.MAX_STRING_LENGTH, Constants.EMBEDDING_DIM, 
 numIntermediateConvLayers=Constants.NUM_INTERMEDIATE_CONV_LAYERS,
  numFCLayers=Constants.NUM_FC_LAYERS, FCAct=Constants.FC_ACTIVATION, ConvAct=Constants.CONV_ACTIVATION,
 withBatchnorm=Constants.WITH_BATCHNORM, withDropout=Constants.WITH_DROPOUT,
 c=Constants.OUT_CHANNELS,k=Constants.KERNEL_SIZE, poolingMethod=Constants.POOLING_METHOD
 ) |> DEVICE

trainDatasetHelper = Dataset.TrainingDataset(Constants.NUM_TRAINING_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
Dataset.plotSequenceDistances(trainDatasetHelper.getDistanceMatrix(), maxSamples=1000)
Dataset.plotKNNDistances(trainDatasetHelper.getDistanceMatrix(), trainDatasetHelper.getIdSeqDataMap())

# trainDatasetHelper = Dataset.TrainingDataset(Constants.NUM_TRAINING_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
# trainDataset = evalDataset.getTripletBatch(Constants.NUM_TRAINING_EXAMPLES * Constants.BSIZE)

# save(Constants.DATASET_SAVE_PATH,"trainingDataset", trainDatasetHelper)
@info("Number of triplet collisions encountered is: %s\n", trainDatasetHelper.numCollisions)
@info("Created %s unique triplet pairs\n", Constants.BSIZE * Constants.NUM_BATCHES)

evalDatasetHelper = Dataset.TrainingDataset(Constants.NUM_EVAL_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
# evalDataset = evalDatasetHelper.getTripletBatch(Constants.NUM_EVAL_EXAMPLES)
# evalDatasetHelper.shuffleTripletBatch!(evalDataset)
# evalDatasetBatches = evalDatasetHelper.extractBatches(evalDataset, 1)

trainingLoop!(embeddingModel, trainDatasetHelper, evalDatasetHelper, opt, numEpochs=Constants.NUM_EPOCHS)
