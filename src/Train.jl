include("./src/Utils.jl")
include("./src/Constants.jl")
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


function evaluateModel(evalBatches, model)

    totalMSE = 0
    numTriplets = 0

    averageAbsErrorArray = []
    maxAbsErrorArray = []
    totalAbsError = 0 


    for fullBatch in evalBatches
        # fullBatch = dataHelper.getTripletBatch(Constants.BSIZE)  
        # trainDataHelper.shuffleTripletBatch!(fullBatch)
        # fullBatch = trainDataHelper.extractBatchSubsets(fullBatch, 5)

        reads = fullBatch[1:3]
        batch = fullBatch[4:end]# |> DEVICE
    
        batch = batch |> DEVICE
    
        Xacr, Xpos, Xneg, y12, y13, y23 = batch
    
        Eacr = model(Xacr)
        Epos = model(Xpos)
        Eneg = model(Xneg)
    
        # MSE
        posEmbedDist = Utils.Norm(Eacr, Epos, dims=1) |> DEVICE # 1D dist vector of size bsize
        negEmbedDist =  Utils.Norm(Eacr, Eneg, dims=1) |> DEVICE
        PosNegEmbedDist =  Utils.Norm(Epos, Eneg, dims=1) |> DEVICE

        # @assert maximum(posEmbedDist) <= 1
        @assert maximum(y12) <= 1

        d12 = abs.(posEmbedDist - y12) * Constants.MAX_STRING_LENGTH
        d13 = abs.(negEmbedDist - y13) * Constants.MAX_STRING_LENGTH
        d23 = abs.(PosNegEmbedDist - y23) * Constants.MAX_STRING_LENGTH

        totalAbsError += sum(d12) + sum(d13) + sum(d23)
        
        averageAbsError = mean([mean(d12), mean(d13), mean(d23)])
        push!(averageAbsErrorArray, averageAbsError)
        maxAbsError = maximum([maximum(d12), maximum(d13), maximum(d23)])
        push!(maxAbsErrorArray, maxAbsError)

        MSE = ((posEmbedDist - y12).^2 + (negEmbedDist - y13).^2 + (PosNegEmbedDist - y23).^2) 
        MSE = mean(sqrt.(MSE))
        totalMSE += MSE
        numTriplets += size(Xacr)[3]
    end

    averageAbsError = mean(averageAbsErrorArray)
    maxAbsError = maximum(maxAbsErrorArray)

    averageMSEPerTriplet = totalMSE / numTriplets
    @printf("Total MSE on entire training dataset is %s \n", totalMSE)
    @printf("Average MSE per triplet is %s \n", round(averageMSEPerTriplet, digits=4))
    @printf("Average abs error is %s \n", round(averageAbsError, digits=4))
    @printf("Max abs error is %s \n", round(maxAbsError, digits=4))
    @printf("Total abs error is %s \n", totalAbsError, )
    @printf("Number of triplets compared %s \n", numTriplets)

    return totalMSE, averageMSEPerTriplet, averageAbsError, maxAbsError, numTriplets
end

function trainingLoop!(model, trainDataHelper, evalBatches, opt; numEpochs=100)
    local trainingLoss

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

                lReg, rReg = Model.getRegularization(epoch, regularizationSteps, lReg, rReg)
                gs = gradient(ps) do
                    trainingLoss = Model.tripletLoss(tensorBatch..., embeddingModel=model, lReg=lReg, rReg=rReg)
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

            averageEpochLoss = sumEpochLoss / batchNum
            @printf("-----Training dataset-----\n")
            @printf("Epoch %s stats:\n", epoch)
            @printf("Average loss: %s, lReg: %s, rReg: %s\n", averageEpochLoss, lReg, rReg)
            @printf("maxGS: %s, minGS: %s, meanGS: %s\n", maximum(maxgsArray), minimum(mingsArray), mean(meangsArray))

            # Set to test mode
            trainmode!(model, false)
            totalMSE, averageMSEPerTriplet, averageAbsError, maxAbsError, numTriplets = evaluateModel(evalBatches, model)
            @printf("-----Evaluation dataset-----")
            @printf("totalMSE: %s, averageMSEPerTriplet: %s, averageAbsError: %s, maxAbsError: %s, numTriplets: %s\n",
            totalMSE, averageMSEPerTriplet, averageAbsError, maxAbsError, numTriplets)

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

# trainDatasetHelper = Dataset.TrainingDataset(Constants.NUM_TRAINING_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
# trainDataset = evalDataset.getTripletBatch(Constants.NUM_TRAINING_EXAMPLES * Constants.BSIZE)

# save(Constants.DATASET_SAVE_PATH,"trainingDataset", trainDatasetHelper)
@info("Number of triplet collisions encountered is: %s\n", trainDatasetHelper.numCollisions)
@info("Created %s unique triplet pairs\n", Constants.BSIZE * Constants.NUM_BATCHES)

evalDatasetHelper = Dataset.TrainingDataset(Constants.NUM_EVAL_EXAMPLES, Constants.MAX_STRING_LENGTH, Constants.MAX_STRING_LENGTH, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwiseHammingDistance)
evalDataset = evalDatasetHelper.getTripletBatch(Constants.NUM_EVAL_EXAMPLES * Constants.EVAL_BSIZE)
evalDatasetHelper.shuffleTripletBatch!(evalDataset)
evalDatasetBatches = evalDatasetHelper.extractBatches(evalDataset, Constants.NUM_BATCHES)

trainingLoop!(embeddingModel, trainDatasetHelper, evalDatasetBatches, opt, numEpochs=Constants.NUM_EPOCHS)
