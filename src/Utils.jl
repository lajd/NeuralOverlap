

module Utils

    using Distances
    using Statistics
    using LinearAlgebra
    using Printf
    using Flux
    using Plots
    using DataStructures
    using Lathe

    try
        using CUDA
        # TODO: Don't require allow scalar
        global DEVICE = Flux.gpu
        global ArrayType = Union{Vector{Float32}, Matrix{Float32}, CuArray{Float32}}
    catch e
        global DEVICE = Flux.cpu
        global ArrayType = Union{Vector{Float32}, Matrix{Float32}}
    end
    

    function l2Norm(A; dims)
        A = sum(x -> x^2, A; dims=dims)
        A = sqrt.(A)
        return A
    end

    function pairwiseHammingDistance(seqArr::Array{String})
        sequenceIDMap = Dict([(refSeq, i) for (i, refSeq) in enumerate(seqArr)])
        distanceMatrix=pairwise(hamming, seqArr)
        return sequenceIDMap, distanceMatrix
    end

    function pairwiseDistance(X, method="l2")
        if method == "l2"
            return pairwise(euclidean, X)
        elseif method == "cosine"
            return pairwise(cosine_dist, X)
        else
            thow("Invalid method")
        end
    end

    function getTrainingScore(modelName::String; modelExtension=".jld2")::Float32
        suffix = split(modelName, "_")[end]
        trainingLoss = split(suffix, modelExtension)[1]
        return parse(Float32, trainingLoss)
    end
    
    
    function removeOldModels(modelSaveDir::String, keepN::Int64=3)
        # Remove any old models
        trainingScoreToModelNameDict = Dict()
        allModels = readdir(modelSaveDir)
    
        for modelName in allModels
            trainingLoss = getTrainingScore(modelName)
            trainingScoreToModelNameDict[trainingLoss] = modelName
        end
    
        i = 0
        for (trainingLoss, modelName) in sort(collect(trainingScoreToModelNameDict), by=x->x[1], rev=false)
            i = i + 1
            if i > keepN
                @printf("Deleting old model %s\n", modelName)
                rm(joinpath(modelSaveDir, modelName))
            end
        end
    end

    function getBestModelPath(modelSaveDir::String)::String
        trainingScoreToModelNameDict = Dict()
        allModels = readdir(modelSaveDir)
    
        for modelName in allModels
            trainingLoss = getTrainingScore(modelName)
            trainingScoreToModelNameDict[trainingLoss] = modelName
        end

        for (trainingLoss, modelName) in sort(collect(trainingScoreToModelNameDict), by=x->x[1], rev=false)
            return joinpath(modelSaveDir, modelName)
        end
    end

    function validateGradients(gs)
        maxGS = -1
        minGS = 1e6

        maxGSArr = []
        minGSArr = []
        avgGSArr = []
        for elements in gs
            max_ = maximum(elements)
            min_ = minimum(elements)
            avg_ = mean(elements)

            if max_ > maxGS
                maxGS = max_
            end

            if min_ < minGS
                minGS = min_
            end

            if any(isnan.(elements))
                @error("max %s, min %s", max_, min_)
                @error("Encountered NAN value in gs with max %s, min %s", maxGS, minGS)
                throw("Encuntered NaN")
            end
            push!(maxGSArr, max_)
            push!(minGSArr, min_)
            push!(avgGSArr, avg_)
       end

       return maximum(maxGSArr), minimum(minGSArr), mean(avgGSArr)
    end

    function norm2(A; dims)
        B = sum(x -> x^2, A; dims=dims)
        C = sqrt.(B)
        return C
    end

    function rowCosineSimilarity(a, b)
        denom = vec(Utils.norm2(a, dims=1)) .* vec(Utils.norm2(b, dims=1))
        return 1 .- diag((transpose(a)* b)) ./ denom
    end

    function EmbeddingDistance(x1, x2, method; dims=1)
        if method == "l2"
            return vec(Utils.l2Norm(x1 - x2, dims=dims)) |> DEVICE
        elseif method == "cosine"
            return Utils.rowCosineSimilarity(x1, x2) |> DEVICE
        else
            throw("Method not recognized")
        end
    end

    function recallTopTAtK(predKNN, actualKNN; T=100,K=100)
        """
        T is the number of relevant documents
        K is the numer of "true" documents
        """
        # If the items ranking top T contain k of the true top-k neighbors, the recall is k′/k.
        # Recall = (Relevant_Items_Recommended in top-k) / (Relevant_Items)
        
        T = min(T, length(predKNN))
        K = min(K, length(actualKNN))

        # Get the top T highest ranked (predicted) items
        predTNN = predKNN[1:T]

        # Get the top K ground truth items
        actualKNN = actualKNN[1: K]

        # Get the k' elements of T that are in the top K
        kPrime = sum([1 for i in predTNN if i in actualKNN])
        
        # Recall is k'/k
        recall = kPrime / T
        return recall
    end

    function getTopTRecallAtK(idSeqDataMap, predictedDistanceMatrix; plotsSavePath=".", identifier="", numNN=100, kStart=1, kEnd=100, kStep=10)
        n = length(idSeqDataMap)

        numNN = min(n, numNN)

        # Get k-nns and recall
        kValues = [k for k in range(kStart, kEnd + 1, step=kStep)]
        recallDict = Dict(
            "top1Recall" => Dict([(k, 0.) for k in kValues]),
            "top5Recall" => Dict([(k, 0.) for k in kValues]),
            "top10Recall" => Dict([(k, 0.) for k in kValues]),
            "top25Recall" => Dict([(k, 0.) for k in kValues]),
            "top50Recall" => Dict([(k, 0.) for k in kValues]),
            "top100Recall" => Dict([(k, 0.) for k in kValues])
        )

        # Get the total sums across all samples
        # Note that we start at index 2 for both predicted/actual KNNs,
        # so that the sequence is not compared with itself (skewing results, especially T=1)
        startIndex = 2
        for k in kValues
            for id in 1:n
                predicted_knns = sortperm(predictedDistanceMatrix[id, 1:end])[startIndex:numNN]
                actual_knns = idSeqDataMap[id]["k100NN"][startIndex:end]

                @assert length(predicted_knns) == length(actual_knns)
                recallDict["top1Recall"][k] += Utils.recallTopTAtK(predicted_knns, actual_knns, T=1, K=k)
                recallDict["top5Recall"][k] += Utils.recallTopTAtK(predicted_knns, actual_knns, T=5, K=k)
                recallDict["top10Recall"][k] += Utils.recallTopTAtK(predicted_knns, actual_knns, T=10, K=k)
                recallDict["top25Recall"][k] += Utils.recallTopTAtK(predicted_knns, actual_knns, T=25, K=k)
                recallDict["top50Recall"][k] += Utils.recallTopTAtK(predicted_knns, actual_knns, T=50, K=k)
                recallDict["top100Recall"][k] += Utils.recallTopTAtK(predicted_knns, actual_knns, T=100, K=k)
            end
        end

        # Normalize by the number of samples
        for (topNKey, sumRecallAtKDict) in recallDict
            for (kValue, sumValue) in sumRecallAtKDict
                recallDict[topNKey][kValue] = sumValue/length(idSeqDataMap)
            end
        end

        # Create a single recall-item plot containing topN curves
        averageRecallDict = Dict()
        for (topNRecallAtK, averageRecallAtKDict) in recallDict
            recallAtKArray = []
            # Sort the keys
            sortedK = sort(collect(keys(averageRecallAtKDict)))
            for k in sortedK
                averageRecallAtK = averageRecallAtKDict[k]
                push!(recallAtKArray, averageRecallAtK)
            end
            averageRecallDict[topNRecallAtK] = recallAtKArray
        end
        
        # Save a plot to a new directory
        saveDir = joinpath(plotsSavePath, "recall_at_k")
        mkpath(saveDir)
        fig = plot(kValues, [averageRecallDict["top1Recall"], averageRecallDict["top5Recall"],
            averageRecallDict["top10Recall"], averageRecallDict["top25Recall"], averageRecallDict["top50Recall"],
            averageRecallDict["top100Recall"]], label=["top1Recall" "top5Recall" "top10Recall" "top25Recall" "top50Recall" "top100Recall"],
            title=string("Average recall at K of ", length(idSeqDataMap), " samples"), 
            xlabel="Number of items",
            ylabel="Recall"
            )
        savefig(fig, joinpath(saveDir, string("epoch", "_", identifier,  ".png")))
        
        return recallDict
    end

    function embedSequenceData(datasetHelper, idSeqDataMap, embeddingModel; bsize=128)
        
        n = length(idSeqDataMap)  # Number of sequences
        
        # Xarray's index is the same as idSeqDataMap
        Xarray = []
        for k in 1:n
            v = idSeqDataMap[k]
            push!(Xarray, v["oneHotSeq"])
        end

        X = datasetHelper.formatOneHotSequenceArray(Xarray)

        Earray = []

        i = 1
        j = bsize
        while j < n
            Xm=X[1:end, 1:end, i:j] |> DEVICE
            push!(Earray, embeddingModel(Xm))
            i += bsize
            j += bsize
        end
        Xm = X[1:end, 1:end, i:n] |> DEVICE
        push!(Earray, embeddingModel(Xm))

        E = hcat(Earray...)
        @assert size(E)[2] == n
        return E
    end

    function _getEstimationErrorAndPlot(predDistanceArray, trueDistanceArray, plotsSavePath, identifier; calibrationModel=nothing, estimationErrorIdentifier="")
        # Fit a linear model mapping predicted distances to the true distances
        # TODO: This should be fit on the training set, not the validation set
        if isnothing(calibrationModel)
            calibrationModel = Lathe.models.LinearLeastSquare(
                predDistanceArray,
                trueDistanceArray,
            )
        end
        
        # Apply calibration model to the predicted distances
        calibratedPredictedDistanceArray = calibrationModel.predict(predDistanceArray)
        
        # Use the calibrated predictions for error estimation
        absErrorArray = [abs(i - j) for (i, j) in zip(calibratedPredictedDistanceArray, trueDistanceArray)]
        estimationErrorArray = [absError / trueDist for (absError, trueDist) in zip(absErrorArray, trueDistanceArray)]
        
        meanAbsError = mean(absErrorArray)
        maxAbsError = maximum(absErrorArray)
        minAbsError = minimum(absErrorArray)
        totalAbsError = sum(absErrorArray)
        meanEstimationError = mean(estimationErrorArray)
        
        # Plot the true/predicted estimation error
        saveDir = joinpath(plotsSavePath, "true_vs_pred_edit_distance")
        mkpath(saveDir)
        fig = plot(scatter(trueDistanceArray, [predDistanceArray, calibratedPredictedDistanceArray],
            label=["pred" "calibrated-pred"],
            title="Edit distanance predictions"),
            xlabel="true edit distance",
            ylabel="predicted edit distance",
            xlims=(0, Inf),
            ylims=(0, Inf)
        )

        # Add the line y=x to the figure for reference
        plot!(fig, trueDistanceArray, trueDistanceArray)
        
        savefig(fig, joinpath(saveDir, string("epoch", "_", identifier, estimationErrorIdentifier,  ".png")))
        
        @printf("---------------%s--------------\n", estimationErrorIdentifier)
        @printf("Mean Absolute Error is %s \n", round(meanAbsError, digits=4))
        @printf("Max Absolute Error is %s \n", round(maxAbsError, digits=4))
        @printf("Min abs error is %s \n", round(minAbsError, digits=4))
        @printf("Total abs error is %s \n", round(totalAbsError, digits=4))
        @printf("Mean relative estimation error is %s \n", round(meanEstimationError, digits=4))   
        @printf("-------------------------------\n")     
        return meanAbsError, maxAbsError, minAbsError, totalAbsError, meanEstimationError, calibrationModel

    end

    function _getRandomlySampledTruePredDistances(distanceMatrix, predictedDistanceMatrix, n, estErrorN, denormFactor)
        predDistanceArray = []
        trueDistanceArray = []
        # Randomly sample pairs and obtain true distance
        for _ in 1:estErrorN
            local trueDist = 0
            local id1 = id2 = nothing
            while isapprox(trueDist, 0.) == true
                id1 = rand(1:n)
                id2 = rand(1:n)
                trueDist = distanceMatrix[id1, id2] * denormFactor
            end

            predDist = predictedDistanceMatrix[id1, id2] * denormFactor

            push!(predDistanceArray, predDist)
            push!(trueDistanceArray, trueDist)
        end
        return trueDistanceArray, predDistanceArray
    end

    function _getNNSampledTruePredDistances(idSeqDataMap, distanceMatrix, predictedDistanceMatrix, n, estErrorN, denormFactor; startIndex=2)

        numNNsPerSample = 5

        predDistanceArray = []
        trueDistanceArray = []
        # Randomly sample pairs and obtain true distance
        for id1 in 1:n
            actual_knns = idSeqDataMap[id1]["k100NN"][startIndex:numNNsPerSample]
            for id2 in actual_knns
                trueDist = distanceMatrix[id1, id2] * denormFactor
                predDist = predictedDistanceMatrix[id1, id2] * denormFactor
                push!(predDistanceArray, predDist)
                push!(trueDistanceArray, trueDist)
            end
        end
        
        trueDistanceArray = trueDistanceArray[1: estErrorN]
        predDistanceArray = predDistanceArray[1: estErrorN]

        return trueDistanceArray, predDistanceArray
    end

    function getEstimationError(idSeqDataMap, distanceMatrix, predictedDistanceMatrix, maxStringLength; estErrorN=1000, plotsSavePath=".", identifier="", distanceMatrixNormMethod="max", calibrationModel=nothing)
        n = size(distanceMatrix)[1]
        
        if distanceMatrixNormMethod == "max"
            denormFactor = maxStringLength
        elseif distanceMatrixNormMethod == "mean"
            denormFactor = mean(distanceMatrix)
        else
            throw("Invalidate distanceMatrixNormMethod")
        end
        
        # Randomly sample pairs and obtain true distance
        trueDistanceArray, predDistanceArray = _getRandomlySampledTruePredDistances(distanceMatrix, predictedDistanceMatrix, n, estErrorN, denormFactor)
        
        meanAbsError, maxAbsError, minAbsError, totalAbsError, meanEstimationError, calibrationModel = _getEstimationErrorAndPlot(
            predDistanceArray, trueDistanceArray, plotsSavePath, identifier, calibrationModel=nothing, estimationErrorIdentifier="_random_samples"
        )

        # Now consider only N samples pairs that are nearest, and recalibrate. Since we are predominantly interested
        # in accurately predicting nearest neighbours, this is a useful data point. Here we sample estErrorN from the N nearest
        # neighbours for each point.
        trueDistanceArrayNNSampled, predDistanceArrayNNSampled = _getNNSampledTruePredDistances(
            idSeqDataMap, distanceMatrix, predictedDistanceMatrix, n, estErrorN, denormFactor
        )

        meanAbsError, maxAbsError, minAbsError, totalAbsError, meanEstimationError, calibrationModel = _getEstimationErrorAndPlot(
            predDistanceArrayNNSampled, trueDistanceArrayNNSampled, plotsSavePath, identifier, calibrationModel=nothing, estimationErrorIdentifier="_sampled_nns"
        )
        
        return meanAbsError, maxAbsError, minAbsError, totalAbsError, meanEstimationError, calibrationModel
    end


    function evaluateModel(datasetHelper, embeddingModel, maxStringLength; bsize=512,
         method="l2", numNN=100, estErrorN=1000, plotsSavePath=".",
         identifier="", distanceMatrixNormMethod="max", kStart=kStart,
         kEnd=kEnd, kStep=kStep
        )

        idSeqDataMap = datasetHelper.getIdSeqDataMap()
        n = length(idSeqDataMap)

        distanceMatrix = datasetHelper.getDistanceMatrix()
        numSeqs = length(idSeqDataMap)

        timeEmbedSequences = @elapsed begin
            Etensor = embedSequenceData(
                datasetHelper, idSeqDataMap, embeddingModel, bsize=bsize
            )
        end

        # Convert back to CPU
        Etensor = Etensor |> cpu

        @assert size(Etensor)[2] == n


        timeGetPredictedDistanceMatrix = @elapsed begin
            # Obtain inferred/predicted distance matrix
            predictedDistanceMatrix=pairwiseDistance(Etensor, method)
        end


        @assert size(predictedDistanceMatrix)[1] == n
        @assert size(predictedDistanceMatrix)[2] == n

        timeGetRecallAtK = @elapsed begin
            # Obtain the recall dictionary for each T value, for each K value
            recallDict = getTopTRecallAtK(
                idSeqDataMap, predictedDistanceMatrix,
                plotsSavePath=plotsSavePath, identifier=identifier, numNN=numNN,
                kStart=kStart, kEnd=kEnd, kStep=kStep
            )
        end

        timeGetEstimationError = @elapsed begin
            # Obtain estimation error
            meanAbsError, maxAbsError, minAbsError,
            totalAbsError, meanEstimationError, calibrationModel = getEstimationError(
                idSeqDataMap, distanceMatrix, predictedDistanceMatrix, maxStringLength,
                estErrorN=estErrorN, plotsSavePath=plotsSavePath, identifier=identifier,
                distanceMatrixNormMethod=distanceMatrixNormMethod
            )
        end

        # Log results

        @printf("Time to embed sequences: %s \n", timeEmbedSequences)
        @printf("Time to compute pred distance matrix: %s \n", timeGetPredictedDistanceMatrix)
        @printf("Time to get recall at K: %s \n", timeGetRecallAtK)
        @printf("Time to get error estimation: %s \n", timeGetEstimationError)
        return meanAbsError, maxAbsError, minAbsError, totalAbsError, meanEstimationError, recallDict, calibrationModel
    end

    
    
    anynan(x) = any(y -> any(isnan, y), x)
end
