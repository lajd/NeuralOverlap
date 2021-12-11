

module Utils

    using Distances
    using Statistics
    using LinearAlgebra
    using Printf
    using Flux
    using Plots
    using DataStructures

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
        n = length(seqArr)
        distanceMatrix = zeros(n, n)
        sequenceIDMap = Dict()

        for (i, refSeq) in enumerate(seqArr)
            sequenceIDMap[refSeq] = i
            for (j, compSeq) in enumerate(seqArr)
                d = hamming(refSeq, compSeq)
                distanceMatrix[i, j] = d
            end
        end
        return sequenceIDMap, distanceMatrix
    end

    function getTrainingScore(modelName::String, modelSaveSuffix::String)::Float32
        suffix = split(modelName, "_")[end - 1]
        trainingLoss = split(suffix, modelSaveSuffix)[1]
        return parse(Float32, trainingLoss)
    end
    
    
    function removeOldModels(modelSaveDir::String, modelSaveSuffix::String, keepN::Int64=3)
        # Remove any old models
        trainingScoreToModelNameDict = Dict()
        allModels = readdir(modelSaveDir)
    
        for modelName in allModels
            trainingLoss = getTrainingScore(modelName, modelSaveSuffix::String)
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

    function getBestModelPath(modelSaveDir::String, modelSaveSuffix::String)::String
        trainingScoreToModelNameDict = Dict()
        allModels = readdir(modelSaveDir)
    
        for modelName in allModels
            trainingLoss = getTrainingScore(modelName, modelSaveSuffix)
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
                break
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

    function recallTopN(predKNN, actualKNN; T=100,K=100)
        """
        T is the number of relevant documents
        K is the numer of "true" documents
        """
        # If the items ranking  top T contain k of the true top-k neighbors, the recall is k′/k.
        # Recall= (Relevant_Items_Recommended in top-k) / (Relevant_Items)
        predTNN = predKNN[1:min(T, length(predKNN))]
        actualKNN = actualKNN[1: min(K, length(actualKNN))]
        intersection = sum([1 for i in predTNN if i in actualKNN])
        recall = intersection / T
        return recall
    end

    function getTopTRecallAtK(idSeqDataMap, distanceMatrix, predictedDistanceMatrix; plotsSavePath=".", identifier="", numNN=100, kStep=10)
        # Get k-nns and recall
        kValues = [k for k in range(1, numNN + 1, step=kStep)]
        recallDict = Dict(
            "top1Recall" => Dict([(k, 0.) for k in kValues]),
            "top10Recall" => Dict([(k, 0.) for k in kValues]),
            "top50Recall" => Dict([(k, 0.) for k in kValues]),
            "top100Recall" => Dict([(k, 0.) for k in kValues])
        )

        # Get the total sums across all samples
        # Note that we start at index 2 for both predicted/actual KNNs,
        # so that the sequence is not compared with itself (skewing results, especially T=1)
        startIndex = 2
        for k in kValues
            for id in 1:length(idSeqDataMap)
                predicted_knns = sortperm(predictedDistanceMatrix[id, 1:end])[startIndex:numNN]
                actual_knns = idSeqDataMap[id]["k100NN"][startIndex:end]
                recallDict["top1Recall"][k] += Utils.recallTopN(predicted_knns, actual_knns, T=1, K=k)
                recallDict["top10Recall"][k] += Utils.recallTopN(predicted_knns, actual_knns, T=10, K=k)
                recallDict["top50Recall"][k] += Utils.recallTopN(predicted_knns, actual_knns, T=50, K=k)
                recallDict["top100Recall"][k] += Utils.recallTopN(predicted_knns, actual_knns, T=100, K=k)
            end
        end

        # Normalize by the number of samples
        for (topNKey, sumRecallAtKDict) in recallDict
            for (kValue, sumValue) in sumRecallAtKDict
                recallDict[topNKey][kValue] = sumValue/length(idSeqDataMap)
            end
        end

        # Create recall-item plots
        for (topNRecallAtK, averageRecallAtKDict) in recallDict
            kArray = []
            recallAtKArray = []
            # Sort the keys
            sortedK = sort(collect(keys(averageRecallAtKDict)))
            for k in sortedK
                averageRecallAtK = averageRecallAtKDict[k]
                push!(recallAtKArray, averageRecallAtK)
                push!(kArray, k)
            end
            # Save a plot
            fig = plot(kArray, recallAtKArray, title=topNRecallAtK, xlabel="K-value", ylabel="Recall", label=["Recall"])
            savefig(fig, joinpath(plotsSavePath, string(topNRecallAtK, "_", identifier,  ".png")))
        end
        return recallDict
    end

    function getPredictedDistanceMatrix(datasetHelper, idSeqDataMap, embeddingModel; bsize=256, method="l2")
        Xarray = []
        for k in 1:length(idSeqDataMap)
            v = idSeqDataMap[k]
            push!(Xarray, v["oneHotSeq"])
        end

        X = datasetHelper.formatOneHotSequenceArray(Xarray) |> DEVICE

        Earray = []

        i = 1
        j = bsize
        n = size(X)[3]
        while j < n
            push!(Earray, embeddingModel(X[1:end, 1:end, i:j]))
            i += bsize
            j += bsize
        end
        push!(Earray, embeddingModel(X[1:end, 1:end, i:n]))

        E = hcat(Earray...)
        @assert size(E)[2] == n

        predictedDistanceMatrix = convert.(Float32, zeros(n, n))

        # Get pairwise distance
        for refIdx in 1:n
            e1 = E[1:end, refIdx]
            refE = hcat([e1 for _ in 1:n]...)
            compE = hcat([E[1:end, compIdx] for compIdx in 1:n]...)
            D = Utils.EmbeddingDistance(refE, compE, method, dims=1)
            predictedDistanceMatrix[refIdx, 1:end] = D
        end
        return predictedDistanceMatrix
    end
    
    function getEstimationError(distanceMatrix, predictedDistanceMatrix, maxStringLength; estErrorN=1000)
        n = size(distanceMatrix)[1]
        # Get average estimation error
        predDistanceArray = []
        trueDistanceArray = []
        absErrorArray = []
        for _ in 1:estErrorN
            id1 = rand(1:n)
            id2 = rand(1:n)
            trueDist = distanceMatrix[id1, id2] * maxStringLength
            predDist = predictedDistanceMatrix[id1, id2] * maxStringLength

            push!(predDistanceArray, predDist)
            push!(trueDistanceArray, trueDist)

            absError = abs(predDist - trueDist)
            push!(absErrorArray, absError)
        end
        meanAbsError = mean(absErrorArray)
        maxAbsError = maximum(absErrorArray)
        minAbsError = minimum(absErrorArray)
        totalAbsError = sum(absErrorArray)
        return meanAbsError, maxAbsError, minAbsError, totalAbsError
    end


    function evaluateModel(datasetHelper, embeddingModel, maxStringLength; bsize=512, method="l2", numNN=100, estErrorN=1000, plotsSavePath=".", identifier="")
        idSeqDataMap = datasetHelper.getIdSeqDataMap()
        distanceMatrix = datasetHelper.getDistanceMatrix()

        # Truncate
        for (k, _) in idSeqDataMap
            if k > 100
                delete!(idSeqDataMap, k)
            end
        end

        numSeqs = length(idSeqDataMap)
        distanceMatrix = distanceMatrix[1:100, 1:100]

        identifier = "test"
        plotsSavePath="."





        # Obtain inferred/predicted distance matrix
        predictedDistanceMatrix = getPredictedDistanceMatrix(
            datasetHelper, idSeqDataMap, embeddingModel, bsize=bsize, method=method
        )

        # Obtain the recall dictionary for each T value, for each K value
        recallDict = getTopTRecallAtK(
            idSeqDataMap, distanceMatrix, predictedDistanceMatrix,
            plotsSavePath=plotsSavePath, identifier=identifier, numNN=numNN,
        )

        # Obtain estimation error
        meanAbsError, maxAbsError, minAbsError, totalAbsError = getEstimationError(
            distanceMatrix, predictedDistanceMatrix,
            maxStringLength, estErrorN=estErrorN,
        )

        # Log results
        @printf("Mean Absolute Error is %s \n", round(meanAbsError, digits=4))
        @printf("Max Absolute Error is %s \n", round(maxAbsError, digits=4))
        @printf("Min abs error is %s \n", round(minAbsError, digits=4))
        @printf("Total abs error is %s \n", round(totalAbsError, digits=4))
        @printf("Number of triplets compared %s \n", estErrorN)
        return meanAbsError, maxAbsError, minAbsError, totalAbsError, recallDict
    end
    anynan(x) = any(y -> any(isnan, y), x)
end
