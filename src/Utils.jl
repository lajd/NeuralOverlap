

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

        seqId = 1
        for refSeq in seqArr
            sequenceIDMap[refSeq] = seqId
            for (j, compSeq) in enumerate(seqArr)
                d = hamming(refSeq, compSeq)
                distanceMatrix[seqId, j] = d
            end
            seqId += 1
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

    function evaluateModel(datasetHelper, embeddingModel, maxStringLength; bsize=128, method="l2", numNN=100, estErrorN=1000, plotsSavePath=".", identifier="")
        idSeqDataMap = datasetHelper.getIdSeqDataMap()
        distanceMatrix = datasetHelper.getDistanceMatrix()
        numSeqs = length(idSeqDataMap)

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

        predictedDistanceMatrix = convert.(Float32, zeros(n, n))

        # Get pairwise distance

        for refIdx in 1:n
            e1 = E[1:end, refIdx]
            refE = hcat([e1 for _ in 1:n]...)
            compE = hcat([E[1:end, compIdx] for compIdx in 1:n]...)
            D = Utils.EmbeddingDistance(refE, compE, method, dims=1)
            predictedDistanceMatrix[refIdx, 1:end] = D
        end

        # Get k-nns and recall
        recallDict = Dict(
            "top1Recall" => DefaultDict(0.),
            "top10Recall" => DefaultDict(0.),
            "top50Recall" => DefaultDict(0.),
            "top100Recall" => DefaultDict(0.),
        )

        for id in 1:n
            # Don't include identical examples
            predicted_knns = sortperm(predictedDistanceMatrix[id, 2:end])[1:numNN]
            actual_knns = idSeqDataMap[id]["k100NN"][2:end]
            for k in range(1, 101, step=10)
                recallDict["top1Recall"][k] += Utils.recallTopN(predicted_knns, actual_knns, T=1, K=k)
                recallDict["top10Recall"][k] += Utils.recallTopN(predicted_knns, actual_knns, T=10, K=k)
                recallDict["top50Recall"][k] += Utils.recallTopN(predicted_knns, actual_knns, T=50, K=k)
                recallDict["top100Recall"][k] += Utils.recallTopN(predicted_knns, actual_knns, T=100, K=k)
            end
        end

        # Get average estimation error
        predDistanceArray = []
        trueDistanceArray = []
        # distanceEstimationErrorArray = []
        absErrorArray = []
        for _ in 1:estErrorN
            id1 = rand(1:numSeqs)
            id2 = rand(1:numSeqs)
            trueDist = distanceMatrix[id1, id2] * maxStringLength
            predDist = predictedDistanceMatrix[id1, id2] * maxStringLength
            push!(predDistanceArray, predDist)
            push!(trueDistanceArray, trueDist)
            # push!(distanceEstimationErrorArray, abs(predDist - trueDist)/trueDist * maxStringLength)

            absError = abs(predDist - trueDist)
            push!(absErrorArray, absError)
        end

        for (topNKey, recallAtK) in recallDict
            for (kValue, recallScoreSum) in recallAtK
                recallDict[topNKey][kValue] = recallScoreSum/n
                x = collect(keys(recallDict[topNKey]))
                perm = sortperm(x)
                x = x[perm]
                y = collect(values(recallDict[topNKey]))[perm]
                fig = plot(x, y, title=topNKey, xlabel="K-value", ylabel="Recall", label=["Recall"])
                savefig(fig, joinpath(plotsSavePath, string(topNKey, "_", identifier,  ".png")))
            end
        end

        # meanRecall = mean(recallArray)
        # meanEstimationError = mean(distanceEstimationErrorArray)
        meanAbsError = mean(absErrorArray)
        maxAbsError = maximum(absErrorArray)
        minAbsError = minimum(absErrorArray)
        totalAbsError = sum(absErrorArray)
        @printf("Mean Absolute Error is %s \n", round(meanAbsError, digits=4))
        @printf("Max Absolute Error is %s \n", round(maxAbsError, digits=4))
        @printf("Min abs error is %s \n", round(minAbsError, digits=4))
        @printf("Total abs error is %s \n", round(totalAbsError, digits=4))
        @printf("Number of triplets compared %s \n", estErrorN)
        return meanAbsError, maxAbsError, minAbsError, totalAbsError
    end
    anynan(x) = any(y -> any(isnan, y), x)
end
