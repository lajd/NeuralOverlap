

module Utils

    using Distances
    using Statistics
    using LinearAlgebra
    using Printf
    using Flux
    using Plots

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
                @error("Reads are %s", reads)
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

    function EmbeddingDistance(x1, x2; dims=1, method="cosine")
        if method == "l2"
            return vec(Utils.l2Norm(x1 - x2, dims=dims)) |> DEVICE
        elseif method == "cosine"
            return Utils.rowCosineSimilarity(x1, x2) |> DEVICE
        else
            throw("Method not recognized")
        end
    end

    function evaluateModel(evalBatches, model, maxStringLength; figSavePath="test.png", distanceMethod="l2")

        totalMSE = 0
        numTriplets = 0
    
        averageAbsErrorArray = []
        maxAbsErrorArray = []
        totalAbsError = 0 
        trueDistanceArray = []
        predictedDistanceArray = []

        model = model |> DEVICE
    
    
        for fullBatch in evalBatches
            # fullBatch = dataHelper.getTripletBatch(Constants.BSIZE)  
            # trainDataHelper.shuffleTripletBatch!(fullBatch)
            # fullBatch = trainDataHelper.extractBatchSubsets(fullBatch, 5)
    
            reads = fullBatch[1:3]
            batch = fullBatch[4:end]
        
            batch = batch |> DEVICE
        
            Xacr, Xpos, Xneg, y12, y13, y23 = batch
        
            Eacr = model(Xacr)
            Epos = model(Xpos)
            Eneg = model(Xneg)
        
            # MSE
            posEmbedDist = Utils.EmbeddingDistance(Eacr, Epos, dims=1, method=distanceMethod) |> DEVICE # 1D dist vector of size bsize
            negEmbedDist =  Utils.EmbeddingDistance(Eacr, Eneg, dims=1, method=distanceMethod) |> DEVICE
            PosNegEmbedDist =  Utils.EmbeddingDistance(Epos, Eneg, dims=1, method=distanceMethod) |> DEVICE
    
            # @assert maximum(posEmbedDist) <= 1
            @assert maximum(y12) <= 1
    
            d12 = abs.(posEmbedDist - y12) * maxStringLength
            d13 = abs.(negEmbedDist - y13) * maxStringLength
            d23 = abs.(PosNegEmbedDist - y23) * maxStringLength

            push!(trueDistanceArray, y12...)
            push!(trueDistanceArray, y13...)
            push!(trueDistanceArray, y23...)


            push!(predictedDistanceArray, posEmbedDist...)
            push!(predictedDistanceArray, negEmbedDist...)
            push!(predictedDistanceArray, PosNegEmbedDist...)

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
        @printf("Epoch MSE is %s \n", totalMSE)
        @printf("Average MSE per triplet is %s \n", round(averageMSEPerTriplet, digits=4))
        @printf("Average abs error is %s \n", round(averageAbsError, digits=4))
        @printf("Max abs error is %s \n", round(maxAbsError, digits=4))
        @printf("Total abs error is %s \n", totalAbsError, )
        @printf("Number of triplets compared %s \n", numTriplets)
    

        perm = sortperm(trueDistanceArray)

        fig = plot(scatter(trueDistanceArray[perm], predictedDistanceArray[perm],  label = ["True ED", "Predicted ED"]), title="True vs. Predicted edit distance")
        savefig(fig, figSavePath)

        return totalMSE, averageMSEPerTriplet, averageAbsError, maxAbsError, numTriplets
    end

    anynan(x) = any(y -> any(isnan, y), x)
end
