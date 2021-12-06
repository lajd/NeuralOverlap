module Utils

    using Distances
    using Statistics
    using LinearAlgebra
    using Printf
    using Flux

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
            seqId = seqId + 1

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
        return B
    end

    function rowCosineSimilarity(a, b)
        denom = vec(Utils.norm2(a, dims=1)) .* vec(Utils.norm2(b, dims=1))
        return 1 .- (diag((transpose(a)* b)) ./ denom)
    end

    function Norm(x1, x2; dims=1, method="l2")
        return Utils.rowCosineSimilarity(x1, x2) |> DEVICE
        # return vec(Utils.l2Norm(x1 - x2, dims=dims))
    end


    anynan(x) = any(y -> any(isnan, y), x)
end
