
module MiscUtils

    using Printf
    using ...ExperimentHelper

    function get_training_score(modelName::String; modelExtension=".jld2")::Float32
        suffix = split(modelName, "_")[end]
        trainingLoss = split(suffix, modelExtension)[1]
        return parse(Float32, trainingLoss)
    end


    function remove_old_models(modelSaveDir::String, keepN::Int64=3)
        # Remove any old models
        trainingScoreToModelNameDict = Dict()
        allModels = readdir(modelSaveDir)

        for modelName in allModels
            trainingLoss = get_training_score(modelName)
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

    function get_best_model_path(modelSaveDir::String)::String
        trainingScoreToModelNameDict = Dict()
        allModels = readdir(modelSaveDir)

        for modelName in allModels
            trainingLoss = get_training_score(modelName)
            trainingScoreToModelNameDict[trainingLoss] = modelName
        end

        for (trainingLoss, modelName) in sort(collect(trainingScoreToModelNameDict), by=x->x[1], rev=false)
            return joinpath(modelSaveDir, modelName)
        end
    end

    function validate_gradients(gs)
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


    anynan(x) = any(y -> any(isnan, y), x)

end
