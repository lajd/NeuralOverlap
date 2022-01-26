include("./Embedding.jl")
include("./Distance.jl")

module EvaluationUtils

    using Distances
    using Statistics
    using LinearAlgebra
    using Printf
    using Plots
    using Lathe
    using Flux

    using ..EmbeddingUtils: embed_sequence_data
    using ..DistanceUtils: pairwise_distance


    function _get_estimation_error_and_plot(pred_distance_array::Vector{Any}, true_distance_array::Vector{Any}, plot_save_path::String, identifier::String; epoch_calibration_model=nothing, estimationErrorIdentifier::String="")
        # Fit a linear model mapping predicted distances to the true distances
        # TODO: This should be fit on the training set, not the validation set
        if isnothing(epoch_calibration_model)
            epoch_calibration_model = Lathe.models.SimpleLinearRegression(
                pred_distance_array,
                true_distance_array,
            )
        end

        # Apply calibration model to the predicted distances
        calibratedPredictedDistanceArray = epoch_calibration_model.predict(pred_distance_array)

        # Use the calibrated predictions for error estimation
        absErrorArray = [abs(i - j) for (i, j) in zip(calibratedPredictedDistanceArray, true_distance_array)]
        estimationErrorArray = [absError / true_dist for (absError, true_dist) in zip(absErrorArray, true_distance_array)]

        epoch_mean_abs_error = mean(absErrorArray)
        epoch_max_abs_error = maximum(absErrorArray)
        epoch_min_abs_error = minimum(absErrorArray)
        epoch_total_abs_error = sum(absErrorArray)
        mean_estimation_error = mean(estimationErrorArray)

        # Plot the true/predicted estimation error
        saveDir = joinpath(plot_save_path, "true_vs_pred_edit_distance")
        mkpath(saveDir)
        fig = plot(scatter(true_distance_array, [pred_distance_array, calibratedPredictedDistanceArray],
            label=["pred" "calibrated-pred"],
            title="Edit distance predictions"),
            xlabel="true edit distance",
            ylabel="predicted edit distance",
            xlims=(0, Inf),
            ylims=(0, Inf)
        )

        # Add the line y=x to the figure for reference
        plot!(fig, true_distance_array, true_distance_array)

        savefig(fig, joinpath(saveDir, string("epoch", "_", identifier, estimationErrorIdentifier,  ".png")))

        @printf("---------------%s--------------\n", estimationErrorIdentifier)
        @printf("Mean Absolute Error is %s \n", round(epoch_mean_abs_error, digits=4))
        @printf("Max Absolute Error is %s \n", round(epoch_max_abs_error, digits=4))
        @printf("Min abs error is %s \n", round(epoch_min_abs_error, digits=4))
        @printf("Total abs error is %s \n", round(epoch_total_abs_error, digits=4))
        @printf("Mean relative estimation error is %s \n", round(mean_estimation_error, digits=4))
        @printf("-------------------------------\n")
        return epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error, epoch_total_abs_error, mean_estimation_error, epoch_calibration_model

    end

    function _get_randomly_sampled_true_pred_distances(true_distance_matrix::Matrix, predicted_distance_matrix::Matrix, n::Int64, est_error_n::Int64, denorm_factor::Float64)
        pred_distance_array = []
        true_distance_array = []
        # Randomly sample pairs and obtain true distance
        for _ in 1:est_error_n
            local true_dist = 0
            local id1 = id2 = nothing
            while isapprox(true_dist, 0.) == true
                id1 = rand(1:n)
                id2 = rand(1:n)
                true_dist = true_distance_matrix[id1, id2] * denorm_factor
            end

            pred_dist = predicted_distance_matrix[id1, id2] * denorm_factor

            push!(pred_distance_array, pred_dist)
            push!(true_distance_array, true_dist)
        end
        return true_distance_array, pred_distance_array
    end

    function _get_nn_sampled_true_pred_distances(id_seq_data_map::Dict, true_distance_matrix::Matrix, predicted_distance_matrix::Matrix, n::Int64, est_error_n::Int64, denorm_factor::Float64; startIndex::Int64=2)

        numNNsPerSample = 5

        pred_distance_array = []
        true_distance_array = []
        # Randomly sample pairs and obtain true distance
        for id1 in 1:n
            actual_knns = id_seq_data_map[id1]["topKNN"][startIndex:numNNsPerSample]
            for id2 in actual_knns
                true_dist = true_distance_matrix[id1, id2] * denorm_factor
                pred_dist = predicted_distance_matrix[id1, id2] * denorm_factor
                push!(pred_distance_array, pred_dist)
                push!(true_distance_array, true_dist)
            end
        end

        true_distance_array = true_distance_array[1: est_error_n]
        pred_distance_array = pred_distance_array[1: est_error_n]

        return true_distance_array, pred_distance_array
    end

    function get_estimation_error(id_seq_data_map::Dict, true_distance_matrix::Matrix, predicted_distance_matrix::Matrix, denorm_factor::Float64; est_error_n::Int64=1000, plot_save_path::String=".", identifier::String="", epoch_calibration_model=nothing)
        n = size(true_distance_matrix)[1]

        est_error_n = min(length(id_seq_data_map), est_error_n)

        # Randomly sample pairs and obtain true distance
        true_distance_array, pred_distance_array = _get_randomly_sampled_true_pred_distances(true_distance_matrix, predicted_distance_matrix, n, est_error_n, denorm_factor)

        epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error, epoch_total_abs_error, mean_estimation_error, epoch_calibration_model = _get_estimation_error_and_plot(
            pred_distance_array, true_distance_array, plot_save_path, identifier, epoch_calibration_model=nothing, estimationErrorIdentifier="_random_samples"
        )

        # Now consider only N samples pairs that are nearest, and recalibrate. Since we are predominantly interested
        # in accurately predicting nearest neighbours, this is a useful data point. Here we sample est_error_n from the N nearest
        # neighbours for each point.
        true_distance_arrayNNSampled, pred_distance_arrayNNSampled = _get_nn_sampled_true_pred_distances(
            id_seq_data_map, true_distance_matrix, predicted_distance_matrix, n, est_error_n, denorm_factor
        )

        epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error, epoch_total_abs_error, mean_estimation_error, epoch_calibration_model = _get_estimation_error_and_plot(
            pred_distance_arrayNNSampled, true_distance_arrayNNSampled, plot_save_path, identifier, epoch_calibration_model=nothing, estimationErrorIdentifier="_sampled_nns"
        )

        return epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error, epoch_total_abs_error, mean_estimation_error, epoch_calibration_model
    end


    function evaluate_model(dataset_helper, embedding_model::Flux.Chain, denorm_factor::Float64,  bSize::Int64;
         method="l2", numNN=100, est_error_n=1000, plot_save_path=".",
         identifier="", kStart=kStart,
         kEnd=kEnd, kStep=kStep
        )

        id_seq_data_map = dataset_helper.get_id_seq_data_map()
        n = length(id_seq_data_map)

        true_distance_matrix = dataset_helper.get_distance_matrix()
        numSeqs = length(id_seq_data_map)

        timeEmbedSequences = @elapsed begin
            Etensor = embed_sequence_data(
                dataset_helper, id_seq_data_map, embedding_model, bSize
            )
        end

        # Convert back to CPU
        Etensor = Etensor |> cpu

        @assert size(Etensor)[2] == n


        timeGetPredictedtrue_distance_matrix = @elapsed begin
            # Obtain inferred/predicted distance matrix
            predicted_distance_matrix=pairwise_distance(Etensor, method)
        end


        @assert size(predicted_distance_matrix)[1] == n
        @assert size(predicted_distance_matrix)[2] == n

        timeGetRecallAtK = @elapsed begin
            # Obtain the recall dictionary for each T value, for each K value
            epcoh_recall_dict = get_top_t_recall_at_k(plot_save_path, identifier, n, numNN=numNN,
                kStart=kStart, kEnd=kEnd, kStep=kStep,trueid_seq_data_map=id_seq_data_map, predicted_distance_matrix=predicted_distance_matrix,
            )
        end

        timeGetEstimationError = @elapsed begin
            # Obtain estimation error
            epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error,
            epoch_total_abs_error, mean_estimation_error, epoch_calibration_model = get_estimation_error(
                id_seq_data_map, true_distance_matrix, predicted_distance_matrix,
                denorm_factor, est_error_n=est_error_n, plot_save_path=plot_save_path,
                identifier=identifier
            )
        end

        # Log results
        @printf("Time to embed sequences: %s \n", timeEmbedSequences)
        @printf("Time to compute pred distance matrix: %s \n", timeGetPredictedtrue_distance_matrix)
        @printf("Time to get recall at K: %s \n", timeGetRecallAtK)
        @printf("Time to get error estimation: %s \n", timeGetEstimationError)
        return epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error, epoch_total_abs_error, mean_estimation_error, epcoh_recall_dict, epoch_calibration_model
    end


        function top_k_recall(predKNN::Array, actualKNN::Array; T::Int64=100, K::Int64=100)
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

        function get_top_t_recall_at_k(plot_save_path::String, identifier::String, n_sequences::Int64; numNN::Int64=1000,
            kStart::Int64=1, kEnd::Int64=1001, kStep::Int64=100, startIndex::Int64=2,
            truetrue_distance_matrix=nothing, trueid_seq_data_map=nothing, predicted_distance_matrix=nothing,
            predicted_id_seq_data_map=nothing, nSample=1000
        )
            # Get k-nns and recall
            kValues = [k for k in range(kStart, kEnd + 1, step=kStep)]
            epcoh_recall_dict = Dict(
                "top1Recall" => Dict([(k, 0.) for k in kValues]),
                "top5Recall" => Dict([(k, 0.) for k in kValues]),
                "top10Recall" => Dict([(k, 0.) for k in kValues]),
                "top25Recall" => Dict([(k, 0.) for k in kValues]),
                "top50Recall" => Dict([(k, 0.) for k in kValues]),
                "top100Recall" => Dict([(k, 0.) for k in kValues])
            )

            # TODO Randomly sample the IDs?
            startIndex = 2
            nSample = min(nSample, n_sequences)
            for k in kValues
                for id in 1:nSample
                    actual_knns = nothing
                    predicted_knns = nothing

                    if predicted_id_seq_data_map != nothing
                        predicted_knns = predicted_id_seq_data_map[id]["topKNN"][startIndex:numNN]
                    elseif predicted_distance_matrix != nothing
                        predicted_knns = sortperm(predicted_distance_matrix[id, 1:end])[startIndex:numNN]
                    else
                        throw("Invalid")
                    end

                    if trueid_seq_data_map != nothing
                        actual_knns = trueid_seq_data_map[id]["topKNN"][startIndex:numNN]
                    elseif truetrue_distance_matrix != nothing
                        actual_knns = sortperm(truetrue_distance_matrix[id, 1:end])[startIndex:numNN]
                    else
                        throw("Invalid")
                    end

                    @assert length(predicted_knns) == length(actual_knns) (length(predicted_knns), length(actual_knns))
                    epcoh_recall_dict["top1Recall"][k] += top_k_recall(predicted_knns, actual_knns, T=1, K=k)
                    epcoh_recall_dict["top5Recall"][k] += top_k_recall(predicted_knns, actual_knns, T=5, K=k)
                    epcoh_recall_dict["top10Recall"][k] += top_k_recall(predicted_knns, actual_knns, T=10, K=k)
                    epcoh_recall_dict["top25Recall"][k] += top_k_recall(predicted_knns, actual_knns, T=25, K=k)
                    epcoh_recall_dict["top50Recall"][k] += top_k_recall(predicted_knns, actual_knns, T=50, K=k)
                    epcoh_recall_dict["top100Recall"][k] += top_k_recall(predicted_knns, actual_knns, T=100, K=k)

                end
            end

            # Normalize by the number of samples used to compute the recall sum
            for (topNKey, sumRecallAtKDict) in epcoh_recall_dict
                for (kValue, sumValue) in sumRecallAtKDict
                    epcoh_recall_dict[topNKey][kValue] = sumValue/nSample
                end
            end

            # Create a single recall-item plot containing topN curves
            averageepcoh_recall_dict = Dict()
            for (topNRecallAtK, averageRecallAtKDict) in epcoh_recall_dict
                recallAtKArray = []
                # Sort the keys
                sortedK = sort(collect(keys(averageRecallAtKDict)))
                for k in sortedK
                    averageRecallAtK = averageRecallAtKDict[k]
                    push!(recallAtKArray, averageRecallAtK)
                end
                averageepcoh_recall_dict[topNRecallAtK] = recallAtKArray
            end

            # Save a plot to a new directory
            saveDir = joinpath(plot_save_path, "recall_at_k")
            mkpath(saveDir)
            fig = plot(kValues, [averageepcoh_recall_dict["top1Recall"], averageepcoh_recall_dict["top5Recall"],
                averageepcoh_recall_dict["top10Recall"], averageepcoh_recall_dict["top25Recall"], averageepcoh_recall_dict["top50Recall"],
                averageepcoh_recall_dict["top100Recall"]], label=["top1Recall" "top5Recall" "top10Recall" "top25Recall" "top50Recall" "top100Recall"],
                title=string("Average recall at for samples", nSample, " samples"),
                xlabel="Number of items",
                ylabel="Recall"
                )
            savefig(fig, joinpath(saveDir, string("epoch", "_", identifier,  ".png")))

            return epcoh_recall_dict
        end

end
