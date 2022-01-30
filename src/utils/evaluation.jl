include("misc.jl")

using Distances
using Statistics
using LinearAlgebra
using Printf
using Plots
using Lathe
using Flux
using ProgressBars

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
    save_dir = joinpath(plot_save_path, "true_vs_pred_edit_distance")
    mkpath(save_dir)
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

    savefig(fig, joinpath(save_dir, string("epoch", "_", identifier, estimationErrorIdentifier,  ".png")))

    @info("---------------$(estimationErrorIdentifier)--------------")
    @info("Mean Abs Error: $(toscientific(epoch_mean_abs_error))")
    @info("Max Abs Error: $(toscientific(epoch_max_abs_error))")
    @info("Min Abs Error: $(toscientific(epoch_min_abs_error))")
    @info("Total Abs Error: $(toscientific(epoch_total_abs_error))")
    @info("Mean Relative Estimation Error: $(toscientific(mean_estimation_error))")
    return epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error, epoch_total_abs_error, mean_estimation_error, epoch_calibration_model

end

function _get_randomly_sampled_true_pred_distances(true_distance_matrix::Matrix, predicted_distance_matrix::Matrix, n::Int64, est_error_n::Int64, denorm_factor::Float64)
    pred_distance_array = []
    true_distance_array = []
    # Randomly sample pairs and obtain true distance
    for _ in ProgressBar(1:est_error_n)
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

function _get_nn_sampled_true_pred_distances(id_seq_data_map::Dict, true_distance_matrix::Matrix, predicted_distance_matrix::Matrix, n::Int64, est_error_n::Int64, denorm_factor::Float64; nn_start_index::Int64=2)
    @info("Getting sampled true/pred distances")
    numNNsPerSample = 5

    pred_distance_array = []
    true_distance_array = []
    # Randomly sample pairs and obtain true distance
    for id1 in ProgressBar(1:n)
        actual_knns = id_seq_data_map[id1]["topKNN"][nn_start_index:numNNsPerSample]
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
     method::String="l2", numNN::Int64=100, est_error_n::Int64=1000, plot_save_path::String=".",
     identifier="", kStart=kStart,
     kEnd=kEnd, kStep=kStep
    )

    id_seq_data_map = dataset_helper.get_id_seq_data_map()
    n = length(id_seq_data_map)

    true_distance_matrix = dataset_helper.get_distance_matrix()
    numSeqs = length(id_seq_data_map)

    embed_sequence_time = @elapsed begin
        Etensor = embed_sequence_data(
            dataset_helper, id_seq_data_map, embedding_model, bSize
        )
    end

    # Convert back to CPU
    Etensor = Etensor |> cpu

    @assert size(Etensor)[2] == n


    time_get_pred_dist_mat = @elapsed begin
        # Obtain inferred/predicted distance matrix
        predicted_distance_matrix=pairwise_distance(Etensor, method)
    end


    @assert size(predicted_distance_matrix)[1] == n
    @assert size(predicted_distance_matrix)[2] == n

    time_get_recall_at_k = @elapsed begin
        # Obtain the recall dictionary for each T value, for each K value
        epoch_recall_dict = get_top_t_recall_at_k(plot_save_path, identifier, n, numNN=numNN,
            kStart=kStart, kEnd=kEnd, kStep=kStep,trueid_seq_data_map=id_seq_data_map,
            predicted_distance_matrix=predicted_distance_matrix,
        )
    end

    time_get_estimation_error = @elapsed begin
        # Obtain estimation error
        epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error,
        epoch_total_abs_error, mean_estimation_error, epoch_calibration_model = get_estimation_error(
            id_seq_data_map, true_distance_matrix, predicted_distance_matrix,
            denorm_factor, est_error_n=est_error_n, plot_save_path=plot_save_path,
            identifier=identifier
        )
    end

    # Log results
    @info("Time to embed sequences: $(round(embed_sequence_time))s")
    @info("Time to compute pred distance matrix: $(round(time_get_pred_dist_mat))s")
    @info("Time to get recall at K: $(round(time_get_recall_at_k))s")
    @info("Time to get error estimation: $(round(time_get_estimation_error))s")
    return epoch_mean_abs_error, epoch_max_abs_error, epoch_min_abs_error, epoch_total_abs_error, mean_estimation_error, epoch_recall_dict, epoch_calibration_model
end

function recallatk(kvalues::Array{Int64})

    tvalues = [1, 5, 10, 25, 50, 100]

    recall_at_k_dict = Dict()
    for t in tvalues
        recall_at_k_dict[t] = Dict([(k, 0.) for k in kvalues])
    end


    () -> (tvalues;recall_at_k_dict;)

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

    # # Get the k' elements of T that are in the top K
    # kPrime = sum([1 for i in predTNN if i in actualKNN])

    kPrime = length(intersect(predTNN, actualKNN))
    # Recall is k'/k
    recall = kPrime / T
    return recall
end

function _get_nns(
        id::Int64,
        id_seq_data_map_::Union{Nothing, Dict},
        distance_matrix_::Union{Nothing, Matrix},
        nn_start_index::Int64,
        numNN::Int64
    )
    if id_seq_data_map_ != nothing
        nns = id_seq_data_map_[id]["topKNN"][nn_start_index:numNN]
    elseif distance_matrix_ != nothing
        nns = sortperm(distance_matrix_[id, 1:end])[nn_start_index:numNN]
    else
        throw("Invalid")
    end
    return nns
end


function get_top_t_recall_at_k(plot_save_path::String, identifier::String,
    n_sequences::Int64; numNN::Int64=1000,
    kStart::Int64=1, kEnd::Int64=1001, kStep::Int64=100, nn_start_index::Int64=2,
    true_distance_matrix=nothing, trueid_seq_data_map=nothing, predicted_distance_matrix=nothing,
    predicted_id_seq_data_map=nothing, num_samples::Int64=1000, index_start::Int64=1
)
    # TODO: This is very slow
    @info("Getting top T recall at K")

    # Get k-nns and recall
    kvalues = [k for k in range(kStart, kEnd + 1, step=kStep)]
    epoch_recall_at_k = recallatk(kvalues)

    # TODO Randomly sample the IDs?
    nn_start_index = 2
    num_samples = Int64(min(num_samples, n_sequences))
    @info("Obtaining recall for $(length(kvalues)) k values of range $(kStart):$(kEnd):$(kStep)")

    # epoch recall at k
    epoch_recall_dict = epoch_recall_at_k.recall_at_k_dict

    # Iterate over each sampled sequences
    for id in ProgressBar(1:num_samples)
        # Get true/pred NNs for this sequence
        predicted_knns = _get_nns(id, predicted_id_seq_data_map, predicted_distance_matrix, nn_start_index, numNN)
        actual_knns = _get_nns(id, trueid_seq_data_map, true_distance_matrix, nn_start_index, numNN)
        @assert length(predicted_knns) == length(actual_knns) (length(predicted_knns), length(actual_knns))

        for kidx in 1:length(kvalues)
            k = kvalues[kidx]
            for t in epoch_recall_at_k.tvalues
                epoch_recall_dict[t][k] += top_k_recall(predicted_knns, actual_knns, T=t, K=k)
            end
        end
    end

    # Normalize by the number of samples used to compute the recall sum
    for (t, rdict) in epoch_recall_dict
        for (k, r) in rdict
            epoch_recall_dict[t][k] = r/num_samples
        end
    end

    # Create a single recall-item plot containing topN curves
    avg_epoch_recall_dict = Dict()
    for (t, rdict) in epoch_recall_dict
        # average recall at each k value, for the given t
        ratk = [rdict[k] for k in kvalues]
        avg_epoch_recall_dict[t] = ratk
    end

    # Save a plot to a new directory
    save_dir = joinpath(plot_save_path, "recall_at_k")
    plot_title = string("Average recall at K Averaged over: ", num_samples, " sequencesWith dataset size: ", n_sequences)
    mkpath(save_dir)
    fig = plot(kvalues, [avg_epoch_recall_dict[i] for i in [1, 5, 10, 25, 50, 100]],
     label=["top1Recall" "top5Recall" "top10Recall" "top25Recall" "top50Recall" "top100Recall"],
        title=plot_title,
        xlabel="Number of items",
        ylabel="Recall"
    )
    savefig(fig, joinpath(save_dir, string(identifier,  ".png")))

    return epoch_recall_dict
end
