module DatasetUtils
    using Random
    using Statistics
    using StatsBase

    using Flux
    using Flux: onehot, onehotbatch

    using Zygote: gradient, @ignore
    using Distances

    using BenchmarkTools
    using Plots
    using Distributions
    using Printf

    function one_hot_encode_sequences(sequence_array:: Array{String}, max_sequence_length::Int64, alphabet_symbols::Vector{Symbol})::Array
        one_hot_encode_sequences = []
        for seq in sequence_array
            push!(one_hot_encode_sequences, Flux.unsqueeze(_one_hot_encode_sequence(seq, max_sequence_length, alphabet_symbols), 1))
        end
        return one_hot_encode_sequences
    end

    function _one_hot_encode_sequence(s::String, max_sequence_length::Int64, alphabet_symbols::Vector{Symbol})::Matrix
            symbols = [Symbol(char) for char in s]
            # TODO: Add padding
            @assert length(symbols) == max_sequence_length
            one_hot_batch = onehotbatch(symbols, alphabet_symbols)
            one_hot_batch = float.(one_hot_batch)
            return one_hot_batch
    end

    function one_hot_encode_sequence_batch(sequence_array:: Array{String}, max_sequence_length::Int64, b_size::Int64, alphabet_symbols::Vector{Symbol}; do_padding::Bool=true)
        one_hot_sequence_array = one_hot_encode_sequences(sequence_array, max_sequence_length, alphabet_symbols)
        if do_padding
            vpad = zeros(b_size - length(sequence_array), length(alphabet_symbols), max_sequence_length)
            one_hot_sequence_array = vcat(one_hot_sequence_array..., vpad)
        else
            one_hot_sequence_array = vcat(one_hot_sequence_array...)
        end
        one_hot_sequence_array = convert(Array{Float32}, one_hot_sequence_array)
        return one_hot_sequence_array
    end

    function format_one_hot_sequence_array(one_hot_sequence_array::Vector{Any})
        n = length(one_hot_sequence_array)
        one_hot_sequence_array = convert.(Float32, vcat(one_hot_sequence_array...))
        one_hot_sequence_array = permutedims(one_hot_sequence_array, (3, 2, 1))
        one_hot_sequence_array = reshape(one_hot_sequence_array, :, 1, n)
        return one_hot_sequence_array
    end


    function plot_sequence_distances(distance_matrix::Array; max_samples::Int64 = 2000, plot_save_path::String=".", identifier::String="")
        n = size(distance_matrix)[1]
        max_samples = min(max_samples, n)
        randSamplesX = a = sample(1:n, max_samples, replace = false)
        randSamplesY = a = sample(1:n, max_samples, replace = false)
        dists = [distance_matrix[i, j] for (i, j) in  zip(randSamplesX, randSamplesY)]
        fig = plot(scatter(1:max_samples, dists), label=["Sequence number" "Distance"], title="True distances")
        savefig(fig, joinpath(plot_save_path, string(identifier, "_", "true_sequence_distances.png")))
    end

    function plot_knn_distances(distance_matrix::Array, id_seq_data_map::Dict; plot_save_path::String=".", num_samples::Int64=10, identifier::String="", sampled_top_k_nns::Int64=100)
        for i in 1:num_samples
            refID, refData = rand(id_seq_data_map)
            nns = refData["topKNN"][1:sampled_top_k_nns]
            distances = [distance_matrix[refID, j] for j in nns]
            fig = plot(scatter(1:length(distances), distances), label=["KNN index" "Distance"], title="Random KNN distances")
            saveDir = joinpath(plot_save_path, "random_knn_distances")
            mkpath(saveDir)
            savefig(fig, joinpath(saveDir, string(identifier, "_", string(i), ".png")))
        end
    end

    function plot_triplet_batch_distances(tripletBatchDict::Dict, plot_save_path::String=".")
        Dpos = tripletBatchDict["Dpos"]
        Dneg = tripletBatchDict["Dneg"]
        Dposneg = tripletBatchDict["DPosNeg"]
        fig = plot(scatter(1:length(Dpos), [Dpos, Dneg, Dposneg], label=["Dpos" "Dneg" "Dposneg"], title="Triplet batch distances"))
        savefig(fig, joinpath(plot_save_path, string("triplet_batch_distances", ".png")))
    end

end
