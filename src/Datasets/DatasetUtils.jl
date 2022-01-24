"""
# module DatasetUtils

- Julia version:
- Author: jon
- Date: 2021-12-12

# Examples

```jldoctest
julia>
```
"""

module DatasetUtils
    using Random
    using Statistics
    using StatsBase

    using Flux
    using Flux: onehotbatch

    using Zygote: gradient, @ignore
    using Distances

    using BenchmarkTools
    using Plots
    using Distributions
    using Printf

    function one_hot_encode_sequences(seqArr:: Array{String}, maxSeqLen::Int64, alphabetSymbols::Vector{Symbol})::Array
        one_hot_encode_sequences = []
        for seq in seqArr
            push!(one_hot_encode_sequences, Flux.unsqueeze(_one_hot_encode_sequence(seq, maxSeqLen, alphabetSymbols), 1))
        end
        return one_hot_encode_sequences
    end

    function _one_hot_encode_sequence(s::String, maxSeqLen::Int64, alphabetSymbols::Vector{Symbol})::Matrix
            symbols = [Symbol(char) for char in s]
            # TODO: Add padding
            @assert length(symbols) == maxSeqLen
            oneHotBatch = onehotbatch(symbols, alphabetSymbols)
            oneHotBatch = float.(oneHotBatch)
            return oneHotBatch
    end

    function one_hot_encode_sequence_batch(seqArr:: Array{String}, maxSeqLen::Int64, bSize::Int64, alphabetSymbols::Vector{Symbol}; doPad=true)
        oneHotSeqs = one_hot_encode_sequences(seqArr, maxSeqLen, alphabetSymbols)
        if doPad
            vpad = zeros(bSize - length(seqArr), length(alphabetSymbols), maxSeqLen)
            oneHotSeqs = vcat(oneHotSeqs..., vpad)
        else
            oneHotSeqs = vcat(oneHotSeqs...)
        end
        oneHotSeqs = convert(Array{Float32}, oneHotSeqs)
        return oneHotSeqs
    end

    function format_one_hot_sequence_array(oneHotSequence)
        n = length(oneHotSequence)
        oneHotSequence = convert.(Float32, vcat(oneHotSequence...))
        oneHotSequence = permutedims(oneHotSequence, (3, 2, 1))
        oneHotSequence = reshape(oneHotSequence, :, 1, n)
        return oneHotSequence
    end


    function plot_sequence_distances(distanceMat; maxSamples = 2000, plotsSavePath=".", identifier="")
        n = size(distanceMat)[1]
        maxSamples = min(maxSamples, n)
        randSamplesX = a = sample(1:n, maxSamples, replace = false)
        randSamplesY = a = sample(1:n, maxSamples, replace = false)
        dists = [distanceMat[i, j] for (i, j) in  zip(randSamplesX, randSamplesY)]
        fig = plot(scatter(1:maxSamples, dists), label=["Sequence number" "Distance"], title="True distances")
        savefig(fig, joinpath(plotsSavePath, string(identifier, "_", "true_sequence_distances.png")))
    end

    function plot_knn_distances(distanceMat, idSeqDataMap; plotsSavePath=".", num_samples=10, identifier="", sampledTopKNNs=100)
        for i in 1:num_samples
            refID, refData = rand(idSeqDataMap)
            nns = refData["topKNN"][1:sampledTopKNNs]
            distances = [distanceMat[refID, j] for j in nns]
            fig = plot(scatter(1:length(distances), distances), label=["KNN index" "Distance"], title="Random KNN distances")
            saveDir = joinpath(plotsSavePath, "random_knn_distances")
            mkpath(saveDir)
            savefig(fig, joinpath(saveDir, string(identifier, "_", string(i), ".png")))
        end
    end

    function plot_triplet_batch_distances(tripletBatchDict, plotsSavePath=".")
        Dpos = tripletBatchDict["Dpos"]
        Dneg = tripletBatchDict["Dneg"]
        Dposneg = tripletBatchDict["DPosNeg"]
        fig = plot(scatter(1:length(Dpos), [Dpos, Dneg, Dposneg], label=["Dpos" "Dneg" "Dposneg"], title="Triplet batch distances"))
        savefig(fig, joinpath(plotsSavePath, string("triplet_batch_distances", ".png")))
    end

end
