module DistanceUtils

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


    function pairwise_hamming_distance(sequence_array::Array{String})
        sequenceIDMap = Dict([(refSeq, i) for (i, refSeq) in enumerate(sequence_array)])
        true_distance_matrix=pairwise(hamming, sequence_array)
        return sequenceIDMap, true_distance_matrix
    end

    function pairwise_distance(X::ArrayType, method="l2")
        if method == "l2"
            return pairwise(euclidean, X)
        elseif method == "cosine"
            return pairwise(cosine_dist, X)
        else
            thow("Invalid method")
        end
    end


    function l2_norm(A::ArrayType; dims::Int64)
        B = sum(x -> x^2, A; dims=dims)
        C = sqrt.(B)
        return C
    end

    function row_cosine_similarity(a::ArrayType, b::ArrayType)
        denom = vec(l2_norm(a, dims=1)) .* vec(l2_norm(b, dims=1))
        return 1 .- diag((transpose(a)* b)) ./ denom
    end

    function embedding_distance(x1::ArrayType, x2::ArrayType, method::String; dims::Int64=1)
        if method == "l2"
            return vec(l2_norm(x1 - x2, dims=dims))
        elseif method == "cosine"
            return row_cosine_similarity(x1, x2)
        else
            throw("Method not recognized")
        end
    end


end
