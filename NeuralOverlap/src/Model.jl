

module Model
    using BenchmarkTools
    using Random
    using Statistics

    using CUDA
    CUDA.allowscalar(false)
    using Flux
    using StatsBase: wsample
    using Base.Iterators: partition
    using Distances
    using LinearAlgebra
    using Zygote

    DEVICE = gpu

    function getModel(maxSeqLen, bSize, flatSize, embDim)::Chain
        embeddingModel = Chain(
            x -> reshape(x, :, 1, maxSeqLen),
            Conv((3,), 1 => 8, relu; bias = false, stride=1, pad=1),
            x -> permutedims(x, (3, 1, 2)),
            MaxPool((2,); pad=0),
            x -> permutedims(x, (2, 3, 1)),
            x -> reshape(x, (bSize, :)),
            x -> transpose(x),
            Dense(flatSize, embDim),
            x -> transpose(x),
        )

        return embeddingModel
    end

    function tripletLoss(Xacr, Xpos, Xneg, y12, y13, y23; embeddingModel, norm)
        Embacr = embeddingModel(Xacr)
        Embpos = embeddingModel(Xpos)
        Embneg = embeddingModel(Xneg)

        posEmbedDist = norm(Embacr - Embpos, dims=2)
        negEmbedDist = norm(Embacr - Embneg, dims=2)
        PosNegEmbedDist = norm(Embpos - Embneg, dims=2)
        threshold = y13 - y12
        rankLoss = relu.(posEmbedDist - negEmbedDist + threshold)

        mseLoss = (posEmbedDist - y12).^2 + (negEmbedDist - y13).^2 + (PosNegEmbedDist - y23).^2

        L = 1
        R = 0.1

        return mean(R * rankLoss + L * sqrt.(mseLoss))
    end
end
