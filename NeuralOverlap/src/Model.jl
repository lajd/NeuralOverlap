

module Model
    using BenchmarkTools
    using Random
    using Statistics
    using Flux

    using StatsBase: wsample
    using Base.Iterators: partition
    using Distances
    using LinearAlgebra
    using Zygote

    try
        using CUDA
        CUDA.allowscalar(false)
        global DEVICE = Flux.gpu
    catch
        global DEVICE = Flux.cpu
    end
    
    function getLossRegularizationSteps(maxEpochs::Int64)::Dict
        step = Int32(maxEpochs / 5)
        ls = Dict(
            step * 0 => (0., 10.),
            step * 1 => (10., 10.),
            step * 2 => (10., 1.),
            step * 3 => (5., 0.1),
            step * 4 => (1., 0.01),
        )
        return ls
    end

    function getRegularization(epoch::Int64, regularizationSteps::Dict, lReg::Float64, rReg::Float64)::Tuple{Float64, Float64}
        if haskey(regularizationSteps, epoch)
            return regularizationSteps[epoch]
        end
        return lReg, rReg
    end


    function getModel(maxSeqLen::Int64, bSize::Int64, flatSize::Int64, embDim::Int64)::Chain
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

    function tripletLoss(Xacr::Array{Float32}, Xpos::Array{Float32},
         Xneg::Array{Float32}, y12::Array{Float32}, y13::Array{Float32},
         y23::Array{Float32}; embeddingModel, norm, lReg::Float64=1.0, rReg::Float64=0.1)
        Embacr = embeddingModel(Xacr)
        Embpos = embeddingModel(Xpos)
        Embneg = embeddingModel(Xneg)

        posEmbedDist = norm(Embacr - Embpos, dims=2)
        negEmbedDist = norm(Embacr - Embneg, dims=2)
        PosNegEmbedDist = norm(Embpos - Embneg, dims=2)
        threshold = y13 - y12
        rankLoss = relu.(posEmbedDist - negEmbedDist + threshold)

        mseLoss = (posEmbedDist - y12).^2 + (negEmbedDist - y13).^2 + (PosNegEmbedDist - y23).^2

        return mean(lReg * rankLoss + rReg * sqrt.(mseLoss))
    end
end


# self.conv = nn.Sequential(
#     nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
#     nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
#     nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
#     nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
#     nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
#     nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
#     nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
#     nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
#     nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
#     nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
#     POOL(2),
# )
