

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

    function _getInputConv1D(n::Int64=3, k::Int64 = 8)::Conv
        return Conv((n,), 1 => k, relu; bias = false, stride=1, pad=1)
    end


    function _getIntermediateConv1D(n::Int64=3, k::Int64=8)::Conv
        return Conv((n,), k => k, relu; bias = false, stride=1, pad=1)
    end

    function _getMaxPool1D(n::Int64 = 2)::MaxPool
        return MaxPool((n,); pad=0)
    end

    function _getIntermediateConvLayers(numLayers::Int64)::Array
        layers = [
            _getInputConv1D(),
            _getMaxPool1D(),
        ]

        for _ in 1:numLayers
            push!(layers, _getMaxPool1D())
            push!(layers, _getIntermediateConv1D())
        end
        return layers
    end

    function _getFCOutput(inputSize::Int64, outputSize::Int64, numLayers::Int64; activation=relu)::Array
        layers = []

        for _ in 1:numLayers - 1
            push!(layers, Dense(inputSize, inputSize))
            push!(layers, x -> activation.(x))
        end

        push!(layers, Dense(inputSize, outputSize))

        return layers
    end



    function getModel(maxSeqLen::Int64, bSize::Int64, flatSize::Int64, embDim::Int64; numConvLayers::Int64=3, numFCLayers::Int64=2)::Chain
        
        embeddingModel = Chain(
            # Input
            x -> reshape(x, :, 1, maxSeqLen),
            # Reshaped
            # Conv layers
            _getIntermediateConvLayers(numConvLayers)...,
            # Output
            x -> reshape(x, (bSize, :)),
            x -> transpose(x),
            # FC layers
            _getFCOutput(flatSize, embDim, numFCLayers)...,
            x -> transpose(x),
        )

        # if true
        #     # x = zeros(512, 4, 512)
        #     # x1 = reshape(x, :, 1, 512)  # (2048, 1, 512)
        #     # x2 = c1(x1)  # (2048, 8, 512)
        #     # x3 = mp(x2)  # (1024, 8, 512)
        #     # x4 = c2(x3)  # (1024, 8, 512)
        #     # x5 = mp(x4)  # (512, 8, 512)
        #     # x6 = c3(x5)  # (512, 8, 512)
        #     # x7 = mp(x6)  # (256, 8, 512)

        #     # x8 = c3(x7)  # (256, 8, 512)
        #     # x9 = mp(x8)  # (128, 8, 512)

        #     # x10 = c3(x9)  # (128, 8, 512)
        #     # x11 = mp(x10)  # (64, 8, 512)

        #     # x12 = c3(x11)  # (64, 8, 512)
        #     # x13 = mp(x12)  # (32, 8, 512)

        #     # x14 = c3(x13)  # (32, 8, 512)
        #     # x15 = mp(x14)  # (16, 8, 512)

        #     # x16 = c3(x15)  # (16, 8, 512)
        #     # x17 = mp(x16)  # (8, 8, 512)

        #     # x18 = c3(x17)  # (8, 8, 512)
        #     # x19 = mp(x18)  # (4, 8, 512)

        #     # c1 = Conv((3,), 1 => 8, relu; bias = false, stride=1, pad=1)
        #     # c2 = Conv((3,), 8 => 8, relu; bias = false, stride=1, pad=1)
        #     # c3 = Conv((3,), 8 => 8, relu; bias = false, stride=1, pad=1)
        #     # c4 = Conv((3,), 8 => 8, relu; bias = false, stride=1, pad=1)
        #     # c5 = Conv((3,), 8 => 8, relu; bias = false, stride=1, pad=1)
        #     # c6 = Conv((3,), 8 => 8, relu; bias = false, stride=1, pad=1)
        #     # c7 = Conv((3,), 8 => 8, relu; bias = false, stride=1, pad=1)
        #     # c8 = Conv((3,), 8 => 8, relu; bias = false, stride=1, pad=1)
        #     # c9 = Conv((3,), 8 => 8, relu; bias = false, stride=1, pad=1)
    
        #     # mp = MaxPool((2,); pad=0)
        #     n = 3
        #     flatSize = Int(bSize * 8 * 4 / (2^(n + 1)))

        #     # 1 -> 8192
        #     # 3 -> 1024
        #     # 4 -> 512
        #     # Int(MAX_STRING_LENGTH / 2 * ALPHABET_DIM * OUT_CHANNELS)
        #     # 512*4/1024*8
        #     fc1 = Dense(flatSize, embDim)
        #     fc2 = Dense(flatSize, embDim)

        #     embeddingModel = Chain(
        #         x -> reshape(x, :, 1, maxSeqLen),
        #         _getIntermediateConvLayers(n)...,
        #         # _getInputConv1D(),
        #         # _getMaxPool1D(),
        #         # _getIntermediateConv1D(), 
        #         # _getMaxPool1D(),
        #         # _getIntermediateConv1D(),
        #         # _getMaxPool1D(),
        #         # _getIntermediateConv1D(),
        #         # _getMaxPool1D(),
        #         # # Works until here
        #         # _getIntermediateConv1D(),
        #         # _getMaxPool1D(),
        #         # This worked too

        #         # _getIntermediateConv1D(),
        #         # _getMaxPool1D(),
        #         # # This worked too...
                
        #         # # The below did not work
        #         # _getIntermediateConv1D(),
        #         # _getMaxPool1D(),
                
        #         # _getIntermediateConv1D(),
        #         # _getMaxPool1D(),
        #         # _getIntermediateConv1D(),
        #         # _getMaxPool1D(),
        #         # Output
        #         x -> reshape(x, (bSize, :)),
        #         x -> transpose(x),
        #         fc1,
        #         # x -> relu.(x),
        #         # fc2,
        #         x -> transpose(x),
        #     )
        # else
        #     flatSize = bSize * 8
        #     embeddingModel = Chain(
        #     # Input [bsize, alphabetDim, maxSequenceLength]
        #     x -> reshape(x, :, 1, maxSeqLen),
        #     # Reshape as [-1, 1, maxLength]
        #     _getInputConv1D(),
        #     # (bsize*alphabetDim, 8, maxSequenceLength)
        #     # x -> permutedims(x, (3, 1, 2)),
        #     # MaxPool((2,); pad=0),
        #     # x -> permutedims(x, (2, 3, 1)),
        #     _getMaxPool1D(),
        #     #
        #     # x -> reshape(x, :, 8, maxSeqLen),
        #     # Conv((3,), 8 => 8, relu; bias = false, stride=1, pad=1),
        #     _getIntermediateConvLayers(1),
        #     # _getIntermediateConv1D(),
        #     # x -> permutedims(x, (3, 1, 2)),
        #     # MaxPool((2,); pad=0),
        #     # x -> permutedims(x, (2, 3, 1)),
        #     # _getMaxPool1D(),
        #     # Output
        #     x -> reshape(x, (bSize, :)),
        #     x -> transpose(x),
        #     Dense(flatSize, embDim),
        #     x -> transpose(x),
        # )
        # end
        
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
