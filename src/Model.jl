
include("./Constants.jl")
include("./Utils.jl")

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
    using Printf

    using ..Utils
    using ..Constants


    try
        using CUDA
        CUDA.allowscalar(Constants.ALLOW_SCALAR)
        global DEVICE = Flux.gpu
        global ArrayType = Union{Vector{Float32}, Matrix{Float32}, CuArray{Float32}, Array{Float32}}
    catch e
        global DEVICE = Flux.cpu
        global ArrayType = Union{Vector{Float32}, Matrix{Float32}, Array{Float32}}
    end

    function getLossScaling(epoch::Int64, regularizationSteps::Dict, lReg::Float64, rReg::Float64)::Tuple{Float64, Float64}
        if haskey(regularizationSteps, epoch)
            return regularizationSteps[epoch]
        end
        return lReg, rReg
    end


    function _getInputConvLayer(;k=3, c=8, activation=relu, withBatchnorm=false, withInputBatchnorm=false, poolingMethod="max", poolKernel=2)::Array
        # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        layers = []

        if poolingMethod == "max"
            pool = MaxPool((poolKernel,); pad=0)
        elseif poolingMethod == "mean"
            pool = MeanPool((poolKernel,); pad=0)
        else
            thow("Invalid")
        end

        if withBatchnorm
            if withInputBatchnorm
                push!(layers, BatchNorm(1, identity))
            end
            push!(layers, Conv((k,), 1 => c, identity; bias = false, stride=1, pad=1))
            push!(layers, BatchNorm(c, activation))
            push!(layers, pool)
        else
            push!(layers, Conv((k,), 1 => c, activation; bias = false, stride=1, pad=1))
            push!(layers, pool)
        end
        return layers
    end

    function _getIntermediateConvLayers(fin, numLayers::Int64; k=3, c=8, activation=relu, withBatchnorm=false, poolingMethod="max", poolKernel=2)::Array
        # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        layers = []
        
        if poolingMethod == "max"
            pool = MaxPool((poolKernel,); pad=0)
        elseif poolingMethod == "mean"
            pool = MeanPool((poolKernel,); pad=0)
        else
            thow("Invalid")
        end
    
        bnDim = fin
        if numLayers > 0
            for _ in 1:numLayers
                if withBatchnorm
                    push!(layers, Conv((k,), c => c, identity; bias = false, stride=1, pad=1))
                    push!(layers, BatchNorm(c, activation))
                    push!(layers, pool)
                    bnDim = getConvMPSize(bnDim, 1, convK=k, poolK=poolKernel)  # Add one for input
                else
                    push!(layers, Conv((k,), c => c, activation; bias = false, stride=1, pad=1))
                    push!(layers, pool)
                end
            end
        end
        return layers
    end

    function _getFCOutput(inputSize::Int64, outputSize::Int64, numLayers::Int64; activation=relu, withDropout=false, dropoutP=0.4, withBatchnorm=false)::Array
        # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        layers = []
        
        if numLayers > 1
            for _ in 1:numLayers -1
                if withBatchnorm
                    push!(layers, Dense(inputSize, inputSize, identity))
                    push!(layers, BatchNorm(inputSize, activation))
                else
                    push!(layers, Dense(inputSize, inputSize, activation))
                end
                if withDropout
                    push!(layers, Dropout(dropoutP))
                end
            end
        end

        push!(layers, Dense(inputSize, outputSize))

        return layers
    end

    function getKernalOutputDim(l, k;p=1, s=1, d=1)
        """
        https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        """
        out = Int(floor((l + 2*p - d*(k-1) -1)/s + 1))
        return out
    end

    function convOutDim(l, k)
        return getKernalOutputDim(l, k)
    end

    function poolOutDim(l, k)
        return getKernalOutputDim(l, k, p=0, s=k)
    end

    function getConvMPSize(l_in, numConvLayers; convK=3, poolK=2)
        out(in) = poolOutDim(convOutDim(in, convK), poolK)
        l_out = l_in
        for _ in 1:numConvLayers
            l_out = out(l_out)
        end
        return l_out
    end

    function getFlatSize(l_in, numConvLayers; c=8, convK=3, poolK=2)
        out(in) = poolOutDim(convOutDim(in, convK), poolK)
        l_out = l_in
        for _ in 1:numConvLayers
            l_out = out(l_out)
        end
        return l_out * c
    end


    function getModel(maxSeqLen::Int64, embDim::Int64; numIntermediateConvLayers::Int64=0,
        numFCLayers::Int64=1, FCAct=relu, ConvAct=relu, k=3, c=8, withBatchnorm=false,
        withInputBatchnorm=false, withDropout=false, poolingMethod="max", poolKernel=2)::Chain

        l_in = maxSeqLen * Constants.ALPHABET_DIM
        flatSize = getFlatSize(l_in, numIntermediateConvLayers + 1, c=c, convK=k, poolK=poolKernel)  # Add one for input
        f1 = getFlatSize(l_in, 1, c=c, convK=k, poolK=poolKernel)  # one for input

        embeddingModel = Chain(
            # Input conv
            _getInputConvLayer(activation=ConvAct, k=k, c=c, withBatchnorm=withBatchnorm, withInputBatchnorm=withInputBatchnorm, poolingMethod=poolingMethod, poolKernel=poolKernel)...,
            # Intermediate convs
            _getIntermediateConvLayers(f1, numIntermediateConvLayers, activation=ConvAct, k=k, c=c, withBatchnorm=withBatchnorm, poolingMethod=poolingMethod, poolKernel=poolKernel)...,
            flatten,
            _getFCOutput(flatSize, embDim, numFCLayers, withDropout=withDropout, activation=FCAct, withBatchnorm=withBatchnorm)...,
        ) |> DEVICE
        
        return embeddingModel
    end

    function formatOneHotSeq(x)
        x = permutedims(x, (3, 2, 1))
        # View as 1d
        x = reshape(x, :, 1, Constants.BSIZE)
        return x
    end


    function formatOneHotSequences(XSeq)
        output = []
        for x in XSeq
            x = formatOneHotSeq(x)
            push!(output, x)
        end
        return output
    end

    function tripletLoss(Xacr::ArrayType, Xpos::ArrayType,
         Xneg::ArrayType, y12::ArrayType, y13::ArrayType,
         y23::ArrayType; embeddingModel, lReg::Float64=1.0, rReg::Float64=0.1)

        # Xacr, Xpos, Xneg, y12, y13, y23 = tensorBatch
        # FIXME: The below are done on CPU to avoid scalar indexing issues
        Embacr = embeddingModel(Xacr)
        Embpos = embeddingModel(Xpos)
        Embneg = embeddingModel(Xneg)

        if Constants.DEBUG
            @assert any(isnan,Embacr) == false
            @assert any(isnan,Embpos) == false
            @assert any(isnan,Embneg) == false
        end

        posEmbedDist = Utils.EmbeddingDistance(Embacr, Embpos, Constants.DISTANCE_METHOD, dims=1) |> DEVICE  # 1D dist vector of size bsize
        negEmbedDist =  Utils.EmbeddingDistance(Embacr, Embneg, Constants.DISTANCE_METHOD, dims=1) |> DEVICE
        PosNegEmbedDist =  Utils.EmbeddingDistance(Embpos, Embneg, Constants.DISTANCE_METHOD, dims=1) |> DEVICE

        threshold = y13 - y12  # Positive

        if Constants.DEBUG
            @assert any(isnan,posEmbedDist) == false
            @assert any(isnan,negEmbedDist) == false
            @assert any(isnan,PosNegEmbedDist) == false
            @assert any(isnan,threshold) == false
            @assert minimum(threshold) >= 0
        end

        rankLoss = relu.(posEmbedDist - negEmbedDist + threshold)
        mseLoss = ((posEmbedDist - y12).^2 + (negEmbedDist - y13).^2 + (PosNegEmbedDist - y23).^2)

        if Constants.DEBUG
            @assert minimum(mseLoss) >= 0
        end

        rankLoss = lReg * rankLoss
        mseLoss = rReg * sqrt.(mseLoss)

        return mean(rankLoss), mean(mseLoss), mean(rankLoss + mseLoss)
    end
end
