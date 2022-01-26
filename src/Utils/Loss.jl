include("Distance.jl")

module LossUtils

    using Flux
    using Distances
    using Statistics

    using ..LossUtils
    using ..DistanceUtils

    try
        using CUDA
        global DEVICE = Flux.gpu
        global ArrayType = Union{Vector{Float32}, Matrix{Float32}, CuArray{Float32}}
    catch e
        global DEVICE = Flux.cpu
        global ArrayType = Union{Vector{Float32}, Matrix{Float32}}
    end


    function triplet_loss(args, Xacr::ArrayType, Xpos::ArrayType,
         Xneg::ArrayType, y12::ArrayType, y13::ArrayType,
         y23::ArrayType; embedding_model::Flux.Chain, l_reg::Float64=1.0, r_reg::Float64=0.1)

        # Xacr, Xpos, Xneg, y12, y13, y23 = tensorBatch

        Embacr = embedding_model(Xacr)
        Embpos = embedding_model(Xpos)
        Embneg = embedding_model(Xneg)


        if args.DEBUG
            @assert any(isnan,Embacr) == false
            @assert any(isnan,Embpos) == false
            @assert any(isnan,Embneg) == false
        end

        posEmbedDist = DistanceUtils.embedding_distance(Embacr, Embpos, args.DISTANCE_METHOD, dims=1)  # 1D dist vector of size bsize
        negEmbedDist =  DistanceUtils.embedding_distance(Embacr, Embneg, args.DISTANCE_METHOD, dims=1)
        PosNegEmbedDist =  DistanceUtils.embedding_distance(Embpos, Embneg, args.DISTANCE_METHOD, dims=1)

        threshold = y13 - y12  # Positive

        if args.DEBUG
            @assert any(isnan,posEmbedDist) == false
            @assert any(isnan,negEmbedDist) == false
            @assert any(isnan,PosNegEmbedDist) == false
            @assert any(isnan,threshold) == false
            @assert minimum(threshold) >= 0
        end

        rankLoss = relu.(posEmbedDist - negEmbedDist + threshold)
        mseLoss = ((posEmbedDist - y12).^2 + (negEmbedDist - y13).^2 + (PosNegEmbedDist - y23).^2)

        if args.DEBUG
            @assert minimum(mseLoss) >= 0
        end

        rankLoss = l_reg * mean(rankLoss)
        mseLoss = r_reg * mean(sqrt.(mseLoss))

        return rankLoss, mseLoss, rankLoss + mseLoss
    end

end
