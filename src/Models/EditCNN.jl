
module EditCNN
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
    using Lathe.models: predict, SimpleLinearRegression

    try
        using CUDA
        global DEVICE = Flux.gpu
        global ArrayType = Union{Vector{Float32}, Matrix{Float32}, CuArray{Float32}, Array{Float32}}
    catch e
        global DEVICE = Flux.cpu
        global ArrayType = Union{Vector{Float32}, Matrix{Float32}, Array{Float32}}
    end

    function get_loss_scaling(epoch::Int64, regularization_steps::Dict, l_reg::Float64, r_reg::Float64)::Tuple{Float64, Float64}
        if haskey(regularization_steps, epoch)
            return regularization_steps[epoch]
        end
        return l_reg, r_reg
    end


    function _get_input_conv_layer(;k::Int64=3, c::Int64=8, activation=relu, use_batchnorm::Bool=false,
         use_input_batchnorm=false, pooling_method="max", pool_kernel=2)::Array
        # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        layers = []

        if pooling_method == "max"
            pool = MaxPool((pool_kernel,); pad=0)
        elseif pooling_method == "mean"
            pool = MeanPool((pool_kernel,); pad=0)
        else
            throw("Invalid pooling method")
        end

        if use_batchnorm
            if use_input_batchnorm
                push!(layers, BatchNorm(1, identity))
            end
            push!(layers, Conv((k,), 1 => c, identity; bias = false, stride=1, pad=1))
            push!(layers, BatchNorm(c, identity))
            push!(layers, pool)
        else
            push!(layers, Conv((k,), 1 => c, identity; bias = false, stride=1, pad=1))
            push!(layers, pool)
        end
        return layers
    end

    function _get_intermediate_conv_layers(l_in::Int64, n_layers::Int64; k::Int64=3, c::Int64=8, activation=relu, use_batchnorm::Bool=false, pooling_method::String="max", pool_kernel::Int64=2, conv_activation_fn_mod::Int64=1)::Array
        # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        layers = []

        if pooling_method == "max"
            pool = MaxPool((pool_kernel,); pad=0)
        elseif pooling_method == "mean"
            pool = MeanPool((pool_kernel,); pad=0)
        else
            throw("Invalid")
        end

        bn_dim = l_in
        i = 1  # Starts with input conv
        if n_layers > 0
            for _ in 1:n_layers
                i += 1
                if use_batchnorm
                    push!(layers, Conv((k,), c => c, identity; bias = false, stride=1, pad=1))
                    push!(layers, BatchNorm(c, identity))
                    push!(layers, pool)
                    if mod(i, conv_activation_fn_mod) == 0
                        push!(layers, x -> activation.(x))
                    end
                    bn_dim = _get_conv_mp_output_size(bn_dim, 1, conv_kernel=k, pool_kernel=pool_kernel)  # Add one for input
                else
                    push!(layers, Conv((k,), c => c, identity; bias = false, stride=1, pad=1))
                    push!(layers, pool)
                    if mod(i, conv_activation_fn_mod) == 0
                        push!(layers, x -> activation.(x))
                    end
                end
            end
        end
        return layers
    end

    function _get_readout_layers(l_in::Int64, embedding_dim::Int64, n_layers::Int64; activation=relu, use_dropout::Bool=false, dropout_p=0.4, use_batchnorm=false)::Array
        # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        layers = []

        if n_layers > 1
            for _ in 1:n_layers -1
                if use_batchnorm
                    push!(layers, Dense(l_in, l_in, identity))
                    push!(layers, BatchNorm(l_in, activation))
                else
                    push!(layers, Dense(l_in, l_in, activation))
                end
                if use_dropout
                    push!(layers, Dropout(dropout_p))
                end
            end
        end

        # Identity activation on output layer
        push!(layers, Dense(l_in, embedding_dim))

        return layers
    end

    function get_kernel_output_dim(l::Int64, k::Int64;p::Int64=1, s::Int64=1, d::Int64=1)
        """
        https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        """
        out = Int(floor((l + 2*p - d*(k-1) -1)/s + 1))
        return out
    end

    _get_conv_output_dim(l::Int64, k::Int64) = get_kernel_output_dim(l, k)

    _get_pool_output_dim(l::Int64, k::Int64) = get_kernel_output_dim(l, k, p=0, s=k)

    function _get_conv_mp_output_size(l_in::Int64, num_conv_layers::Int64; conv_kernel::Int64=3, pool_kernel::Int64=2)
        out(in) = _get_pool_output_dim(_get_conv_output_dim(in, conv_kernel), pool_kernel)
        l_out = l_in
        for _ in 1:num_conv_layers
            l_out = out(l_out)
        end
        return l_out
    end

    function _get_flat_size(l_in::Int64, num_conv_layers::Int64; c::Int64=8, conv_kernel::Int64=3, pool_kernel::Int64=2)
        conv_mp_size = _get_conv_mp_output_size(l_in, num_conv_layers, conv_kernel=conv_kernel, pool_kernel=pool_kernel)
        flat_size = conv_mp_size * c
        return flat_size
    end


    function get_embedding_model(max_sequence_length::Int64, alphabet_dim::Int64, embedding_dim::Int64; n_intermediate_conv_layers::Int64=0,
        num_readout_layers::Int64=1, readout_activation_fn=relu, conv_activation_fn=relu, k::Int64=3, c::Int64=8, use_batchnorm::Bool=false,
        use_input_batchnorm::Bool=false, use_dropout::Bool=false, pooling_method::String="max", pool_kernel::Int64=2,
        conv_activation_fn_mod::Int64=1, use_input_linear_model::Bool=true, normalize_embeddings::Bool=true)::Chain

        l_in = max_sequence_length * alphabet_dim

        flat_size = _get_flat_size(l_in, n_intermediate_conv_layers + 1, c=c, conv_kernel=k, pool_kernel=pool_kernel)  # Add one for input
        l_intermediate = _get_flat_size(l_in, 1, c=c, conv_kernel=k, pool_kernel=pool_kernel)  # one for input

        modelLayers = [
            # Input Convolution
            _get_input_conv_layer(activation=conv_activation_fn, k=k, c=c, use_batchnorm=use_batchnorm, use_input_batchnorm=use_input_batchnorm, pooling_method=pooling_method, pool_kernel=pool_kernel)...,
            # Intermediate Convolutions
            _get_intermediate_conv_layers(l_intermediate, n_intermediate_conv_layers, activation=conv_activation_fn, k=k, c=c, use_batchnorm=use_batchnorm, pooling_method=pooling_method, pool_kernel=pool_kernel, conv_activation_fn_mod=conv_activation_fn_mod)...,
            # Flatten to vector
            flatten,
            # FC Readouts
            _get_readout_layers(flat_size, embedding_dim, num_readout_layers, use_dropout=use_dropout, activation=readout_activation_fn, use_batchnorm=use_batchnorm)...,
        ]

        if use_input_linear_model == true
            # Input linear model
            modelLayers = ([Dense(l_in, l_in)]..., modelLayers...)
        end

        if normalize_embeddings == true
            # L2 Normalize across the embedding dimension
            # X shape is <embedding_dim x bsize>
            modelLayers = (modelLayers..., [x -> x ./ sqrt.(sum(x .^ 2, dims=1))]...)
        end


        embedding_model = Chain(modelLayers...) |> DEVICE

        return embedding_model
    end

end
