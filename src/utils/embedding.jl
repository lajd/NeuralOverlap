include("../dataset/dataset_utils.jl")

using Flux
using ProgressBars

try
    using CUDA
    # TODO: Don't require allow scalar
    global DEVICE = Flux.gpu
    global ArrayType = Union{Vector{Float32}, Matrix{Float32}, CuArray{Float32}}
catch e
    global DEVICE = Flux.cpu
    global ArrayType = Union{Vector{Float32}, Matrix{Float32}}
end


function _encodeBatch!(embedding_model::Flux.Chain, xarr::Vector, earr::Vector)
    formattedOneHotSeqs = format_one_hot_sequence_array(xarr)
    formattedOneHotSeqs = formattedOneHotSeqs |> DEVICE
    emb = embedding_model(formattedOneHotSeqs) |> Flux.cpu
    push!(earr, embedding_model(formattedOneHotSeqs))
end


function embed_sequence_data(dataset_helper, id_seq_data_map::Dict, embedding_model::Flux.Chain, bSize::Int64)

    n = length(id_seq_data_map)  # Number of sequences

    # Xarray's index is the same as id_seq_data_map
    Xarray = []
    Earray = []

    for k in ProgressBar(1:n)
        v = id_seq_data_map[k]
        push!(Xarray, v["oneHotSeq"])

        if length(Xarray) == bSize
            _encodeBatch!(embedding_model, Xarray, Earray)
            Xarray = []
        end
    end
    _encodeBatch!(embedding_model, Xarray, Earray)
    Xarray = []

    # TODO: This does not scale to very large datasets
    E = hcat(Earray...)
    @assert size(E)[2] == n
    return E
end
