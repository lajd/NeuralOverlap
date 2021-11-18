using BenchmarkTools
using Random
using Statistics

using CUDA
CUDA.allowscalar(false)
using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy, params, update!
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using Distances
using LinearAlgebra
using Zygote
using Zygote: gradient, @ignore


# configure device
DEVICE = gpu

ALPHABET = ['A';'T';'C';'G']

ALPHABET_SYMBOLS = [Symbol(i) for i in ALPHABET]
ALPHABET_DIM = length(ALPHABET_SYMBOLS)
MAX_STRING_LENGTH = 500


BSIZE = 256
READ_LENGTH = 300
OVERLAP_LENGTH = 50
PROB_SAME = 0.9


OUT_CHANNELS = 8
# Build module
EMBEDDING_DIM = 128
FLAT_SIZE = Int(MAX_STRING_LENGTH / 2 * ALPHABET_DIM * OUT_CHANNELS)


function generateReadsWithOverlap(n::Int64, p_same::Float64)
    """
    Generate reads with some overlaps, with modifucations

    Returns the concatenated reads, and the hamming distance between their overlaps
    """
    
    r1 = randstring(ALPHABET, n)
    r2 = ""
    r3 = ""

    for char in r1
        if rand() <= p_same
            r2 = string(r2, char)
        else
            r2 = string(r2, randstring(ALPHABET, 1))
        end

        if rand() <= p_same
            r3 = string(r3, char)
        else
            r3 = string(r3, randstring(ALPHABET, 1))
        end

    end

    y12 = hamming(r1, r2)
    y13 = hamming(r1, r3)
    y23 = hamming(r2, r3)

    Ranc = r1  # Ancor
    if y12 < y13
        Rpos = r2
        Ypos = y12
        Rneg = r3
        Yneg = y13
    else
        Rpos = r3
        Ypos = y13
        Rneg = r2
        Yneg = y12
    end

    @assert Ypos <= Yneg
 
    return Ranc, Rpos, Rneg, Ypos, Yneg, y23
end


function oneHotPositionalEmbeddingString(s::String)::Matrix
    ls = length(s)
    oneHotMatrix = Float32.(zeros(4, MAX_STRING_LENGTH))

    i = 1

    indicator = Float32(1.)
    for char in s
        char = string(char)
        if char == "A"
            oneHotMatrix[1, i] = indicator
        elseif char == "T"
            oneHotMatrix[2, i] = indicator
        elseif char == "C"
            oneHotMatrix[3, i] = indicator
        elseif char == "G"
            oneHotMatrix[4, i] = indicator
        end
        i = i + 1
    end

    return oneHotMatrix
end



function getRawDataBatch()

    x1 = []
    x2 = []
    x3 = []
    y12 = Vector{Float32}()
    y13 = Vector{Float32}()
    y23 = Vector{Float32}()

    for i in 1:BSIZE
        r1, r2, r3, y12_, y13_, y23_ = generateReadsWithOverlap(READ_LENGTH, PROB_SAME)
        push!(x1, Flux.unsqueeze(oneHotPositionalEmbeddingString(r1), 1))
        push!(x2, Flux.unsqueeze(oneHotPositionalEmbeddingString(r2), 1))
        push!(x3, Flux.unsqueeze(oneHotPositionalEmbeddingString(r3), 1))

        push!(y12, y12_)
        push!(y13, y13_)
        push!(y23, y23_)

    end
    
    x1 = convert.(Float32, vcat(x1...))
    x2 = convert.(Float32, vcat(x2...))
    x3 = convert.(Float32, vcat(x3...))
    
    return x1, x2, x3, y12, y13, y23
end


# X shape [bsize, D, MAX_LEN]

function getModel()::Chain
    embeddingModel = Chain(
        x -> reshape(x, :, 1, MAX_STRING_LENGTH),
        Conv((3,), 1 => 8, relu; bias = false, stride=1, pad=1),
        x -> permutedims(x, (3, 1, 2)),
        MaxPool((2,); pad=0),
        x -> permutedims(x, (2, 3, 1)),
        x -> reshape(x, (BSIZE, :)),
        x -> transpose(x),
        Dense(FLAT_SIZE, EMBEDDING_DIM),
        x -> transpose(x),
    )

    return embeddingModel
end

function getEmbeddedDataBatch(x1, x2, x3)
    x1 = embeddingModel(x1)
    x2 = embeddingModel(x2)
    x3 = embeddingModel(x3)
    return x1, x2, x3
end


function norm2(A; dims)
    A = sum(x -> x^2, A; dims=dims)
    A = sqrt.(A)
    return A
end


function tripletLoss(Xacr, Xpos, Xneg, y12, y13, y23)
    Embacr, Embpos, Embneg = getEmbeddedDataBatch(Xacr, Xpos, Xneg)

    posEmbedDist = norm2(Embacr - Embpos, dims=2)
    negEmbedDist = norm2(Embacr - Embneg, dims=2)
    PosNegEmbedDist = norm2(Embpos - Embneg, dims=2)
    threshold = y13 - y12
    rankLoss = relu.(posEmbedDist - negEmbedDist + threshold)

    mseLoss = (posEmbedDist - y12).^2 + (negEmbedDist - y13).^2 + (PosNegEmbedDist - y23).^2

    L = 1
    R = 1

    return mean(R * rankLoss + L * sqrt.(mseLoss))

end

function getoneHotTrainingDataset(num_batches::Int)
    dataset = []
    for i in 1:num_batches
        s1, s2, s3, y12, y13, y23 = convert.(Array{Float32}, @ignore getRawDataBatch())
        push!(dataset, (s1, s2, s3, y12, y13, y23))
    end
    return dataset
end


function trainingLoop!(loss, model, train_dataset, opt; numEpochs=100)
    @info("Beginning training loop...")
    local training_loss
    ps = params(model)

    for epoch in 1:numEpochs
        for (batch_idx, batch) in enumerate(train_dataset)
            gs = gradient(ps) do
                training_loss = loss(batch...)
                return training_loss
            end
            @info("Epoch: %s, Batch: %s, Training loss: %s", epoch, batch_idx, training_loss)
            update!(opt, ps, gs)
        end
    end
end


opt = ADAM(0.01, (0.9, 0.999))
oneHotTrainingDataset = getoneHotTrainingDataset(500) .|> DEVICE

embeddingModel = getModel() |> DEVICE

trainingLoop!(tripletLoss, embeddingModel, oneHotTrainingDataset, opt)

