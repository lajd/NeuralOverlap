include("./Constants.jl")
include("./Utils.jl")
include("./Datasets/SyntheticDataset.jl")

using BenchmarkTools
using Random
using Statistics

using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy, params, update!
using StatsBase: wsample
using Base.Iterators: partition
using Parameters: @with_kw
using Distances
using LinearAlgebra
using Zygote
using Zygote: gradient, @ignore
using BSON: @save

using Test

using .Dataset
using .Constants
using .Model


"""
x1 shape torch.Size([64, 26, 30])
x2 shape torch.Size([1664, 1, 30])
x3 shape torch.Size([1664, 8, 30])
x4 shape torch.Size([1664, 8, 15])
x5 shape torch.Size([64, 3120])
"""

# max_sequence_length = 30
# outChannels = 8
# alphabet_dim = 26
# bSize = 64
# embedding_dim = 128
# num_conv_layers = 1
# num_readout_layers = 1


@testset "Constants tests" begin
    @testset "Test flat size" begin
        computedflat_size = Constants.__get_flat_size(30, 8, 26, 1)
        @test computedflat_size == 3120
    end
end

@testset "Dataset tests" begin
    @testset "Test one_hot_encode_sequences" begin
        s1 = "ATCG"
        one_hot_encode_sequences = Dataset.one_hot_encode_sequences([s1], 4, Constants.ALPHABET_SYMBOLS)
        ohs = one_hot_encode_sequences[1]

        ohs = ohs[1, :, :]
        for i in 1:4
            @assert ohs[i, i] == true
        end
    end

    @testset "Test generate_synethetic_sequences" begin
        sequences = Dataset.generate_synethetic_sequences(10, 2, 10, Constants.ALPHABET)
        @assert length(sequences) == 10
    end


    @testset "Test TrainingDataset" begin
        trainingDataset = Dataset.dataset_helper(10, 10, 10, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwise_hamming_distance, args.DISTANCE_MATRIX_NORM_METHOD)

        distMat = trainingDataset.get_distance_matrix()
        seqIdMap = trainingDataset.getSeqIdMap()
        id_seq_data_map = trainingDataset.get_id_seq_data_map()

        id1 = 1
        id2 = 5

        expectedDistance = hamming(id_seq_data_map[id1]["seq"], id_seq_data_map[id2]["seq"])

        @assert trainingDataset.getDistance(id1, id2) == expectedDistance

        randomTriplet = trainingDataset.getRandomTripletData()

    end

end


@testset "Model tests" begin

    @testset "load model no error" begin
        model = Model.get_embedding_model(30, 4, 64, 3120, 128, num_conv_layers=1, num_readout_layers=1)
    end

    @testset "Test single layer model" begin
        flat_size = Constants.__get_flat_size(30, 8, 26, 1)
        model = Model.get_embedding_model(30, 4, 64, flat_size, 128, num_conv_layers=1, num_readout_layers=1)
        layers = model.layers

        x = convert.(Float32, zeros(64, 26, 30))

        @test size(x) == (64, 26, 30)

        # Input reshape
        x1 = layers[1](x)
        @test size(x1) == (1664, 1, 30)

        # Conv1
        x2 = layers[2](x1)
        @test size(x2) == (1664, 8, 30)

        # Reshape for MP
        x3 = layers[3](x2)
        @test size(x3) == (30, 1664, 8)
        # MP
        x4 = layers[4](x3)
        @test size(x4) == (15, 1664, 8)
        # Reshape back
        x5 = layers[5](x4)
        @test size(x5) == (1664, 8, 15)

        # Reshape to extract batch size
        x6 = layers[6](x5)
        @test size(x6) == (64, 3120)

        # Transpose for FC layer
        x7 = layers[7](x6)
        @test size(x7) == (3120, 64)

        # FC layer
        x8 = layers[8](x7)
        @test size(x8) == (128, 64)

        # Transpose for output
        x9 = layers[9](x8)
        @test size(x9) == (64, 128)
    end


    @testset "Test multi layer model" begin
        flat_size = Constants.__get_flat_size(30, 8, 26, 3)
        model = Model.get_embedding_model(30, 4, 64, flat_size, 128, num_conv_layers=3, num_readout_layers=2)
        layers = model.layers

        x = convert.(Float32, zeros(64, 26, 30))

        @test size(x) == (64, 26, 30)

        # Input reshape
        x1 = layers[1](x)
        @test size(x1) == (1664, 1, 30)

        # Conv1
        x2 = layers[2](x1)
        @test size(x2) == (1664, 8, 30)

        # Reshape for MP
        x3 = layers[3](x2)
        @test size(x3) == (30, 1664, 8)
        # MP
        x4 = layers[4](x3)
        @test size(x4) == (15, 1664, 8)
        # Reshape back
        x5 = layers[5](x4)
        @test size(x5) == (1664, 8, 15)

        # Conv2
        x6 = layers[6](x5)
        @test size(x6) == (1664, 8, 15)

        # Reshape
        x7 = layers[7](x6)
        @test size(x7) == (15, 1664, 8)

        # MP
        x8 = layers[8](x7)
        @test size(x8) == (7, 1664, 8)

        # Reshape back
        x9 = layers[9](x8)
        @test size(x9) == (1664, 8, 7)

        # Conv3
        x10 = layers[10](x9)
        @test size(x10) == (1664, 8, 7)

        # Reshape
        x11 = layers[11](x10)
        @test size(x11) == (7, 1664, 8)

        # MP
        x12 = layers[12](x11)
        @test size(x12) == (3, 1664, 8)

        # Reshape back
        x13 = layers[13](x12)
        @test size(x13) == (1664, 8, 3)

        # Reshape to extract batch size
        x14 = layers[14](x13)
        @test size(x14) == (64, 624)

        # Transpose for FC layer
        x15 = layers[15](x14)
        @test size(x15) == (624, 64)

        # FC1 layer
        x16 = layers[16](x15)
        @test size(x16) == (624, 64)

        # FC2 layer
        x17 = layers[17](x16)
        @test size(x17) == (128, 64)

        # Transpose for output
        x18 = layers[18](x17)
        @test size(x18) == (64, 128)
    end


    @testset "Test Evaluation" begin
        maxStringLength = 10
        numSeqs = 10
        dataset = Dataset.dataset_helper(numSeqs, maxStringLength, maxStringLength, Constants.ALPHABET, Constants.ALPHABET_SYMBOLS, Utils.pairwise_hamming_distance, args.DISTANCE_MATRIX_NORM_METHOD)
        batchDict = dataset.get_triplet_batch(numSeqs)
        batchTuple = dataset.batchToTuple(batchDict)

        reads = batchTuple[1:3]
        tensorBatch = batchTuple[4:end]

        r1, r2, r3 = reads
        Xacr, Xpos, Xneg, y12, y13, y23 = tensorBatch

        realDists = []
        for i in 1:numSeqs
            push!(realDists, hamming(r1[i], r2[i]))
        end

        for j in 1:numSeqs
            x1 = realDists[j]
            x2 = y12[j]
            @assert isapprox(realDists[j], y12[j] * maxStringLength)  (x1, x2)
        end
    end

    @testset "Test Recall" begin
        @testset "Test k=1,t=20" begin
            # https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
            t = 20
            predKNN = [1, -1, 3, 4, 5, 6, 7, -1, 9,-1]
            expectedRecall = [1/20, 1/20, 2/20, 3/20, 4/20, 5/20, 6/20, 6/20, 7/20, 7/20]
            actualKNN = Array(1:10)
            for k in 1:10
                @assert isapprox(Utils.top_k_recallpredKNN(actualKNN;T=t, K=k), expectedRecall[k])
            end
        end
    end

end
