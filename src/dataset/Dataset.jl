module Dataset
    include("dataset_utils.jl")
    include("sequence_dataset.jl")
    include("synthetic_dataset.jl")

    using Random
    using Statistics
    using StatsBase

    using Flux: onehot

    using Zygote: gradient, @ignore
    using Distances

    using BenchmarkTools
    using Plots
    using Distributions
    using Printf

    function get_sequences(args, n_sequences::Int64; fastq_filepath::String)::Array{String}
        if args.USE_SYNTHETIC_DATA == true
            sequences = Dataset.generate_synethetic_sequences(
                n_sequences, args.MAX_STRING_LENGTH,
                args.MAX_STRING_LENGTH, args.ALPHABET,
                ratio_of_random=args.RATIO_OF_RANDOM_SAMPLES,
                similarity_min=args.SIMILARITY_MIN,
                similarity_max=args.SIMILARITY_MAX
            )
        elseif args.USE_SEQUENCE_DATA == true
            sequences = Dataset.read_sequence_data(
                args,
                n_sequences,
                fastq_filepath=fastq_filepath
            )
        else
            throw("Must provide a valid dataset type")
        end
        return sequences
    end

    function dataset_helper(sequences::Array{String}, max_seq_length::Int64,
         alphabet::Vector{Char}, alphabet_symbols::Vector{Symbol},
         pairwise_distanceFun::Function, weightFunctionMethod::String,
         distance_matrixNormMethod::String="max", numNN::Int64=100, sampled_triplet_nns::Int64=100)

        sampled_triplet_nns = min(sampled_triplet_nns, numNN - 1)
        n_sequences = length(sequences)
        timeGetone_hot_encode_sequences = @elapsed begin
            oneHotEncodedSequences = one_hot_encode_sequences(sequences, max_seq_length, alphabet_symbols)
        end

        batch_keys = ["Idacr", "Idpos", "Idneg", "Sacr", "Spos", "Sneg", "Xacr", "Xpos", "Xneg", "Dpos", "Dneg", "DPosNeg"]

        timeGetpairwise_distances = @elapsed begin
            seqIdMap, true_distance_matrix = pairwise_distanceFun(sequences)
            meanDistance = mean(true_distance_matrix)
            # Normalize distance matrix by maximum length
            if distance_matrixNormMethod == "mean"
                true_distance_matrix = true_distance_matrix / meanDistance

            elseif distance_matrixNormMethod == "max"
                true_distance_matrix = true_distance_matrix / max_seq_length

            else
                throw("Invalid normalization scheme")
            end
        end

        time_create_id_seq_data_map = @elapsed begin
            id_seq_data_map = Dict()

            numNN = min(n_sequences, numNN)

            numCollisions = 0
            for (seq, id) in seqIdMap
                id_seq_data_map[id] = Dict(
                    "seq" => seq,
                    "oneHotSeq" => oneHotEncodedSequences[id],
                    "topKNN" => sortperm(true_distance_matrix[id, 1:end])[1:numNN]  # Index 1 is always itself
                )
            end
        end
        get_distance_matrix() = true_distance_matrix
        getSeqIdMap() = seqIdMap
        get_id_seq_data_map() = id_seq_data_map
        getDistance(id1, id2) = true_distance_matrix[id1, id2]
        get_mean_distance() = meanDistance
        numSeqs() = n_sequences

        function getNNs(id)
            return id_seq_data_map[id]["topKNN"]
        end

        function _get_valid_triplet(nn_start_index::Int64 = 2)
            areUnique = false

            ida = dpn = dai = daj = i = j = Nothing

            numCollisions = 0

            while areUnique != true
                ida = rand(1:n_sequences)  # Anchor
                # Choose positive/negative sample from top K nearest neighbours
                # Don't allow the pos/neg ID to be the same as the anchor (start index 2)
                a_nns = id_seq_data_map[ida]["topKNN"][nn_start_index:sampled_triplet_nns + 1]  # Only consider the top K NNs when creating the triplets

                # Sample both I and J from NNs
                if weightFunctionMethod == "uniform"
                    i = sample(a_nns)
                    j = sample(a_nns)
                elseif weightFunctionMethod == "ranked"
                    # If only I is sampled using the ranked formulation, J
                    # will typically be a sequence which is ranom WRT the reference
                    sampling_weights = ProbabilityWeights(Array(range(sampled_triplet_nns, 1,step=-1) / sum(range(sampled_triplet_nns, 1, step=-1))))
                    i = sample(a_nns, sampling_weights)
                    j = sample(a_nns, sampling_weights)
                elseif weightFunctionMethod == "inverseDistance"
                    # Sample based on similarity such that similar sequences are significantly likely to be sampled
                    # We do this so the algorithm is biased to see more similar samples than non-similar samples
                    sampling_weights =ProbabilityWeights( (1 ./ a_nns) ./ sum(1 ./ a_nns) )
                    i = sample(a_nns, sampling_weights)
                    j = sample(a_nns, sampling_weights)
                else
                    throw("Invalid sampling method $(weightFunctionMethod)")
                end

                dai = true_distance_matrix[ida, i]
                daj = true_distance_matrix[ida, j]

                # TODO: Assumes distance is symmetric
                dpn = true_distance_matrix[i, j]

                # Note: Here we don't allow the same string to be compared with itself
                # Obtain sampling weight function
                if any([isapprox(dai, 0), isapprox(daj, 0), isapprox(dpn, 0)])
                    numCollisions += 1
                    continue
                else
                    areUnique = true
                end
            end

            if dai < daj
                idp, idn = i, j
                dap, dan = dai, daj
            else
                idp, idn = j, i
                dap, dan = daj, dai
            end

            return ida, idp, idn, dap, dan, dpn, numCollisions

        end

        function getTripletDict()

            idAcr, idPos, idNeg, dPos, dNeg, dPosNeg, colls = _get_valid_triplet()

            return Dict(
                # Seqs
                "Sacr" => id_seq_data_map[idAcr]["seq"],
                "Spos" => id_seq_data_map[idPos]["seq"],
                "Sneg" => id_seq_data_map[idNeg]["seq"],
                # Ids
                "Idacr" => idAcr,
                "Idpos" => idPos,
                "Idneg" => idNeg,
                # OneHots
                "Xacr" => id_seq_data_map[idAcr]["oneHotSeq"],
                "Xpos" => id_seq_data_map[idPos]["oneHotSeq"],
                "Xneg" => id_seq_data_map[idNeg]["oneHotSeq"],
                # Distances
                "Dpos" => dPos,
                "Dneg" => dNeg,
                "DPosNeg" => dPosNeg
            )
        end

        function get_triplet_batch(n::Int64)

            batch = Dict(
                # Seqs
                "Sacr" => Vector{String}(),
                "Spos" => Vector{String}(),
                "Sneg" => Vector{String}(),
                # Ids
                "Idacr" => Vector{Int64}(),
                "Idpos" => Vector{Int64}(),
                "Idneg" => Vector{Int64}(),
                # OneHots
                "Xacr" => [],
                "Xpos" => [],
                "Xneg" => [],
                # Distances
                "Dpos" => Vector{Float32}(),
                "Dneg" => Vector{Float32}(),
                "DPosNeg" => Vector{Float32}(),
                "bsize" => n,
                "perm" => Array(range(1, n, step=1))
            )
            for _ in 1:n
                s = getTripletDict()
                for (k, v) in s
                    push!(batch[k], v)
                end
            end

            # Reshape as (-1, 1, N)
            for s in ["Xacr", "Xpos", "Xneg"]
                batch[s] = format_one_hot_sequence_array(batch[s])
            end

            return batch
        end

        function shuffleTripletBatch!(batch::Dict)
            perm = Array(range(1, batch["bsize"], step=1))
            randperm!(perm)

            for (k, v) in batch
                dim = length(size(v))
                if dim == 0
                    continue
                elseif  dim == 1
                    batch[k] = v[perm]
                elseif dim == 3
                    batch[k] = v[:, :, perm]
                else
                    throw((k, dim))
                end
            end
        end

        function batchToTuple(batch::Dict)
            output = []
            for k in batch_keys
                v = batch[k]
                push!(output, v)
            end
            return tuple(output...)
        end

        function extractBatchTupleSubset(batch::Dict, i::Int64, j::Int64)::Tuple
            output = []
            for k in batch_keys
                v = batch[k]
                dim = length(size(v))
                if dim == 1
                    push!(output, v[i:j])
                elseif dim ==3
                    push!(output, v[:, :, i:j])
                else
                    throw("Not implemented")
                end
            end
            return tuple(output...)
        end

        function extractBatches(batch::Dict, numBatches::Int64)::Array
            bsize = batch["bsize"]
            chunkSize = Int(floor(bsize/numBatches))
            i = 1
            j = chunkSize

            outputChunks = []
            while j <= bsize
                push!(outputChunks, extractBatchTupleSubset(batch, i, j))
                j += chunkSize
            end

            return outputChunks
        end

        function get_batch_tuples_producer(chnl::Channel{Any}, numBatches::Int64, bsize::Int64, device)
            for i in 1:numBatches
                batchDict = get_triplet_batch(bsize)
                batchTuple = batchToTuple(batchDict)
                ids_and_reads = batchTuple[1:6]
                tensorBatch = batchTuple[7:end] |> device
                put!(chnl, (ids_and_reads, tensorBatch))
            end
        end

        ()->(numSeqs;get_distance_matrix;getSeqIdMap;get_id_seq_data_map;getDistance;getTripletDict;
        get_triplet_batch;numCollisions;shuffleTripletBatch!;extractBatches;batchToTuple;
        getNNs;get_batch_tuples_producer;get_mean_distance)
    end
end
