
module Dataset
    using Random
    using Statistics
    using StatsBase

    using Flux
    using Flux: onehotbatch

    using Zygote: gradient, @ignore
    using Distances

    using BenchmarkTools
    using Plots
    using Distributions

    using ..Utils

    function generateSequences(numSequences::Int64, minSequenceLength::Int64,
         maxSequenceLength::Int64, alphabet::Vector{Char},
         ratioOfRandom=0.1, similarityMin=0.6, similarityMaxProb=0.95)::Array{String}

        # It's very likely for a sequence to be similar to another sequence, but 
        # most sequences are dissimilar to each other 
        
        n = rand(minSequenceLength:maxSequenceLength)
        sequenceSet = Set()
        push!(sequenceSet, randstring(alphabet, n))
        while length(sequenceSet) < numSequences
            n = rand(minSequenceLength:maxSequenceLength)
            if rand() < ratioOfRandom
                s = randstring(alphabet, n)
                push!(sequenceSet, s)
            else
                ref = rand(sequenceSet)
                s = []
                r = rand(Uniform(similarityMin, similarityMaxProb), 1)[1]
                for char in ref
                    if rand() < r
                        push!(s, char)
                    else
                        push!(s, rand(alphabet))
                    end
                end
                s = join(s)
                push!(sequenceSet, s)
            end
        end
        return collect(sequenceSet)
    end


    function oneHotSequences(seqArr:: Array{String}, maxSeqLen::Int64, alphabetSymbols::Vector{Symbol})::Array
        oneHotSequences = []
        for seq in seqArr
            push!(oneHotSequences, Flux.unsqueeze(oneHotEncodeSequence(seq, maxSeqLen, alphabetSymbols), 1))
        end
        return oneHotSequences
    end

    function TrainingDataset(numSequences::Int64, minSequenceLength::Int64, maxSequenceLength::Int64, alphabet::Vector{Char}, alphabetSymbols::Vector{Symbol}, pairwiseDistanceFun)
        @info("Creating dataset... with %s sequences", numSequences)
        @time begin
            generatedSequences = Dataset.generateSequences(numSequences, minSequenceLength, maxSequenceLength, alphabet)
            oneHotEncodedSequences = Dataset.oneHotSequences(generatedSequences, maxSequenceLength, alphabetSymbols)
    
            seqIdMap, distanceMatrix = pairwiseDistanceFun(generatedSequences)
    
            # Normalize distance matrix by maximum length
            distanceMatrix = distanceMatrix / maxSequenceLength
    
            idSeqDataMap = Dict()
    
            numNN = min(numSequences, 100)

            numCollisions = 0
            for (seq, id) in seqIdMap
                idSeqDataMap[id] = Dict(
                    "seq" => seq,
                    "oneHotSeq" => oneHotEncodedSequences[id],
                    "k100NN" => sortperm(distanceMatrix[id, 1:end])[2:numNN + 1]  # Don't compare example to itself
                )
            end
    
            getDistanceMatrix() = distanceMatrix
            getSeqIdMap() = seqIdMap
            getIdSeqDataMap() = idSeqDataMap
            getDistance(id1, id2) = distanceMatrix[id1, id2]
            numSeqs() = numSequences
        end

        function getNNs(id)
            return idSeqDataMap[id]["k100NN"]
        end

        @info("Dataset contains: %s unique sequences", numSequences)
        function _get_valid_triplet()
            areUnique = false

            ida = dpn = dai = daj = i = j = Nothing

            numCollisions = 0 
            while areUnique != true
                ida = rand(1:numSequences)
                # Choose positive/negative sample from 100 nearest neighbours
                # TODO: For negative sample, sample randomly?
                a_nns = idSeqDataMap[ida]["k100NN"]

                # Note: Here we don't allow the same string to be compared with itself
                i = sample(a_nns[1:end])
                j = sample(a_nns[1:end])

                dai = distanceMatrix[ida, i]
                daj = distanceMatrix[ida, j]

                # TODO: Assumes distance is symmetric
                dpn = distanceMatrix[i, j]
                
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
                "Sacr" => idSeqDataMap[idAcr]["seq"],
                "Spos" => idSeqDataMap[idPos]["seq"],
                "Sneg" => idSeqDataMap[idNeg]["seq"],
                # Ids
                "Idacr" => idAcr,
                "Idpos" => idPos,
                "Idneg" => idNeg,
                # OneHots
                "Xacr" => idSeqDataMap[idAcr]["oneHotSeq"],
                "Xpos" => idSeqDataMap[idPos]["oneHotSeq"],
                "Xneg" => idSeqDataMap[idNeg]["oneHotSeq"],
                # Distances
                "Dpos" => dPos,
                "Dneg" => dNeg,
                "DPosNeg" => dPosNeg
            )
        end

        function formatOneHotSequenceArray(Xarray)
            n = length(Xarray)
            Xarray = convert.(Float32, vcat(Xarray...))
            Xarray = permutedims(Xarray, (3, 2, 1))
            Xarray = reshape(Xarray, :, 1, n)
            return Xarray
        end

        function getTripletBatch(n::Int64)

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

                batch[s] = formatOneHotSequenceArray(batch[s])
            end

            return batch
        end

        function shuffleTripletBatch!(batch)
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

        function batchToTuple(batch)
            order = ["Idacr", "Idpos", "Idneg", "Sacr", "Spos", "Sneg", "Xacr", "Xpos", "Xneg", "Dpos", "Dneg", "DPosNeg"]
            output = []
            for k in order                
                v = batch[k]
                push!(output, v)
            end
            return tuple(output...)
        end

        function extractBatchTupleSubset(batch, i, j)::Tuple
            order = ["Idacr", "Idpos", "Idneg", "Sacr", "Spos", "Sneg", "Xacr", "Xpos", "Xneg", "Dpos", "Dneg", "DPosNeg"]
            output = []
            for k in order                
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

        function extractBatches(batch, numBatches)::Array
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

        ()->(numSeqs;getDistanceMatrix;getSeqIdMap;getIdSeqDataMap;getDistance;getTripletDict;
        getTripletBatch;numCollisions;shuffleTripletBatch!;extractBatches;batchToTuple;
        formatOneHotSequenceArray;getNNs)
    end
    

    function oneHotEncodeSequence(s::String, maxSeqLen::Int64, alphabetSymbols::Vector{Symbol})::Matrix
            symbols = [Symbol(char) for char in s]
            # TODO: Add padding
            @assert length(symbols) == maxSeqLen
            oneHotBatch = onehotbatch(symbols, alphabetSymbols)
            oneHotBatch = float.(oneHotBatch)
            return oneHotBatch
    end
    
    function oneHotBatchSequences(seqArr:: Array{String}, maxSeqLen::Int64, bSize::Int64, alphabetSymbols::Vector{Symbol})
        embeddings = Dataset.oneHotSequences(seqArr, maxSeqLen, alphabetSymbols)
        vpad = zeros(bSize - length(seqArr), 4, maxSeqLen)        
        embeddings = vcat(embeddings..., vpad)
        embeddings = convert(Array{Float32}, embeddings)
        return embeddings
    end

    function plotSequenceDistances(distanceMat; maxSamples = 500)
        n = size(distanceMat)[1]
        maxSamples = min(maxSamples, n)
        randSamplesX = a = sample(1:n, maxSamples, replace = false)
        randSamplesY = a = sample(1:n, maxSamples, replace = false)
        dists = [distanceMat[i, j] for (i, j) in  zip(randSamplesX, randSamplesY)]        
        fig = plot(scatter(1:maxSamples, dists), title="True distances")
        savefig(fig, "sequence_distances.png")
    end

    function plotKNNDistances(distanceMat, idSeqDataMap)
        refID, refData = rand(idSeqDataMap)
        nns = refData["k100NN"]
        distances = [distanceMat[refID, j] for j in nns]
        fig = plot(scatter(1:length(distances), distances), title="Random KNN distances")
        savefig(fig, "knn_distances.png")
    end

end
