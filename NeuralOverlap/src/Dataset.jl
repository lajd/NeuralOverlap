
module Dataset
    using Random
    using Statistics

    using Flux
    using Zygote: gradient, @ignore
    using Distances

    function generateReadsWithOverlap(n::Int64, p_same::Float64, alphabet, )
        """
        Generate reads with some overlaps, with modifucations

        Returns the concatenated reads, and the hamming distance between their overlaps
        """
        
        r1 = randstring(alphabet, n)
        r2 = ""
        r3 = ""

        for char in r1
            if rand() <= p_same
                r2 = string(r2, char)
            else
                r2 = string(r2, randstring(alphabet, 1))
            end

            if rand() <= p_same
                r3 = string(r3, char)
            else
                r3 = string(r3, randstring(alphabet, 1))
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

        averageY = Ypos + Yneg + y23
    
        return Ranc, Rpos, Rneg, Ypos, Yneg, y23, averageY
    end


    function oneHotPositionalEmbeddingString(s::String, maxSeqLen)::Matrix
        ls = length(s)
        oneHotSequnece = Float32.(zeros(4, maxSeqLen))

        i = 1

        indicator = Float32(1.)
        for char in s
            char = string(char)
            if char == "A"
                oneHotSequnece[1, i] = indicator
            elseif char == "T"
                oneHotSequnece[2, i] = indicator
            elseif char == "C"
                oneHotSequnece[3, i] = indicator
            elseif char == "G"
                oneHotSequnece[4, i] = indicator
            end
            i = i + 1
        end

        return oneHotSequnece
    end



    function getRawDataBatch(maxSeqLen, probSame, alphabet, bsize)

        x1 = []
        x2 = []
        x3 = []
        y12 = Vector{Float32}()
        y13 = Vector{Float32}()
        y23 = Vector{Float32}()
        yAvg = []

        for i in 1:bsize
            r1, r2, r3, y12_, y13_, y23_, yAvg_ = generateReadsWithOverlap(maxSeqLen, probSame, alphabet)
            push!(x1, Flux.unsqueeze(oneHotPositionalEmbeddingString(r1, maxSeqLen), 1))
            push!(x2, Flux.unsqueeze(oneHotPositionalEmbeddingString(r2, maxSeqLen), 1))
            push!(x3, Flux.unsqueeze(oneHotPositionalEmbeddingString(r3, maxSeqLen), 1))

            push!(y12, y12_)
            push!(y13, y13_)
            push!(y23, y23_)
            push!(yAvg, yAvg_)
        end
        
        x1 = convert.(Float32, vcat(x1...))
        x2 = convert.(Float32, vcat(x2...))
        x3 = convert.(Float32, vcat(x3...))

        # yAvg = mean(yAvg)
        yAvg=1

        # Normalize all y values
        y12 = y12 / yAvg
        y13 = y13 / yAvg
        y23 = y23 / yAvg

        return x1, x2, x3, y12, y13, y23
    end


    function getoneHotTrainingDataset(num_batches::Int, maxSeqLen, pSame, alphabet, bSize)
        dataset = []
        for i in 1:num_batches
            s1, s2, s3, y12, y13, y23 = convert.(Array{Float32}, @ignore getRawDataBatch(maxSeqLen, pSame, alphabet, bSize))
            push!(dataset, (s1, s2, s3, y12, y13, y23))
        end
        return dataset
    end

    function oneHotBatchSequences(seqArr:: Array{String}, maxSeqLen, bSize)
        
        embeddings = []
    
        for seq in seqArr
            push!(embeddings, Flux.unsqueeze(oneHotPositionalEmbeddingString(seq,maxSeqLen), 1))
        end

        vpad = zeros(bSize - length(seqArr), 4, maxSeqLen)        
        embeddings = vcat(embeddings..., vpad)

        embeddings = convert.(Array{Float32}, embeddings)
        return embeddings
    end
    

end
