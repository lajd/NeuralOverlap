"""
# module SequenceDataset

- Julia version: 
- Author: jon
- Date: 2021-12-11

# Examples

```jldoctest
julia>
```
"""
module SequenceDataset
    using BioSequences
    using FASTX
    using StatsBase

    using Printf



    function getReadSequenceData(maxSequences, maxSeqLen; fastqFilePath="/home/jon/PycharmProjects/NeuralOverlap/data_fetch/phis174_simulated_reads.fa_R1.fastq")
        reader = open(FASTQ.Reader, fastqFilePath)
        i = 0
        local data
        sequenceSet = Set{String}()

        for r in reader
            ## Do something
            seq = String(sequence(r))
            l = length(seq)
            prefix = seq[1:maxSeqLen]
            @assert length(prefix) == maxSeqLen
            suffix = seq[l-maxSeqLen + 1:end]
            @assert length(suffix) == maxSeqLen
            push!(sequenceSet, [prefix, suffix]...)
            i += 2
        end

        sequenceArray = collect(sequenceSet)
        
        n = min(maxSequences, length(sequenceArray))
        if isnothing(maxSequences) == false
            sequenceArray = sample(sequenceArray, n, replace=false)
        end
        return sequenceArray
    end
end
