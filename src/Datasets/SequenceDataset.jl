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
        sequenceSet = Set{String}()

        for r in reader
            ## Do something
            seq = String(sequence(r))
            push!(sequenceSet, seq[1:maxSeqLen])
        end

        sequenceArray = collect(sequenceSet)
        
        n = min(maxSequences, length(sequenceArray))
        if isnothing(maxSequences) == false
            sequenceArray = sample(sequenceArray, n, replace=false)
        end
        return sequenceArray
    end
end
