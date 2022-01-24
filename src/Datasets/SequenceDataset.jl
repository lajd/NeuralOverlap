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

    function read_sequence_data(maxSequences, maxSeqLen; fastqFilePath="/home/jon/JuliaProjects/NeuralOverlap/data_fetch/phix_train.fastq")
        reader = open(FASTQ.Reader, fastqFilePath)
        sequenceSet = Set{String}()

        for r in reader
            ## Do something
            seq = String(sequence(r))

            # Prefix
            push!(sequenceSet, seq[1:maxSeqLen])

            # Suffix
            push!(sequenceSet, seq[end - maxSeqLen + 1: end])
        end

        sequenceArray = collect(sequenceSet)

        n = min(maxSequences, length(sequenceArray))
        if isnothing(maxSequences) == false
            sequenceArray = sample(sequenceArray, n, replace=false)
        end
        return sequenceArray
    end
end
