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


    function getReadSequenceData(maxSequences=nothing; fastqFilePath="/home/jon/JuliaProjects/NeuralOverlap/data/raw_data/miseq_reads_R1.fastq")
        reader = open(FASTQ.Reader, fastqFilePath)
        i = 0
        local data
        sequenceSet = Set{String}()

        prefixSuffixLength = 128
        for r in reader
            ## Do something
            seq = String(sequence(r))
            l = length(seq)
            prefix = seq[1:prefixSuffixLength]
            suffix = seq[l-prefixSuffixLength + 1:end]
            push!(sequenceSet, [prefix, suffix]...)
            i += 2
        end

        sequenceArray = collect(sequenceSet)

        if isnothing(maxSequences) == false
            sequenceArray = sample(sequenceArray, maxSequences, replace=false)
        end
        return sequenceArray
    end
end