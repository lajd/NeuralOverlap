"""
# module SyntheticDataset

- Julia version: 
- Author: jon
- Date: 2021-12-12

# Examples

```jldoctest
julia>
```
"""

module SyntheticDataset
    using Random
    using Distributions


    function generateSequences(numSequences::Int64, minSequenceLength::Int64,
         maxSequenceLength::Int64, alphabet::Vector{Char};
         ratioOfRandom=0.05, similarityMin=0.6, similarityMax=0.95)::Array{String}

        # It's very likely for a sequence to be similar to another sequence, but
        # most sequences are dissimilar to each other

        # First we populate the sequence set with random sequences
        coreSequenceSet = Set()
        sequenceSet = Set()

        # Core samples from which the rest are based off of
        for i in 1:Int(floor(numSequences*ratioOfRandom))
            n = rand(minSequenceLength:maxSequenceLength)
            push!(coreSequenceSet, randstring(alphabet, n))
        end

        # Make each core sequence have, on average, 100 similar sequences
        while length(sequenceSet) + length(coreSequenceSet) < numSequences
            n = rand(minSequenceLength:maxSequenceLength)
            ref = rand(coreSequenceSet)
            s = []
            r = rand(Uniform(similarityMin, similarityMax), 1)[1]
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

        sequenceSet = union(sequenceSet, coreSequenceSet)
        return collect(sequenceSet)
    end
end