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


    function generate_synethetic_sequences(n_sequences::Int64, min_seq_length::Int64,
         max_seq_length::Int64, alphabet::Vector{Char};
         ratio_of_random::Float64=0.05, similarity_min::Float64=0.6, similarity_max::Float64=0.95)::Array{String}

        # It's very likely for a sequence to be similar to another sequence, but
        # most sequences are dissimilar to each other

        # First we populate the sequence set with random sequences
        core_sequence_set = Set()
        sequence_set = Set()

        # Core samples from which the rest are based off of
        for i in 1:Int(floor(n_sequences*ratio_of_random))
            n = rand(min_seq_length:max_seq_length)
            push!(core_sequence_set, randstring(alphabet, n))
        end

        # Make each core sequence have, on average, 100 similar sequences
        while length(sequence_set) + length(core_sequence_set) < n_sequences
            n = rand(min_seq_length:max_seq_length)
            ref = rand(core_sequence_set)
            s = []
            r = rand(Uniform(similarity_min, similarity_max), 1)[1]
            for char in ref
                if rand() < r
                    push!(s, char)
                else
                    push!(s, rand(alphabet))
                end
            end
            s = join(s)
            push!(sequence_set, s)
        end

        sequence_set = union(sequence_set, core_sequence_set)
        return collect(sequence_set)
    end
end
