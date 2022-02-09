using BioSequences
using FASTX
using StatsBase

using Printf

function read_sequence_data(args, max_sequences::Union{Int64, Float64}; fastq_filepath::String="/home/jon/JuliaProjects/NeuralOverlap/data_fetch/phix_train.fastq")
    reader = open(FASTQ.Reader, fastq_filepath)
    sequence_set = Set{String}()

    expected_set = Set(args.ALPHABET)
    max_sequence_length = args.MAX_STRING_LENGTH

    for r in reader
        ## Do something
        seq = String(sequence(r))

        # Prefix
        prefix = seq[1:max_sequence_length]
        if Set(prefix) == expected_set
            push!(sequence_set, prefix)
        end

        # Suffix
        suffix = seq[end - max_sequence_length + 1: end]
        if Set(suffix) == expected_set
            push!(sequence_set, suffix)
        end
    end

    sequence_array = collect(sequence_set)

    n = Int64(min(max_sequences, length(sequence_array)))
    if isnothing(max_sequences) == false
        sequence_array = sample(sequence_array, n, replace=false)
    end
    return sequence_array
end
