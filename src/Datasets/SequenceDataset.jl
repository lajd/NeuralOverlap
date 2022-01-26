module SequenceDataset
    using BioSequences
    using FASTX
    using StatsBase

    using Printf

    function read_sequence_data(max_sequences::Int64, max_sequence_length::Int64; fastq_filepath::String="/home/jon/JuliaProjects/NeuralOverlap/data_fetch/phix_train.fastq")
        reader = open(FASTQ.Reader, fastq_filepath)
        sequence_set = Set{String}()

        for r in reader
            ## Do something
            seq = String(sequence(r))

            # Prefix
            push!(sequence_set, seq[1:max_sequence_length])

            # Suffix
            push!(sequence_set, seq[end - max_sequence_length + 1: end])
        end

        sequence_array = collect(sequence_set)

        n = min(max_sequences, length(sequence_array))
        if isnothing(max_sequences) == false
            sequence_array = sample(sequence_array, n, replace=false)
        end
        return sequence_array
    end
end
