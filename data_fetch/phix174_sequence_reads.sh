# Dataset is downloaded from the following URL
# https://www.ncbi.nlm.nih.gov/nuccore/NC_001422.1?report=fasta

pip install InSilicoSeq

# Train/validation
iss generate --genomes phix174.fa --model miseq --output phis174_simulated_reads.fa -n 10000

# Inference
iss generate --genomes phix174.fa --model miseq --output phis174_simulated_reads_inference.fa -n 100000
