include("./src/NeuralOverlap.jl")

using ..NeuralOverlap

predicted_nn_map, true_nn_map, epoch_recall_dict, faiss_index = NeuralOverlap.run_experiment()
