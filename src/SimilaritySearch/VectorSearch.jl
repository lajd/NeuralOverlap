

module VectorSearch

    using Faiss

    """
        FaissIndex(embeddingDim::Int; indexTyp::String="PQ", numCodes::Int=16, numBits::Int=12)

    Creates a Faiss index and provides utilities for interaction
    """

    function FaissIndex(embeddingDim::Int; BNFString::String="IVF4096,Flat")

        # Create index
        index = Faiss.Index(embeddingDim, BNFString)

        """
            train!(vs::AbstractMatrix)

        Train the index on data
        """
        function train!(vs::AbstractArray)
            @assert size(vs)[1] == embeddingDim
            Faiss.train!(index, vs)
        end

        """
            add!(vs::AbstractMatrix)

        Add the columns of `vs` to the index.
        """
        function add!(vs::AbstractArray)
            @assert size(vs)[1] == embeddingDim
            Faiss.add!(index, vs)
        end

        """
            search(vs::AbstractMatrix, k::Integer)

        Search the index for the `k` nearest neighbours of each column of `vs`.

        Return `(D, I)` where `I` is a matrix where each column gives the ids of the `k` nearest
        neighbours of the corresponding column of `vs` and `D` is the corresponding matrix of
        distances.
        """
        function knn(vs::AbstractArray, k::Int)
            @assert size(vs)[1] == embeddingDim
            return Faiss.search(index, vs, k)
        end

        ()->(index;train!;add!;knn)
    end
end
