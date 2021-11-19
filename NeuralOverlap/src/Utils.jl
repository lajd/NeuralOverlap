module Utils
    function norm2(A; dims)
        A = sum(x -> x^2, A; dims=dims)
        A = sqrt.(A)
        return A
    end
end
