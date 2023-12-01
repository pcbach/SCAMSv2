using Random
using LinearAlgebra
using SparseArrays
"""
    reinitialize!(a::Arnoldi, j::Int = 0) → a

Generate a random `j+1`th column orthonormal against V[:,1:j]

Returns true if the column is a valid new basis vector.
Returns false if the column is numerically in the span of the previous vectors.
"""
function reinitialize!(arnoldi::Arnoldi{T}, j::Int=0) where {T}
    V = arnoldi.V
    v = view(V, :, j + 1)

    # Generate a new random column
    rand!(v)

    # Norm before orthogonalization
    rnorm = norm(v)

    # Just normalize, don't orthogonalize
    if j == 0
        v ./= rnorm
        return true
    end

    # Constant used by ARPACK.
    η = √2 / 2
    Vprev = view(V, :, 1:j)

    # Orthogonalize: h = Vprev' * v, v ← v - Vprev * Vprev' * v = v - Vprev * h
    h = Vprev' * v
    mul!(v, Vprev, h, -one(T), one(T))

    # Norm after orthogonalization
    wnorm = norm(v)

    # Reorthogonalize once
    if wnorm < η * rnorm
        rnorm = wnorm
        mul!(h, Vprev', v)
        mul!(v, Vprev, h, -one(T), one(T))
        wnorm = norm(v)
    end

    if wnorm ≤ η * rnorm
        # If we have to reorthogonalize thrice, then we're just numerically in the span
        return false
    else
        # Otherwise we just normalize this new basis vector
        v ./= wnorm
        return true
    end
end

"""
    orthogonalize!(arnoldi, j) → Bool

Orthogonalize arnoldi.V[:, j+1] against arnoldi.V[:, 1:j].

Returns true if the column is a valid new basis vector.
Returns false if the column is numerically in the span of the previous vectors.
"""
function orthogonalize!(arnoldi::Arnoldi{T}, j::Integer) where {T}
    V = arnoldi.V
    H = arnoldi.H

    # Constant used by ARPACK.
    η = √2 / 2

    Vprev = view(V, :, 1:j)
    v = view(V, :, j + 1)
    h = view(H, 1:j, j)

    # Norm before orthogonalization
    rnorm = norm(v)

    # Orthogonalize: h = Vprev' * v, v ← v - Vprev * Vprev' * v = v - Vprev * h
    mul!(h, Vprev', v)
    mul!(v, Vprev, h, -one(T), one(T))

    # Norm after orthogonalization
    wnorm = norm(v)

    # Reorthogonalize once
    if wnorm < η * rnorm
        rnorm = wnorm
        correction = Vprev' * v
        mul!(v, Vprev, correction, -one(T), one(T))
        h .+= correction
        wnorm = norm(v)
    end

    if wnorm ≤ η * rnorm
        # If we have to reorthogonalize thrice, then we're just numerically in the span
        H[j+1, j] = zero(T)
        return false
    else
        # Otherwise we just normalize this new basis vector
        H[j+1, j] = wnorm
        v ./= wnorm
        return true
    end
end


function lanczosSubRoutine(A, v, b)
    wp = zeros(m)
    for i in 1:n
        a = A[:, i]
        c = -1 / (2 * sqrt(v[i]))
        w_ = a' * b
        w_ = c * a * w_
        wp = wp + w_
    end
    return wp
end



"""
    iterate_arnoldi!(A, arnoldi, from:to) → arnoldi

Perform Arnoldi from `from` to `to`.
"""
function iterate_arnoldi!(C, y, arnoldi::Arnoldi{T}, range::UnitRange{Int}) where {T}
    V, H = arnoldi.V, arnoldi.H
    for j = range
        # Generate a new column of the Krylov subspace
        #mul!(view(V, :, j+1), A, view(V, :,j))
        ##BEGIN MODIFICATION!!!!################################################################
        #view(V, :, j+1) is 0
        #wp = view(V, :, j + 1)
        #w_ = zeros(1, 1)
        #w__ = zeros(size(A, 1), 1)
        #w___ = zeros(size(A, 1), 1)

        V[:, j+1] = sqrt.(y) .* V[:, j]
        V[:, j+1] = C * V[:, j+1]
        V[:, j+1] = sqrt.(y) .* V[:, j+1]
        if orthogonalize!(arnoldi, j) === false && j != size(V, 1)
            reinitialize!(arnoldi, j)
        end
    end

    return arnoldi
end
