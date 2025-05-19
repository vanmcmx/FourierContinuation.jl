export rowmod_gramschmidt

"""
    modified_gramschmidt(A::Matrix{T}; tol=eps(T)) where {T}

Compute a QR Factorization of a real matrix A using the row-wise modified Gram-Schmidt Algorithm
See §3.1 of Leon S. J., et al. *Gram-Schmidt orthogonalization: 100 years and more.*
"""
function rowmod_gramschmidt(A::AbstractMatrix; tol=eps(eltype(A)), verbose::Bool)
    n = size(A, 2)
    Q = copy(A) # orthonormal matrix initialization
    R = zeros(eltype(A), n, n) |> UpperTriangular # upper triangular matrix initialization
    
    @views for k = 1:n
        R[k, k] = Q[:, k] |> norm
        @assert R[k, k] ≥ tol "rank deficient matrix detected by Gram-Schmidt"
        Q[:, k] ./= R[k, k]
        for j = k+1:n
            R[k, j] = dot(Q[:, j], Q[:, k])
            Q[:, j] .-= R[k, j] * Q[:, k]
        end
    end
    
    #=
    @info eltype(A)
    @info tol
    for k = 1:n
        R[k, k] = Q[:, k] |> norm
        @assert R[k, k] ≥ tol @sprintf "rank deficient matrix detected by Gram-Schmidt %1.3e" R[k,k]
        Q[:, k] ./= R[k, k]
        R[k, k+1:n] .= Q[:, k+1:n]'Q[:, k]
        Q[:, k+1:n] .-= Q[:, k] * R[k, k+1:n]'
    end
    =#
    verbose && info_qr(A, Q, R)
    Q, R
end

function info_qr(A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix)
    errA = norm(vec(A - Q * R), Inf)
    errO = orthoerror(Q)
    @info @sprintf "QR\n‖A - QR‖∞ = %1.3e ‖QᵀQ - I‖∞ = %1.3e" errA errO
end