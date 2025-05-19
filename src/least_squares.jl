export leastsquares_svd

"""
    leastsquares_svd(A::AbstractMatrix, b::AbstractVecOrMat; rtol::Real) * eps(T)) where {T}

Compute `xLS = arg minₓ‖Ax-[b;0]‖₂` using a SVD factorization `A=U diag(S) Vᵀ`, that is,
> `xLS =  V[:, 1:r] * Diagonal(1 ./ S[1:r]) * U[1:q, 1:r]'b`, 

where `q=size(b,1)` and `r=rank(A)`. The SVD is computed by QR iteration for better accuracy.
The numerical rank is controled by `rtol`. By default `rtol=min(size(A)...)*eps(eltype(A))`.
"""
function leastsquares_svd(
    A::AbstractMatrix,
    b::AbstractVecOrMat;
    verbose=false,
    rtol=reduce(min, size(A)) * eps(eltype(A)))

    q = size(b, 1)
    nrows = size(A, 1)
    q ≤ nrows || throw(DimensionMismatch("nrows A = $nrows < $q = nrows b"))

    F = svd(A; alg=QRIteration()) # singular value decomposition
    r = numrank(F, rtol) # numerical rank
    C = F.U[1:q, 1:r]'b # coefficients for the expansion in right singular vector
    @views for i = 1:r # scaling by the singular values
        C[i, :] ./= F.S[i]
    end
    xLS = F.V[:, 1:r] * C # projection onto the basis of left singular vectors 

    if verbose
        info_svd(F, r) # matrix conditioning and orthogonallity loss 
        info_lsq(A, xLS, b) # residual error
    end
    return xLS
end