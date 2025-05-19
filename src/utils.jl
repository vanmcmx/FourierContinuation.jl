export orthoerror, numrank, relerr
export info_svd, info_lsq, info_relerr

"""
    orthoerror(Q::AbstractMatrix{T}) where {T}

Compute ‖vec(QᵀQ-I)‖∞ without storing the matrix QᵀQ.
"""
function orthoerror(Q::AbstractMatrix)
    err0(v, w) = dot(v, w) |> abs
    err1(v) = dot(v, v) - oneunit(eltype(v)) |> abs
    n = size(Q, 2)
    E = mapreduce(err1, max, eachcol(Q))
    @views for i = 1:n
        for j in setdiff(1:n, i)
            E = max(E, err0(Q[:, i], Q[:, j]))
        end
    end
    return E
end

function relerr(x::AbstractVector, xapprox::AbstractVector) 
    mx = length(x)
    ma = length(xapprox)
    nx = norm(x, Inf) 
    msg1 = "length(x) = $mx ≠ $ma = length(xapprox)"
    msg2 = "division by small norm in relative error"
    mx == ma || throw(DimensionMismatch(msg1))
    @assert nx ≥ eps(eltype(x)) msg2
    norm(x - xapprox, Inf) / nx
end

function numrank(F::SVD, rtol::AbstractFloat)
    S = F.S
    check_threshold(σ) = σ > rtol * first(S)
    count(check_threshold, S) 
end

"""
    info_svd(F::SVD; r::Integer)

Show the condition number and the numerical rank using the SVD factorization provided by `F`.
In addition compute the orthogonallity loss of the factors `U` and `V` of the SVD.
"""
function info_svd(F::SVD, r::Integer)
    U, S, V = F
    m = size(U, 1)
    n = size(V, 1)
    κ = reduce(\, extrema(S)) # condition number 
    errU = orthoerror(U)
    errV = orthoerror(V)
    @info @sprintf "SVD\nκ₂(Bos) = %1.3e rank(Bos) = %i \nsize(Bos) = %i × %i" κ r m n
    @info @sprintf "‖UᵀU - I‖∞ = %1.3e ‖VᵀV - I‖∞ = %1.3e" errU errV
end

"""
    info_lsq(A::AbstractMatrix{T}, x::AbstractVecOrMat{T}, b::AbstractVecOrMat{T}) where {T}

Show thw residual error =  max{‖A⋅x - bⱼ‖₂², bⱼ ∈ Cols(b)}
"""
function info_lsq(A::AbstractMatrix, x::AbstractVecOrMat, b::AbstractVecOrMat)
    m, n = size(A)
    p = size(x, 1)
    q = size(b, 1)
    q ≤ m || throw(DimensionMismatch("nrows A = $m < $q = nrows b"))
    p == n || throw(DimensionMismatch("ncols A = $n ≠ $p = nrows x"))
    Ax = A * x
    @views Ax[1:q, :] .-= b
    reserr = mapreduce(norm, max, eachcol(Ax))
    @info @sprintf "Least Squares\nmax{‖A⋅x - bⱼ‖², bⱼ ∈ Cols(b)} = %1.3e" reserr
end

function info_relerr(x::AbstractVector, xapprox::AbstractVector, msg::AbstractString)
    @info @sprintf "%s %1.3e" msg relerr(x, xapprox)
end