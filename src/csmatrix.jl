export CSmatrix

"""
    CSmatrix(v::AbstractVector, P::AbstractFloat, M::Integer)

Given a n-vector `v`, generate the n×(2M+1) matrix `C` with the following column view:

> `C = [1 cospi.(2/P v) ⋯  cospi.(2M/P v) sinpi.(2/P v) ⋯  sinpi.(2M/P v)]`

Example:

```julia
julia> CSmatrix(2,2,ones(3))
3×5 CSmatrix{Float64, Int64, Int64, Vector{Float64}}:
 1.0  -1.0  1.0  0.0  0.0
 1.0  -1.0  1.0  0.0  0.0
 1.0  -1.0  1.0  0.0  0.0

julia>C = CSmatrix(3, 5//1, big.(1:5))
5×7 CSmatrix{BigFloat, Int64, Float32, UnitRange{BigInt}}:
 1.0   0.309017  -0.809017  -0.809017   0.951057   0.587785  -0.587785
 1.0  -0.809017   0.309017   0.309017   0.587785  -0.951057   0.951057
 1.0  -0.809017   0.309017   0.309017  -0.587785   0.951057  -0.951057
 1.0   0.309017  -0.809017  -0.809017  -0.951057  -0.587785   0.587785
 1.0   1.0        1.0        1.0        0.0        0.0        0.0

julia> eltype(C)
BigFloat
````
"""
struct CSmatrix{T, I<:Integer, R<:Real, V<:AbstractVector} <: AbstractMatrix{T}
    M::I # bandwith
    P::R # period
    v::V # evaluation vector

    function CSmatrix(M::I, P::R, v::V) where {I<:Integer, R<:Real, V<:AbstractVector}
        M > 0 || throw(ArgumentError("bandwith $M ≤ 0"))
        P > 0 || throw(ArgumentError("period $P ≤ 0"))
        T = oneunit(R) * oneunit(eltype(v)) |> cospi |> eltype
        new{T,I,R,V}(M, P, v)
    end
end

CSmatrix(J::FCInterval, P::AbstractFloat, M::Integer) = CSmatrix(M, P, get_grid(J))

CSmatrix(v::AbstractVector, P::AbstractFloat, M::Integer) = CSmatrix(M, P, v)

Base.size(C::CSmatrix) = (length(C.v), 2C.M + 1)

@inline Base.@propagate_inbounds function Base.getindex(
    C::CSmatrix{T,I,R,V},
    i::Int,
    j::Int,
) where {T,I,R,V}
    @boundscheck checkbounds(C, i, j)
    u = oneunit(T) 
    factor = (@inbounds 2C.v[i]) / C.P
    isone(j) ? u : 2 ≤ j ≤ C.M + 1 ? (j - u) * factor |> cospi : (j - C.M - u) * factor |> sinpi
end

#=
    if isone(j)
        one(T)
    elseif 2 ≤ j ≤ C.M + 1
         (j - 1) * factor |> cospi
    elseif C.M + 2 ≤ j ≤ 2C.M + 1
         (j - C.M - 1) * factor |> sinpi
    end

"""
    csbasis_ImUIz(; d::T, C::T, E::T, Z::T, no::T, modes::T, h::Real) where {T<:Real}

Generate the real DFT matrix 
> `Bos = [1 cos.(2π/P*u*(1:M)') sin.(2π/P*u*(1:M)')]`

where `M=(d+C+Z+E)/2-modes`, `P=d+C+Z+E-1`, and `u` is the union of  uniform partitions on `MatchInterval` and `ZeroInterval` 
with the same stepsize `h/no`.
"""
function csbasis_ImUIz(; N::T, d::T, C::T, E::T, Z::T, no::T, modes::T) where {T<:Integer}
    M = BlendOpBandwith(d=d, C=C, Z=Z, E=E, modes=modes)
    P = BlendOpPeriod(d=d, C=C, Z=Z, E=E)
    MatchInterval = grid_MatchInterval(N=N, d=d, no=no)
    ZeroInterval = grid_ZeroInterval(N=N, d=d, C=C, Z=Z, no=no)
    ImUIz = vcat(MatchInterval, ZeroInterval)
     csmatrix(ImUIz, P=P, M=M) # cosine-sine basis matrix for oversampled grids on MatchInterval ∪ ZeroInterval
end

"""
    csbasis_BlendInterval(; d::T, C::T, E::T, Z::T, modes::T, h::Real) where {T<:Real}

Generate the real DFT matrix 
> `Bos = [1 cos.(2π/P*u*(1:M)') sin.(2π/P*u*(1:M)')]`

where `M=(d+C+Z+E)/2-modes`, `P=d+C+Z+E-1`, and `u` is a uniform partition on `BlendInterval` 
with stepsize `h`. The paramaters `d`, `C`, `E`, `Z` and `modes` are given by `params`.
"""
function csbasis_BlendInterval(; N::T, d::T, C::T, E::T, Z::T, modes::T) where {T<:Integer}
    M = BlendOpBandwith(d=d, C=C, Z=Z, E=E, modes=modes)
    P = BlendOpPeriod(d=d, C=C, Z=Z, E=E)
    P *= one(T) / (N - 1) # FC period
     csmatrix(BlendInterval.grid, P=P, M=M) # cosine-sine basis matrix for BlendInterval
end
=#