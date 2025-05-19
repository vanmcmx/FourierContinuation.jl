export Vandermonde

"""
   Vandermonde(x::AbstractVector{T}; d::Integer=length(x), rowidx::Integer=0) where {T<:Number}

Generate the modified Vandermonde matrix `V` with the following entries:
>  `vᵢⱼ = (j-1)xᵢʲ⁻²` if `i=r` and `vᵢⱼ = xᵢʲ⁻¹` otherwise for `i∈ eachindex(x)` and `j=1:d`.

The corresponding polynomial interpolates the slope at `x=xᵣ` instead of the function value, 
where `r=rowidx`.

Example:

```julia

```julia
julia> Vandermonde(1:6)
6×6 Vandermonde{Int64}:
 1  1   1    1     1     1
 1  2   4    8    16    32
 1  3   9   27    81   243
 1  4  16   64   256  1024
 1  5  25  125   625  3125
 1  6  36  216  1296  7776

 julia> Vandermonde(1:6, d=5)
6×5 Vandermonde{Int64}:
 1  1   1    1     1
 1  2   4    8    16
 1  3   9   27    81
 1  4  16   64   256
 1  5  25  125   625
 1  6  36  216  1296

julia> Vandermonde(1:6, d=5, rowidx=6)
6×5 Vandermonde{Int64}:
 1  1   1    1    1
 1  2   4    8   16
 1  3   9   27   81
 1  4  16   64  256
 1  5  25  125  625
 0  1  12  108  864
 ```
"""
struct Vandermonde{T, V<:AbstractVector{T}, I<:Integer} <: AbstractMatrix{T}
    x::V
    d::I
    rowidx::I
end

Vandermonde(x::AbstractVector; d::Integer=length(x), rowidx::Integer=zero(d)) = Vandermonde(x, d, rowidx)

@inline Base.@propagate_inbounds function getindex(
    V::Vandermonde,
    i::Integer,
    j::Integer,
)
    @boundscheck checkbounds(V, i, j)
    T = V.x |> eltype
     if i == V.rowidx
        if isone(j)
            V.x |> eltype |> zero
        else
            T(j-1) * (@inbounds V.x[i])^T(j-2)
        end
     else
        (@inbounds V.x[i])^T(j-1)
     end
end

size(V::Vandermonde) = (length(V.x), V.d)