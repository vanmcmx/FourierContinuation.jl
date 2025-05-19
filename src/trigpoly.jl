export FCTrigPoly
export leastsquares_trigpoly

"""
    FCTrigPoly(c::AbstractVecOrMat, P::Real, M::Integer)

Evaluate the trigonometric polynomial with coefficients `a`, period `P` and bandwith `M`.

> `p(x) = [1 cos(2πx) ⋯  cos(2πMx) sin(2πx) ⋯  sin(2πMx)] a`
"""
struct FCTrigPoly{T<:AbstractVecOrMat}
    a::T
    P::Real
    M::Integer

    function FCTrigPoly(a::T, P::Real, M::Integer) where {T<:AbstractVecOrMat}
        n = size(a, 1)
        nrows = 2M + 1
        M > 0 || throw(ArgumentError("bandwith $M ≤ 0"))
        P > 0 || throw(ArgumentError("period $P ≤ 0"))
        nrows == n || return throw(DimensionMismatch("nrows of coefficient matrix ≠ $nrows, got $n "))
        new{T}(a, P, M)
    end
end

(p::FCTrigPoly)(v::AbstractVector) = _eval(p, v)

(p::FCTrigPoly)(J::FCInterval) = J |> get_grid |> p

function _eval(p::FCTrigPoly, v::AbstractVector)
    C = CSmatrix(v, p.P, p.M)
    C * p.a
end

"""
    leastsquares_trigpoly(J::FCInterval, y::AbstractVecOrMat, M::Integer, P)

Generate the vector `a` of Fourier coefficients for the function values `y` at the interval `J` 
by solving the following linear least squares problem
> `a=arg minₓ ‖Cx-y‖₂`

where `C = CSmatrix(get_grid(J), P, M)`.
"""
function leastsquares_trigpoly(J::FCInterval, y::AbstractVecOrMat, P::AbstractFloat, M::Integer; verbose)
    B = CSmatrix(J, P, M)
    a = leastsquares_svd(B, y; verbose=verbose)
    FCTrigPoly(a, P, M)
end