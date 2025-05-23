export FCDerivative
export info_fc, derivative_error, npts_Iunit, get_period

"""
    FCDerivative(op::O, k::Integer; σ::Function=identity)
"""
struct FCDerivative{O<:FCOperator,F,I}
    op::O
    σ::F
    k::I

    function FCDerivative(op::O, σ::F, k::I) where {O<:FCOperator,F,I}
        k ≥ zero(k) || throw(ArgumentError("derivative order < 0"))
        new{O,F,I}(op, σ, k)
    end
end

function FCDerivative(::Type{T}, Op::FCGram, k::Integer=0; σ=identity) where {T<:AbstractFloat}
    change_type(x) = map(T, x)
    change_matrix(op::FCGramBlendZero) = op |> get_matrix |> change_type
    change_interval(op::FCGramBlendZero) = op |> get_interval |> get_grid |> change_type |> FCInterval
    
    N = Int(Op.N)
    C = Int(Op.C)
    Al = change_matrix(Op.leftOp)
    Ar = change_matrix(Op.rightOp)
    Il = change_interval(Op.leftOp)
    Ir = change_interval(Op.rightOp)
    Opl = FCGramBlendZero(Il, Al)
    Opr = FCGramBlendZero(Ir, Ar)
    
    Opnew = FCGram(Opl, Opr, C, N)
    FCDerivative(Opnew, σ, k)
end

(D::FCDerivative)(fvalues::AbstractVector) = spectral_derivative(D.op, fvalues, D.k; σ=D.σ)

function spectral_derivative(op::FCOperator, fvalues::AbstractVector, k::Integer; σ=identity)
    Af = op(fvalues)
    C = length(Af)
    N = length(fvalues)
    b = N + C
    b /= N - oneunit(N)
    cvalues = vcat(fvalues, Af)
    fftderivative(cvalues, b, k; (freqfilter!)=σ)
end

"""
    fftderivative(fvalues::AbstractVector{T}; b::T, k::Integer freqfilter!::Function=identity) where {T<:Real}

Compute a vector with the values of the `k`-th order derivative of a real-valued function `f` of period `P`
on a uniform partition. The derivatives are approximated by spectral differentiation, that is,
> `derivatives = real IFFT( coefficients (2π/b)ᵏ frequenciesᵏ)`

where `coefficients = real FFT(fvalues)` and `frequencies = eachindex(coeficients) - 1`.
This routine assumes that the first and the last entries of `fvalues` are different.
In addition, the Fourier coefficients can be filtered by a function `freqfilter!`.
"""
function fftderivative(fvalues::AbstractVector, b::Real, k::Integer; (freqfilter!)=identity)
    N = length(fvalues)
    coeffs = rfft(fvalues)
    freqfilter!(coeffs)
    factor = (π * im / b)^k * exp2(k)
    freqs = eachindex(coeffs)
    coeffs .*= @. factor * (freqs - 1)^k
    irfft(coeffs, N)
end

npts_Iunit(D::FCDerivative) = npts_Iunit(D.op)

npts_Iblend(D::FCDerivative) = npts_Iblend(D.op)

get_period(D::FCDerivative) = get_period(D.op)

function info_fc(D::FCDerivative, f, df)
    msg = "|fᶜ⁽ᵏ⁾(grid) - f⁽ᵏ⁾(grid) |∞/|f⁽ᵏ⁾(grid)|∞ ="
    @info @sprintf "%s %1.3e" msg derivative_error(D, f, df)
end

function derivative_error(D::FCDerivative, f, df)
    N = npts_Iunit(D)
    Iunit = grid_unit(N)
    dfvalues = Iunit .|> df
    dfcvalues = Iunit .|> f |> D
    relerr(dfvalues, dfcvalues[1:N])
end
