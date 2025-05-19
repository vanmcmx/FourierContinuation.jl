export FCGramBlendZero
export get_matrix, get_interval
export npts_Iunit, npts_Iblend

"""
    FCGramBlendZero(BM::InterpolationMatrix, BMder::InterpolationMatrix, intervals::FCGramIntervals; params::FCGramParameters)

"""
struct FCGramBlendZero{T, I<:FCInterval,M<:AbstractMatrix{T}} <: OneSidedFC{T}
    Iblend::I
    AQt::M

    function FCGramBlendZero(Iblend::I, AQt::M) where {T, I<:FCInterval, M<:AbstractMatrix{T}}
        C = get_npts(Iblend)
        m = size(AQt, 1)
        msg = "nrows AQᵀ = $m ≠ $C = npts Iblend"
        C == m || throw(DimensionMismatch(msg))
        new{T,I,M}(Iblend, AQt)
    end
end

"""
    FCGramBlend0Op(::Type{DirichletBC}, params::FCGramParameters)
    FCGramBlend0Op(::Type{NeumannBC}, params::FCGramParameters)
"""
function FCGramBlendZero(::Type{DirichletBC}, params::FCGramParameters; verbose)
    M = get_bandwith(params)
    b = get_period(params)

    Is = FCGramIntervals(params)
    Ifit = fit_interval(Is, params.no)

    G = GramBasisDirichlet(Is.Imatch, verbose=verbose)
    Gos = oversample_basis(G, params.no)

    Q = get_basis(G)
    Qos = get_basis(Gos)

    p = leastsquares_trigpoly(Ifit, Qos, b, M, verbose=verbose)
    A = p(Is.Iblend)
    projection!(A, Q)
    FCGramBlendZero(Is.Iblend, A)
end

function FCGramBlendZero(::Type{NeumannBC}, params::FCGramParameters; verbose)
    M = get_bandwith(params)
    b = get_period(params)

    Is = FCGramIntervals(params)
    Ifit = fit_interval(Is, params.no)

    G = GramBasisDirichlet(Is.Imatch, verbose=verbose)
    Gder = GramBasisNeumann(Is.Imatch, verbose=verbose)
    Gos = oversample_basis(G, params.no)

    R = get_coefficients(G)
    Rder = get_coefficients(Gder)
    Qder = get_basis(Gder)
    Qos = get_basis(Gos)

    p = leastsquares_trigpoly(Ifit, Qos, M, b; verbose=verbose)
    A = p(Is.Iblend)
    projection!(A, Qder, Rder, R)
    FCGramBlendZero(Is.Iblend, A)
end

"""
    projection!(A::AbstractMatrix, Q::AbstractMatrix)

Overwrtite `A` with `AQᵀ`
"""
function projection!(A::AbstractMatrix, Q::AbstractMatrix)
    nA = size(A, 2)
    nQ = size(Q, 2)
    msg = "ncols(A) = $nA ≠ $nQ = ncols(Q)."
    nA == nQ || return throw(DimensionMismatch(msg))
    for r in eachrow(A) # compute AQᵗ
        r .= Q * r
    end
end

"""
     projection!(A::AbstractMatrix, Qder::AbstractMatrix, Rder::AbstractMatrix, R::AbstractMatrix)

Overwrtite `A` with `AQᵀ`, where `Qᵀ=R⋅Rder⁻¹⋅Qder`
"""
function projection!(A::AbstractMatrix, Qder::AbstractMatrix, Rder::AbstractMatrix, R::AbstractMatrix)
    n = size(A, 2)
    mR = size(R, 1)
    mD = size(Rder, 1)
    msg1 = "nrows(R) = $mR ≠ $mD = nrows(Rder)."
    msg2 = "ncols(A) = $n ≠ $mR = nrows(R)."
    mR == mD || return throw(DimensionMismatch(msg1))
    n == mR || return throw(DimensionMismatch(msg2))
    Qt = R / Rder # 
    for r in eachrow(Qt)
        r .= Qder * r
    end
    for c in eachcol(A)
        c .= Qt * c
    end
end

"""
    get_bandwith(params::FCGramParameters)

Compute the bandwith `M = (d + C + Z + E) / 2 - modes` of the trigonometric polynomial,
where `C`,`Z`, `E` and `modes` are paramaters of `params`.
"""
function get_bandwith(params::FCGramParameters)
    C = params.C
    d = params.d
    E = params.E
    Z = params.Z
    modes = params.modes
    (d + C + Z + E) ÷ 2 - modes
end

"""
    get_period(params::FCGramParameters)

Compute the period `P = (d + C + Z + E - 1)⋅ h` of trigonometric polynomial,
where `d`, `C`, `E`, `Z` and `h` are paramaters of `params`.
"""
function get_period(params::FCGramParameters)
    d = params.d
    C = params.C
    E = params.E
    Z = params.Z
    h = params.h
    b = d + C + Z + E - oneunit(d)
    b * h
end

get_matrix(op::FCGramBlendZero) = op.AQt

get_interval(op::FCGramBlendZero) =  op.Iblend

npts_Iblend(op::FCGramBlendZero) = op |> get_interval |> get_npts