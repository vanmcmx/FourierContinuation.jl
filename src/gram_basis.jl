export GramBasis, FCGramBasis
export GramBasisDirichlet, GramBasisNeumann
export get_basis, get_coefficients, get_dim
export oversample_basis

"""
    GramBasis(V::AbstractMatrix)
    GramBasis(J::FCGramIntervals; r::Int)
"""
struct GramBasis{T1,T2<:AbstractMatrix}
    Q::T1 # Gram-basis matrix
    R::T2 # Gram-coefficient matrix 

    function GramBasis(Q::T1, R::T2) where {T1,T2<:AbstractMatrix}
        n = size(Q, 2)
        m = size(R, 1)
        m == n || throw(DimensionMismatch("ncols Q = $m ≠ $n = nrows R"))
        new{T1,T2}(Q, R)
    end
end

GramBasis(P::AbstractMatrix; verbose) = reduce(GramBasis, rowmod_gramschmidt(P, verbose=verbose))

struct FCGramBasis{TI<:FCInterval,TG<:GramBasis}
    D::TI # discretization of the matching interval
    G::TG # Gram basis

    function FCGramBasis(D::TI, G::TG) where {TI<:FCInterval,TG<:GramBasis}
        m = get_dim(G)
        n = get_npts(D)
        m == n || throw(DimensionMismatch("dim Gram basis = $m ≠ $n = num partition pts"))
        new{TI,TG}(D, G)
    end
end

function GramBasisDirichlet(D::FCInterval; verbose)
    V = D |> get_grid |> Vandermonde
    G = GramBasis(V, verbose=verbose)
    return FCGramBasis(D, G)
end

function GramBasisNeumann(D::FCInterval; verbose)
    grid = D |> get_grid
    d = length(grid)
    V = Vandermonde(grid, rowidx=d)
    G = GramBasis(V, verbose=verbose)
    return FCGramBasis(D, G)
end

get_basis(G::GramBasis) = G.Q

get_basis(P::FCGramBasis) = P.G |> get_basis

get_coefficients(G::GramBasis) = G.R

get_coefficients(P::FCGramBasis) = P.G |> get_coefficients

get_dim(G::GramBasis) = size(G.Q, 1)

get_dim(P::FCGramBasis) = P.G |> get_dim

function oversample_basis(P::FCGramBasis, no::Integer)
    R = get_coefficients(P)
    d = get_dim(P)
    vandermonde_over(x) = Vandermonde(x, d=d)
    Dos = oversample_partition(P.D, no)
    Pos = Dos |> get_grid |> vandermonde_over
    Qos = Pos / R
    Gos = GramBasis(Qos, R)
    FCGramBasis(Dos, Gos)
end