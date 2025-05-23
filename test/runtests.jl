using Test
using FourierContinuation

include("vandermonde.jl")
include("csmatrix.jl")

@testset "FC" begin
    f(x) = log(x + oneunit(x))
    df(x) = inv(x + oneunit(x))
    params = FCGramParameters{BigInt,BigFloat}(N=50) # high-precision arithmethic
    Op = FCGram(params) # FC operator with Dirichlet boundary conditions
    D = FCDerivative(Float64, Op, 1) # first derivative approx
    @test derivative_error(D, f, df) < 1e-6 #error(function derivative, continuation derivative)
end
