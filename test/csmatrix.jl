@testset "CSMatrix" begin
    n = 5
    M = 3
    P = 5//1
    v = big.(1:n)
    C = CSmatrix(M, P, v)
    T = eltype(v)
    r = one(T):convert(T, M)
    fac = convert(T, 2)/convert(T, P) 
    A = fac*v*r'
    S = hcat(ones(n), cospi.(A), sinpi.(A)) 
    @test C ≈ S
end