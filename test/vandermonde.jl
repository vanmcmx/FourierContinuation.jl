@testset "vandermonde" begin
    n = 6
    d = 5
    v = 1:n
    V = Vandermonde(v, d=5, rowidx=0) 
    @test V == [vi^(j-1) for vi in v, j=1:d]
end