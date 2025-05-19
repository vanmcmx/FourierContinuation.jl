module FC

import Printf: @sprintf
import FourierTools: rfft, irfft
import Base: size, getindex
import LinearAlgebra: Bidiagonal, norm, dot, rdiv!, SVD, UpperTriangular, QRIteration
import GenericLinearAlgebra: svd
import PlotlyLight as PL

include("fcoperators.jl")
include("bconditions.jl")
include("parameters.jl")
include("utils.jl")
include("intervals.jl")
include("vandermonde.jl")
include("csmatrix.jl")
include("least_squares.jl")
include("trigpoly.jl")
include("gram_schmidt.jl")
include("gram_basis.jl")
include("blendzero.jl")
include("fcgram.jl")
include("derivatives.jl")
include("fcplots.jl")

end # module FC
