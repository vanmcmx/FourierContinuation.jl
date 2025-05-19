export FCParameters, FCGramParameters

abstract type FCParameters{T} end

"""
    FCGramParameters(d, C, Z, E, no)

Parameters for accelarated FC Gram with default values.

| parameter | default | description |
|---:|:---:|:---|
| N | 50 | number of points of the uniform partition on `[0,1]` |
| dl | 5 | number of leftward matching points |
| dr | 5 | number of rigfhtward subgrid matching points |
| C | 25 | number of blend-to-zero points |
| E | 25 | number of extra zeros |
| Z | 12 | number of match-to-zero points |
| no | 20 | oversampling factor |
| modes | 3 | number of modes to reduce |
"""
@kwdef mutable struct FCGramParameters{T<:Integer, R<:Real} <: FCParameters{T}
    N::T = 50
    C::T = 25
    dl::T = 5
    dr::T = 5
    d::T = dr
    E::T = 25
    Z::T = 12
    no::T = 20
    modes::T = 3
    h::R = inv(N)
end
