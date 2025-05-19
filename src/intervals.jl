export FCInterval, FCGramIntervals
export get_grid, get_npts, merge_partition, oversample_partition
export grid_unit, grid_blend

struct FCInterval{V<:AbstractVector}
    grid::V

    function FCInterval(grid::V) where {V<:AbstractVector}
        issorted(grid) || throw(ArgumentError("unsorted vector"))
        new{V}(grid)
    end
end

get_grid(J::FCInterval) = J.grid

get_npts(J::FCInterval) = J |> get_grid |> length

firstpoint(J::FCInterval) = J |> get_grid |> first

lastpoint(J::FCInterval) = J |> get_grid |> last

get_step(J::FCInterval{<:AbstractRange}) = J |> get_grid |> step

merge_partition(I1::FCInterval, I2::FCInterval) = mapreduce(get_grid, union, (I1, I2)) |> sort |> FCInterval

function oversample_partition(J::FCInterval{<:AbstractRange}, no::Integer)
    u = oneunit(no)
    n = get_npts(J)
    h = get_step(J) 
    h /= no
    xstart = firstpoint(J)
    npts = (n - u) * no + u
    range(start=xstart, step=h, length=npts) |> FCInterval
end


"""
    FCGramIntervals(params::fcFCGramParameters)
    FCGramIntervals(; N::T, d::T, C::T, Z::T, no::T) where {T<:Integer}

Type for the intervals required by the FC Gram method, it has the following properties 

| Property | Description |
|---:|:---|
| `D` | Uniform grid on the match-to-function interval `[0, (d-1)h]` |
| `Im` | Oversampled grid on the match-to-function interval `[0, (d-1)h]` |
| `Iz` | Oversampled grid on the match-to-zero interval `[(d+C)h, (d+C+Z-1)h]` |
| `Ib` | Uniform grid on the blend-to-zero interval `[dh, (d+C-1)h]` |

where `h=1/(N-1)`.
"""
struct FCGramIntervals{T<:FCInterval}
    Imatch::T
    Iblend::T
    Izero::T

    function FCGramIntervals(Imatch::T, Iblend::T, Izero::T) where {T<:FCInterval{<:AbstractRange}}
        msg = "The FC intervals have different stepsizes"
        get_step(Imatch) ≈ get_step(Iblend) ≈ get_step(Izero) || throw(ArgumentError(msg))
        new{T}(Imatch, Iblend, Izero)
    end
end

function FCGramIntervals(params::FCGramParameters) 
    Imatch = interval_match(params)
    Iblend = interval_blend(params)
    Izero = interval_zero(params)
    FCGramIntervals(Imatch,Iblend,Izero)
end

function fit_interval(Is::FCGramIntervals, no::Integer)
    oversample(J::FCInterval) = oversample_partition(J, no)
    mapreduce(oversample, merge_partition, (Is.Imatch, Is.Izero))
end

"""
   interval_match(params::FCGramParameters)

Discretization of the matching interval for the FC Gram operator using a uniform partition 
`grid` with `(d-1)no+1` points that starts at `x=0` with stepsize `1/(N⋅no)`.
"""
function interval_match(params::FCGramParameters)
    d = params.d
    h = params.h
    x0 = zero(h)
    range(start=x0, step=h, length=d) |> FCInterval
end

"""
    interval_blend(; N::T, d::T, C::T) where {T<:Integer}

Discretization of the blending interval for the FC Gram operator using an uniform partition 
of `C` points with stepsize `h=1/N` that starts at `x = d⋅h`.
"""
interval_blend(params::FCGramParameters) = grid_blend(params.d, params.C, params.h) |> FCInterval

"""
    interval_zero(params::FCGramParameters)

Discretization of the match-to-zero interval for the FC Gram operator using an uniform partition 
of `(Z-1)no+1` points with stepsize `h/no` that starts at `x=(d+C)h`, where `h=1/N`.
"""
function interval_zero(params::FCGramParameters)
    C = params.C
    d = params.d
    h = params.h
    Z = params.Z
    xstart = (C + d) * h
    range(start=xstart, step=h, length=Z) |> FCInterval
end

grid_unit(N::Integer) = range(start=0.0, stop=1.0, length=N)

function grid_blend(d::Integer, C::Integer, h::Real)
    x0 = d * h
    range(start=x0, step=h, length=C)
end
