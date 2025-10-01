### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 4e68de32-9565-11f0-3060-d7f0568ee26d
begin # load packages
	import Printf: @sprintf
	import LinearAlgebra: norm
	import DelimitedFiles: writedlm
	import PlotlyLight as PL
	import FourierContinuation: FCGramParameters, FCDerivative, FCGram
	import PlutoUI: NumberField, Slider, Select, CheckBox, TextField, TableOfContents
end

# ╔═╡ 84abb2c2-16b7-4cd0-a7c3-1b29ff83af77
TableOfContents()

# ╔═╡ 657174ac-27c7-424c-b178-84f25bf8de4e
md"""
# Fourier Continuation Notebook
*A periodic extension framework for high-order PDE solvers*

## Introduction

Given the values $f_i$ of a smooth function $f:[0,1]\to\mathbb R$
on a uniform partition $G$ of $[0,1]$, generate a trigonometric polynomial 
$f^c$ with period $b>1$ that interpolates $f$ on $G$.

The idea is to generate a blending operator $B$ that preserves
first $d_\ell$ values, ${\bf f}_\ell$, and last $d_r$ values, ${\bf f}_r$. 
The values of the Fourier Continuation are obtained by

$${\bf f}^c=\begin{pmatrix} {\bf f} \\ B{\bf f} \end{pmatrix}$$

The operator $B$ is decomposed as the sum of two blend-to-zero operators 

$$B = B_\ell + B_r,$$

where $B_\ell$ preserves ${\bf f}_\ell$ and makes zero ${\bf f}_r$, while $B_r$
makes the opposite.
"""

# ╔═╡ 9841f2b7-94a9-44e8-93f9-efded9c3c72a
md"""
## Accelerated Fourier Continuation FC(Gram)

The rightward operator $B_r$ projects the function $f$ on a basis of Gram polynomials $q_j$ on $I_\text{match}$ which are blended by trigonometric interpolants $p_j$ (via least squares) that vanish at $I_\text{zero}$, where

-  $I_\text{match}$ is the set of the last $d_r$ points of $G$,
-  $I_\text{zero}$ is the set of the first $d_\ell$ points of $G$ translated by $b$,
-  $I_\text{blend}$ is the extended grid $G$ on $[1,b]$.

The operator $B_r$ is expressed in matrix form as

$$B_r{\bf f}=PQ^T{\bf f_r},$$
where

$$P=
\begin{bmatrix}
	p_1(I_\text{blend}) & \cdots & p_{d_r}(I_\text{blend})
\end{bmatrix}$$

$$\begin{matrix}
	\begin{matrix}
		\text{QR Factorization of} \\
		\text{Vandermonde Matrix on } I_\text{match}
	\end{matrix} & 
	\longrightarrow &
	\text{values of }q_j \\
	& & \\
	\begin{matrix}
		\text{Singular Value Decomposition of }\\
		\text{Sinusoidal Basis Matrix on } I_\text{match}\cup I_\text{zero} 
	\end{matrix}
	& \longrightarrow &
	\text{coefficients of }p_j
\end{matrix}$$

**High-precision arithmetic and oversampling are required to mitigate ill-conditioning.**

- The leftward operator can be obtained as
$$B_\ell{\bf f}= \text{reverse}(PQ^T \text{reverse}({\bf f}_\ell)).$$

- The matrix $PQ^T$ is independent of the step size and the function, so it can be saved for other computations.
"""

# ╔═╡ 83e79e9b-3ce0-4faf-813d-040cf66b9f81
md"""
## Higher-Order Derivative Approximation

The higher order derivatives of a family of non-periodic smooth functions are approximated on a uniform grid of $N$ points by the FFT-based differentiation of the Fourier Continuation:

$${\bf f}^{(k)} \approx {\bf f}^{c(k)} =
\text{IFFT}\left( \left( \dfrac{2\pi i}{b}\cdot \text{freq}({\bf f}^c) \right)^k  \text{filter}(\text{FFT}({\bf f}^c)) 
\right)$$
"""

# ╔═╡ fbf35e98-7127-4a43-b1ae-c51027edb432
md"""
**Function**

$$f(x)= A\cdot \cos(\pi(k_1\cdot x -\omega_1))\cdot\sin(\pi(k_2\cdot x - \omega_2))$$
"""

# ╔═╡ 71a51662-2343-4399-9ded-06d3e7f20812
begin
	sgn(k::Integer) = iseven(k) ? one(k) : -one(k)
	
	function sincos(x; order=0, A=0.5, k₁=2.5, k₂=6.5, ω₁=420, ω₂=-ω₁)
    	Ak = sgn(order ÷ 2) * pi^order * A/2
    	kp = k₁+k₂
    	km = k₁-k₂
    	dp = kp^order
    	dm = km^order

    	θp(x) = kp*x - ω₁ - ω₂
    	θm(x) = km*x - ω₁ + ω₂
    	sp = sinpi ∘ θp
    	sm = sinpi ∘ θm
    	cp = cospi ∘ θp
    	cm = cospi ∘ θm

    	df = iseven(order) ? dp*sp(x) - dm*sm(x) : dp*cp(x) - dm*cm(x)
    	Ak*df
	end
end

# ╔═╡ 68dc0b20-6db4-41c5-8d2a-f376bb8ee53f
md"""
| | |
|:---|---:|
| Save Continuation Matrix $( @bind save_matrix CheckBox(default=false) ) | name: $( @bind namem TextField(default="FCmatrix") ) .csv |
"""

# ╔═╡ 9ebd5639-be02-4648-b184-061d51d6ddc1
begin # routines for plot

function draw_plot(figs...; layout)
    plt = PL.Config.(fig for fig in figs) |> PL.Plot # merge figures
    plt.layout = PL.Config(layout) # set layout
    return plt
end

function draw_points(x, y; color, label, symbol="circle", width=2)
    Dict(:x => x,
        :y => y,
        :type => "scatter",
        :mode => "markers",
        :marker => Dict(:width => 1, :color => "white", 
						:opacity => 1.0, :symbol => symbol,
					    :line => Dict(:width => width, :color => color)),
        :text => map(string, eachindex(x)),
        :hoverinfo => "text",
        :name => label)
end

function draw_pointlines(x, y, Dl, Dm, label)
	 Dict(:x => x,
        :y => y,
        :type => "scatter",
        :mode => "markers+lines",
        :marker => Dm,
        :line => Dl,
		:hoverinfo => "text",
        :name => label)
end

function ptlines(x, y; color, label, symbol, width=2)
	Dml = Dict(:width => width, :color => color)
	Dm = Dict(:width => 1, :opacity => 1.0, :color => "white",  
			  :symbol => symbol, :line => Dml)
	Dl = Dict(:width => width, :color => color)
	draw_pointlines(x, y, Dl, Dm, label)
end

function ptlines_gradient(x, y; color, cgrad, label, symbol, rev=false, width=2)
	idx = eachindex(x) |> collect
	rev && reverse!(idx)
	Dml = Dict(:width => width, :colorscale => cgrad, :color => idx)
	Dm = Dict(:width => 1, :opacity => 1.0, :color => "white",  
			  :symbol => symbol, :line => Dml)
	Dl = Dict(:width => width, :color => color)
	draw_pointlines(x, y, Dl, Dm, label)
end

function draw_lines(x, y; color, label, width=3)
    Dict(:x => x,
        :y => y,
        :type => "scatter",
        :mode => "lines",
        :line => Dict(:width => width, :color => color),
        :name => label,
        :hoverinfo => "name")
end

function BottomLegend(titlename)
    Dl = Dict(
        :orientation => "h",
        :font => Dict(:size => 15))
	Df = Dict(:family => "Times New Roman", :size => 20)
	Dt = Dict(:text => titlename, :side => "top center", :font => Df)
    Dict(:legend => Dl, :title => Dt)
end

function plot_comparison(Gos, G, Gb, dfos, dfc1, dfcb, order)
	msg = string(order, "-th order derivative approximation by Fourier Continuation")
    plt1 = ptlines(G, dfc1, symbol="circle", color="#5017B3", label="fᶜ⁽ᵏ⁾ [0,1]")
	plt2 = draw_lines(Gos, dfos, color="#176DB3", label="f⁽ᵏ⁾")
	plt3 = ptlines(Gb, dfcb, symbol="square", color="#5017B3", label="fᶜ⁽ᵏ⁾ [1,b]")
	draw_plot(plt1, plt2, plt3, layout=BottomLegend(msg))
end

function plot_fcblending(grid, Gtrans, Gblend, Dl, Dr, fungrid, fl, fr, fcr, fcl; label)

	fc = fcr + fcl
	plt1 = ptlines(grid, fungrid, symbol="circle", color="#176DB3", label="f(x)")
	plt2 = ptlines(Gtrans, fungrid, symbol="circle", color="#176DB3", label="f(x-b)")
	plt3 = ptlines_gradient(Gblend, fcr, symbol="square", color="blue", 
							cgrad=:Blues, label="Bᵣf")
	plt4 = ptlines_gradient(Gblend, fcl, symbol="diamond", color="blue",
							cgrad=:Blues, label="Bℓf", rev=true)
	plt5 = ptlines(Gblend, fc, symbol="circle", color="#003399", label="Bf", width=3)
	plt6 = ptlines(Dl, fl, symbol="diamond", color="#4B0082", label="fℓ", width=3)
	plt7 = ptlines(Dr, fr, symbol="square", color="#4B0082", label="fr", width=3)
	
	draw_plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, layout=BottomLegend(label))
end

end

# ╔═╡ c9fac3b4-a179-4340-96c1-d8a389bf4779
begin # filters
	fcfilter(x::Number, p::Integer, α::Real) = exp(-α * x^(2p))

	filter_zero(x::Real, N::Integer) = fcfilter(x, 3N ÷ 5, 16log(10))

	function filter_zero!(c::AbstractVector)
    	M = length(c)
		for i in eachindex(c)
			c[i] *= filter_zero( (i-1)/M, 2M)
		end
	end

	filter_decimate(x::Real) = fcfilter(x, 4, -log(0.01) / 5)

	function filter_decimate!(c::AbstractVector)
    	M = length(c)
		for i in eachindex(c)
			c[i] *= filter_decimate( (i-1)/M)
		end
	end
end

# ╔═╡ b5b6d869-3eaf-49fe-9d26-7e4740405e26
function fcGramOp(C::Integer, d::Integer)
	params = FCGramParameters{BigInt,BigFloat}(h=1.0, C=C, dl=d, dr=d)
    FCGram(params)
end

# ╔═╡ d014ff3c-56e8-410e-90a3-47bf1770be06
function fcblendig(fun, N0, C0, d0; label)
	b = (N0+C0)/(N0-1) # period
	# FC operators
	B = fcGramOp(C0, d0)
	Br = map(Float64, B.rightOp.AQt)
	# grids
	grid = range(start=0, stop=1, length=N0)
	Dr = grid[N0-d0+1:N0]
	Dl = grid[1:d0] .+ b
	Gtrans = grid .+ b
	Gblend = range(start=grid[N0-d0+1], step=step(grid), length=C0+2d0)
	# function values
	fungrid = map(fun, grid)
	fl = fungrid[1:d0]
	fr = fungrid[N0-d0+1:N0]
	fun_duplicate = repeat(fungrid,2)
	# FC values
	fcr = vcat(fr, Br*fr, zeros(d0))
	fcl = vcat(zeros(d0), reverse(Br*reverse(fl)), fl)
	# plot FC blending
	plot_fcblending(grid, Gtrans, Gblend, Dl, Dr, fungrid, fl, fr, fcr, fcl; label=label)
end

# ╔═╡ 3acc65c9-0507-4831-856d-5ad827a78ab7
begin # example of FC blending
	N0 = 100; C0 = 25; d0 = 5 # FC parameters
	fun(x) = sin(5.4π*x-2.7π) - cos(2π*x) |> exp # function
	fcblendig(fun, N0, C0, d0; label="Fourier Continuation by blending")
end

# ╔═╡ 6329fb28-6974-416b-8d23-7d94b67db733
begin # range of function parameters
	rangeA = 0.1:0.1:10
	rangek₁ = 0.1:0.1:10
	rangek₂ = 0.1:0.1:10
	rangeω₁ = -1:0.05:1
	rangeω₂ = -1:0.05:1
	# range of FC parameters
	ranged = 5:15
	rangeo = 0:4
	rangeC = [25,35]
	rangeN = [50, 100, 200, 500, 1000, 2000]
	filterlabel = [ identity => "none", 
					filter_zero! => "trunctated ends" , 
					filter_decimate! => "smooth ends" ]
end;

# ╔═╡ 824dc0de-f722-4dec-82df-9efa1aef962c
md"""

**Shape Parameters**

| | |
|:---|---:|
| A  | $(@bind A Slider(rangeA, default=0.5, show_value=true)) |
| k₁ | $(@bind k₁ Slider(rangek₁, default=2.5, show_value=true)) |
| k₂ | $(@bind k₂ Slider(rangek₂, default=6.5, show_value=true)) |
| ω₁ | $(@bind ω₁ Slider(rangeω₁, default=0.05, show_value=true)) |
| ω₂ | $(@bind ω₂ Slider(rangeω₂, default=-0.05, show_value=true)) |

**Fourier Continuation Parameters**

| | |
|:---|---:|
| Polynomial degree d | $(@bind d NumberField(ranged)) |
| Derivative order | $(@bind order NumberField(rangeo)) |
| Filter σ| $(@bind σ Select(filterlabel)) |
| Number of Grid points N | $(@bind N Slider(rangeN, default=rangeN[2], show_value=true)) |
| Number of blending points C | $(@bind C Select(rangeC)) |
"""

# ╔═╡ 4e0c661a-9903-49ac-afde-ec9337bb4e2c
begin
	@info "Computation of the FC Operator"
	@time Op = fcGramOp(C, d)
	@info @sprintf "Period = %1.4e" (N+C)/(N-1)
end;

# ╔═╡ e36fe870-cce9-4a8c-b0a3-b651e1631945
if save_matrix
	
	opmatrix = map(Float64, Op.rightOp.AQt) 
	writedlm( string(namem,".cvs"), opmatrix)
end

# ╔═╡ 95e2c2b6-3e79-4589-9324-0642fee87232
begin # function and higher-order derivatives
	f(x) = sincos(x; order=0, A=A, k₁=k₁, k₂=k₂, ω₁=ω₁, ω₂=ω₂)
	df(x) = sincos(x; order=order, A=A, k₁=k₁, k₂=k₂, ω₁=ω₁, ω₂=ω₂)
end;

# ╔═╡ 53ad4494-9bd7-4446-a12d-1b212587871d
begin # grids
	G = range(start=0, stop=1, length=N) # uniform grid on the unit interval
	Gos = range(start=0, stop=1, length=10N)
	Gb = range(start=1, step=step(G), length=C+2) # extended grid
    fG = map(f, G) # function values on the uniform grid
	dfG = map(df, G)
	dfos = map(df, Gos)
end;

# ╔═╡ 7ab5074d-06bb-41fc-ba7f-bb458a431c0b
D = FCDerivative(Float64, Op, order, σ=σ);

# ╔═╡ 6d6f540b-4256-4488-a7c9-8a657da2d56b
begin
	@info "Spectral differentiation"
	@time dfc = D(fG)
	dfc1 = @view dfc[1:N]
	dfcb = vcat( dfc[N:end], first(dfc))
	errfc = norm(dfG - dfc1, Inf)/norm(dfG, Inf)
	@info @sprintf "‖f⁽ᵏ⁾ - fᶜ⁽ᵏ⁾‖∞/‖f⁽ᵏ⁾‖∞ = %1.3e " errfc
end;

# ╔═╡ 90ff5b00-be19-4786-b1b9-86f7f5fe9089
plot_comparison(Gos, G, Gb, dfos, dfc1, dfcb, order)

# ╔═╡ a69ddcea-2c22-4ccd-8be9-bf4c5c5f0e22
md"""
## References

1. F. Amlani & O. P. Bruno (2016) [An FC-based spectral solver for elastodynamic problems in general three-dimensional domains](https://doi.org/10.1016/j.jcp.2015.11.060)

2. O. P. Bruno et al (2007) [Accurate, high-order representation of complex three-dimensional surfaces via Fourier continuation analysis](https://doi.org/10.1016/j.jcp.2007.08.029)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
FourierContinuation = "7a7d29dc-0a27-42a9-9145-d1eba3778da3"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlotlyLight = "ca7969ec-10b3-423e-8d99-40f33abb42bf"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
FourierContinuation = "~0.1.0"
PlotlyLight = "~0.11.1"
PlutoUI = "~0.7.66"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "e61334ac59a522db231e3c3465ab6b0d9a1d82e3"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractNFFTs]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "292e21e99dedb8621c15f185b8fdb4260bb3c429"
uuid = "7f219486-4aa7-41d6-80a7-e08ef20ceed7"
version = "0.8.2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BasicInterpolators]]
deps = ["LinearAlgebra", "Memoize", "Random"]
git-tree-sha1 = "3f7be532673fc4a22825e7884e9e0e876236b12a"
uuid = "26cce99e-4866-4b6d-ab74-862489e035e0"
version = "0.7.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "06ee8d1aa558d2833aa799f6f0b31b30cada405f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.2"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Cobweb]]
deps = ["DefaultApplication", "OrderedCollections", "Scratch"]
git-tree-sha1 = "6665ec6b16446379fb76ad58a2a7b65687c77271"
uuid = "ec354790-cf28-43e8-bb59-b484409b7bad"
version = "0.7.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "3a3dfb30697e96a440e4149c8c51bf32f818c0f3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.17.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefaultApplication]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "c0dfa5a35710a193d83f03124356eef3386688fc"
uuid = "3f0dd361-4fe0-5fc6-8523-80b14ec94d85"
version = "1.1.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EasyConfig]]
deps = ["JSON3", "OrderedCollections", "StructTypes"]
git-tree-sha1 = "11fa8ecd53631b01a2af60e16795f8b4731eb391"
uuid = "acab07b0-f158-46d4-8913-50acef6d41fe"
version = "0.1.16"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "797762812ed063b9b94f6cc7742bc8883bb5e69e"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.9.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FourierContinuation]]
deps = ["FourierTools", "GenericLinearAlgebra", "LinearAlgebra", "PlotlyLight", "Printf", "Test"]
git-tree-sha1 = "1bf512b3059c8ee71392d87a5245720bb5e7412f"
uuid = "7a7d29dc-0a27-42a9-9145-d1eba3778da3"
version = "0.1.0"

[[deps.FourierTools]]
deps = ["ChainRulesCore", "FFTW", "IndexFunArrays", "LinearAlgebra", "NDTools", "NFFT", "PaddedViews", "Reexport", "ShiftedArrays"]
git-tree-sha1 = "acbe6d6c9ef39cce526ec3fb31db2650beada18b"
uuid = "b18b359b-aebc-45ac-a139-9c0ccbb2871e"
version = "0.4.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random", "libblastrampoline_jll"]
git-tree-sha1 = "ad599869948d79efd63a030c970e2c6e21fecf4a"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.3.17"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IndexFunArrays]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f78703c7a4ba06299cddd8694799c91de0157ac"
uuid = "613c443e-d742-454e-bfc6-1d7f8dd76566"
version = "0.2.7"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NDTools]]
deps = ["LinearAlgebra", "OffsetArrays", "PaddedViews", "Random", "Statistics"]
git-tree-sha1 = "3e5105ea7d08354014613c96bdfeaa0d151f1c1a"
uuid = "98581153-e998-4eef-8d0d-5ec2c052313d"
version = "0.7.1"

[[deps.NFFT]]
deps = ["AbstractNFFTs", "BasicInterpolators", "Distributed", "FFTW", "FLoops", "LinearAlgebra", "PrecompileTools", "Printf", "Random", "Reexport", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "2dfd2514a10b49ee99e06ceee6515d192f7d11be"
uuid = "efe261a4-0d2b-5849-be55-fc731d526b0d"
version = "0.13.7"

    [deps.NFFT.extensions]
    NFFTGPUArraysExt = ["Adapt", "GPUArrays"]

    [deps.NFFT.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

    [deps.OffsetArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyLight]]
deps = ["Artifacts", "Cobweb", "Dates", "Downloads", "EasyConfig", "JSON3", "REPL", "Random"]
git-tree-sha1 = "bbebe9360e0a88f4e53b2c75576192574a6b8b26"
uuid = "ca7969ec-10b3-423e-8d99-40f33abb42bf"
version = "0.11.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "2b2127e64c1221b8204afe4eb71662b641f33b82"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.66"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─4e68de32-9565-11f0-3060-d7f0568ee26d
# ╟─84abb2c2-16b7-4cd0-a7c3-1b29ff83af77
# ╟─657174ac-27c7-424c-b178-84f25bf8de4e
# ╟─3acc65c9-0507-4831-856d-5ad827a78ab7
# ╟─d014ff3c-56e8-410e-90a3-47bf1770be06
# ╟─9841f2b7-94a9-44e8-93f9-efded9c3c72a
# ╟─83e79e9b-3ce0-4faf-813d-040cf66b9f81
# ╟─fbf35e98-7127-4a43-b1ae-c51027edb432
# ╟─71a51662-2343-4399-9ded-06d3e7f20812
# ╟─824dc0de-f722-4dec-82df-9efa1aef962c
# ╟─90ff5b00-be19-4786-b1b9-86f7f5fe9089
# ╟─4e0c661a-9903-49ac-afde-ec9337bb4e2c
# ╟─68dc0b20-6db4-41c5-8d2a-f376bb8ee53f
# ╟─e36fe870-cce9-4a8c-b0a3-b651e1631945
# ╟─6d6f540b-4256-4488-a7c9-8a657da2d56b
# ╟─9ebd5639-be02-4648-b184-061d51d6ddc1
# ╟─c9fac3b4-a179-4340-96c1-d8a389bf4779
# ╟─b5b6d869-3eaf-49fe-9d26-7e4740405e26
# ╟─6329fb28-6974-416b-8d23-7d94b67db733
# ╟─95e2c2b6-3e79-4589-9324-0642fee87232
# ╟─53ad4494-9bd7-4446-a12d-1b212587871d
# ╟─7ab5074d-06bb-41fc-ba7f-bb458a431c0b
# ╟─a69ddcea-2c22-4ccd-8be9-bf4c5c5f0e22
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
