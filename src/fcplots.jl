export fcplot

"""
    fcplot(D::FCDerivative, f, df; savefig=true, titlename)

Draw the function derivative `df` and is Fourier continuation given by the derivative operator 
`D` using `PlotlyLight`. The plot is saved by default as a html file. 
"""
function fcplot(D::FCDerivative, f, df; savefig=true, titlename="Derivative Comparison")
    k = D.k # derivative order
    N = npts_Iunit(D) # npts in [0,1]
    Iunit = grid_unit(N) # uniform partition in [0,1]    
    dfvalues = map(df, Iunit) # derivative values of the function  
    dfcvalues = map(f, Iunit) |> D  # derivative values of the continuation
    ## figures
    fig1 = draw_pointlines(Iunit, dfcvalues[1:N], color="green", label="$k-th derivative of fᶜ")
    fig2 = draw_points(Iunit, dfvalues, color="blue", label="$k-th derivative of f")
    plt = draw_plot(fig1, fig2, layout=BottomLegend(titlename))
    savefig && PL.save("testfc_derivative.html", plt)
    plt
end

"""
    fcplot(D::FCDerivative, f; savefig=true, titlename)

Plot the function `f` on `[0,1]∪[b,b+1]` and its Fourier continuation `fc` on the blendinf interval `[1,b]`.
"""
function fcplot(Op::FCGram, f; savefig=true, titlename="Periodic Continuation")
    N = Op.N # npts in [0,1]
    C = Op.C # npts in Iblend
    b = get_period(Op) # period 
    ## grids
    Iunit = grid_unit(N) # partition of [0,1]
    Iext = vcat(Iunit, Iunit .+ b) # partition of [0,1]∪[b,b+1]
    Iblend = range(start=oneunit(b), stop=b, length=C + 2) # partition of [1,b]
    ## evaluation
    fvalues = map(f, Iunit) # function values
    fcvalues = Op(fvalues) # continuation values
    # ploting values
    fext = repeat(fvalues, 2)
    fcblend = zeros(eltype(fcvalues), C + 2)
    fcblend[1] = last(fvalues)
    fcblend[end] = first(fvalues)
    fcblend[2:end-1] .= fcvalues
    ## figures
    fig1 = draw_pointlines(Iblend, fcblend, color="red", label="continuation fᶜ")
    fig2 = draw_points(Iext, fext, color="blue", label="function f")
    plt = draw_plot(fig1, fig2, layout=BottomLegend(titlename))
    savefig && PL.save("testfc_function.html", plt)
    plt
end

"""
    draw_plot(figs...)

Draw the figures stored in the dictionary of traces `figs` using the same axis.
The legend is shown on bottom of the plot.
"""
function draw_plot(figs...; layout)
    plt = PL.Config.(fig for fig in figs) |> PL.Plot # merge figures in the same plot
    plt.layout = PL.Config(layout) # set layout
    return plt
end

function draw_points(x, y; color, label, width=2)
    Dict(:x => x,
        :y => y,
        :type => "scatter",
        :mode => "markers",
        :marker => Dict(:width => width, :color => color),
        :text => map(string, eachindex(x)),
        :hoverinfo => "text",
        :name => label)
end

function draw_pointlines(x, y; color, label, width=2)
    Dict(:x => x,
        :y => y,
        :type => "scatter",
        :mode => "markers+lines",
        :marker => Dict(:width => width, :color => color),
        :line => Dict(:width => width, :color => color),
        :name => label)
end

function draw_lines(x, y; color, label, width=2)
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
        :title => Dict(:text => titlename, :side => "top center"),
        :font => Dict(:size => 18))
    Dict(:legend => Dl)
end