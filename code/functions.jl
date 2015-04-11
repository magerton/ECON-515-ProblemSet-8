


## Normal PDF
function normpdf(x::Union(Vector{Float64}, Float64, DataArray) ;mean=0,var=1) # a type-union should work here and keep code cleaner
    out = Distributions.pdf(Distributions.Normal(mean,var), x) 
    out + (out .== 0.0)*eps(1.0) - (out .== 1.0)*eps(1.0) 
end

## Normal CDF
function normcdf(x::Union(Vector{Float64}, Float64, DataArray);mean=0,var=1) 
    out = Distributions.cdf(Distributions.Normal(mean,var), x)
    out + (out .== 0.0)*eps(1.0) - (out .== 1.0)*eps(1.0) 
end

function leisure_value_t(a::Int64)
	γ_1 + (1 + γ_2).*(df[:Y])[df[:A] .== a] 
end

function wage_eqn(
	X_a::Union(Array{Int64}, Int64, DataArray),
	e  ::Union(Array{Float64}, Float64, DataArray)
	)

	exp( α_1 + α_2.*X_a + e )
end

function obs_wage_eqn(	
	X_a::Union(Array{Int64}, Int64, DataArray),
	e  ::Union(Array{Float64}, Float64, DataArray),
	v  ::Union(Array{Float64}, Float64, DataArray)
	)

	exp( log(wage_eqn(X_a,e)) + v )
end


function g(
	X_a::Union(Array{Int64}, Int64, DataArray),
	a  ::Int64)
	
	x        = unique(X_a)
	EV_a1_x  = symbol("EV_$(a+1)_x$(x[1])")
	EV_a1_x1 = symbol("EV_$(a+1)_x$(x[1]+1)")
	EV_1     = (df[EV_a1_x1])[df[:A] .== a] 
	EV_0     = (df[EV_a1_x])[df[:A] .== a] 
	y        = (df[:Y])[df[:A] .== a] 

	log( 
		leisure_value_t(a) 
		- y + β*(EV_0 - EV_1)
		)  - wage_eqn(X_a,zeros(N),a) 
end
	

function Π_true(
	X_a::Union(Array{Int64}, Int64, DataArray),
	a  ::Int64
	)
	
	1 - normcdf( g(X_a,a)./σ_e )
end
