
# TODO
# we could make small functions accept θ
# then we could use leisure_value_t, etc accept
# θ_true in DGP and θ_guess in estimation
# right now can't small functions for estimation

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



function unpackparams(θ::Array{Float64})
  d = minimum(size(θ))
  θ = squeeze(θ,d)
  γ_1 = θ[1]
  γ_2 = θ[2]
  α_1 = θ[3]
  α_2 = θ[4]
  σ_e = θ[5]
  σ_v = θ[6]

  return [ 
  "γ_1" => γ_1,
  "γ_2" => γ_2,
  "α_1" => α_1,
  "α_2" => α_2,
  "σ_e" => σ_e,
  "σ_v" => σ_v ]
end