

function leisure_value_t(
	θ::Array{Float64},
	a::Int64)

    p_vec = unpackparams(θ)
    γ_1 = p_vec["γ_1"]
    γ_2 = p_vec["γ_2"]

    # calls df but for :Y and :A same as DATA
  	γ_1 + (1 + γ_2).*(df[:Y])[df[:A] .== a] 
end

function wage_eqn(
	θ::Array{Float64},
	X_a::Union(Array{Int64}, Int64, DataArray),
	e  ::Union(Array{Float64}, Float64, DataArray)
	)

  p_vec = unpackparams(θ)
  α_1 = p_vec["α_1"]
  α_2 = p_vec["α_2"]
  α_3 = p_vec["α_3"]

	exp( α_1 + α_2.*X_a + α_3.*(X_a.^2) + e )
end

function obs_wage_eqn(	
	θ::Array{Float64},
	X_a::Union(Array{Int64}, Int64, DataArray),
	e  ::Union(Array{Float64}, Float64, DataArray),
	v  ::Union(Array{Float64}, Float64, DataArray)
	)

	exp( log(wage_eqn(θ,X_a,e)) + v )
end

function g(
	θ::Array{Float64},
	X_a::Union(Array{Int64}, Int64, DataArray),
	a  ::Int64)

	x        = unique(X_a)
	EV_a1_x  = symbol("EV_$(a+1)_x$(x[1])")
	EV_a1_x1 = symbol("EV_$(a+1)_x$(x[1]+1)")
	EV_1     = (df[EV_a1_x1])[df[:A] .== a] 
	EV_0     = (df[EV_a1_x])[df[:A] .== a] 
	y        = (df[:Y])[df[:A] .== a]

	log( 
		 leisure_value_t(θ,a) 
		 - y + β*(EV_0 - EV_1)  )
    - wage_eqn(θ,X_a,zeros(N)) 
end

function Π_work(
	θ::Array{Float64},
	X_a::Union(Array{Int64}, Int64, DataArray),
	a  ::Int64
	)

  p_vec = unpackparams(θ)
  σ_e = p_vec["σ_e"]
	
	1 - normcdf( g(θ,X_a,a)./σ_e )
end

function E_g_fun(
    θ::Array{Float64},
    X_a::Union(Array{Int64}, DataArray),
    tt::Int64)

    p_vec = unpackparams(θ)
    γ_1   = p_vec["γ_1"]
    γ_2   = p_vec["γ_2"]
    α_1   = p_vec["α_1"]
    α_2   = p_vec["α_2"]
    α_3   = p_vec["α_3"]
    σ_e   = p_vec["σ_e"]

    X_a      = DATA[symbol("X_$(tt)")][DATA[:A].==tt]
    X_a_p1   = X_a + 1
    y        = DATA[:Y][DATA[:A].== tt] # N-vec of non-labor income at a
    # will update entry for current period
    EV_a_x   = symbol("EV_$(tt)_x") # exp of being at a with x
    EV_a_x1  = symbol("EV_$(tt)_x1") # exp of being at a with x+1
    # use next period's in calculation
    EV_a1_x  = symbol("EV_$(tt)_x") # exp of being at a with x
    EV_a1_x1 = symbol("EV_$(tt)_x1") # exp of being at a with x+1
    EV_a1_1  = (DATA[EV_a_x1])[DATA[:A] .== tt] 
    EV_a1_0 = (DATA[EV_a_x])[DATA[:A] .== tt] 
    
    log_term = leisure_value_t(θ,tt) - y + β*(EV_a1_0 - EV_a1_1) 
    log_term[log_term .<= 0] = eps()
    g_fun = log( log_term ) - wage_eqn(θ,X_a,zeros(N))     
    
end


function EV_hat(
    θ::Array{Float64},
    tt::Int64
    )

    p_vec = unpackparams(θ)
    γ_1   = p_vec["γ_1"]
    γ_2   = p_vec["γ_2"]
    α_1   = p_vec["α_1"]
    α_2   = p_vec["α_2"]
    α_3   = p_vec["α_3"]
    σ_e   = p_vec["σ_e"]
    
    X_a      = DATA[symbol("X_$(tt)")][DATA[:A].==tt]
    X_a_p1   = X_a + 1
    y        = DATA[:Y][DATA[:A].== tt] # N-vec of non-labor income at a
    # will update entry for current period
    EV_a_x   = symbol("EV_$(tt)_x") # exp of being at a with x
    EV_a_x1  = symbol("EV_$(tt)_x1") # exp of being at a with x+1
    # use next period's in calculation
    EV_a1_x  = symbol("EV_$(tt)_x") # exp of being at a with x
    EV_a1_x1 = symbol("EV_$(tt)_x1") # exp of being at a with x+1
    EV_a1_1  = (DATA[EV_a_x1])[DATA[:A] .== tt] 
    EV_a1_0 = (DATA[EV_a_x])[DATA[:A] .== tt] 
    

    # EV if not working in a
    Π = 1 - normcdf( E_g_fun(θ,X_a,tt)./σ_e )
    (DATA[EV_a_x])[DATA[:A].==tt] = 
        (1-Π).*( leisure_value_t(θ,tt) 
        + β*DATA[EV_a1_x][DATA[:A].==tt] )
        + Π.*( y + β*DATA[EV_a1_x1][df[:A].==tt] )
        + exp(.5*σ_e^2)*wage_eqn(θ,X_a, zeros(N)).*
        ( 1 - normcdf( (E_g_fun(θ,X_a,tt) - σ_e^2)/σ_e ) )    

    # EV if working in a
    Π = 1 - normcdf( E_g_fun(θ,X_a,tt)./σ_e )
    (DATA[EV_a_x1])[DATA[:A].==tt] = 
        (1-Π).*( leisure_value_t(θ,tt) 
        + β*DATA[EV_a1_x][DATA[:A].==tt] )
        + Π.*( y + β*DATA[EV_a1_x1][df[:A].==tt] )
        + exp(.5*σ_e^2)*wage_eqn(θ,X_a_p1, zeros(N)).*
        ( 1 - normcdf( (E_g_fun(θ,X_a_p1,tt) - σ_e^2)/σ_e ) )    
end


function unpackparams(θ::Array{Float64})
  d = minimum(size(θ))
  θ = squeeze(θ,d)
  γ_1 = θ[1]
  γ_2 = θ[2]
  α_1 = θ[3]
  α_2 = θ[4]
  α_3 = θ[5]
  σ_e = θ[6]

  return [ 
  "γ_1" => γ_1,
  "γ_2" => γ_2,
  "α_1" => α_1,
  "α_2" => α_2,
  "α_3" => α_3,
  "σ_e" => σ_e
  ]
end





function least_sq(
	X::Union(Array,DataArray,Float64),
	Y::Union(Array,DataArray,Float64);
	N=int(size(X,1)), W=1
	)

  l = minimum(size(X))
  A = X'*W*X
  if sum(size(A))== 1
    inv_term = 1./A
  else
    inv_term = A\eye(int(size(X,2)))
  end
  β = inv_term * X'*W*Y
  if l == 1
    sigma_hat = sqrt(sum((1/N).* (Y - (β*X')')'*(Y - (β*X')'  ) ) ) #sum converts to Float64
  else
    sigma_hat = sqrt(sum((1/N).* (Y - (X*β))'*(Y - (X*β)  ) ) ) #sum converts to Float64
  end
  VCV = (sigma_hat).^2 * inv_term * eye(l)
  return β, sigma_hat, VCV
end



function λ(t)
    normpdf( probit_input(t) )./(1-normcdf(probit_input(t)))
end
















