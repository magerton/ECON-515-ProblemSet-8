

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


function probit_wrapper(θ::Array{Float64})

	probit_LL(θ)
end

function probit_LL(θ::Vector{Float64})

	P = DATA[symbol("P_$(AAA)")][DATA[:A].== AAA]

	g_over_sig = normcdf( probit_input(θ))
	out = P.*log( 1 - g_over_sig ) + (1-P).*log(g_over_sig)

	# clean output
	out[isnan(out).==1] = - 1e50
	out = - sum( out )

	countPlus!(out)
	return out
end

function probit_input(θ::Array{Float64})
	p_vec = unpackparams(θ)
  	γ_1 = p_vec["γ_1"]
  	γ_2 = p_vec["γ_2"]
  	α_1 = p_vec["α_1"]
  	α_2 = p_vec["α_2"]
  	α_3 = p_vec["α_3"]
  	σ_e = p_vec["σ_e"]

  	Y_a = DATA[:Y][DATA[:A].== AAA]
  	X_a = DATA[symbol("X_$(AAA)")][DATA[:A].==AAA]


	term = [ones(N) Y_a]*[γ_1; γ_2 ]
	term[term .<= 0] = NaN
	
	g_over_sig = (log(term) - [ones(N) X_a X_a.^2]*[α_1; α_2;α_3])./σ_e

	return g_over_sig
end






























function printCounter(count)
	if count <= 5
		denom = 1
	elseif count <= 50
		denom = 10
	elseif count <= 200
		denom = 25
	elseif count <= 500
		denom = 50
	elseif count <= 2000
		denom = 100
	else
		denom = 500
	end
	mod(count, denom) == 0 
end


function countPlus!()
  global count += 1
  if printCounter(count) 
    println("Eval $(count)")
  end
end


function countPlus!(out::Float64)
  global count += 1
  if printCounter(count) 
    println("Eval $(count): value = $(round(out,5))")
  end
    return count
end