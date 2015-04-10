


using DataFrames
using Distributions
using NLopt
using Optim


initials = ones(numparams)


probit_opt = []
for i =1:5
  probit_opt = optimize(ML_PROBIT,vec(initials),autodiff = true,
      ftol=1e-12)
  initials = probit_opt.minimum
end
probit_opt
θ = probit_opt.minimum

  probit_opt = optimize(ENDOG_PROBIT,g!,vec(initials),
    method = :cg,ftol = 1e-20,grtol = 1e-12)
  initials = probit_opt.minimum

N = size(A,1)

function fun_val(θ)
	γ_1 = θ[1]
	γ_2 = θ[2]
	f1 = θ[3]
	f2 = θ[4]
	σ_e = θ[5]
	term = [ones(N) A[:,5]]*[γ_1; γ_2]
	term[term .<=0] = NaN
	try 
		value = (log(term) - [ones(N) A[:,6]]*[f1; f2])./σ_e
	catch
		value = squeeze(ones(N,1).*NaN,2)
	end
end


function ML_PROBIT(θ::Vector)
	# llf

	t = fun_val(θ)
	F = normcdf(t)
	# llf
	# F[F.== 0.0] = 1e-50
	# F[1-F.== 0.0] = 1e-50
	# maximum(F)
	# minimum(F)
	# LLF = P.*log(F) + (1-P).*log(1-F)
	# try	
	term1 = log(F)
	term2 = log(1-F)
	# catch
	# 	F[isnan(F) .== 1] = 0
	# 	term1 = log(F)
	# 	term2 = log(1-F)		
	# end
	# maximum(term1 +term2)
	# minimum(term1 + term2)
	P = A[:,3]
	term1[isnan(term1).==1] = -1e50
	term2[isnan(term2).==1] = -1e50
	LLF = P.*term1 + (1-P).*term2
	LLF[isnan(LLF) .== 1] = 0
	log_like = - sum(LLF)

	return log_like[1,1]
end












