

# Set Working Directory
cd("C:/Users/nick/skydrive/projects/laborecon/ps8")
# Call Packages
using DataFrames
using Optim
using Distributions
using PyPlot
using KernelDensity
pwd()

include("code/functions.jl")
include("code/functions_probit.jl")

###########################
# Specify Model Parameters
###########################
A    = 11         # Number of Periods
γ_1  = 0.30        # Leisure Coefficient
γ_2  = 0.50        # Consumption-Leisure Interaction Coefficient
δ    = 0.1        # Discount Rate
N    = 15000          # Number of Individuals
σ_e  = 1.0        # Standard Error of Wage Shock
σ_v  = 0.1       # Standard Error of Measurement Error
α_1  = 5.0          # Wage Function Parameter
α_2  = 0.5         # Wage Function Parameter
α_3  = -0.1         # wage function parameter
μ_y  = 0.0          # Mean of non-labor income
σ_y  = 2.0          # std dev of non-labor income
# yL = 0          # Minimum Non-Labor Income
# yH = 1000          # Top Non-Labor Income

θ_real = [γ_1; γ_2; α_1; α_2; α_3; σ_e]

# srand(12345)

β = 1/(1+δ)

################
# Start Dataset
################

Y_nl = exp(rand(Normal(μ_y,σ_y^2),N))
# Y_nl = rand(Uniform(yL,yH),N)
df = DataFrame(
    ID = squeeze(kron([1:N],int(ones(A,1))),2), # ID
    A  = repmat([1:A],N),                       # Indicate Period
    Y  = squeeze(kron(Y_nl,ones(A,1)),2),       # Non-Labor Income
    e  = rand(Normal(0,σ_e^2),N*A),               # Wage Shock
    v  = rand(Normal(0,σ_v^2),N*A)                # Measurement Error
    )
head(df)

# generate empty value function
# pre-allocation makes code cleaner later
for tt in A+1:-1:1
    for x in 0:tt
        df[symbol("EV_$(tt)_x$(x)")] = 0.0   # what are the $$'s? can i get some? is that a different pkg?
        df[symbol("V_$(tt)_x$(x)")]  = 0.0
        df[symbol("w_$(tt)_x$(x)")]  = 0.0
        df[symbol("V_$(tt)_x$(x)_no")] = 0.0
        df[symbol("V_$(tt)_x$(x)_yes")] = 0.0
        df[symbol("p_$(tt)_x$(x)")]      = 0.0
    end
end
head(df)

# fill in values
tt = 3
x = 0
for tt in A:-1:1
    for x in 0:tt
        println("$tt and $x")

		# Convenience
		X_a   = int(x.*ones(N)) # give everyone X = x at a
		E_a   = (df[:e])[df[:A].==tt] # errors this period
		y     = df[:Y][df[:A].== tt] # N-vec of non-labor income at a
		
		##### WAGES
		w_a_x = symbol("w_$(tt)_x$(x)") # wage[ A= a, X = x] 
		w_a   = symbol("w_$(tt)") # observed wage at a    
		
		#####  VALUES
		# generated before
		EV_a1_x   = symbol("EV_$(tt+1)_x$(x)") # of being at a+1 with x (didn't work)
		EV_a1_x1  = symbol("EV_$(tt+1)_x$(x+1)") # being at a+1 with x+1 (did work)
		
		# generated here
		V_a_x     = symbol("V_$(tt)_x$(x)") # being at a with x
		EV_a_x    = symbol("EV_$(tt)_x$(x)") # exp of being at a with x
		V_a_x_no  = symbol("V_$(tt)_x$(x)_no") # of working this period
		V_a_x_yes = symbol("V_$(tt)_x$(x)_yes") # not working this period
		
		##### POLICY FUNCITON
		p_a_x     = symbol("p_$(tt)_x$(x)") # actual policy function
		

		
		# Add Wages to Dataset
		# True Wage
		df[w_a_x][df[:A]      .==tt] = wage_eqn(θ_real,X_a,E_a)          
		
		# value of not working
		df[V_a_x_no][df[:A]   .==tt] = leisure_value_t(θ_real, tt) + (df[EV_a1_x])[df[:A].==tt]
		# value of working       
		df[V_a_x_yes][df[:A]  .==tt] = y + wage_eqn(θ_real, X_a,E_a) + df[EV_a1_x1][df[:A].==tt]
		
		# Choose between and record
		df[p_a_x][df[:A]      .==tt] = df[V_a_x_no][df[:A].==tt] .< df[V_a_x_yes][df[:A].==tt]
		(df[V_a_x])[df[p_a_x] .==false] = (df[V_a_x_no])[df[p_a_x].==false]
		(df[V_a_x])[df[p_a_x] .==true] = (df[V_a_x_yes])[df[p_a_x].==true]

        # calculate expected value of being at current period
        # Correct expectations for selection
        # Weight values by probability of occurence (cond'l on e_{ia})
        Π = Π_work(θ_real, X_a,tt)
        (df[EV_a_x])[df[:A].==tt] = 
            (1-Π).*( leisure_value_t(θ_real, tt) 
            + β*df[EV_a1_x][df[:A].==tt] )
            + Π.*( y + β*df[EV_a1_x1][df[:A].==tt] )
            + exp(.5*σ_e^2)*wage_eqn(θ_real,X_a, zeros(N)).*
            ( 1 - normcdf( (g(θ_real,X_a,tt) - σ_e^2)./σ_e ) )    
    end
end
head(df)


DATA = DataFrame(
    ID = squeeze(kron([1:N],int(ones(A,1))),2), # ID
    A  = repmat([1:A],N),                       # Indicate Period
    Y  = squeeze(kron(Y_nl,ones(A,1)),2),       # Non-Labor Income
    )
# generate empty observed data set
for tt in 1:A
    DATA[symbol("P_$(tt)")] = 0.0   # do they work in period a
    DATA[symbol("W_$(tt)")] = NaN # wage in period a
    DATA[symbol("X_$(tt)")] = 0.0  # experience in period a
end
head(DATA)

for jj in 1:A
    
    P_a = symbol("P_$(jj)") # observed decision
    W_a = symbol("W_$(jj)")
    X_a = symbol("X_$(jj)")
    X_a1 = symbol("X_$(jj+1)")

    X_vec = convert(Array{Float64},(DATA[X_a])[df[:A].== jj])
    for x in 0:jj
        # println("$jj,$x")
        tP_a = (df[symbol("p_$(jj)_x$(x)")][ df[:A].==jj ])
        tP = DATA[P_a][df[:A].== jj]
        tP[x.==X_vec] = tP_a[x.==X_vec]
        DATA[P_a][df[:A].== jj] = tP

        WORKED = DATA[P_a][df[:A].== jj]
        tW_a = (df[symbol("w_$(jj)_x$(x)")][ df[:A].==jj ])
        # don't forget measurement error, + v
        tW = DATA[W_a][df[:A].== jj] + df[:v][df[:A].== jj]
        tW[WORKED.==true] = tW_a[WORKED.==true]
        DATA[W_a][df[:A].== jj] = tW
    end 

    if jj < A
        (DATA[X_a1])[df[:A].== jj+1] = X_vec + (DATA[P_a][df[:A].== jj])
    end
end
# DATA


# percentage that work in period A
perc_a = zeros(A)
for a in 1:A
    perc_a[a] = sum( DATA[symbol("P_$(a)")][DATA[:A].== a ]  )/N
end
perc_a

##################################
######## Estimation
##################################

# # map g function
# AAA = 1
# k = kde(probit_input(θ_real))

# fig2 = figure
# fig2 = plot(k)

# fig2 = title("Kernel density of \$g\$")
# savefig("./plots/KdenX.jpg")





θ = θ_real
probit_LL(θ_real)

ntheta = length(θ)
initials = ones(ntheta)
initials = θ 

AAA = A
probit_opt = []
global count = 1
for i =1:5
  probit_opt = optimize(probit_wrapper,vec(initials),autodiff = true,
      ftol=1e-12,
      iterations = 2000)
   initials = probit_opt.minimum
  initials = probit_opt.minimum 
end
show(probit_opt)
println("\n")
println("$(probit_opt.minimum)")
println("\n")
println("$θ_real \n")

println("$perc_a")
# θ = probit_opt.minimum
# θ_star = [ 2.0  1.0 1.0  5.0  10.0 ]

# show([θ'; θ_star])

θ_hat = probit_opt.minimum


for tt in A:A

	W_a = DATA[symbol("W_$(tt)")][DATA[:A].==tt]
	X_a = DATA[symbol("X_$(tt)")][DATA[:A].==tt]
	P_a = DATA[symbol("P_$(tt)")][DATA[:A].==tt]

	W_a = W_a[P_a .== true]
	X_a = X_a[P_a.== true]

	Y_mat = log(W_a) 
	X_mat = [ones(int(sum(P_a))) X_a X_a.^2 λ(θ_hat)[P_a.==true]]

	(α_w_ols, Σ_w, Σ_α) = least_sq(X_mat,Y_mat)

end




# In progress

function EV_hat(X_a,Y_a)

    p_vec = unpackparams(θ)
    γ_1 = p_vec["γ_1"]
    γ_2 = p_vec["γ_2"]
    α_1 = p_vec["α_1"]
    α_2 = p_vec["α_2"]
    α_3 = p_vec["α_3"]
    σ_e = p_vec["σ_e"]

    Π = Π_work(θ_hat, X_a,tt)   
    (df[EV_a_x])[df[:A].==tt] = 
        (1-Π).*( leisure_value_t(θ_hat,tt) 
        + β*df[EV_a1_x][df[:A].==tt] )
        + Π.*( y + β*df[EV_a1_x1][df[:A].==tt] )
        + exp(.5*σ_e^2)*wage_eqn(θ_hat,X_a, zeros(N)).*
        ( 1 - normcdf( (g(θ_hat,X_a,tt) - σ_e^2)/σ_e ) )    



end
