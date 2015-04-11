

# Set Working Directory
cd("C:/Users/nick/skydrive/projects/laborecon/ps8")
# Call Packages
using DataFrames
using Optim
using Distributions
using PyPlot
pwd()

include("code/functions.jl")
include("code/functions_probit.jl")

###########################
# Specify Model Parameters
###########################
A    = 4          # Number of Periods
γ_1  = 2.0        # Leisure Coefficient
γ_2  = 1.0        # Consumption-Leisure Interaction Coefficient
δ    = 0.1        # Discount Rate
N    = 4          # Number of Individuals
σ_e  = 10         # Standard Error of Wage Shock
σ_v  = 1000       # Standard Error of Measurement Error
α_1  = 1          # Wage Function Parameter
α_2  = 5         # Wage Function Parameter
μ_y  = 0          # Mean of non-labor income
σ_y  = 1          # std dev of non-labor income
# yL = 0          # Minimum Non-Labor Income
# yH = 3          # Top Non-Labor Income

srand(12345)

β = 1/(1+δ)

################
# Start Dataset
################

Y_nl = exp(rand(Normal(μ_y,σ_y^2),N))
df = DataFrame(
    ID = squeeze(kron([1:N],int(ones(A,1))),2), # ID
    A  = repmat([1:A],N),                       # Indicate Period
    Y  = squeeze(kron(Y_nl,ones(A,1)),2),       # Non-Labor Income
    e  = rand(Normal(0,σ_e),N*A),               # Wage Shock
    v  = rand(Normal(0,σ_v),N*A)                # Measurement Error
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

tt = A
x = 1

for tt in A:-1:1
    for x in 0:tt
        println("$tt and $x")

        # Convenience
        X_a = int(x.*ones(N)) # give everyone X = x at a
        E_a = (df[:e])[df[:A].==tt] # errors this period
        y = df[:Y][df[:A].== tt] # N-vec of non-labor income at a

        ##### WAGES
        w_a_x  = symbol("w_$(tt)_x$(x)") # wage[ A= a, X = x] 
        w_a    = symbol("w_$(tt)") # observed wage at a    

        #####  VALUES
        # generated before
        EV_a1_x    = symbol("EV_$(tt+1)_x$(x)") # of being at a+1 with x (didn't work)
        EV_a1_x1   = symbol("EV_$(tt+1)_x$(x+1)") # being at a+1 with x+1 (did work)
        # generated here
        V_a_x      = symbol("V_$(tt)_x$(x)") # being at a with x
        EV_a_x     = symbol("EV_$(tt)_x$(x)") # exp of being at a with x
        V_a_x_no   = symbol("V_$(tt)_x$(x)_no") # of working this period
        V_a_x_yes  = symbol("V_$(tt)_x$(x)_yes") # not working this period

        ##### POLICY FUNCITON
        p_a_x      = symbol("p_$(tt)_x$(x)") # actual policy function
        


        # Add Wages to Dataset
        # True Wage
        df[w_a_x][df[:A].==tt] = wage_eqn(X_a,(df[:e])[df[:A].==tt])          


# take out of Dataframe
        # value of not working
        df[V_a_x_no][df[:A].==tt] = leisure_value_t(tt) + (df[EV_a1_x])[df[:A].==tt]
        # value of working       
        df[V_a_x_yes][df[:A].==tt] = y + wage_eqn(X_a,E_a) + df[EV_a1_x1][df[:A].==tt]

        # Choose between and record
        df[p_a_x][df[:A].==tt] = df[V_a_x_no][df[:A].==tt] .< df[V_a_x_yes][df[:A].==tt]
        (df[V_a_x])[df[p_a_x].==false] = (df[V_a_x_no])[df[p_a_x].==false]
        (df[V_a_x])[df[p_a_x].==true] = (df[V_a_x_yes])[df[p_a_x].==true]

        # calculate expected value of being at current period
        # Correct expectations for selection
        # Weight values by probability of occurence (cond'l on e_{ia})
        Π = Π_true(X_a,tt)
        (df[EV_a_x])[df[:A].==tt] = 
            (1-Π).*( leisure_value_t(tt) + β*EV_0 )
            + Π.*( y + β*EV_1 )
            + exp(.5*σ_e^2)*wage_eqn(X_a, zeros(N)).*
            ( 1 - normcdf( (g(X_a,tt) - σ_e^2)/σ_e ) )    
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
        println("$jj,$x")
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
DATA


##################################
######## Estimation
##################################

# want to find 


global count = 1


initials = ones(6)

aa = 1
probit_opt = []
for i =1:5
  probit_opt = optimize(probit_wrapper,vec(initials),autodiff = true,
      ftol=1e-12)
  initials = probit_opt.minimum
end
probit_opt
θ = probit_opt.minimum