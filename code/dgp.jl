

# Set Working Directory
cd("C:/Users/nick/skydrive/projects/laborecon/ps8")
# Call Packages
using DataFrames
using Optim
using Distributions
using PyPlot
pwd()

###########################
# Specify Model Parameters
###########################
A   = 3          # Number of Periods
γ_1 = 0.0        # Leisure Coefficient
γ_2 = 1.0        # Consumption-Leisure Interaction Coefficient
δ   = 0.1        # Discount Rate
N   = 4          # Number of Individuals
σ_e = 3          # Standard Error of Wage Shock
σ_v = 0.3        # Standard Error of Measurement Error
f1  = 1          # Wage Function Parameter
f2  = .5         # Wage Function Parameter
μ_y = 0          # Mean of non-labor income
σ_y = 2          # std dev of non-labor income
# yL  = 0          # Minimum Non-Labor Income
# yH  = 3          # Top Non-Labor Income

srand(12345)

################
# Start Dataset
################

Y_nl = e.^(rand(Normal(μ_y,σ_y^2),N))
df = DataFrame(
    ID = squeeze(kron([1:N],int(ones(A,1))),2), # ID
    A  = repmat([1:A],N),                       # Indicate Period
    Y  = squeeze(kron(Y_nl,ones(A,1)),2),       # Non-Labor Income
    e  = rand(Normal(0,σ_e),N*A),               # Wage Shock
    v  = rand(Normal(0,σ_v),N*A)                # Measurement Error
    )
# head(df)


df = DataFrame(
    ID = AAA[:,1],
    A  = AAA[:,2],
    Y  = AAA[:,6],
    e  = AAA[:,3],
    v  = AAA[:,4]
    )

for tt in 1:A+1
    for x in 0:A+1
        df[symbol("EV_$(tt)_x$(x)")] = 0.0   # what are the $$'s? can i get some? is that a different pkg?
        df[symbol("V_$(tt)_x$(x)")]  = 0.0
    end
end
head(df)


for tt in A:-1:1
    for x in 0:tt
        println("$tt and $x")


        p_a_x      = symbol("p$(tt)_x$(x)")
        V_a_x_no   = symbol("V_$(tt)_x$(x)_no")
        V_a_x_yes  = symbol("V_$(tt)_x$(x)_yes")
        V_a_x      = symbol("V_$(tt)_x$(x)")
        EV_a_x     = symbol("EV_$(tt)_x$(x)")
        EV_a1_x    = symbol("EV_$(tt+1)_x$(x)")
        EV_a1_x1   = symbol("EV_$(tt+1)_x$(x+1)")

        w_x  = symbol("w_x$(x)")
        Ew_x = symbol("Ew_x$(x)")


        # Add Wages to Dataset
        df[w_x] = exp(f1+f2*x+df[:e])          # True Wage

        # ##################################################
        # Don't we need to correct for selection???
        # ##################################################
        df[Ew_x] = exp(f1+f2*x)                # Expected True Wage
           
        df[V_a_x_no] = γ_1+(1+γ_2).*df[:Y] + df[EV_a1_x]
        df[V_a_x_yes] = df[w_x]+df[:Y] + df[EV_a1_x1]
        
        #######################################################
        # Need to use selection corrected wages here. 
        # also weight by probability of working/not working.
        # Right now, agents aren't rational.
        #######################################################
        df[EV_a_x] = df[Ew_x]+df[:Y] + df[EV_a1_x1]      
        

        df[p_a_x] = df[V_a_x_no] .< df[V_a_x_yes]

        (df[V_a_x])[df[p_a_x].==false] = (df[V_a_x_no])[df[p_a_x].==false]
        (df[V_a_x])[df[p_a_x].==true] = (df[V_a_x_yes])[df[p_a_x].==true]

        # df[symbol("EV_$(tt)_x$(x)")] = max(df[symbol("V_$(tt)_x$(x)_no")],df[symbol("EV_$(tt)_x$(x)_yes")])

        
        # observed wage
        df[symbol("ow_x$(x)")] = exp(f1+f2*x+df[:e]+df[:v])  # Wage with Measurement Error

        # delete!(df,V_a_x_no)
        # delete!(df,V_a_x_yes)
        # delete!(df,EV_a_x)
    end
end

#  get rid of Value of A+1 since always 0.0

for x in 0:A
    delete!(df,symbol("EV_$(A+1)_x$(x)"))
    delete!(df,symbol("V_$(A+1)_x$(x)"))
end


head(df)



# ################################
# # Clean Dataset for Regressions
# ################################
# # Delete Error Terms
# delete!(df,:e)
# delete!(df,:v)
# # Delete True Wages and Values
# for x in 0:2
#     delete!(df,symbol("wT$x"))
#     delete!(df,symbol("V$x"))
# end
# # Delete State Contingent Values and Expected State Contingent Values
# for x in 0:1
#     delete!(df,symbol("fV0x$x"))
#     delete!(df,symbol("fV1x$x"))
#     delete!(df,symbol("V0x$x"))
#     delete!(df,symbol("V1x$x"))
# end
# # Censor Wages
# df[:W] = 0.0
# for x = 0:2
#     for i in 1:N*A
#         if x == df[i,:Xs]
#             if df[i,:X] == 1
#                 df[i,:W] = df[i,symbol("wF$x")]
#             end
#         end
#     end
#     delete!(df,symbol("wF$x"))
# end
# df

