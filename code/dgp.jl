
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

A_max = 25
time_taken = zeros(A_max)

for A in 1:A_max

    tic()

    ###########################
    # Specify Model Parameters
    ###########################
    # A    = 11         # Number of Periods
    γ_1  = 0.300        # Leisure Coefficient
    γ_2  = 0.500        # Consumption-Leisure Interaction Coefficient
    δ    = 0.1          # Discount Rate
    N    = 10000        # Number of Individuals
    σ_e  = 1.000        # Standard Error of Wage Shock
    σ_v  = 0.100        # Standard Error of Measurement Error
    α_1  = 5.000        # Wage Function Parameter
    α_2  = 0.500        # Wage Function Parameter
    α_3  = -0.100       # wage function parameter
    μ_y  = 0.0          # Mean of non-labor income
    σ_y  = 2.0          # std dev of non-labor income
    # yL = 0          # Minimum Non-Labor Income
    # yH = 1000          # Top Non-Labor Income

    θ_real = [γ_1; γ_2; α_1; α_2; α_3; σ_e]
    β_real = [α_1; α_2; α_3; σ_e]
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

    ##################################
    ######## Estimation
    ##################################


    θ = θ_real

    ntheta = length(θ_real)
    nbeta  = length(β_real)

    θ_MLE = zeros(A,ntheta)
    β_ols = zeros(A,nbeta)
    Σ_w       = zeros(A)
    Σ_ols     = zeros(nbeta,nbeta,A)


    for tt in A+1:-1:1
        DATA[symbol("EV_$(tt)_x")] = 0.0  
        DATA[symbol("EV_$(tt)_x1")] = 0.0  
    end


    # map g function
    k = kde(probit_input(θ_real))

    fig2 = figure
    fig2 = plot(k)

    fig2 = title("Kernel density of Work Probit Input")
    savefig("./plots/KdenX.jpg")


    k_y = kde(log(df[:Y][df[:A].== 1]))
    fig3 = 0
    fig3 = figure
    fig3 = plot(k_y)
    fig3 = title("Kernel density of Log(Non-Labor Income)")
    savefig("./plots/Yden_normal.jpg")


    for tt in A:-1:1  # for every period, starting at A and working backward

        initials = ones(ntheta)
        initials = θ 

        probit_opt = []
        global count = 1
        for i =1:5
          probit_opt = optimize(probit_wrapper,vec(initials),autodiff = true,
              ftol=1e-12,
              iterations = 3000)
          initials = probit_opt.minimum 
        end
        show(probit_opt)

        θ_MLE[tt,:] = probit_opt.minimum



        # Second stage

        θ_hat = probit_opt.minimum

        W_a = DATA[symbol("W_$(tt)")][DATA[:A].==tt]
        X_a = DATA[symbol("X_$(tt)")][DATA[:A].==tt]
        P_a = DATA[symbol("P_$(tt)")][DATA[:A].==tt]

        W_a = W_a[P_a .== true]
        X_a = X_a[P_a.== true]

        Y_mat = log(W_a) 

        X_mat = [ones(int(sum(P_a))) X_a X_a.^2 λ(θ_hat)[P_a.==true]]

        if det(X_mat'*X_mat) > 0.0
            (β_ols[tt,:], Σ_w[tt], Σ_ols[:,:,tt]) = least_sq(X_mat,Y_mat)
        else
            X_mat = [ones(int(sum(P_a))) λ(θ_hat)[P_a.==true]]       
            (test, ~, ~) = least_sq(X_mat,Y_mat)
            β_ols[tt,:] = [test[1] NaN NaN test[2]]
        end


        # Calculate EV_a_x and EV_a_x1 (current period)
        ## assumes value for next period already exists
        EV_hat(θ_hat,tt)

    end

    time_taken[A] = toc()

    θ_names = ["γ_1" "γ_2" "α_1" "α_2" "α_3" "σ_e"] 
    β_names = ["α_1" "α_2" "α_3" "σ_e"] 

    for tt in 1:A
        println("There where $N workings and $A periods")
        println("It took $(round(time_taken[A],3)) seconds to run" )

        println("\n Percentage that worked in period $tt:")
        println("$(round(perc_a,2)) \n")
        println("\n LLN value at θ true: $(round(probit_LL(θ_MLE[tt,:][:]),3)) \n \n")

        println("\n MLE Parameters: \n \t TRUE \t ESTIMATED")
        for ii in 1:ntheta
            println("")
            println("$(θ_names[ii]) \t $(round(θ_real[ii],3)) \t $(round(θ_MLE[tt,ii],3))")
        end

        println("\n OLS Parameters: \n \t TRUE \t ESTIMATED")
        for ii in 1:nbeta
            println("")
            println("$(β_names[ii]) \t $(round(β_real[ii],3)) \t $(round(β_ols[tt,ii],3))")
        end

    end

end # end of loop over A


for ii in 1:N
    temp = 1:A
    println("NumPeriods: Time Taken:")
    println("$(temp[ii]) \t\t $(time_taken[ii])")
end



fig5 = figure
fig5 = plot([1:11],time_taken)
fig5 = title("Time Taken to Run Model, N=$(N)")
fig5 = xlabel("Number of Periods")
fig5 = ylabel("Seconds")
savefig("./plots/time_taken.jpg")
