    ###########################
    # Specify Model Parameters
    ###########################
    # A    = 11         # Number of Periods
    γ_1  = 0.300        # Leisure Coefficient
    γ_2  = 0.500        # Consumption-Leisure Interaction Coefficient
    δ    = 0.1          # Discount Rate
    N    = 35000        # Number of Individuals
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