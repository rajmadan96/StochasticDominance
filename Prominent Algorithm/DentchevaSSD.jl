"""
	Second order stochastic algorithm and dataset from OPTIMIZATION WITH STOCHASTIC DOMINANCE
    CONSTRAINTS by D DENTCHEVA and A RUSZCZYŃSKI  2003
	created: 2024, November
	author©: Rajmadan Lakshmanan
"""

# Import necessary packages
using GLPK
using JuMP
using DataFrames

# Define the safe_exponent function to handle 0^0 cases
function safe_exponent(base, exponent)
    if base == 0 && exponent == 0
        return 0.0
    else
        return base^exponent
    end
end

# DARINKA DENTCHEVA and  ANDRZEJ RUSZCZYŃSKI dataset
returns = [
    7.5 -5.8 -14.8 -18.5 -30.2 2.3 -14.9 67.7;
    8.4 2 -26.5 -28.4 -33.8 0.2 -23.2 72.2;
    6.1 5.6 37.1 38.5 31.8 12.3 35.4 -24;
    5.2 17.5 23.6 26.6 28 15.6 2.5 -4;
    5.5 0.2 -7.4 -2.6 9.3 3 18.1 20;
    7.7 -1.8 6.4 9.3 14.6 1.2 32.6 29.5;
    10.9 -2.2 18.4 25.6 30.7 2.3 4.8 21.2;
    12.7 -5.3 32.3 33.7 36.7 3.1 22.6 29.6;
    15.6 0.3 -5.1 -3.7 -1 7.3 -2.3 -31.2;
    11.7 46.5 21.5 18.7 21.3 31.1 -1.9 8.4;
    9.2 -1.5 22.4 23.5 21.7 8 23.7 -12.8;
    10.3 15.9 6.1 3 -9.7 15 7.4 -17.5;
    8 36.6 31.6 32.6 33.3 21.3 56.2 0.6;
    6.3 30.9 18.6 16.1 8.6 15.6 69.4 21.6;
    6.1 -7.5 5.2 2.3 -4.1 2.3 24.6 24.4;
    7.1 8.6 16.5 17.9 16.5 7.6 28.3 -13.9;
    8.7 21.2 31.6 29.2 20.4 14.2 10.5 -2.3;
    8 5.4 -3.2 -6.2 -17 8.3 -23.4 -7.8;
    5.7 19.3 30.4 34.2 59.4 16.1 12.1 -4.2;
    3.6 7.9 7.6 9 17.4 7.6 -12.2 -7.4;
    3.1 21.7 10 11.3 16.2 11 32.6 14.6;
    4.5 -11.1 1.2 -0.1 -3.2 -3.5 7.8 -1;
]
# Step 1: Create the data (as provided)
data = DataFrame(
    Year = 1:22,
    Asset1 = returns[:,1],
    Asset2 = returns[:,2],
    Asset3 = returns[:,3],
    Asset4 = returns[:,4],
    Asset5 = returns[:,5],
    Asset6 = returns[:,6],
    Asset7 = returns[:,7],
    Asset8 = returns[:,8]
)

# Step 2: Extract the returns matrix ξ (22 scenarios by 9 assets)
ξ = (Matrix(select(data, Not(:Year))))'

# Step 3: Define the number of scenarios and assets
d, n = size(ξ)  # d = 8 assets, n = 22 scenarios, 

# Step 4: Define portfolio τ weights (equal weighting across assets for simplicity)
τ = fill(1/d, d)  # This assumes equal weights across all 8 assets

p_ξ = fill(1/n, n)  # Equal probability for each scenarios (can be replaced with actual probability)
p_ξ0 = fill(1/n, n)  # Equal probability for each scenarios (can be replaced with actual probability)

ξ0 = vec(τ'ξ)  # Returns of reference portfolio over m periods 
v =  [p_ξ0'*(max.(ξ0s .- ξ0, 0)) for ξ0s in ξ0] # E(\xis-\xi) forall \xis

# Model definition
model = Model(GLPK.Optimizer)

# Decision Variables
@variable(model, x[1:d] >= 0)  # Portfolio weights with small positive bound
@variable(model, s[1:n, 1:n] >= 0)  # Shortfall variables

# Objective: Maximize expected return
@objective(model, Max, sum(p_ξ[j] * sum(ξ[i, j] * x[i] for i in 1:d) for j in 1:n)) 

# Constraint 1: Portfolio weights sum to 1 (fully invested)
@constraint(model, sum(x[i] for i in 1:d) == 1)

# Constraint 2: Ensure second-order stochastic dominance (Shortfall constraint)
@constraint(model, [j=1:n, k=1:n], sum(ξ[i, k] * x[i] for i in 1:d) + s[j,k] >= ξ0[j])

# Constraint 3: Limit the expected shortfall for the portfolio
@constraint(model, [i=1:n], sum(p_ξ[k] * s[i,k] for k in 1:n) <= v[i])

# Solve the problem
@time optimize!(model)

# Print results
println("Optimal portfolio weights: ", value.(x))
println("Optimal expected return: ", objective_value(model))
