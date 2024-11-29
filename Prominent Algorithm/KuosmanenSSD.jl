"""
	Kuosmanen Second Order Stochastic Dominance
	created: 2024, October
	author©: Rajmadan Lakshmanan
"""

using JuMP
#using GLPK
using DataFrames
using Dates

# Step 1: Create the data (as provided)
data = DataFrame(
    Date = Date.([Date("2024-07-01"), Date("2024-07-02"), Date("2024-07-03"), Date("2024-07-05"), Date("2024-07-08"), Date("2024-07-09"), Date("2024-07-10"), Date("2024-07-11"), Date("2024-07-12"), Date("2024-07-15"), Date("2024-07-16"), Date("2024-07-17"), Date("2024-07-18"), Date("2024-07-19"), Date("2024-07-22"), Date("2024-07-23"), Date("2024-07-24"), Date("2024-07-25"), Date("2024-07-26"), Date("2024-07-29"), Date("2024-07-30"), Date("2024-07-31")]),
    Agric = [-1.01, -2.50, -0.38, -1.11, 0.44, 0.05, -0.34, 4.00, 1.76, 1.53, 2.77, 0.71, -2.24, 0.08, 0.35, 2.55, -2.21, 1.67, 0.17, -0.97, -0.11, 0.24],
    Food = [-0.72, -0.22, 0.27, 0.15, -0.22, -1.49, 0.79, 1.60, 1.04, 0.02, 1.28, 0.69, -1.24, -1.51, 0.29, -0.09, 2.88, -0.21, 1.26, -0.79, 0.79, 1.89],
    Soda = [1.10, -0.86, -0.21, -0.33, -0.64, 1.52, 1.31, 2.53, -0.06, -3.42, 2.59, -1.44, -1.74, -0.23, -0.54, 0.08, -1.61, 1.05, 2.29, -0.48, 0.98, 0.72],
    Beer = [-1.80, -0.09, 0.41, 2.10, 0.59, -2.59, 2.75, 0.73, 0.41, -2.50, 0.40, 1.12, -1.72, -2.67, -0.42, -0.39, -2.51, 0.00, 0.93, -0.48, -1.65, -1.14],
    Smoke = [-0.65, 0.64, 0.41, -1.61, 0.33, -0.26, 1.93, 3.72, 2.17, -0.63, 1.83, 0.94, -1.78, -1.81, -0.25, 1.14, -0.41, 2.21, 2.13, -1.13, 0.07, -1.23]
)

# Step 2: Extract the returns matrix X (22 scenarios by 5 assets)
X = Matrix(select(data, Not(:Date)))

# Step 3: Define the number of scenarios and assets
T, N = size(X)  # T = 22 scenarios, N = 5 assets

# Step 4: Define portfolio τ weights (equal weighting across assets for simplicity)
τ = fill(1/N, N)  # This assumes equal weights across all 5 assets

# Step 5: Define the optimization model
model = Model(GLPK.Optimizer)

# Step 6: Define the decision variable λ for portfolio weights (non-negative, sum to 1)
@variable(model, λ[1:N] >= 0)  # λ must be non-negative
@constraint(model, sum(λ) == 1)  # Sum of λ must equal 1

# Step 7: Define the optimization variable W (T×T double stochastic matrix)
@variable(model, W[1:T, 1:T] >= 0)  # W must have non-negative elements

# Double stochastic constraints for W
# 1. Rows of W must sum to 1
@constraint(model, [i=1:T], sum(W[i, j] for j in 1:T) == 1)

# 2. Columns of W must sum to 1
@constraint(model, [j=1:T], sum(W[i, j] for i in 1:T) == 1)

# Step 8: Add the SSD constraint: Xλ ≥ W * Xτ
@constraint(model, [i=1:T], sum(X[i, j] * λ[j] for j in 1:N) >= sum(W[i, k] * sum(X[k, j] * τ[j] for j in 1:N) for k in 1:T))

# Step 9: Define the objective function (maximize portfolio return)
@objective(model, Max, 1/T*sum(sum(X[i, j] * λ[j] for j in 1:N) for i in 1:T))

# Step 10: Solve the model
optimize!(model)

# Step 11: Extract the solution (optimal portfolio weights and W matrix)
optimal_portfolio = value.(λ)
optimal_W = value.(W)

# Output the results
println("Optimal Portfolio Weights (λ): ", optimal_portfolio)
println("Optimal Double Stochastic Matrix (W):")
for i in 1:T
    println(optimal_W[i, :])
end
# Output the value of the objective function
println("Value of the Objective Function (Maximized Portfolio Return): ", objective_value(model))
