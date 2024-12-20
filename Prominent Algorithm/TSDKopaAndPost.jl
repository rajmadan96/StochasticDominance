"""
    Portfolio Choice Based on Third-Degree Stochastic Dominance by T Post and M Kopa  2017
	created: 2024, December
	author©: Rajmadan Lakshmanan
"""

using JuMP
using Ipopt
using DataFrames
using AmplNLWriter
using Couenne_jll
using HiGHS

#using MathOptInterface

function construct_reduced_qcp(S_τ, E_τ, ξ_0, ξ, P)
    # Input Parameters
    # S_τ: Vector of semivariances of the benchmark at threshold levels
    # E_τ: Vector of expected shortfalls of the benchmark at threshold levels
    # ξ_0: Vector of discretized thresholds (benchmark)
    # ξ: Matrix of scenario returns
    # P: Vector of probabilities for scenarios

    # Problem Dimensions
    d, n = size(ξ) # d assets, n scenarios

    # Create a JuMP Model
    
    model = Model(Ipopt.Optimizer)
    #model = Model(() -> AmplNLWriter.Optimizer(Couenne_jll.amplexe))
    #set_optimizer_attribute(model, "tol", 1e-6)
    #set_optimizer_attribute(model, "max_iter", 10000)

    # Example input for classification
    x_feasible = find_feasible_weights(d, n, P, ξ, ξ_0)
    T_zero, T_minus, T_plus = classify_thresholds(x_feasible .*ξ, ξ_0)

    # Variables
    @variable(model, x[1:d] >= 0, start = 1/d)  # Portfolio weights

    # Create a dictionary for θ variables for each scenario
    θ = Dict()
    for s in 1:n
        θ[s] = @variable(model, [1:length(T_zero[s])], start = 0.01, lower_bound = 0)
    end

    # Objective: Minimize semivariance
    @objective(model, Min, sum(sum(P[T_zero[s][i]] * θ[s][i]^2 for i in 1:length(T_zero[s])) for s in 1:n))
    #println("Theta:", θ)
    # Reduced Constraints
    for s in 1:n
        # Constraints for T_zero
        for i in 1:length(T_zero[s])
            @constraint(model, -θ[s][i] - sum(ξ[k,  T_zero[s][i]] * x[k] for k = 1:d) <= -ξ_0[s])
        end

        # Semivariance constraints with reduced terms
          @constraint(model,
              (1 + tolerance(S_τ, E_τ, ξ_0, s)) *
              (sum(P[T_zero[s][i]] * θ[s][i]^2 for i in 1:length(T_zero[s])) +
               sum(P[T_plus[s][j]] * (ξ_0[s] - sum(ξ[k, T_plus[s][j]] * x[k] for k = 1:d))^2 for j in 1:length(T_plus[s]))) <= S_τ[s] + 10
          )
        
    end

    # Portfolio feasibility constraint
    @constraint(model, sum(x[i] for i = 1:d) == 1)

    # Mean Dominance Constraint
    @constraint(model, -sum(P[t] * sum(ξ[k, t] * x[k] for k = 1:d) for t = 1:n) <= -(sum(P[t] * ξ_0[t] for t = 1:n))*1.035)

    return model, x, θ
end
#

function classify_thresholds(ξ_num, ξ_0)
    # Input:
    # ξ_num: Matrix of numerical returns (d assets, n scenarios)
    # ξ_0: Vector of benchmark threshold values (size n)
    # Output:
    # T_minus, T_zero, T_plus: Sets of indices as defined mathematically

    d, n = size(ξ_num)  # d assets, n scenarios
    T_minus = Vector{Vector{Int}}(undef, n)
    T_zero = Vector{Vector{Int}}(undef, n)
    T_plus = Vector{Vector{Int}}(undef, n)

    # Compute the sets T_minus, T_zero, and T_plus
    for s in 1:n
        T_minus[s] = [t for t in 1:n if minimum(ξ_num[:, t]) >= ξ_0[s]]  # T_minus
        T_plus[s] = [t for t in 1:n if maximum(ξ_num[:, t]) <= ξ_0[s]]  # T_plus
        T_zero[s] = [t for t in 1:n if minimum(ξ_num[:, t]) < ξ_0[s] < maximum(ξ_num[:, t])]  # T_zero
    end

    # Ensure disjoint sets: remove overlaps in T_zero
    for s in 1:n
        T_zero[s] = setdiff(T_zero[s], union(T_minus[s], T_plus[s]))
    end

    # Return the sets T_minus, T_zero, and T_plus
    return T_minus, T_zero, T_plus
end




function tolerance(S_τ, E_τ, ξ_0, i)
    if i <= 2
        return 0.0
    else
        return max(
            (S_τ[i] / (S_τ[i - 1] + 2 * E_τ[i - 1] * (ξ_0[i] - ξ_0[i - 1]))) - 1,
            0.0
        )
    end
end

function find_feasible_weights(d, n, P, ξ, ξ_0)
    # Inputs:
    # d: Number of assets
    # n: Number of scenarios
    # P: Probabilities for scenarios (vector of size n)
    # ξ: Matrix of asset returns (d x n)
    # ξ_0: Vector of benchmark threshold values (size n)
    # Outputs:
    # Feasible weights x (or error message if no solution exists)

    # Create the optimization model
    model = Model(HiGHS.Optimizer)

    # Decision variables: x[k], the weights for each asset
    @variable(model, x[1:d] >= 0)

    # Constraints
    # 1. Weighted sum over scenarios satisfies the inequality
    @constraint(
        model,
        sum(P[t] * sum(ξ[k, t] * x[k] for k = 1:d) for t = 1:n) ≥
        sum(P[t] * ξ_0[t] for t = 1:n)
    )

    # 2. First scenario constraint
    @constraint(model, sum(x[k] * ξ[k, 1] for k = 1:d) ≥ ξ_0[1])

    # 3. Budget constraint: sum(x) = 1
    @constraint(model, sum(x) == 1)

    # Solve the optimization problem
    optimize!(model)

    return value.(x)
    # Check the result
    # if termination_status(model) == MOI.OPTIMAL
    #     
    # else
    #     error("No feasible solution found. Status: ", termination_status(model))
    # end
end


# Example usage:

# DARINKA DENTCHEVA and ANDRZEJ RUSZCZYŃSKI dataset
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

# Create the data
data = DataFrame(
    Year = 1:22,
    Asset1 = returns[:, 1],
    Asset2 = returns[:, 2],
    Asset3 = returns[:, 3],
    Asset4 = returns[:, 4],
    Asset5 = returns[:, 5],
    Asset6 = returns[:, 6],
    Asset7 = returns[:, 7],
    Asset8 = returns[:, 8]
)

ξ = Matrix(select(data, Not(:Year)))'
d, n = size(ξ)

τ = ones(d) / d
ξ_0 = sort(vec(τ' * ξ))
#ξ_0 = vec(τ' * ξ)
sorted_indices = sortperm(vec(τ' * ξ))
ξ = ξ[:, sorted_indices]
P = ones(n) / n

S_τ = [(P' * (max.((ξ_0[i] .- ξ_0), 0.0)).^2) for i = 1:n]
E_τ = [(P' * max.((ξ_0[i] .- ξ_0), 0.0)) for i = 1:n]

model, x, θ = construct_reduced_qcp(S_τ, E_τ, ξ_0, ξ, P)
optimize!(model)
status = termination_status(model)
println("Solver Status:", status)
println("Optimal Portfolio Weights: ", value.(x))
# if status != MOI.OPTIMAL
#     println("Solver failed to find an optimal solution.")
# end


println("Optimal Objective Value : ", objective_value(model))


