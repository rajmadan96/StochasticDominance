"""
Portfolio optimization under higher order stochastic dominance constraints (from order two); Objective: Minimize Risk measure
created: 2024, September
author©: Rajmadan Lakshmanan
"""

using LinearAlgebra
using ForwardDiff
include("Newton.jl")
include("StocksToAnnualizedReturn.jl")
Random.seed!(123) 


"""
    safe_exponent(base, exponent)

Handles cases of 0^0 by returning 0.0. Otherwise, computes base^exponent.

# Arguments
- `base`: The base of the exponentiation.
- `exponent`: The exponent.

# Returns
- The result of base^exponent or 0.0 if base == 0 and exponent == 0.
"""
function safe_exponent(base, exponent)
    if base == 0 && exponent == 0
        return 0.0
    else
        return base^exponent
    end
end

# ---------------------------
# Gradient Computations
# ---------------------------

"""
    ∇gp_t(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)

Computes the gradient of g_p with respect to `t`.

# Arguments
- `t`: Threshold variable.
- `x`: Portfolio weights.
- `ξ`, `ξ_0`: Portfolio scenarios and benchmark.
- `p`: Power parameter.
- `p_ξ`, `p_ξ_0`: Probability weights.

# Returns
- The gradient of g_p with respect to t.
"""
function ∇gp_t(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    term1 = (safe_exponent.(max.(t .- x' * ξ, 0), p-1) * p_ξ)
    term2 = (safe_exponent.(max.(t .- ξ_0, 0), p-1)' * p_ξ_0)
    return (p) * (term1[1] - term2[1])
end


"""
    ∇gp_x(t, x, ξ, p, p_ξ)

Computes the gradient of g_p with respect to `x`.

# Arguments
- `t`: Threshold variable.
- `x`: Portfolio weights.
- `ξ`: Portfolio scenarios.
- `p`: Power parameter.
- `p_ξ`: Probability weights.

# Returns
- The gradient of g_p with respect to x.
"""
function ∇gp_x(t, x, ξ, p, p_ξ)
    grad_x = -p * ((safe_exponent.(max.(t .- x' * ξ, 0), p-1) .* ξ) * p_ξ)
    return grad_x 
end

"""
    ∂gp_minus_1_∂t(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)

Computes the gradient of g_{p-1} with respect to `t`.

# Arguments
- `t`: Threshold variable.
- `x`: Portfolio weights.
- `ξ`, `ξ_0`: Portfolio scenarios and benchmark.
- `p`: Power parameter.
- `p_ξ`, `p_ξ_0`: Probability weights.

# Returns
- The gradient of g_{p-1} with respect to t.
"""
function ∂gp_minus_1_∂t(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    term1 = (safe_exponent.(max.(t .- x' * ξ, 0), p-2) * p_ξ)
    term2 = (safe_exponent.(max.(t .- ξ_0, 0), p-2)' * p_ξ_0)
    return (p-1) * (term1[1] - term2[1])
end

"""
    ∇gp_minus_1_x(t, x, ξ, p, p_ξ)

Computes the gradient of g_{p-1} with respect to `x`.

# Arguments
- `t`: Threshold variable.
- `x`: Portfolio weights.
- `ξ`: Portfolio scenarios.
- `p`: Power parameter.
- `p_ξ`: Probability weights.

# Returns
- The gradient of g_{p-1} with respect to x.
"""
function ∇gp_minus_1_x(t, x, ξ, p, p_ξ)
    grad_x = -(p-1) * ((safe_exponent.(max.(t .- x' * ξ, 0), p-2) .* ξ) * p_ξ)
    return grad_x 
end


# ---------------------------
# Lagrangian and Optimization
# ---------------------------

"""
    lagrangian(x, q, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t, β)

Calculates the Lagrangian function for portfolio optimization, combining the objective and constraints into a single expression.

# Arguments
- `x`: Portfolio weights (variables to optimize).
- `q`: Quantile threshold for risk adjustment.
- `λ`: Lagrange multiplier for ensuring the portfolio weights sum to 1.
- `μ`: Lagrange multiplier for stochastic dominance constraints.
- `ν`: Lagrange multipliers for non-negativity of weights.
- `ξ`: Portfolio scenarios (matrix of asset returns).
- `ξ_0`: Benchmark scenarios.
- `p`: Power parameter for risk and dominance constraints.
- `p_ξ`: Probability weights for `ξ`.
- `p_ξ_0`: Probability weights for `ξ_0`.
- `t`: Threshold for stochastic dominance.
- `β`: Risk aversion parameter.

"""
function lagrangian(x, q, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t, β)
    term1 = q + (1 / (1 - β)) * (sum(p_ξ[i] * safe_exponent(max(-dot(x, ξ[:, i]) - q, 0), p) for i in 1:size(ξ, 2)))^(1/p)
    term2 = λ * (1 - sum(x))
    term3 = μ * (sum(p_ξ[i] * safe_exponent(max(t - dot(x, ξ[:, i]), 0), p) for i in 1:size(ξ, 2)) - sum(p_ξ_0[i] * safe_exponent(max(t - ξ_0[i], 0), p) for i in 1:length(ξ_0)))
    term4 = sum(ν[j] * max(-x[j], 0) for j in 1:length(x))
    return term1 + term2 + term3 + term4
end


"""
    grad_lagrangian_x(x, q, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t, β)

Computes the gradient of the Lagrangian function with respect to `x` (portfolio weights).

# Arguments
- `x`: Portfolio weights.
- `q`: Quantile threshold variable.
- `λ`: Lagrange multiplier for the equality constraint.
- `μ`: Lagrange multiplier for the stochastic dominance constraint.
- `ν`: Lagrange multipliers for the inequality constraints.
- `ξ`: Portfolio scenarios matrix.
- `ξ_0`: Benchmark scenario.
- `p`: Power parameter for the constraints.
- `p_ξ`: Probability weights for ξ.
- `p_ξ_0`: Probability weights for ξ_0.
- `t`: Threshold variable.
- `β`: Risk parameter.

# Returns 
- The gradient of the Lagrangian function with respect to `x`.
"""
function grad_lagrangian_x(x, q, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t, β)
    d = length(x)
    ∇gp_t_val = ∇gp_t(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    ∇gp_x_val = ∇gp_x(t, x, ξ, p, p_ξ)
    ∇gp_minus_1_x_val = ∇gp_minus_1_x(t, x, ξ, p, p_ξ)
    ∂gp_minus_1_∂t_val = ∂gp_minus_1_∂t(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    
    ∇t_x = -∇gp_minus_1_x_val ./ ∂gp_minus_1_∂t_val
    ∇overline_g = -∇gp_t_val * ∇t_x + ∇gp_x_val
    
    term1_factor = (1 / (1 - β)) * (safe_exponent.(max.(-x' * ξ .- q, 0), p) * p_ξ).^((1/p) - 1)
    term1 = term1_factor[1] * ((p .* (safe_exponent.(max.(-x' * ξ .- q, 0), p-1) .* (-ξ)) * p_ξ))#
    
    term2 = -λ
    
    term3 = ν .* (x .< 0)  
    
    grad_x = term1 .+ term2 .+ term3 .+ μ * ∇overline_g
    
    return grad_x
end

"""
    grad_lagrangian_q(x, q, ξ, p, p_ξ, β)

Computes the gradient of the Lagrangian function with respect to `q` (quantile threshold variable).

# Arguments
- `x`: Portfolio weights.
- `q`: Quantile threshold variable.
- `ξ`: Portfolio scenarios matrix.
- `p`: Power parameter for the constraint.
- `p_ξ`: Probability weights for ξ.
- `β`: Risk parameter.

# Returns
- The gradient of the Lagrangian function with respect to `q`.
"""
function grad_lagrangian_q(x, q, ξ, p, p_ξ, β)
    term1 = 1 .- (1 / (1 - β)) * (safe_exponent.(max.(-x' * ξ .- q, 0), p) * p_ξ).^((1/p) - 1)[1] *
                (safe_exponent.(max.(-x' * ξ .- q, 0), p-1) * p_ξ)[1]
    return term1[1]
end


"""
    grad_lagrangian_λ(x)

Computes the gradient of the Lagrangian function with respect to λ, the multiplier for the simplex constraint (sum to 1).

# Arguments
- `x`: Portfolio weights.

# Returns
- The gradient of the Lagrangian with respect to λ as `1 - sum(x)`, enforcing the sum-to-one constraint for portfolio weights.
"""
function grad_lagrangian_λ(x)
    result = 1 - sum(x)
    return result
end

"""
    grad_lagrangian_μ(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)

Computes the gradient of the Lagrangian function with respect to μ, the multiplier for the stochastic dominance constraint.

# Arguments
- `t`: Threshold variable.
- `x`: Portfolio weights.
- `ξ`: Portfolio scenarios matrix.
- `ξ_0`: Benchmark scenario.
- `p`: Power parameter for the constraint.
- `p_ξ`: Probability weights for ξ.
- `p_ξ_0`: Probability weights for ξ_0.

# Returns
- The gradient of the Lagrangian with respect to μ as the difference between two terms involving stochastic dominance constraints.
"""
function grad_lagrangian_μ(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    term1 = (safe_exponent.(max.(t .- x' * ξ, 0), p-1) * p_ξ)
    term2 = (safe_exponent.(max.(t .- ξ_0, 0), p-1)' * p_ξ_0)
    return term1[1] - term2[1]
end


"""
    grad_lagrangian_ν(x)

Computes the gradient of the Lagrangian function with respect to ν, the multiplier for the simplex constraints (ensures non-negativity).

# Arguments
- `x`: Portfolio weights.

# Returns
- A vector representing the inequality constraint gradient as `x .* (x .< 0)`.
"""
function grad_lagrangian_ν(x)
    result = x .* (x .< 0)
    return x .* (x .< 0)
end

"""
    g_p_minus_1(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)

Defines the g_{p-1}(x, t)  stochastic dominance constraint.

# Arguments
- `t`: Threshold variable.
- `x`: Portfolio weights.
- `ξ`: Portfolio scenarios matrix.
- `ξ_0`: Benchmark scenario.
- `p`: Power parameter for the constraint.
- `p_ξ`: Probability weights for ξ.
- `p_ξ_0`: Probability weights for ξ_0.

# Returns
- The value of the  g_{p-1}  constraint, which represents the derivative of higher-order stochastic dominance condition.
"""
function g_p_minus_1(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    term1 = (safe_exponent.(max.(t .- x' * ξ, 0), p-1) * p_ξ)
    term2 = (safe_exponent.(max.(t .- ξ_0, 0), p-1)' * p_ξ_0)
    return term1[1] - term2[1]
end


"""
    g_p(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)

Defines the g_p(x, t)  stochastic dominance constraint.

# Arguments
- `t`: Threshold variable.
- `x`: Portfolio weights.
- `ξ`: Portfolio scenarios matrix.
- `ξ_0`: Benchmark scenario.
- `p`: Power parameter for the constraint.
- `p_ξ`: Probability weights for ξ.
- `p_ξ_0`: Probability weights for ξ_0.

# Returns
- The value of the g_p  constraint, which represents a higher-order stochastic dominance condition.
"""
function g_p(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    term1 = (safe_exponent.(max.(t .- x' * ξ, 0), p) * p_ξ)
    term2 = (safe_exponent.(max.(t .- ξ_0, 0), p)' * p_ξ_0)
    return term1[1] - term2[1]
end

"""
    gInd_p(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)

Defines g_p(x, t)  with an indicator function, ensuring the constraint is non-negative.

# Arguments
- `t`: Threshold variable.
- `x`: Portfolio weights.
- `ξ`: Portfolio scenarios matrix.
- `ξ_0`: Benchmark scenario.
- `p`: Power parameter for the constraint.
- `p_ξ`: Probability weights for ξ.
- `p_ξ_0`: Probability weights for ξ_0.

# Returns
- The value of g_p , with 0 returned if the result is negative.
"""
function gInd_p(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    term1 = (safe_exponent.(max.(t .- x' * ξ, 0), p) * p_ξ)
    term2 = (safe_exponent.(max.(t .- ξ_0, 0), p)' * p_ξ_0)
    result = term1[1] - term2[1]
    
    # Return 0 if result is negative or zero, else return result
    return result <= 0 ? 0 : result
end

"""
    test_g_p(t_range, x_opt, ξ, ξ_0, p, p_ξ, p_ξ_0)

Tests whether g_p(x, t)  is satisfied over a given range of `t` values.

# Arguments
- `t_range`: Range of threshold values to test.
- `x_opt`: Optimal portfolio weights.
- `ξ`: Portfolio scenarios matrix.
- `ξ_0`: Benchmark scenario.
- `p`: Power parameter for the constraint.
- `p_ξ`: Probability weights for ξ.
- `p_ξ_0`: Probability weights for ξ_0.

# Returns
- `tilde_t_1`: The value of `t` where g_p  is maximal (if violated).
- `has_positive_gp`: Boolean indicating whether g_p  is violated (True if any value > 0).
"""
function test_g_p(t_range, x_opt, ξ, ξ_0, p, p_ξ, p_ξ_0)
    # Compute g_p for each t in t_range
    g_p_values = [g_p(t, x_opt, ξ, ξ_0, p, p_ξ, p_ξ_0) for t in t_range]
    
    # Check if all values are non-positive
    if any(g_p_val -> g_p_val > 0, g_p_values)
        # Find the index where g_p is maximal (i.e., argmax)
        tilde_t_1 = t_range[argmax(g_p_values)]
        return tilde_t_1, true
    else
        return nothing, false
    end
end

"""
    RiskFunction(x, q, ξ, p, p_ξ, β)

Calculates the risk-adjusted portfolio return based on quantile-based risk measures.

# Arguments
- `x`: Portfolio weights (decision variables to optimize).
- `q`: Quantile threshold for the risk measure.
- `ξ`: Portfolio scenarios (matrix of asset returns).
- `p`: Power parameter for the risk adjustment.
- `p_ξ`: Probability weights for the scenarios in `ξ`.
- `β`: Risk aversion parameter (closer to 1 indicates higher aversion).

# Returns
- The risk-adjusted return, which accounts for portfolio performance and risk under extreme loss scenarios.
"""
function RiskFunction(x, q, ξ, p, p_ξ, β)
    return q + (1 / (1 - β)) * (sum(p_ξ[i] * safe_exponent(max(-dot(x, ξ[:, i]) - q, 0), p) for i in 1:size(ξ, 2)))^(1/p)
end


"""
    optimize_lagrangian_newton(x, q, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t; γ=0.1, ε=1e-7, max_iter=200)

Optimizes the Lagrangian function using Newton's method, iteratively adjusting constraints to satisfy stochastic dominance.

# Arguments
- `x`: Initial portfolio weights.
- `q`: Initial quantile threshold variable.
- `λ`: Initial Lagrange multiplier for the equality constraint.
- `μ`: Initial Lagrange multiplier for the stochastic dominance constraint.
- `ν`: Initial Lagrange multipliers for non-negativity constraints.
- `ξ`: Portfolio scenarios (matrix of asset returns).
- `ξ_0`: Benchmark scenarios.
- `p`: Power parameter for the constraints.
- `p_ξ`: Probability weights for `ξ`.
- `p_ξ_0`: Probability weights for `ξ_0`.
- `t`: Initial threshold for stochastic dominance.
- `γ`: Step size for optimization (default: 0.1).
- `ε`: Convergence tolerance (default: 1e-7).
- `max_iter`: Maximum number of iterations (default: 200).

# Returns
- `x_opt`: Optimal portfolio weights.
- `q_opt`: Optimal quantile threshold.
- `λ_opt`: Optimal multiplier for the equality constraint.
- `μ_opt`: Optimal multiplier for stochastic dominance constraints.
- `ν_opt`: Optimal multipliers for inequality constraints.
- `t_opt`: Optimal threshold for stochastic dominance.
"""
function optimize_lagrangian_newton(x,q, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t; γ=0.1, ε=1e-7, max_iter=200)
    length_x = length(x)
    constraints = []  # List to store the constraints from each iteration

    # Define the combined function without passing tilde_t_1 explicitly
    function combined_fun(vars)
        x, q, λ, μ, ν, t = unpack_vars(vars)  # Pass `d` to unpack_vars
        base_grad = vcat(
            grad_lagrangian_x(x, q, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t, β),
            grad_lagrangian_q(x, q, ξ, p, p_ξ, β),
            grad_lagrangian_λ(x),
            grad_lagrangian_μ(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0),
            grad_lagrangian_ν(x),
            g_p(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0),
            g_p_minus_1(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
        )
        # Add all constraints from previous iterations
        for tilde_t in constraints
            gInd_constraint = gInd_p(tilde_t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
            base_grad = vcat(base_grad, gInd_constraint)
        end

        return base_grad
    end
    function unpack_vars(vars)
        x = vars[1:length_x]                # Extract the first d elements for x
        q = vars[length_x + 1]              # Next element is q
        λ = vars[length_x + 2]              # Next element is λ
        μ = vars[length_x + 3]              # Next element is μ
        ν = vars[(length_x + 4):(length_x + 3 + d)]# d elements for ν
        t = vars[end]                # Last element is t
        return x, q, λ, μ, ν, t
    end
  
    vars0 = vcat(x, q, λ, μ, ν, t)

    # Define the function to compute the Jacobian using ForwardDiff
    function combined_jacobian(vars)
        return ForwardDiff.jacobian(combined_fun, vars)
    end

    # Main optimization loop
    vars_opt = vars0
    tilde_t_1 = nothing
    has_positive_gp = true  # Initialize the loop condition

    iteration = 0  # Track number of iterations

    while has_positive_gp
        iteration += 1
        println("Running iteration $iteration")

        # Perform the optimization using Newton's method
        @show result = Newton(combined_fun, combined_jacobian, vars_opt; maxEval=max_iter, εAccuracy=ε)

        x_opt, q_opt, λ_opt, μ_opt, ν_opt, t_opt = unpack_vars(result.xMin)

        # Define t_range and test for non-positive g_p values #[minimum(vec(x_opt' * ξ)),maximum(vec(x_opt' * ξ))]#
        t_range = minimum(vec(ξ_0)):0.001:maximum(vec(ξ_0))
        tilde_t_1, has_positive_gp = test_g_p(t_range, x_opt, ξ, ξ_0, p, p_ξ, p_ξ_0)

        if has_positive_gp
            println("Positive g_p values found at tilde_t_1 = $tilde_t_1. Adding constraint and rerunning optimization.")
            # Add the new constraint to the list of constraints
            push!(constraints, tilde_t_1)
        else
            println("No positive g_p values found. Optimization complete.")
            return  x_opt, q_opt, λ_opt, μ_opt, ν_opt, t_opt
        end

        # Update vars_opt with the current optimal values for the next iteration
        vars_opt = vcat(x_opt, q_opt, λ_opt, μ_opt, ν_opt, t_opt)
    end

    return  x_opt, q_opt, λ_opt, μ_opt, ν_opt, t_opt
end



# Define Portfolio matrix ξ ∈ R^{d×n} d assets and n scenarios
ξ = rand(5,15) # replace randome value with interested portfolio
d, n = size(ξ) 

# Define Benchmark 
τ = (ones(d)/d)'  # equally weights
ξ_0 = vec((ones(d)/d)' *ξ)


#parameter
p = 2.0  # Power parameter
β = 0.5

# Probability vectors for ξ and ξ_0
p_ξ = (ones(n)/n) 
p_ξ_0 = (ones(n)/n) 

# Initial guesses for x, λ, μ, and ν
x = rand(d)#
x /= sum(x)  # Ensure x lies in the simplex
q = rand(1) #Initial guess for q
λ = 0.0 #Initial guess for λ
μ = 0.0 # Initial guess for μ
ν = zeros(d) # Initial guess for ν
t = minimum(ξ_0)  # Initial guess for t


# # Run the optimization
@time x_opt, q_opt, λ_opt, μ_opt, ν_opt, t_opt = optimize_lagrangian_newton(x, q, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t)
println("Optimal x: ", x_opt)
println("Optimal q: ", q_opt)
println("Optimal λ: ", λ_opt)
println("Optimal μ: ", μ_opt)
println("Optimal ν: ", ν_opt)
println("Optimal t: ", t_opt)

 