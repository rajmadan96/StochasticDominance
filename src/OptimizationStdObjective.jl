"""
Portfolio optimization under higher order stochastic dominance constraints (from order two); ; Objective: Maximize Expected Return
created: 2024, September
author©: Rajmadan Lakshmanan
"""

using LinearAlgebra
using ForwardDiff
include("Newton.jl")
#include("StocksToAnnualizedReturn.jl")
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
    grad_lagrangian_x(x, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t)

Computes the gradient of the Lagrangian with respect to `x`.

# Arguments
- `x`: Portfolio weights.
- `λ, μ, ν`: Lagrange multipliers.
- `ξ`, `ξ_0`: Portfolio scenarios and benchmark.
- `p`: Power parameter.
- `p_ξ`, `p_ξ_0`: Probability weights.
- `t`: Threshold variable.

# Returns
- The gradient of the Lagrangian with respect to x.
"""
function grad_lagrangian_x(x, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t)
    d = length(x)
    ∇gp_t_val = ∇gp_t(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    ∇gp_x_val = ∇gp_x(t, x, ξ, p, p_ξ)
    ∇gp_minus_1_x_val = ∇gp_minus_1_x(t, x, ξ, p, p_ξ)
    ∂gp_minus_1_∂t_val = ∂gp_minus_1_∂t(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
    
    ∇t_x = -∇gp_minus_1_x_val ./ ∂gp_minus_1_∂t_val
    ∇overline_g = -∇gp_t_val * ∇t_x + ∇gp_x_val
    
    term1 =  vec((p_ξ)'*ξ')
    
    term2 = -λ
    
    term3 = ν .* (x .< 0)  
    
    grad_x = term1 .+ term2 .+ term3 .+ μ * ∇overline_g
    
    return grad_x
end



"""
    lagrangian(x, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t)

Defines the Lagrangian function.

# Arguments
- `x`: Portfolio weights.
- `λ, μ, ν`: Lagrange multipliers.
- `ξ`, `ξ_0`: Portfolio scenarios and benchmark.
- `p`: Power parameter.
- `p_ξ`, `p_ξ_0`: Probability weights.
- `t`: Threshold variable.

# Returns
- The value of the Lagrangian function.
"""
function lagrangian(x,λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t)
    term1 = dot(vec((p_ξ)'*ξ'),x)
    term2 = λ * (1 - sum(x))
    term3 = μ * (sum(p_ξ[i] * safe_exponent(max(t - dot(x, ξ[:, i]), 0), p) for i in 1:size(ξ, 2)) - sum(p_ξ_0[i] * safe_exponent(max(t - ξ_0[i], 0), p) for i in 1:length(ξ_0)))
    term4 = sum(ν[j] * max(-x[j], 0) for j in 1:length(x))
    return term1 + term2 + term3 + term4
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
    result = (1 - sum(x))
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
    return (term1[1] - term2[1]) 
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
- The value of the  g_{p-1}  constraint, which represents a lower-order stochastic dominance condition.
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
    RiskFunction(x, ξ, p_ξ)

Computes the expected portfolio return as Return = E x^Tξ .

# Arguments
- `x`: Portfolio weights.
- `ξ`: Portfolio scenarios matrix.
- `p_ξ`: Probability weights for ξ.

# Returns
- The expected portfolio return.
"""
function RiskFunction(x,ξ,p_ξ)
   return  dot(vec((p_ξ)'*ξ'),x)
end
 

# ---------------------------
# Optimization Framework
# ---------------------------

"""
    optimize_lagrangian_newton(x, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t; γ=0.1, ε=1e-7, max_iter=1000)

Optimizes the Lagrangian function using Newton's method.

# Arguments
- `x`: Initial portfolio weights.
- `λ, μ, ν`: Initial Lagrange multipliers.
- `ξ`, `ξ_0`: Portfolio scenarios and benchmark.
- `p`: Power parameter.
- `p_ξ`, `p_ξ_0`: Probability weights.
- `t`: Initial threshold variable.
- `γ`: Step size for optimization.
- `ε`: Convergence tolerance.
- `max_iter`: Maximum number of iterations.

# Returns
- `x_opt`: Optimal portfolio weights.
- `λ_opt`: Optimal multiplier for the equality constraint.
- `μ_opt`: Optimal multiplier for stochastic dominance constraints.
- `ν_opt`: Optimal multipliers for inequality constraints.
- `t_opt`: Optimal threshold for stochastic dominance.
"""
function optimize_lagrangian_newton(x, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t; γ=0.1, ε=1e-7, max_iter=1000)
    length_x = length(x)
    constraints = []  # List to store the constraints from each iteration

    # Define the combined function without passing tilde_t_1 explicitly
    function combined_fun(vars)
        x, λ, μ, ν, t = unpack_vars(vars)
        base_grad = vcat(grad_lagrangian_x(x, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t),                    
                         grad_lagrangian_λ(x),
                         grad_lagrangian_μ(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0),
                         grad_lagrangian_ν(x),
                         g_p(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0),
                         g_p_minus_1(t, x, ξ, ξ_0, p, p_ξ, p_ξ_0))
        
        # Add all constraints from previous iterations
        for tilde_t in constraints
            gInd_constraint = gInd_p(tilde_t, x, ξ, ξ_0, p, p_ξ, p_ξ_0)
            base_grad = vcat(base_grad, gInd_constraint)
        end

        return base_grad
    end
   function unpack_vars(vars)
        x = vars[1:length_x]
        λ = vars[length_x + 1]
        μ = vars[length_x + 2]
        ν = vars[length_x + 3:length_x + 2 + length_x]
        t = vars[end]
        return x, λ, μ, ν, t
    end

    vars0 = vcat(x, λ, μ, ν, t)

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

        x_opt, λ_opt, μ_opt, ν_opt, t_opt = unpack_vars(result.xMin)

        # Define t_range and test for non-positive g_p values #[minimum(vec(x_opt' * ξ)),maximum(vec(x_opt' * ξ))]#
        t_range = minimum(ξ_0):0.001:maximum(ξ_0)
        tilde_t_1, has_positive_gp = test_g_p(t_range, x_opt, ξ, ξ_0, p, p_ξ, p_ξ_0)

        if has_positive_gp
            println("Positive g_p values found at tilde_t_1 = $tilde_t_1. Adding constraint and rerunning optimization.")
            # Add the new constraint to the list of constraints
            push!(constraints, tilde_t_1)
        else
            println("No positive g_p values found. Optimization complete.")
            return x_opt, λ_opt, μ_opt, ν_opt, t_opt
        end

        # Update vars_opt with the current optimal values for the next iteration
        vars_opt = vcat(x_opt, λ_opt, μ_opt, ν_opt, t_opt)
    end

    return x_opt, λ_opt, μ_opt, ν_opt, t_opt
end

# Define Portfolio matrix ξ ∈ R^{d×n} d assets and n scenarios
ξ = rand(5,10) # replace randome value with interested portfolio
d, n = size(ξ) 

# Define Benchmark 
τ = (ones(d)/d)'  # equally weights
ξ_0 = vec((ones(d)/d)' *ξ)


#parameter
p = 2.0  # Power parameter

# Probability vectors for ξ and ξ_0
p_ξ = (ones(n)/n) 
p_ξ_0 = (ones(n)/n) 

# Initial guesses for x, λ, μ, and ν
x = rand(d)#
x /= sum(x)  # Ensure x lies in the simplex
λ = 0.0 #Initial guess for λ
μ = 0.0 # Initial guess for μ
ν = zeros(d) # Initial guess for ν
t = minimum(ξ_0)  # Initial guess for t

# Run the optimization
@time x_opt, λ_opt, μ_opt, ν_opt, t_opt = optimize_lagrangian_newton(x, λ, μ, ν, ξ, ξ_0, p, p_ξ, p_ξ_0, t)
println("Optimal x: ", x_opt)
println("Optimal λ: ", λ_opt)
println("Optimal μ: ", μ_opt)
println("Optimal ν: ", ν_opt)
println("Optimal t: ", t_opt)
println("Maximized Portfolio Return: ", RiskFunction(x_opt,ξ,p_ξ))

