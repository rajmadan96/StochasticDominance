"""
	Stocks to annualized return 
	created: 2024, July
	author©: Rajmadan Lakshmanan
"""

using Random
using Statistics

# Function to calculate annualized return for multiple portfolios
"""
    annualized_return(S::Any, t::Any) -> Matrix{Float64}

Compute the annualized return for multiple portfolios over multiple time points.

# Arguments
- `S::Any`: Matrix of stock prices where each row corresponds to a portfolio of assets (d) and each column corresponds to a time point (n).
- `t::Any`: Vector of time points.

# Returns
- `Matrix{Float64}`: Matrix of annualized returns where each row corresponds to a portfolio.
"""
function annualized_return(S::Any, t::Any)
    d, n = size(S)
    ξ = zeros(Float64, d, n-1)
    for i in 2:n
        ξ[:, i-1] = (1 / (t[i] - t[i-1])) .* log.(S[:, i] ./ S[:, i-1])
    end
    return ξ
end

# Function to calculate weights (probabilities)
"""
    weights(t::Any) -> Vector{Float64}

Compute the weights (probabilities) for given time points.

# Arguments
- `t::Any`: Vector of time points.

# Returns
- `Vector{Float64}`: Vector of weights.
"""
function weights(t::Any)
    n = length(t)
    p = zeros(Float64, n-1)
    for i in 2:n
        p[i-1] = (t[i] - t[i-1]) / (t[end] - t[1])
    end
    return p
end


