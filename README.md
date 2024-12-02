# Higher-Order Stochastic Dominance Constraints in Optimization

This repository provides an efficient framework to solve optimization problems involving **higher-order stochastic dominance (HOSD) constraints**. These constraints are often uncountably infinite, but this implementation reduces the problem to a finite set of test points, making it computationally feasible.

## Key Features
- **Finite Reduction of Constraints:** 
   - We simplify uncountable stochastic dominance constraints into a finite, computationally verifiable set of test points.
- **Two Formulations:** 
   - Based on **expectation operators**.
   - Based on **risk measures**.
- **Optimization Framework:**
   - Incorporates both formulations for verification and efficient computation.
- **Generalized Application:** 
   - Can be applied to portfolio optimization and other areas involving risk and dominance constraints.

---

## Getting Started

### Prerequisites
Before running the code, ensure you have the following installed:
- Julia (v1.7 or later)
- Dependencies for scientific computation: `LinearAlgebra`, `Plots`, `ForwardDiff`, `Random`

Install dependencies via Julia's package manager:
```julia
using Pkg
Pkg.add("Plots")
Pkg.add("ForwardDiff")
Pkg.add("Random")
