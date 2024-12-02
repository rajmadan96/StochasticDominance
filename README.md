# Higher-Order Stochastic Dominance Constraints in Optimization

This repository provides an efficient framework to solve optimization problems involving **higher-order stochastic dominance (HOSD) constraints**. These constraints are often uncountably infinite, but this implementation reduces the problem to a finite set of test points, making it computationally feasible.

## Key Features
- **Finite Reduction of Constraints:** 
   - We simplify uncountable stochastic dominance constraints into a finite, computationally verifiable set of test points.
- **Optimization Framework:**
   - Incorporates theoretical verification setup using Newton method
- **Two Variants:** 
   - Objective: **Maximize Expected Return**
   - Objective: Minimize **Risk measure**
---

## Getting Started

### Prerequisites
Before running the code, ensure you have the following installed:
- Julia (v1.7 or later)
- Dependencies for scientific computation: `LinearAlgebra`, `ForwardDiff`

Install dependencies via Julia's package manager:
```julia
using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("ForwardDiff")
```

## Code Structure

The directory is organized as follows:

| File/Folder         | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `src/`              | Contains source code, including the main implementation and utilities. |
| `Dataset/`         | Provides datasets used for testing and experiments.                    |
| `Prominent Algorithm/`       | Contains implementations of prominent stochastic dominance approaches. |
| `README.md`         | Overview and instructions for the project.                             |

---

## Citing This Work

If you use this code, please cite the corresponding research paper:

**TODO: Add link to the paper**

This paper comprehensively explains the implementation and methodology of our proposed approach.

---

## Contributions
We welcome contributions! Please feel free to open an issue or submit a pull request if you have suggestions, bug reports, or feature requests.
