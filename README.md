# Capacitated VRP Branch-and-Cut in Python

Educational exact Branch-and-Cut implementation for small symmetric Capacitated Vehicle Routing Problem (CVRP) instances.

## Model scope

The solver minimizes total routing distance with one depot, exactly `k` identical vehicle routes, customer-once service, and vehicle-capacity constraints. It accepts symmetric nonnegative distance matrices and rejects asymmetric instances.

## Branch-and-Cut design

The implementation combines:

- LP relaxation solved with SciPy/HiGHS;
- dynamic subtour-elimination constraints;
- rounded-capacity inequalities;
- branching on fractional arc variables;
- greedy incumbent generation;
- explicit `OPTIMAL`, `TIME_LIMIT`, and `INFEASIBLE` statuses.

For exact small-instance validation, separation enumerates every nonempty proper customer subset. This is exponential and intentionally limits the practical problem size.

## Reference instance

For the reproducible 6-customer instance (`seed=42`, 2 vehicles, capacity 30):

- optimal cost: `319.7050045735702`;
- one equivalent optimum is:
  - `0 -> 3 -> 5 -> 0`
  - `0 -> 4 -> 1 -> 6 -> 2 -> 0`
- the exact objective was independently verified by exhaustive enumeration.

## Validation

The test suite covers:

- the reference optimum;
- exhaustive-oracle agreement on many small instances;
- infeasibility detection;
- a capacity-cut counterexample that the earlier implementation accepted incorrectly;
- asymmetric-matrix rejection;
- timeout status behavior.

Run:

```bash
python -m unittest discover -s tests -v
```

## Usage

```python
from cvrp_branch_and_cut import CVRPBranchAndCut, create_test_instance

d, demands, _ = create_test_instance(6, seed=42)
result = CVRPBranchAndCut(d, demands, 2, 30).solve(time_limit=60)

print(result.status)
print(result.cost)
print(result.routes)
```

## Branch-and-Cut vs Branch-and-Bound

Cuts can substantially reduce the search-tree size by strengthening LP bounds. On very small instances, however, LP solves and separation overhead can make Branch-and-Cut slower in wall-clock time than a lightweight exact Branch-and-Bound implementation. This repository therefore avoids simulated performance claims.

## Requirements

- Python 3.10+
- NumPy
- SciPy

## Limitations

This is an educational exact solver for small CVRP instances, not a commercial-scale routing engine. It does not implement advanced cut pools, max-flow separation, strong branching, presolve, parallel search, time windows, heterogeneous fleets, or pickup-and-delivery.
