from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Cut:
    coefficients: Dict[Tuple[int, int], float]
    rhs: float
    sense: str
    cut_type: str
    subset: frozenset[int]


@dataclass
class SolveResult:
    status: str
    routes: Optional[List[List[int]]]
    cost: float
    best_bound: float
    gap: float
    elapsed_time: float
    nodes_explored: int
    lp_relaxations_solved: int
    cuts_added: int
    subtour_cuts: int
    capacity_cuts: int


class CVRPBranchAndCut:
    """
    Educational exact Branch-and-Cut solver for small symmetric CVRP instances.

    Exactness scope:
    - symmetric nonnegative distance matrix
    - depot is node 0
    - identical vehicles
    - exact number of vehicles used
    - each customer visited exactly once
    - capacity enforced through exhaustive rounded-capacity cut separation
      over all nonempty proper customer subsets

    This implementation is intended for small instances because exact subset
    separation is exponential.
    """

    def __init__(self, distance_matrix: np.ndarray, demands: List[int],
                 num_vehicles: int, vehicle_capacity: int):
        d = np.asarray(distance_matrix, dtype=float)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError("distance_matrix must be square")
        if not np.all(np.isfinite(d)) or np.any(d < -1e-12):
            raise ValueError("distances must be finite and nonnegative")
        if not np.allclose(d, d.T, atol=1e-10):
            raise ValueError("exact solver requires a symmetric distance matrix")
        if len(demands) != d.shape[0] or demands[0] != 0:
            raise ValueError("demands must match matrix size and depot demand must be 0")
        if num_vehicles <= 0 or vehicle_capacity <= 0:
            raise ValueError("num_vehicles and vehicle_capacity must be positive")
        if any(int(x) < 0 for x in demands):
            raise ValueError("demands must be nonnegative")

        self.d = d
        self.demands = tuple(int(x) for x in demands)
        self.k = int(num_vehicles)
        self.Q = int(vehicle_capacity)
        self.n = d.shape[0]
        self.customers = tuple(range(1, self.n))

        self.edge_to_var = {}
        self.var_to_edge = {}
        idx = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.edge_to_var[(i, j)] = idx
                    self.var_to_edge[idx] = (i, j)
                    idx += 1
        self.num_vars = idx

        self.nodes_explored = 0
        self.lp_relaxations_solved = 0
        self.cuts_added = 0
        self.subtour_cuts = 0
        self.capacity_cuts = 0
        self.best_solution = None
        self.best_cost = math.inf

        self.trivially_infeasible = (
            any(self.demands[i] > self.Q for i in self.customers)
            or sum(self.demands[1:]) > self.k * self.Q
            or self.k > len(self.customers)
        )

    def _objective(self):
        c = np.zeros(self.num_vars)
        for idx, (i, j) in self.var_to_edge.items():
            c[idx] = self.d[i, j]
        return c

    def _solve_lp(self, fixed_vars, cuts):
        self.lp_relaxations_solved += 1
        c = self._objective()
        A_eq, b_eq, A_ub, b_ub = [], [], [], []

        for i in self.customers:
            row = np.zeros(self.num_vars)
            for j in range(self.n):
                if i != j:
                    row[self.edge_to_var[(i, j)]] = 1.0
            A_eq.append(row); b_eq.append(1.0)

        for j in self.customers:
            row = np.zeros(self.num_vars)
            for i in range(self.n):
                if i != j:
                    row[self.edge_to_var[(i, j)]] = 1.0
            A_eq.append(row); b_eq.append(1.0)

        row = np.zeros(self.num_vars)
        for j in self.customers:
            row[self.edge_to_var[(0, j)]] = 1.0
        A_eq.append(row); b_eq.append(float(self.k))

        row = np.zeros(self.num_vars)
        for i in self.customers:
            row[self.edge_to_var[(i, 0)]] = 1.0
        A_eq.append(row); b_eq.append(float(self.k))

        for var_idx, value in fixed_vars.items():
            row = np.zeros(self.num_vars); row[var_idx] = 1.0
            A_eq.append(row); b_eq.append(float(value))

        for cut in cuts:
            row = np.zeros(self.num_vars)
            for edge, coef in cut.coefficients.items():
                row[self.edge_to_var[edge]] = coef
            if cut.sense == "<=":
                A_ub.append(row); b_ub.append(cut.rhs)
            else:
                A_ub.append(-row); b_ub.append(-cut.rhs)

        result = linprog(
            c,
            A_ub=np.asarray(A_ub) if A_ub else None,
            b_ub=np.asarray(b_ub) if b_ub else None,
            A_eq=np.asarray(A_eq),
            b_eq=np.asarray(b_eq),
            bounds=[(0.0, 1.0)] * self.num_vars,
            method="highs",
        )
        if not result.success:
            return math.inf, None
        return float(result.fun), result.x

    def _subset_internal_value(self, x, S):
        return sum(x[self.edge_to_var[(i, j)]] for i in S for j in S if i != j)

    def _subset_out_value(self, x, S):
        return sum(x[self.edge_to_var[(i, j)]] for i in S for j in range(self.n) if j not in S)

    def _sec_cut(self, S):
        return Cut({(i, j): 1.0 for i in S for j in S if i != j}, float(len(S)-1), "<=", "subtour", frozenset(S))

    def _capacity_cut(self, S):
        required = math.ceil(sum(self.demands[i] for i in S) / self.Q)
        return Cut({(i, j): 1.0 for i in S for j in range(self.n) if j not in S}, float(required), ">=", "capacity", frozenset(S))

    def _separate_all_violated_cuts(self, x, existing_keys, tol=1e-9):
        violated = []
        customers = list(self.customers)
        for size in range(1, len(customers)):
            for subset in itertools.combinations(customers, size):
                S = set(subset)
                sec_key = ("subtour", frozenset(S))
                if size >= 2 and sec_key not in existing_keys:
                    if self._subset_internal_value(x, S) > len(S)-1 + tol:
                        violated.append(self._sec_cut(S))
                cap_key = ("capacity", frozenset(S))
                if cap_key not in existing_keys:
                    required = math.ceil(sum(self.demands[i] for i in S) / self.Q)
                    if self._subset_out_value(x, S) < required - tol:
                        violated.append(self._capacity_cut(S))
        return violated

    @staticmethod
    def _is_integral(x, tol=1e-8):
        return bool(np.all(np.abs(x - np.round(x)) <= tol))

    def _extract_routes(self, x):
        succ = {i: [] for i in range(self.n)}
        for idx, value in enumerate(x):
            if value > 0.5:
                i, j = self.var_to_edge[idx]
                succ[i].append(j)
        starts = list(succ[0])
        if len(starts) != self.k:
            raise RuntimeError("integer solution does not have exactly k depot departures")

        routes = []
        globally_visited = set()
        for start in starts:
            route = [0, start]
            current = start
            local_seen = {0, start}
            globally_visited.add(start)
            while current != 0:
                nexts = succ[current]
                if len(nexts) != 1:
                    raise RuntimeError("customer must have exactly one successor")
                nxt = nexts[0]
                if nxt != 0:
                    if nxt in local_seen:
                        raise RuntimeError("subtour detected in integer solution")
                    local_seen.add(nxt)
                    globally_visited.add(nxt)
                route.append(nxt)
                current = nxt
            routes.append(route)
        if globally_visited != set(self.customers):
            raise RuntimeError("integer solution does not cover every customer exactly once")
        return routes

    def _validate_routes(self, routes):
        if len(routes) != self.k:
            raise RuntimeError("wrong number of routes")
        seen = []
        total = 0.0
        for route in routes:
            if route[0] != 0 or route[-1] != 0:
                raise RuntimeError("every route must start and end at depot")
            customers = route[1:-1]
            if sum(self.demands[v] for v in customers) > self.Q:
                raise RuntimeError("capacity violation in integer route")
            seen.extend(customers)
            total += sum(self.d[a, b] for a, b in zip(route, route[1:]))
        if sorted(seen) != list(self.customers):
            raise RuntimeError("customer coverage violation")
        return total

    def _greedy_upper_bound(self):
        remaining = set(self.customers)
        routes = []
        total = 0.0
        for _ in range(self.k):
            if not remaining:
                break
            route = [0]; load = 0; cur = 0
            while remaining:
                feasible = [j for j in remaining if load + self.demands[j] <= self.Q]
                if not feasible:
                    break
                nxt = min(feasible, key=lambda j: self.d[cur, j])
                total += self.d[cur, nxt]
                route.append(nxt)
                load += self.demands[nxt]
                remaining.remove(nxt)
                cur = nxt
            if len(route) > 1:
                total += self.d[cur, 0]
                route.append(0)
                routes.append(route)
        if remaining or len(routes) != self.k:
            return None, math.inf
        return routes, total

    def _routes_to_x(self, routes):
        x = np.zeros(self.num_vars)
        for route in routes:
            for a, b in zip(route, route[1:]):
                x[self.edge_to_var[(a, b)]] = 1.0
        return x

    def solve(self, time_limit=120.0):
        start = time.time()
        if self.trivially_infeasible:
            return SolveResult("INFEASIBLE", None, math.inf, math.inf, math.inf, 0.0, 0, 0, 0, 0, 0)

        routes, ub = self._greedy_upper_bound()
        if routes is not None:
            self.best_cost = ub
            self.best_solution = self._routes_to_x(routes)

        best_open_bound = math.inf
        timed_out = False

        def recurse(fixed_vars, inherited_cuts):
            nonlocal best_open_bound, timed_out
            if timed_out:
                return
            if time.time() - start >= time_limit:
                timed_out = True
                return
            self.nodes_explored += 1
            cuts = list(inherited_cuts)
            existing_keys = {(c.cut_type, c.subset) for c in cuts}

            while True:
                if time.time() - start >= time_limit:
                    timed_out = True
                    return
                lb, x = self._solve_lp(fixed_vars, cuts)
                if x is None:
                    return
                best_open_bound = min(best_open_bound, lb)
                if lb >= self.best_cost - 1e-10:
                    return
                new_cuts = self._separate_all_violated_cuts(x, existing_keys)
                if new_cuts:
                    cuts.extend(new_cuts)
                    for cut in new_cuts:
                        existing_keys.add((cut.cut_type, cut.subset))
                        self.cuts_added += 1
                        if cut.cut_type == "subtour":
                            self.subtour_cuts += 1
                        else:
                            self.capacity_cuts += 1
                    continue
                if self._is_integral(x):
                    candidate_routes = self._extract_routes(x)
                    candidate_cost = self._validate_routes(candidate_routes)
                    if not math.isclose(candidate_cost, lb, rel_tol=0, abs_tol=1e-6):
                        raise RuntimeError("route reconstruction cost differs from LP objective")
                    if candidate_cost < self.best_cost - 1e-10:
                        self.best_cost = candidate_cost
                        self.best_solution = x.copy()
                    return
                fractional = [(abs(x[i]-0.5), i) for i in range(self.num_vars) if 1e-8 < x[i] < 1-1e-8]
                if not fractional:
                    raise RuntimeError("nonintegral node has no branchable variable")
                _, branch_var = min(fractional)
                child0 = dict(fixed_vars); child0[branch_var] = 0
                recurse(child0, cuts)
                if timed_out:
                    return
                child1 = dict(fixed_vars); child1[branch_var] = 1
                recurse(child1, cuts)
                return

        recurse({}, [])
        elapsed = time.time() - start

        if self.best_solution is None:
            status = "TIME_LIMIT" if timed_out else "INFEASIBLE"
            return SolveResult(status, None, math.inf, best_open_bound if math.isfinite(best_open_bound) else math.inf, math.inf, elapsed, self.nodes_explored, self.lp_relaxations_solved, self.cuts_added, self.subtour_cuts, self.capacity_cuts)

        final_routes = self._extract_routes(self.best_solution)
        final_cost = self._validate_routes(final_routes)
        if timed_out:
            bound = best_open_bound if math.isfinite(best_open_bound) else -math.inf
            gap = math.inf if not math.isfinite(bound) else max(0.0, (final_cost-bound)/max(abs(final_cost),1e-12))
            status = "TIME_LIMIT"
        else:
            bound = final_cost
            gap = 0.0
            status = "OPTIMAL"
        return SolveResult(status, final_routes, final_cost, bound, gap, elapsed, self.nodes_explored, self.lp_relaxations_solved, self.cuts_added, self.subtour_cuts, self.capacity_cuts)


def create_test_instance(n_customers: int, seed: int = 42):
    np.random.seed(seed)
    coords = np.random.rand(n_customers + 1, 2) * 100
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    demands = [0] + list(np.random.randint(5, 15, n_customers))
    return d, demands, coords


if __name__ == "__main__":
    d, demands, _ = create_test_instance(6, 42)
    result = CVRPBranchAndCut(d, demands, 2, 30).solve(60)
    print(result)
