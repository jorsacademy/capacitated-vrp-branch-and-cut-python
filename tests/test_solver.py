import itertools
import math
import unittest

import numpy as np

from cvrp_branch_and_cut import CVRPBranchAndCut, create_test_instance


def exhaustive_exact_k(distance_matrix, demands, vehicles, capacity):
    n = len(distance_matrix) - 1
    customers = list(range(1, n + 1))
    if vehicles > n:
        return math.inf
    best = math.inf
    for perm in itertools.permutations(customers):
        for cuts in itertools.combinations(range(1, n), vehicles - 1):
            prev = 0
            cost = 0.0
            feasible = True
            for cut in cuts + (n,):
                part = perm[prev:cut]
                prev = cut
                if not part or sum(demands[i] for i in part) > capacity:
                    feasible = False
                    break
                route = (0,) + part + (0,)
                cost += sum(distance_matrix[a, b] for a, b in zip(route, route[1:]))
            if feasible:
                best = min(best, cost)
    return best


class BranchAndCutTests(unittest.TestCase):
    def test_reference_instance(self):
        d, demands, _ = create_test_instance(6, 42)
        result = CVRPBranchAndCut(d, demands, 2, 30).solve(30)
        self.assertEqual(result.status, "OPTIMAL")
        self.assertTrue(math.isclose(result.cost, 319.7050045735702, abs_tol=1e-8))
        self.assertEqual(len(result.routes), 2)

    def test_capacity_counterexample_is_infeasible(self):
        n = 5
        d = np.full((n + 1, n + 1), 10.0)
        np.fill_diagonal(d, 0)
        for i in range(1, n):
            d[i, i + 1] = d[i + 1, i] = 1
        d[1, 5] = d[5, 1] = 1
        result = CVRPBranchAndCut(d, [0, 3, 3, 3, 3, 3], 1, 14).solve(30)
        self.assertEqual(result.status, "INFEASIBLE")

    def test_asymmetric_matrix_rejected(self):
        d = np.array([[0, 1, 5], [2, 0, 1], [1, 4, 0]], dtype=float)
        with self.assertRaises(ValueError):
            CVRPBranchAndCut(d, [0, 1, 1], 1, 3)

    def test_small_instances_match_exhaustive_oracle(self):
        for n in [3, 4, 5]:
            for seed in range(3):
                pts = np.random.default_rng(seed).random((n + 1, 2)) * 50
                d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
                demands = [0] + list(np.random.default_rng(seed + 100).integers(1, 7, size=n))
                for vehicles in range(1, min(3, n) + 1):
                    oracle = exhaustive_exact_k(d, demands, vehicles, 8)
                    result = CVRPBranchAndCut(d, demands, vehicles, 8).solve(30)
                    if math.isfinite(oracle):
                        self.assertEqual(result.status, "OPTIMAL")
                        self.assertTrue(math.isclose(result.cost, oracle, abs_tol=1e-8))
                    else:
                        self.assertEqual(result.status, "INFEASIBLE")

    def test_timeout_status(self):
        d, demands, _ = create_test_instance(7, 42)
        result = CVRPBranchAndCut(d, demands, 3, 30).solve(1e-6)
        self.assertIn(result.status, {"TIME_LIMIT", "OPTIMAL"})


if __name__ == "__main__":
    unittest.main()
