"""Check the article's finite-grid search against an independent graph solver."""
import math
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                     "content/posts/planner/task-and-motion-planning"))
import tamp_search as lab


def reference_distance(parked, radius):
    """Exhaustive Dijkstra, using no heuristic and a separate visited-set loop."""
    point = lambda n: (n[0] * .2, n[1] * .2)
    remaining = {(x, y) for x in range(-10, 11) for y in range(-6, 7)}
    costs = {(-10, 0): 0.}
    while remaining:
        node = min(remaining, key=lambda n: costs.get(n, math.inf))
        cost = costs.get(node, math.inf)
        if not math.isfinite(cost):
            return math.inf
        if node == (10, 0):
            return cost
        remaining.remove(node)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nxt = node[0] + dx, node[1] + dy
                if nxt not in remaining:
                    continue
                edge = [point(node), point(nxt)]
                if lab.geometry.verify_path(edge, radius, parked)["valid"]:
                    costs[nxt] = min(costs.get(nxt, math.inf), cost + math.dist(*edge))
    return math.inf


class SearchTests(unittest.TestCase):
    def test_published_grid_result_matches_search(self):
        reference = Path(lab.__file__).parent / "assets/grid-results.json"
        actual = json.dumps(lab.grid_motion((1., 0.), .25), indent=2, allow_nan=False) + "\n"
        self.assertEqual(actual, reference.read_text())

    def test_grid_matches_independent_shortest_paths(self):
        rng = random.Random(20261029)
        cases = [((1., 3.), .25), ((1., 0.), .25), ((0., .05), .25)]
        cases += [((rng.uniform(-1.5, 1.5), rng.uniform(-1, 1)),
                   rng.choice([.15, .25, .35])) for _ in range(12)]
        for parked, radius in cases:
            with self.subTest(parked=parked, radius=radius):
                expected = reference_distance(parked, radius)
                result = lab.grid_motion(parked, radius)
                if math.isfinite(expected):
                    self.assertEqual(result["status"], "feasible")
                    self.assertAlmostEqual(result["path_length_m"], expected, places=12)
                    self.assertTrue(lab.geometry.verify_path(result["path_m"], radius, parked)["valid"])
                    self.assertEqual((result["path_m"][0], result["path_m"][-1]), ((-2., 0.), (2., 0.)))
                else:
                    self.assertEqual(result["status"], "unknown")
                self.assertLessEqual(result["expanded_nodes"], 273)

    def test_clear_scene_is_straight_and_detour_is_longer(self):
        clear = lab.grid_motion((1., 3.), .25)
        detour = lab.grid_motion((1., 0.), .25)
        self.assertAlmostEqual(clear["path_length_m"], 4., places=12)
        self.assertGreater(detour["path_length_m"], 4.)
        self.assertTrue(any(abs(p[1]) > .65 for p in detour["path_m"]))
        length = sum(math.dist(a, b) for a, b in zip(detour["path_m"], detour["path_m"][1:]))
        self.assertAlmostEqual(length, detour["path_length_m"], places=12)

    def test_grid_failure_and_budgets_stay_unknown(self):
        for result in [lab.grid_motion((1., 0.), .25, max_expansions=0),
                       lab.grid_motion((1., 3.), .6),
                       lab.grid_motion((-2., 0.), .25)]:
            self.assertEqual(result["status"], "unknown")
            self.assertNotIn("path_m", result)
        self.assertEqual(lab.grid_motion((1., 0.), .25, max_expansions=0)["expanded_nodes"], 0)

    def test_invalid_geometry_and_expansion_budget(self):
        for parked, radius, budget in [((1.,), .25, 10), ((math.nan, 0.), .25, 10),
                ((1., 0.), -.1, 10), ((1., 0.), math.inf, 10), ((1., 0.), .25, -1),
                ((1., 0.), .25, 1.5), ((1., 0.), .25, True)]:
            with self.subTest(parked=parked, radius=radius, budget=budget), self.assertRaises(ValueError):
                lab.grid_motion(parked, radius, budget)

    def test_outer_search_revisits_unknown_and_reuses_only_proofs(self):
        result = lab.self_check()
        trace = result["detour_with_revisit"]["trace"]
        self.assertEqual([r["status"] for r in trace], ["infeasible", "unknown", "feasible"])
        self.assertEqual(trace[-1]["reason"], "verified_grid_path")
        self.assertEqual(result["blocked_with_cache"]["refinement_calls"], 3)
        self.assertEqual(result["detour_with_one_attempt"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
