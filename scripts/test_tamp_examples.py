"""Check grid search, shared-grasp kinematics and the published scheduler."""
import math
import cmath
from collections import deque
from decimal import Decimal, localcontext
import json
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                     "content/posts/planner/task-and-motion-planning"))
import tamp_search as lab
import tamp_kinematics as kin
from validate_blog import code_blocks


def reference_distance(parked, radius, step=.2, bounds=(-10, 10, -6, 6)):
    """Exhaustive Dijkstra, using no heuristic and a separate visited-set loop."""
    point = lambda n: (n[0] * step, n[1] * step)
    remaining = {(x, y) for x in range(bounds[0], bounds[1] + 1)
                 for y in range(bounds[2], bounds[3] + 1)}
    costs = {(bounds[0], 0): 0.}
    while remaining:
        node = min(remaining, key=lambda n: costs.get(n, math.inf))
        cost = costs.get(node, math.inf)
        if not math.isfinite(cost):
            return math.inf
        if node == (bounds[1], 0):
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
    def test_segment_distances_against_high_precision_convex_minimization(self):
        # Minimize the convex squared distance along the segment using Decimal
        # ternary search, independently of the implementation's dot projection
        # and segment/ray intersection branches.
        rng = random.Random(20261030)
        cases = [[tuple(rng.uniform(-2, 2) for _ in range(2)) for _ in range(3)] for _ in range(60)]
        cases += [((0., 0.), (-1., .5000000001), (1., .5000000001)),
                  ((0., 0.), (-1., .4999999999), (1., .4999999999)),
                  ((0., 1e-12), (-1e-12, 0.), (1e-12, 0.)),
                  ((0., 0.), (.3, -1.), (.3, 1.)),
                  ((0., 0.), (.4, .2), (.4, .2))]
        with localcontext() as context:
            context.prec = 70
            zero, one, three = Decimal(0), Decimal(1), Decimal(3)
            for point, start, end in cases:
                p, a, b = [[Decimal.from_float(float(value)) for value in coordinates]
                           for coordinates in (point, start, end)]
                def point_at(t):
                    return [a[i] + t * (b[i] - a[i]) for i in range(2)]
                def minimum(cost):
                    left, right = zero, one
                    for _ in range(220):
                        first, second = (2*left + right)/three, (left + 2*right)/three
                        if cost(first) <= cost(second):
                            right = second
                        else:
                            left = first
                    return float(min(cost(zero), cost(one), cost((left+right)/2)).sqrt())
                expected = minimum(lambda t: sum((x-y)**2 for x, y in zip(point_at(t), p)))
                self.assertAlmostEqual(lab.geometry.point_segment_distance(point, start, end), expected, delta=2e-14)
                for sign in (-1, 1):
                    def ray_cost(t):
                        x, y = point_at(t)
                        gap = max(zero, Decimal.from_float(lab.geometry.HALF_OPENING) - sign*y)
                        return x*x + gap*gap
                    expected = minimum(ray_cost)
                    self.assertAlmostEqual(lab.geometry.segment_wall_ray_distance(start, end, sign), expected, delta=2e-14)
        with mock.patch.object(lab.geometry, "BLOCKER_RADIUS", .25), mock.patch.object(lab.geometry, "EPS", 0.):
            for clearance, valid in ((.5-1e-12, False), (.5, False), (.5+1e-12, True)):
                self.assertEqual(lab.geometry.verify_path([(.3, clearance), (2., clearance)], .25, (1., 0.))["valid"], valid)

    def test_geometry_range_loss_never_becomes_clearance(self):
        # Squared lengths overflowed to infinity; inf/inf was clamped into a
        # false clear path even though the segment crosses the blocker centre.
        for scale in (1e160, 1e200, 1e308, 10**160, 10**200):
            path = [(-scale, 0.), (scale, 0.)]
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                lab.geometry.verify_path(path, .25, (0., 0.))
            with self.assertRaises(ValueError):
                lab.geometry.segment_wall_ray_distance(*path, 1)
        with self.assertRaises(ValueError):
            lab.geometry.point_segment_distance((0., 0.), (0., 0.), (1e-200, 0.))
        self.assertEqual(lab.geometry.point_segment_distance((0., 0.), (-1e150, 0.), (1e150, 0.)), 0.)

    def test_invalid_scene_never_returns_a_witness_or_certificate(self):
        path = [(-2., 0.), (2., 0.)]
        bad_points = [None, (), (1.,), (0., 0., 0.), (math.inf, 0.),
                      (0., math.nan), (True, 0.), ("0", 0.), (10**1000, 0.)]
        for parked in bad_points:
            for function, arguments in (
                    (lab.geometry.verify_path, (path, .25, parked)),
                    (lab.geometry.crossing_intervals, (.6, parked)),
                    (lab.geometry.refine, (parked, .6)),
                    (lab.grid_motion, (parked, .25)),
                    (lab.search, (parked, False))):
                with self.subTest(function=function.__name__, parked=parked), self.assertRaises(ValueError):
                    function(*arguments)
        for radius in (-1., math.nan, math.inf, True, "0.25", None):
            for function, arguments in (
                    (lab.geometry.verify_path, (path, radius, (0., 0.))),
                    (lab.geometry.crossing_intervals, (radius, (0., 0.))),
                    (lab.geometry.refine, ((0., 0.), radius)),
                    (lab.grid_motion, ((0., 0.), radius))):
                with self.subTest(function=function.__name__, radius=radius), self.assertRaises(ValueError):
                    function(*arguments)
        for points in (None, [(-2., 0.), (math.nan, 0.)], [(0., 0., 0.), (2., 0.)]):
            with self.subTest(path=points), self.assertRaises(ValueError):
                lab.geometry.verify_path(points, .25, (1., 1.))
        for name, value in (("EPS", -1.), ("BLOCKER_RADIUS", -.1), ("HALF_OPENING", 0.), ("EPS", math.nan)):
            with mock.patch.object(lab.geometry, name, value), self.assertRaises(ValueError):
                lab.geometry.refine((0., 0.), .6)
        self.assertTrue(lab.geometry.verify_path(path, 0., (1., 1.))["valid"])

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

    def test_resolution_study_and_continuous_counterexample(self):
        result = lab.grid_resolution_demo()
        self.assertEqual([r["grid_node_count"] for r in result["results"]], [3, 15, 77, 273, 1025])
        coarse, *resolved = result["results"]
        self.assertEqual((coarse["status"], coarse["reason"]), ("unknown", "finite_grid_exhausted"))
        # A separately specified continuous detour still exists in this scene.
        witness = [(-2., 0.), (.3, 0.), (.3, .8), (1.8, .8), (2., 0.)]
        self.assertTrue(lab.geometry.verify_path(witness, .25, (1., 0.))["valid"])
        for row in resolved:
            self.assertEqual(row["status"], "feasible")
            self.assertTrue(lab.geometry.verify_path(row["path_m"], .25, (1., 0.))["valid"])
        self.assertEqual(resolved[2], lab.grid_motion((1., 0.), .25))
        reference = Path(lab.__file__).parent / "assets/grid-resolution-results.json"
        self.assertEqual(json.dumps(result, indent=2, allow_nan=False) + "\n", reference.read_text())

    def test_other_resolutions_against_independent_shortest_paths(self):
        # Bounds are declared here rather than taken from the layout under test.
        for step, bounds in ((2., (-1, 1, 0, 0)), (1., (-2, 2, -1, 1)),
                             (.4, (-5, 5, -3, 3)), (.1, (-20, 20, -12, 12))):
            for parked, radius in (((1., 0.), .25), ((1., 3.), .25), ((0., .05), .25)):
                with self.subTest(step=step, parked=parked):
                    expected = reference_distance(parked, radius, step, bounds)
                    result = lab.grid_motion(parked, radius, step=step)
                    if math.isfinite(expected):
                        self.assertEqual(result["status"], "feasible")
                        self.assertAlmostEqual(result["path_length_m"], expected, places=12)
                    else:
                        self.assertEqual(result["status"], "unknown")

    def test_invalid_resolution_and_nondefault_budget(self):
        for step in (0., -.1, math.inf, math.nan, True, "0.2", 1e-300, .3, 3., 10**1000):
            with self.subTest(step=step), self.assertRaises(ValueError):
                lab.grid_motion((1., 0.), .25, step=step)
        for step in (2., 1., .4, .1):
            result = lab.grid_motion((1., 0.), .25, max_expansions=0, step=step)
            self.assertEqual((result["status"], result["reason"], result["expanded_nodes"]),
                             ("unknown", "grid_budget_exhausted", 0))

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

    def test_outer_search_rejects_invalid_states_and_unverified_proofs(self):
        for result in (None, {}, {"status": "timeout", "reason": "budget"},
                       {"status": "infeasible", "reason": "local_optimizer_failed"}):
            with self.subTest(result=result), mock.patch.object(lab, "refine_one", return_value=result):
                with self.assertRaises(ValueError):
                    lab.search()
        # A known reason string is insufficient when the actual geometry
        # does not satisfy that certificate's preconditions.
        candidate = lab.Candidate("direct", (1., 1.), "narrow")
        with mock.patch.object(lab, "candidates", return_value=deque([candidate])), \
             mock.patch.object(lab, "refine_one", return_value={"status": "infeasible", "reason": "door_width_certificate"}):
            with self.assertRaises(ValueError):
                lab.search()

    def test_safe_path_must_also_connect_the_task_endpoints(self):
        # A clear segment entirely to the left of the wall is not a transfer
        # from the prescribed start to goal. Collision checks alone accept it.
        path = [(-2., 0.), (-1., 0.)]
        self.assertTrue(lab.geometry.verify_path(path, .6, (0., .05))["valid"])
        for invalid in (path, list(reversed(path)), [(2., 0.), (-2., 0.)],
                        [(-2., 0.), (math.nan, 1.), (2., 0.)],
                        [(-2., 0.), (10**1000, 0.), (2., 0.)],
                        [(-2., 0.), (True, 0.), (2., 0.)],
                        [(-2., 0.), ("0", 0.), (2., 0.)],
                        [(-2., 0.), (2., 0., 0.)], []):
            result = {"status": "feasible", "reason": "unverified", "path_m": invalid}
            with self.subTest(path=invalid), mock.patch.object(lab, "refine_one", return_value=result):
                with self.assertRaises(ValueError):
                    lab.search()


class SharedGraspTests(unittest.TestCase):
    def test_published_kinematic_result(self):
        expected = Path(kin.__file__).parent / "assets/grasp-results.json"
        actual = json.dumps(kin.self_check(), indent=2, allow_nan=False) + "\n"
        self.assertEqual(actual, expected.read_text())

    def test_transform_direction_and_noncommutativity(self):
        parent, child = kin.Pose(1., 2., math.pi / 2), kin.Pose(.5, 0., -math.pi / 2)
        result = kin.compose(parent, child)
        self.assertAlmostEqual(result.x, 1.)
        self.assertAlmostEqual(result.y, 2.5)
        self.assertAlmostEqual(result.yaw, 0.)
        reversed_result = kin.compose(child, parent)
        self.assertGreater(kin.residual(result, reversed_result)["position_m"], 1.)
        for p in (parent, child):
            identity = kin.compose(p, kin.inverse(p))
            self.assertLess(math.hypot(identity.x, identity.y), 1e-12)
            self.assertLess(abs(identity.yaw), 1e-12)

    def test_endpoint_grasp_intersection_and_regrasp(self):
        result = kin.demo()
        self.assertEqual(result["direct_shared_grasps"], [])
        self.assertEqual(result["via_middle_shared_grasps"], {
            "left_to_middle": ["back"], "middle_to_right": ["front"]})
        for placement, grasp, expected in (("left", "front", 2.25), ("left", "back", 1.75),
                                          ("right", "front", 1.75), ("right", "back", 2.25)):
            self.assertAlmostEqual(result["endpoint_witnesses"][placement][grasp]["wrist_radius_m"], expected)
        self.assertIn("connecting paths", result["unverified"])
        self.assertIn("support and grasp stability", result["unverified"])

    def test_ik_witnesses_with_independent_complex_fk(self):
        rng = random.Random(12)
        for _ in range(200):
            joints = [rng.uniform(-math.pi, math.pi) for _ in range(3)]
            # Evaluate each link with a complex rotation rather than the
            # companion's incremental x/y recursion.
            tip = sum(length * cmath.exp(1j * sum(joints[:index + 1]))
                      for index, length in enumerate(kin.LINKS))
            target = kin.Pose(tip.real, tip.imag, kin.wrap(sum(joints)))
            solutions = kin.hand_ik(target)
            self.assertTrue(solutions)
            for q in solutions:
                actual = sum(length * cmath.exp(1j * sum(q[:index + 1]))
                             for index, length in enumerate(kin.LINKS))
                self.assertLess(abs(actual - tip), 1e-10)
                self.assertLess(abs(kin.wrap(sum(q) - target.yaw)), 1e-10)

    def test_singular_representatives_and_clear_unreachability(self):
        self.assertEqual(len(kin.hand_ik(kin.Pose(2.25, 0., 0.))), 1)
        self.assertTrue(kin.hand_ik(kin.Pose(-.25, 0., math.pi)))
        self.assertEqual(kin.hand_ik(kin.Pose(2.25 + 1e-6, 0., 0.)), [])
        for invalid in ((True, 0., 0.), (0., math.nan, 0.), (0., 0., "0")):
            with self.assertRaises(ValueError):
                kin.Pose(*invalid)
        for invalid in ([], [0., 0.], [0., 0., math.inf], [0., 0., True]):
            with self.assertRaises(ValueError):
                kin.forward(invalid)


class PublishedSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        article = Path(lab.__file__).parent / "index.md"
        source = next(code for language, code, _ in code_blocks(article.read_text())
                      if language == "python" and "def schedule(" in code)
        namespace = {}
        exec(compile(source, str(article), "exec"), namespace)
        cls.schedule = staticmethod(namespace["schedule"])

    def test_budget_preserves_unknown_and_pending_counts(self):
        calls = []
        def reject(candidate):
            calls.append(candidate)
            return {"status": "infeasible"}
        for budget in range(3):
            result = self.schedule(["A", "B"], reject, lambda _: None, max_calls=budget)
            self.assertEqual(result["pending"], 2 - budget)
            self.assertEqual(result["status"], "unknown" if budget < 2 else "finite_candidates_infeasible")
        self.assertEqual(calls, ["A", "A", "B"])

    def test_round_robin_retry(self):
        calls = []
        def refine(candidate):
            calls.append(candidate)
            return {"status": {"A0": "unknown", "B0": "infeasible", "A1": "feasible"}[candidate]}
        result = self.schedule(["A0", "B0"], refine, lambda _: "A1", max_calls=3)
        self.assertEqual(result["status"], "feasible")
        self.assertEqual(calls, ["A0", "B0", "A1"])

    def test_invalid_statuses_and_budgets_are_rejected(self):
        for result in (None, {}, {"status": "timed_out"}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.schedule([1], lambda _: result, lambda _: None)
        for budget in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.schedule([1], lambda _: {"status": "feasible"}, lambda _: None, max_calls=budget)


if __name__ == "__main__":
    unittest.main()
