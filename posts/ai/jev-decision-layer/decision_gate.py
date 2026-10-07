"""Offline typed-decision exercise; no model calls and no robot control.

Python 3.10+, standard library only. Thresholds are synthetic test fixtures.
Observation metadata belongs to the request wrapper, not the Jev answer.
"""
from dataclasses import dataclass, replace
import argparse
import json
import math
from pathlib import Path
import unittest


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class RequestContext:
    observation_id: int
    observed_at: float
    sent_at: float
    options: frozenset[str]
    goal_version: int = 1
    confidence_floor: float = 0.8
    max_observation_age: float = 0.5
    max_request_age: float = 0.3

    def __post_init__(self):
        for version in (self.observation_id, self.goal_version):
            if type(version) is not int or version < 0:
                raise ValueError("nonnegative integer versions required")
        if (type(self.options) is not frozenset or not self.options
                or not all(isinstance(x, str) and x.strip() for x in self.options)):
            raise ValueError("nonempty immutable set of string options required")
        if not all(finite_number(t) for t in (self.observed_at, self.sent_at)):
            raise ValueError("finite timestamps required")
        if self.observed_at > self.sent_at:
            raise ValueError("observation must precede the request")
        for duration in (self.max_observation_age, self.max_request_age):
            if not finite_number(duration) or duration <= 0:
                raise ValueError("positive finite time limits required")
        if not finite_number(self.confidence_floor) or not 0 <= self.confidence_floor <= 1:
            raise ValueError("confidence floor must be in [0, 1]")


def route(answer, request, *, observation_id, goal_version, now,
          execution_allowed, goal_verified):
    """Return a routing instruction. This function never executes an action.

    observed_at, sent_at and now share a monotonic time base. Observation age
    includes preprocessing and queueing. Request age also includes any waiting
    after response arrival; it is not the network round-trip latency.
    The caller owns the active observation/goal versions and execution checks.
    Recheck at controller submission to avoid check/use races.
    """
    if not finite_number(now):
        return "invalid_clock"
    if any(type(v) is not int or v < 0 for v in (observation_id, goal_version)):
        return "invalid_context"
    if now < request.sent_at:
        return "invalid_clock"
    if goal_version != request.goal_version:
        return "replan"
    if (observation_id != request.observation_id
            or now - request.observed_at > request.max_observation_age):
        return "observe"
    if now - request.sent_at > request.max_request_age:
        return "request_expired"
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return "invalid_response"
    choice = answer.get("choice")
    probs = answer.get("probabilities")
    confidence = answer.get("confidence")
    if not isinstance(choice, str) or choice not in request.options:
        return "invalid_response"
    if not isinstance(probs, dict) or set(probs) != request.options:
        return "invalid_response"
    if not all(finite_number(p) and 0 <= p <= 1 for p in probs.values()):
        return "invalid_response"
    if not math.isclose(math.fsum(probs.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        return "invalid_response"
    if probs[choice] < max(probs.values()) - 1e-9:
        return "invalid_response"
    if not finite_number(confidence) or not 0 <= confidence <= 1:
        return "invalid_response"
    # These paths cannot submit a physical action.
    if choice == "blocked":
        return "replan"
    if choice == "done":
        return "handoff" if goal_verified is True else "verify_goal"
    if choice == "observe":
        # Here observe means reading sensors, not moving a camera/robot.
        return "observe"
    if execution_allowed is not True:
        return "reject_execution"
    if confidence < request.confidence_floor:
        return "replan"
    return "execute:" + choice


def selective_metrics(confidences, correct, threshold):
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("nonempty aligned samples required")
    if not finite_number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if any(not finite_number(c) or not 0 <= c <= 1 for c in confidences):
        raise ValueError("invalid confidence")
    if any(type(y) is not bool for y in correct):
        raise ValueError("boolean correctness labels required")
    accepted = [y for c, y in zip(confidences, correct) if c >= threshold]
    return {"accept_rate": len(accepted) / len(correct),
            "selective_error": ((len(accepted) - sum(accepted)) / len(accepted)) if accepted else None}


OPTIONS = frozenset({"approach", "observe", "done", "blocked"})
CONTEXT = RequestContext(42, 10.0, 10.125, OPTIONS)


def fixture(choice="approach", confidence=0.9):
    return {"type": "choice", "choice": choice, "confidence": confidence,
            "probabilities": {k: (0.97 if k == choice else 0.01) for k in sorted(OPTIONS)}}


def decide(answer=None, **overrides):
    args = dict(observation_id=42, goal_version=1, now=10.25,
                execution_allowed=True, goal_verified=False)
    args.update(overrides)
    return route(fixture() if answer is None else answer, CONTEXT, **args)


class GateTests(unittest.TestCase):
    def test_valid_action(self):
        self.assertEqual(decide(), "execute:approach")

    def test_freshness(self):
        for args in ({"observation_id": 43}, {"now": 10.6}):
            self.assertEqual(decide(**args), "observe")
        self.assertEqual(decide(now=float("nan")), "invalid_clock")
        self.assertEqual(decide(now=9.9), "invalid_clock")

    def test_fast_response_can_have_old_observation(self):
        request = replace(CONTEXT, observed_at=9.0)
        result = route(fixture(), request, observation_id=42, goal_version=1,
                       now=10.25, execution_allowed=True, goal_verified=False)
        self.assertEqual(result, "observe")

    def test_request_budget_is_separate_from_observation_age(self):
        request = replace(CONTEXT, max_request_age=0.125)
        args = dict(observation_id=42, goal_version=1,
                    execution_allowed=True, goal_verified=False)
        self.assertEqual(route(fixture(), request, now=10.25, **args), "execute:approach")
        self.assertEqual(route(fixture(), request, now=10.375, **args), "request_expired")

    def test_goal_change_invalidates_semantics(self):
        self.assertEqual(decide(goal_version=2), "replan")

    def test_version_types_and_nonfinite_clock(self):
        self.assertEqual(decide(observation_id=42.0), "invalid_context")
        self.assertEqual(decide(goal_version=True), "invalid_context")
        self.assertEqual(decide(now=10 ** 1000), "invalid_clock")

    def test_invalid_answers(self):
        answers = [[], {}, {**fixture(), "choice": "invented"},
                   {**fixture(), "confidence": True},
                   {**fixture(), "confidence": float("nan")},
                   {**fixture(), "confidence": 10 ** 1000},
                   {**fixture(), "choice": "observe"}]
        for values in ({"approach": 1.0}, {k: 0.5 for k in OPTIONS},
                       {k: float("nan") for k in OPTIONS}, {k: True for k in OPTIONS}):
            answers.append({**fixture(), "probabilities": values})
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertEqual(decide(answer), "invalid_response")

    def test_confidence_is_not_top_probability(self):
        self.assertEqual(decide(fixture(confidence=0.3)), "replan")

    def test_execution_gate(self):
        for allowed in (False, None, 1, "yes"):
            self.assertEqual(decide(execution_allowed=allowed), "reject_execution")

    def test_control_handoffs(self):
        self.assertEqual(decide(fixture("done")), "verify_goal")
        self.assertEqual(decide(fixture("done"), goal_verified=True), "handoff")
        self.assertEqual(decide(fixture("done"), goal_verified="yes"), "verify_goal")
        self.assertEqual(decide(fixture("blocked")), "replan")
        self.assertEqual(decide(fixture("observe"), execution_allowed=False), "observe")

    def test_invalid_configuration(self):
        for changes in ({"max_observation_age": 0}, {"max_request_age": -1},
                        {"confidence_floor": 2}, {"sent_at": float("inf")},
                        {"observed_at": 11.0}, {"observation_id": True},
                        {"goal_version": -1}, {"options": set(OPTIONS)},
                        {"options": frozenset({""})}):
            with self.assertRaises(ValueError):
                replace(CONTEXT, **changes)

    def test_recheck_after_queueing_before_execution(self):
        self.assertEqual(decide(now=10.25), "execute:approach")
        # The same answer must not be trusted after waiting in a command queue.
        self.assertEqual(decide(now=10.625), "observe")

    def test_selective_metrics(self):
        self.assertEqual(selective_metrics([0.9, 0.8, 0.2], [True, False, True], 0.8),
                         {"accept_rate": 2 / 3, "selective_error": 0.5})
        self.assertIsNone(selective_metrics([0.1], [True], 0.8)["selective_error"])
        with self.assertRaises(ValueError):
            selective_metrics([0.9], [], 0.8)


def example_results():
    common = dict(observation_id=42, goal_version=1, now=10.25,
                  execution_allowed=True, goal_verified=False)
    reported_planner_cost = 0.387810
    reported_selector_cost = 0.006192942
    reported_total = math.fsum((reported_planner_cost, reported_selector_cost))
    return {
        "scope": "Offline protocol fixtures and arithmetic on reported values; no Jev or robot calls.",
        "freshness_example_seconds": {
            "fresh": {"observation_age": 0.25, "request_age": 0.125,
                      "route": route(fixture(), CONTEXT, **common)},
            "old_observation_fast_response": {
                "observation_age": 1.25, "request_age": 0.125,
                "route": route(fixture(), replace(CONTEXT, observed_at=9.0), **common)},
        },
        "synthetic_metrics": selective_metrics([0.9, 0.8, 0.2], [True, False, True], 0.8),
        "synthetic_costs": {"success_conditioned_mean": (0.1 + 0.2) / 2,
                            "all_attempts_per_success": (0.1 + 0.2 + 0.6) / 2},
        "jev_mobile_recomputed": {"time_reduction": 1 - 132.67 / 197.21,
                                  "cost_reduction": 1 - 0.072744 / 0.273694,
                                  "source": "https://arxiv.org/html/2609.30186v1#S6"},
        "community_drawer_recomputed": {
            "scope": "One reported successful development run; not a new measurement.",
            "requests": 12 + 63,
            "reported_model_cost_usd": reported_total,
            "jev_fraction_of_model_cost": reported_selector_cost / reported_total,
            "source": "https://github.com/FBddcz/embodied-jev/blob/"
                      "f08de2e4e20d6cd69fea9c57ac1062c3ef510f1e/"
                      "docs/results/libero-vision/COSTS.json",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write fixture results as JSON")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GateTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    output = json.dumps(example_results(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
