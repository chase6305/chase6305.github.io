#!/usr/bin/env python3
"""Small VLA arithmetic examples; Python 3.8+, standard library only.

This is not a policy implementation or a robot controller. The quantizer uses
K equal-width intervals, not OpenVLA's exact digitize boundary convention.
Flow examples use analytic fields to check direction and probability transport;
no field is learned from training data.
"""

import argparse
import json
import math
from statistics import NormalDist
import unittest


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a finite number")
    if not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def uniform_quantize(value, low=-0.1, high=0.1, bins=256):
    """Return a K-bin index and midpoint reconstruction after clipping."""
    value, low, high = (finite(x, n) for x, n in
                        [(value, "value"), (low, "low"), (high, "high")])
    positive_int(bins, "bins")
    if high <= low:
        raise ValueError("high must exceed low")
    clipped = min(high, max(low, value))
    width = (high - low) / bins
    index = min(bins - 1, int(math.floor((clipped - low) / width)))
    center = low + (index + 0.5) * width
    return {"index": index, "reconstructed_m": center,
            "bin_width_m": width, "in_range_error_bound_m": width / 2,
            "clipped": clipped != value}


def symmetric_risk(prediction):
    """Expected scalar L1 and L2 loss for equally likely -1 and +1."""
    p = finite(prediction, "prediction")
    return {"l1": (abs(p + 1) + abs(p - 1)) / 2,
            "l2": ((p + 1) ** 2 + (p - 1) ** 2) / 2}


def oracle_flow(target=2.0, noise=-1.0, steps=10, wrong_sign=False):
    """Check s=1 noise -> s=0 data using an oracle, not a learned field."""
    target, noise = finite(target, "target"), finite(noise, "noise")
    positive_int(steps, "steps")
    velocity = noise - target
    delta_s = (1 if wrong_sign else -1) / steps
    value = noise
    path = [value]
    for _ in range(steps):
        value += delta_s * velocity
        path.append(value)
    return {"velocity": velocity, "delta_s": delta_s, "path": path,
            "endpoint": value, "target": target}


def action_clocks(horizon=50, frequency_hz=50.0, execute=25, flow_steps=10):
    """Ideal timing excludes inference, network, scheduling and reset time."""
    positive_int(horizon, "horizon")
    positive_int(execute, "execute")
    positive_int(flow_steps, "flow_steps")
    frequency_hz = finite(frequency_hz, "frequency_hz")
    if frequency_hz <= 0 or execute > horizon:
        raise ValueError("frequency must be positive and execute <= horizon")
    return {"action_period_s": 1 / frequency_hz,
            "chunk_coverage_s": horizon / frequency_hz,
            "last_target_offset_s": (horizon - 1) / frequency_hz,
            "prefix_duration_s": execute / frequency_hz,
            "ideal_replans_per_s": frequency_hz / execute,
            "flow_steps_per_chunk": flow_steps}


def mixture_probabilities(counts, exponent=0.43):
    """Group sampling probability proportional to count ** exponent."""
    exponent = finite(exponent, "exponent")
    if not counts or exponent < 0:
        raise ValueError("need nonempty counts and a nonnegative exponent")
    for count in counts:
        positive_int(count, "count")
    # Log-space rescaling avoids overflow for large counts/exponents.
    logs = [exponent * math.log(count) for count in counts]
    maximum = max(logs)
    weights = [math.exp(value - maximum) for value in logs]
    total = sum(weights)
    return [weight / total for weight in weights]


def bimodal_velocity(value, time):
    """Exact conditional mean velocity for A in {-1,+1}, s=1 noise -> 0 data.

    This analytic vector field is not a neural network or trained controller.
    Gaussian noise has unit variance and both action modes have equal mass.
    """
    value, time = finite(value, "value"), finite(time, "time")
    if not 0 < time <= 1:
        raise ValueError("time must lie in (0, 1]")
    conditional_action = math.tanh((1 - time) * value / (time * time))
    return (value - conditional_action) / time


def bimodal_flow(noise, steps=1000, end_time=0.02):
    """Euler trajectory, ending before the singular discrete-data boundary."""
    value = finite(noise, "noise")
    positive_int(steps, "steps")
    end_time = finite(end_time, "end_time")
    if not 0 < end_time < 1:
        raise ValueError("end_time must lie in (0, 1)")
    delta = (end_time - 1) / steps
    path = [(1.0, value)]
    for i in range(steps):
        time = 1 + i * delta
        value += delta * bimodal_velocity(value, time)
        path.append((1 + (i + 1) * delta, value))
    return path


def mixture_cdf(value, time):
    """Exact CDF of the interpolated two-Gaussian marginal distribution."""
    value, time = finite(value, "value"), finite(time, "time")
    if not 0 < time <= 1:
        raise ValueError("time must lie in (0, 1]")
    gaussian = NormalDist()
    return 0.5 * (gaussian.cdf((value - (1 - time)) / time)
                  + gaussian.cdf((value + (1 - time)) / time))


def sparse_episode_returns(length, success, failure_penalty=1000):
    """Undiscounted returns: -1 per nonterminal step, 0 / -C at terminal.

    Raw constructed values; RECAP additionally normalizes per task and learns
    a distributional critic. This function does neither of those operations.
    """
    positive_int(length, "length")
    if not isinstance(success, bool):
        raise ValueError("success must be boolean")
    failure_penalty = finite(failure_penalty, "failure_penalty")
    if failure_penalty <= 0:
        raise ValueError("failure_penalty must be positive")
    terminal_reward = 0 if success else -failure_penalty
    return [terminal_reward - (length - 1 - t) for t in range(length)]


def n_step_advantage(rewards, value_now, value_next, terminal=False):
    """Constructed undiscounted estimate; never bootstrap past a terminal."""
    value_now, value_next = finite(value_now, "value_now"), finite(value_next, "value_next")
    if not isinstance(terminal, bool) or not rewards:
        raise ValueError("need nonempty rewards and boolean terminal")
    rewards = [finite(r, "reward") for r in rewards]
    return sum(rewards) + (0 if terminal else value_next) - value_now


def report():
    probabilities = mixture_probabilities([1000000, 10000])
    return {
        "scope": "Constructed arithmetic examples; no model or robot evaluation",
        "uniform_quantization": uniform_quantize(0.02),
        "symmetric_actions": {str(p): symmetric_risk(p) for p in [-1, 0, 1]},
        "oracle_flow_correct_endpoint": oracle_flow()["endpoint"],
        "oracle_flow_wrong_endpoint": oracle_flow(wrong_sign=True)["endpoint"],
        "bimodal_flow_endpoints": {
            str(z): bimodal_flow(z)[-1][1] for z in [-1.0, -0.2, 0.2, 1.0]
        },
        "reward_and_advantage": {
            "success_returns_4_steps": sparse_episode_returns(4, True),
            "failure_returns_4_steps": sparse_episode_returns(4, False),
            "ten_step_advantage": n_step_advantage([-1] * 10, -100, -70),
        },
        "clocks": action_clocks(),
        "sampling": {"probabilities": probabilities,
                     "weight_ratio": probabilities[0] / probabilities[1]},
    }


class Checks(unittest.TestCase):
    def test_quantizer_accuracy_over_interval(self):
        # Check many independent in-range values against the analytic bound.
        for bins in [1, 2, 16, 256]:
            for i in range(2001):
                value = -0.1 + 0.2 * i / 2000
                out = uniform_quantize(value, bins=bins)
                self.assertTrue(0 <= out["index"] < bins)
                self.assertLessEqual(abs(out["reconstructed_m"] - value),
                                     out["in_range_error_bound_m"] + 1e-15)
        out = uniform_quantize(0.02)
        self.assertEqual(out["index"], 153)
        self.assertAlmostEqual(out["reconstructed_m"], 0.019921875)
        self.assertTrue(uniform_quantize(1)["clipped"])

    def test_l2_mean_and_nonunique_l1(self):
        for p in [-1, -0.5, 0, 0.5, 1]:
            self.assertAlmostEqual(symmetric_risk(p)["l1"], 1)
            self.assertAlmostEqual(symmetric_risk(p)["l2"], 1 + p * p)
        self.assertGreater(symmetric_risk(2)["l1"], 1)

    def test_flow_endpoints_and_direction(self):
        for target, noise in [(2, -1), (-2, 3), (0, 0), (0.2, -0.9)]:
            for steps in [1, 2, 10, 100]:
                out = oracle_flow(target, noise, steps)
                self.assertAlmostEqual(out["endpoint"], target)
                self.assertEqual(len(out["path"]), steps + 1)
                bad = oracle_flow(target, noise, steps, wrong_sign=True)
                self.assertAlmostEqual(bad["endpoint"], 2 * noise - target)

    def test_clocks_independent_of_sampling_steps(self):
        for n in [1, 5, 10, 20]:
            out = action_clocks(flow_steps=n)
            self.assertEqual(out["chunk_coverage_s"], 1)
            self.assertEqual(out["last_target_offset_s"], 0.98)
            self.assertEqual(out["prefix_duration_s"], 0.5)
            self.assertEqual(out["ideal_replans_per_s"], 2)
        slower = action_clocks(frequency_hz=20, execute=16)
        self.assertEqual(slower["prefix_duration_s"], 0.8)
        self.assertEqual(slower["chunk_coverage_s"], 2.5)

    def test_mixture_invariants(self):
        self.assertEqual(mixture_probabilities([100, 1], 0), [0.5, 0.5])
        p = mixture_probabilities([1000000, 10000])
        self.assertAlmostEqual(sum(p), 1)
        self.assertAlmostEqual(p[0] / p[1], 100 ** 0.43)
        q = mixture_probabilities([10000000, 100000])
        for a, b in zip(p, q):
            self.assertAlmostEqual(a, b)

    def test_bimodal_flow_preserves_probability_quantiles(self):
        # CDF conservation checks the ODE against the independently known
        # marginal distribution, rather than merely checking its own update.
        for noise in [-2.0, -1.0, -0.2, 0.2, 1.0, 2.0]:
            path = bimodal_flow(noise, steps=4000)
            quantile = NormalDist().cdf(noise)
            for time, value in path[::200]:
                self.assertAlmostEqual(mixture_cdf(value, time), quantile, delta=0.0002)
            end_time, endpoint = path[-1]
            self.assertGreater(noise * endpoint, 0)
            self.assertGreater(abs(endpoint), 0.94)
            self.assertAlmostEqual(mixture_cdf(endpoint, end_time), quantile, delta=0.0002)
        # Exactly zero is a zero-probability Gaussian event and stays on the
        # symmetry boundary; it must not be presented as a sampled mode.
        self.assertEqual(bimodal_flow(0)[-1][1], 0)
        left, right = bimodal_flow(-0.2)[-1][1], bimodal_flow(0.2)[-1][1]
        self.assertAlmostEqual(left, -right)

    def test_rewards_and_terminal_bootstrap(self):
        self.assertEqual(sparse_episode_returns(4, True), [-3, -2, -1, 0])
        self.assertEqual(sparse_episode_returns(4, False), [-1003, -1002, -1001, -1000])
        self.assertEqual(n_step_advantage([-1] * 10, -100, -70), 20)
        # A terminal outcome excludes any value assigned after termination.
        for next_value in [-1000, 0, 999]:
            self.assertEqual(n_step_advantage([-1, 0], -5, next_value, True), 4)
        # Failure penalties must be chosen relative to the task time limit;
        # in this bounded example no early failure beats a successful rollout.
        for length in range(1, 101):
            self.assertLess(sparse_episode_returns(length, False)[0],
                            sparse_episode_returns(100, True)[0])

    def test_invalid_inputs(self):
        calls = [
            lambda: uniform_quantize(float("nan")),
            lambda: uniform_quantize(0, low=1, high=0),
            lambda: uniform_quantize(0, bins=True),
            lambda: oracle_flow(steps=0),
            lambda: action_clocks(frequency_hz=0),
            lambda: action_clocks(execute=51),
            lambda: action_clocks(frequency_hz=float("inf")),
            lambda: mixture_probabilities([]),
            lambda: mixture_probabilities([0]),
            lambda: mixture_probabilities([1], -1),
            lambda: bimodal_velocity(1, 0),
            lambda: bimodal_flow(1, end_time=1),
            lambda: sparse_episode_returns(0, True),
            lambda: sparse_episode_returns(4, 1),
            lambda: sparse_episode_returns(4, False, -1),
            lambda: n_step_advantage([], 0, 1),
            lambda: n_step_advantage([float("nan")], 0, 1),
        ]
        for call in calls:
            with self.assertRaises(ValueError):
                call()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    print(json.dumps(report(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
