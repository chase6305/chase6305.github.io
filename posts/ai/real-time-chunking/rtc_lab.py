#!/usr/bin/env python3
"""Deterministic RTC teaching checks; NumPy only, no learned policy or robot.

Optional --source-model extracts get_prefix_weights from the local reference
source and compares its array math using NumPy in place of jax.numpy.
"""

import argparse
import ast
from pathlib import Path

import numpy as np


def prefix_weights(delay, overlap, horizon, schedule="exp"):
    if not 0 <= delay <= horizon or not 0 <= overlap <= horizon:
        raise ValueError("delay and overlap must be within the horizon")
    delay = min(delay, overlap)
    index = np.arange(horizon)
    if schedule == "zeros":
        weight = (index < delay).astype(float)
    elif schedule == "ones":
        weight = np.ones(horizon)
    elif schedule in ("linear", "exp"):
        weight = np.clip((overlap - index) / (overlap - delay + 1), 0, 1)
        if schedule == "exp":
            weight *= np.expm1(weight) / np.expm1(1.0)
    else:
        raise ValueError(f"unknown schedule: {schedule}")
    return np.where(index < overlap, weight, 0.0)


def simulate_iteration(old, new, delay, advance):
    """One serialized simulated iteration; arrays already share a time origin."""
    if old.shape != new.shape:
        raise ValueError("old and new must have the same shape")
    horizon = len(new)
    if not (0 <= delay <= advance <= horizon - delay and advance > 0):
        raise ValueError("requires d <= s <= H-d and s > 0")
    executed = np.concatenate((old[:delay], new[delay:advance]))
    shifted = np.concatenate((new[advance:], np.zeros_like(new[:advance])))
    return executed, shifted


def live_suffix(chunk, observation_step, next_command_step, *, committed_until_step=None):
    """Same-grid toy installer; commitment boundary is exclusive."""
    if observation_step > next_command_step:
        raise ValueError("observation time is in the future")
    if committed_until_step is None:
        committed_until_step = next_command_step
    index = max(next_command_step, committed_until_step) - observation_step
    if not 0 <= index < len(chunk):
        raise ValueError("no replaceable suffix or its time origin is in the future")
    return chunk[index:]


def candidate_suffix(chunk, *, result_session, active_session, request_id,
                     installed_request_id, observation_step, next_command_step,
                     committed_until_step):
    """Pure teaching decision; a real caller must check and install under one lock."""
    if result_session != active_session:
        raise ValueError("result belongs to a previous control session")
    if request_id <= installed_request_id:
        raise ValueError("duplicate or out-of-order result")
    return live_suffix(chunk, observation_step, next_command_step,
                       committed_until_step=committed_until_step)


def check_session_and_freshness():
    chunk = np.arange(200, 212)
    common = dict(active_session=8, installed_request_id=2, observation_step=0,
                  next_command_step=3, committed_until_step=4)
    # Old session's high ID and valid-looking time must not defeat the session guard.
    for session, request in ((7, 99), (8, 2), (8, 1)):
        try:
            candidate_suffix(chunk, result_session=session, request_id=request, **common)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted old-session, duplicate, or out-of-order result")
    actual = candidate_suffix(chunk, result_session=8, request_id=3, **common)
    np.testing.assert_array_equal(actual, [204, 205, 206, 207, 208, 209, 210, 211])
    # Equal 80 ms request-to-install time, but different sensor age at request time.
    horizon_ms, wait_ms, old_queue_ms = 16 * 20, 80, 200
    remaining = [horizon_ms - age - wait_ms for age in (0, 200)]
    assert remaining == [240, 40] and old_queue_ms - wait_ms == 120
    assert horizon_ms - 240 - wait_ms == 0  # fresh suffix completely exhausted
    print("session guard: old-session ID 99 rejected; active-session ID 3 accepted")
    print("freshness: equal 80 ms waits leave 240/40 ms of prediction; queue margin stays 120 ms")


def check_variable_latency():
    # Serialized requests, same time grid, no request/installation overhead.
    horizon = 8
    def feasible(previous_delay, current_delay, spacing):
        return spacing >= previous_delay and spacing + current_delay <= horizon
    assert feasible(2, 5, 2)  # an isolated delay > H/2 may be covered
    assert not feasible(5, 5, 5)  # sustained delay cannot be hidden
    assert feasible(5, 2, 5)
    assert not feasible(2, 5, 1)  # would overlap inference requests
    assert not feasible(2, 5, 4)  # waiting too long consumes the coverage margin
    for previous in range(9):
        for current in range(9):
            exists = any(feasible(previous, current, spacing) for spacing in range(9))
            assert exists == (previous + current <= horizon)
    print("variable latency: H=8 covers delays 2 then 5 at spacing 2; two consecutive 5s fail")


def conditioned_input(action, noise, time, delays):
    """action/noise: [B,H,D], time/delays: [B]."""
    prefix = np.arange(action.shape[1])[None, :] < delays[:, None]
    token_time = np.where(prefix, 1.0, time[:, None])
    mixed = (1 - token_time[..., None]) * noise + token_time[..., None] * action
    return mixed, token_time, ~prefix


def source_style_loss(prediction, target, suffix):
    squared = (prediction - target) ** 2
    return np.sum(squared * suffix[..., None]) / (np.sum(suffix) + 1e-8)


def check_timeline():
    old = np.arange(100, 108)
    new = np.arange(200, 208)
    executed, shifted = simulate_iteration(old, new, delay=2, advance=3)
    np.testing.assert_array_equal(executed, [100, 101, 202])
    np.testing.assert_array_equal(shifted, [203, 204, 205, 206, 207, 0, 0, 0])
    first, buffer = simulate_iteration(old, new, delay=2, advance=2)
    second, _ = simulate_iteration(buffer, np.arange(300, 308), delay=2, advance=2)
    np.testing.assert_array_equal(first, [100, 101])
    np.testing.assert_array_equal(second, [202, 203])
    print("s=d=2: generated chunk enters execution in the next iteration")
    # Every valid (d,s) executes exactly s commands and advances the origin by s.
    for delay in range(5):
        for advance in range(max(1, delay), 8 - delay + 1):
            actual, buffer = simulate_iteration(old, new, delay, advance)
            assert len(actual) == advance and len(buffer) == 8
            np.testing.assert_array_equal(actual[:delay], old[:delay])
            np.testing.assert_array_equal(actual[delay:], new[delay:advance])
            np.testing.assert_array_equal(buffer[: 8 - advance], new[advance:])
    np.testing.assert_array_equal(live_suffix(new, 100, 106), [206, 207])
    for now in (99, 108, 109):
        try:
            live_suffix(new, 100, now)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a result with no usable current action")
    print("timeline: old[0:2] + new[2:3]; late result starts at actual index 6")
    longer = np.arange(200, 212)
    suffix = live_suffix(longer, 100, 106, committed_until_step=108)
    np.testing.assert_array_equal(suffix, [208, 209, 210, 211])
    # Commands through 107 have already been committed and survive installation.
    queued = np.arange(100, 112)
    installed = np.concatenate((queued[:8], suffix))
    np.testing.assert_array_equal(installed[:8], queued[:8])
    np.testing.assert_array_equal(installed[8:], longer[8:])
    for now, boundary in ((106, 112), (106, 113), (99, 108)):
        try:
            live_suffix(longer, 100, now, committed_until_step=boundary)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an unusable chunk or future observation")
    print("queued commands through 107 survive; replacement starts at chunk index 8")


def check_committed_prefix():
    """Toy command adapter, not dynamics or a Kinetix implementation feature."""
    raw = np.array([0., .4, .8])
    previous = 0.
    committed = []
    for target in raw:
        previous += np.clip(target - previous, -.1, .1)
        committed.append(previous)
    committed = np.asarray(committed)
    np.testing.assert_allclose(committed, [0., .1, .2])
    assert np.all(np.abs(np.diff(np.r_[0., committed])) <= .1 + 1e-12)
    assert not np.allclose(raw, committed)
    # Same policy and controller space, same timestamps, all three committed.
    conditioned, _, suffix = conditioned_input(
        committed[None, :, None], np.ones((1, 3, 1)),
        np.array([.2]), np.array([3]))
    np.testing.assert_allclose(conditioned[0, :, 0], committed)
    assert not suffix.any()
    print(f"committed prefix: raw={raw.tolist()}, rate-limited={committed.round(2).tolist()}")


def check_masks(source_model=None):
    for schedule in ("linear", "exp", "zeros", "ones"):
        print(f"{schedule:>6}: {prefix_weights(2, 5, 8, schedule).round(3)}")
    np.testing.assert_allclose(prefix_weights(2, 5, 8, "linear"),
                               [1, 1, .75, .5, .25, 0, 0, 0])
    np.testing.assert_array_equal(prefix_weights(2, 5, 8, "zeros"),
                                  [1, 1, 0, 0, 0, 0, 0, 0])
    reference = None
    if source_model:
        tree = ast.parse(Path(source_model).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "get_prefix_weights")
        function.returns = None
        for arg in function.args.args:
            arg.annotation = None
        isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {"jnp": np}
        exec(compile(isolated, str(source_model), "exec"), namespace)
        reference = namespace["get_prefix_weights"]
    count = 0
    for horizon in (1, 8, 16):
        for delay in range(horizon + 1):
            for overlap in range(horizon + 1):
                for schedule in ("linear", "exp", "zeros", "ones"):
                    actual = prefix_weights(delay, overlap, horizon, schedule)
                    assert np.all((actual >= 0) & (actual <= 1 + 1e-12))
                    assert np.all(np.diff(actual) <= 1e-12)
                    assert np.all(actual[overlap:] == 0)
                    if reference:
                        np.testing.assert_allclose(actual, reference(
                            delay, overlap, horizon, schedule), atol=1e-7)
                    count += 1
    print(f"mask boundaries: {count} cases; source parity: {bool(reference)}")


def check_vjp():
    # Scalar velocity v(x)=3x, tau=.5: F(x)=2.5x includes the identity branch.
    tau, velocity_jacobian = .5, 3.
    endpoint_jacobian = 1 + (1 - tau) * velocity_jacobian
    state, target = 0., 1.
    residual = target - endpoint_jacobian * state
    correct_scalar = endpoint_jacobian * residual
    np.testing.assert_allclose(correct_scalar, 2.5)
    np.testing.assert_allclose(residual + (1 - tau) * velocity_jacobian * residual,
                               correct_scalar)
    epsilon = 1e-6
    energy_plus = .5 * (target - endpoint_jacobian * (state + epsilon))**2
    energy_minus = .5 * (target - endpoint_jacobian * (state - epsilon))**2
    np.testing.assert_allclose(correct_scalar, -(energy_plus - energy_minus) / (2 * epsilon))
    pseudo_inverse_result = np.linalg.pinv(np.array([[endpoint_jacobian]])) @ [residual]
    np.testing.assert_allclose(pseudo_inverse_result, [.4])
    assert correct_scalar != velocity_jacobian * residual
    assert correct_scalar != residual
    print("Endpoint VJP: 2.5; velocity-only: 3; detached velocity: 1; pseudoinverse: 0.4")
    coupled = np.array([[1.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(coupled.T @ np.array([1.0, 0.0]), [1.0, 1.0])
    rng = np.random.default_rng(42)
    horizon = 8
    matrix = np.eye(horizon) + .08 * rng.normal(size=(horizon, horizon))
    bias = rng.normal(size=horizon)
    x, old = rng.normal(size=(2, horizon))
    weights = prefix_weights(2, 5, horizon)

    def energy(state):
        residual = old - (matrix @ state + bias)
        return .5 * np.sum(weights * residual**2)

    correction = matrix.T @ (weights * (old - (matrix @ x + bias)))
    eps = 1e-6
    finite_diff = np.array([
        (energy(x + eps * axis) - energy(x - eps * axis)) / (2 * eps)
        for axis in np.eye(horizon)
    ])
    np.testing.assert_allclose(correction, -finite_diff, atol=2e-8)
    wrong = matrix.T @ (weights**2 * (old - (matrix @ x + bias)))
    assert np.linalg.norm(wrong - correction) > .01
    # Coupling in F means even coordinates with zero direct mask can be updated.
    assert np.linalg.norm(correction[5:]) > 0
    before, after = energy(x), energy(x + .01 * correction)
    assert after < before
    print(f"VJP: finite difference agrees; energy {before:.6f} -> {after:.6f}")


def check_training():
    rng = np.random.default_rng(7)
    action, noise = rng.normal(size=(2, 2, 8, 3))
    x, token_time, suffix = conditioned_input(
        action, noise, np.array([.25, .75]), np.array([2, 4]))
    np.testing.assert_array_equal(x[~suffix], action[~suffix])
    np.testing.assert_array_equal(token_time[~suffix], 1)
    target = action - noise
    prediction = target + 1
    baseline = source_style_loss(prediction, target, suffix)
    prediction[~suffix] += 1000
    assert np.isclose(source_style_loss(prediction, target, suffix), baseline)
    prediction[suffix] += 1
    assert source_style_loss(prediction, target, suffix) > baseline
    assert np.isclose(baseline, action.shape[-1])
    # At d=0, source-style loss sums D coordinates per token; base FM averages D.
    all_suffix = np.ones((2, 8), dtype=bool)
    source_loss = source_style_loss(target + 1, target, all_suffix)
    mean_loss = np.mean(((target + 1) - target) ** 2)
    assert np.isclose(source_loss / mean_loss, action.shape[-1])
    # Different suffix lengths distinguish token averaging from sample averaging.
    uneven_mask = np.arange(8)[None, :] >= np.array([0, 4])[:, None]
    target_zero = np.zeros((2, 8, 1))
    pred_uneven = np.ones_like(target_zero)
    pred_uneven[1] = 2
    token_average = source_style_loss(pred_uneven, target_zero, uneven_mask)
    sample_average = np.mean([
        np.mean(pred_uneven[b, uneven_mask[b]] ** 2) for b in range(2)
    ])
    assert np.isclose(token_average, 2) and np.isclose(sample_average, 2.5)
    print("unequal suffix lengths: token-average loss=2, sample-average loss=2.5")
    # A toy nonzero prefix velocity exposes the last-update detail in the source.
    state = noise[0].copy()
    for _ in range(5):
        state[:2] = action[0, :2]
        np.testing.assert_array_equal(state[:2], action[0, :2])
        state += .2 * np.ones_like(state)
    np.testing.assert_allclose(state[:2] - action[0, :2], .2)
    print("training: clean prefix, per-token time, suffix-only loss, D=3 scaling verified")
    print("prefix is fixed at network input; final toy output prefix can still drift")


def check_guidance_grid():
    time = np.arange(1, 5) / 5
    raw = (time**2 + (1 - time)**2) / (time * (1 - time))
    np.testing.assert_allclose(raw, [4.25, 13 / 6, 13 / 6, 4.25])
    five_step = np.r_[5.0, np.minimum(raw, 5.0)]
    np.testing.assert_allclose(five_step, [5, 4.25, 13 / 6, 13 / 6, 4.25])
    print(f"five-step guidance (beta=5): {five_step.round(6)}")


def check_episode_targets():
    # The source masks labels starting at the first done, including that action.
    action = np.array([10, 11, 12, 20, 21, 22, 23, 24], dtype=float)[:, None]

    def prepare(done):
        first = int(np.argmax(done)) if np.any(done) else len(done)
        return np.where(np.arange(len(done))[:, None] >= first, 0, action)

    done = np.array([False, False, True, False, False, False, False, False])
    target = prepare(done)
    np.testing.assert_array_equal(target[:, 0], [10, 11, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(prepare(np.zeros(8, dtype=bool)), action)
    np.testing.assert_array_equal(prepare(np.ones(8, dtype=bool)), np.zeros_like(action))
    last_done = np.zeros(8, dtype=bool)
    last_done[-1] = True
    np.testing.assert_array_equal(prepare(last_done)[:, 0], [10, 11, 12, 20, 21, 22, 23, 0])
    # Zero labels still count in the suffix objective: they are not a loss mask.
    noise = np.ones_like(target)
    velocity_target = target - noise
    prediction = velocity_target.copy()
    prediction[2:] += 1
    suffix = np.arange(8)[None, :] >= 2
    loss = source_style_loss(prediction[None], velocity_target[None], suffix)
    assert np.isclose(loss, 1)
    print("episode boundary: terminal action included in zeroing; zero targets still incur suffix loss")


def check_delay_coverage():
    maximum = 5  # exclusive upper bound, as in simulated_delay=5
    probability = np.exp(np.arange(maximum)[::-1])
    probability /= probability.sum()
    positions = np.arange(8)
    coverage = np.array([probability[: min(i + 1, maximum)].sum() for i in positions])
    analytic = (1 - np.exp(-np.minimum(positions + 1, maximum))) / (1 - np.exp(-maximum))
    np.testing.assert_allclose(coverage, analytic)
    np.testing.assert_allclose(coverage[:4], [.6364, .8705, .9567, .9883], atol=5e-5)
    np.testing.assert_allclose(coverage[4:], 1)
    uniform = np.ones(maximum) / maximum
    uniform_coverage = [uniform[: min(i + 1, maximum)].sum() for i in positions]
    np.testing.assert_allclose(uniform_coverage, [.2, .4, .6, .8, 1, 1, 1, 1])
    print(f"direct supervision coverage (%): {(coverage * 100).round(2)}")


def check_action_metric():
    # One position channel, with both its center and scale converted to mm.
    center_m, scale_m = .3, .1
    old_m, new_m = .42, .41
    old_z = (old_m - center_m) / scale_m
    new_z = (new_m - center_m) / scale_m
    normalized_mm = ((1000 * old_m - 1000 * center_m) / (1000 * scale_m)
                     - (1000 * new_m - 1000 * center_m) / (1000 * scale_m))
    residual_z = old_z - new_z
    np.testing.assert_allclose(residual_z, .1)
    np.testing.assert_allclose(normalized_mm, residual_z)
    wrong_units = (1000 * (old_m - new_m)) / scale_m
    np.testing.assert_allclose((wrong_units / residual_z)**2, 1_000_000)

    # Flatten [H=2,D=2]; each time weight is broadcast to both channels.
    matrix = np.array([[1., .2, .1, 0.], [.3, 1., 0., .1],
                       [.1, 0., 1., .2], [0., .1, .3, 1.]])
    state = np.array([.2, -.1, .3, .4])
    target_z = np.array([.5, .2, -.2, .1])
    scales = np.tile([.1, 2.], 2)
    centers = np.tile([.3, -.2], 2)
    weights = np.repeat([1., .25], 2)
    target_a = centers + scales * target_z
    residual_a = target_a - (centers + scales * (matrix @ state))

    def physical_energy(x):
        residual = target_a - (centers + scales * (matrix @ x))
        return .5 * np.sum(weights * residual**2)

    correction = matrix.T @ (scales * weights * residual_a)
    equivalent = matrix.T @ (scales**2 * weights * (target_z - matrix @ state))
    np.testing.assert_allclose(correction, equivalent)
    eps = 1e-6
    gradient = np.array([(physical_energy(state + eps * axis)
                          - physical_energy(state - eps * axis)) / (2 * eps)
                         for axis in np.eye(4)])
    np.testing.assert_allclose(correction, -gradient, atol=1e-9)
    missing_chain_factor = matrix.T @ (weights * residual_a)
    assert np.linalg.norm(correction - missing_chain_factor) > .01
    print("Action metric: m/mm normalized residual agrees; mixed scales inflate energy 1,000,000x")
    print("Physical-space VJP: finite differences verify the inverse-normalization chain factor")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path)
    args = parser.parse_args()
    check_timeline()
    check_session_and_freshness()
    check_variable_latency()
    check_committed_prefix()
    check_masks(args.source_model)
    check_vjp()
    check_guidance_grid()
    check_training()
    check_episode_targets()
    check_delay_coverage()
    check_action_metric()
    print("All RTC teaching checks passed. No learned policy or robot was evaluated.")


if __name__ == "__main__":
    main()
