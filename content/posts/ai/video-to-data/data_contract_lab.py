#!/usr/bin/env python3
"""Offline V2D contract examples; no GPU, robot, model, or external packages.

The height checks follow the released tissue_box_lift_hold profile at
193382a184308f6800331f3db829f76dc4294cfe. Traces are constructed examples,
not measured robot results. The residual example follows the same commit's
ReferenceResidualJointPositionAction scaling/filtering/clipping order.
Reordering measured positions never creates expert action labels.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HeightRules:
    lift: float = 0.10
    hold: float = 0.05
    hold_samples: int = 20
    placement: float = 0.03
    placement_samples: int = 20
    final_floor: float = -0.01
    descent_step: float = 0.01
    terminal_step: float = 0.005
    terminal_range: float = 0.01


def _finite(values, expected=None):
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("expected nonempty finite numeric samples")
    if expected is not None and len(result) != expected:
        raise ValueError(f"expected {expected} values, got {len(result)}")
    return result


def height_result(heights, rules=HeightRules()):
    """First height supplies the baseline; later pre-action samples are scored."""
    trace = _finite(heights)
    lifts = [value - trace[0] for value in trace[1:]]
    peak = run = longest = previous = largest_descent = 0
    for value in lifts:
        peak = max(peak, value)
        if peak >= rules.hold:
            largest_descent = max(largest_descent, previous - value)
        run = run + 1 if value >= rules.hold else 0
        longest = max(longest, run)
        previous = value
    terminal = lifts[-rules.placement_samples:]
    enough = len(lifts) >= rules.placement_samples
    step = max((abs(b - a) for a, b in zip(terminal, terminal[1:])), default=0)
    span = max(terminal) - min(terminal) if terminal else 0
    stable = (enough and max(terminal) <= rules.placement
              and step <= rules.terminal_step and span <= rules.terminal_range)
    lift_hold = peak >= rules.lift and longest >= rules.hold_samples
    final = lifts[-1] if lifts else 0
    return {
        "success": bool(lift_hold and stable and final >= rules.final_floor
                        and largest_descent <= rules.descent_step),
        "lift_hold_success": bool(lift_hold),
        "stable_placement": bool(stable),
        "max_lift_m": float(peak),
        "max_hold_samples": int(longest),
        "max_descent_m_per_sample": float(largest_descent),
        "final_lift_m": float(final),
        "sample_count": len(lifts),
    }


def example_traces():
    """Three 20 Hz examples sharing the same rise and hold phase."""
    rise = [0.0] + [0.11 * i / 22 for i in range(1, 23)]
    common = rise + [0.11] * 25
    descent = [0.11 * (1 - i / 22) for i in range(1, 23)]
    return {
        "stable_place": common + descent + [0.0] * 25,
        "held_aloft": common + [0.11] * 47,
        "abrupt_drop": common + [0.0] * 47,
    }


def assemble_state(arm_positions, finger_positions):
    """State order: R arm, L arm, R fingers, L fingers; angles in radians."""
    return _finite(arm_positions, 14) + _finite(finger_positions, 44)


def positions_in_action_order(state):
    """Permute one measured pose for comparison; never infer control targets."""
    state = _finite(state, 58)
    return state[:7] + state[14:36] + state[7:14] + state[36:58]


def residual_targets(reference, policy_output, *, scale=0.15, ema=0.3,
                     residual_clip=1.0, joint_limits=None):
    """One-joint, zero-initial-residual example in double precision.

    Scale the raw policy output, update and clip the residual, then add the
    current reference and clamp the target. This computes controller targets,
    not measured joint positions; it does not simulate a PD actuator.
    """
    reference = _finite(reference)
    policy_output = _finite(policy_output, len(reference))
    scale, ema, residual_clip = _finite([scale, ema, residual_clip], 3)
    if scale <= 0 or not 0 <= ema <= 1 or residual_clip < 0:
        raise ValueError("require scale > 0, 0 <= ema <= 1 and residual_clip >= 0")
    limits = _finite(joint_limits, 2) if joint_limits is not None else None
    if limits is not None and limits[0] > limits[1]:
        raise ValueError("joint lower limit must not exceed upper limit")
    residual = 0.0
    rows = []
    for step, (ref, raw) in enumerate(zip(reference, policy_output)):
        filtered = ema * residual + (1 - ema) * (scale * raw)
        residual = max(-residual_clip, min(residual_clip, filtered))
        target = ref + residual
        if limits is not None:
            target = max(limits[0], min(limits[1], target))
        rows.append({"step": step, "reference_rad": ref, "raw_policy_output": raw,
                     "filtered_residual_rad": residual, "target_rad": target})
    return rows


def self_test():
    traces = example_traces()
    results = {name: height_result(trace) for name, trace in traces.items()}
    assert results["stable_place"]["success"]
    assert results["held_aloft"]["lift_hold_success"]
    assert not results["held_aloft"]["stable_placement"]
    assert not results["held_aloft"]["success"]
    assert results["abrupt_drop"]["stable_placement"]
    assert not results["abrupt_drop"]["success"]
    assert not height_result([0.0] * 100)["lift_hold_success"]
    assert height_result([0.0, 0.11, 0.0] * 30)["max_hold_samples"] == 1
    assert height_result([0.0])["sample_count"] == 0
    for bad in ([], [float("nan")], [0.0, float("inf")]):
        try:
            height_result(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid trace accepted")
    # Sentinels encode identity; they are not physically valid joint angles.
    arm = list(range(100, 107)) + list(range(200, 207))
    fingers = list(range(300, 322)) + list(range(400, 422))
    state = assemble_state(arm, fingers)
    reordered = positions_in_action_order(state)
    assert state[7] == 200 and reordered[7] == 300
    assert reordered[29] == state[7] and reordered[36:] == state[36:]
    assert sorted(reordered) == sorted(state)
    for name, trace in traces.items():
        assert results[name]["sample_count"] == len(trace) - 1
    residual = residual_targets([.5, .55], [2, 0])
    assert math.isclose(residual[0]["target_rad"], .71)
    assert math.isclose(residual[1]["filtered_residual_rad"], .063)
    assert math.isclose(residual[1]["target_rad"], .613)
    clipped = residual_targets([0, 0], [100, 0])
    assert clipped[0]["filtered_residual_rad"] == 1
    assert math.isclose(clipped[1]["filtered_residual_rad"], .3)
    limited = residual_targets([.5], [2], joint_limits=(-.6, .6))
    assert limited[0]["target_rad"] == .6
    print("height criteria, sample counting, input validation, layouts and residual targets passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--show-trace", choices=tuple(example_traces()),
                      help="print one constructed 20 Hz height trace as JSON")
    mode.add_argument("--residual-example", action="store_true",
                      help="print a constructed one-joint residual-to-target example")
    mode.add_argument("--trace", type=Path,
                      help="read a JSON height list, or {fps:20, heights_m:[...]} object")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.residual_example:
        print(json.dumps({"illustrative": True, "units": "joint radians; raw policy output unscaled",
                          "scale": .15, "ema": .3, "residual_clip_rad": 1,
                          "initial_residual_rad": 0, "joint_limit_assumption": "not reached in this constructed example",
                          "samples": residual_targets([.5, .55], [2, 0])}, indent=2))
    elif args.show_trace:
        print(json.dumps({"illustrative": True, "fps": 20,
                          "heights_m": example_traces()[args.show_trace]}, indent=2))
    elif args.trace:
        try:
            payload = json.loads(args.trace.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if float(payload.get("fps", 20)) != 20:
                    raise ValueError("these sample-count thresholds require 20 Hz input")
                values = payload["heights_m"]
            else:
                values = payload
            if not isinstance(values, list):
                raise ValueError("heights_m must be a JSON list in meters")
            result = height_result(values)
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps({"fps": 20, "baseline": "first sample",
                          "rules": asdict(HeightRules()), "result": result}, indent=2))
    else:
        print(json.dumps({"illustrative": True, "fps": 20,
                          "rules": asdict(HeightRules()),
                          "results": {name: height_result(trace)
                                      for name, trace in example_traces().items()}},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
