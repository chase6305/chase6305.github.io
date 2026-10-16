#!/usr/bin/env python3
"""Configured Vega VOC scales, not measured forces or robot success rates.

Source: video_to_data 193382a184308f6800331f3db829f76dc4294cfe,
VegaManipCurriculumCfg and TrackingCommand._update_command.
Only the Python standard library is required.
"""

import argparse
from bisect import bisect_right
import json
import math


THRESHOLDS = tuple(range(1250, 12501, 1250))
SCALES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
CONTROL_HZ = 20
STEPS_PER_PPO_UPDATE = 24


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def global_target(ppo_update):
    """Configured curriculum at environment_step_counter / 24.

    This is a sampling-budget clock, not a checkpoint's zero-based iteration ID.
    """
    _nonnegative(ppo_update, "ppo_update")
    return SCALES[bisect_right(THRESHOLDS, ppo_update)]


def reset_scale(step, target, freeze=50, decay=10, reset=1.0):
    """Scale after `step` command updates; zero is the resample state.

    The reference clock stays frozen through max(freeze, decay) updates. With decay=0,
    reset initializes the per-environment scale directly to the global target.
    IsaacLab advances physics before the command update; this does not report
    force application time or elapsed wall time after an automatic reset.
    """
    for name, value in [("step", step), ("target", target), ("freeze", freeze),
                        ("decay", decay), ("reset", reset)]:
        _nonnegative(value, name)
    if step != int(step) or freeze != int(freeze) or decay != int(decay):
        raise ValueError("step, freeze and decay must be integer control-step counts")
    if decay == 0:
        return target
    progress = min(max(step - max(freeze - decay, 0), 0) / decay, 1.0)
    return max(reset + (target - reset) * progress, 0.0)


def reference_advances(step, freeze=50, decay=10):
    """How many reference-frame increments occurred since the reset."""
    for name, value in [("step", step), ("freeze", freeze), ("decay", decay)]:
        _nonnegative(value, name)
        if value != int(value):
            raise ValueError(f"{name} must be an integer control-step count")
    return max(0, int(step) - max(int(freeze), int(decay)))


def self_test():
    assert global_target(0) == 1.0
    assert global_target(1249) == 1.0 and global_target(1250) == 0.9
    assert global_target(12499) == 0.1 and global_target(12500) == 0.0
    assert global_target(20000) == 0.0
    assert [reset_scale(k, 0.0) for k in [0, 40, 45, 50, 51]] == [1, 1, .5, 0, 0]
    assert math.isclose(reset_scale(45, .6), .8)
    assert reference_advances(50) == 0 and reference_advances(51) == 1
    assert all(reset_scale(k, 0.0, freeze=0, decay=0, reset=0) == 0 for k in range(80))
    assert reference_advances(1, freeze=0, decay=0) == 1
    for args in [(-1, 0), (float("nan"), 0), (1.5, 0)]:
        try:
            reset_scale(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid step accepted")
    print("curriculum boundaries, reset ramp, reference freeze and VOC-free checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--iteration", type=int, default=12500,
                        help="environment-step counter divided by 24; zero-scale threshold is 12500")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 40, 45, 50, 51],
                        help="command-update counts since resample; 0 is the resample state")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    try:
        target = global_target(args.iteration)
        rows = [{"command_updates": step, "nominal_command_time_s": step / CONTROL_HZ,
                 "training_scale": reset_scale(step, target),
                 "training_reference_advances": reference_advances(step),
                 "collection_scale": reset_scale(step, 0, freeze=0, decay=0, reset=0),
                 "collection_reference_advances": reference_advances(step, freeze=0, decay=0)}
                for step in args.steps]
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"kind": "configured_scales_not_measured_forces",
                      "clock": "environment_step_counter / 24",
                      "ppo_update_equivalent": args.iteration, "scheduled_target": target,
                      "environment_step_counter": args.iteration * STEPS_PER_PPO_UPDATE,
                      "control_hz": CONTROL_HZ, "samples": rows}, indent=2))


if __name__ == "__main__":
    main()
