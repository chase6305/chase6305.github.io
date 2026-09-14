#!/usr/bin/env python3
"""Small, offline data-contract experiments; no robot or training dependencies.

Run: python data_contract_lab.py
Only synthetic examples are used. This is not an RLDS/LeRobot validator.
Times are seconds in a monotonic common clock after explicit calibration.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from typing import Callable


TICKS_PER_SECOND = 1_000_000_000


class ContractError(ValueError):
    """A sample cannot satisfy this lab's explicitly chosen data contract."""


def finite(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def time_tick(seconds: float) -> int:
    """Quantize relative common-clock seconds to this lab's declared 1 ns grid.

    This removes binary-float comparison artifacts, not sensor timing error.
    Production logs should preserve their original integer ticks and clock domain.
    """
    if not finite(seconds):
        raise ContractError("nonfinite timestamp")
    return round(seconds * TICKS_PER_SECOND)


@dataclass(frozen=True)
class Clock:
    scale: float
    offset: float

    def to_common(self, timestamp: float) -> float:
        if not all(finite(x) for x in (self.scale, self.offset, timestamp)) or self.scale <= 0:
            raise ContractError("invalid clock calibration")
        return self.scale * timestamp + self.offset


@dataclass(frozen=True)
class Frame:
    name: str
    capture_device: float
    received_common: float
    tracking_epoch: int
    valid: bool = True


def select_observation(frames: list[Frame], clock: Clock, decision: float,
                       max_age: float, epoch: int) -> tuple[str, float]:
    """Latest usable frame at a decision, respecting capture AND availability.

    This is an online-observation example. Future training targets are handled
    separately in command_window and must not be mistaken for future inputs.
    """
    if not finite(decision) or not finite(max_age) or max_age < 0:
        raise ContractError("invalid decision time or age budget")
    decision_tick, age_budget_tick = time_tick(decision), time_tick(max_age)
    candidates = []
    for frame in frames:
        if not frame.valid or frame.tracking_epoch != epoch:
            continue
        capture = time_tick(clock.to_common(frame.capture_device))
        received = time_tick(frame.received_common)
        if capture > received:
            raise ContractError("capture later than receipt: clock contract violated")
        if received <= decision_tick and capture <= decision_tick:
            candidates.append((capture, frame.name))
    if not candidates:
        raise ContractError("no observation available in this tracking epoch")
    capture, name = max(candidates)
    age_tick = decision_tick - capture
    if age_tick > age_budget_tick:
        raise ContractError("available observation is too old")
    return name, age_tick / TICKS_PER_SECOND


@dataclass(frozen=True)
class Command:
    effective_time: float
    target: tuple[float, ...]
    tracking_epoch: int


def command_window(commands: list[Command], start: float, horizon: int,
                   period: float, coverage_end: float, epoch: int,
                   control_mode: str = "joint_position_target") -> list[tuple[float, ...]]:
    """Sample held POSITION TARGETS at future training-label times.

    Commands become effective at effective_time and remain active until the
    next command. coverage_end is an exclusive episode end, not inferred from
    the last logged target. The whole [start, start + horizon*period) must exist.
    This convention is unsuitable for incremental displacements or impulses.
    No interpolation, extrapolation beyond the episode, or epoch crossing.
    """
    if control_mode != "joint_position_target":
        raise ContractError("zero-order hold here is defined only for position targets")
    if (not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
            or not all(finite(x) for x in (start, period, coverage_end)) or period <= 0):
        raise ContractError("invalid target window")
    if not commands:
        raise ContractError("missing robot commands")
    times = [time_tick(item.effective_time) for item in commands]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ContractError("command timestamps must be strictly increasing")
    dimensions = len(commands[0].target)
    if not dimensions or any(len(item.target) != dimensions or not all(map(finite, item.target))
                             for item in commands):
        raise ContractError("inconsistent or nonfinite command targets")
    start_tick, period_tick = time_tick(start), time_tick(period)
    if period_tick <= 0:
        raise ContractError("period below the declared time resolution")
    end_tick = start_tick + horizon * period_tick
    if end_tick > time_tick(coverage_end) or start_tick < times[0]:
        raise ContractError("window falls outside recorded execution coverage")
    first = bisect_right(times, start_tick) - 1
    # Include every event inside the window, even between grid points. Otherwise
    # a reset can be missed by low-rate sampling.
    active_events = [commands[first]] + [c for c, tick in zip(commands, times)
                                        if start_tick < tick < end_tick]
    if any(c.tracking_epoch != epoch for c in active_events):
        raise ContractError("target window crosses a tracking epoch")
    return [commands[bisect_right(times, start_tick + k * period_tick) - 1].target
            for k in range(horizon)]


def masked_mse(predictions: list[list[float]], targets: list[list[float | None]],
               valid: list[list[bool]]) -> float | None:
    """Mean over valid coordinates only; None means skip this action loss.

    The mask is selected BEFORE arithmetic, so masked NaN/None labels do not
    contaminate the result. Shapes are checked, including masked rows.
    """
    if not len(predictions) == len(targets) == len(valid):
        raise ContractError("batch sizes differ")
    errors = []
    for pred, target, mask in zip(predictions, targets, valid):
        if not len(pred) == len(target) == len(mask):
            raise ContractError("action dimensions differ")
        for p, t, enabled in zip(pred, target, mask):
            if not isinstance(enabled, bool):
                raise ContractError("mask must be boolean")
            if enabled:
                if not finite(p) or not finite(t):
                    raise ContractError("valid target or prediction is not finite")
                errors.append((p - t) ** 2)
    return sum(errors) / len(errors) if errors else None


def audit_split(records: list[dict], require_unseen_scenes: bool = False) -> None:
    """Family identifiers must include all descendants of a source episode.

    This checks declared metadata, not perceptual duplicates or an automatically
    discovered provenance graph. Optional scene isolation tests a stronger goal.
    """
    ids = set()
    groups = {}
    keys = ("family", "scene") if require_unseen_scenes else ("family",)
    for row in records:
        if any(not isinstance(row.get(key), str) or not row[key] for key in ("id", "split", *keys)):
            raise ContractError("missing provenance fields")
        if row["id"] in ids:
            raise ContractError("duplicate episode ID")
        if row["split"] not in {"train", "validation", "test"}:
            raise ContractError("unknown split")
        ids.add(row["id"])
        for key in keys:
            group = (key, row[key])
            if group in groups and groups[group] != row["split"]:
                raise ContractError(f"{key} spans multiple splits: {row[key]}")
            groups[group] = row["split"]


def require_real_command_provenance(origin: str, environment: str) -> None:
    """Gate metadata for one target: command-conditioned REAL dynamics.

    This checks declared provenance, not whether hardware obeyed the command.
    State differences, human-retargeted and latent actions are useful for other
    objectives; they cannot silently be relabeled as recorded robot commands.
    """
    if origin != "recorded_robot_command" or environment != "real":
        raise ContractError("not a recorded command from real robot interaction")


def run_examples() -> dict:
    checks = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name: str, operation: Callable, message: str) -> None:
        try:
            operation()
        except ContractError as error:
            check(name, message in str(error))
        else:
            raise AssertionError(f"{name}: invalid sample was accepted")

    clock = Clock(1.0, -4.0)
    frames = [Frame("available", 5.000, 1.020, 0),
              Frame("newer_but_late", 5.025, 1.080, 0)]
    chosen, age = select_observation(frames, clock, 1.040, 0.050, 0)
    check("exclude_not_yet_received_frame", chosen == "available")
    check("age_uses_capture_not_receipt", math.isclose(age, 0.040))
    check("clock_scale_and_offset", math.isclose(Clock(1.001, -4).to_common(5), 1.005))
    boundary = select_observation([Frame("exact_boundary", 5.025, 1.025, 0)],
                                  clock, 1.025, 0.050, 0)
    check("calibrated_frame_on_decision_boundary_is_usable", boundary == ("exact_boundary", 0.0))
    rejects("stale_observation", lambda: select_observation(frames, clock, 1.200, 0.050, 0), "too old")
    rejects("reset_invalidates_old_frames", lambda: select_observation(frames, clock, 1.040, 0.050, 1), "no observation")
    rejects("bad_clock_is_not_silently_repaired", lambda: select_observation(frames, Clock(1, 0), 6, 9, 0), "clock contract")

    commands = [Command(1.000, (0.1, 0.2), 0), Command(1.020, (0.3, 0.4), 0),
                Command(1.040, (0.5, 0.6), 0), Command(1.060, (0.7, 0.8), 0),
                Command(1.080, (0.9, 1.0), 0)]
    # This example anchors labels at the decision time, using an aged frame.
    # Other policies anchor chunks at capture time and compensate differently.
    window = command_window(commands, 1.040, 3, 0.020, 1.100, 0)
    check("future_training_targets_are_allowed", window == [c.target for c in commands[2:]])
    held = command_window(commands, 1.011, 1, 0.005, 1.060, 0)
    check("held_target_is_not_nearest_future_command", held == [(0.1, 0.2)])
    rejects("no_padding_past_episode_end", lambda: command_window(commands, 1.060, 3, 0.020, 1.100, 0), "coverage")
    resets = [Command(1.000, (0.1,), 0), Command(1.010, (0.2,), 1), Command(1.020, (0.3,), 0)]
    rejects("detect_reset_between_sample_grid_points", lambda: command_window(resets, 1.000, 2, 0.020, 1.040, 0), "epoch")
    rejects("hold_does_not_apply_to_delta_commands", lambda: command_window(commands, 1, 1, .02, 1.06, 0, "delta_pose"), "position targets")
    rejects("reject_duplicate_command_times", lambda: command_window([commands[0], commands[0]], 1, 1, .02, 1.06, 0), "increasing")

    for dt in (0.001, 0.01, 0.02, 0.05, 0.1):
        regular = [Command(i * dt, (float(i),), 0) for i in range(100)]
        for first_index in range(80):
            actual = command_window(regular, first_index * dt, 10, dt, 100 * dt, 0)
            expected = [(float(i),) for i in range(first_index, first_index + 10)]
            if actual != expected:
                raise AssertionError("nominal equal-rate window shifted by float arithmetic")
    check("400_regular_windows_preserve_boundary_indices", True)

    loss = masked_mse([[1.0, 99.0], [2.0, 99.0]], [[0.0, None], [0.0, float("nan")]],
                      [[True, False], [True, False]])
    check("mask_before_arithmetic_and_normalization", loss == 2.5)
    check("all_missing_means_skip_action_loss", masked_mse([[1.0]], [[None]], [[False]]) is None)
    rejects("valid_nan_is_rejected", lambda: masked_mse([[1.0]], [[float("nan")]], [[True]]), "not finite")
    rejects("mask_does_not_hide_shape_errors", lambda: masked_mse([[1.0, 2.0]], [[None]], [[False]]), "dimensions")

    records = [dict(id="seed-a", family="family-a", scene="kitchen-a", split="train"),
               dict(id="child-a", family="family-a", scene="kitchen-b", split="train"),
               dict(id="seed-b", family="family-b", scene="kitchen-c", split="test")]
    audit_split(records, require_unseen_scenes=True)
    check("distinct_families_and_scenes_pass", True)
    leaked_family = records + [dict(id="child-a2", family="family-a", scene="kitchen-d", split="test")]
    rejects("synthetic_descendant_leakage", lambda: audit_split(leaked_family), "family spans")
    shared_scene = records + [dict(id="seed-c", family="family-c", scene="kitchen-a", split="test")]
    audit_split(shared_scene)
    check("scene_sharing_depends_on_evaluation_goal", True)
    rejects("unseen_scene_protocol_rejects_shared_scene", lambda: audit_split(shared_scene, True), "scene spans")

    require_real_command_provenance("recorded_robot_command", "real")
    check("real_recorded_command_gate", True)
    rejects("human_retarget_is_not_executed_command", lambda: require_real_command_provenance("human_retargeted", "real"), "not a recorded")
    rejects("state_difference_is_not_executed_command", lambda: require_real_command_provenance("state_difference", "real"), "not a recorded")
    rejects("human_operated_sim_is_still_sim", lambda: require_real_command_provenance("recorded_robot_command", "simulation"), "not a recorded")
    return {
        "scope": "synthetic teaching examples; no policy training or robot validation",
        "checks_passed": len(checks),
        "checks": checks,
        "observation": {"chosen": chosen, "age_ms": round(age * 1000, 6)},
        "action_target_times_seconds": [1.040, 1.060, 1.080],
        "future_training_action_window": window,
        "masked_mse": loss,
        "all_missing_action_loss": None,
    }


if __name__ == "__main__":
    print(json.dumps(run_examples(), ensure_ascii=False, indent=2, allow_nan=False))
