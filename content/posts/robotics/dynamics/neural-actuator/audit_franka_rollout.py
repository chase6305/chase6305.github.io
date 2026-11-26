#!/usr/bin/env python3
"""Verify Franka MJX dumps against the pinned CSVs and recompute force metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import numpy as np

from audit_nad import (
    FORCE_COLUMNS, FRANKA_LIFT, FRANKA_TEST, FRANKA_TEST_FILES, REVISION,
    manifest, parse_csv, read_blob, row_vectors,
)

ARM_COLUMNS = tuple(f"pos{joint}" for joint in range(1, 8))
WINDOWS = (10, 100, 200, 300, 400, 500, 600)
LOOKAHEAD = 5
CHECKPOINT_REVISION = "cd58d48d465450ffb734ed8ba22e48266ccdb2f2"
CHECKPOINT_SHA256 = "0894168cac986744b76d7d4e19d1995401cff4579aff63f49c54f7de447d64dd"
GROUPS = {"all_five": tuple(name.split("_")[0] + "g" for name in FRANKA_TEST_FILES),
          "without_duplicate_200g": ("300g", "400g", "500g", "600g")}


def nominal_vertical_label(rows):
    """The public benchmark provides a nominal vertical label and no lateral measurements."""
    labels, = set(row_vectors(rows, FORCE_COLUMNS))
    x, y, z = labels
    if (x, y) != (-999.0, -999.0) or z >= 0 or z == -999:
        raise ValueError("Expected a fixed vertical label and two missing lateral labels")
    return z


def load_dumps(rollouts, evaluation, rows, task):
    expected_steps = len(rows) - 1
    if evaluation["n_steps"] != expected_steps or evaluation["has_force"] is not True:
        raise ValueError(f"Incorrect evaluated length or force availability for {task}")

    shapes = {
        "force_pred": (expected_steps, 3),
        "force_gt": (expected_steps, 3),
        "sim_q": (expected_steps, 8),
        "gt_q": (expected_steps, 8),
        "gate": (expected_steps,),
    }
    with np.load(rollouts / f"{task}_rollout.npz", allow_pickle=False) as saved:
        data = {key: saved[key].astype(np.float64) for key in shapes}
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid shape or nonfinite values for {task}/{key}")
    if data["gate"].min() < 0 or data["gate"].max() > 1:
        raise ValueError(f"Gate lies outside [0, 1] for {task}")

    expected_q = np.asarray(row_vectors(rows[1:], ARM_COLUMNS + ("gripper_width",)))
    expected_q[:, 7] *= 0.04
    expected_force = np.asarray(row_vectors(rows[:-1], FORCE_COLUMNS))
    expected_force[expected_force == -999] = 0.0
    if not np.allclose(data["gt_q"], expected_q, rtol=1e-7, atol=1e-7):
        raise ValueError(f"{task} state reference does not match the next CSV rows")
    if not np.allclose(data["force_gt"], expected_force, rtol=1e-7, atol=1e-7):
        raise ValueError(f"{task} force reference does not match the current CSV rows")
    return data


def metrics(data, indices, nominal_force_z):
    force_gt = data["force_gt"][indices]
    prediction = data["force_pred"][indices]
    force_error = prediction - force_gt
    force_mae = np.mean(np.abs(force_error), axis=0)
    constant_mae = float(np.mean(np.abs(force_gt[:, 2] - nominal_force_z)))
    dq = np.abs(data["sim_q"][indices] - data["gt_q"][indices]).mean(axis=0)
    return {
        "mjx_xyz_mae_n": force_mae.tolist(),
        "mjx_force_mae_n": float(force_mae.mean()),
        "mjx_forcez_mae_n": float(force_mae[2]),
        "mjx_forcez_mean_n": float(prediction[:, 2].mean()),
        "mjx_forcez_mean_signed_error_n": float(force_error[:, 2].mean()),
        "constant_force_mae_n": constant_mae / 3,
        "constant_forcez_mae_n": constant_mae,
        "arm_mae_deg": np.rad2deg(dq[:7]).tolist(),
        "finger_mae_mm": float(dq[7] * 1000),
    }


def verify_metrics(evaluation, calculated, label):
    suffix = "" if label == "full" else f"@{label}"
    expected = {"Force": calculated["mjx_force_mae_n"], "ForceZ": calculated["mjx_forcez_mae_n"],
                "J8": calculated["finger_mae_mm"]}
    expected.update((f"J{i}", value) for i, value in enumerate(calculated["arm_mae_deg"], 1))
    for key, value in expected.items():
        if not math.isclose(evaluation[key + suffix], value, rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError(f"Recomputed metric disagrees with upstream JSON: {key + suffix}")


def aggregate(records):
    """Take the mean of task MAEs, with the same set of tasks at every retained window."""
    averages = {}
    for group, names in GROUPS.items():
        windows = [label for label in records[names[0]]
                   if all(label in records[name] for name in names)]
        averages[group] = {}
        for label in windows:
            task_metrics = [records[name][label] for name in names]
            averages[group][label] = {
                key: np.mean([row[key] for row in task_metrics], axis=0).tolist()
                for key in task_metrics[0]
            }
    return averages


def evaluate(rollouts, evaluation_path, tasks, trim, nominal_force_z):
    evaluation_bytes = evaluation_path.read_bytes()
    evaluation = json.loads(evaluation_bytes)
    if set(evaluation) != set(tasks) | {"AVERAGE"}:
        raise ValueError(f"Unexpected task names in {evaluation_path}")
    report = {}
    dumps = {}
    for task, (path, raw_rows) in tasks.items():
        start = int(len(raw_rows) * trim[0])
        stop = int(len(raw_rows) * (1 - trim[1]))
        kept = raw_rows[start:stop]
        dumps[task] = load_dumps(rollouts, evaluation[task], kept, task)
        steps = len(kept) - 1
        labels = [str(window) for window in WINDOWS if window <= steps] + ["full"]
        if any(f"Force@{window}" in evaluation[task] for window in WINDOWS if window > steps):
            raise ValueError(f"An out-of-range force window entered the upstream output for {task}")
        by_window = {}
        for label in labels:
            length = steps if label == "full" else int(label)
            value = metrics(dumps[task], slice(length), nominal_force_z)
            verify_metrics(evaluation[task], value, label)
            by_window[label] = value
        report[task] = {
            "csv": path,
            "raw_sample_rows": [start, stop],
            "rollout_steps": steps,
            "nominal_forcez_n": nominal_vertical_label(kept),
            "minimum_gate": float(dumps[task]["gate"].min()),
            "metrics": by_window,
        }
    task_means = aggregate({task: row["metrics"] for task, row in report.items()})
    for label, calculated in task_means["all_five"].items():
        suffix = "" if label == "full" else f"@{label}"
        upstream_mean = evaluation["AVERAGE"]["Force" + suffix]
        if not math.isclose(upstream_mean, calculated["mjx_force_mae_n"], abs_tol=2e-6):
            raise ValueError(f"Upstream task mean disagrees for Force{suffix}")
    return {
        "upstream_evaluation_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        "trim_start_end": trim,
        "minimum_gate_across_tasks": min(row["minimum_gate"] for row in report.values()),
        "tasks": report,
        "task_means": task_means,
    }, dumps


def build_report(args):
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint digest differs from the pinned Franka release")
    entries = manifest(args.source)

    def rows(path):
        return parse_csv(read_blob(args.source, args.csv_root, entries, path))

    train_paths = sorted(path for path in entries if path.startswith(f"{FRANKA_LIFT}/train/"))
    train_force = [nominal_vertical_label(rows(path)) for path in train_paths]
    constant = statistics.median(train_force)
    tasks = {}
    for filename in FRANKA_TEST_FILES:
        path = f"{FRANKA_LIFT}/test/{filename}"
        task = filename.split("_")[0] + "g"
        tasks[task] = (path, rows(path))

    released, crop_dumps = evaluate(args.released_rollouts, args.released_eval,
                                    tasks, (0.30, 0.25), constant)
    extended, extended_dumps = evaluate(args.extended_rollouts, args.extended_eval,
                                        tasks, (0.30, 0.0), constant)
    uncropped, full_dumps = evaluate(args.uncropped_rollouts, args.uncropped_eval,
                                     tasks, (0.0, 0.0), constant)
    matched_windows, unchanged_prefix = {}, {}
    for task, item in released["tasks"].items():
        start, stop = item["raw_sample_rows"]
        # Force on raw rows [start, stop - 1), state on [start + 1, stop).
        raw_window = slice(start, stop - 1)
        local_window = slice(stop - start - 1)
        reference = crop_dumps[task]
        for key in ("gt_q", "force_gt"):
            if not (np.array_equal(reference[key], full_dumps[task][key][raw_window])
                    and np.array_equal(reference[key], extended_dumps[task][key][local_window])):
                raise ValueError(f"Raw sampling rows disagree for {task}/{key}")
        # The recorded target pose agrees up to the first frame where the shorter
        # crop clamps the five-frame lookahead index to its final sample.
        prefix_steps = stop - start - LOOKAHEAD
        prefix = slice(prefix_steps)
        force_delta = np.max(np.abs(reference["force_pred"][prefix]
                                    - extended_dumps[task]["force_pred"][prefix])).item()
        state_delta = np.max(np.abs(reference["sim_q"][prefix]
                                    - extended_dumps[task]["sim_q"][prefix])).item()
        if max(force_delta, state_delta) > 1e-6:
            raise ValueError(f"Tail extension changed output before it could affect goals for {task}")
        unchanged_prefix[task] = {"steps": prefix_steps, "max_force_abs_delta_n": force_delta,
                                  "max_state_abs_delta_rad_or_m": state_delta}
        matched_windows[task] = {
            "released_reinitialized": item["metrics"]["full"],
            "extended_same_initial_row": metrics(extended_dumps[task], local_window, constant),
            "uncropped_from_row_zero": metrics(full_dumps[task], raw_window, constant),
        }
    report = {
        "revision": REVISION,
        "checkpoint": {
            "revision": CHECKPOINT_REVISION,
            "sha256": checkpoint_sha,
            "default_weight_key": "params",
        },
        "scoring": {
            "raw_sample_rows": "zero-based, start-inclusive and stop-exclusive",
            "reference": "state at t+1; nominal Z at t with lateral sentinels mapped to zero",
            "task_average": "unweighted mean of per-task MAEs on a common window",
        },
        "excluded_test_path": FRANKA_TEST,
        "fixed_baseline": {
            "train_path_count": len(train_paths),
            "selection": "median of per-record nominal training Z labels; lateral components zero",
            "force_xyz_n": [0.0, 0.0, constant],
        },
        "released_crop": released,
        "extend_end_only": extended,
        "uncropped": uncropped,
        "same_start_prefix_check": unchanged_prefix,
        "same_raw_rows_task_means": aggregate(matched_windows),
    }
    if args.ema_eval is not None:
        ema, _ = evaluate(args.ema_rollouts, args.ema_eval, tasks, (0.30, 0.0), constant)
        report["extend_end_only_ema"] = {
            "weight_key": "ema_params",
            "upstream_evaluation_sha256": ema["upstream_evaluation_sha256"],
            "task_means": ema["task_means"],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Dynamics-Modeling Git repository")
    parser.add_argument("--csv-root", type=Path, help="Local alternative to Git blobs (NeuralActuator/data)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned franka_lift_hold.pkl")
    parser.add_argument("--released-eval", type=Path, required=True)
    parser.add_argument("--released-rollouts", type=Path, required=True)
    parser.add_argument("--extended-eval", type=Path, required=True)
    parser.add_argument("--extended-rollouts", type=Path, required=True)
    parser.add_argument("--uncropped-eval", type=Path, required=True)
    parser.add_argument("--uncropped-rollouts", type=Path, required=True)
    parser.add_argument("--ema-eval", type=Path, help="Optional end extension with EMA weights")
    parser.add_argument("--ema-rollouts", type=Path)
    args = parser.parse_args()
    if (args.ema_eval is None) != (args.ema_rollouts is None):
        parser.error("--ema-eval and --ema-rollouts must be supplied together")
    try:
        print(json.dumps(build_report(args), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Franka rollout audit failed: {error}\n")


if __name__ == "__main__":
    main()
