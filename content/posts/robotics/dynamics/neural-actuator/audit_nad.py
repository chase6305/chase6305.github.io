#!/usr/bin/env python3
"""Inspect selected NAD paths and CSVs at an immutable upstream Git revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

REVISION = "541a93aac4368e73883e86a51f31a7add6bde059"
PREFIX = "NeuralActuator/data/"
FORCE_COLUMNS = ("force_x", "force_y", "force_z")
JOINT_SIGNATURE = tuple(f"{field}{joint}" for field in ("pos", "current") for joint in range(1, 6))
OMX_WEIGHT = "weight/pick_place_object_200g"
OMX_CONDITION = "motor_condition/pick_place_object_200g"
FRANKA_LIFT = "franka/lift_hold"
FRANKA_TEST = f"{FRANKA_LIFT}/test/200_006.csv"
FRANKA_TRAIN = f"{FRANKA_LIFT}/train/200_007.csv"
FRANKA_TEST_FILES = ("200_006.csv", "300_001.csv", "400_001.csv", "500_006.csv", "600_003.csv")
FRANKA_SIGNATURE = tuple(
    f"{field}{joint}" for field in ("pos", "tau_d", "motor_pos") for joint in range(1, 8)
)
SAMPLES = {
    "omx_push": "force_sensor/force_x_plus/test/001.csv",
    "omx_reference": "force_sensor/force_x_plus_ref/test/001.csv",
    "omx_gripper": "force_unlabeled/joint_sweep_motor15/test/001.csv",
    "so101_pick_place": "so101/pick_and_place/300g/test/001.csv",
    "franka_300g": "franka/lift_hold/test/300_001.csv",
}


def git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(source), *args], timeout=45
    )


def manifest(source: Path) -> dict[str, str]:
    git(source, "cat-file", "-e", REVISION + "^{commit}")
    output = git(source, "ls-tree", "-rz", REVISION, "--", PREFIX)
    entries = {}
    for entry in output.decode().split("\0"):
        if not entry:
            continue
        metadata, path = entry.split("\t", 1)
        mode, kind, sha = metadata.split()
        if path.endswith(".csv") and kind == "blob" and mode == "100644":
            entries[path.removeprefix(PREFIX)] = sha
    if not entries:
        raise ValueError("The pinned tree contains no data CSV entries")
    return entries


def read_blob(source: Path, csv_root: Path | None, entries: dict[str, str], path: str) -> bytes:
    if csv_root is None:
        data = git(source, "show", f"{REVISION}:{PREFIX}{path}")
    else:
        data = (csv_root / path).read_bytes()
    header = f"blob {len(data)}\0".encode()
    if hashlib.sha1(header + data).hexdigest() != entries[path]:
        raise ValueError(f"CSV bytes differ from the pinned Git blob: {path}")
    return data


def parse_csv(raw: bytes) -> list[dict[str, float]]:
    reader = csv.DictReader(io.StringIO(raw.decode(), newline=""))
    rows = [{key: float(value) for key, value in row.items()} for row in reader]
    if len(rows) < 2 or any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("Expected at least two finite telemetry samples")
    return rows


def row_vectors(rows: list[dict[str, float]], columns: tuple[str, ...]) -> list[tuple[float, ...]]:
    return [tuple(row[column] for column in columns) for row in rows]


def overlap(short, long) -> tuple[int, float, float]:
    """Find a full contiguous telemetry match, then verify every other field."""
    short_signature = row_vectors(short, JOINT_SIGNATURE)
    long_signature = row_vectors(long, JOINT_SIGNATURE)
    offsets = [
        offset for offset, signature in enumerate(long_signature)
        if signature == short_signature[0]
        and long_signature[offset:offset + len(short)] == short_signature
    ]
    if len(offsets) != 1:
        raise ValueError("Expected a unique contiguous match in positions and currents")
    offset, = offsets
    if short[0].keys() != long[0].keys():
        raise ValueError("The files do not share the same columns")
    time_base = long[offset]["timestamp"] - short[0]["timestamp"]
    pairs = list(zip(short, long[offset:offset + len(short)], strict=True))
    values_error = max(abs(a[key] - b[key]) for a, b in pairs for key in a if key != "timestamp")
    clock_error = max(abs((a["timestamp"] + time_base) - b["timestamp"]) for a, b in pairs)
    if values_error > 1e-12 or clock_error > 1e-12:
        raise ValueError("Aligned samples disagree beyond float rounding")
    return offset, time_base, values_error


def sample_summary(rows):
    times = [row["timestamp"] for row in rows]
    deltas = [next_time - time for time, next_time in zip(times, times[1:])]
    if min(deltas) <= 0:
        raise ValueError("Expected strictly increasing timestamps")
    forces = row_vectors(rows, FORCE_COLUMNS)
    norms = [math.hypot(*(value if value != -999 else 0.0 for value in sample)) for sample in forces]
    positives = [norm > 0.01 for norm in norms]
    summary = {
        "rows": len(rows),
        "columns": len(rows[0]),
        "median_interval_ms": round(statistics.median(deltas) * 1000, 6),
        "valid_xyz": [sum(value != -999 for value in values) for values in zip(*forces)],
        "min_filled_force_norm_n": round(min(norms), 9),
        "max_filled_force_norm_n": round(max(norms), 9),
        "threshold_positive_rows": sum(positives),
        "threshold_transition_indices": [
            index for index in range(1, len(rows)) if positives[index - 1] != positives[index]
        ],
    }
    spans_ms = [(last - first) * 1000 for first, last in zip(times, times[8:])]
    if spans_ms:
        summary["nine_frame_span_ms"] = {
            "min": round(min(spans_ms), 6),
            "median": round(statistics.median(spans_ms), 6),
            "max": round(max(spans_ms), 6),
        }
    return summary


def franka_overlaps(entries, rows) -> dict:
    """Find the longest contiguous match for each pair of telemetry files."""
    train_paths = sorted(path for path in entries if path.startswith(f"{FRANKA_LIFT}/train/"))
    test_paths = sorted(path for path in entries if path.startswith(f"{FRANKA_LIFT}/test/"))
    minimum_rows = 9
    test_index = defaultdict(list)
    for path in test_paths:
        for index, signature in enumerate(row_vectors(rows(path), FRANKA_SIGNATURE)):
            test_index[signature].append((path, index))

    matches = []
    for train_path in train_paths:
        active = {}
        longest = {}
        for train_row, signature in enumerate(row_vectors(rows(train_path), FRANKA_SIGNATURE)):
            # A matching row extends a run only if both files advance by one row.
            next_active = {}
            for test_path, test_row in test_index.get(signature, ()):
                length = active.get((test_path, test_row - 1), 0) + 1
                next_active[test_path, test_row] = length
                if length >= minimum_rows and length > longest.get(test_path, (0,))[0]:
                    longest[test_path] = (length, train_row + 1 - length, test_row + 1 - length)
            active = next_active
        for test_path, (length, train_start, test_start) in sorted(longest.items()):
            matches.append({
                "train": train_path, "test": test_path, "train_start_row": train_start,
                "test_start_row": test_start, "matching_rows": length,
            })

    return {
        "train_paths": len(train_paths),
        "test_paths": len(test_paths),
        "compared_pairs": len(train_paths) * len(test_paths),
        "match_columns": list(FRANKA_SIGNATURE),
        "minimum_contiguous_rows": minimum_rows,
        "matches": matches,
    }


def audit(source: Path, csv_root: Path | None, scan_franka: bool = False) -> dict:
    entries = manifest(source)
    groups = defaultdict(list)
    for path, oid in entries.items():
        groups[oid].append(path)

    def read(path):
        return read_blob(source, csv_root, entries, path)

    def rows(path):
        return parse_csv(read(path))

    aliases = []
    for weight_path in sorted(entries):
        if not weight_path.startswith(OMX_WEIGHT + "/"):
            continue
        condition_path = OMX_CONDITION + weight_path[len(OMX_WEIGHT):]
        short, long = rows(weight_path), rows(condition_path)
        offset, time_base, value_error = overlap(short, long)
        aliases.append({
            "weight_path": weight_path,
            "motor_condition_path": condition_path,
            "weight_rows": len(short),
            "motor_condition_rows": len(long),
            "first_matching_index": offset,
            "last_matching_index": offset + len(short) - 1,
            "timestamp_offset_s": time_base,
            "max_non_timestamp_abs_delta": value_error,
        })

    train, test = read(FRANKA_TRAIN), read(FRANKA_TEST)
    if train != test:
        raise ValueError("The Franka duplicate is no longer byte identical")
    samples = {name: rows(path) for name, path in SAMPLES.items()}
    summaries = {name: {"path": SAMPLES[name], **sample_summary(data)}
                 for name, data in samples.items()}

    omx_goal = tuple(f"goal_pos{joint}" for joint in range(1, 6))
    commands = row_vectors(samples["omx_push"], omx_goal)
    reference = row_vectors(samples["omx_reference"], omx_goal)
    max_command_delta = max(abs(a - b) for first, second in zip(commands, reference, strict=True)
                            for a, b in zip(first, second, strict=True))
    gripper = [row["aperture"] for row in samples["omx_gripper"]]

    franka = samples["franka_300g"]
    first = int(0.30 * len(franka))
    stop = int((1 - 0.25) * len(franka))
    cropped = franka[first:stop]
    franka_goal = tuple(f"cmd_pos{joint}" for joint in range(1, 8))
    times = [row["timestamp"] for row in cropped]
    lookahead = [times[min(i + 5, len(times) - 1)] - time for i, time in enumerate(times)]
    franka_lengths = {}
    for filename in FRANKA_TEST_FILES:
        count = len(rows(f"{FRANKA_LIFT}/test/{filename}"))
        retained = int((1 - 0.25) * count) - int(0.30 * count)
        franka_lengths[filename] = {
            "raw_rows": count, "cropped_rows": retained, "rollout_steps": retained - 1,
        }

    # The force-sensor config starts with the matching +X file in each split.
    # The validator's start index comes from validation, but its closure reads train.
    first_train = rows("force_sensor/force_x_plus/train/001.csv")
    first_val = rows("force_sensor/force_x_plus/validation/001.csv")
    index = 100
    pose_delta = [first_train[index][f"pos{j}"] - first_val[index][f"pos{j}"] for j in range(1, 5)]

    result = {
        "revision": REVISION,
        "csv_paths": len(entries),
        "distinct_git_blobs": len(groups),
        "paths_by_family": dict(Counter(path.split("/")[0] for path in entries)),
        "exact_blob_aliases": [paths for paths in groups.values() if len(paths) > 1],
        "omx_loaded_crop_aliases": aliases,
        "franka_cross_split": {
            "train": FRANKA_TRAIN, "test": FRANKA_TEST, "git_blob": entries[FRANKA_TEST],
            "sha256": hashlib.sha256(test).hexdigest(), "bytes": len(test),
            "rows": len(parse_csv(test)),
        },
        "samples": summaries,
        "franka_test_lengths": franka_lengths,
        "omx_validation_initial_state_example": {
            "row_index": index,
            "train_minus_validation_pose_deg": [math.degrees(delta) for delta in pose_delta],
        },
        "omx_max_paired_goal_delta_rad": max_command_delta,
        "omx_single_finger_min_max_mm": [min(gripper), max(gripper)],
        "franka_proxy": {
            "first_row": first, "stop_row_exclusive": stop, "retained_rows": len(cropped),
            "unique_logged_goals_full": len(set(row_vectors(franka, franka_goal))),
            "unique_logged_goals_cropped": len(set(row_vectors(cropped, franka_goal))),
            "median_lookahead_ms": round(statistics.median(lookahead) * 1000, 6),
            "last_lookahead_ms": lookahead[-1] * 1000,
        },
    }
    if scan_franka:
        result["franka_telemetry_overlap"] = franka_overlaps(entries, rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Dynamics-Modeling Git repository")
    parser.add_argument("--csv-root", type=Path, help="Optional local copy of NeuralActuator/data")
    parser.add_argument("--franka-overlaps", action="store_true",
                        help="Also scan all Franka train/test pairs for shared telemetry runs")
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.source, args.csv_root, args.franka_overlaps),
                         indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Audit failed: {error}\n")


if __name__ == "__main__":
    main()
