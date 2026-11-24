#!/usr/bin/env python3
"""Compare paired, timestamp-aligned probes near the object-local plane x=0.

Coordinates are in metres, time in seconds, and object rotations in wxyz.
Run without arguments for the three article examples. Use --example to
print an input JSON document, or pass an input path to evaluate your own.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean


def components(value, size):
    if (len(value) != size
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError(f"expected {size} finite numbers")
    return value


def local_points(pose):
    tx, ty, tz = components(pose["object_pos_m"], 3)
    quat = components(pose["object_rot_wxyz"], 4)
    norm = math.hypot(*quat)
    if not math.isclose(norm, 1, rel_tol=1e-3):
        raise ValueError("expected a unit wxyz quaternion")
    w, x, y, z = (v / norm for v in quat)
    # Rows of the inverse rotation. The outward plane normal is its first row.
    inverse = (
        (1 - 2*(y*y + z*z), 2*(x*y + z*w), 2*(x*z - y*w)),
        (2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w)),
        (2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x*x + y*y)),
    )
    result = {}
    for probe, world in pose["probes_world_m"].items():
        px, py, pz = components(world, 3)
        delta = (px - tx, py - ty, pz - tz)
        result[probe] = tuple(sum(a*b for a, b in zip(row, delta)) for row in inverse)
    return result


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def mean_mm(values):
    return 1000 * fmean(values) if values else None


def audit(frames, threshold_mm=20):
    if not frames or not math.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError("expected frames and a positive finite threshold")
    threshold = threshold_mm / 1000
    expected = frames[0]["reference"]["probes_world_m"].keys()
    if not expected:
        raise ValueError("expected at least one probe")
    previous_time = -math.inf
    near_ref = near_candidate = overlap = labelled_missed = unexpected_near = 0
    depth, surface, anchors = [], [], []
    penetration = 0.0

    for frame in frames:
        time, = components([frame["t_s"]], 1)
        if time <= previous_time:
            raise ValueError("t_s must strictly increase")
        previous_time = time
        reference = local_points(frame["reference"])
        candidate = local_points(frame["candidate"])
        labelled = frame["anchors_local_m"]
        if reference.keys() != expected or candidate.keys() != expected or not labelled.keys() <= expected:
            raise ValueError("each frame needs the same probes and anchors must name existing probes")
        for probe, source in reference.items():
            target = candidate[probe]
            ref_near = source[0] < threshold
            candidate_near = target[0] < threshold
            near_ref += ref_near
            near_candidate += candidate_near
            overlap += ref_near and candidate_near
            penetration = max(penetration, -target[0])
            if ref_near:
                depth.append(abs(target[0] - max(source[0], 0)))
                surface.append(math.dist(source[1:], target[1:]))
            if probe in labelled:
                anchors.append(math.dist(target, components(labelled[probe], 3)))
                labelled_missed += not candidate_near
            else:
                unexpected_near += candidate_near

    return {
        "samples": len(frames) * len(expected),
        "threshold_mm": threshold_mm,
        "reference_near": near_ref,
        "candidate_near": near_candidate,
        "near_intersection": overlap,
        "near_jaccard": ratio(overlap, near_ref + near_candidate - overlap),
        "near_precision": ratio(overlap, near_candidate),
        "near_recall": ratio(overlap, near_ref),
        "reference_near_depth_mae_mm": mean_mm(depth),
        "reference_near_surface_shift_mean_mm": mean_mm(surface),
        "labelled_samples": len(anchors),
        "labelled_anchor_mean_mm": mean_mm(anchors),
        "labelled_samples_outside_threshold": labelled_missed,
        "unlabelled_candidate_near": unexpected_near,
        "max_penetration_mm": 1000 * penetration,
    }


def example(gaps_mm, tangent_mm=0):
    identity = [1, 0, 0, 0]

    def pose(gap, tangent=0):
        return {"object_pos_m": [0, 0, 0], "object_rot_wxyz": identity,
                "probes_world_m": {"palm": [gap / 1000, tangent / 1000, 0]}}

    return [
        {"t_s": i / 30, "reference": pose(ref_gap),
         "candidate": pose(gap, tangent_mm if ref_gap == 0 else 0),
         "anchors_local_m": {"palm": [0, 0, 0]} if ref_gap == 0 else {}}
        for i, (ref_gap, gap) in enumerate(zip((40, 0, 0, 35), gaps_mm))
    ]


def demos():
    return {"tangential_shift": example((40, 0, 0, 35), 200),
            "late_release": example((40, 0, 0, 0)),
            "penetration": example((40, -10, -10, 35))}


def check():
    expected = [(1, 0, 200, 0, 0), (2/3, 0, 0, 1, 0), (1, 10, 10, 0, 10)]
    keys = ("near_jaccard", "reference_near_depth_mae_mm", "labelled_anchor_mean_mm",
            "unlabelled_candidate_near", "max_penetration_mm")
    for frames, result in zip(demos().values(), expected):
        report = audit(frames)
        assert tuple(report[key] for key in keys) == result

    # Rotate the candidate 90 degrees about +Z and move it by (1, 2, 3).
    # Its local probe (0, 0.2, 0) ends up at world (0.8, 2, 3).
    frame = demos()["tangential_shift"][1]
    frame["candidate"] = {"object_pos_m": [1, 2, 3],
                          "object_rot_wxyz": [-math.sqrt(0.5), 0, 0, -math.sqrt(0.5)],
                          "probes_world_m": {"palm": [0.8, 2, 3]}}
    report = audit([frame])
    assert math.isclose(report["reference_near_surface_shift_mean_mm"], 200)
    assert math.isclose(report["labelled_anchor_mean_mm"], 200)
    assert report["reference_near_depth_mae_mm"] < 1e-9

    empty = audit(demos()["late_release"][:1], threshold_mm=40)
    assert empty["near_jaccard"] is None and empty["labelled_anchor_mean_mm"] is None
    assert empty["reference_near_depth_mae_mm"] is None
    try:
        audit([frame, frame])
    except ValueError as exc:
        assert "increase" in str(exc)
    else:
        raise AssertionError("duplicate frame times must fail")
    print("article cases, moving object, empty supports and duplicate times passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--example", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--threshold-mm", type=float, default=20.0)
    args = parser.parse_args()
    if args.input and (args.example or args.check):
        parser.error("a file cannot be combined with --example or --check")
    try:
        if args.check:
            check()
            return
        if args.input:
            with args.input.open(encoding="utf-8") as stream:
                document = json.load(stream)
            result = audit(document["frames"], args.threshold_mm)
        elif args.example:
            result = {"frames": demos()["tangential_shift"]}
        else:
            result = {name: audit(frames, args.threshold_mm) for name, frames in demos().items()}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
