#!/usr/bin/env python3
"""Recover source timestamps for the pinned GMR, UMR and FRDV single-clip exports.

GMR uses reported PKL metadata. UMR and FRDV read numeric NPZ fields
with NumPy. With --csv, write one row per output frame.
"""

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Sequence


@dataclass(frozen=True)
class Timeline:
    source_indices: Sequence[float]
    source_step_s: float
    export_fps: float

    def rows(self):
        for index, source_index in enumerate(self.source_indices):
            yield index, source_index, source_index * self.source_step_s, index / self.export_fps

    def summary(self):
        first = self.source_indices[0] * self.source_step_s
        last = self.source_indices[-1] * self.source_step_s
        nominal_span = (len(self.source_indices) - 1) / self.export_fps
        # All three inputs use uniform grids, so maximum drift is at an endpoint.
        last_offset = last - nominal_span
        return {
            "exported_frames": len(self.source_indices),
            "export_fps": self.export_fps,
            "first_source_index": self.source_indices[0],
            "last_source_index": self.source_indices[-1],
            "first_source_time_s": first,
            "last_source_time_s": last,
            "nominal_sample_span_s": nominal_span,
            "source_sample_span_s": last - first,
            "max_abs_source_minus_export_s": max(abs(first), abs(last_offset)),
            "source_span_minus_nominal_span_s": last_offset - first,
        }


def require_positive_finite(value, label):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")


def validate_export(indices, step, expected_fps, frame_count, recorded_fps):
    if not indices:
        raise ValueError("the source and frame range would export no samples")
    if len(indices) != frame_count:
        raise ValueError(f"expected {len(indices)} exported frames, received {frame_count}")
    require_positive_finite(step, "source sample interval")
    require_positive_finite(recorded_fps, "export FPS")
    if not math.isclose(expected_fps, recorded_fps, rel_tol=1e-9):
        raise ValueError(f"expected export FPS {expected_fps:.12g}, received {recorded_fps}")
    return Timeline(indices, step, recorded_fps)


def gmr_smplx(source_frames, source_fps, exported_frames, export_fps):
    require_positive_finite(source_fps, "source FPS")
    if source_frames < 2:
        raise ValueError("GMR SMPL-X needs at least two source samples for a nonempty export")
    if source_fps > 30:
        adapted = source_frames // int(source_fps / 30)
        if adapted < 2:
            raise ValueError("the GMR SMPL-X adapter leaves fewer than two samples")
        grid_step = (source_frames - 1) / (adapted - 1)
        # linspace includes its endpoint, even when multiplication rounds away from it.
        indices = [index * grid_step for index in range(1, adapted - 1)] + [source_frames - 1]
        expected_fps = adapted / source_frames * source_fps
    else:
        indices = range(1, source_frames)
        expected_fps = 30
    return validate_export(indices, 1 / source_fps, expected_fps, exported_frames, export_fps)


def read_bvh_clock(path):
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip() == "MOTION":
                break
        else:
            raise ValueError("missing MOTION section in BVH")
        frames = next(source, "").strip().split()
        period = next(source, "").strip().split()
    if len(frames) != 2 or frames[0] != "Frames:" or len(period) != 3 or period[:2] != ["Frame", "Time:"]:
        raise ValueError("expected Frames: and Frame Time: immediately after MOTION")
    return int(frames[1]), float(period[2])


def gmr_bvh(source_frames, source_step, exported_frames, export_fps):
    if not math.isfinite(export_fps) or export_fps < 1 or export_fps != int(export_fps):
        raise ValueError("the pinned GMR BVH CLI requires an integer --motion_fps")
    return validate_export(range(source_frames - 1), source_step, export_fps, exported_frames, export_fps)


def frdv_bvh(source_frames, source_step, start, end, exported_frames, export_fps):
    require_positive_finite(source_step, "BVH Frame Time")
    indices = range(source_frames)[start:end]
    return validate_export(indices, source_step, 1 / source_step, exported_frames, export_fps)


def umr_timeline(frame_ids, rows, source_fps, export_fps):
    require_positive_finite(source_fps, "source FPS")
    require_positive_finite(export_fps, "export FPS")
    source_step = 1 / source_fps
    require_positive_finite(source_step, "source sample interval")
    if not frame_ids or len(frame_ids) != rows or frame_ids[0] < 0:
        raise ValueError("frame_ids must be nonnegative and match the number of qpos rows")
    if rows >= 2:
        stride = frame_ids[1] - frame_ids[0]
        if stride <= 0 or any(next_id - index != stride for index, next_id in zip(frame_ids, frame_ids[1:])):
            raise ValueError("expected increasing frame_ids at a constant stride")
        expected_fps = source_fps / stride
        if not math.isclose(expected_fps, export_fps, rel_tol=1e-6):
            raise ValueError(f"frame_ids and source FPS imply {expected_fps:.12g} FPS, received {export_fps}")
    return Timeline(frame_ids, source_step, export_fps)


def npz_metadata(export):
    qpos = export["qpos"]
    fps = export["fps"]
    if qpos.ndim != 2 or qpos.shape[1] == 0 or qpos.dtype.kind not in "iuf":
        raise ValueError("expected numeric qpos with one row per sample")
    if fps.size != 1 or fps.dtype.kind not in "iuf":
        raise ValueError("expected one numeric fps value")
    return qpos.shape[0], fps.item()


def read_umr(path, source_fps):
    import numpy as np

    with np.load(path, allow_pickle=False) as export:
        frame_ids = export["frame_ids"]
        if frame_ids.ndim != 1 or frame_ids.dtype.kind not in "iu":
            raise ValueError("expected a vector of integer frame_ids")
        rows, fps = npz_metadata(export)
        return umr_timeline(frame_ids.tolist(), rows, source_fps, fps)


def read_frdv(bvh, npz, start, end):
    import numpy as np

    frames, step = read_bvh_clock(bvh)
    with np.load(npz, allow_pickle=False) as export:
        return frdv_bvh(frames, step, start, end, *npz_metadata(export))


def check():
    low = gmr_smplx(60, 24, 59, 30).summary()
    assert math.isclose(low["max_abs_source_minus_export_s"], 0.525)
    high = gmr_smplx(60, 60, 29, 30)
    assert math.isclose(high.source_indices[0], 59 / 29)
    assert high.source_indices[-1] == 59
    assert math.isclose(high.summary()["source_span_minus_nominal_span_s"], 14 / 870)
    assert list(gmr_smplx(4, 60, 1, 30).rows()) == [(0, 3, 0.05, 0)]
    assert list(gmr_bvh(4, 0.04, 3, 30).rows())[-1] == (2, 2, 0.08, 2 / 30)
    assert list(frdv_bvh(5, 0.04, 1, 3, 2, 25).source_indices) == [1, 2]
    assert list(frdv_bvh(5, 0.04, -2, None, 2, 25).source_indices) == [3, 4]
    assert umr_timeline([12, 14, 16], 3, 60, 30).summary()["first_source_time_s"] == 0.2
    assert umr_timeline([12], 1, 60, 30).summary()["nominal_sample_span_s"] == 0
    for function, args in (
        (gmr_smplx, (3, 90, 0, 30)),
        (gmr_smplx, (60, 24, 60, 30)),
        (gmr_smplx, (60, 24, 59, 24)),
        (gmr_bvh, (1, 0.04, 0, 30)),
        (gmr_bvh, (4, 0.04, 3, 29.97)),
        (frdv_bvh, (5, 0.04, 1, 3, 3, 25)),
        (frdv_bvh, (5, 0.04, 1, 3, 2, 30)),
        (umr_timeline, ([12, 14, 15], 3, 60, 30)),
        (umr_timeline, ([12, 12], 2, 60, 30)),
        (umr_timeline, ([12, 14], 2, 60, 60)),
    ):
        try:
            function(*args)
        except ValueError:
            continue
        raise AssertionError(f"{function.__name__}{args} must reject inconsistent metadata")
    print("Motion timeline checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check", help="run the standard-library examples")
    smplx = modes.add_parser("gmr-smplx", help="non-looping GMR SMPL-X single-clip export")
    smplx.add_argument("--source-frames", type=int, required=True)
    smplx.add_argument("--source-fps", type=float, required=True)
    bvh = modes.add_parser("gmr-bvh", help="non-looping GMR BVH single-clip export")
    bvh.add_argument("bvh", type=Path)
    for mode in (smplx, bvh):
        mode.add_argument("--exported-frames", type=int, required=True)
        mode.add_argument("--export-fps", type=float, required=True)
    umr = modes.add_parser("umr", help="UMR single-clip NPZ with frame_ids (requires NumPy)")
    umr.add_argument("npz", type=Path)
    umr.add_argument("--source-fps", type=float, required=True)
    frdv = modes.add_parser("frdv-bvh", help="FRDV BVH to NPZ with an optional slice (requires NumPy)")
    frdv.add_argument("bvh", type=Path)
    frdv.add_argument("npz", type=Path)
    frdv.add_argument("--start", type=int, default=0)
    frdv.add_argument("--end", type=int, help="exclusive end from frdv_config.yaml")
    for mode in (smplx, bvh, umr, frdv):
        mode.add_argument("--csv", action="store_true", help="stream the frame mapping to stdout")
    args = parser.parse_args()
    try:
        if args.mode == "check":
            check()
            return
        if args.mode == "gmr-smplx":
            timeline = gmr_smplx(args.source_frames, args.source_fps, args.exported_frames, args.export_fps)
        elif args.mode == "gmr-bvh":
            timeline = gmr_bvh(*read_bvh_clock(args.bvh), args.exported_frames, args.export_fps)
        elif args.mode == "umr":
            timeline = read_umr(args.npz, args.source_fps)
        else:
            timeline = read_frdv(args.bvh, args.npz, args.start, args.end)
        if args.csv:
            writer = csv.writer(sys.stdout)
            writer.writerow(("export_index", "source_index", "source_time_s", "nominal_export_time_s"))
            writer.writerows(timeline.rows())
        else:
            print(json.dumps(timeline.summary(), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
