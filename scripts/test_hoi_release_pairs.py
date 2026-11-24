#!/usr/bin/env python3
"""Check the two-row CoRoleHOI CLI against known offsets and signed rotations."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "content/posts/robotics/contact-preserving-retargeting/inspect_hoi_release.py"
)


def write_row(path, row, truncated=()):
    path.write_text(
        json.dumps({"rows": [{"row": row, "truncated_cells": list(truncated)}]}),
        encoding="utf-8",
    )


def run(first, second):
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), str(first), "--partner", str(second)],
        check=False, capture_output=True, text=True,
    )


def check():
    first = {
        "schema_version": 2,
        "dataset": "corolehoi",
        "robot": "fixture",
        "sequence_id": "event",
        "clip_id": "actor-a",
        "partner_clip_id": "actor-b",
        "n_actors": 2,
        "pair_composable": True,
        "partner_z_offset_cm": 2,
        "n_frames": 3,
        "fps": 30,
        "duration_s": 0.1,
        "dof_names": ["joint"],
        "contact_link_names": ["palm"],
        "object_model_path": "box.urdf",
        "object_scale": 0.83,
        "object_aug_scale": 1,
        "scale_object_mesh": True,
        "scale_object_trajectory": True,
        "qc_pass": True,
        "qc_flags": [],
        "root_pos": [[0, 0, 1]] * 3,
        "root_rot": [[1, 0, 0, 0]] * 3,
        "object_pos": [[0, 0, 1], [0.1, 0, 1], [0.2, 0, 1]],
        "object_rot": [[1, 0, 0, 0]] * 3,
        "dof_pos": [[0]] * 3,
        "per_link_contact_flags": [[False]] * 3,
        "fixed_contact_points_per_frame_in_object_frame": [[[None, None, None]]] * 3,
    }
    second = copy.deepcopy(first)
    second.update(
        clip_id="actor-b",
        partner_clip_id="actor-a",
        partner_z_offset_cm=-2,
        object_pos=[[-0.03, 0, 0.98], [0.07, 0, 0.98], [0.18, 0, 0.98]],
        object_rot=[[-1, 0, 0, 0], [1, 0, 0, 0], [math.cos(math.pi / 12), 0, 0, math.sin(math.pi / 12)]],
    )
    with tempfile.TemporaryDirectory(prefix="hoi-release-pairs-") as directory:
        a, b = Path(directory) / "a.json", Path(directory) / "b.json"
        write_row(a, first)
        write_row(b, second)
        forward = run(a, b)
        reverse = run(b, a)
        assert forward.returncode == reverse.returncode == 0, (forward.stderr, reverse.stderr)
        pair = json.loads(forward.stdout)["pair"]
        assert pair["partner_clip_id"] == "actor-b"
        assert pair["published_partner_to_reference_z_m"] == 0.02
        errors = pair["object_position_error_m"]
        assert math.isclose(errors["after_published_z_shift"]["max"], 0.03)
        assert math.isclose(errors["after_first_frame_xyz_shift"]["max"], 0.01)
        assert math.isclose(errors["after_first_frame_xyz_shift"]["mean"], 0.01 / 3)
        assert math.isclose(pair["object_rotation_error_deg"]["mean"], 10)
        assert math.isclose(pair["object_rotation_error_deg"]["max"], 30)
        pair_reverse = json.loads(reverse.stdout)["pair"]
        assert pair_reverse["published_partner_to_reference_z_m"] == -0.02
        assert pair_reverse["first_frame_partner_to_reference_xyz_m"] == [
            -value for value in pair["first_frame_partner_to_reference_xyz_m"]
        ]
        assert pair_reverse["object_position_error_m"] == errors
        for counterexample, message in (
            ({"robot": "other"}, "robot"),
            ({"fps": 60, "duration_s": 0.05}, "fps"),
            ({"object_scale": 1}, "object_scale"),
            ({"partner_clip_id": "missing"}, "partner_clip_id"),
            ({"partner_z_offset_cm": 2}, "opposite"),
            ({"pair_composable": False}, "pair_composable"),
        ):
            write_row(b, {**second, **counterexample})
            failure = run(a, b)
            assert failure.returncode == 1 and not failure.stdout and message in failure.stderr
        write_row(b, second, truncated=("object_rot",))
        failure = run(a, b)
        assert failure.returncode == 1 and not failure.stdout and "all cells present" in failure.stderr
        write_row(a, {**first, "partner_z_offset_cm": 4, "pair_composable": False})
        write_row(b, {**second, "partner_z_offset_cm": -4, "pair_composable": False,
                      "object_pos": [[x, y, z - 0.02] for x, y, z in second["object_pos"]]})
        flagged = run(a, b)
        assert flagged.returncode == 0 and json.loads(flagged.stdout)["pair"]["dataset_pair_composable"] is False
    print("CoRoleHOI paired release checks passed.")


if __name__ == "__main__":
    check()
