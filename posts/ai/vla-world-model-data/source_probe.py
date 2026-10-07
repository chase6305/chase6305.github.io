#!/usr/bin/env python3
"""Read two fixed upstream Git objects and run small, hardware-free probes.

Requires NumPy and local clones containing the commits below. Does not import
robot controllers, connect to devices, or modify either repository.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
import types
from pathlib import Path

import numpy as np

XR_COMMIT = "79e5cb8a56e3455515ce1b476e993c764ec58739"
UMI_COMMIT = "d095ba9590df789df5189eea5ee7e431689038a6"


def git_module(root: Path, commit: str, relative: str, name: str):
    # Read the pinned Git object, independent of current branch or working edits.
    source = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{relative}"],
                            check=True, capture_output=True).stdout
    module = types.ModuleType(name)
    exec(compile(source, f"{commit}:{relative}", "exec"), module.__dict__)
    return module, {"commit": commit, "path": relative,
                    "sha256": hashlib.sha256(source).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xr-root", required=True, type=Path)
    parser.add_argument("--umi-root", required=True, type=Path)
    args = parser.parse_args()
    logger_module, xr_source = git_module(args.xr_root, XR_COMMIT,
        "xrobotoolkit_teleop/common/data_logger.py", "fixed_logger")
    with tempfile.TemporaryDirectory(prefix="robot-data-logger-") as directory:
        logger = logger_module.DataLogger(log_dir=directory)
        entry = {"qpos": [0.1]}
        logger.add_entry(entry)
        timestamp_added = "timestamp" in logger.log_data[0]
        if timestamp_added:
            raise AssertionError("fixed logger behavior changed")
    pose_module, umi_source = git_module(args.umi_root, UMI_COMMIT,
        "diffusion_policy/common/pose_repr_util.py", "fixed_pose_repr")
    base = np.eye(4)
    base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    base[:3, 3] = [0.3, 0.2, 0.0]
    target = base.copy()
    target[:3, 3] = [0.4, 0.2, 0.0]
    convert = pose_module.convert_pose_mat_rep
    relative = convert(target, base, "relative")
    legacy = convert(target, base, "rel")
    wrong = convert(legacy, base, "relative", backward=True)
    proper_roundtrips = all(np.allclose(convert(value, base, mode, backward=True), target)
                            for mode, value in (("relative", relative), ("rel", legacy)))
    mixed_error = float(np.linalg.norm(wrong[:3, 3] - target[:3, 3]))
    if not proper_roundtrips or not np.isclose(mixed_error, np.sqrt(0.02)):
        raise AssertionError("fixed pose-conversion behavior changed")
    return {
        "scope": "fixed upstream functions with artificial inputs; no robot execution",
        "numpy_version": np.__version__,
        "sources": [xr_source, umi_source],
        "xr_logger_adds_timestamp": timestamp_added,
        "umi": {
            "relative_translation_m": relative[:3, 3].round(6).tolist(),
            "legacy_rel_translation_m": legacy[:3, 3].round(6).tolist(),
            "target_translation_m": target[:3, 3].round(6).tolist(),
            "mixed_decoder_translation_m": wrong[:3, 3].round(6).tolist(),
            "consistent_roundtrips_pass": proper_roundtrips,
            "mixed_translation_error_m": mixed_error,
        },
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2, allow_nan=False))
