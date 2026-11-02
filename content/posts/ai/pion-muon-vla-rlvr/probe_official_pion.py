"""Probe the pinned official BF16 polynomial on CPU; requires PyTorch.

Pass VLA/VLAAdapter/pion_optim/muon.py from commit 87965014b360bde604632cba0aa12e423c6c63c6.
The source hash is checked before importing it. This is not GPU training or a
cross-device bitwise reproducibility promise.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

SOURCE_SHA256 = "8e7ca8d6f40ebc4fb066acb15387e618d515896cca5d1eb0ad234e5310e9157e"
COMMIT = "87965014b360bde604632cba0aa12e423c6c63c6"


def run(source):
    source = Path(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official source does not match the pinned Pion commit")
    import torch
    from pion_matrix import transform
    import numpy as np
    torch.set_num_threads(1)
    specification = importlib.util.spec_from_file_location("pinned_pion_probe", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    rows = []
    for scale in [1., 1e-8, 1e-200]:
        matrix = np.diag([.8, .4, .2, .02])*scale
        actual = module.high_pass_ns(torch.tensor(matrix, device="cpu"), 1, 4, epsilon=1e-7)
        rows.append({"input_scale": scale,
                     "ideal_float64_epsilon0": np.diag(transform(matrix, 1, 0.)).tolist(),
                     "float64_epsilon1e_7": np.diag(transform(matrix, 1, 1e-7)).tolist(),
                     "official_cpu_bfloat16_epsilon1e_7": actual.double().diag().tolist()})
    # Integer outer product: exactly rank one before floating-point operations.
    # Per-entry BF16 rounding can change the singular subspace as well as size.
    matrix = np.outer([1., 2., 3.], [1., -2., 3., -4.])
    unit = matrix / math.hypot(*matrix.flat)
    rectangle = []
    for kp in [0, 1, 2]:
        actual = module.high_pass_ns(torch.tensor(matrix, device="cpu"), kp, 5-kp, epsilon=1e-7).double().numpy()
        parallel = float(np.sum(actual*unit))
        rectangle.append({"promotion_steps": kp, "suppression_steps": 5-kp,
                          "float64_output": transform(matrix, kp, 1e-7).tolist(),
                          "official_cpu_bfloat16_output": actual.tolist(),
                          "official_singular_values_float64_svd": np.linalg.svd(actual, compute_uv=False).tolist(),
                          "residual_from_input_direction_frobenius": math.hypot(*(actual-parallel*unit).flat)})
    return {"scope": "Pinned polynomial helper on CPU, not model training or GPU accuracy",
            "official_commit": COMMIT, "official_source_sha256": SOURCE_SHA256,
            "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
            "threads": torch.get_num_threads(), "promotion_steps": 1,
            "suppression_steps": 4, "rows": rows,
            "rank_one_rectangle": {"input": matrix.tolist(), "rows": rectangle}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.official_file)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
