"""Check pinned per-head Pion routing and scaling with rank-one CPU blocks.

Constructed inputs isolate geometry and optimizer ordering from general-matrix
BF16 rounding. No model, attention forward pass or distributed shard is used.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[3] /
                        "content/posts/ai/pion-muon-vla-rlvr")
    args = parser.parse_args()
    try:
        digest = hashlib.sha256(args.official_file.read_bytes()).hexdigest()
    except OSError as error:
        parser.error(str(error))
    if digest != "15c1331453ef3db4b565a514d1e43f74692e2e910eb8cb939894d54371dd402d":
        parser.error("Official RL source does not match pinned Pion commit 8796501")
    import numpy as np
    import torch
    sys.path.insert(0, str(args.bundle))
    from pion_step import step

    spec = importlib.util.spec_from_file_location("pinned_pion_rl", args.official_file)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    torch.set_num_threads(1)
    layouts = [("Q", (8, 6), 4, 0), ("K/V", (4, 6), 2, 0),
               ("O", (6, 8), 4, 1), ("tall rows", (6, 4), 3, 0),
               ("wide columns", (4, 6), 3, 1),
               ("square rows", (4, 4), 2, 0), ("square columns", (4, 4), 2, 1),
               ("row fallback", (5, 4), 2, 0), ("column fallback", (4, 5), 2, 1)]
    cases = []
    for label, shape, heads, axis in layouts:
        compatible = shape[axis] % heads == 0
        count = heads if compatible else 1
        block_width = shape[axis] // heads if compatible else shape[axis]
        observations = []
        for values in ([.8, -.4, .2, 0.], [0., 0., 0., 0.], None,
                       [-1.6, .8, -.4, .2]):
            if values is None:
                observations.append(None)
                continue
            gradient = np.zeros(shape)
            for head in range(count):
                index = (head * block_width, 0) if axis == 0 else (0, head * block_width)
                gradient[index] = values[head]
            observations.append(gradient)
        for beta in (.5, .95):
            for nesterov in (False, True):
                for kp in (0, 1, 5):
                    weight = np.arange(1, 1 + np.prod(shape), dtype=float).reshape(shape) / 10
                    buffer = np.zeros(shape)
                    parameter = torch.nn.Parameter(torch.from_numpy(weight.copy()))
                    optimizer = official.PerHeadPionAdamW(
                        [parameter], lr=.01, weight_decay=.1, momentum=beta,
                        nesterov=nesterov, promotion_steps=kp, ns_steps=5,
                        scale_factor=.2, num_heads=heads, head_split_dim=axis)
                    rows = []
                    for number, gradient in enumerate(observations, 1):
                        parameter.grad = None if gradient is None else torch.from_numpy(gradient.copy())
                        optimizer.step()
                        slices = [slice(None)] * 2
                        parts, buffers, scales = [], [], []
                        for head in range(count):
                            slices[axis] = slice(head * block_width, (head + 1) * block_width)
                            region = tuple(slices)
                            after, current, trace = step(
                                weight[region], None if gradient is None else gradient[region],
                                buffer[region], beta=beta, nesterov=nesterov,
                                promotion_steps=kp, epsilon=1e-7)
                            parts.append(after)
                            buffers.append(current)
                            scales.append(trace["dimension_scale"])
                        weight = np.concatenate(parts, axis=axis)
                        buffer = np.concatenate(buffers, axis=axis)
                        actual_weight = parameter.detach().numpy()
                        actual_buffer = optimizer.state[parameter]["momentum_buffer"].numpy()
                        np.testing.assert_allclose(actual_weight, weight, rtol=0, atol=4e-15)
                        np.testing.assert_allclose(actual_buffer, buffer, rtol=0, atol=2e-16)
                        if optimizer.state[parameter]["step"] != number - (number >= 3):
                            raise RuntimeError("Missing gradient changed the optimizer step counter")
                        rows.append({"step": number, "dimension_scales": scales,
                                     "max_weight_difference": float(np.max(np.abs(actual_weight - weight))),
                                     "max_buffer_difference": float(np.max(np.abs(actual_buffer - buffer)))})
                    cases.append({"layout": label, "shape": shape, "heads": heads, "axis": axis,
                                  "per_head": compatible, "beta": beta, "nesterov": nesterov,
                                  "promotion_steps": kp, "steps": rows})
    result = {"scope": __doc__, "official_source_sha256": digest,
              "official_commit": "87965014b360bde604632cba0aa12e423c6c63c6",
              "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
              "device": "cpu", "parameter_dtype": "float64", "internal_dtype": "bfloat16",
              "lr": .01, "weight_decay": .1, "scale_factor": .2, "epsilon": 1e-7,
              "reference_sha256": {name: hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
                                   for name in ("pion_step.py", "pion_matrix.py", "pion_spectral.py")},
              "cases": cases}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"configurations": len(cases), "calls": sum(len(c["steps"]) for c in cases),
                      "max_weight_difference": max(r["max_weight_difference"] for c in cases for r in c["steps"]),
                      "max_buffer_difference": max(r["max_buffer_difference"] for c in cases for r in c["steps"])}))


if __name__ == "__main__":
    main()
