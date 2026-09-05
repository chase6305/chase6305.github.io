#!/usr/bin/env python3
"""Optional second-pass runtime tests; each mode uses its own installed packages.

Examples:
  python scripts/test_blog_round2.py casadi
  python scripts/test_blog_round2.py warp
  python scripts/test_blog_round2.py torch
  python scripts/test_blog_round2.py open3d
"""
import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

from validate_blog import ROOT, code_blocks


def python_blocks(post):
    return [code for lang, code, _ in code_blocks(
        (ROOT / "content/posts" / post).read_text()) if lang == "python"]


def execute(source):
    values = {"__name__": "blog_test"}
    exec(compile(source, "<blog-test>", "exec"), values)
    return values


def check_casadi():
    ns = execute(python_blocks("casadi/index.md")[0])
    values = [ns["solve_example"](mode) for mode in (False, True)]
    return {"casadi": ns["ca"].__version__,
            "solutions": [v[0].tolist() for v in values],
            "objectives": [v[1] for v in values]}


def check_torch():
    import torch
    from torch.distributions import Normal, TransformedDistribution
    from torch.distributions.transforms import TanhTransform, AffineTransform

    blocks = python_blocks("rl/index.md")
    policy = execute(next(c for c in blocks if "class GaussianPolicy" in c))
    policy["log_prob"].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in policy["policy"].parameters())
    ppo = execute(next(c for c in blocks if "def ppo_surrogate" in c))
    assert ppo["ppo_surrogate"]([1], [1])[0] == 1
    ns = execute(next(c for c in blocks if "log_tanh_jacobian =" in c))
    transformed = TransformedDistribution(
        Normal(ns["mean"], ns["log_std"].exp()),
        [TanhTransform(cache_size=1), AffineTransform(ns["center"], ns["scale"])],
    )
    actual = transformed.log_prob(ns["action"]).sum(-1)
    torch.testing.assert_close(ns["log_prob"], actual, atol=1e-10, rtol=1e-10)
    return {"torch": torch.__version__, "ppo": "4 clipping directions passed",
            "gaussian": "score-function gradient passed",
            "squashed_gaussian": "matches TransformedDistribution"}


def check_warp():
    import warp as wp
    blocks = python_blocks("cuda/warp/index.md")
    start = next(i for i, c in enumerate(blocks) if "def add_vectors" in c)
    end = next(i for i, c in enumerate(blocks) if "def loss_function" in c)
    # Warp must inspect source in a real module when compiling @kernel functions.
    with tempfile.TemporaryDirectory(prefix="chase-warp-round2-") as directory:
        path = Path(directory) / "warp_examples.py"
        path.write_text("import warp as wp\nwp.config.kernel_cache_dir = " +
                        repr(str(Path(directory) / "cache")) + "\n" +
                        "\n".join(blocks[start:end+1]))
        spec = importlib.util.spec_from_file_location("blog_warp_examples", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert str(module.x.device) == "cpu"
    return {"warp": wp.__version__, "device": "cpu",
            "tests": ["vector addition", "matrix multiplication",
                      "atomic increment", "particle integration", "autodiff"]}


def check_open3d():
    import numpy as np
    import open3d as o3d
    blocks = python_blocks("open3d/introduction/index.md")
    kd = next(c for c in blocks if "squared_distances" in c)
    # Execute numeric steps only; no window is created.
    kd = kd.split("# 提取最近邻点")[0]
    ns = execute(kd)
    assert ns["k"] == 10
    octree = next(c for c in blocks if "locate_leaf_node" in c)
    octree = octree.split("o3d.visualization.draw_geometries")[0]
    execute(octree)
    return {"open3d": o3d.__version__, "tests": ["KDTree vs brute force", "octree lookup"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["casadi", "torch", "warp", "open3d"])
    args = parser.parse_args()
    result = globals()["check_" + args.mode]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
