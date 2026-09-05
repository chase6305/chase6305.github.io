#!/usr/bin/env python3
"""Third-pass CPU tests execute article snippets, not independent replacement examples.

Use a disposable environment with numpy, torch, cv2, tqdm, matplotlib, toppra,
ruckig and pytorch-kinematics. No robot, network service or GUI is used.
"""
import argparse
import ast
import io
import json
import math
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from validate_blog import ROOT, code_blocks


def blocks(post):
    return [code for lang, code, _ in code_blocks(
        (ROOT / "content/posts" / post).read_text()) if lang == "python"]


def run(source, values=None, definitions=False):
    values = {"__name__": "blog_test"} if values is None else values
    tree = ast.parse(source)
    if definitions:
        tree.body = [node for node in tree.body if isinstance(
            node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    exec(compile(tree, "<article>", "exec"), values)
    return values


def expect_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("invalid input was not rejected")


def trajectory():
    import numpy as np
    ns = run(blocks("trajectory/ruckig/index.md")[0])
    for target in ([1.0], [-0.1], [0.0], [1, .5, .25, 0, -1, -.5, -.25]):
        ns["simulate"](target)
    expect_error(ns["simulate"], [1], 0)
    expect_error(ns["simulate"], [float("nan")])
    ns = run(blocks("trajectory/toppra/index.md")[0])
    durations = [ns["plan_and_check"](n)[0][-1] for n in (101, 201, 401)]
    assert max(durations) - min(durations) < .02
    expect_error(ns["plan_and_check"], 1)
    return {"ruckig": version("ruckig"), "toppra": version("toppra"),
            "toppra_durations": list(map(float, durations)), "ruckig_targets": 4}


def calibration():
    import numpy as np
    handeye = run(blocks("calibration/model/index.md")[0])
    zero = run(blocks("calibration/zero/index.md")[0])
    q = np.array([[.4, .8], [-.3, 1.2]])
    _, jacobian = zero["fk_and_jacobian"](q)
    for j in range(2):
        delta = np.zeros_like(q)
        delta[:, j] = 1e-6
        numeric = (zero["fk_and_jacobian"](q + delta)[0] -
                   zero["fk_and_jacobian"](q - delta)[0]) / 2e-6
        np.testing.assert_allclose(numeric, jacobian[:, :, j], atol=1e-9)
    _, degenerate = zero["fk_and_jacobian"](np.zeros((10, 2)))
    assert np.linalg.matrix_rank(degenerate.reshape(-1, 2)) == 1
    # Independently verify the article's eye-to-hand AZ=ZB relation.
    G, inv = handeye["G"], handeye["rigid_inverse"]
    X, Z = handeye["Y_true"], handeye["X_true"]
    C = [inv(X) @ g @ Z for g in G]
    A, B = inv(G[1]) @ G[0], inv(C[1]) @ C[0]
    np.testing.assert_allclose(A @ Z, Z @ B, atol=1e-12)
    for g, c in zip(G, C):
        np.testing.assert_allclose(g @ Z @ inv(c), X, atol=1e-12)
    return {"opencv": handeye["cv2"].__version__, "handeye_holdout": 6,
            "zero_holdout": 10, "parameter_jacobian": "central difference passed",
            "degenerate_pose": "rank deficiency detected"}


def python():
    import numpy as np
    first, second = blocks("python/tqdm/index.md")
    ns = run(first)
    assert ns["process_items"]([]) == []
    assert ns["process_items"]([1, 2, 3]) == [1, 4, 9]
    assert ns["process_batches"]([[1, 2], [], [3]]) == 3
    assert ns["process_batches"]([]) == 0
    assert len(ns["nested"]()) == 6
    run(second, ns)
    with tempfile.TemporaryDirectory(prefix="chase-round3-bytes-") as folder:
        path = Path(folder) / "sample.bin"
        for data in (b"", "汉字\nabc".encode("utf-8"), bytes(range(256))):
            path.write_bytes(data)
            assert ns["scan_bytes"](path, 3) == len(data)
        expect_error(ns["scan_bytes"], path, 0)
    # Exercise actual progress increments with a deterministic non-terminal sink.
    sink = io.StringIO()
    with ns["tqdm"](total=3, file=sink, disable=False) as bar:
        bar.update(2)
        bar.update(1)
        assert bar.n == 3
    workspace = blocks("robotics/workspace/whole-workspace/index.md")[0]
    ns = run(workspace.split("# 绘制可达空间")[0])
    assert ns["positions"].shape == (10000, 2)
    for count in (1, 17, 100000):
        rng = np.random.default_rng(count)
        q = rng.uniform(-np.pi, np.pi, (count, 2))
        xy = np.column_stack(ns["forward_kinematics"](q[:, 0], q[:, 1]))
        r = np.linalg.norm(xy, axis=1)
        assert np.all((r >= .4-1e-12) & (r <= 1.6+1e-12))
    return {"tqdm": version("tqdm"), "byte_cases": 3,
            "workspace_samples": 110018, "invalid_inputs": "passed"}


def torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(42)
    ns = {"__name__": "blog_test", "torch": torch, "nn": nn, "F": F, "math": math}
    source = blocks("ai/transformer-attention/index.md")
    for marker in ("class SimpleAttention", "class SimpleMultiHeadAttention",
                   "class InputEmbedding", "class FeedForward", "class TransformerBlock",
                   "class MiniGPT", "def make_lm_batch"):
        run(next(c for c in source if marker in c), ns, definitions=True)
    single = ns["SimpleAttention"](8).double()
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    manual, _ = single(x, causal=True)
    builtin, _ = single(x, causal=True, use_sdpa=True)
    torch.testing.assert_close(manual, builtin, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(torch.autograd.grad(manual.sum(), x)[0],
                               torch.autograd.grad(builtin.sum(), x)[0], atol=1e-10, rtol=1e-10)
    multi = ns["SimpleMultiHeadAttention"](8, 2).double()
    valid = torch.tensor([[False, False, True, True], [False]*4])
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    out, attn = multi(x, causal=True, valid_tokens=valid, return_attn=True)
    assert torch.isfinite(out).all() and torch.isfinite(attn).all()
    assert torch.count_nonzero(out[~valid]) == 0
    assert torch.count_nonzero(attn[1]) == 0
    assert torch.count_nonzero(attn.triu(1)) == 0
    torch.testing.assert_close(attn[0, :, 2:].sum(-1), torch.ones(2, 2, dtype=torch.float64))
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in multi.parameters())
    expect_error(ns["SimpleMultiHeadAttention"], 8, 0)
    expect_error(ns["make_lm_batch"], [], 0)
    expect_error(ns["make_lm_batch"], [[1]], 0)
    model = ns["MiniGPT"](40, 16, 8, 2, 16, 1, dropout=0).double()
    batch = ns["make_lm_batch"]([[1, 2, 3, 4], [1, 3]], 0)
    logits, loss = model(batch[0], targets=batch[1], valid_tokens=batch[2])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    expect_error(model, batch[0], torch.full_like(batch[0], -100), batch[2])
    model.eval()
    left = torch.tensor([[0, 0, 1, 3]])
    mask = left != 0
    left_logits, _ = model(left, valid_tokens=mask)
    plain_logits, _ = model(torch.tensor([[1, 3]]))
    torch.testing.assert_close(left_logits[:, 2:], plain_logits, atol=1e-10, rtol=1e-10)
    # Both causal implementations must ignore perturbations to future inputs.
    a = torch.randn(1, 4, 8, dtype=torch.float64)
    b = a.clone(); b[:, 2:] += 100
    torch.testing.assert_close(multi(a, causal=True)[:, :2], multi(b, causal=True)[:, :2])

    # The full DDPM example is smoke-tested for 20 CPU training steps and sampling.
    # This does not establish convergence or quality of the 3000-step experiment.
    diffusion = next(c for c in blocks("ai/diffusion-models/index.md") if "class NoisePredictor" in c)
    diffusion = diffusion.split("real = sample_data(4000)")[0]
    diffusion = diffusion.replace('device = "cuda" if torch.cuda.is_available() else "cpu"', 'device = "cpu"')
    diffusion = diffusion.replace("range(3000)", "range(20)")
    ns = run(diffusion)
    generated = ns["sample"](ns["model"], count=16)
    assert generated.shape == (16, 2) and torch.isfinite(generated).all()
    assert torch.all(ns["beta"] > 0) and torch.all(ns["beta"] < 1)
    assert torch.all(ns["alpha_bar"].diff() < 0)
    assert ns["posterior_var"][0] == 0 and torch.all(ns["posterior_var"] >= 0)
    final_signal = float(ns["alpha_bar"][-1])
    memory = run(blocks("ai/distributed-training-memory/index.md")[0])
    estimate = memory["model_state_gb"]
    assert list(estimate(1, 8).values()) == [16, 5.5, 3.75, 2]
    assert len(set(estimate(1, 1).values())) == 1
    for world_size in (0, -1, 1.5, True):
        expect_error(estimate, 1, world_size)
    expect_error(estimate, -1, 8)
    return {"torch": torch.__version__, "attention": "SDPA output/gradient, padding, causal and LM tests passed",
            "diffusion_smoke_steps": 20, "diffusion_final_alpha_bar": final_signal,
            "memory_estimator": "numbers and invalid inputs passed"}


def kinematics():
    import torch
    source = blocks("robotics/kinematics/pytorch/index.md")[0]
    old_argv = sys.argv
    try:
        sys.argv = ["check_ik.py", str(ROOT / "content/posts/robotics/kinematics/pytorch/planar2r.urdf"), "tip"]
        ns = run(source)
    finally:
        sys.argv = old_argv
    chain = ns["chain"]
    q = torch.tensor([[.3, .7]], dtype=torch.float64, requires_grad=True)
    analytic = chain.jacobian(q)[0, :3]
    numeric = torch.autograd.functional.jacobian(
        lambda values: chain.forward_kinematics(values).get_matrix()[0, :3, 3], q)[:, 0]
    torch.testing.assert_close(analytic, numeric, atol=1e-10, rtol=1e-10)
    return {"pytorch_kinematics": version("pytorch-kinematics"),
            "fk_ik_fk": "finite joint-limit fixture passed", "jacobian": "autograd comparison passed"}


def diffusion():
    """Optional full teaching experiment, separate from the quick regression suite."""
    import matplotlib
    import torch
    matplotlib.use("Agg")
    torch.set_num_threads(2)
    source = next(c for c in blocks("ai/diffusion-models/index.md") if "class NoisePredictor" in c)
    source = source.replace('device = "cuda" if torch.cuda.is_available() else "cpu"', 'device = "cpu"')
    directory = tempfile.mkdtemp(prefix="chase-diffusion-round3-")
    previous = Path.cwd()
    try:
        os.chdir(directory)
        ns = run(source)
    finally:
        os.chdir(previous)
    x = ns["generated"]
    assert x.shape == (4000, 2) and torch.isfinite(x).all()
    radii = torch.linalg.vector_norm(x, dim=1)
    return {"torch": torch.__version__, "device": "cpu", "seed": 0,
            "training_steps": 3000, "samples": len(x),
            "mean_radius": float(radii.mean()), "median_radius": float(radii.median()),
            "last_loss": float(ns["loss"].detach()),
            "plot": str(Path(directory) / "diffusion_2d_result.png")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "trajectory", "calibration", "python", "torch", "kinematics", "diffusion"], default="all", nargs="?")
    args = parser.parse_args()
    modes = ["trajectory", "calibration", "python", "torch", "kinematics"] if args.mode == "all" else [args.mode]
    results = {mode: globals()[mode]() for mode in modes}
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
