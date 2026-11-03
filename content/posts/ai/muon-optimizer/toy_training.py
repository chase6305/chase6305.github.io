"""A complete CPU Muon + AdamW example on synthetic regression data.

Requires PyTorch with torch.optim.Muon; tested with 2.14.1. No dataset download,
GPU, or NumPy is needed. Input projection and output head use AdamW; only the
hidden matrix uses Muon. This checks integration, not optimizer superiority.
"""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import tempfile


def torch_runtime():
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Install the article's PyTorch CPU environment first.") from error
    if not hasattr(torch.optim, "Muon"):
        raise SystemExit("This example needs torch.optim.Muon; tested with PyTorch 2.14.1.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def make_data(torch):
    generator = torch.Generator().manual_seed(20261103)
    x = torch.randn(640, 6, generator=generator)
    a = torch.randn(6, 8, generator=generator)/6**0.5
    b = torch.randn(8, 3, generator=generator)/8**0.5
    y = torch.tanh(x@a)@b + 0.02*torch.randn(640, 3, generator=generator)
    return (x[:512], y[:512]), (x[512:], y[512:])


def make_model(torch):
    return torch.nn.Sequential(OrderedDict([
        ("input", torch.nn.Linear(6, 12)),
        ("input_activation", torch.nn.Tanh()),
        ("hidden", torch.nn.Linear(12, 12)),
        ("hidden_activation", torch.nn.Tanh()),
        ("head", torch.nn.Linear(12, 3)),
    ]))


def optimizers(torch, model):
    # Explicit module role, rather than selecting every tensor with ndim == 2.
    selected = [model.hidden.weight]
    selected_ids = {id(p) for p in selected}
    if selected_ids & {id(model.input.weight), id(model.head.weight)}:
        raise ValueError("hidden matrix shares an input/output role; choose one optimizer policy explicitly")
    remaining = [p for p in model.parameters() if id(p) not in selected_ids]
    all_ids = [id(p) for p in selected+remaining]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != {id(p) for p in model.parameters()}:
        raise ValueError("optimizer groups must cover every parameter exactly once")
    muon = torch.optim.Muon(selected, lr=0.01, weight_decay=0.01,
                            momentum=0.95, nesterov=True,
                            adjust_lr_fn="match_rms_adamw")
    adamw = torch.optim.AdamW(remaining, lr=0.01, weight_decay=0.01)
    return muon, adamw


def evaluate(torch, model, data):
    with torch.no_grad():
        return float((model(data[0])-data[1]).square().mean())


def backward_batch(model, data, indices, microbatch):
    # Each MSE averages over rows and the same three output coordinates.
    # Weight by actual row count, including the shorter final microbatch.
    for chunk in indices.split(microbatch):
        loss = (model(data[0][chunk])-data[1][chunk]).square().mean()
        (loss * (len(chunk)/len(indices))).backward()


def advance(torch, model, opts, data, sampler, start, end, batch, microbatch):
    muon, adamw = opts
    last = {}
    for _ in range(start, end):
        indices = torch.randperm(len(data[0]), generator=sampler)[:batch]
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        backward_batch(model, data, indices, microbatch)
        before = model.hidden.weight.detach().clone()
        grad_rms = float(model.hidden.weight.grad.square().mean().sqrt())
        muon.step()
        adamw.step()
        with torch.no_grad():
            delta = model.hidden.weight-before
            group = muon.param_groups[0]
            decay_delta = -group["lr"]*group["weight_decay"]*before
            last = {"hidden_gradient_rms": grad_rms,
                    "hidden_gradient_update_rms": float((delta-decay_delta).square().mean().sqrt()),
                    "hidden_total_relative_update": float(delta.norm()/before.norm())}
    return last


def maximum_difference(torch, first, second):
    with torch.no_grad():
        return max(float((a-b).abs().max()) for a, b in zip(first.parameters(), second.parameters()))


def parameter_digest(torch, model):
    """CPU byte fingerprint without requiring a NumPy installation."""
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in model.named_parameters():
            digest.update(name.encode("utf-8"))
            digest.update(bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def checkpoint_payload(torch, model, opts, sampler, step, batch, microbatch):
    return {"format": "muon-toy-v1", "data": "synthetic-regression-v1",
            "torch": str(torch.__version__), "model": model.state_dict(),
            "muon": opts[0].state_dict(), "adamw": opts[1].state_dict(),
            "sampler": sampler.get_state(), "step": step,
            "batch": batch, "microbatch": microbatch}


def restore_checkpoint(torch, path, *, restore_muon=True):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "muon-toy-v1" or saved.get("data") != "synthetic-regression-v1":
        raise ValueError("checkpoint format or synthetic dataset version does not match")
    model = make_model(torch)
    opts = optimizers(torch, model)
    sampler = torch.Generator()
    model.load_state_dict(saved["model"])
    if restore_muon:
        opts[0].load_state_dict(saved["muon"])
    opts[1].load_state_dict(saved["adamw"])
    sampler.set_state(saved["sampler"])
    return model, opts, sampler, saved


def save_training(path, steps=80, batch=32, microbatch=13):
    """Save after 'steps' completed updates; refuse to overwrite any file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("checkpoint already exists: " + str(path))
    if steps < 1 or not 1 <= batch <= 512 or microbatch < 1:
        raise ValueError("require steps >= 1, 1 <= batch <= 512, microbatch >= 1")
    torch = torch_runtime()
    torch.manual_seed(31)
    train, validation = make_data(torch)
    model = make_model(torch)
    opts = optimizers(torch, model)
    sampler = torch.Generator().manual_seed(403)
    advance(torch, model, opts, train, sampler, 0, steps, batch, microbatch)
    # Exclusive creation protects a reader's pre-existing checkpoint too.
    with path.open("xb") as stream:
        torch.save(checkpoint_payload(torch, model, opts, sampler, steps, batch, microbatch), stream)
    return {"mode": "save", "checkpoint": str(path.resolve()), "completed_steps": steps,
            "batch": batch, "microbatch": microbatch,
            "parameter_sha256": parameter_digest(torch, model),
            "train_mse": evaluate(torch, model, train), "validation_mse": evaluate(torch, model, validation)}


def resume_training(path, steps=160, batch=None, microbatch=None):
    """Continue in a fresh process to an absolute completed-update count."""
    torch = torch_runtime()
    model, opts, sampler, saved = restore_checkpoint(torch, path)
    if steps < saved["step"]:
        raise ValueError("target steps precede the saved position")
    if (batch is not None and batch != saved["batch"]) or (microbatch is not None and microbatch != saved["microbatch"]):
        raise ValueError("this reproducibility example preserves the saved batch and microbatch")
    train, validation = make_data(torch)
    advance(torch, model, opts, train, sampler, saved["step"], steps, saved["batch"], saved["microbatch"])
    return {"mode": "resume", "completed_steps": steps, "continued_steps": steps-saved["step"],
            "batch": saved["batch"], "microbatch": saved["microbatch"],
            "parameter_sha256": parameter_digest(torch, model),
            "train_mse": evaluate(torch, model, train), "validation_mse": evaluate(torch, model, validation)}


def run(steps=160, batch=32, microbatch=13):
    if steps < 2 or not 1 <= batch <= 512 or microbatch < 1:
        raise ValueError("require steps >= 2, 1 <= batch <= 512, microbatch >= 1")
    torch = torch_runtime()
    torch.manual_seed(31)
    train, validation = make_data(torch)
    model = make_model(torch)
    opts = optimizers(torch, model)
    sampler = torch.Generator().manual_seed(403)
    initial = {"train": evaluate(torch, model, train),
               "validation": evaluate(torch, model, validation)}
    split = steps//2
    advance(torch, model, opts, train, sampler, 0, split, batch, microbatch)
    with tempfile.TemporaryDirectory(prefix="muon-checkpoint-") as directory:
        checkpoint = Path(directory)/"checkpoint.pt"
        # Store primitives and tensors: compatible with weights_only=True.
        torch.save(checkpoint_payload(torch, model, opts, sampler, split, batch, microbatch), checkpoint)
        last = advance(torch, model, opts, train, sampler, split, steps, batch, microbatch)
        restored, restored_opts, restored_sampler, saved = restore_checkpoint(torch, checkpoint)
        advance(torch, restored, restored_opts, train, restored_sampler, saved["step"], steps,
                saved["batch"], saved["microbatch"])
        resume_difference = maximum_difference(torch, model, restored)

        # A model checkpoint alone does not restore Muon's momentum history.
        # load_state_dict can reuse state tensors; re-read the original file so
        # the preceding restored run cannot change this counterfactual's state.
        fresh_momentum, fresh_opts, fresh_sampler, saved = restore_checkpoint(torch, checkpoint, restore_muon=False)
        advance(torch, fresh_momentum, fresh_opts, train, fresh_sampler, saved["step"], steps,
                saved["batch"], saved["microbatch"])
        missing_momentum_difference = maximum_difference(torch, model, fresh_momentum)

    final = {"train": evaluate(torch, model, train),
             "validation": evaluate(torch, model, validation)}
    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    if not finite or resume_difference != 0 or final["train"] >= initial["train"]:
        raise ArithmeticError("toy integration check failed")
    inventory = [{"name": name, "shape": list(p.shape),
                  "optimizer": "Muon" if id(p) == id(model.hidden.weight) else "AdamW"}
                 for name, p in model.named_parameters()]
    return {"torch": str(torch.__version__), "device": "cpu", "parameter_dtype": "float32",
            "steps": steps, "batch": batch, "microbatch": microbatch,
            "processed_training_rows": steps*batch, "initial_mse": initial, "final_mse": final,
            "last_step": last, "resume_max_parameter_difference": resume_difference,
            "parameter_sha256": parameter_digest(torch, model),
            "missing_muon_momentum_max_difference": missing_momentum_difference,
            "parameter_groups": inventory,
            "scope": "synthetic integration example; no optimizer ranking or GPU reproducibility claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--microbatch", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--save-checkpoint", type=Path, help="save after --steps updates; path must not exist")
    mode.add_argument("--resume", type=Path, help="continue to an absolute --steps count using the saved batch settings")
    args = parser.parse_args()
    fresh_batch = 32 if args.batch is None else args.batch
    fresh_microbatch = 13 if args.microbatch is None else args.microbatch
    if args.resume is not None:
        result = resume_training(args.resume, args.steps, args.batch, args.microbatch)
    elif args.save_checkpoint is not None:
        result = save_training(args.save_checkpoint, args.steps, fresh_batch, fresh_microbatch)
    else:
        result = run(args.steps, fresh_batch, fresh_microbatch)
    print(json.dumps(result, indent=2, allow_nan=False))
