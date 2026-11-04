"""Capture raw matrix gradients at a fixed synthetic CPU model position.

Use the complete two-article download or this repository checkout. Capture
requires the article's PyTorch environment; replay uses only the standard
library. Sampling is independent WITH replacement, before momentum or NS.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path


def local_module(name, path):
    if not path.is_file():
        raise ValueError("Missing local Muon example source; use the complete download: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def statistics_module():
    folder = Path(__file__).resolve().parent
    batch_folder = folder.parent/("batch" if (folder.parent/"muon-download.json").is_file()
                                  else "muon-learning-rate-batch-size")
    return local_module("muon_noise_statistics", batch_folder/"batch_math.py")


def count(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))


def collect_gradients(torch, model, parameter, data, *, batches=64, batch=16, seed=724):
    """Return independent sample-mean gradients and the sampled row indices.

    This example's loss averages rows and output coordinates. autograd.grad
    returns the selected gradient without accumulating into parameter.grad.
    The deterministic toy model has no cross-row operations or running buffers.
    """
    count(batches, "batches", 2)
    count(batch, "batch")
    count(seed, "seed", 0)
    if len(data[0]) == 0 or len(data[0]) != len(data[1]):
        raise ValueError("need equally sized, nonempty inputs and targets")
    if parameter.device.type != "cpu" or parameter.ndim != 2 or not parameter.requires_grad:
        raise ValueError("selected CPU matrix parameter must still require gradients")
    generator = torch.Generator().manual_seed(seed)
    gradients, sampled_indices = [], []
    for _ in range(batches):
        indices = torch.randint(len(data[0]), (batch,), generator=generator)
        loss = (model(data[0][indices])-data[1][indices]).square().mean()
        gradient = torch.autograd.grad(loss, parameter)[0].detach()
        if not bool(torch.isfinite(gradient).all()):
            raise ArithmeticError("captured gradient is not finite")
        gradients.append(gradient.to(dtype=torch.float64).reshape(-1).tolist())
        sampled_indices.append(indices.tolist())
    return gradients, sampled_indices


def capture(batches=64, batch=16, seed=724):
    folder = Path(__file__).resolve().parent
    toy = local_module("muon_noise_toy", folder/"toy_training.py")
    torch = toy.torch_runtime()
    torch.manual_seed(31)
    train, _ = toy.make_data(torch)
    model = toy.make_model(torch)
    before = toy.parameter_digest(torch, model)
    gradients, indices = collect_gradients(torch, model, model.hidden.weight, train,
                                          batches=batches, batch=batch, seed=seed)
    after = toy.parameter_digest(torch, model)
    if before != after:
        raise ArithmeticError("model parameters changed during gradient capture")
    return {
        "format": "muon-gradient-records-v1",
        "sampling": "iid-with-replacement", "loss": "mean-squared-error",
        "model": "synthetic-muon-mlp-v1", "layer": "hidden.weight",
        "torch": str(torch.__version__), "parameter_dtype": "float32",
        "parameter_shape": list(model.hidden.weight.shape),
        "parameter_sha256_before": before, "parameter_sha256_after": after,
        "dataset_rows": len(train[0]), "batch_size": batch,
        "batch_count": batches, "sampling_seed": seed,
        "gradients": gradients, "sampled_indices": indices,
    }


def validate_record(record):
    if not isinstance(record, dict) or record.get("format") != "muon-gradient-records-v1":
        raise ValueError("unknown gradient record format")
    if record.get("sampling") != "iid-with-replacement":
        raise ValueError("this estimator example requires IID sampling with replacement")
    if not all(isinstance(record.get(key), str) and record[key] for key in ("layer", "torch")):
        raise ValueError("layer and recorded torch version must be nonempty strings")
    for key, minimum in (("dataset_rows", 1), ("batch_size", 1), ("batch_count", 2)):
        count(record.get(key), key, minimum)
    shape = record.get("parameter_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("parameter_shape must contain two positive dimensions")
    for size in shape:
        count(size, "parameter dimension")
    gradients, indices = record.get("gradients"), record.get("sampled_indices")
    if not isinstance(gradients, list) or not isinstance(indices, list) or len(gradients) != record["batch_count"] or len(indices) != len(gradients):
        raise ValueError("record counts do not match batch_count")
    for gradient, rows in zip(gradients, indices):
        if not isinstance(gradient, list) or len(gradient) != math.prod(shape):
            raise ValueError("gradient size does not match parameter_shape")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in gradient):
            raise ValueError("gradient entries must be finite real numbers")
        if not isinstance(rows, list) or len(rows) != record["batch_size"] or not all(
                isinstance(i, int) and not isinstance(i, bool) and 0 <= i < record["dataset_rows"] for i in rows):
            raise ValueError("sampled row indices do not match the dataset and batch")
    before, after = record.get("parameter_sha256_before"), record.get("parameter_sha256_after")
    if not isinstance(before, str) or len(before) != 64 or any(c not in "0123456789abcdef" for c in before) or before != after:
        raise ValueError("recorded parameter fingerprints must match")
    return record


def summarize(record):
    validate_record(record)
    estimate = statistics_module().noise_estimate(record["gradients"], record["batch_size"])
    variance = estimate["single_sample_noise"]/record["batch_size"]
    return {
        "batch_count": record["batch_count"], "batch_size": record["batch_size"],
        "dataset_rows": record["dataset_rows"], "layer": record["layer"],
        "parameter_shape": record["parameter_shape"], "sampling": record["sampling"],
        "torch_recorded": record["torch"],
        "model_parameter_sha256": record["parameter_sha256_before"],
        "observed_batch_gradient_variance": variance,
        "observed_mean_gradient_power": estimate["signal"]+variance/record["batch_count"],
        "estimate": estimate,
        "scope": "fixed synthetic model; raw gradients before momentum or NS; no optimal-LR claim",
    }


def save_records(path, record):
    validate_record(record)
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--save-records", type=Path, help="capture and save; refuses existing files")
    mode.add_argument("--replay", type=Path, help="recompute saved statistics without PyTorch")
    parser.add_argument("--batches", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.replay is not None:
        if any(value is not None for value in (args.batches, args.batch, args.seed)):
            parser.error("replay uses the recorded batch settings and does not resample")
        record = json.loads(args.replay.read_text(encoding="utf-8"))
    else:
        if args.save_records is not None and args.save_records.exists():
            raise FileExistsError("gradient record already exists: " + str(args.save_records))
        record = capture(batches=64 if args.batches is None else args.batches,
                         batch=16 if args.batch is None else args.batch,
                         seed=724 if args.seed is None else args.seed)
        if args.save_records is not None:
            save_records(args.save_records, record)
    print(json.dumps(summarize(record), indent=2, allow_nan=False))
