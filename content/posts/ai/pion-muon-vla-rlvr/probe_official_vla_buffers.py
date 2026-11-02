"""Inspect missing-gradient buffer reuse in pinned VLA DefaultPion.

CPU replays redirect the official CUDA buffer allocation and vary collective
completion order. --gloo adds real one-rank CPU collectives with a flat output
view. --nccl adds unwrapped one-rank CUDA/NCCL calls using the original allocator.
Only tiny matrices are used. No model is loaded and source is unedited.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

SOURCE_SHA256 = "8e7ca8d6f40ebc4fb066acb15387e618d515896cca5d1eb0ad234e5310e9157e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-file", type=Path, required=True)
    parser.add_argument("--gloo", action="store_true", help="Also run real one-rank CPU/Gloo collectives")
    parser.add_argument("--nccl", action="store_true", help="Also run 40 real one-rank CUDA/NCCL cases; requires a CUDA GPU")
    args = parser.parse_args()
    try:
        digest = hashlib.sha256(args.official_file.read_bytes()).hexdigest()
    except OSError as error:
        parser.error(str(error))
    if digest != SOURCE_SHA256:
        parser.error("Official VLA source does not match pinned Pion commit 8796501")

    import torch
    import torch.distributed as dist

    spec = importlib.util.spec_from_file_location("pinned_vla_buffer_probe", args.official_file)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    torch.set_num_threads(1)
    original_empty = torch.empty
    original_gather = dist.all_gather_into_tensor

    def cpu_buffer(*positional, **keywords):
        if keywords.get("device") == "cuda":
            keywords = {**keywords, "device": "cpu"}
        return original_empty(*positional, **keywords)

    class Completion:
        def __init__(self, output, snapshot, immediate):
            self.output, self.snapshot = output, snapshot
            self.done = False
            if immediate:
                self.wait()

        def wait(self):
            if not self.done:
                self.output.view(-1).copy_(self.snapshot.view(-1))
                self.done = True

    def replay(immediate):
        def gather(output, value, *, async_op):
            assert async_op
            # Independent snapshot represents launch input. The replay varies
            # when its result reaches the shared receive storage.
            return Completion(output, value.detach().clone(), immediate)
        return gather

    def gloo(output, value, *, async_op):
        # Gloo requires the concatenated flat shape. This view retains exactly
        # the receive buffer's storage, including the source/receive alias.
        return original_gather(output.view(-1), value, async_op=async_op)

    def run(mode, gather, pattern, same_group, device="cpu", repeat=0):
        shapes = [(2, 2), (2, 2) if same_group else (2, 3)]
        params = [torch.nn.Parameter(torch.ones(shape, dtype=torch.float64, device=device)) for shape in shapes]
        expected = []
        for parameter, kind in zip(params, pattern):
            target = torch.full_like(parameter, .999)
            if kind == "none":
                parameter.grad = None
            else:
                parameter.grad = torch.zeros_like(parameter)
                if kind == "nonzero":
                    parameter.grad[0, 0] = 1.
                    target[0, 0] -= .01 * .2 * math.sqrt(max(parameter.shape))
            expected.append(target)
        with patch.object(torch, "empty", cpu_buffer) if device == "cpu" else nullcontext():
            optimizer = official.DefaultPion(params, lr=.01, weight_decay=.1,
                                            momentum=.5, nesterov=False,
                                            promotion_steps=1, ns_steps=5,
                                            scale_factor=.2, rank=0, world_size=1)
        with patch.object(dist, "all_gather_into_tensor", gather) if gather is not None else nullcontext():
            optimizer.step()
        if device != "cpu":
            torch.cuda.synchronize()
        errors = [float((parameter.detach()-target).abs().max()) for parameter, target in zip(params, expected)]
        result = {
            "completion": mode, "device": device, "repeat": repeat,
            "gradients": list(pattern), "same_numel_group": same_group,
            "expected_weights": [target.tolist() for target in expected],
            "actual_weights": [parameter.detach().tolist() for parameter in params],
            "max_weight_differences": errors, "matches_independent_updates": max(errors) < 4e-15,
        }
        # Controls must agree; the one deliberately hazardous adjacent pattern
        # must fail under both replay orders. Real backend timing is observed.
        hazardous = same_group and pattern == ("nonzero", "none")
        if not hazardous:
            assert result["matches_independent_updates"], result
        elif mode.endswith("_replay"):
            assert not result["matches_independent_updates"], result
        return result

    patterns = [("nonzero", "zero"), ("nonzero", "none"), ("none", "nonzero"), ("none", "none")]
    cases = []
    for mode, gather in [("immediate_replay", replay(True)), ("wait_completion_replay", replay(False))]:
        for same_group in (True, False):
            for pattern in patterns:
                cases.append(run(mode, gather, pattern, same_group))
    if args.gloo:
        if not dist.is_gloo_available():
            parser.error("This PyTorch build has no Gloo backend")
        with tempfile.TemporaryDirectory(prefix="pion-vla-gloo-") as folder:
            rendezvous = (Path(folder)/"rendezvous").as_uri()
            dist.init_process_group("gloo", init_method=rendezvous, rank=0, world_size=1,
                                    timeout=timedelta(seconds=30))
            try:
                for same_group in (True, False):
                    for pattern in patterns:
                        cases.append(run("gloo", gloo, pattern, same_group))
            finally:
                dist.destroy_process_group()
    gpu = None
    if args.nccl:
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            parser.error("--nccl requires an available CUDA GPU and NCCL backend")
        torch.cuda.set_device(0)
        gpu = {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
               "nccl": list(torch.cuda.nccl.version()), "repeats": 5}
        with tempfile.TemporaryDirectory(prefix="pion-vla-nccl-") as folder:
            rendezvous = (Path(folder)/"rendezvous").as_uri()
            dist.init_process_group("nccl", init_method=rendezvous, rank=0, world_size=1,
                                    timeout=timedelta(seconds=30), device_id=torch.device("cuda:0"))
            try:
                for repeat in range(5):
                    for same_group in (True, False):
                        for pattern in patterns:
                            cases.append(run("nccl", None, pattern, same_group, "cuda:0", repeat))
            finally:
                dist.destroy_process_group()
    report = {
        "scope": "CPU allocation adaptation and completion replays; optional one-rank Gloo and unwrapped CUDA/NCCL. Tiny parameters only; no model or multi-rank training.",
        "official_commit": "87965014b360bde604632cba0aa12e423c6c63c6",
        "official_source_sha256": digest,
        "torch": torch.__version__, "devices": ["cpu", "cuda:0"] if args.nccl else ["cpu"],
        "gpu": gpu, "parameter_dtype": "float64",
        "internal_dtype": "bfloat16", "source_modified": False,
        "adaptations": {"cpu_replay": "Redirect CUDA receive-buffer allocation to CPU and replay explicit completion order",
                        "gloo": "Redirect buffer to CPU and flatten output through a storage-sharing view",
                        "nccl": "No source, allocator or collective wrapper modifications"},
        "settings": {"lr": .01, "weight_decay": .1, "momentum": .5, "nesterov": False, "promotion_steps": 1, "ns_steps": 5, "scale_factor": .2},
        "cases": cases,
        "summary": {"cases": len(cases), "controls": sum(not (row["same_numel_group"] and row["gradients"] == ["nonzero", "none"]) for row in cases), "independent_update_mismatches": sum(not row["matches_independent_updates"] for row in cases),
                    "by_completion": {mode: {"cases": sum(row["completion"] == mode for row in cases),
                                            "independent_update_mismatches": sum(row["completion"] == mode and not row["matches_independent_updates"] for row in cases)}
                                      for mode in sorted({row["completion"] for row in cases})}},
        "interpretation": "A missing gradient aliases the receive buffer before the preceding collective is waited and consumed. Completion order can lose the preceding update or reuse it for the missing parameter. Real backend outcomes describe this device, software and these tiny matrices, not every training run.",
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
