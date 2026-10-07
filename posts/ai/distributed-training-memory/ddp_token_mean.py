"""Verify token-normalized DDP accumulation against a single-process reference.

Run as a file: python -B ddp_token_mean.py --output ddp-token-results.json
Requires PyTorch with CPU Gloo. Two local spawned processes, float64, no GPU.
No torchrun is needed: this program creates its own two-process group.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import cross_entropy


IGNORE = -100


def batches(rank, mode):
    labels = [[[0, 1, IGNORE], [1, IGNORE, IGNORE], [2, 3, IGNORE]],
              [[2, 3, IGNORE], [IGNORE, IGNORE, IGNORE], [1, IGNORE, IGNORE]]][rank]
    if (mode == "empty_rank" and rank == 1) or mode == "empty_window":
        labels = [[IGNORE] * 3 for _ in labels]
    generator = torch.Generator().manual_seed(100 + rank)
    return [(torch.randn(3, 3, generator=generator, dtype=torch.float64),
             torch.tensor(y, dtype=torch.long)) for y in labels]


def make_model():
    model = torch.nn.Linear(3, 4, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.arange(12, dtype=torch.float64).reshape(4, 3) / 20)
        model.bias.copy_(torch.arange(4, dtype=torch.float64) / 10)
    return model


def flat_grad(model):
    return torch.cat([parameter.grad.flatten() for parameter in model.parameters()])


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    # This fixture is local-only; use the loopback interface on Linux.
    if os.name == "posix" and Path("/sys/class/net/lo").exists():
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        results = {}
        for mode in ("uneven_tokens", "empty_rank", "empty_window"):
            model = DDP(make_model())
            reference = make_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
            local_batches = batches(rank, mode)
            records = []
            for start in (0, 2):  # windows of two and one micro-batches
                window = local_batches[start:start + 2]
                local_valid = sum(int((labels != IGNORE).sum()) for _, labels in window)
                total = torch.tensor(local_valid, dtype=torch.int64)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                if total.item() == 0:
                    # The global count gives every rank the same branch.
                    records.append({"start_micro_batch": start, "valid_tokens": 0,
                                    "action": "skip optimizer update on all ranks"})
                    continue
                for index, (inputs, labels) in enumerate(window):
                    context = nullcontext() if index == len(window) - 1 else model.no_sync()
                    with context:
                        loss_sum = cross_entropy(model(inputs), labels,
                                                 ignore_index=IGNORE, reduction="sum")
                        (loss_sum * dist.get_world_size() / total.item()).backward()

                # Independent reference: concatenate all valid-token losses in
                # the whole window, then divide once. No DDP or accumulation.
                all_batches = [batch for r in range(2) for batch in batches(r, mode)[start:start + 2]]
                all_inputs = torch.cat([x for x, _ in all_batches])
                all_labels = torch.cat([y for _, y in all_batches])
                reference_loss = cross_entropy(reference(all_inputs), all_labels,
                                               ignore_index=IGNORE, reduction="sum") / total.item()
                reference_loss.backward()
                error = float((flat_grad(model) - flat_grad(reference)).abs().max())
                torch.testing.assert_close(flat_grad(model), flat_grad(reference), rtol=1e-12, atol=1e-12)
                naive_error = None
                if mode == "uneven_tokens":
                    naive = make_model()
                    naive.load_state_dict(reference.state_dict())
                    for other_rank in range(2):
                        other = batches(other_rank, mode)[start:start + 2]
                        inputs = torch.cat([x for x, _ in other])
                        labels = torch.cat([y for _, y in other])
                        # Incorrect objective: give each rank's local mean equal
                        # weight despite different valid-token counts.
                        (cross_entropy(naive(inputs), labels, ignore_index=IGNORE,
                                       reduction="mean") / 2).backward()
                    naive_error = float((flat_grad(naive) - flat_grad(reference)).abs().max())
                    assert naive_error > 1e-4
                optimizer.step()
                reference_optimizer.step()
                for parameter, expected in zip(model.module.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter, expected, rtol=1e-12, atol=1e-12)
                records.append({"start_micro_batch": start, "micro_batches": len(window),
                                "valid_tokens": total.item(), "gradient_max_abs_error": error,
                                "incorrect_rank_mean_gradient_error": naive_error})
            if mode == "empty_window":
                for parameter, initial in zip(model.module.parameters(), make_model().parameters()):
                    torch.testing.assert_close(parameter, initial, rtol=0, atol=0)
            results[mode] = records
        if rank == 0:
            Path(output).write_text(json.dumps({"torch": torch.__version__, "backend": "gloo",
                "world_size": 2, "dtype": "float64", "seed_per_rank": [100, 101],
                "results": results}, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ddp-token-results.json"))
    args = parser.parse_args()
    if not dist.is_gloo_available():
        raise RuntimeError("This PyTorch installation has no Gloo backend")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ddp-token-check-") as directory:
        rendezvous = str(Path(directory) / "rendezvous")  # initially absent
        mp.spawn(worker, args=(rendezvous, str(output)), nprocs=2, join=True)
    print(output.read_text())


if __name__ == "__main__":
    main()
