"""GSPO v1 objective/gradient laboratory; CPU PyTorch, no model downloads.

Run: python gspo_lab.py --output results.json
Uses fixed response log-probabilities to isolate objectives, not an RL benchmark.
"""
import argparse
import json
import math
from pathlib import Path

import torch


def group_advantages(rewards, eps=1e-8):
    """rewards: [prompts, group]; population std, with zero-variance groups -> zero."""
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not torch.isfinite(rewards).all():
        raise ValueError("Expected finite rewards with group size >= 2")
    r = rewards.detach()
    return (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True, correction=0) + eps)


def objective(current, old, mask, advantages, kind="gspo", epsilon=.2):
    """Maximized surrogate. Rows are equally weighted responses of equal-size groups.

    Logps: [responses, padded_tokens]; mask: bool; advantages: [responses].
    GSPO-token here uses the same scalar advantage on every response token.
    """
    if kind not in ("grpo", "gspo", "gspo-token"):
        raise ValueError("Unknown objective")
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must lie strictly between 0 and 1")
    if current.ndim != 2 or old.shape != current.shape or mask.shape != current.shape:
        raise ValueError("Logps and mask must have the same 2D shape")
    if mask.dtype != torch.bool or advantages.shape != (current.shape[0],):
        raise ValueError("Expected boolean mask and one advantage per response")
    lengths = mask.sum(-1)
    if current.shape[0] == 0 or torch.any(lengths == 0):
        raise ValueError("Every response must contain at least one valid token")
    if not (torch.isfinite(current[mask]).all() and torch.isfinite(old[mask]).all()
            and torch.isfinite(advantages).all()):
        raise ValueError("Valid-token logps and advantages must be finite")
    # Select before arithmetic: padding may contain NaN and must not enter the graph.
    c = torch.where(mask, current, torch.zeros_like(current))
    o = torch.where(mask, old.detach(), torch.zeros_like(old))
    delta = c - o
    sequence_ratio = torch.exp(delta.sum(-1) / lengths)
    if not torch.isfinite(sequence_ratio).all():
        raise ValueError("Sequence ratio overflow: inspect policy drift and precision")
    adv = advantages.detach()
    if kind == "gspo":
        terms = torch.minimum(sequence_ratio * adv,
                              sequence_ratio.clamp(1-epsilon, 1+epsilon) * adv)
        return terms.mean()
    if kind == "grpo":
        ratio = delta.exp()
    else:
        # Forward equals s_i; only current token logp supplies the local derivative.
        ratio = sequence_ratio.detach()[:, None] * (c - c.detach()).exp()
    if not torch.isfinite(ratio[mask]).all():
        raise ValueError("Token ratio overflow")
    terms = torch.minimum(ratio * adv[:, None],
                          ratio.clamp(1-epsilon, 1+epsilon) * adv[:, None])
    return ((terms * mask).sum(-1) / lengths).mean()


def inspect(ratios, advantage, kind):
    old = torch.full((1, len(ratios)), -3., dtype=torch.float64)
    current = (old + torch.tensor(ratios, dtype=torch.float64).log()).requires_grad_()
    value = objective(current, old, torch.ones_like(old, dtype=torch.bool),
                      torch.tensor([advantage], dtype=torch.float64), kind)
    gradient, = torch.autograd.grad(value, current)
    return dict(value=value.item(), gradient=gradient[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    report = dict(torch=torch.__version__, epsilon=.2,
                  scope="fixed log-probability objective and first-gradient comparison; no RL training",
                  cancellation={}, clipping={})
    for advantage in (1., -1.):
        report["cancellation"][str(advantage)] = {
            kind: inspect([.5, 2.], advantage, kind) for kind in ("grpo", "gspo", "gspo-token")}
    for ratio, advantage in ((1.3, 1.), (.7, 1.), (1.3, -1.), (.7, -1.)):
        report["clipping"][f"s={ratio}, A={advantage}"] = inspect([ratio, ratio], advantage, "gspo")
    report["length_normalization"] = [dict(length=n, raw_ratio=1.01**n, gspo_ratio=1.01)
                                       for n in (2, 100, 1000)]
    # Exact IS uses the full likelihood ratio. Taking a root loses that identity.
    old_prob = torch.tensor([.5, .5], dtype=torch.float64)
    new_prob = torch.tensor([.8, .2], dtype=torch.float64)
    reward = torch.tensor([1., 0.], dtype=torch.float64)
    report["is_identity"] = dict(target=float(new_prob @ reward),
        exact=float(old_prob @ ((new_prob / old_prob) * reward)),
        root_weight=float(old_prob @ ((new_prob / old_prob).sqrt() * reward)))
    report["advantages"] = group_advantages(torch.tensor([[0., 1.], [.5, .5]], dtype=torch.float64)).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
