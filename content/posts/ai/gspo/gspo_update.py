"""One frozen, enumerated rollout batch and real SGD updates; not an RL benchmark."""
import argparse
import json
from pathlib import Path

import torch
from gspo_lab import group_advantages, objective


def run(steps=5, learning_rate=0.5):
    # Two independent binary decisions form a normalized autoregressive policy.
    # All four responses occur once: an exactly balanced batch under uniform old.
    responses = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    logits = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float64))

    def logps():
        return logits.log_softmax(-1).unsqueeze(0).expand(4, -1, -1).gather(
            -1, responses.unsqueeze(-1)).squeeze(-1)

    old = logps().detach().clone()
    rewards = responses.to(torch.float64).mean(-1)
    advantages = group_advantages(rewards[None])[0]
    mask = torch.ones_like(responses, dtype=torch.bool)
    optimizer = torch.optim.SGD([logits], lr=learning_rate)
    history = []
    for step in range(steps + 1):
        current = logps()
        value = objective(current, old, mask, advantages)
        ratio = ((current - old).mean(-1)).exp()
        flat = ((advantages > 0) & (ratio > 1.2)) | ((advantages < 0) & (ratio < .8))
        history.append(dict(step=step, objective=value.item(),
                            exact_expected_reward=logits.softmax(-1)[:, 1].mean().item(),
                            ratios=ratio.detach().tolist(), flat_fraction=flat.double().mean().item()))
        if step < steps:
            optimizer.zero_grad()
            (-value).backward()
            optimizer.step()
    return dict(scope='fixed enumerated batch; no stochastic rollout, KL, MoE or benchmark',
                learning_rate=learning_rate, epsilon=.2, responses=responses.tolist(),
                rewards=rewards.tolist(), advantages=advantages.tolist(), history=history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('update-results.json'))
    args = parser.parse_args()
    report = json.dumps(run(), indent=2) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding='utf-8')
    print(report, end='')
