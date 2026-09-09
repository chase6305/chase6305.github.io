"""Seeded CPU GSPO loop for two independent binary decisions; not a paper reproduction."""
import argparse
import json
import math
from pathlib import Path

import torch
from gspo_lab import group_advantages, objective


def exact_reference_kl(logits, reference_logps):
    """Exact forward sequence KL for this independent-position toy policy only."""
    current = logits.log_softmax(-1)
    return (current.exp() * (current - reference_logps.detach())).sum()


def run(seed=0, rounds=20, groups=8, group_size=8, updates=3, beta=1.0):
    if min(rounds, groups, updates) < 1 or group_size < 2:
        raise ValueError('Use positive rounds/groups/updates and group_size >= 2')
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    logits = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float64))
    reference = logits.detach().log_softmax(-1).clone()
    optimizer = torch.optim.SGD([logits], lr=.5)
    history = []
    for round_id in range(rounds):
        with torch.no_grad():
            responses = torch.multinomial(logits.softmax(-1), groups * group_size,
                                          replacement=True, generator=generator).T.contiguous()
            old = logits.log_softmax(-1).unsqueeze(0).expand(len(responses), -1, -1).gather(
                -1, responses.unsqueeze(-1)).squeeze(-1).clone()
            rewards = responses.double().mean(-1).reshape(groups, group_size)
            advantages = group_advantages(rewards).reshape(-1)
        old_copy = old.clone()
        mask = torch.ones_like(responses, dtype=torch.bool)
        initial_ratio_error = None
        for update in range(updates):
            current = logits.log_softmax(-1).unsqueeze(0).expand(len(responses), -1, -1).gather(
                -1, responses.unsqueeze(-1)).squeeze(-1)
            if update == 0:
                initial_ratio_error = (((current.detach() - old).mean(-1)).exp() - 1).abs().max().item()
            policy = objective(current, old, mask, advantages)
            loss = -policy + beta * exact_reference_kl(logits, reference)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            history.append(dict(round=round_id + 1,
                sampled_reward_before_update=rewards.mean().item(),
                exact_reward_after_update=logits.softmax(-1)[:, 1].mean().item(),
                exact_reference_kl_after_update=exact_reference_kl(logits, reference).item(),
                zero_variance_group_fraction=(rewards.std(-1, correction=0) == 0).double().mean().item(),
                initial_ratio_max_error=initial_ratio_error,
                old_logps_max_change=(old - old_copy).abs().max().item()))
    return dict(seed=seed, rounds=rounds, groups=groups, group_size=group_size,
                updates_per_round=updates, learning_rate=.5, epsilon=.2, beta=beta,
                sampled_responses=rounds*groups*group_size,
                sampled_tokens=2*rounds*groups*group_size, history=history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('online-results.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = dict(torch=torch.__version__, device='cpu', dtype='float64',
                  scope='two independent binary decisions; identical prompt; no LLM, MoE or benchmark',
                  runs=[run(seed=seed, beta=beta) for beta in (0., 1.) for seed in (0, 1, 2)])
    output = json.dumps(report, indent=2) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding='utf-8')
    print(json.dumps([dict(seed=r['seed'], beta=r['beta'], **r['history'][-1]) for r in report['runs']], indent=2))
