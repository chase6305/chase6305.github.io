"""Two-step PPO + GAE experiment on CPU; no Gym or model download.

State 0: action 0 -> state 1, action 1 -> state 2, both with reward 0.
State 1: terminal rewards [1, 0]. State 2: terminal rewards [0, 0.2].
Every episode has exactly two transitions; gamma=1 and no reference KL.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from rl_lab import DTYPE, categorical_kl, clipped_policy_loss, gae


@torch.no_grad()
def rollout(actor, critic, generator, episodes=64):
    """Flatten complete episodes in episode-major order: e0t0, e0t1, e1t0..."""
    logps = actor.log_softmax(-1)
    start = torch.zeros(episodes, dtype=torch.long)
    first = torch.multinomial(logps[start].exp(), 1, generator=generator).squeeze(1)
    second_state = first + 1
    second = torch.multinomial(logps[second_state].exp(), 1, generator=generator).squeeze(1)
    terminal_rewards = torch.tensor([[1., 0.], [0., .2]], dtype=DTYPE)
    final_reward = terminal_rewards[first, second]
    states = torch.stack((start, second_state), dim=1).flatten()
    actions = torch.stack((first, second), dim=1).flatten()
    rewards = torch.stack((torch.zeros_like(final_reward), final_reward), dim=1).flatten()
    next_values = torch.stack((critic[second_state], torch.zeros_like(final_reward)), dim=1).flatten()
    terminated = torch.tensor([False, True]).repeat(episodes)
    return dict(states=states, actions=actions, rewards=rewards,
                values=critic[states].clone(), next_values=next_values,
                old_logp=logps[states, actions], terminated=terminated,
                boundary=terminated.clone())


@torch.no_grad()
def evaluate(actor, critic):
    p = actor.softmax(-1)
    # Enumerate all four complete paths; no sampled evaluation noise.
    expected_return = p[0, 0] * p[1, 0] + .2 * p[0, 1] * p[2, 1]
    return dict(expected_return=expected_return.item(),
                left_probability=p[0, 0].item(),
                left_success_probability=p[1, 0].item(),
                value_start=critic[0].item())


def train(*, steps=120, seed=7, gae_lambda=.95):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    if not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError("gae_lambda must be in [0, 1]")
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    actor = torch.nn.Parameter(torch.zeros(3, 2, dtype=DTYPE))
    critic = torch.nn.Parameter(torch.zeros(3, dtype=DTYPE))
    optimizer = torch.optim.Adam([actor, critic], lr=.08)
    episodes, update_epochs, epsilon, target_kl = 64, 4, .2, .05
    rows = [dict(step=0, optimizer_steps=0, sampled_episodes=0,
                 sampled_transitions=0, old_policy_kl=0., **evaluate(actor, critic))]
    updates = 0
    first_trace = None
    for step in range(1, steps + 1):
        old = actor.detach().clone()
        batch = rollout(old, critic.detach(), generator, episodes)
        advantage, returns = gae(
            batch["rewards"], batch["values"], batch["next_values"],
            batch["terminated"], batch["boundary"], gamma=1., lam=gae_lambda,
        )
        if first_trace is None:
            first_trace = {key: value[:8].tolist() for key, value in batch.items()}
            first_trace.update(advantages=advantage[:8].tolist(), returns=returns[:8].tolist())
        states, actions = batch["states"], batch["actions"]
        for _ in range(update_epochs):
            # Old-state visitation weights, exact sum over both possible actions.
            with torch.no_grad():
                drift = categorical_kl(old[states], actor.detach()[states]).mean()
            if drift.item() > target_kl:
                break
            logp = actor.log_softmax(-1)[states, actions]
            policy_loss = clipped_policy_loss(logp, batch["old_logp"], advantage, epsilon).mean()
            value_loss = (critic[states] - returns).square().mean()
            optimizer.zero_grad(set_to_none=True)
            (policy_loss + .5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm_([actor, critic], 1.)
            optimizer.step()
            updates += 1
        with torch.no_grad():
            drift = categorical_kl(old[states], actor.detach()[states]).mean().item()
        rows.append(dict(step=step, optimizer_steps=updates,
                         sampled_episodes=step * episodes,
                         sampled_transitions=step * episodes * 2,
                         old_policy_kl=drift, **evaluate(actor, critic)))
    settings = dict(seed=seed, steps=steps, gae_lambda=gae_lambda, gamma=1.,
                    episodes_per_batch=episodes, transitions_per_episode=2,
                    learning_rate=.08, update_epochs=update_epochs,
                    clip_epsilon=epsilon, target_old_policy_kl=target_kl,
                    reference_kl=False, advantage_normalization=False,
                    value_target="GAE lambda return (advantage + old value)",
                    kl_reduction="old||current; rollout-state mean, exact action sum",
                    device="cpu", dtype=str(DTYPE), torch=torch.__version__,
                    scope="two-step tabular decision tree, complete terminal episodes")
    return rows, dict(settings=settings, initial=rows[0], final=rows[-1],
                     first_rollout_trace=first_trace,
                     final_action_probabilities=actor.detach().softmax(-1).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--output", type=Path, default=Path("results-chain"))
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.seed < 2**63 or not 0 <= args.gae_lambda <= 1:
        parser.error("Require steps >= 1, seed in [0, 2**63), and gae-lambda in [0, 1]")
    rows, report = train(steps=args.steps, seed=args.seed, gae_lambda=args.gae_lambda)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "ppo-chain.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "ppo-chain.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"initial": report["initial"], "final": report["final"],
                      "report": str(args.output / "ppo-chain.json")}, indent=2))


if __name__ == "__main__":
    main()
