"""CPU PyTorch laboratory: atomic losses and trainable contextual-bandit demos.

No model downloads, language generation, Gym dependency or external reward API.
The one-step demos exercise optimization mechanics, not LLM benchmark quality.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

DTYPE = torch.float64


def finite_tensor(value, name):
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite tensor")
    return value


def clipped_policy_loss(logp, old_logp, advantages, epsilon=0.2):
    """Elementwise MINIMIZATION loss. Caller chooses token/sequence reduction."""
    for name, value in (("logp", logp), ("old_logp", old_logp), ("advantages", advantages)):
        finite_tensor(value, name)
    if not (logp.shape == old_logp.shape == advantages.shape):
        raise ValueError("logp, old_logp and advantages must have equal shapes")
    if not math.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0, 1)")
    ratio = (logp - old_logp.detach()).exp()
    finite_tensor(ratio, "ratio")
    advantage = advantages.detach()
    return -torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - epsilon, 1 + epsilon) * advantage,
    )


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.5):
    """Original sequence-summed DPO loss; no length normalization."""
    tensors = (chosen, rejected, ref_chosen, ref_rejected)
    for value in tensors:
        finite_tensor(value, "sequence log-probability")
    if any(value.shape != chosen.shape for value in tensors):
        raise ValueError("All four log-probability shapes must match")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    margin = (chosen - rejected) - (ref_chosen - ref_rejected).detach()
    return -F.logsigmoid(beta * margin)


def group_advantages(rewards, epsilon=1e-8):
    """[prompts, samples] -> detached population-standardized advantages."""
    finite_tensor(rewards, "rewards")
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Each prompt requires at least two rewards")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    rewards = rewards.detach()
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, correction=0, keepdim=True)
    return centered / (scale + epsilon)


def categorical_kl(logits, reference_logits):
    """Exact forward KL(current || reference) for a small discrete action set."""
    finite_tensor(logits, "logits")
    finite_tensor(reference_logits, "reference_logits")
    if logits.shape != reference_logits.shape or logits.ndim < 1:
        raise ValueError("Current and reference logits must have equal shapes")
    logp = logits.log_softmax(dim=-1)
    logq = reference_logits.detach().log_softmax(dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def response_token_logps(logits, input_ids, response_mask):
    """Causal shift -> masked token logps and target mask, both [B,T-1].

    EOS may be marked True. Prompt/padding must be False. All sequences must
    contain at least one scored response token. No model is called here.
    """
    finite_tensor(logits, "logits")
    if logits.ndim != 3 or input_ids.shape != logits.shape[:2]:
        raise ValueError("Expected logits [B,T,V] and input_ids [B,T]")
    if input_ids.dtype != torch.long or response_mask.dtype != torch.bool:
        raise ValueError("Expected int64 token IDs and a boolean response mask")
    if response_mask.shape != input_ids.shape or input_ids.shape[1] < 2:
        raise ValueError("Invalid response mask or sequence length")
    if (input_ids < 0).any() or (input_ids >= logits.shape[-1]).any():
        raise ValueError("Token IDs are outside the vocabulary")
    if response_mask[:, 0].any():
        raise ValueError("The first token has no previous logit to score it")
    valid = response_mask[:, 1:]
    if not valid.any(dim=1).all():
        raise ValueError("Every response must contain a scored token")
    logp = logits[:, :-1].log_softmax(dim=-1)
    selected = logp.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    return selected.masked_fill(~valid, 0), valid


def response_logps(logits, input_ids, response_mask):
    """Response-only sequence SUM, using the same validated token alignment."""
    token_logps, _ = response_token_logps(logits, input_ids, response_mask)
    return token_logps.sum(dim=-1)


def gae(rewards, values, next_values, terminated, boundary, gamma=0.99, lam=0.95):
    """One temporal stream [T], with separate bootstrap and trace boundaries.

    terminated: true MDP terminal, suppress bootstrap.
    boundary: episode reset OR rollout cut, stop the recursive GAE trace.
    For a time limit use terminated=False, boundary=True and value(final_obs).
    """
    for name, value in (("rewards", rewards), ("values", values), ("next_values", next_values)):
        finite_tensor(value, name)
    if rewards.ndim != 1 or not rewards.numel():
        raise ValueError("Expected a nonempty temporal stream")
    if any(v.shape != rewards.shape for v in (values, next_values, terminated, boundary)):
        raise ValueError("All trajectory tensors must have equal shapes")
    if terminated.dtype != torch.bool or boundary.dtype != torch.bool:
        raise ValueError("Boundary masks must be boolean")
    if (terminated & ~boundary).any():
        raise ValueError("A terminal transition must also be a trace boundary")
    if not (math.isfinite(gamma) and math.isfinite(lam) and 0 <= gamma <= 1 and 0 <= lam <= 1):
        raise ValueError("gamma and lambda must be in [0, 1]")
    with torch.no_grad():
        # Convert masks first: float * bool otherwise uses the default float32,
        # silently rounding gamma/lambda even when the trajectory is float64.
        bootstrap = (~terminated).to(dtype=rewards.dtype)
        trace = (~boundary).to(dtype=rewards.dtype)
        delta = rewards + gamma * bootstrap * next_values - values
        advantages = torch.zeros_like(rewards)
        carry = torch.zeros_like(rewards[0])
        for t in reversed(range(len(rewards))):
            carry = delta[t] + gamma * lam * trace[t] * carry
            advantages[t] = carry
        returns = advantages + values
    return advantages, returns


def reward_table():
    # Four contexts, three candidate actions; one best action per context.
    return torch.tensor([
        [1.0, 0.2, -0.5],
        [-0.5, 1.0, 0.2],
        [0.2, -0.5, 1.0],
        [1.0, -0.5, 0.2],
    ], dtype=DTYPE)


@torch.no_grad()
def evaluate(logits, reference, rewards):
    logp = logits.log_softmax(-1)
    probabilities = logp.exp()
    best = rewards.argmax(-1, keepdim=True)
    return {
        "expected_reward": (probabilities * rewards).sum(-1).mean().item(),
        "best_action_probability": probabilities.gather(-1, best).mean().item(),
        "entropy": -(probabilities * logp).sum(-1).mean().item(),
        "kl_reference": categorical_kl(logits, reference).mean().item(),
    }


def train(algorithm, *, steps=120, seed=7, group_size=8, preference_flips=0, grpo_prompts=16):
    if algorithm not in ("ppo", "dpo", "grpo"):
        raise ValueError("Unknown algorithm")
    if type(steps) is not int or steps < 1 or type(group_size) is not int or group_size < 2:
        raise ValueError("steps >= 1 and group_size >= 2 are required")
    if type(preference_flips) is not int or not 0 <= preference_flips <= 12:
        raise ValueError("preference_flips must be an integer in [0, 12]")
    if preference_flips and algorithm != "dpo":
        raise ValueError("Preference flips apply only to DPO")
    if type(grpo_prompts) is not int or grpo_prompts < 1:
        raise ValueError("grpo_prompts must be a positive integer")
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rewards = reward_table()
    contexts, actions_count = rewards.shape
    actor = torch.nn.Parameter(torch.zeros_like(rewards))
    critic = torch.nn.Parameter(torch.zeros(contexts, dtype=DTYPE))
    reference = torch.zeros_like(rewards)  # fixed uniform reference, never updated
    optimizer = torch.optim.Adam(
        [actor, critic] if algorithm == "ppo" else [actor], lr=0.08,
    )
    epsilon, kl_beta, dpo_beta = 0.2, 0.03, 0.5
    target_kl, update_epochs = 0.05, 4
    rows = []
    updates = 0
    sampled_actions = 0
    pair_ids = [
        (x, winner, loser)
        for x in range(contexts)
        for winner in range(actions_count)
        for loser in range(actions_count)
        if rewards[x, winner] > rewards[x, loser]
    ]
    pairs = torch.tensor(pair_ids, dtype=torch.long)
    clean_pairs = pairs.clone()
    flipped_indices = []
    if preference_flips:
        # Choose once without replacement; reuse the same noisy labels each step.
        flipped_indices = torch.randperm(len(pairs), generator=generator)[:preference_flips].sort().values.tolist()
        pairs[flipped_indices] = pairs[flipped_indices][:, [0, 2, 1]]
    px, winner, loser = pairs.unbind(dim=1)
    ref_logps = reference.log_softmax(-1)

    def record(step, **stats):
        row = dict(
            algorithm=algorithm, step=step, optimizer_steps=updates,
            sampled_actions=sampled_actions, pair_presentations=0,
            **evaluate(actor, reference, rewards),
            loss=0.0, value_loss=0.0, clip_fraction=0.0,
            old_policy_kl=0.0, zero_group_fraction=0.0,
        )
        row.update(stats)
        if algorithm == "dpo":
            with torch.no_grad():
                current = actor.log_softmax(-1)
                for field, dataset in (("dpo_training_loss", pairs), ("dpo_clean_loss", clean_pairs)):
                    dx, dw, dl = dataset.unbind(dim=1)
                    row[field] = dpo_loss(current[dx, dw], current[dx, dl],
                                          ref_logps[dx, dw], ref_logps[dx, dl], beta=dpo_beta).mean().item()
        rows.append(row)

    record(0)
    for step in range(1, steps + 1):
        if algorithm == "dpo":
            logp = actor.log_softmax(-1)
            loss = dpo_loss(
                logp[px, winner], logp[px, loser],
                ref_logps[px, winner], ref_logps[px, loser], beta=dpo_beta,
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([actor], 1.0)
            optimizer.step()
            updates += 1
            record(step, loss=loss.item(), pair_presentations=step * len(pairs))
            continue

        with torch.no_grad():
            old_logits = actor.detach().clone()
            old_all = old_logits.log_softmax(-1)
            if algorithm == "ppo":
                x = torch.randint(contexts, (128,), generator=generator)
                a = torch.multinomial(old_all[x].exp(), 1, generator=generator).squeeze(-1)
                old_logp = old_all[x, a]
                score = rewards[x, a]
                # One terminal step: GAE reduces to shaped_reward - V_old(x).
                returns = score - kl_beta * (old_logp - ref_logps[x, a])
                advantage = returns - critic[x].detach()
                zero_fraction = 0.0
            else:
                x = torch.randint(contexts, (grpo_prompts,), generator=generator)
                a = torch.multinomial(old_all[x].exp(), group_size, replacement=True,
                                      generator=generator)
                old_logp = old_all[x].gather(-1, a)
                score = rewards[x[:, None], a]
                advantage = group_advantages(score)
                zero_fraction = (score.std(-1, correction=0) == 0).double().mean().item()
            sampled_actions += a.numel()

        value_loss = torch.zeros((), dtype=DTYPE)
        for _ in range(update_epochs):
            # Exact old||current KL over the tiny action space, before an update.
            with torch.no_grad():
                drift = categorical_kl(old_logits, actor.detach()).mean()
            if drift.item() > target_kl:
                break
            current = actor.log_softmax(-1)
            if algorithm == "ppo":
                logp = current[x, a]
                value_loss = F.mse_loss(critic[x], returns)
                loss = clipped_policy_loss(logp, old_logp, advantage, epsilon).mean()
                loss = loss + 0.5 * value_loss
            else:
                logp = current[x].gather(-1, a)
                loss = clipped_policy_loss(logp, old_logp, advantage, epsilon).mean()
                # Exact current||reference KL replaces the sampled token estimator.
                loss = loss + kl_beta * categorical_kl(actor[x], reference[x]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [actor, critic] if algorithm == "ppo" else [actor]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            updates += 1

        with torch.no_grad():
            new_all = actor.log_softmax(-1)
            new = new_all[x, a] if algorithm == "ppo" else new_all[x].gather(-1, a)
            ratio = (new - old_logp).exp()
            clip_fraction = ((ratio - 1).abs() > epsilon).double().mean().item()
            drift = categorical_kl(old_logits, actor.detach()).mean().item()
        record(step, loss=loss.item(), value_loss=value_loss.item(),
               clip_fraction=clip_fraction, old_policy_kl=drift,
               zero_group_fraction=zero_fraction)

    settings = {
        "algorithm": algorithm, "seed": seed, "steps": steps, "dtype": str(DTYPE),
        "device": "cpu", "torch": torch.__version__, "learning_rate": 0.08,
        "update_epochs": 1 if algorithm == "dpo" else update_epochs,
        "clip_epsilon": epsilon, "target_old_policy_kl": target_kl,
        "kl_beta": 0 if algorithm == "dpo" else kl_beta, "dpo_beta": dpo_beta,
        "group_size": group_size if algorithm == "grpo" else None,
        "online_batch_prompts": 128 if algorithm == "ppo" else grpo_prompts if algorithm == "grpo" else 0,
        "offline_preference_pairs": len(pairs) if algorithm == "dpo" else 0,
        "reward_table": rewards.tolist(), "reference": "fixed uniform",
        "scope": "one-step tabular contextual bandit; no language model or held-out generalization",
    }
    if algorithm == "dpo":
        settings.update(preference_flips=preference_flips,
                        flipped_pair_indices=flipped_indices,
                        clean_preference_pairs=clean_pairs.tolist(),
                        training_preference_pairs=pairs.tolist(),
                        preference_columns=["context", "chosen", "rejected"])
    return rows, settings


def atomic_demo():
    rewards = torch.tensor([0., 0., 1.], dtype=DTYPE)
    zeros = torch.zeros(3, dtype=DTYPE)
    terminal = torch.tensor([False, False, True])
    advantages, _ = gae(rewards, zeros, zeros, terminal, terminal, gamma=1, lam=1)
    ratios = torch.tensor([1.5, .5, 1.5, .5], dtype=DTYPE)
    signs = torch.tensor([1., -1., -1., 1.], dtype=DTYPE)
    values = -clipped_policy_loss(ratios.log(), torch.zeros_like(ratios), signs)
    return {
        "gae_terminal_reward": advantages.tolist(),
        "clipped_maximization_objective": values.tolist(),
        "group_0011": group_advantages(torch.tensor([[0., 0., 1., 1.]], dtype=DTYPE)).tolist(),
        "equal_policy_dpo_loss": dpo_loss(zeros, zeros, zeros, zeros).mean().item(),
        "identical_policy_kl": categorical_kl(zeros, zeros).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("atoms", "ppo", "dpo", "grpo", "all"), default="all")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--grpo-prompts", type=int, default=16,
                        help="Prompt groups per GRPO rollout; actions per batch = prompts * group-size")
    parser.add_argument("--preference-flips", type=int, default=0,
                        help="Swap labels in N of 12 fixed DPO pairs, once per run")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.steps < 1 or args.group_size < 2 or not 0 <= args.seed < 2**63:
        parser.error("Require steps >= 1, group-size >= 2, and seed in [0, 2**63)")
    if args.grpo_prompts < 1:
        parser.error("grpo-prompts must be positive")
    if not 0 <= args.preference_flips <= 12:
        parser.error("preference-flips must be in [0, 12]")
    if args.preference_flips and args.algorithm not in ("dpo", "all"):
        parser.error("preference-flips requires algorithm dpo or all")
    if args.algorithm == "atoms":
        print(json.dumps(atomic_demo(), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    algorithms = ("ppo", "dpo", "grpo") if args.algorithm == "all" else (args.algorithm,)
    for name in algorithms:
        rows, settings = train(name, steps=args.steps, seed=args.seed, group_size=args.group_size,
                               preference_flips=args.preference_flips if name == "dpo" else 0,
                               grpo_prompts=args.grpo_prompts)
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report = {"settings": settings, "initial": rows[0], "final": rows[-1]}
        (args.output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"algorithm": name, "initial_reward": rows[0]["expected_reward"],
                          "final_reward": rows[-1]["expected_reward"],
                          "best_action_probability": rows[-1]["best_action_probability"],
                          "report": str(args.output / f"{name}.json")}, indent=2))


if __name__ == "__main__":
    main()
