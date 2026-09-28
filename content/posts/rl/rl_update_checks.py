"""CPU checks for discounted REINFORCE and TD3 gradient paths.

Run: python -B rl_update_checks.py
Requires PyTorch. This is a numerical check, not an environment trainer.
"""

import itertools
import json

import torch
from torch import nn
from torch.distributions import Bernoulli


def policy_gradient_check():
    """Enumerate all four trajectories, avoiding Monte Carlo uncertainty."""
    theta = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
    gamma = 0.9
    probability = theta.sigmoid()
    objective = probability[0] + 2 * gamma * probability[1]
    exact, = torch.autograd.grad(objective, theta)
    estimates = {}
    for name in ("return_to_go", "state_baseline", "missing_outer_discount"):
        surrogate = torch.zeros((), dtype=theta.dtype)
        for a0, a1 in itertools.product((0.0, 1.0), repeat=2):
            action = theta.new_tensor([a0, a1])
            log_probability = Bernoulli(logits=theta).log_prob(action)
            trajectory_probability = log_probability.sum().exp().detach()
            returns = theta.new_tensor([a0 + gamma * 2 * a1, 2 * a1])
            if name == "state_baseline":
                returns = returns - theta.new_tensor([0.3, 0.7])
            time_weight = theta.new_tensor([1.0, gamma])
            if name == "missing_outer_discount":
                time_weight = torch.ones_like(time_weight)
            surrogate = surrogate + trajectory_probability * (
                time_weight * returns * log_probability
            ).sum()
        gradient, = torch.autograd.grad(surrogate, theta)
        estimates[name] = gradient
    torch.testing.assert_close(estimates["return_to_go"], exact)
    torch.testing.assert_close(estimates["state_baseline"], exact)
    wrong = estimates["missing_outer_discount"]
    assert not torch.allclose(wrong, exact)
    torch.testing.assert_close(wrong[1] * gamma, exact[1])
    # Central differences independently check the closed-form objective.
    def value(x):
        p = x.sigmoid()
        return p[0] + 2 * gamma * p[1]
    step = 1e-5
    finite_difference = []
    for axis in torch.eye(2, dtype=theta.dtype):
        finite_difference.append(
            (value(theta.detach() + step * axis)
             - value(theta.detach() - step * axis)) / (2 * step)
        )
    torch.testing.assert_close(torch.stack(finite_difference), exact,
                               atol=1e-10, rtol=1e-8)
    return {"exact_gradient": exact.tolist(),
            "missing_outer_discount": wrong.tolist(),
            "baseline_and_finite_difference": "passed"}


def td3_gradient_check():
    torch.manual_seed(42)
    actor = nn.Sequential(nn.Linear(2, 1), nn.Tanh()).double()
    critics = nn.ModuleList([nn.Linear(3, 1).double() for _ in range(2)])
    target_actor = nn.Sequential(nn.Linear(2, 1), nn.Tanh()).double()
    target_critics = nn.ModuleList([nn.Linear(3, 1).double() for _ in range(2)])
    state = torch.randn(4, 2, dtype=torch.float64)
    next_state = torch.randn_like(state)
    stored_action = torch.tensor([[-0.8], [0.7], [-0.2], [0.4]],
                                 dtype=torch.float64)
    reward = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    terminated = torch.tensor([[False], [True], [False], [True]])
    with torch.no_grad():
        noise = (0.2 * torch.randn_like(stored_action)).clamp(-0.5, 0.5)
        next_action = (target_actor(next_state) + noise).clamp(-1.0, 1.0)
        next_input = torch.cat((next_state, next_action), dim=-1)
        target_q = torch.minimum(*(q(next_input) for q in target_critics))
        target = reward + 0.99 * (~terminated) * target_q
    assert not target.requires_grad
    torch.testing.assert_close(target[terminated], reward[terminated])
    recorded_input = torch.cat((state, stored_action), dim=-1)
    critic_loss = sum((q(recorded_input) - target).square().mean()
                      for q in critics)
    critic_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in critics.parameters())
    assert all(p.grad is None for p in actor.parameters())
    assert all(p.grad is None for model in (target_actor, target_critics)
               for p in model.parameters())
    critics.zero_grad(set_to_none=True)
    for parameter in critics.parameters():
        parameter.requires_grad_(False)
    actor_loss = -critics[0](torch.cat((state, actor(state)), dim=-1)).mean()
    actor_loss.backward()
    assert all(p.grad is None for p in critics.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in actor.parameters())
    actor_gradient_norm = sum(p.grad.square().sum()
                              for p in actor.parameters()).sqrt().item()
    assert actor_gradient_norm > 0.0
    return {"target_detached": True, "terminal_target_equals_reward": True,
            "critic_update_uses_replay_action": True,
            "actor_gradient_norm_with_frozen_critic": actor_gradient_norm}


if __name__ == "__main__":
    print(json.dumps({"torch": torch.__version__,
                      "reinforce": policy_gradient_check(),
                      "td3": td3_gradient_check()}, indent=2))
