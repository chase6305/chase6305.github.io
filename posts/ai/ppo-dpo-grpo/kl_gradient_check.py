"""A KL value identity does not specify the gradient of a sampled surrogate.

Enumerates three actions exactly: no Monte Carlo error, no training benchmark.
Requires PyTorch; Matplotlib only for --figure.
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('kl-gradient-results.json'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    dtype = torch.float64
    initial = torch.tensor([.79, .11, .10], dtype=dtype)
    reference = torch.tensor([.50, .10, .40], dtype=dtype)
    old = torch.tensor([.20, .30, .50], dtype=dtype)
    logits = initial.log().requires_grad_()
    logp = torch.log_softmax(logits, dim=-1)
    current = logp.exp()
    ell = reference.log()-logp
    k3 = ell.expm1()-ell
    direct = (current*(logp-reference.log())).sum()
    frozen = (current.detach()*k3).sum()
    differentiable = (current*k3).sum()
    score_part = (current*k3.detach()).sum()
    importance = (old*(current/old)*k3).sum()
    old_uncorrected = (old*k3).sum()
    losses = {'direct_forward_kl': direct, 'frozen_sampling_weights': frozen,
              'differentiate_entire_expectation': differentiable,
              'score_function_part_only': score_part,
              'differentiable_importance_weights': importance,
              'old_sampling_without_correction': old_uncorrected}
    gradients = {name: torch.autograd.grad(value, logits, retain_graph=True)[0]
                 for name, value in losses.items()}
    expected = current.detach()*((logp-reference.log()).detach()-direct.detach())
    torch.testing.assert_close(gradients['direct_forward_kl'], expected, atol=1e-14, rtol=0)
    torch.testing.assert_close(gradients['frozen_sampling_weights'], current.detach()-reference, atol=1e-14, rtol=0)
    for name in ('differentiate_entire_expectation', 'differentiable_importance_weights'):
        torch.testing.assert_close(gradients[name], gradients['direct_forward_kl'], atol=1e-14, rtol=0)
        torch.testing.assert_close(losses[name], direct, atol=1e-14, rtol=0)
    torch.testing.assert_close(frozen, direct, atol=1e-14, rtol=0)
    torch.testing.assert_close(gradients['frozen_sampling_weights']+gradients['score_function_part_only'],
                               gradients['direct_forward_kl'], atol=1e-14, rtol=0)
    assert gradients['frozen_sampling_weights'][1] > 0
    assert gradients['direct_forward_kl'][1] < 0
    assert abs(old_uncorrected.item()-direct.item()) > .1
    # Finite differences of the exact forward KL as a separate derivative check.
    def exact_value(z):
        lp = z.log_softmax(-1)
        return (lp.exp()*(lp-reference.log())).sum()
    numeric = torch.stack([(exact_value(logits.detach()+h)-exact_value(logits.detach()-h))/2e-6
                           for h in torch.eye(3, dtype=dtype)*1e-6])
    torch.testing.assert_close(numeric, gradients['direct_forward_kl'], atol=1e-9, rtol=0)
    results = {name: {'value': value.item(), 'logit_gradient': gradients[name].tolist()}
               for name, value in losses.items()}
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, ax = plt.subplots(figsize=(9.4, 4.2), constrained_layout=True)
        x = torch.arange(3).numpy()
        ax.bar(x-.18, gradients['direct_forward_kl'].numpy(), width=.36,
               color='#528bb5', label='Exact forward-KL gradient')
        ax.bar(x+.18, gradients['frozen_sampling_weights'].numpy(), width=.36,
               color='#ae89c5', label='Frozen sampling weights: k3 derivative')
        ax.axhline(0, color='#708090', linewidth=.8)
        ax.set(xticks=x, xticklabels=['Logit 1', 'Logit 2', 'Logit 3'], ylabel='Derivative w.r.t. logit',
               title=f'Same value ({direct.item():.6f} nats), different gradients')
        ax.legend(); ax.grid(axis='y', alpha=.15)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=160); plt.close(fig)
    report = {'torch': torch.__version__, 'dtype': 'float64', 'current': initial.tolist(),
              'reference': reference.tolist(), 'old_for_importance_check': old.tolist(),
              'results': results,
              'finite_difference_max_error': (numeric-gradients['direct_forward_kl']).abs().max().item(),
              'scope': 'Exact categorical sum at one fixed context; no state/prefix distribution derivative, clipping, finite-sample variance or full GRPO implementation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
