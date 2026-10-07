"""Plot stored toy runs without rerunning training or reconstructing paper figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('online-results.json'))
    parser.add_argument('--output', type=Path, default=Path('online-training.png'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, layout='constrained')
    palette = ['#2563eb', '#ea580c', '#059669', '#9333ea']
    colors = {beta: palette[i % len(palette)]
              for i, beta in enumerate(sorted({run['beta'] for run in data['runs']}))}
    seen = set()
    labels = [('exact_reward_after_update', 'Exact reward · after update'),
              ('exact_reference_kl_after_update', 'Exact reference KL (nat) · after update'),
              ('zero_variance_group_fraction', 'Same-reward groups · before update')]
    for run in data['runs']:
        steps = [row['round'] * run['groups'] * run['group_size'] for row in run['history']]
        for axis, (key, label) in zip(axes, labels):
            axis.plot(steps, [row[key] for row in run['history']],
                      color=colors[run['beta']], alpha=.65, linewidth=1.5,
                      label=f"beta={run['beta']:g}" if run['beta'] not in seen else None)
            axis.set_ylabel(label, fontsize=9)
            axis.grid(alpha=.2)
        seen.add(run['beta'])
    axes[0].set_ylim(0, 1.02)
    axes[2].set_ylim(-.03, 1.03)
    axes[0].legend(loc='lower right')
    axes[-1].set_xlabel('Cumulative sampled responses')
    fig.suptitle('Two-token toy policy · runs grouped by beta\nIndividual runs, not confidence intervals', fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
