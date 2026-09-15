#!/usr/bin/env python3
"""Plot X-VLA's post-warmup LR schedule, without loading a model.

Requires matplotlib: python -m pip install matplotlib
Run beside this file: python lr_schedule_plot.py
Formula reference: 2toinf/X-VLA, train.py, commit 6bc2513.
The plotted endpoints are mathematical boundaries, not extra optimizer updates.
"""
import math
from pathlib import Path

FREEZE = 1_000
WARMUP = 2_000
BASE_LR = 1e-4
MIN_RATIO = 0.1


def cosine_lr(step, total):
    """Post-warmup branch only; require a nonempty cosine interval."""
    start = FREEZE + WARMUP
    if step < start or total <= start:
        raise ValueError('require step >= freeze + warmup and total > freeze + warmup')
    progress = min(1.0, (step - start) / (total - start))
    return BASE_LR * (MIN_RATIO + (1 - MIN_RATIO) * (1 + math.cos(math.pi * progress)) / 2)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'xvla-lr-budget'})
    fig, ax = plt.subplots(figsize=(10, 5.7), layout='constrained')
    fig.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')
    for total, color, label in [(100_000, '#2563eb', '100k training budget'),
                                 (200_000, '#d97706', '200k training budget')]:
        steps = [FREEZE + WARMUP + (total - FREEZE - WARMUP) * i / 500 for i in range(501)]
        values = [cosine_lr(step, total) / BASE_LR for step in steps]
        ax.plot([step / 1000 for step in steps], values, color=color, lw=2.8, label=label)
        for step in (50_000, 100_000):
            value = cosine_lr(step, total)
            ax.scatter(step / 1000, value / BASE_LR, s=45, color=color, zorder=4)
            ax.annotate(f'{value:.2e}', (step / 1000, value / BASE_LR),
                        xytext=(9, 10 if total == 200_000 else -19), textcoords='offset points',
                        color=color, fontsize=11)
        print(f'total={total}: ' + ', '.join(f'lr({step})={cosine_lr(step, total):.10g}'
                                          for step in (50_000, 100_000)))
    ax.axhline(MIN_RATIO, color='#64748b', lw=1, ls='--', label='Minimum: 10% of base LR')
    ax.set(xlim=(0, 205), ylim=(-0.02, 1.12), xlabel='Scheduler step (thousands)',
           ylabel='Learning rate / base learning rate', xticks=[0, 50, 100, 150, 200],
           yticks=[0.1, 0.25, 0.5, 0.75, 1.0])
    ax.grid(color='#e2e8f0', lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
    ax.tick_params(colors='#334155')
    ax.set_title('Changing the budget changes the cosine curve', loc='left',
                 fontsize=17, weight='bold', color='#0f172a', pad=20)
    ax.text(0, 1.02, 'Cosine phase starts at step 3,000  |  Base LR = 1e-4',
            transform=ax.transAxes, fontsize=11, color='#475569')
    ax.legend(loc='upper right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', fontsize=10)
    output = Path(__file__).resolve().parent / 'assets' / 'learning-rate-budget.svg'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, metadata={'Date': None, 'Title': 'X-VLA training budget and learning rate',
                                 'Description': 'Formula plot after warmup; no training results measured.'})
    plt.close(fig)
    print(output.name)


if __name__ == '__main__':
    main()
