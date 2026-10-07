"""Synthetic paired bootstrap: rows versus independent question groups.

NumPy required; Matplotlib only for --figure. No model outputs or downloads.
Twenty independent-group assumptions are contrasted with falsely treating
repeated within-group outcomes as independent rows. This is not a real benchmark.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_means(values, repeats, seed):
    rng = np.random.default_rng(seed)
    samples = []
    # Bounded memory; each resample draws the same number of units as observed.
    for start in range(0, repeats, 500):
        indices = rng.integers(0, len(values), size=(min(500, repeats-start), len(values)))
        samples.extend(values[indices].mean(axis=1))
    return np.asarray(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('cluster-evaluation.json'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    repeats, seed, rows_per_group = 20000, 20260928, 20
    # Six improved groups, four degraded groups, ten tied groups.
    baseline = np.array([0]*6 + [1]*4 + [1]*5 + [0]*5)
    candidate = np.array([1]*6 + [0]*4 + [1]*5 + [0]*5)
    differences = candidate-baseline
    repeated_differences = np.repeat(differences, rows_per_group)
    # This reduction is exact only because this demo has equal-size groups.
    per_group = bootstrap_means(differences, repeats, seed)
    per_row = bootstrap_means(repeated_differences, repeats, seed)
    analytic_se_group = np.sqrt(np.var(differences, ddof=0)/len(differences))
    analytic_se_row = np.sqrt(np.var(repeated_differences, ddof=0)/len(repeated_differences))
    np.testing.assert_allclose(analytic_se_group/analytic_se_row, np.sqrt(rows_per_group), atol=1e-14)
    for distribution, expected in [(per_group, analytic_se_group), (per_row, analytic_se_row)]:
        assert abs(distribution.mean()-differences.mean()) < .005
        assert abs(distribution.std()-expected)/expected < .03
    lower_row, upper_row = np.quantile(per_row, [.025, .975])
    lower_group, upper_group = np.quantile(per_group, [.025, .975])
    assert lower_row > 0 and lower_group < 0 < upper_group
    # Unequal group sizes change the estimand if each group gets equal weight.
    unequal_sizes = np.array([2, 8])
    unequal_means = np.array([1., -1.])
    micro = np.average(unequal_means, weights=unequal_sizes)
    macro = unequal_means.mean()
    np.testing.assert_allclose([micro, macro], [-.6, 0.])
    report = {'data_kind': 'synthetic_example', 'numpy': np.__version__,
              'groups': len(differences), 'rows_per_group': rows_per_group,
              'total_rows': len(repeated_differences), 'bootstrap_repeats': repeats, 'bootstrap_seed': seed,
              'baseline_accuracy': float(baseline.mean()), 'candidate_accuracy': float(candidate.mean()),
              'difference_pp': float(differences.mean()*100),
              'paired_rows_as_iid': {'percentile_95_pp': [lower_row*100, upper_row*100],
                                      'exact_empirical_bootstrap_se_pp': float(analytic_se_row*100)},
              'paired_groups': {'percentile_95_pp': [lower_group*100, upper_group*100],
                                'exact_empirical_bootstrap_se_pp': float(analytic_se_group*100)},
              'unequal_group_example': {'sizes': unequal_sizes.tolist(), 'differences': unequal_means.tolist(),
                                       'question_weighted_difference_pp': float(micro*100),
                                       'group_weighted_difference_pp': float(macro*100)},
              'scope': 'Fixed synthetic paired scores. Equal-size independent groups with identical outcomes within each group. No training or generation variability, no coverage study, and no guarantee that a 20-group percentile interval is well calibrated.'}
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        for row, interval, color in [(1, (lower_row, upper_row), '#ae89c5'),
                                     (0, (lower_group, upper_group), '#528bb5')]:
            low, high = np.asarray(interval)*100
            axes[0].plot([low, high], [row, row], color=color, linewidth=4, solid_capstyle='round')
            axes[0].scatter([10], [row], color=color, s=60, zorder=3)
            axes[0].text((low+high)/2, row+.13, f'[{low:g}, {high:g}] pp', ha='center', fontsize=10)
        axes[0].axvline(0, color='#b98045', linestyle=':')
        axes[0].set(yticks=[0, 1], yticklabels=['20 paired groups', '400 rows as IID\n(incorrect here)'],
                    ylim=(-.5, 1.55), xlabel='Accuracy difference [percentage points]',
                    title='Same +10 pp; different resampling units')
        copies = np.array([1, 2, 5, 10, 20, 50])
        axes[1].plot(copies, np.full(len(copies), analytic_se_group*100), 'o-', color='#528bb5', label='Independent groups remain 20')
        axes[1].plot(copies, analytic_se_group*100/np.sqrt(copies), 's--', color='#ae89c5', label='Incorrect independent-row assumption')
        axes[1].set(xscale='log', xlabel='Repeated identical rows per group',
                    ylabel='Empirical bootstrap standard error [pp]', title='Copying records does not add information')
        axes[1].legend(fontsize=8, loc='center right')
        for ax in axes: ax.grid(alpha=.15)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=160); plt.close(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
