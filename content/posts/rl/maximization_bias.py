"""Isolate selection bias with unbiased Gaussian estimates, not RL training.

Requires NumPy and Matplotlib. Writes a JSON summary and a two-panel plot.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("bias-results"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    trials, actions = 100000, 10
    # Identical true values = 0, independent unit-variance estimation errors.
    selector = rng.normal(size=(trials, actions))
    evaluator = rng.normal(size=selector.shape)
    chosen = selector.argmax(axis=1)
    coupled = selector[np.arange(trials), chosen]
    decoupled = evaluator[np.arange(trials), chosen]
    samples = {"fixed_action": selector[:, 0],
               "same_estimator": coupled, "independent_evaluator": decoupled}
    report = {"numpy": np.__version__, "seed": 42, "trials": trials,
              "actions": actions, "all_true_values": 0.0,
              "error_distribution": "independent N(0,1)",
              "results": {k: {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                              "mean_standard_error": float(v.std(ddof=1)/np.sqrt(trials))}
                          for k, v in samples.items()},
              "scope": "Independent synthetic estimators; learned Double Q tables need not be independent."}
    assert abs(samples["fixed_action"].mean()) < .02
    assert 1.4 < coupled.mean() < 1.7
    assert abs(decoupled.mean()) < .02
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"maximization-bias-results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    bins = np.linspace(-5, 6, 90)
    axes[0].hist(coupled, bins=bins, density=True, histtype="step", linewidth=2,
                 color="#9866c0", label="Select and evaluate with A")
    axes[0].hist(decoupled, bins=bins, density=True, histtype="step", linewidth=2,
                 color="#3989ce", label="Select with A, evaluate with B")
    axes[0].axvline(0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Reported value of selected action", ylabel="Density",
                title="Each action is unbiased before selection")
    axes[0].legend(fontsize=9)
    keys = list(samples)
    means = [report["results"][k]["mean"] for k in keys]
    sems = [report["results"][k]["mean_standard_error"] for k in keys]
    axes[1].bar(range(3), means, color=["#77ab8a", "#9866c0", "#3989ce"],
                yerr=np.array(sems)*1.96, capsize=5, alpha=.8)
    axes[1].axhline(0, color="#333333", linewidth=1)
    for i, value in enumerate(means):
        axes[1].text(i, value+.08, f"{value:.3f}", ha="center")
    axes[1].set(xticks=range(3), xticklabels=["Fixed action", "Select A\nEvaluate A", "Select A\nEvaluate B"],
                ylabel="Mean estimated value", ylim=(-.18, 1.85),
                title="Mean +/- 1.96 Monte Carlo standard errors")
    fig.suptitle("10 actions, true values = 0; 100,000 independent trials")
    fig.savefig(args.output_dir/"maximization-bias.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
