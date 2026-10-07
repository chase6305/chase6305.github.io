"""Deterministic counterexamples for confidence-gated routing.

All records are deliberately constructed, not Jev outputs or robot trials.
The validation/test names illustrate a split protocol; no data are sampled.
Core calculations and checks use Python 3.10+ standard library only.
Matplotlib is needed only when --figure is requested.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
import unittest


GROUPS = {
    "validation": {
        "label": "合成验证组", "description": "仅用来选阈值，40 条构造记录。",
        "bins": [(0.95, 10, 10), (0.85, 10, 9), (0.65, 10, 7), (0.45, 10, 5)],
    },
    "heldout": {
        "label": "合成留出组", "description": "记录 ID 与验证组不重叠，80 条构造记录。",
        "bins": [(0.95, 20, 19), (0.85, 20, 17), (0.65, 20, 14), (0.45, 20, 10)],
    },
    "shifted": {
        "label": "合成分布变化组", "description": "高分段加入更多错误，80 条构造记录。",
        "bins": [(0.95, 20, 12), (0.85, 20, 18), (0.65, 20, 15), (0.45, 20, 10)],
    },
}
THRESHOLDS = (0.0, 0.6, 0.8, 0.9, 1.0)


def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def unit_interval(value):
    return finite(value) and 0 <= value <= 1


def fixture_records():
    records = []
    for group, spec in GROUPS.items():
        for score, count, correct in spec["bins"]:
            for index in range(count):
                records.append({"id": f"{group}-{round(score * 100)}-{index + 1:02d}",
                                "group": group, "routing_score": score,
                                "acceptable": index < correct})
    return records


def validate_records(records):
    if not records:
        raise ValueError("nonempty records required")
    ids = set()
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("record ID required")
        if row["id"] in ids:
            raise ValueError("duplicate record ID")
        ids.add(row["id"])
        if not unit_interval(row.get("routing_score")) or type(row.get("acceptable")) is not bool:
            raise ValueError("finite routing score and boolean label required")


def route_metrics(records, threshold, fast_cost=1.0, fallback_cost=20.0):
    validate_records(records)
    if not unit_interval(threshold):
        raise ValueError("threshold must be in [0, 1]")
    if any(not finite(x) or x < 0 for x in (fast_cost, fallback_cost)):
        raise ValueError("costs must be finite and nonnegative")
    accepted = [r for r in records if r["routing_score"] >= threshold]
    errors = sum(not r["acceptable"] for r in accepted)
    rejected = len(records) - len(accepted)
    total_cost = len(records) * fast_cost + rejected * fallback_cost
    if not finite(total_cost):
        raise ValueError("cost total overflow")
    return {
        "threshold": threshold, "records": len(records), "accepted": len(accepted),
        "rejected": rejected, "accepted_errors": errors,
        "accept_rate": len(accepted) / len(records),
        "selective_error": errors / len(accepted) if accepted else None,
        "total_call_cost": total_cost, "mean_call_cost": total_cost / len(records),
        "always_fallback_mean_cost": fallback_cost,
    }


def select_threshold(validation, thresholds, max_error=0.05, min_accepted=10):
    """Empirical selection only; this is not a population risk guarantee."""
    if not unit_interval(max_error) or type(min_accepted) is not int or min_accepted < 1:
        raise ValueError("invalid selection criterion")
    if not thresholds:
        raise ValueError("candidate thresholds required")
    rows = [route_metrics(validation, t) for t in thresholds]
    feasible = [r for r in rows if r["accepted"] >= min_accepted
                and r["selective_error"] <= max_error]
    return min(feasible, key=lambda r: (-r["accepted"], r["threshold"])) if feasible else None


def binary_calibration(probabilities, labels, bins=10):
    """Scores an explicitly defined binary-event probability, not confidence."""
    if (not probabilities or len(probabilities) != len(labels)
            or type(bins) is not int or bins < 1):
        raise ValueError("nonempty aligned arrays and positive bin count required")
    if any(not unit_interval(p) for p in probabilities) or any(type(y) is not bool for y in labels):
        raise ValueError("invalid probability or label")
    groups = [[] for _ in range(bins)]
    for p, y in zip(probabilities, labels):
        groups[min(int(p * bins), bins - 1)].append((p, y))
    reliability = []
    for index, group in enumerate(groups):
        if not group:
            continue
        reliability.append({"bin": index, "count": len(group),
                            "mean_probability": math.fsum(p for p, _ in group) / len(group),
                            "event_frequency": sum(y for _, y in group) / len(group)})
    brier = math.fsum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
    ece = math.fsum(r["count"] * abs(r["mean_probability"] - r["event_frequency"])
                    for r in reliability) / len(labels)
    return {"brier": brier, "ece": ece, "bins": reliability}


def wilson_interval(successes, trials, confidence=0.95):
    if (type(trials) is not int or type(successes) is not int
            or trials < 1 or not 0 <= successes <= trials):
        raise ValueError("valid binomial counts required")
    if not finite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def build_results():
    records = fixture_records()
    grouped = {g: [r for r in records if r["group"] == g] for g in GROUPS}
    selected = select_threshold(grouped["validation"], THRESHOLDS)
    labels_a = [True] * 10
    labels_b = [True] * 8 + [False] * 2
    return {
        "scope": "Constructed offline records, not Jev measurements or sampled robot trials.",
        "cost_unit": "arbitrary call-cost unit; excludes execution, retries and fallback outcomes",
        "threshold_selection": {"group": "validation", "grid": list(THRESHOLDS),
                                "max_empirical_error": 0.05, "min_accepted": 10,
                                "selected": selected},
        "groups": {g: {"label": GROUPS[g]["label"], "description": GROUPS[g]["description"],
                       "curve": [route_metrics(rows, t) for t in THRESHOLDS],
                       "selected_threshold": route_metrics(rows, selected["threshold"])}
                   for g, rows in grouped.items()},
        "calibration_counterexample": {
            "scope": "Separate binary probability fixture; not a transformation of routing scores.",
            "aggregate": binary_calibration([0.9] * 20, labels_a + labels_b),
            "group_a": binary_calibration([0.9] * 10, labels_a),
            "group_b": binary_calibration([0.9] * 10, labels_b),
        },
        "wilson_arithmetic_example": {
            "assumption": "Hypothetical independent Bernoulli trials, not an interval claim for these constructed records.",
            "errors": 1, "trials": 20, "interval_95": wilson_interval(1, 20),
        },
    }


def save_figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"validation": "#4779c5", "heldout": "#7b60ac", "shifted": "#ce7d35"}
    styles = {"validation": ("o", "-"), "heldout": ("s", "--"), "shifted": ("^", "-.")}
    labels = {"validation": "Constructed validation", "heldout": "Constructed held-out", "shifted": "Constructed shift"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for group, data in results["groups"].items():
        rows = [r for r in data["curve"] if r["selective_error"] is not None]
        axes[0].plot([r["threshold"] for r in rows], [100 * r["selective_error"] for r in rows],
                     marker=styles[group][0], linestyle=styles[group][1],
                     color=colors[group], label=labels[group])
    axes[0].axvline(0.8, color="#455568", linestyle=":", label="Selected on validation: 0.80")
    axes[0].set(xlabel="Routing threshold", ylabel="Error among accepted decisions (%)",
                title="Higher threshold need not mean lower error", xlim=(-0.02, 1.02), ylim=(-1, 46))
    axes[0].text(1, 43, "No accepted\nrecords at 1.0", ha="right", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left")
    rows = results["groups"]["heldout"]["curve"]
    axes[1].plot([100 * r["accept_rate"] for r in rows], [r["mean_call_cost"] for r in rows],
                 marker="o", color="#447c63", label="Selector = 1, fallback = 20")
    axes[1].axhline(20, color="#8e96a0", linestyle="--", label="Always call fallback")
    for row in rows:
        axes[1].annotate(f"t={row['threshold']:.1f}",
                         (100 * row["accept_rate"], row["mean_call_cost"]),
                         xytext=(4, 5), textcoords="offset points", fontsize=8)
    axes[1].set(xlabel="Accepted decisions (%)", ylabel="Mean call cost (arbitrary units)",
                title="A rejected decision still pays for the selector", xlim=(-3, 109), ylim=(0, 25))
    axes[1].legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Synthetic routing counterexamples — not model measurements", fontsize=13)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.records = fixture_records()
        self.validation = [r for r in self.records if r["group"] == "validation"]

    def test_split_counts_and_identity(self):
        self.assertEqual(len(self.records), 200)
        self.assertEqual(len({r["id"] for r in self.records}), 200)
        self.assertEqual([sum(r["group"] == g for r in self.records) for g in GROUPS], [40, 80, 80])

    def test_selection_uses_only_validation(self):
        picked = select_threshold(self.validation, THRESHOLDS)
        self.assertEqual(picked["threshold"], 0.8)
        self.assertEqual((picked["accepted"], picked["accepted_errors"]), (20, 1))

    def test_shift_can_reverse_threshold_ordering(self):
        rows = [r for r in self.records if r["group"] == "shifted"]
        self.assertEqual(route_metrics(rows, 0.8)["selective_error"], 0.25)
        self.assertEqual(route_metrics(rows, 0.9)["selective_error"], 0.4)

    def test_empty_acceptance_is_not_zero_error(self):
        row = route_metrics(self.validation, 1)
        self.assertIsNone(row["selective_error"])
        self.assertEqual(row["mean_call_cost"], 21)
        self.assertIsNone(select_threshold(self.validation, [1]))

    def test_cost_counts_both_calls(self):
        self.assertEqual(route_metrics(self.validation, 0.8)["mean_call_cost"], 11)
        self.assertEqual(route_metrics(self.validation, 0.8, fallback_cost=1)["mean_call_cost"], 1.5)

    def test_subgroup_calibration_is_hidden_by_aggregation(self):
        result = build_results()["calibration_counterexample"]
        self.assertAlmostEqual(result["aggregate"]["ece"], 0)
        self.assertAlmostEqual(result["aggregate"]["brier"], 0.09)
        self.assertAlmostEqual(result["group_a"]["ece"], 0.1)
        self.assertAlmostEqual(result["group_b"]["ece"], 0.1)

    def test_probability_endpoints_and_wilson_symmetry(self):
        result = binary_calibration([0.0, 1.0], [False, True])
        self.assertEqual((result["brier"], result["ece"]), (0, 0))
        low = wilson_interval(1, 20)
        high = wilson_interval(19, 20)
        self.assertAlmostEqual(low[0], 1 - high[1])
        self.assertAlmostEqual(low[1], 1 - high[0])
        self.assertGreater(wilson_interval(0, 10)[1], 0)

    def test_invalid_data(self):
        for records in ([], [self.records[0], self.records[0]],
                        [{**self.records[0], "routing_score": float("nan")}],
                        [{**self.records[0], "acceptable": 1}]):
            with self.assertRaises(ValueError):
                route_metrics(records, 0.8)
        with self.assertRaises(ValueError):
            route_metrics(self.validation, 0.8, fast_cost=10 ** 1000)
        with self.assertRaises(ValueError):
            binary_calibration([0.9], [True, False])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RoutingTests))
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    result = build_results()
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    records = fixture_records()
    if args.samples:
        args.samples.write_text(json.dumps({"scope": result["scope"], "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "group", "routing_score", "acceptable"],
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    if args.figure:
        save_figure(result, args.figure)
    print(json.dumps({"selected_threshold": result["threshold_selection"]["selected"]["threshold"],
                      "groups": {g: row["selected_threshold"] for g, row in result["groups"].items()},
                      "wilson_arithmetic_example": result["wilson_arithmetic_example"]}, ensure_ascii=False, indent=2))
