"""Compare fixed models on paired IID binary-scored questions.

No arguments: run the article's explicitly synthetic 100-question example.
Real files: --baseline baseline.jsonl --candidate candidate.jsonl --ids ids.json

Each JSONL row must contain a unique nonempty string id, a boolean correct,
and status='ok' or 'error'. Error rows must have correct=false. The expected
ID manifest prevents silently dropping questions from both model files.
This does not score raw answers, infer correctness or handle clustered data.
"""
import argparse
import json
import statistics
from pathlib import Path

from metrics_lab import paired_bootstrap, wilson_interval


def validate_rows(rows, label):
    result = {}
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {number}: expected an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{label} row {number}: id must be a nonempty string")
        if identifier in result:
            raise ValueError(f"{label}: duplicate id {identifier!r}")
        if type(row.get("correct")) is not bool:
            raise ValueError(f"{label} {identifier}: correct must be a JSON boolean")
        if row.get("status") not in ("ok", "error"):
            raise ValueError(f"{label} {identifier}: status must be ok or error")
        if row["status"] == "error" and row["correct"]:
            raise ValueError(f"{label} {identifier}: an error cannot be scored correct")
        result[identifier] = row
    return result


def compare(baseline_rows, candidate_rows, expected_ids, repeats=10000, seed=17):
    if (not isinstance(expected_ids, list) or not expected_ids
            or any(not isinstance(x, str) or not x.strip() for x in expected_ids)
            or len(set(expected_ids)) != len(expected_ids)):
        raise ValueError("Expected IDs must be a nonempty list of unique nonempty strings")
    baseline = validate_rows(baseline_rows, "baseline")
    candidate = validate_rows(candidate_rows, "candidate")
    expected = set(expected_ids)
    for name, rows in (("baseline", baseline), ("candidate", candidate)):
        if set(rows) != expected:
            missing = sorted(expected - set(rows))
            extra = sorted(set(rows) - expected)
            raise ValueError(f"{name}: ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
    # Stable ordering makes seeded resampling invariant to JSONL row order.
    identifiers = sorted(expected)
    a = [int(baseline[x]["correct"]) for x in identifiers]
    b = [int(candidate[x]["correct"]) for x in identifiers]
    gain, lower, upper = paired_bootstrap(a, b, repeats=repeats, seed=seed)
    both_correct = sum(x and y for x, y in zip(a, b))
    only_baseline = sum(x and not y for x, y in zip(a, b))
    only_candidate = sum(not x and y for x, y in zip(a, b))
    both_wrong = sum(not x and not y for x, y in zip(a, b))
    warnings = [
        "Assumes independent representative questions; use group-aware methods for related questions.",
        "Conditions on these two fixed checkpoints; does not measure training-seed variability.",
        "Resamples supplied scores only; does not rerun generation or isolate decoding variability.",
        "Percentile bootstrap can be unreliable at boundaries or with very few discordant questions.",
    ]
    if lower == upper:
        warnings.append("Bootstrap interval is degenerate; it does not establish zero population uncertainty.")
    return {
        "questions": len(identifiers),
        "baseline_accuracy": statistics.mean(a),
        "candidate_accuracy": statistics.mean(b),
        "baseline_wilson_95": list(wilson_interval(sum(a), len(a))),
        "candidate_wilson_95": list(wilson_interval(sum(b), len(b))),
        "difference_pp": gain * 100,
        "difference_percentile_95_pp": [lower * 100, upper * 100],
        "contingency": {
            "both_correct": both_correct,
            "only_baseline_correct": only_baseline,
            "only_candidate_correct": only_candidate,
            "both_wrong": both_wrong,
        },
        "errors_counted_in_denominator": {
            "baseline": sum(row["status"] == "error" for row in baseline.values()),
            "candidate": sum(row["status"] == "error" for row in candidate.values()),
        },
        "resampling_unit": "paired_question",
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "warnings": warnings,
    }


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def demo():
    a = [1] * 60 + [1] * 10 + [0] * 15 + [0] * 15
    b = [1] * 60 + [0] * 10 + [1] * 15 + [0] * 15
    ids = [f"q{i:03d}" for i in range(100)]
    baseline = [{"id": i, "correct": bool(x), "status": "ok"} for i, x in zip(ids, a)]
    candidate = [{"id": i, "correct": bool(x), "status": "ok"} for i, x in zip(ids, b)]
    return baseline, candidate, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--ids", type=Path, help="JSON array of all expected test question IDs")
    parser.add_argument("--repeats", type=int, default=10000,
                        help="bootstrap resamples, not generated answers (default: 10000)")
    parser.add_argument("--seed", type=int, default=17,
                        help="bootstrap RNG seed, not a training or generation seed (default: 17)")
    args = parser.parse_args()
    supplied = (args.baseline is not None, args.candidate is not None, args.ids is not None)
    if any(supplied) and not all(supplied):
        parser.error("provide --baseline, --candidate and --ids together")
    try:
        if all(supplied):
            baseline, candidate = load_jsonl(args.baseline), load_jsonl(args.candidate)
            ids = json.loads(args.ids.read_text(encoding="utf-8"))
        else:
            baseline, candidate, ids = demo()
        report = compare(baseline, candidate, ids, args.repeats, args.seed)
        report["data_kind"] = "user_supplied_scores" if all(supplied) else "synthetic_example"
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
