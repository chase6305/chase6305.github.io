"""Reproduce the article's illustrative metrics; no model training or downloads."""
import math
import random
import statistics
import json
from pathlib import Path

# Synthetic percentages for the article's 2x2 ablation, not measured results.
# A: neither factor; B: dedup only; C: response mask only; D: both.
ABLATION_SCORES = {"A": 70.0, "B": 72.0, "C": 71.0, "D": 76.0}


def weighted_nll(loss_sums, counts):
    """Aggregate NLL sums using counts after causal shift and loss masking."""
    if (not counts or len(loss_sums) != len(counts)
            or any(type(n) is not int or n < 0 for n in counts)):
        raise ValueError("Provide aligned NLL sums and nonnegative target counts")
    if any(not math.isfinite(v) or v < 0 for v in loss_sums):
        raise ValueError("NLL sums must be finite and nonnegative")
    if any(n == 0 and v != 0 for n, v in zip(counts, loss_sums)):
        raise ValueError("An empty target group must contribute zero NLL")
    if sum(counts) == 0:
        raise ValueError("At least one supervised target is required")
    return math.fsum(loss_sums) / sum(counts)


def pass_at_k(n, c, k):
    """HumanEval-style per-task estimator; candidates use one sampling protocol."""
    if any(type(x) is not int for x in (n, c, k)) or not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("Require 0 <= c <= n and 1 <= k <= n")
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0.0)


def percentile(sorted_values, q):
    """Linearly interpolated empirical quantile; input must already be sorted."""
    if not sorted_values or not 0 <= q <= 1:
        raise ValueError("Need values and a quantile in [0, 1]")
    index = (len(sorted_values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def paired_bootstrap(baseline, candidate, repeats=10000, seed=17):
    """Percentile interval over paired IID evaluation units, NOT training seeds."""
    if (not baseline or len(baseline) != len(candidate)
            or type(repeats) is not int or repeats < 2):
        raise ValueError("Need nonempty aligned predictions and >= 2 resamples")
    if any(not math.isfinite(v) for v in (*baseline, *candidate)):
        raise ValueError("Scores must be finite")
    differences = [b - a for a, b in zip(baseline, candidate)]
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(statistics.mean(rng.choices(differences, k=n))
                     for _ in range(repeats))
    return statistics.mean(differences), percentile(samples, 0.025), percentile(samples, 0.975)


def wilson_interval(successes, trials, confidence=0.95):
    """Two-sided Wilson interval for IID Bernoulli trials, not paired differences."""
    if (type(successes) is not int or type(trials) is not int
            or not 0 <= successes <= trials or trials <= 0 or not 0 < confidence < 1):
        raise ValueError("Invalid binomial counts or confidence level")
    z = statistics.NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == trials else min(1.0, center + radius)
    return lower, upper


def binary_calibration(probabilities, outcomes, bins=5):
    """Event-probability ECE and Brier; ECE uses fixed equal-width bins.

    This is calibration of P(event=1), not max-class-confidence ECE.
    Bins are [left, right), except the final bin includes probability 1.
    """
    if (not probabilities or len(probabilities) != len(outcomes)
            or type(bins) is not int or bins <= 0):
        raise ValueError("Need aligned predictions, outcomes and positive bin count")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must lie in [0, 1]")
    if any(type(y) not in (int, bool) or y not in (0, 1) for y in outcomes):
        raise ValueError("Outcomes must be binary")
    groups = [[] for _ in range(bins)]
    for p, y in zip(probabilities, outcomes):
        groups[min(int(p * bins), bins - 1)].append((p, y))
    n = len(outcomes)
    ece = sum(len(g) / n * abs(statistics.mean(p for p, _ in g)
                             - statistics.mean(y for _, y in g)) for g in groups if g)
    brier = statistics.mean((p - y)**2 for p, y in zip(probabilities, outcomes))
    return ece, brier


def qwen_parameter_breakdown(config):
    """Count the pinned Qwen2 dense architecture, including QKV bias and RMSNorm.

    Not a generic Transformer, quantized checkpoint or MoE parameter counter.
    """
    if config.get("model_type") != "qwen2" or config.get("tie_word_embeddings") is not False:
        raise ValueError("This example requires dense Qwen2 with untied embeddings")
    d, layers = config["hidden_size"], config["num_hidden_layers"]
    ff, vocab = config["intermediate_size"], config["vocab_size"]
    q_heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    if any(type(v) is not int or v <= 0 for v in (d, layers, ff, vocab, q_heads, kv_heads)):
        raise ValueError("Dimensions must be positive integers")
    if d % q_heads or q_heads % kv_heads:
        raise ValueError("Head dimensions must divide evenly")
    kv_width = kv_heads * (d // q_heads)
    return {
        "embedding": vocab * d,
        "lm_head": vocab * d,
        "attention_matrices": layers * (2 * d * d + 2 * d * kv_width),
        "qkv_bias": layers * (d + 2 * kv_width),
        "gated_mlp": layers * 3 * d * ff,
        "layer_norms": layers * 2 * d,
        "final_norm": d,
    }


def main():
    weights = 7_000_000_000 * 2
    print(f"7B BF16: {weights / 10**9:.2f} GB = {weights / 2**30:.2f} GiB")
    kv = 2 * 32 * 1 * 4096 * 8 * 128 * 2
    print(f"KV cache: {kv / 2**20:.0f} MiB")
    nll = weighted_nll([2 * 1.0, 8 * 3.0], [2, 8])
    print(f"Weighted NLL: {nll:.4f}; PPL: {math.exp(nll):.4f}")
    probabilities = [0.5, 0.25, 0.125]
    ppl = math.exp(-statistics.mean(math.log(p) for p in probabilities))
    print(f"Toy probabilities PPL: {ppl:.4f}")
    # Each position has exactly two candidate tokens. These are probabilities
    # of the true target; the other candidate has probability 1-p.
    for name, targets in (("A", [0.6, 0.6, 0.6, 0.49]),
                          ("B", [0.99, 0.99, 0.99, 0.01]),
                          ("C", [0.51] * 4)):
        accuracy = statistics.mean(p > 0.5 for p in targets)
        mean_nll = -statistics.mean(math.log(p) for p in targets)
        print(f"Toy {name}: token accuracy={accuracy:.2%}; NLL={mean_nll:.4f}")
    response_sums, response_counts = [2.0, 24.0], [2, 8]
    token_mean = weighted_nll(response_sums, response_counts)
    response_mean = statistics.mean(s / n for s, n in zip(response_sums, response_counts))
    print(f"Token-mean NLL: {token_mean:.4f}; response-mean NLL: {response_mean:.4f}")
    print(f"pass@3 (n=10, c=2): {pass_at_k(10, 2, 3):.2%}")
    wins, losses, ties = 48, 32, 20
    print(f"Win rate (ties=0.5): {(wins + 0.5 * ties) / (wins + losses + ties):.2%}")
    full = weighted_nll([100 * 0.2, 20 * 2.0], [100, 20])
    print(f"Full-sequence NLL: {full:.4f}; response-only NLL: {2.0:.4f}")
    margins = [0.2, -0.1, 0.4, 0.0]
    print(f"DPO pair accuracy: {statistics.mean(m > 0 for m in margins):.2%}; "
          f"mean margin: {statistics.mean(margins):.4f}")
    print(f"TPOT: {(5.75 - 0.8) / (100 - 1) * 1000:.2f} ms")
    # Contingency counts: both pass, correct only, on-time only, neither.
    # The two conditions are observed per request, not assumed independent.
    for name, counts in (("A", (78, 2, 12, 8)), ("B", (60, 25, 10, 5))):
        both, _, _, _ = counts
        seconds = 10
        print(f"Quality goodput {name}: {both / seconds:.2f} requests/s; "
              f"joint pass: {both / sum(counts):.2%}")
    baseline, candidate = [70, 72, 71], [73, 73, 74]
    deltas = [b - a for a, b in zip(baseline, candidate)]
    print(f"Paired gain: {statistics.mean(deltas):.2f} pp; "
          f"sample std: {statistics.stdev(deltas):.2f} pp")
    print(f"Relative gain: {statistics.mean(deltas) / statistics.mean(baseline):.2%}")
    a_score, b_score, c_score, d_score = (ABLATION_SCORES[k] for k in "ABCD")
    effect_full = b_score - a_score
    effect_response = d_score - c_score
    print(f"Dedup effects: {effect_full:.2f} / {effect_response:.2f} pp; "
          f"interaction contrast: {effect_response - effect_full:.2f} pp")
    # Synthetic binary outcomes for 100 paired questions: both correct=60,
    # only baseline correct=10, only candidate correct=15, both wrong=15.
    a = [1] * 60 + [1] * 10 + [0] * 15 + [0] * 15
    b = [1] * 60 + [0] * 10 + [1] * 15 + [0] * 15
    delta, low, high = paired_bootstrap(a, b)
    print(f"Toy paired bootstrap: {delta * 100:.2f} pp; "
          f"95% percentile interval [{low * 100:.2f}, {high * 100:.2f}] pp")
    lower, upper = wilson_interval(0, 100)
    print(f"Wilson 95% interval (0/100): [{lower:.2%}, {upper:.2%}]")
    ece, brier = binary_calibration([0.8] * 10, [1] * 8 + [0] * 2)
    print(f"Binary event calibration: ECE={ece:.4f}; Brier={brier:.4f}")
    snapshot = Path(__file__).with_name("qwen2.5-7b-budget.json")
    if snapshot.is_file():
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        total = sum(qwen_parameter_breakdown(data["config"]).values())
        if total != data["expected_parameters"] or total * 2 != data["expected_weight_bytes"]:
            raise ValueError("Qwen configuration and expected metadata disagree")
        print(f"Qwen2.5-7B pinned config: {total:,} parameters; {total * 2 / 2**30:.4f} GiB BF16")
    else:
        print("Optional Qwen audit skipped: download qwen2.5-7b-budget.json beside this script.")
    print("Illustrative arithmetic and configuration audit; no model training or quality measurement.")


if __name__ == "__main__":
    main()
