"""Independently recompute the revised examples and bind the tested Torch snippet."""
import argparse
import contextlib
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {slug: ROOT / f"content/posts/ai/{slug}/index.md"
             for slug in ("supermimic", "rlhf-rlvr-rlcd")}
    sources = {slug: p.read_text() for slug, p in paths.items()}
    llm = sources["rlhf-rlvr-rlcd"]
    assert r"$\sigma(z)=1/(1+e^{-z})$" in llm
    probability_rows = re.findall(r"^\| ([02]) \| ([\d.]+) \| ([\d.]+) \|$", llm, re.M)
    assert len(probability_rows) == 2
    checked_probabilities = []
    with localcontext() as ctx:
        ctx.prec = 90
        for difference, probability, loss in probability_rows:
            delta = Decimal(difference)
            expected_probability = 1 / (1 + (-delta).exp())
            expected_loss = -expected_probability.ln()
            assert expected_probability.quantize(Decimal("0.000001")) == Decimal(probability)
            assert expected_loss.quantize(Decimal("0.000001")) == Decimal(loss)
            checked_probabilities.append({"difference": difference, "published_probability": probability,
                                          "published_loss": loss, "probability_90_digits": str(expected_probability),
                                          "loss_90_digits": str(expected_loss)})
    critic_rows = re.findall(r"^\| ([AB]) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$",
                             sources["supermimic"], re.M)
    assert len(critic_rows) == 2
    checked_critic = []
    for name, window, value, total in critic_rows:
        expected = Fraction(window) + Fraction(1, 2) * Fraction(value)
        assert expected == Fraction(total)
        checked_critic.append({"trajectory": name, "window": window, "value": value,
                               "terminal_coefficient": "1/2", "total": total})
    assert "用总体标准差" in llm and "$(1,1,0,0)$" in llm and "$(1,1,-1,-1)$" in llm
    rewards = tuple(map(Fraction, (1, 1, 0, 0)))
    mean = sum(rewards) / 4
    variance = sum((r - mean) ** 2 for r in rewards) / 4
    assert mean == Fraction(1, 2) and variance == Fraction(1, 4)
    advantages = [(r - mean) / Fraction(1, 2) for r in rewards]
    assert advantages == [1, 1, -1, -1]

    def guarded_divide(a, b):
        return None if b == 0 else a / b

    def direct_divide(a, b):
        return a / b

    verifier_scores = []
    for function in (guarded_divide, direct_divide, guarded_divide):
        try:
            passed = function(6, 3) == 2 and function(5, 0) is None
        except ZeroDivisionError:
            passed = False
        verifier_scores.append(int(passed))
    assert verifier_scores == [1, 0, 1]
    snippets = re.findall(r"```python\n(.*?)```", sources["supermimic"], re.S)
    finite_difference = next(s for s in snippets if "return_two_steps" in s)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(finite_difference, str(paths["supermimic"]), "exec"), {})
    assert stdout.getvalue() == "gradient=8.0, return=-1.000\n"
    torch_code = next(s for s in snippets if "import torch" in s)
    torch_sha256 = hashlib.sha256(torch_code.encode()).hexdigest()
    receipt = json.loads(args.torch_receipt.read_text())
    assert torch_sha256 == receipt["published_snippet_sha256"]
    assert receipt["published_output"] == "actor gradient=8.0\nmodel gradient=None\n"
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
              "bradley_terry": checked_probabilities, "critic_comparison": checked_critic,
              "grpo_population_mean": str(mean), "grpo_population_variance": str(variance),
              "grpo_advantages": list(map(str, advantages)), "constructed_verifier_scores": verifier_scores,
              "published_finite_difference_output": stdout.getvalue(),
              "torch_snippet_unchanged_since_cpu_execution": True, "torch_snippet_sha256": torch_sha256,
              "scope": "Arithmetic on values extracted from current article tables, the published standard-library snippet, two constructed function-interface tests, and a hash binding to earlier CPU autograd evidence. No human-judgment quality test, robot, GPU or model training."}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"preference_rows": len(checked_probabilities), "critic_rows": len(checked_critic),
                      "verifier_scores": verifier_scores, "torch_snippet_unchanged": True}))


if __name__ == "__main__":
    main()
