"""Known-hypothesis Bayesian example; no learned model or GEN-1.5 inference.

Python 3.10+, standard library. Likelihoods below are assumed numbers,
not measured probabilities. Run with --output results.json to save results.
"""
import argparse
import json
import math
from pathlib import Path


def distribution(values):
    values = tuple(values)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Nonempty finite nonnegative weights required')
    total = sum(values)
    if total <= 0:
        raise ValueError('Total weight must be positive')
    return tuple(v / total for v in values)


def posterior(prior, likelihood):
    prior = distribution(prior)
    likelihood = tuple(likelihood)
    if len(prior) != len(likelihood):
        raise ValueError('One likelihood for each hypothesis required')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in likelihood):
        raise ValueError('Likelihoods must be probabilities')
    return distribution(p * value for p, value in zip(prior, likelihood))


def entropy(probabilities):
    return -sum(p * math.log2(p) for p in distribution(probabilities) if p > 0)


def expected_information(prior, outcomes_by_hypothesis):
    """Mutual information, before observing the result of a candidate setup.

    Rows are hypotheses, columns are mutually exclusive observed outcomes.
    This assumes that this likelihood model describes a new observation.
    """
    prior = distribution(prior)
    rows = tuple(tuple(row) for row in outcomes_by_hypothesis)
    if len(rows) != len(prior) or not rows or not rows[0]:
        raise ValueError('A nonempty likelihood row for each hypothesis required')
    if any(len(row) != len(rows[0]) or any(not math.isfinite(v) or v < 0 for v in row)
           or not math.isclose(sum(row), 1., abs_tol=1e-12) for row in rows):
        raise ValueError('Every likelihood row must sum to one')
    expected_entropy = 0.
    for likelihood in zip(*rows):
        probability = sum(p * value for p, value in zip(prior, likelihood))
        if probability > 0:
            expected_entropy += probability * entropy(posterior(prior, likelihood))
    return entropy(prior) - expected_entropy


def example():
    prior = (1/3, 1/3, 1/3)
    ambiguous = posterior(prior, (.95, .95, .95))
    informative = posterior(prior, (.05, .95, .05))
    repeated_as_independent = posterior(informative, (.05, .95, .05))
    redundant = expected_information(prior, ((.95, .05),) * 3)
    distinguishing = expected_information(prior, ((.05, .95), (.95, .05), (.05, .95)))
    assert all(math.isclose(a, b) for a, b in zip(prior, ambiguous))
    assert all(math.isclose(a, b) for a, b in zip(informative, (1/21, 19/21, 1/21)))
    assert math.isclose(redundant, 0, abs_tol=1e-12)
    assert 0 < distinguishing < entropy(prior)
    assert math.isclose(entropy((1, 0, 0)), 0)
    assert math.isclose(entropy(tuple(reversed(informative))), entropy(informative))
    # Perfectly distinguishing observations reveal exactly the prior entropy.
    assert math.isclose(expected_information(prior, ((1,0,0),(0,1,0),(0,0,1))), entropy(prior))
    for bad_prior, bad_likelihood in [((0,0,0), (1,1,1)), (prior, (0,0,0)),
                                      (prior, (.1,.9)), (prior, (.1,float('nan'),.3))]:
        try:
            posterior(bad_prior, bad_likelihood)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid evidence accepted')
    return {'scope': 'Constructed hypotheses and assumed likelihoods; not GEN-1.5 probabilities or a trained task recognizer.',
            'hypotheses': ['left container', 'blue container', 'old absolute position'],
            'prior': prior, 'ambiguous_posterior': ambiguous,
            'new_evidence_posterior': informative,
            'prior_entropy_bits': entropy(prior),
            'new_evidence_entropy_bits': entropy(informative),
            'repeat_counted_as_independent_posterior': repeated_as_independent,
            'expected_information_bits': {'redundant_setup': redundant,
                                          'distinguishing_setup': distinguishing},
            'checks': ['uninformative evidence', 'Bayes normalization', 'entropy bounds',
                       'permutation invariance', 'perfect identification', 'invalid input rejection']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(example(), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end='')
