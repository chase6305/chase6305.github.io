"""Check positive Q/K scale cancellation on constructed one-head activations.

Only the hash-checked Qwen3RMSNorm class is executed from the supplied official
file. No model, attention module, weights, RoPE, mask, GQA or training is loaded.
An independent Decimal reference separates epsilon effects from FP32 rounding.
"""
import argparse
import ast
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path


SOURCE_SHA256 = '704c914530530a1acb0b443add1f520404e3ac2c28c0ab7e16f80f86cfe8ccb2'


def reference_norm(values, weights, epsilon):
    with localcontext() as context:
        context.prec = 80
        vector = [Decimal.from_float(float(value)) for value in values]
        variance = sum(value * value for value in vector) / len(vector)
        denominator = (variance + Decimal.from_float(epsilon)).sqrt()
        return [float(Decimal.from_float(weight) * value / denominator)
                for weight, value in zip(weights, vector)]


def probabilities(query, keys):
    logits = [math.fsum(q * k for q, k in zip(query, key)) / math.sqrt(len(query))
              for key in keys]
    exponentials = [math.exp(value - max(logits)) for value in logits]
    total = math.fsum(exponentials)
    return [value / total for value in exponentials]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.official_file.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        parser.error('Expected the pinned Transformers v4.51.3 Qwen3 source')
    import torch
    torch.set_num_threads(1)
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name == 'Qwen3RMSNorm')
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(args.official_file), 'exec'), namespace)
    query = [1., 2., -1., 3.]
    keys = [[1., 0., 2., -1.], [-2., 1., .5, 2.], [.5, -1., -2., 1.]]
    q_weight, k_weight = [1., .5, 1.25, -.75], [.8, 1.2, -.5, 1.5]
    scales = [(1., 1.), (.01, .01), (.1, 10.), (10., .1),
              (10., 10.), (1e-6, 1e-6), (1000., .001)]
    cases = []
    for epsilon in [0., 1e-6]:
        q_norm = namespace['Qwen3RMSNorm'](4, eps=epsilon).double()
        k_norm = namespace['Qwen3RMSNorm'](4, eps=epsilon).double()
        with torch.no_grad():
            q_norm.weight.copy_(torch.tensor(q_weight, dtype=torch.float64))
            k_norm.weight.copy_(torch.tensor(k_weight, dtype=torch.float64))
        base_query = reference_norm(query, q_weight, epsilon)
        base_keys = [reference_norm(key, k_weight, epsilon) for key in keys]
        base_probabilities = probabilities(base_query, base_keys)
        for q_scale, k_scale in scales:
            scaled_query = [value * q_scale for value in query]
            scaled_keys = [[value * k_scale for value in key] for key in keys]
            expected_query = reference_norm(scaled_query, q_weight, epsilon)
            expected_keys = [reference_norm(key, k_weight, epsilon) for key in scaled_keys]
            with torch.no_grad():
                actual_query = q_norm(torch.tensor(scaled_query, dtype=torch.float64)).tolist()
                actual_keys = k_norm(torch.tensor(scaled_keys, dtype=torch.float64)).tolist()
            expected_probabilities = probabilities(expected_query, expected_keys)
            actual_probabilities = probabilities(actual_query, actual_keys)
            norm_error = max(abs(actual - expected)
                             for actual, expected in zip(actual_query + sum(actual_keys, []),
                                                         expected_query + sum(expected_keys, [])))
            assert norm_error < 8e-7, norm_error
            ideal_change = max(abs(a - b) for a, b in zip(expected_probabilities, base_probabilities))
            if epsilon == 0.:
                assert ideal_change < 8e-16, ideal_change
            cases.append({
                'epsilon': epsilon, 'q_scale': q_scale, 'k_scale': k_scale,
                'decimal_reference_probabilities': expected_probabilities,
                'official_class_probabilities': actual_probabilities,
                'reference_probability_change_from_unit_scale': ideal_change,
                'official_vs_reference_max_norm_error': norm_error,
                'official_vs_reference_max_probability_error': max(
                    abs(a - b) for a, b in zip(actual_probabilities, expected_probabilities)),
            })
    raw_baseline = probabilities(query, keys)
    raw_scaled = probabilities([10. * value for value in query],
                               [[10. * value for value in key] for key in keys])
    epsilon_case = next(row for row in cases if row['epsilon'] == 1e-6 and row['q_scale'] == 1e-6)
    assert epsilon_case['reference_probability_change_from_unit_scale'] > .1
    assert max(abs(a - b) for a, b in zip(raw_baseline, raw_scaled)) > .1
    report = {
        'scope': 'Constructed one-head activations and three keys; fixed nonuniform learned RMS weights. Only the original Qwen3RMSNorm class runs on CPU. No model, attention module, RoPE, mask, GQA, dropout or training.',
        'source_url': 'https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py',
        'source_sha256': SOURCE_SHA256, 'source_class_lines': [62, 79],
        'torch': torch.__version__, 'input_and_weight_dtype': 'float64',
        'official_internal_normalization_dtype': 'float32',
        'decimal_precision': 80, 'query': query, 'keys': keys,
        'q_norm_weight': q_weight, 'k_norm_weight': k_weight,
        'cases': cases,
        'raw_attention_control': {'unit_scale': raw_baseline, 'q_and_k_scaled_by_ten': raw_scaled},
        'interpretation': 'Positive scale cancels in the epsilon-zero mathematical normalization with fixed learned weights. Fixed positive epsilon breaks exact cancellation at small activation scale, and official FP32 arithmetic has rounding even at epsilon zero. This does not measure a pretrained model or establish a training effect.',
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'cases': len(cases),
                      'maximum_official_vs_reference_norm_error': max(row['official_vs_reference_max_norm_error'] for row in cases),
                      'maximum_official_vs_reference_probability_error': max(row['official_vs_reference_max_probability_error'] for row in cases),
                      'small_scale_epsilon_probability_change': epsilon_case['reference_probability_change_from_unit_scale']}))


if __name__ == '__main__':
    main()
