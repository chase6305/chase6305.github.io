#!/usr/bin/env python3
"""Compare the printed RTC mask with a hash-pinned official pure function.

The extracted function runs with NumPy replacing jnp. This checks its Float64
array algebra, not JAX compilation, accelerator precision or policy inference.
The checkout is only read and its source is verified before extraction.
"""
import argparse
import ast
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
COMMIT = '9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b'
SHA256 = 'f59e9479c4fe4386ca027ba1a5c0b4940552adfae8cdb38d177cefec1427482e'


def check(source_root):
    revision = subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT:
        raise RuntimeError('Expected RTC source commit '+COMMIT)
    source = source_root/'src/model.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError('RTC model.py source hash mismatch')
    import numpy as np
    tree = ast.parse(source.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_prefix_weights')
    namespace = {'jnp': np, 'jax': SimpleNamespace(Array=np.ndarray), 'PrefixAttentionSchedule': str}
    unit = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(unit, str(source), 'exec'), namespace)
    official = namespace['get_prefix_weights']
    spec = importlib.util.spec_from_file_location('article_contracts', ROOT/'content/posts/ai/rma-umi-rtc-deployment/deployment_contracts.py')
    article = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(article)
    cases, error_source, error_article = 0, 0., 0.
    fixtures = []
    for horizon in range(1, 33):
        for stride in range(1, horizon+1):
            for delay in range(min(stride, horizon-stride)+1):
                end = horizon-stride
                observed = official(delay, end, horizon, 'exp')
                helper = article.rtc_weights(horizon, stride, delay)
                with localcontext() as context:
                    context.prec = 70
                    one, expected = Decimal(1), []
                    for i in range(horizon):
                        c = Decimal(end-i)/Decimal(end-delay+1)
                        expected.append(1. if i < delay else float(c*(c.exp()-1)/(one.exp()-1)) if i < end else 0.)
                error_source = max(error_source, float(np.max(np.abs(observed-expected))))
                error_article = max(error_article, float(np.max(np.abs(np.asarray(helper)-expected))))
                if (horizon, stride, delay) in ((8, 3, 2), (16, 5, 4)):
                    fixtures.append({'horizon': horizon, 'stride': stride, 'delay': delay,
                                     'official_float64_weights': observed.tolist(), 'decimal_70_weights': expected})
                cases += 1
    assert error_source < 1e-14 and error_article < 1e-14
    # The fixed c=1/2 expression is also reducible to a sigmoid, independent
    # of evaluating the exponential numerator/denominator separately.
    half = float(official(2, 5, 8, 'exp')[3])
    assert abs(half-.5/(np.sqrt(np.e)+1)) < 1e-15
    extended = article.timing_budget(16, 4, 20, 40, 60, 20)
    np.testing.assert_allclose(extended['weights'], official(6, 12, 16, 'exp'), rtol=0, atol=1e-14)
    return {'commit': COMMIT, 'model_sha256': SHA256, 'numpy': np.__version__, 'mask_cases': cases,
            'source_maximum_absolute_error_against_decimal70': error_source,
            'article_maximum_absolute_error_against_decimal70': error_article,
            'fixtures': fixtures, 'separate_worker_and_deadline_profile': True,
            'regression': {'c': .5, 'correct_weight': half, 'exponential_only_weight': 1/(np.sqrt(np.e)+1)},
            'scope': 'Verified function body, NumPy Float64 adapter and independent Decimal; no JAX or model execution.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(args.source_root)
    data = json.dumps(result, indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(data)
    print(data, end='')
