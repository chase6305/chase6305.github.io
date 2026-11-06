#!/usr/bin/env python3
"""Pinned SureSim statistical-function probe on constructed arrays, not robots."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import subprocess

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/chase-suresim-review-mpl')
import numpy as np

COMMIT = 'e9e6cb567f8792630044e5c1662ed6e1db9c7480'
SOURCE_SHA256 = 'e5d8025a524f6bab9060965c1bd30d07d19ddc4e43cf6069a7619b4b41b4c691'
ROOT = Path(__file__).resolve().parent


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def independent_log_wealth(values, candidate, alpha, c):
    """Scalar predictable-betting recurrence, independent of NumPy cumulative ops."""
    n = len(values)
    total, squared, variance = 0., 0., .25
    plus, minus = 0., 0.
    result = []
    for i, value in enumerate(values, 1):
        rate = math.sqrt(2*math.log(2/alpha)/(n*variance))
        plus += math.log1p(min(rate,c/candidate)*(value-candidate))
        minus += math.log1p(-min(rate,c/(1-candidate))*(value-candidate))
        result.append(max(plus,minus)-math.log(2))
        total += value
        mean = (.5+total)/(1+i)
        squared += (value-mean)**2
        variance = (.25+squared)/(1+i)
    return result



def independent_interval(values, lower, upper, alpha, c):
    """Invert the scalar prefix-wealth predicate with independent brackets.

    This checks numerical endpoints for these arrays, not statistical validity.
    """
    normalized = [(float(value)-lower)/(upper-lower) for value in values]
    center = math.fsum(normalized)/len(normalized)
    threshold = math.log(1/alpha)
    accepts = lambda candidate: max(independent_log_wealth(normalized,candidate,alpha,c)) < threshold
    if not accepts(center):
        raise ValueError('sample mean is not accepted; this endpoint oracle needs a different bracket')
    left, right = 1e-12, center
    if accepts(left):
        lo = 0.
    else:
        for _ in range(40):
            middle = (left+right)/2
            if accepts(middle):right=middle
            else:left=middle
        lo = (left+right)/2
    left, right = center, 1-1e-12
    if accepts(right):
        hi = 1.
    else:
        for _ in range(40):
            middle = (left+right)/2
            if accepts(middle):left=middle
            else:right=middle
        hi = (left+right)/2
    return [lo*(upper-lower)+lower,hi*(upper-lower)+lower]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-endpoints', action='store_true', help='Independently invert the scalar wealth predicate for each constructed case')
    args = parser.parse_args()
    actual_commit = subprocess.check_output(['git','-C',str(args.source_root),'rev-parse','HEAD'], text=True).strip()
    if actual_commit != COMMIT:
        raise ValueError('source checkout commit does not match the pinned revision')
    source = args.source_root/'src/suresim/intervals/ppi.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('statistical source SHA-256 does not match the pinned file')
    official = module_from_file('official_suresim_ppi', source)
    example = module_from_file('deployment_example', ROOT/'deployment_sanity.py')
    seed, alpha, c = 20261105, .1, .99
    rng = np.random.default_rng(seed)
    y = np.array([0.]*15+[1.]*45)
    rng.shuffle(y)
    moderate = y.copy()
    moderate[np.flatnonzero(y == 0)[:3]] = 1
    moderate[np.flatnonzero(y == 1)[:3]] = 0
    captured = []
    original = official.mean_ci_eff_corrected_membership_accelerated

    def observe(z, risk, **kwargs):
        captured.append({'values':np.array(z,copy=True), 'risk':risk, **kwargs})
        return original(z, risk, **kwargs)

    official.mean_ci_eff_corrected_membership_accelerated = observe
    classical = official.gold_only_est(y, alpha=alpha, c=c)
    results = []
    for name, paired in [('perfect',y),('partly_predictive',moderate),('anticorrelated',1-y)]:
        for n_extra in [20,60,200,700]:
            extra = np.tile([0.,1.,1.,1.] if name != 'anticorrelated' else [1.,0.,0.,0.], n_extra//4)
            rng.shuffle(extra)
            np.random.seed(seed+n_extra)
            calls_before = len(captured)
            interval = official.ppi_uniform(y, paired, extra, alpha=alpha, c=c)
            assert len(captured) == calls_before+1
            call = captured[-1]
            point = example.ppi_pooled(y.tolist(),paired.tolist(),extra.tolist())
            assert abs(float(call['values'].mean())-point) < 2e-14
            assert min(call['values']) >= call['L'] and max(call['values']) <= call['U']
            assert all(math.isfinite(float(x)) for x in interval) and interval[0] <= interval[1]
            clipped = [max(0.,float(interval[0])),min(1.,float(interval[1]))]
            endpoint_check = {}
            if args.check_endpoints:
                independent = independent_interval(call['values'],call['L'],call['U'],alpha,c)
                error = max(abs(float(a)-b) for a,b in zip(interval,independent))/(call['U']-call['L'])
                if error > 2e-6:
                    raise ValueError(f'independent endpoint mismatch: {error}')
                endpoint_check = {'independent_raw_interval':independent,'maximum_normalized_endpoint_error':error}
            results.append({'fixture':name,'paired':len(y),'additional':n_extra,'estimate':point,**endpoint_check,
                            'paired_correlation':float(np.corrcoef(y,paired)[0,1]),
                            'raw_interval':list(map(float,interval)), 'intersection_0_1':clipped,
                            'width':clipped[1]-clipped[0], 'pseudovalue_range':[call['L'],call['U']]})
    # Check the underlying wealth recurrence independently on ordinary bounded
    # data; this verifies implementation arithmetic, not statistical coverage.
    values = [.1,.8,.9,.2,.7,.3,.95,.55]
    count = len(values)
    mu = (.5+np.cumsum(values))/(1+np.arange(1,count+1))
    variances = (.25+np.cumsum((np.array(values)-mu)**2))/(1+np.arange(1,count+1))
    rates = np.sqrt(2*np.log(2/alpha)/(count*np.r_[.25,variances[:-1]]))
    max_error = 0.
    for candidate in [.05,.25,.5,.75,.95]:
        expected = independent_log_wealth(values,candidate,alpha,c)
        actual = official.MARTINGALE(candidate,c,rates,np.array(values))
        error = float(np.max(np.abs(actual-np.array(expected))))
        assert error < 2e-13
        max_error = max(max_error,error)
    report = {'source_commit':COMMIT,'source_sha256':SOURCE_SHA256,'python':platform.python_version(),
              'numpy':np.__version__,'seed':seed,'alpha':alpha,'c':c,
              'classical_interval':list(map(float,classical)),
              'classical_width':float(classical[1]-classical[0]),'cases':results,
              'independent_endpoint_checks':24 if args.check_endpoints else 0,
              'independent_log_wealth_checks':40,'maximum_log_wealth_error':max_error,
              'scope':'Constructed fixed arrays and original statistical functions only. No paper dataset, hardware, policy training, or coverage guarantee is reproduced.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(results),'classical_width':report['classical_width'],
                      'last_case':results[-1],'maximum_log_wealth_error':max_error},indent=2))


if __name__ == '__main__':
    main()
