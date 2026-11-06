#!/usr/bin/env python3
"""Optional NumPy/SciPy cross-checks of the three article helpers.

Run outside the site environment. These are small constructed mathematical
fixtures, not training, hardware tests, or a reproduction of paper results.
No paper implementation is imported by this driver.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy import linalg, optimize, signal, special, stats


ROOT = Path(__file__).resolve().parents[3]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def check():
    deployment = module('deployment_contracts', 'content/posts/ai/rma-umi-rtc-deployment/deployment_contracts.py')
    residual = module('residual_control', 'content/posts/ai/residual-reinforcement-learning/residual_control_sanity.py')
    contact = module('contact_calibration', 'content/posts/calibration/interactive-contact-calibration/contact_calibration_sanity.py')
    rng = np.random.default_rng(20261106)
    report = {'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'scope': 'Constructed mathematical examples; no model, hardware or paper benchmark.'}

    # SciPy valid convolution followed by subsampling is independent of the
    # helper's integer output-length formula; coefficients are irrelevant here.
    sequence = np.arange(50, dtype=float)
    observed = [len(sequence)]
    for kernel, stride in ((8, 4), (5, 1), (5, 1)):
        sequence = signal.convolve(sequence, np.ones(kernel), mode='valid')[::stride]
        observed.append(len(sequence))
    assert observed == deployment.convolution_lengths()
    report['convolution_lengths'] = observed

    # A nonsymmetric Jacobian and a zero output mask expose transpose mistakes
    # and the distinction between output masking and masking input gradients.
    velocity_matrix = np.array([[.2, -.1, .3], [.4, .1, -.2], [-.3, .2, .1]])
    target, offset, mask = np.array([.3, -.1, .4]), np.array([.1, -.2, .05]), np.array([1., .5, 0.])
    guidance_errors = []
    for tau in (0., .001, .1, .4, .5, .9, .999, 1.):
        for value in rng.normal(size=(25, 3)):
            def objective(x):
                endpoint = x+(1-tau)*(velocity_matrix @ x+offset)
                return .5*np.dot(mask, (endpoint-target)**2)
            loss, correction = deployment.linear_guidance(value, tau)
            assert abs(loss-objective(value)) < 1e-13
            # SciPy's finite difference evaluates the independently written
            # NumPy objective, not the helper's loss implementation.
            error = optimize.check_grad(objective, lambda x: -np.asarray(deployment.linear_guidance(x, tau)[1]), value)
            assert error < 2e-6, (tau, value, error)
            guidance_errors.append(float(error))
    assert abs(deployment.linear_guidance()[1][2]) > .001
    report['guidance'] = {'cases': len(guidance_errors), 'maximum_absolute_gradient_error': max(guidance_errors)}

    actor_errors, clipped_cases = [], 0
    scale = np.array([[2., .5], [-.25, 3.]])
    for saturated in (False, True):
        for theta in rng.uniform(-1.5, 1.5, size=(100, 2)):
            _, gradient, raw = residual.actor_chain_example(theta, saturated)
            if saturated and np.min(np.abs(np.abs(raw)-1)) < 1e-4:
                continue
            def objective(x):
                command = np.array([.4, -.2])+scale @ np.array([x[0]+2*x[1], np.prod(x)])
                command = np.clip(command, -1, 1) if saturated else command
                return -np.dot(np.array([1., 2.]), (command-np.array([.1, .3]))**2)
            error = optimize.check_grad(objective, lambda x: np.asarray(residual.actor_chain_example(x, saturated)[1]), theta)
            assert error/max(1., linalg.norm(gradient)) < 2e-6
            actor_errors.append(float(error))
            clipped_cases += int(saturated and np.any(np.abs(raw) > 1))
    report['actor_chain_rule'] = {'cases': len(actor_errors), 'clipped_cases': clipped_cases,
                                  'maximum_absolute_gradient_error': max(actor_errors)}

    # Full SE(3) left perturbations use matrix exponentials, rather than the
    # helper's hand-written axis rotations, to check all six columns.
    jacobian_errors = []
    h = 1e-6
    for _ in range(100):
        point, normal = rng.normal(size=(2, 3))
        normal /= linalg.norm(normal)
        predicted = np.asarray(contact.point_plane_row(point, normal))
        for axis in range(6):
            tangent = np.zeros((4, 4))
            if axis < 3:
                tangent[axis, 3] = 1
            else:
                omega = np.eye(3)[axis-3]
                tangent[:3, :3] = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
            plus = linalg.expm(h*tangent) @ np.r_[point, 1]
            minus = linalg.expm(-h*tangent) @ np.r_[point, 1]
            derivative = normal @ (plus[:3]-minus[:3])/(2*h)
            jacobian_errors.append(float(abs(derivative-predicted[axis])))
    assert max(jacobian_errors) < 1e-8
    horizontal, diverse = map(np.asarray, contact.geometry_rows())
    singular_horizontal = linalg.svdvals(horizontal)
    singular_diverse = linalg.svdvals(diverse)
    nullspace = linalg.null_space(horizontal)
    np.testing.assert_allclose(horizontal @ nullspace, 0, atol=1e-14)
    assert nullspace.shape == (6, 3) and np.min(singular_diverse) > .1
    report['point_plane'] = {'derivatives': len(jacobian_errors), 'maximum_absolute_error': max(jacobian_errors),
                             'horizontal_singular_values': singular_horizontal.tolist(),
                             'diverse_singular_values': singular_diverse.tolist(),
                             'horizontal_nullity': nullspace.shape[1],
                             'units_note': 'Exact fixture coordinates only; these singular values are not a mixed-unit acceptance threshold.'}

    # An independent SciPy Gaussian density agrees with the finite grid when
    # its full covariance and truncation are both represented explicitly.
    axis = np.arange(-20, 21, dtype=float)
    xx, yy = np.meshgrid(axis, axis, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel()])
    posterior_errors, covariance_errors = [], []
    for sigma in (1., 2., 4., 8.):
        for angle in (5., 15., 30., 60., 90.):
            radians = np.deg2rad(angle)
            design = np.array([[0., 1.], [np.sin(radians), np.cos(radians)]])
            covariance = linalg.inv(design.T @ design)*sigma**2
            covariance_errors.append(float(np.max(np.abs(covariance-contact.unbounded_covariance(sigma, angle)))))
            logs = stats.multivariate_normal(mean=[6., -4.], cov=covariance).logpdf(points)
            mass = np.exp(logs-special.logsumexp(logs))
            mean = mass @ points
            spread = mass @ ((points-mean)**2)
            expected = contact.posterior_example(sigma, angle, True)
            error = np.max(np.abs(np.r_[mean, np.sqrt(spread)]-
                                 [expected['meanX'], expected['meanY'], expected['stdX'], expected['stdY']]))
            assert error < 1e-10
            posterior_errors.append(float(error))
    report['posterior'] = {'cases': len(posterior_errors), 'maximum_mean_std_error_mm': max(posterior_errors),
                           'maximum_covariance_error_mm_squared': max(covariance_errors)}

    literal_errors = []
    for parameter in (.0001, .005, .02):
        distances = rng.uniform(-.02, .02, size=(20, 8))
        expected_logs = stats.norm(scale=np.sqrt(parameter)).logpdf(distances).sum(axis=1)
        expected = np.exp(expected_logs-special.logsumexp(expected_logs))
        observed = contact.literal_v1_log_weights(distances, [1]*8, parameter)
        literal_errors.append(float(np.max(np.abs(expected-observed))))
    assert max(literal_errors) < 1e-13
    report['printed_kernel'] = {'cases': len(literal_errors), 'maximum_normalized_weight_error': max(literal_errors),
                                'scope': 'Equal contact counts across particles; shared printed prefactors cancel.'}
    correlation_errors = []
    for count in (2, 5, 25, 100):
        ones = np.ones(count)
        for rho in (0., .1, .5, .9, .999):
            covariance = 4*((1-rho)*np.eye(count)+rho*np.ones((count, count)))
            precision = ones @ linalg.solve(covariance, ones, assume_a='pos')
            expected = np.sqrt(1/precision)
            observed = contact.correlated_contact_scale(count, rho, 2.)
            correlation_errors.append(float(abs(expected-observed)))
    assert max(correlation_errors) < 1e-12
    report['correlated_observations'] = {'cases': len(correlation_errors),
                                        'maximum_fused_standard_deviation_error': max(correlation_errors)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check()
    serialized = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized, end='')
