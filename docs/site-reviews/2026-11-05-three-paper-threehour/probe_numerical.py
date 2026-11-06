#!/usr/bin/env python3
"""Optional independent NumPy/SciPy oracles; no training or robot calibration."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.interpolate import BSpline
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    last = load('last_geometry', 'content/posts/ai/last-action-tokenizer/geometry_sanity.py')
    omni = load('omni_observability', 'content/posts/calibration/omnicalib/observability_sanity.py')
    rng = np.random.default_rng(261119582)
    errors = {key:0. for key in ['se3_exp', 'se3_log', 'so3_rotvec', 'whitening', 'spline_basis', 'spline_ridge', 'profiled_derivative', 'pose_compose', 'pose_inverse', 'tool_adjoint', 'whitening_choice', 'excited_joint_mean', 'excited_joint_covariance', 'held_velocity_projection', 'sample_statistic_decomposition', 'sample_statistic_norm']}
    counts = {key:0 for key in errors}

    def record(name, actual, expected, tolerance):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        assert error < tolerance, (name, error, tolerance)
        errors[name] = max(errors[name], error)
        counts[name] += 1

    for angle in [1e-8, 1e-5, .01, .2, 1., 2., 3., math.pi-.01]:
        for _ in range(15):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            xi = np.r_[rng.normal(size=3), axis * angle]
            matrix = np.zeros((4,4))
            matrix[:3,:3] = np.asarray(last.skew(xi[3:]))
            matrix[:3,3] = xi[:3]
            expected = expm(matrix)
            r, t = last.se3_exp(xi.tolist())
            actual = np.eye(4)
            actual[:3,:3], actual[:3,3] = r, t
            record('se3_exp', actual, expected, 2e-10)
            independent_log = logm(expected)
            assert np.max(np.abs(np.imag(independent_log))) < 1e-9
            independent_log = np.real(independent_log)
            vector = np.r_[independent_log[:3,3], independent_log[2,1], independent_log[0,2], independent_log[1,0]]
            record('se3_log', last.se3_log(expected[:3,:3].tolist(), expected[:3,3].tolist()), vector, 2e-9)
            record('so3_rotvec', last.so3_log(expected[:3,:3].tolist()), Rotation.from_matrix(expected[:3,:3]).as_rotvec(), 2e-9)
    for _ in range(200):
        angle = rng.uniform(-math.pi, math.pi)
        q = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        hi = 10**rng.uniform(-2,3)
        lo = hi / 10**rng.uniform(0,4)
        covariance = q @ np.diag([lo,hi]) @ q.T
        values, vectors = np.linalg.eigh(covariance)
        expected = (vectors / np.sqrt(values)) @ vectors.T
        record('whitening', last.whitening_2d(covariance.tolist()), expected, 2e-7)
    for horizon in [8,16,30,61]:
        u = np.linspace(0,1,horizon)
        for degree in [0,2,4]:
            knots = last.clamped_knots(16,degree)
            expected = BSpline.design_matrix(u, knots, degree).toarray()
            actual = np.array([last.spline_basis(float(x),16,degree) for x in u])
            record('spline_basis',actual,expected,2e-14)
            samples = np.sin(2.3*u)+.2*np.cos(8*u)
            ridge = .01
            # Augmented SVD least squares is independent of normal equations.
            augmented = np.vstack([expected,np.sqrt(ridge)*np.eye(16)])
            target = np.r_[samples,np.zeros(16)]
            controls = np.linalg.lstsq(augmented,target,rcond=None)[0]
            record('spline_ridge',last.fit_spline(samples.tolist(),16,degree,ridge),controls,3e-12)
    for _ in range(100):
        nuisance = rng.normal(size=(9,3))
        g = rng.normal(size=9)
        expected = g - nuisance @ np.linalg.lstsq(nuisance,g,rcond=None)[0]
        record('profiled_derivative',omni.profile_derivative(g.tolist(),nuisance.T.tolist()),expected,3e-13)
    for _ in range(120):
        a, b = np.eye(4), np.eye(4)
        a[:3,:3], b[:3,:3] = Rotation.random(random_state=rng).as_matrix(), Rotation.random(random_state=rng).as_matrix()
        a[:3,3], b[:3,3] = rng.normal(size=3), rng.normal(size=3)
        pose_a = (a[:3,:3].tolist(),a[:3,3].tolist())
        pose_b = (b[:3,:3].tolist(),b[:3,3].tolist())
        rotation,translation = last.pose_compose(pose_a,pose_b)
        record('pose_compose',np.c_[rotation,translation],(a@b)[:3,:],2e-14)
        rotation,translation = last.pose_inverse(pose_a)
        record('pose_inverse',np.c_[rotation,translation],np.linalg.inv(a)[:3,:],2e-14)
        xi = rng.uniform(-.3,.3,size=6)
        hat = np.zeros((4,4));hat[:3,:3]=np.asarray(last.skew(xi[3:]));hat[:3,3]=xi[:3]
        changed = logm(np.linalg.inv(a)@expm(hat)@a).real
        expected = np.r_[changed[:3,3],changed[2,1],changed[0,2],changed[1,0]]
        record('tool_adjoint',last.adjoint(last.pose_inverse(pose_a),xi.tolist()),expected,3e-12)
    for _ in range(200):
        ratio,angle,epsilon = rng.uniform(1,10),rng.uniform(0,90),rng.uniform(0,10)
        result = last.whitening_choice(ratio,angle,epsilon)
        covariance = np.asarray(result['covariance'])
        deltas = np.array([[2.,0.],[0.,-1.]])
        expected = [np.sqrt(d@np.linalg.solve(covariance,d)) for d in deltas]
        record('whitening_choice',[result['metricA'],result['metricB']],expected,2e-13)
    for excitation in [0.,1e-8,1e-4,.01,.1,.5,1.]:
        for _ in range(25):
            prior,std = rng.uniform(-.5,.5),rng.uniform(.005,.3)
            design = np.array([[50.,50.],[50*excitation,-50*excitation],[1/std,0.]])
            target = np.array([5.,15*excitation,prior/std])
            expected = np.linalg.lstsq(design,target,rcond=None)[0]
            _, singular, vh = np.linalg.svd(design,full_matrices=False)
            expected_cov = (vh.T/singular**2)@vh
            k,a,cov = omni.excited_joint_fit(prior,std,excitation)
            record('excited_joint_mean',[k,a],expected,3e-13)
            record('excited_joint_covariance',cov,expected_cov,3e-14)
    for dimension in [3,6]:
        for _ in range(60):
            codes = rng.normal(size=(5,dimension))
            weights = rng.uniform(.01,1,size=5);weights /= weights.sum()
            z = rng.normal(size=dimension)*3
            center = weights@codes
            residual = z-center
            epsilon = 10**rng.uniform(-3,0)
            deviations = codes-center
            spread = deviations.T@np.diag(weights)@deviations
            outer_sum = (z-codes).T@np.diag(weights)@(z-codes)+epsilon*np.eye(dimension)
            a = spread+epsilon*np.eye(dimension)
            record('sample_statistic_decomposition',outer_sum,np.outer(residual,residual)+a,8e-14)
            values,vectors = np.linalg.eigh(outer_sum)
            whitened = (vectors/np.sqrt(values))@vectors.T@residual
            t = residual@np.linalg.solve(a,residual)
            record('sample_statistic_norm',whitened@whitened,t/(1+t),4e-12)
            assert whitened@whitened < 1+4e-12
    deployment = load('deployment','content/posts/ai/machine-learning-for-robotics/deployment_sanity.py')
    # Independent interval projection: enumerate both constraint endpoints.
    for _ in range(200):
        x,dt,limit,disturbance = rng.uniform(-1,.3),rng.uniform(.001,1),rng.uniform(.1,5),rng.uniform(0,6)
        nominal = rng.uniform(-10,10)
        actual = deployment.held_velocity(x,nominal,dt,actuator=limit,disturbance_upper=disturbance)
        upper = min(limit,(.3-x)/dt-disturbance)
        if upper < -limit:
            assert actual is None
            continue
        candidates = [-limit,upper]
        if -limit <= nominal <= upper:candidates.append(nominal)
        expected = min(candidates,key=lambda candidate:(candidate-nominal)**2)
        record('held_velocity_projection',actual,expected,2e-14)
    report = {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'seed':261119582,'counts':counts,'maximum_absolute_errors':errors,
              'scope':'Small-matrix geometry, observability, and scalar held-input constraints only. No LAST training, sensor fitting, calibration accuracy, robot behavior, or statistical coverage is reproduced.'}
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
