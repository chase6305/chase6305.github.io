#!/usr/bin/env python3
"""Analytic observability examples and OmniCalib v2 Table II arithmetic.

No robot, sensor data, calibration solver, or hardware commands are involved.
All scalar angular quantities are in degrees unless explicitly named radians.
"""
import argparse
import math


# Paper Table II. Keep its printed residuals separate from rounded columns.
ROWS = [
    ("left hip pitch", -.278, 2.100, 1.849, .027, .0118),
    ("left hip roll", -.120, -1.550, -1.700, -.030, .0118),
    ("left hip yaw", -.025, 1.350, 1.477, .153, .0238),
    ("left knee", -.107, -2.200, -2.318, -.011, .0090),
    ("left ankle pitch", -.777, 1.700, .912, -.011, .0088),
    ("left ankle roll", .643, -1.250, -.558, .050, .0133),
    ("right hip pitch", -.555, 1.850, 1.309, .014, .0111),
    ("right hip roll", .301, -1.350, -1.015, .034, .0119),
    ("right hip yaw", .926, 1.550, 2.349, -.127, .0236),
    ("right knee", -.120, -2.050, -2.179, -.009, .0089),
    ("right ankle pitch", .010, 1.400, 1.401, -.009, .0087),
    ("right ankle roll", .122, -1.650, -1.573, -.045, .0130),
]


def rms(values):
    values = list(values)
    if not values:
        raise ValueError("RMS needs nonempty data")
    return math.sqrt(sum(x * x for x in values) / len(values))


def knee_ankle_fit(observed_sum, knee_prior, data_precision=100., prior_precision=1.):
    """Fit .5*data_precision*(k+a-y)^2 + .5*prior_precision*(k-b)^2."""
    if data_precision <= 0 or prior_precision <= 0:
        raise ValueError("positive precisions required")
    # For this underdetermined data, any positive prior selects the knee value.
    knee, ankle = knee_prior, observed_sum - knee_prior
    p, s = data_precision, prior_precision
    hessian = [[p + s, p], [p, p]]
    posterior_covariance = [[1 / s, -1 / s], [-1 / s, 1 / s + 1 / p]]
    return knee, ankle, hessian, posterior_covariance


def excited_joint_fit(knee_prior=.4, prior_std=.02, excitation=0.):
    """Two observations plus a knee prior, solved by reorthogonalized QR.

    y1=k+a=.1; y2=e*(k-a)=.3*e. Independent noise std .02 deg.
    The constructed, noise-free observation values correspond to (.2,-.1).
    """
    if not (math.isfinite(knee_prior) and -.5 <= knee_prior <= .5 and
            math.isfinite(prior_std) and .005 <= prior_std <= .3 and
            math.isfinite(excitation) and 0 <= excitation <= 1):
        raise ValueError("prior or excitation outside the example range")
    columns = [[50., 50. * excitation, 1 / prior_std],
               [50., -50. * excitation, 0.]]
    target = [5., 15. * excitation, knee_prior / prior_std]
    q0_norm = math.sqrt(dot(columns[0], columns[0]))
    q0 = [v / q0_norm for v in columns[0]]
    r01 = dot(q0, columns[1])
    v1 = [v - r01 * u for v, u in zip(columns[1], q0)]
    correction = dot(q0, v1)
    r01 += correction
    v1 = [v - correction * u for v, u in zip(v1, q0)]
    r11 = math.sqrt(dot(v1, v1))
    q1 = [v / r11 for v in v1]
    ankle = dot(q1, target) / r11
    knee = (dot(q0, target) - r01 * ankle) / q0_norm
    # Covariance R^-1 R^-T, not the browser's closed-form Hessian inverse.
    inverse_r = [[1 / q0_norm, -r01 / (q0_norm * r11)], [0., 1 / r11]]
    covariance = mm(inverse_r, transpose(inverse_r))
    return knee, ankle, covariance


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def profile_derivative(g, nuisance_columns):
    """Project g off an independent nuisance basis using reorthogonalized QR.

    Equal weights only. Real calibration must whiten residuals/Jacobians first.
    """
    if not g or not all(math.isfinite(x) for x in g):
        raise ValueError("finite, nonempty derivative required")
    basis = []
    for column in nuisance_columns:
        if len(column) != len(g) or not all(math.isfinite(x) for x in column):
            raise ValueError("nuisance shape or value invalid")
        q = list(column)
        for _ in range(2):
            for unit in basis:
                coefficient = dot(q, unit)
                q = [x - coefficient * y for x, y in zip(q, unit)]
        norm = math.sqrt(dot(q, q))
        if norm < 1e-12:
            raise ValueError("dependent nuisance columns")
        basis.append([x / norm for x in q])
    result = list(g)
    for unit in basis:
        coefficient = dot(g, unit)
        result = [x - coefficient * y for x, y in zip(result, unit)]
    return result


def mm(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def transpose(a):
    return [list(x) for x in zip(*a)]


def rz(radians):
    c, s = math.cos(radians), math.sin(radians)
    return [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]


def ry(radians):
    c, s = math.cos(radians), math.sin(radians)
    return [[c, 0., s], [0., 1., 0.], [-s, 0., c]]


def rx(radians):
    c, s = math.cos(radians), math.sin(radians)
    return [[1., 0., 0.], [0., c, -s], [0., s, c]]


def mv(a, x):
    return [dot(row, x) for row in a]


def essential(ri, rj, ti, tj):
    """Camera-to-body rotations and fixed body-frame camera centers."""
    baseline = mv(transpose(rj), [x - y for x, y in zip(ti, tj)])
    x, y, z = baseline
    skew = [[0., -z, y], [z, 0., -x], [-y, x, 0.]]
    return mm(skew, mm(transpose(rj), ri))


def rotation_distance(a, b):
    delta = mm(transpose(a), b)
    cosine = (sum(delta[i][i] for i in range(3)) - 1) / 2
    return math.degrees(math.acos(max(-1., min(1., cosine))))


def check():
    from decimal import Decimal, localcontext
    # Independent high-precision arithmetic on the printed columns.
    with localcontext() as context:
        context.prec = 50
        squared = sum(Decimal(str(row[4]))**2 for row in ROWS)
        expected = float((squared / Decimal(len(ROWS))).sqrt())
    assert abs(rms(row[4] for row in ROWS) - expected) < 1e-15
    for name, base, injection, estimate, delta, sigma in ROWS:
        assert abs(estimate - base - injection - delta) <= .00101, name
        assert sigma > 0
    assert round(expected, 3) == .063
    assert [row[0] for row in ROWS if abs(row[4]) > .1] == ["left hip yaw", "right hip yaw"]
    assert .027 < rms(row[4] for row in ROWS if "yaw" not in row[0]) < .029
    assert max(abs(row[4]) / row[5] for row in ROWS) > 6
    # Data alone have a null direction (1,-1); a prior does not change that.
    for prior in (-.5, 0., .7):
        k, a, h, cov = knee_ankle_fit(.4, prior)
        assert abs(k + a - .4) < 1e-15 and k == prior
        assert all(abs(x - y) < 1e-12 for x, y in zip(mm(h, cov)[0], [1., 0.]))
        assert all(abs(x - y) < 1e-12 for x, y in zip(mm(h, cov)[1], [0., 1.]))
    for prior in (-.5, .2, .4, .5):
        for std in (.005, .02, .3):
            k, a, cov = excited_joint_fit(prior, std, 0.)
            old_k, old_a, _, old_cov = knee_ankle_fit(.1, prior, 2500., 1 / std**2)
            assert abs(k-old_k) < 2e-14 and abs(a-old_a) < 2e-14
            assert max(abs(x-y) for row, other in zip(cov, old_cov) for x,y in zip(row,other)) < 1e-14
            k, a, cov = excited_joint_fit(prior, std, 1.)
            expected_k = (.2 * 5000 + prior / std**2) / (5000 + 1 / std**2)
            assert abs(k-expected_k) < 1e-14 and abs(a+.1) < 1e-14
    k, a, cov = excited_joint_fit(.4, .02, 1.)
    assert abs(k - 4/15) < 1e-14 and abs(a + .1) < 1e-14
    g = [-1., 0., 1.]
    removed = profile_derivative(g, [[1., 1., 1.], [-1., 0., 1.]])
    assert dot(removed, removed) < 1e-28
    # Independent baseline factor has no scale/translation nuisance column.
    extended = profile_derivative(g + [.2], [[1., 1., 1., 0.], [-1., 0., 1., 0.]])
    assert abs(dot(extended, extended) - .04) < 1e-14
    for values in ([1., 2., 4.], [-2., 3., 1.]):
        residual = profile_derivative(values, [[1., 1., 1.]])
        mean = sum(values) / len(values)
        assert max(abs(x - (y - mean)) for x, y in zip(residual, values)) < 1e-14
    rear, camera = rz(.2), mm(rz(.5), ry(.3))
    common = mm(rz(.7), ry(-.4))
    before = rotation_distance(rear, camera)
    after = rotation_distance(mm(common, rear), mm(common, camera))
    assert abs(before - after) < 1e-12
    # Rig-relative angle invariant, yet every global orientation has changed.
    assert rotation_distance(rear, mm(common, rear)) > 40
    # A planar yaw hand-eye pair commutes with any additional yaw.
    for alpha in (-2., -.4, .7):
        a, x = rz(.3), rz(alpha)
        assert max(abs(v - w) for row, other in zip(mm(a, x), mm(x, a)) for v, w in zip(row, other)) < 1e-14
    # Independent 3D scene construction checks the E_ij frame convention.
    ri, rj = mm(rz(.1), ry(.2)), mm(rz(-.2), ry(-.1))
    ti, tj = [-.1, 0., 0.], [.1, 0., 0.]
    e = essential(ri, rj, ti, tj)
    changed = essential(mm(rz(.5), ri), mm(rz(.5), rj), ti, tj)
    mismatch = []
    for point in ([.3, .5, 2.], [-.4, -.3, 3.], [.8, -.6, 2.5]):
        ui = mv(transpose(ri), [x-y for x, y in zip(point, ti)])
        uj = mv(transpose(rj), [x-y for x, y in zip(point, tj)])
        assert abs(dot(uj, mv(e, ui))) < 1e-14
        mismatch.append(abs(dot(uj, mv(changed, ui))))
    assert sum(mismatch) > .01
    # A single x-directed baseline cannot see a common rotation about x.
    invisible = essential(mm(rx(.5), ri), mm(rx(.5), rj), ti, tj)
    assert max(abs(a-b) for row, other in zip(e, invisible) for a, b in zip(row, other)) < 1e-14
    print("PASS: Table II, prior/data distinction, profiled yaw, gauge, and fixed-baseline geometry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    print(f"Printed Table II residual RMS: {rms(row[4] for row in ROWS):.6f} deg")
    for row in ROWS:
        if "yaw" in row[0]:
            print(f"{row[0]}: delta={row[4]:+.3f} deg, sigma={row[5]:.4f} deg, |delta|/sigma={abs(row[4])/row[5]:.3f}")
    for prior in (-.5, 0., .7):
        knee, ankle, _, _ = knee_ankle_fit(.4, prior)
        print(f"same sum=0.4, knee prior={prior:+.1f}: knee={knee:+.1f}, ankle={ankle:+.1f}")


if __name__ == "__main__":
    main()
