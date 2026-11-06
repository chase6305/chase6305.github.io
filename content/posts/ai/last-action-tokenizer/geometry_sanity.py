#!/usr/bin/env python3
"""Small geometry checks accompanying the LAST article; Python standard library.

This is not the LAST tokenizer or a VLA implementation. Coordinates use [v,w]
with translation in a fixed illustrative unit and rotation in radians.
"""
import argparse
import math


def transpose(a):
    return [list(x) for x in zip(*a)]


def mm(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def mv(a, x):
    return [sum(u * v for u, v in zip(row, x)) for row in a]


def eye(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def skew(w):
    x, y, z = w
    return [[0., -z, y], [z, 0., -x], [-y, x, 0.]]


def combine(*terms):
    n, m = len(terms[0][1]), len(terms[0][1][0])
    return [[sum(c * a[i][j] for c, a in terms) for j in range(m)] for i in range(n)]


def so3_exp(w):
    theta = math.hypot(*w)
    k = skew(w)
    if theta < 1e-6:
        a = 1 - theta**2 / 6 + theta**4 / 120
        b = .5 - theta**2 / 24 + theta**4 / 720
    else:
        a, b = math.sin(theta) / theta, (1 - math.cos(theta)) / theta**2
    return combine((1, eye(3)), (a, k), (b, mm(k, k)))


def so3_log(r):
    cosine = max(-1., min(1., (sum(r[i][i] for i in range(3)) - 1) / 2))
    theta = math.acos(cosine)
    # This example deliberately rejects the branch boundary instead of choosing
    # an arbitrary axis there. A production implementation needs a chart policy.
    if math.pi - theta < 1e-6:
        raise ValueError("rotation is too close to the pi branch boundary")
    factor = .5 + theta**2 / 12 if theta < 1e-6 else theta / (2 * math.sin(theta))
    return [factor * (r[2][1] - r[1][2]), factor * (r[0][2] - r[2][0]),
            factor * (r[1][0] - r[0][1])]


def se3_exp(xi):
    v, w = xi[:3], xi[3:]
    theta = math.hypot(*w)
    k = skew(w)
    if theta < 1e-6:
        b, c = .5 - theta**2 / 24, 1 / 6 - theta**2 / 120
    else:
        b, c = (1 - math.cos(theta)) / theta**2, (theta - math.sin(theta)) / theta**3
    v_matrix = combine((1, eye(3)), (b, k), (c, mm(k, k)))
    return so3_exp(w), mv(v_matrix, v)


def se3_log(r, t):
    w = so3_log(r)
    theta = math.hypot(*w)
    k = skew(w)
    a = 1 / 12 + theta**2 / 720 if theta < 1e-5 else (1 - theta / (2 * math.tan(theta / 2))) / theta**2
    inverse_v = combine((1, eye(3)), (-.5, k), (a, mm(k, k)))
    return mv(inverse_v, t) + w


def pose_compose(a, b):
    ra, ta = a
    rb, tb = b
    return mm(ra, rb), [x+y for x, y in zip(ta, mv(ra, tb))]


def pose_inverse(pose):
    r, t = pose
    inverse_r = transpose(r)
    return inverse_r, [-x for x in mv(inverse_r, t)]


def relative_pose(start, pose):
    return pose_compose(pose_inverse(start), pose)


def adjoint(pose, xi):
    """SE(3) adjoint for the [translation, rotation] coordinate convention."""
    r, t = pose
    rotation = mv(r, xi[3:])
    translation = [x+y for x, y in zip(mv(r, xi[:3]), mv(skew(t), rotation))]
    return translation+rotation


def whitening_2d(covariance):
    """Symmetric inverse square root for a positive definite 2x2 covariance."""
    a, b = covariance[0]
    c, d = covariance[1]
    if not all(math.isfinite(x) for x in (a, b, c, d)) or not math.isclose(b, c, abs_tol=1e-12):
        raise ValueError("expected a finite symmetric covariance")
    gap = math.hypot(a - d, 2 * b)
    hi, lo = (a + d + gap) / 2, (a + d - gap) / 2
    if lo <= 0:
        raise ValueError("covariance must be positive definite; regularize first")
    angle = .5 * math.atan2(2 * b, a - d)
    q = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    return mm(mm(q, [[1 / math.sqrt(hi), 0], [0, 1 / math.sqrt(lo)]]), transpose(q))


def whitening_choice(ratio=3., angle_degrees=0., epsilon=0.):
    """Fixed dimensionless 2D residual and decoded code candidates; no learning."""
    if not all(math.isfinite(x) for x in (ratio, angle_degrees, epsilon)) or not (1 <= ratio <= 10 and 0 <= angle_degrees <= 90 and 0 <= epsilon <= 10):
        raise ValueError("ratio 1..10, angle 0..90 degrees, epsilon 0..10 required")
    angle = math.radians(angle_degrees)
    c, s = math.cos(angle), math.sin(angle)
    q = [[c, -s], [s, c]]
    covariance = mm(mm(q, [[ratio**2+epsilon, 0.], [0., 1.+epsilon]]), transpose(q))
    w = whitening_2d(covariance)
    residual, candidates = [2., .2], [[0., .2], [2., 1.2]]
    differences = [[x-y for x, y in zip(residual, candidate)] for candidate in candidates]
    raw = [math.hypot(*d) for d in differences]
    metric = [math.hypot(*mv(w, d)) for d in differences]

    def winner(distances):
        tolerance = 64*math.ulp(1.)*max(1., *(d*d for d in distances))
        if abs(distances[0]**2-distances[1]**2) <= tolerance:
            return 'tie'
        return 'A' if distances[0] < distances[1] else 'B'

    return {'euclideanA':raw[0], 'euclideanB':raw[1], 'metricA':metric[0], 'metricB':metric[1],
            'euclideanWinner':winner(raw), 'metricWinner':winner(metric),
            'condition':(ratio**2+epsilon)/(1+epsilon), 'covariance':covariance,
            'whitener':w, 'residual':residual, 'candidates':candidates}


def clamped_knots(controls, degree):
    if controls <= degree or degree < 0:
        raise ValueError("controls must exceed nonnegative spline degree")
    return [0.] * (degree + 1) + [i / (controls - degree) for i in range(1, controls - degree)] + [1.] * (degree + 1)


def spline_basis(u, controls=16, degree=4):
    if not 0 <= u <= 1:
        raise ValueError("normalized time must be in [0,1]")
    knots = clamped_knots(controls, degree)
    if u == 1:
        return [0.] * (controls - 1) + [1.]
    values = [float(knots[i] <= u < knots[i + 1]) for i in range(len(knots) - 1)]
    for p in range(1, degree + 1):
        next_values = []
        for i in range(len(values) - 1):
            left, right = knots[i + p] - knots[i], knots[i + p + 1] - knots[i + 1]
            next_values.append(((u - knots[i]) / left * values[i] if left else 0.) +
                               ((knots[i + p + 1] - u) / right * values[i + 1] if right else 0.))
        values = next_values
    return values[:controls]


def solve(a, b):
    """Pivoted elimination, used only for the small ridge example."""
    n = len(a)
    rows = [list(row) + [value] for row, value in zip(a, b)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(rows[i][k]))
        rows[k], rows[pivot] = rows[pivot], rows[k]
        if abs(rows[k][k]) < 1e-14:
            raise ValueError("singular system")
        scale = rows[k][k]
        rows[k] = [x / scale for x in rows[k]]
        for i in range(n):
            if i != k:
                factor = rows[i][k]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[k])]
    return [row[-1] for row in rows]


def fit_spline(samples, controls=16, degree=4, ridge=.01):
    if len(samples) < 2 or ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("need at least two samples and finite positive ridge")
    phi = [spline_basis(i / (len(samples) - 1), controls, degree) for i in range(len(samples))]
    gram = mm(transpose(phi), phi)
    for i in range(controls):
        gram[i][i] += ridge
    return solve(gram, mv(transpose(phi), samples))


def covariance(points):
    n = len(points)
    mean = [sum(p[i] for p in points) / n for i in range(len(points[0]))]
    centered = [[x - m for x, m in zip(p, mean)] for p in points]
    return [[sum(p[i] * p[j] for p in centered) / n for j in range(len(mean))] for i in range(len(mean))]


def check():
    import random
    rng = random.Random(260611221)
    for _ in range(100):
        xi = [rng.uniform(-.5, .5) for _ in range(6)]
        restored = se3_log(*se3_exp(xi))
        assert max(abs(a - b) for a, b in zip(xi, restored)) < 2e-12
    for _ in range(60):
        start = se3_exp([rng.uniform(-.4, .4) for _ in range(6)])
        motion = se3_exp([rng.uniform(-.2, .2) for _ in range(6)])
        finish = pose_compose(start, motion)
        world = se3_exp([rng.uniform(-.5, .5) for _ in range(6)])
        original = se3_log(*relative_pose(start, finish))
        changed_world = se3_log(*relative_pose(pose_compose(world,start),pose_compose(world,finish)))
        assert max(abs(x-y) for x, y in zip(original,changed_world)) < 2e-12
        tool = se3_exp([rng.uniform(-.3, .3) for _ in range(6)])
        changed_tool = se3_log(*relative_pose(pose_compose(start,tool),pose_compose(finish,tool)))
        expected = adjoint(pose_inverse(tool), original)
        assert max(abs(x-y) for x, y in zip(changed_tool,expected)) < 2e-12
    pose = (so3_exp([0,0,math.pi/2]), [.3,.2,0.])
    delta = (eye(3), [.01,0.,0.])
    body_step, world_step = pose_compose(pose,delta), pose_compose(delta,pose)
    assert abs(body_step[1][1]-.21) < 1e-14 and abs(body_step[1][0]-.3) < 1e-14
    assert abs(world_step[1][0]-.31) < 1e-14 and abs(world_step[1][1]-.2) < 1e-14
    xi = se3_log(so3_exp([0, 0, math.pi / 2]), [1, 0, 0])
    assert abs(xi[0] - math.pi / 4) < 1e-12 and abs(xi[1] + math.pi / 4) < 1e-12
    for angle in (.2, .1, .05, .025):
        w = so3_log(mm(so3_exp([angle, 0, 0]), so3_exp([0, angle, 0])))
        assert abs(w[2] / angle**2 - .5) < .002
    for _ in range(100):
        b = rng.uniform(-2, 2)
        cov = [[4., b], [b, 2.]]
        w = whitening_2d(cov)
        unit = mm(mm(w, cov), transpose(w))
        assert max(abs(unit[i][j] - (i == j)) for i in range(2) for j in range(2)) < 2e-12
        x = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
        det = cov[0][0] * cov[1][1] - b * b
        direct = (cov[1][1] * x[0]**2 - 2 * b * x[0] * x[1] + cov[0][0] * x[1]**2) / det
        assert abs(sum(v * v for v in mv(w, x)) - direct) < 2e-12
    # Literal single-sample covariance has different behavior from a running
    # chart covariance: the whitened norm approaches one as amplitude grows.
    for amplitude in (.1, 1., 2., 10.):
        epsilon = 1e-4
        w = whitening_2d([[amplitude**2+epsilon, 0.], [0., epsilon]])
        transformed = mv(w, [amplitude, 0.])
        assert abs(transformed[0]-amplitude/math.sqrt(amplitude**2+epsilon)) < 1e-12
    for controls, degree in ((16, 4), (16, 0), (6, 2)):
        for i in range(101):
            row = spline_basis(i / 100, controls, degree)
            assert min(row) >= 0 and abs(sum(row) - 1) < 1e-12
    for horizon in (8, 16, 30):
        a = [math.sin(i / (horizon - 1)) for i in range(horizon)]
        b = [math.cos(2 * i / (horizon - 1)) for i in range(horizon)]
        ca, cb = fit_spline(a), fit_spline(b)
        combined = fit_spline([2 * x - 3 * y for x, y in zip(a, b)])
        assert max(abs(z - (2 * x - 3 * y)) for x, y, z in zip(ca, cb, combined)) < 2e-12
    # Same zero mean and covariance, but an arbitrary pairing is not an isometry.
    points = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
    paired = [points[0], points[2], points[1], points[3]]
    assert covariance(points) == covariance(paired)
    assert math.dist(points[0], points[1]) != math.dist(paired[0], paired[1])
    for bad in ([[1., 1.], [1., 1.]], [[1., 2.], [0., 1.]]):
        try:
            whitening_2d(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid covariance accepted")
    # Exact decomposition of the paper's literal sample-dependent Eq. 11.
    # The codebook spread is not an across-trajectory residual covariance.
    for _ in range(120):
        codes = [[rng.uniform(-2,2) for _ in range(3)] for _ in range(5)]
        weights = [rng.uniform(.1,1) for _ in codes]
        weights = [w/sum(weights) for w in weights]
        z = [rng.uniform(-3,3) for _ in range(3)]
        center = [sum(w*c[i] for w,c in zip(weights,codes)) for i in range(3)]
        residual = [x-y for x,y in zip(z,center)]
        epsilon = .03
        spread = [[sum(w*(c[i]-center[i])*(c[j]-center[j]) for w,c in zip(weights,codes)) for j in range(3)] for i in range(3)]
        a = [[spread[i][j]+(epsilon if i==j else 0.) for j in range(3)] for i in range(3)]
        sigma = [[sum(w*(z[i]-c[i])*(z[j]-c[j]) for w,c in zip(weights,codes))+(epsilon if i==j else 0.) for j in range(3)] for i in range(3)]
        decomposed = [[a[i][j]+residual[i]*residual[j] for j in range(3)] for i in range(3)]
        assert max(abs(x-y) for row,other in zip(sigma,decomposed) for x,y in zip(row,other)) < 2e-14
        quadratic = lambda matrix: sum(x*y for x,y in zip(residual,solve(matrix,residual)))
        t = quadratic(a)
        assert abs(quadratic(sigma)-t/(1+t)) < 3e-13
        assert 0 <= quadratic(sigma) < 1
    assert whitening_choice()['euclideanWinner'] == 'B'
    assert whitening_choice()['metricWinner'] == 'A'
    assert abs(whitening_choice()['metricA']-2/3) < 1e-14
    assert whitening_choice(1)['metricWinner'] == 'B'
    assert whitening_choice(3,90)['metricWinner'] == 'B'
    assert whitening_choice(2)['metricWinner'] == 'tie'
    assert whitening_choice(3,0,3)['metricWinner'] == 'B'
    print("PASS: SE(3) frames/BCH, whitening choices, spline linearity, and covariance counterexample")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        for angle in (.2, .1, .05):
            w = so3_log(mm(so3_exp([angle, 0, 0]), so3_exp([0, angle, 0])))
            print(f"angle={angle:.3f} rad, z commutator={w[2]:.8f}, leading={angle**2 / 2:.8f}")
        print("SE(3) log of Rz(90deg), t=(1,0,0):", [round(x, 6) for x in se3_log(so3_exp([0, 0, math.pi / 2]), [1, 0, 0])])


if __name__ == "__main__":
    main()
