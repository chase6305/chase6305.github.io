#!/usr/bin/env python3
"""Small contact-likelihood and observability examples; standard library.

The 2D translation example assumes known plane associations and uses mm.
The Jacobian example fixes point-plane correspondences and uses an SE(3)
left perturbation. Neither implements the paper's mesh-based 6D filter.
"""
import argparse
from fractions import Fraction
import math
import random


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def normalized_log_weights(distances, contacts, sigma=.005, penetration=.002, epsilon=1e-6):
    """Consistent standard-deviation Gaussian product, up to a shared constant.

    Rows are particles; columns are the same observed events for each row.
    An empty observation sequence preserves an equal-weight input population.
    sigma is explicitly a standard deviation. The literal v1 Eq. 7 uses
    2*sigma_P in its exponent and is covered by literal_v1_log_weights below.
    """
    sigma, penetration, epsilon = [finite(v, "likelihood parameter") for v in (sigma, penetration, epsilon)]
    if sigma <= 0 or penetration < 0 or not 0 < epsilon <= 1:
        raise ValueError("invalid likelihood scale, tolerance or rejection factor")
    contacts = tuple(contacts)
    if any(c not in (0, 1) for c in contacts):
        raise ValueError("contacts must be binary")
    rows = [tuple(finite(d, "signed distance") for d in row) for row in distances]
    if not rows or any(len(row) != len(contacts) for row in rows):
        raise ValueError("expected a nonempty particles by events distance matrix")
    logs = []
    for row in rows:
        score = 0.
        for d, contact in zip(row, contacts):
            if contact:
                ratio = d/sigma
                score -= ratio*ratio/2
            elif d < -penetration:
                score += math.log(epsilon)
        logs.append(score)
    maximum = max(logs)
    if not math.isfinite(maximum):
        raise ValueError("all particles have non-finite log weights; inspect model or particle coverage")
    unnormalized = [math.exp(v-maximum) for v in logs]
    total = math.fsum(unnormalized)
    return [w/total for w in unnormalized]


def literal_v1_log_weights(distances, contacts, parameter=.005, penetration=.002, epsilon=1e-6):
    """Normalized Eq. 7 as printed: exponent -d*d/(2*sigma_P).

    The inconsistent printed prefactor is shared across particles for the
    same observations, so it cancels. Only normalized weights are compared.
    """
    parameter = finite(parameter, "v1 kernel parameter")
    if parameter <= 0:
        raise ValueError("v1 kernel parameter must be positive")
    return normalized_log_weights(distances, contacts, math.sqrt(parameter), penetration, epsilon)


def kernel_parameter_comparison(distance=.01, parameter=.005):
    distance, parameter = finite(distance, "distance"), finite(parameter, "parameter")
    if parameter <= 0:
        raise ValueError("kernel parameter must be positive")
    return {'literal_factor':math.exp(-distance*distance/(2*parameter)),
            'standard_deviation_factor':math.exp(-distance*distance/(2*parameter*parameter)),
            'literal_weights':literal_v1_log_weights([[0.], [distance]], [1], parameter),
            'standard_deviation_weights':normalized_log_weights([[0.], [distance]], [1], parameter)}


def effective_sample_size(weights):
    weights = tuple(finite(w, "weight") for w in weights)
    if not weights or min(weights) < 0 or not math.isclose(math.fsum(weights), 1., abs_tol=1e-12):
        raise ValueError("expected nonnegative normalized weights")
    return 1/math.fsum(w*w for w in weights)


def observation_consistent(distances, contacts, threshold=.002):
    """Literal one-sided branches of v1 Eq. 12, not a collision guarantee.

    A nonempty observed batch is required by this small helper. Deep predicted
    contact penetration is not rejected by this stopping predicate itself.
    """
    distances = tuple(finite(d, "distance") for d in distances)
    contacts = tuple(contacts)
    threshold = finite(threshold, "consistency threshold")
    if not distances or len(distances) != len(contacts) or threshold <= 0 or any(c not in (0, 1) for c in contacts):
        raise ValueError("expected nonempty matched events and a positive threshold")
    return not any(d > threshold if c else d <= -threshold for d, c in zip(distances, contacts))


def diffusion_scale(previous=1., confident=False, stable=False, consistent=False, alpha=2., beta=.5):
    """Eq. 13 branch logic; alpha/beta defaults are constructed examples."""
    previous, alpha, beta = [finite(v, "diffusion scale") for v in (previous, alpha, beta)]
    if previous <= 0 or alpha <= 1 or not 0 < beta < 1 or any(not isinstance(c, bool) for c in (confident, stable, consistent)):
        raise ValueError("invalid diffusion scale or criteria flags")
    result = previous*(alpha if confident and not consistent else beta if stable and consistent else 1.)
    return finite(result, "updated diffusion scale")


def five_round_stop(history):
    """Each record already contains (confidence, stability, consistency).

    The stability criterion's own historical window is computed by the caller;
    this only checks the paper's five-consecutive-round outer stop condition.
    """
    rows = [tuple(record) for record in history]
    if any(len(row) != 3 or any(not isinstance(v, bool) for v in row) for row in rows):
        raise ValueError("expected three boolean criteria per round")
    return len(rows) >= 5 and all(all(row) for row in rows[-5:])


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def transform_point(transform, point):
    """Affine homogeneous point transform; callers supply rigid matrices."""
    rows = [tuple(finite(v, "transform") for v in row) for row in transform]
    point = tuple(finite(v, "point") for v in point)
    if len(rows) != 4 or any(len(row) != 4 for row in rows) or len(point) != 3 or rows[3] != (0., 0., 0., 1.):
        raise ValueError("expected a homogeneous 4x4 transform and 3D point")
    return tuple(math.fsum(rows[i][j]*point[j] for j in range(3))+rows[i][3] for i in range(3))


def sphere_sdf(point, center=(0., 0., 0.), radius=1.):
    point, center = [tuple(finite(v, "sphere coordinate") for v in p) for p in (point, center)]
    radius = finite(radius, "radius")
    if len(point) != 3 or len(center) != 3 or radius <= 0:
        raise ValueError("invalid sphere geometry")
    return math.hypot(*(a-b for a, b in zip(point, center)))-radius


def transformed_cloud_distance(base, joint, points, center=(0., 0., 0.), radius=1.):
    """min SDF(X * Gamma * p), using an analytic sphere rather than a voxel grid."""
    distances = [sphere_sdf(transform_point(base, transform_point(joint, p)), center, radius) for p in points]
    if not distances:
        raise ValueError("expected a nonempty end-effector cloud")
    return min(distances)


def sampled_circle_distance(samples=8, radius_mm=40., center_height_mm=39.):
    """A circle above y=0; phase puts its bottom halfway between samples."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
        raise ValueError("at least three circle samples are required")
    radius_mm, center_height_mm = finite(radius_mm, "radius"), finite(center_height_mm, "height")
    if radius_mm <= 0:
        raise ValueError("radius must be positive")
    points = [(radius_mm*math.cos(-math.pi/2+(i+.5)*math.tau/samples),
               center_height_mm+radius_mm*math.sin(-math.pi/2+(i+.5)*math.tau/samples))
              for i in range(samples)]
    exact = center_height_mm-radius_mm
    sampled = min(y for _, y in points)
    coverage = 2*radius_mm*math.sin(math.pi/(2*samples))
    return {'exact':exact, 'sampled':sampled, 'coverage':coverage, 'points':points}


def point_plane_row(point, normal):
    if len(point) != 3 or len(normal) != 3:
        raise ValueError("expected 3D point and normal")
    point, normal = [tuple(finite(v, "geometry") for v in x) for x in (point, normal)]
    if not math.isclose(math.hypot(*normal), 1., abs_tol=1e-12):
        raise ValueError("normal must be unit length")
    return normal+cross(point, normal)


def exact_rank(rows):
    """Exact rational row reduction, useful for the integer-row fixtures."""
    a = [[Fraction(x) for x in row] for row in rows]
    if not a:
        return 0
    if any(len(row) != len(a[0]) for row in a):
        raise ValueError("ragged matrix")
    rank = 0
    for j in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][j]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        divisor = a[rank][j]
        a[rank] = [v/divisor for v in a[rank]]
        for i in range(len(a)):
            if i != rank:
                multiplier = a[i][j]
                a[i] = [v-multiplier*w for v, w in zip(a[i], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


def geometry_rows():
    horizontal = [point_plane_row(p, (0., 0., 1.)) for p in ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))]
    diverse = horizontal+[point_plane_row((0., 0., 0.), (1., 0., 0.)),
                          point_plane_row((0., 0., 0.), (0., 1., 0.)),
                          point_plane_row((0., 1., 0.), (1., 0., 0.))]
    return horizontal, diverse


def translation_example(sigma_mm=.5, second_plane=False):
    """Uniform grid prior, t_true=(6,-4) mm; known separate plane labels."""
    sigma_mm = finite(sigma_mm, "distance scale")
    if sigma_mm <= 0:
        raise ValueError("distance scale must be positive")
    points = [(float(x), float(y)) for x in range(-20, 21) for y in range(-20, 21)]
    # A measured base-frame point (0,4) touches y=0; a second point
    # (-6,10) touches x=0. The plane association is assumed known here.
    distances = [[y+4, x-6] if second_plane else [y+4] for x, y in points]
    weights = normalized_log_weights(distances, [1, 1] if second_plane else [1], sigma=sigma_mm)
    mean = [math.fsum(p[i]*w for p, w in zip(points, weights)) for i in range(2)]
    variance = [math.fsum(w*(p[i]-mean[i])**2 for p, w in zip(points, weights)) for i in range(2)]
    return {'mean_mm':mean, 'std_mm':[math.sqrt(v) for v in variance],
            'effective_samples':effective_sample_size(weights), 'particles':len(points)}


def posterior_example(sigma_mm=2., angle_degrees=90., second_plane=False):
    """Finite-grid oracle for the inline two-translation experiment.

    The second residual is cos(theta)*(ty+4)+sin(theta)*(tx-6).
    Both residuals, when enabled, have independent equal Gaussian scales.
    """
    sigma_mm, angle_degrees = finite(sigma_mm, "distance scale"), finite(angle_degrees, "normal angle")
    if not 1 <= sigma_mm <= 8 or not 0 <= angle_degrees <= 90 or not isinstance(second_plane, bool):
        raise ValueError("inline posterior settings are out of range")
    angle = math.radians(angle_degrees)
    c, s = (1., 0.) if angle_degrees == 0 else (0., 1.) if angle_degrees == 90 else (math.cos(angle), math.sin(angle))
    axis = list(range(-20, 21))
    points = [(x, y) for x in axis for y in axis]
    distances = [[y+4, c*(y+4)+s*(x-6)] if second_plane else [y+4] for x, y in points]
    weights = normalized_log_weights(distances, [1, 1] if second_plane else [1], sigma=sigma_mm)
    mean = [math.fsum(p[i]*w for p, w in zip(points, weights)) for i in range(2)]
    variance = [math.fsum(w*(p[i]-mean[i])**2 for p, w in zip(points, weights)) for i in range(2)]
    covariance = math.fsum(w*(p[0]-mean[0])*(p[1]-mean[1]) for p, w in zip(points, weights))
    marginals = [[math.fsum(w for p, w in zip(points, weights) if p[i] == value) for value in axis] for i in range(2)]
    return {'meanX':mean[0], 'meanY':mean[1], 'stdX':math.sqrt(variance[0]), 'stdY':math.sqrt(variance[1]),
            'correlation':covariance/math.sqrt(variance[0]*variance[1]),
            'effectiveSamples':effective_sample_size(weights), 'axis':axis, 'marginals':marginals}


def unbounded_covariance(sigma=2., angle_degrees=90.):
    """Two known, independent plane residuals on an unbounded translation plane."""
    sigma, angle_degrees = finite(sigma, "standard deviation"), finite(angle_degrees, "angle")
    if sigma <= 0 or not 0 <= angle_degrees <= 90:
        raise ValueError("invalid scale or normal angle")
    if angle_degrees == 0:
        return None  # An improper flat x direction cannot be normalized here.
    s, c = math.sin(math.radians(angle_degrees)), math.cos(math.radians(angle_degrees))
    if angle_degrees == 90:
        s, c = 1., 0.
    denominator = s*s
    if denominator == 0:
        raise ValueError("angle is below the floating-point range")
    return [[sigma*sigma*(1+c*c)/denominator, -sigma*sigma*c/s],
            [-sigma*sigma*c/s, sigma*sigma]]


def correlated_contact_scale(count=25, correlation=.9, sigma=1.):
    """Scalar flat-prior fusion of equicorrelated equal-noise distances.

    Correlation 1 is the continuous limit, not an invertible covariance.
    This is an observation-noise example, not particle-weight ESS.
    """
    correlation, sigma = finite(correlation, "correlation"), finite(sigma, "standard deviation")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or not 0 <= correlation <= 1 or sigma <= 0:
        raise ValueError("invalid count, correlation or scale")
    return sigma*math.sqrt((1+(count-1)*correlation)/count)


def symmetric_wall_example(half_gap_mm=10., sigma_mm=2.):
    """One point touching either wall of a symmetric 1D gap.

    Solid environment occupies |x| >= half_gap. Its signed distance is
    half_gap - abs(x). Two equally weighted modes touch different walls.
    """
    half_gap_mm, sigma_mm = [finite(v, 'wall example parameter')
                             for v in (half_gap_mm, sigma_mm)]
    if min(half_gap_mm, sigma_mm) <= 0:
        raise ValueError('half-gap and distance standard deviation must be positive')
    hypotheses = [-half_gap_mm, half_gap_mm]
    distances = [half_gap_mm-abs(x) for x in hypotheses]
    weights = normalized_log_weights([[d] for d in distances], [1], sigma_mm)
    mean = math.fsum(x*w for x, w in zip(hypotheses, weights))
    mean_distance = half_gap_mm-abs(mean)
    return {'hypotheses_mm':hypotheses, 'weights':weights, 'mean_mm':mean,
            'mode_distances_mm':distances, 'mean_distance_mm':mean_distance,
            'mean_contact_factor':math.exp(-.5*(mean_distance/sigma_mm)**2)}


def circular_mean(angles, weights=None):
    angles = tuple(finite(a, "angle") for a in angles)
    if not angles:
        raise ValueError("expected at least one angle")
    weights = tuple([1/len(angles)]*len(angles) if weights is None else weights)
    effective_sample_size(weights)  # validates normalization and signs
    if len(weights) != len(angles):
        raise ValueError("angle/weight shape mismatch")
    c = math.fsum(w*math.cos(a) for w, a in zip(weights, angles))
    s = math.fsum(w*math.sin(a) for w, a in zip(weights, angles))
    if math.hypot(c, s) < 1e-12:
        raise ValueError("circular mean is undefined for this symmetric population")
    return math.atan2(s, c)


def normal_sparsity(candidate, history):
    candidate = tuple(finite(v, "normal") for v in candidate)
    history = [tuple(finite(v, "normal") for v in n) for n in history]
    if not history or any(len(n) != 3 or not math.isclose(math.hypot(*n), 1., abs_tol=1e-12) for n in [candidate]+history):
        raise ValueError("expected unit 3D candidate and nonempty normal history")
    return min(math.acos(max(-1., min(1., math.fsum(a*b for a, b in zip(candidate, n))))) for n in history)


def check():
    base = [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    joint = [[0, -1, 0, 0], [1, 0, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert transform_point(base, transform_point(joint, (1, 0, 0))) == (1., 3., 0.)
    # The first point is at the sphere center, the second just outside it.
    # Min absolute distance would select 0.05 and hide the -0.2 penetration.
    signed = transformed_cloud_distance(base, joint, [(1, 0, 0), (1.25, 0, 0)], (1, 3, 0), .2)
    assert signed == -.2
    assert abs(sphere_sdf((1, 3.25, 0), (1, 3, 0), .2)-.05) < 1e-15
    for count in (4, 8, 16, 32, 64):
        circle = sampled_circle_distance(count)
        assert circle['exact'] <= circle['sampled'] <= circle['exact']+circle['coverage']+1e-12
        closed = 39.-40.*math.cos(math.pi/count)
        assert abs(circle['sampled']-closed) < 1e-12
    assert sampled_circle_distance(8)['sampled'] > 2
    assert sampled_circle_distance(32)['sampled'] < 0
    horizontal, diverse = geometry_rows()
    assert exact_rank(horizontal) == 3 and exact_rank(diverse) == 6
    opposite = [point_plane_row((0., 0., 1.), (0., 0., -1.))]
    assert exact_rank(horizontal+opposite) == 3
    assert normal_sparsity((0, 0, -1), [(0, 0, 1)]) > normal_sparsity((1, 0, 0), [(0, 0, 1)])
    rng = random.Random(20261106)
    for _ in range(200):
        p = [rng.uniform(-2, 2) for _ in range(3)]
        n = [rng.uniform(-1, 1) for _ in range(3)]
        norm = math.hypot(*n)
        n = [v/norm for v in n]
        row = point_plane_row(p, n)
        # Independent axis rotations give finite differences for the three
        # rotational columns of a left-perturbed world-frame point.
        h = 1e-6
        for axis in range(3):
            def distance(angle):
                out = p.copy()
                i, j = (axis+1)%3, (axis+2)%3
                out[i] = math.cos(angle)*p[i]-math.sin(angle)*p[j]
                out[j] = math.sin(angle)*p[i]+math.cos(angle)*p[j]
                return math.fsum(a*b for a, b in zip(n, out))
            derivative = (distance(h)-distance(-h))/(2*h)
            assert abs(derivative-row[axis+3]) < 5e-10
    # Direct multiplication is an independent oracle on moderate likelihoods.
    for _ in range(150):
        contacts = [rng.randrange(2) for _ in range(6)]
        rows = [[rng.uniform(-.004, .004) for _ in contacts] for _ in range(10)]
        actual = normalized_log_weights(rows, contacts, sigma=.003, epsilon=.01)
        direct = [math.prod(math.exp(-d*d/(2*.003**2)) if c else .01 if d < -.002 else 1.
                            for d, c in zip(row, contacts)) for row in rows]
        expected = [v/math.fsum(direct) for v in direct]
        assert max(abs(a-b) for a, b in zip(actual, expected)) < 1e-14
    # Repeating 2000 contacts underflows naive products, but not normalization.
    extreme = normalized_log_weights([[.003]*2000, [.0031]*2000], [1]*2000, sigma=.001)
    assert all(math.isfinite(w) for w in extreme) and abs(sum(extreme)-1) < 1e-14
    assert normalized_log_weights([[], []], []) == [.5, .5]
    boundary = normalized_log_weights([[-.002], [math.nextafter(-.002, -math.inf)]], [0], epsilon=.01)
    assert abs(boundary[0]-1/1.01) < 1e-15
    assert observation_consistent([.002], [1])
    assert not observation_consistent([math.nextafter(.002, math.inf)], [1])
    assert observation_consistent([-1.], [1])  # One-sided stopping branch only.
    assert not observation_consistent([-.002], [0])
    assert observation_consistent([math.nextafter(-.002, math.inf)], [0])
    assert not observation_consistent([0., -.002], [1, 0])
    criteria_cases = [
        ((False, False, False), 1.), ((False, False, True), 1.),
        ((False, True, False), 1.), ((False, True, True), .5),
        ((True, False, False), 2.), ((True, False, True), 1.),
        ((True, True, False), 2.), ((True, True, True), .5),
    ]
    for flags, expected in criteria_cases:
        assert diffusion_scale(1., *flags) == expected
    true = (True, True, True)
    assert not five_round_stop([true]*4) and five_round_stop([true]*5)
    assert not five_round_stop([true]*5+[(True, False, True)])
    assert five_round_stop([(False, False, False)]+[true]*5)
    for _ in range(200):
        rows = [tuple(bool(rng.randrange(2)) for _ in range(3)) for _ in range(20)]
        rows.extend([true]*rng.randrange(7))
        trailing = 0
        for row in reversed(rows):
            if row != true:
                break
            trailing += 1
        assert five_round_stop(rows) == (trailing >= 5)
    kernels = kernel_parameter_comparison()
    assert abs(kernels['literal_factor']-math.exp(-.01)) < 1e-15
    assert abs(kernels['standard_deviation_factor']-math.exp(-2)) < 1e-15
    assert kernels['literal_weights'][0] < .503 and kernels['standard_deviation_weights'][0] > .88
    for _ in range(100):
        parameter = rng.uniform(.0001, .05)
        contacts = [rng.randrange(2) for _ in range(6)]
        rows = [[rng.uniform(-.02, .02) for _ in contacts] for _ in range(8)]
        actual = literal_v1_log_weights(rows, contacts, parameter, epsilon=.01)
        # Direct printed prefactors differ from the standard-deviation helper,
        # but are common for all particles and cancel after normalization.
        direct = [math.prod(math.exp(-d*d/(2*parameter))/math.sqrt(2*math.pi*parameter**2) if c else
                            .01 if d < -.002 else 1. for d, c in zip(row, contacts)) for row in rows]
        expected = [w/math.fsum(direct) for w in direct]
        assert max(abs(a-b) for a, b in zip(actual, expected)) < 1e-13
    assert effective_sample_size([.25]*4) == 4.
    # Systematic resampling with exact categorical counts; carrying the old
    # source weight again changes the estimator instead of using equal draws.
    draws = [0. if (i+.5)/20 < .9 else 10. for i in range(20)]
    assert math.fsum(draws)/len(draws) == 1.
    duplicated = math.fsum(x*(.9 if x == 0 else .1) for x in draws)/math.fsum(.9 if x == 0 else .1 for x in draws)
    assert abs(duplicated-0.12195121951219512) < 1e-15
    walls = symmetric_wall_example()
    assert walls['weights'] == [.5, .5]
    assert walls['mean_mm'] == 0. and walls['mean_distance_mm'] == 10.
    assert walls['mode_distances_mm'] == [0., 0.]
    assert observation_consistent(walls['mode_distances_mm'], [1, 1], threshold=5.)
    assert not observation_consistent([walls['mean_distance_mm']], [1], threshold=5.)
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 60
        for gap in (1, 3, 10, 20, 40):
            for sigma in (1, 2, 5, 10, 20, 40):
                result = symmetric_wall_example(gap, sigma)
                reference = (-(Decimal(gap)/Decimal(sigma))**2/2).exp()
                assert math.isclose(result['mean_contact_factor'], float(reference),
                                    rel_tol=1e-13, abs_tol=1e-300)
                assert result['mean_distance_mm'] == gap
    print('PASS: symmetric-contact mixture and 30 independent Decimal contact factors')
    floor = translation_example()
    corner = translation_example(second_plane=True)
    assert abs(floor['mean_mm'][0]) < 1e-12 and abs(floor['mean_mm'][1]+4) < 1e-12
    assert floor['std_mm'][0] > 11 and floor['std_mm'][1] < .5
    assert max(abs(a-b) for a, b in zip(corner['mean_mm'], [6, -4])) < 1e-12
    assert max(corner['std_mm']) < .5
    for sigma in (1., 2., 4., 8.):
        # Orthogonal-plane posteriors factorize into two independent 1D sums.
        result = posterior_example(sigma, 90., True)
        for dim, center in enumerate((6., -4.)):
            mass = [math.exp(-(x-center)**2/(2*sigma*sigma)) for x in range(-20, 21)]
            mass = [v/math.fsum(mass) for v in mass]
            assert max(abs(a-b) for a, b in zip(result['marginals'][dim], mass)) < 1e-13
        parallel = posterior_example(sigma, 0., True)
        assert abs(parallel['stdX']-math.sqrt(140)) < 1e-12
    for angle in (.1, 15., 30., 60., 90.):
        cov = unbounded_covariance(2., angle)
        s, c = math.sin(math.radians(angle)), math.cos(math.radians(angle))
        design = [[0., 1.], [s, c]]
        information = [[math.fsum(row[i]*row[j] for row in design)/4 for j in range(2)] for i in range(2)]
        product = [[math.fsum(information[i][k]*cov[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        assert max(abs(product[i][j]-(i == j)) for i in range(2) for j in range(2)) < 1e-12
    assert unbounded_covariance(2., 0.) is None
    for count in (1, 2, 5, 25, 100):
        assert correlated_contact_scale(count, 1., 2.) == 2.
        assert abs(correlated_contact_scale(count, 0., 2.)-2./math.sqrt(count)) < 1e-15
        for rho in (0., .1, .5, .9, 1.):
            # The average's variance follows from counting N diagonal and
            # N*(N-1) off-diagonal covariance entries, independently of fusion.
            average_variance = (count*4.+count*(count-1)*rho*4.)/count**2
            assert abs(correlated_contact_scale(count, rho, 2.)**2-average_variance) < 1e-14
    assert abs(correlated_contact_scale()-math.sqrt(.904)) < 1e-15
    assert abs(abs(circular_mean([math.radians(179), math.radians(-179)]))-math.pi) < 1e-12
    bad = [lambda: normalized_log_weights([[0]], [1], sigma=0),
           lambda: normalized_log_weights([[0]], [1], epsilon=0),
           lambda: normalized_log_weights([[0]], [2]), lambda: normalized_log_weights([], [1]),
           lambda: normalized_log_weights([[math.nan]], [1]), lambda: effective_sample_size([0, 0]),
           lambda: circular_mean([0, math.pi]), lambda: point_plane_row((0, 0, 0), (0, 0, 2)),
           lambda: correlated_contact_scale(True), lambda: correlated_contact_scale(2, -.1),
           lambda: observation_consistent([], []), lambda: diffusion_scale(alpha=1.),
           lambda: five_round_stop([(1, 1, 1)]),
           lambda: symmetric_wall_example(half_gap_mm=0.),
           lambda: symmetric_wall_example(sigma_mm=-1.)]
    for run in bad:
        try:
            run()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    print('PASS: 600 independent rotational derivatives, 150 standard-deviation and 100 printed-kernel '
          'direct-product cases, exact ranks 3/6, 2D translation gauge, posterior covariance, '
          'resampling weights, correlated observations, signed geometry coverage, '
          'consistency thresholds, 8 diffusion branches, 200 stop histories, circular branch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
        return
    for label, second in [('floor only', False), ('floor + wall', True)]:
        result = translation_example(second_plane=second)
        print(f'{label}: mean={result["mean_mm"]}, std={result["std_mm"]} mm, ESS={result["effective_samples"]:.4f}')
    a, b = geometry_rows()
    print('point-plane ranks:', exact_rank(a), exact_rank(b))
    print('mean of +/-179 degrees:', math.degrees(circular_mean([math.radians(179), math.radians(-179)])))
    kernels = kernel_parameter_comparison()
    print('Contact factor at distance=0.01, parameter=0.005:',
          f'printed v1={kernels["literal_factor"]:.8f}, std convention={kernels["standard_deviation_factor"]:.8f}')
    print('Unbounded two-plane covariance at sigma=2 mm, angle=15 degrees:', unbounded_covariance(2, 15))
    print('25 equicorrelated distances, rho=0.9: fused scale / single scale:', correlated_contact_scale())
    walls = symmetric_wall_example()
    print('Two contact modes at +/-10 mm: mean=', walls['mean_mm'],
          '; distance at mean=', walls['mean_distance_mm'],
          '; relative contact factor=', f"{walls['mean_contact_factor']:.8g}")
    print('Literal Eq12, contact distance -1 versus noncontact distance -0.002:',
          observation_consistent([-1.], [1]), observation_consistent([-.002], [0]))


if __name__ == '__main__':
    main()
