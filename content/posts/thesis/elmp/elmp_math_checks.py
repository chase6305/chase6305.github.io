"""Independent ELMP reading exercise; standard library only, not a reproduction."""
import math


def clearance_and_gradient(q):
    """Two-link planar endpoint sphere versus a circular obstacle, in metres."""
    a, b = q
    l1, l2 = 0.7, 0.4
    x = l1 * math.cos(a) + l2 * math.cos(a + b)
    y = l1 * math.sin(a) + l2 * math.sin(a + b)
    dx, dy = x - 0.9, y - 0.35
    distance = math.hypot(dx, dy)
    if distance < 1e-12:
        raise ValueError("Distance gradient is undefined at the obstacle centre")
    # Obstacle radius 0.16 m, endpoint sphere radius 0.04 m.
    clearance = distance - 0.16 - 0.04
    jac = (
        (-l1 * math.sin(a) - l2 * math.sin(a + b), -l2 * math.sin(a + b)),
        (l1 * math.cos(a) + l2 * math.cos(a + b), l2 * math.cos(a + b)),
    )
    grad = tuple((dx * jac[0][j] + dy * jac[1][j]) / distance for j in range(2))
    return clearance, grad


def penalty(q):
    return max(0.0, 0.03 - clearance_and_gradient(q)[0])


def main():
    q = (0.2, 0.3)
    gap, dg = clearance_and_gradient(q)
    assert gap < 0.03 - 1e-3, "Choose a point safely inside the active hinge branch"
    analytic = tuple(-g for g in dg)
    h = 1e-6
    numerical = []
    for j in range(2):
        plus, minus = list(q), list(q)
        plus[j] += h
        minus[j] -= h
        numerical.append((penalty(plus) - penalty(minus)) / (2 * h))
    error = max(abs(a - b) for a, b in zip(analytic, numerical))
    assert error < 1e-7, (analytic, numerical)
    correct = max(0.0, 0.03 - (0.04 - 0.02))
    incorrect = max(0.0, 0.03 - 0.04)
    assert math.isclose(correct, 0.01) and incorrect == 0.0
    rotation_gradient = 4 * math.sin(math.pi)
    assert abs(rotation_gradient) < 1e-12
    hours = 200_000 * 0.082 / 3600
    print(f"Collision gradient max error: {error:.3e}")
    print(f"Sphere-aware penalty: {correct:.3f} m; centre-only: {incorrect:.3f} m")
    print(f"Chordal derivative at 180 degrees: {rotation_gradient:.3e}")
    print(f"200K problems, serial sampling: {hours:.3f} hours")
    print("Local mathematical checks passed; no policy was trained.")


if __name__ == "__main__":
    main()
