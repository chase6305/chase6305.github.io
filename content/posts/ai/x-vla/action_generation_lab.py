"""Original scalar exercise: fixed-noise remixing versus reverse Euler.

Run: python action_generation_lab.py
Uses only Python's standard library. No learned policy or robot is involved.
"""

from fractions import Fraction as F


def predict_clean(x, t):
    # A deliberately simple teaching function, not an X-VLA checkpoint.
    return x / 2 + 2


def check_dataset_routing():
    # Mirrors the documented lookup keys, not a loaded dataset or registry.
    metas = {
        "lab_day": {"robot_type": "Droid-Left"},
        "lab_night": {"robot_type": "Droid-Left"},
    }
    domain_ids = {"Droid-Left": 13}
    weights = {"lab_day": 1, "lab_night": 3, "Droid-Left": 100}
    routes = {name: domain_ids[meta.get("robot_type", name)]
              for name, meta in metas.items()}
    raw = [weights.get(name, 1) for name in metas]
    probabilities = [F(weight, sum(raw)) for weight in raw]
    assert list(routes.values()) == [13, 13]
    assert probabilities == [F(1, 4), F(3, 4)]
    # A robot-type weight alone is not queried for these metadata keys.
    robot_only = {"Droid-Left": 100}
    assert [robot_only.get(name, 1) for name in metas] == [1, 1]
    print("Dataset routing: same domain 13; dataset-key probabilities 1/4 and 3/4")


def check_action_scaling():
    # One erroneous X coordinate, one arm, B=T=1; xyz MSE averages 3 values.
    error_m = F(1, 100)
    conversion = F(1000)
    loss_m = 500 * error_m**2 / 3
    loss_mm = 500 * (conversion * error_m)**2 / 3
    assert loss_m == F(1, 60)
    assert loss_mm / loss_m == 1_000_000
    assert loss_mm / conversion**2 == loss_m

    time, noise, target = F(1, 2), F(1), F(2)
    scaled_path = conversion * (time * noise + (1 - time) * target)
    data_only = time * noise + (1 - time) * conversion * target
    data_and_noise = time * conversion * noise + (1 - time) * conversion * target
    assert scaled_path == data_and_noise == 1500
    assert data_only == F(2001, 2)
    assert data_only != scaled_path
    print("Units: 1 cm single-axis loss=1/60; unchanged coefficient in mm gives 1,000,000x")
    print("Noise path: scaling data alone gives 1000.5; scaling the whole path gives 1500")


def check_masked_conditioning():
    def masked_predict(position, gripper):
        # Toy predictor after dropping the gripper input; not a learned model.
        return position, 2 * position

    assert masked_predict(F(1), F(-7)) == masked_predict(F(1), F(9))
    step = F(1, 100)
    columns = []
    for delta_p, delta_g in ((step, F(0)), (F(0), step)):
        plus = masked_predict(delta_p, delta_g)
        minus = masked_predict(-delta_p, -delta_g)
        columns.append(tuple((a - b) / (2 * step) for a, b in zip(plus, minus)))
    assert columns == [(F(1), F(2)), (F(0), F(0))]
    residual = (F(0), F(1))
    correction = [sum(j * r for j, r in zip(column, residual)) for column in columns]
    assert correction == [2, 0]
    print("Masked input: changing gripper leaves output unchanged; output residual gives VJP [2, 0]")


def main():
    noise, estimate, steps = F(-1), F(0), 3
    remix_rows = []
    print("Fixed-noise remixing: t | input | clean estimate")
    for i in range(steps, 0, -1):
        t = F(i, steps)
        x = t * noise + (1 - t) * estimate
        estimate = predict_clean(x, t)
        remix_rows.append((t, x, estimate))
        print(f"{str(t):>3} | {str(x):>5} | {estimate}")
    assert [row[1] for row in remix_rows] == [F(-1), F(-1, 6), F(17, 18)]
    assert estimate == F(89, 36)

    repeated = []
    for initial_noise in (F(-1), F(-1), F(1)):
        clean = F(0)
        for i in range(steps, 0, -1):
            t = F(i, steps)
            clean = predict_clean(t * initial_noise + (1 - t) * clean, t)
        repeated.append(clean)
    assert repeated == [F(89, 36), F(89, 36), F(37, 12)]
    assert repeated[2] - repeated[0] == F(11, 18)
    print("Noise pairing: replay gives 89/36 twice; changed noise gives 37/12")

    # Euler retains and updates the current path state instead of remixing
    # the initial noise with the latest clean estimate.
    x, dt = noise, F(1, steps)
    euler_rows = []
    print("Reverse Euler: t | input | clean estimate")
    for i in range(steps, 0, -1):
        t = F(i, steps)
        clean = predict_clean(x, t)
        euler_rows.append((t, x, clean))
        print(f"{str(t):>3} | {str(x):>5} | {clean}")
        velocity = (x - clean) / t
        x -= dt * velocity
    assert euler_rows[1][1] == remix_rows[1][1]
    assert euler_rows[2][1] == F(7, 8)
    assert x == F(39, 16)
    assert x != estimate

    # Check the continuous-channel loss identity at a known path point.
    target, predicted, t = F(2), F(3, 2), F(1, 4)
    path_point = t * noise + (1 - t) * target
    true_velocity = noise - target
    predicted_velocity = (path_point - predicted) / t
    action_error = (predicted - target) ** 2
    velocity_error = (predicted_velocity - true_velocity) ** 2
    assert velocity_error == action_error / t**2
    print(f"Action squared error: {action_error}; velocity squared error: {velocity_error}")

    per_domain = 32 * 1024 + (72 * 1024 + 1024) + (1024 * 20 + 20)
    assert per_domain == 128_020
    print(f"Domain-specific parameters: {per_domain:,} per domain; {30 * per_domain:,} total")
    cycle = F(6, 30) + F(80, 1000)
    print(f"Synchronous cycle: {float(cycle) * 1000:.0f} ms; {float(1 / cycle):.2f} requests/s")
    check_dataset_routing()
    check_action_scaling()
    check_masked_conditioning()
    print("All exact-arithmetic checks passed.")


if __name__ == "__main__":
    main()
