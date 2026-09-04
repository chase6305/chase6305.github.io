"""Estimate M, D and K from a simulated impedance frequency response.

The example uses only the Python standard library. Each frequency is simulated
independently; early cycles are discarded before the complex response is
estimated with a single-frequency DFT.
"""

import cmath
import math


DT_S = 0.0005
MASS_KG = 2.0
DAMPING_N_S_M = 80.0
STIFFNESS_N_M = 1000.0
FORCE_AMPLITUDE_N = 1.0
FREQUENCIES_HZ = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
SETTLE_CYCLES = 8
MEASURE_CYCLES = 8


def frequency_response(frequency_hz: float) -> complex:
    """Return estimated dynamic stiffness F(jw) / X(jw)."""
    omega = 2.0 * math.pi * frequency_hz
    total_s = (SETTLE_CYCLES + MEASURE_CYCLES) / frequency_hz
    settle_s = SETTLE_CYCLES / frequency_hz
    steps = round(total_s / DT_S)

    position_m = 0.0
    velocity_m_s = 0.0
    force_coefficient = 0.0j
    position_coefficient = 0.0j
    sample_count = 0

    for step in range(steps):
        time_s = step * DT_S
        force_n = FORCE_AMPLITUDE_N * math.sin(omega * time_s)

        # Accumulate force and position from the same instant. Pairing F(t)
        # with the already integrated x(t + dt) creates an artificial phase
        # lead and biases the estimated inertia.
        if time_s >= settle_s:
            basis = cmath.exp(-1j * omega * time_s)
            force_coefficient += force_n * basis
            position_coefficient += position_m * basis
            sample_count += 1

        acceleration_m_s2 = (
            force_n
            - DAMPING_N_S_M * velocity_m_s
            - STIFFNESS_N_M * position_m
        ) / MASS_KG

        velocity_m_s += acceleration_m_s2 * DT_S
        position_m += velocity_m_s * DT_S

    if sample_count == 0 or abs(position_coefficient) < 1e-15:
        raise RuntimeError("insufficient response samples")
    return force_coefficient / position_coefficient


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Fit y = intercept + slope*x with ordinary least squares."""
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0.0:
        raise ValueError("fit requires distinct x values")
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator
    return y_mean - slope * x_mean, slope


def main() -> None:
    responses = []
    print("freq_hz  abs_z_n_m  phase_deg  real_z  imag_z")
    for frequency_hz in FREQUENCIES_HZ:
        response = frequency_response(frequency_hz)
        responses.append((frequency_hz, response))
        print(
            f"{frequency_hz:7.2f}  {abs(response):9.2f}  "
            f"{math.degrees(cmath.phase(response)):9.2f}  "
            f"{response.real:6.2f}  {response.imag:6.2f}"
        )

    omegas = [2.0 * math.pi * frequency for frequency, _ in responses]
    stiffness_estimate, real_slope = linear_fit(
        [omega * omega for omega in omegas],
        [response.real for _, response in responses],
    )
    mass_estimate = -real_slope
    damping_estimate = sum(
        omega * response.imag
        for omega, (_, response) in zip(omegas, responses)
    ) / sum(omega * omega for omega in omegas)

    print()
    print(f"estimated mass:      {mass_estimate:.4f} kg")
    print(f"estimated damping:   {damping_estimate:.4f} N*s/m")
    print(f"estimated stiffness: {stiffness_estimate:.4f} N/m")

    assert abs(mass_estimate - MASS_KG) / MASS_KG < 0.02
    assert abs(damping_estimate - DAMPING_N_S_M) / DAMPING_N_S_M < 0.02
    assert abs(stiffness_estimate - STIFFNESS_N_M) / STIFFNESS_N_M < 0.02


if __name__ == "__main__":
    main()
