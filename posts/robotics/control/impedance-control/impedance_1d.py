"""Minimal one-dimensional impedance response simulation.

The external force is applied from 0.5 s to 1.5 s. The script writes a
downsampled response to impedance_response.csv and checks basic invariants.
It uses only the Python standard library.
"""

import csv


DT_S = 0.001
DURATION_S = 2.5
MASS_KG = 2.0
DAMPING_N_S_M = 80.0
STIFFNESS_N_M = 1000.0
FORCE_N = 10.0


def external_force(time_s: float) -> float:
    """Return the force exerted by the environment on the virtual mass."""
    return FORCE_N if 0.5 <= time_s < 1.5 else 0.0


def simulate() -> list[tuple[float, float, float, float]]:
    """Integrate M*x_ddot + D*x_dot + K*x = F_ext with semi-implicit Euler."""
    position_m = 0.0
    velocity_m_s = 0.0
    rows = []

    for step in range(round(DURATION_S / DT_S) + 1):
        time_s = step * DT_S
        force_n = external_force(time_s)
        acceleration_m_s2 = (
            force_n
            - DAMPING_N_S_M * velocity_m_s
            - STIFFNESS_N_M * position_m
        ) / MASS_KG

        # Semi-implicit Euler: update velocity before position.
        velocity_m_s += acceleration_m_s2 * DT_S
        position_m += velocity_m_s * DT_S

        if step % 10 == 0:
            rows.append((time_s, force_n, position_m, velocity_m_s))

    return rows


def write_csv(rows: list[tuple[float, float, float, float]]) -> None:
    """Write the response using explicit SI-unit column names."""
    with open("impedance_response.csv", "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["time_s", "force_n", "position_m", "velocity_m_s"])
        writer.writerows(rows)


def main() -> None:
    rows = simulate()
    write_csv(rows)

    peak_during_force = max(
        abs(position_m)
        for time_s, _, position_m, _ in rows
        if 0.5 <= time_s < 1.5
    )
    final_position_m = rows[-1][2]
    expected_static_displacement_m = FORCE_N / STIFFNESS_N_M

    print(
        f"expected static displacement: "
        f"{expected_static_displacement_m:.6f} m"
    )
    print(f"peak displacement:            {peak_during_force:.6f} m")
    print(f"final position:               {final_position_m:.6f} m")

    assert peak_during_force < 0.02
    assert abs(final_position_m) < 1e-4


if __name__ == "__main__":
    main()
