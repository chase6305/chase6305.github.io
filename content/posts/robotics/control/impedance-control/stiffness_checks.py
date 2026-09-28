"""Static stiffness and held-position gain scheduling checks.

Python 3.10+ and NumPy. Matplotlib is optional for --figure.
No robot I/O, contact simulation, or full energy-tank controller.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def planar_compliance():
    lengths = np.array([.5, .4])
    joint_stiffness = np.array([100., 80.])
    rows = []
    for degrees in (90., 30., 5., .5, 0.):
        q2 = np.deg2rad(degrees)
        jacobian = np.array([[-lengths[1]*np.sin(q2), -lengths[1]*np.sin(q2)],
                             [lengths[0]+lengths[1]*np.cos(q2), lengths[1]*np.cos(q2)]])
        compliance = (jacobian / joint_stiffness) @ jacobian.T
        weighted_j = jacobian / np.sqrt(joint_stiffness)
        singular = np.linalg.svd(weighted_j, compute_uv=False)
        eigenvalues = singular[::-1]**2
        np.testing.assert_allclose(np.linalg.eigvalsh(compliance), eigenvalues, atol=1e-16)
        # Cartesian work equals the energy of the two joint springs.
        force = np.array([.2, -.1])
        delta_q = (jacobian.T @ force) / joint_stiffness
        np.testing.assert_allclose(force @ (jacobian @ delta_q),
                                   delta_q @ (joint_stiffness * delta_q), atol=1e-16)
        if degrees == 0:
            np.testing.assert_allclose(compliance @ np.array([1., 0.]), 0, atol=1e-16)
            assert np.linalg.matrix_rank(compliance) == 1
        rows.append({'q2_degrees': degrees,
                     'min_compliance_m_per_N': float(eigenvalues[0]),
                     'max_compliance_m_per_N': float(eigenvalues[1]),
                     'max_cartesian_stiffness_N_per_m': float(1/eigenvalues[0]) if degrees else None})
    assert all(rows[i]['min_compliance_m_per_N'] > rows[i+1]['min_compliance_m_per_N']
               for i in range(len(rows)-1))
    return {'lengths_m': lengths.tolist(), 'joint_stiffness_Nm_per_rad': joint_stiffness.tolist(),
            'q1_rad': 0., 'results': rows, 'singular_stiffness': 'inverse undefined, not zero stiffness'}


def held_position_schedule():
    displacement = .01
    initial_stiffness, target_stiffness = 100., 1000.
    initial_tank, tank_floor = .03, .005
    times = np.linspace(0, 1, 501)
    requested = np.linspace(initial_stiffness, target_stiffness, len(times))
    granted = [initial_stiffness]
    tank = [initial_tank]
    for desired in requested[1:]:
        increment = max(0., desired - granted[-1])
        affordable = 2*max(0., tank[-1]-tank_floor)/displacement**2
        applied = min(increment, affordable)
        cost = .5*displacement**2*applied
        granted.append(granted[-1]+applied)
        tank.append(tank[-1]-cost)
    granted, tank = np.array(granted), np.array(tank)
    potential = .5*displacement**2*granted
    assert np.all(tank >= tank_floor-1e-14)
    np.testing.assert_allclose(potential+tank, potential[0]+initial_tank, atol=1e-14)
    expected_final = initial_stiffness+2*(initial_tank-tank_floor)/displacement**2
    np.testing.assert_allclose(granted[-1], expected_final, atol=1e-10)
    rows = []
    for duration in (.1, 1., 10.):
        power = .5*displacement**2*(target_stiffness-initial_stiffness)/duration
        work = power*duration
        np.testing.assert_allclose(work, .045, atol=1e-15)
        rows.append({'duration_s': duration, 'gain_change_power_W': power,
                     'potential_increase_J': work, 'mechanical_port_work_J': 0.})
    result = {'displacement_m': displacement, 'initial_stiffness_N_per_m': initial_stiffness,
              'requested_final_stiffness_N_per_m': target_stiffness,
              'initial_tank_J': initial_tank, 'tank_floor_J': tank_floor,
              'granted_final_stiffness_N_per_m': float(granted[-1]),
              'tank_final_J': float(tank[-1]),
              'storage_conservation_max_error_J': float(np.max(np.abs(potential+tank-potential[0]-initial_tank))),
              'ramps_without_tank': rows}
    return result, (times, requested, granted, tank, potential)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('stiffness-results.json'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    static = planar_compliance()
    energy, series = held_position_schedule()
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        times, requested, granted, tank, potential = series
        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
        axes[0].plot(times, requested, '--', color='#af84c5', label='Requested')
        axes[0].plot(times, granted, color='#497eaa', label='Granted by energy budget', linewidth=2)
        axes[0].set(xlabel='Time [s]', ylabel='Stiffness [N/m]', title='Held at 10 mm displacement')
        axes[1].plot(times, potential*1000, color='#497eaa', label='Spring potential')
        axes[1].plot(times, tank*1000, color='#76a791', label='Tank energy')
        axes[1].plot(times, (tank+potential)*1000, '--', color='#af84c5', label='Total storage')
        axes[1].axhline(5, color='#b98045', linestyle=':', label='Tank floor')
        axes[1].set(xlabel='Time [s]', ylabel='Energy [mJ]', title='Zero mechanical work; no energy harvesting')
        for ax in axes:
            ax.grid(alpha=.18)
            ax.legend(fontsize=9)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=160)
        plt.close(fig)
    report = {'numpy': np.__version__, 'static_compliance': static, 'held_position': energy,
              'scope': 'Rigid 2R links, joint springs, zero preload, first-order statics; separate scalar gain update at fixed position and zero velocity. No closed-loop passivity guarantee.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
