"""Compare velocity and torque null spaces on a planar three-link robot.

Python 3.10+, NumPy; Matplotlib only when --figure is requested.
Exact frozen-configuration algebra, not a closed-loop robot simulation.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def kinematics_and_mass(q):
    lengths = np.array([.7, .5, .3])
    masses = np.array([2., 1.5, .8])
    angles = np.cumsum(q)
    vectors = lengths[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    origins = np.vstack((np.zeros(2), np.cumsum(vectors, axis=0)))
    inertia = np.zeros((3, 3))
    for link in range(3):
        center = origins[link] + .5 * vectors[link]
        linear = np.zeros((2, 3))
        angular = np.zeros(3)
        for joint in range(link + 1):
            arm = center - origins[joint]
            linear[:, joint] = [-arm[1], arm[0]]
            angular[joint] = 1
        rotational_inertia = masses[link] * lengths[link]**2 / 12
        inertia += masses[link] * linear.T @ linear + rotational_inertia * np.outer(angular, angular)
    tip_arms = origins[-1] - origins[:-1]
    jacobian = np.vstack((-tip_arms[:, 1], tip_arms[:, 0]))
    return jacobian, inertia, origins[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('nullspace-projection.json'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    q = np.array([.4, -.9, .6])
    jacobian, mass, _ = kinematics_and_mass(q)
    assert np.linalg.matrix_rank(jacobian) == 2
    assert np.linalg.eigvalsh(mass).min() > 0
    # Check geometry with independent finite differences of the tip.
    numeric = np.column_stack([(kinematics_and_mass(q + np.eye(3)[i]*1e-6)[2] -
                                 kinematics_and_mass(q - np.eye(3)[i]*1e-6)[2]) / 2e-6
                                for i in range(3)])
    np.testing.assert_allclose(jacobian, numeric, atol=1e-9)
    rng = np.random.default_rng(20260928)
    dq = rng.normal(size=3)
    # Kinetic energy from the mass matrix must agree with explicit link velocities.
    angles = np.cumsum(q); lengths = np.array([.7, .5, .3]); masses = np.array([2., 1.5, .8])
    endpoint_velocity = np.zeros(2); cumulative_rate = 0.; energy = 0.
    for length, link_mass, angle, rate in zip(lengths, masses, angles, dq):
        cumulative_rate += rate
        link_tip_velocity = length * cumulative_rate * np.array([-np.sin(angle), np.cos(angle)])
        center_velocity = endpoint_velocity + .5 * link_tip_velocity
        energy += .5 * link_mass * (center_velocity @ center_velocity)
        energy += .5 * (link_mass * length**2 / 12) * cumulative_rate**2
        endpoint_velocity += link_tip_velocity
    np.testing.assert_allclose(.5 * dq @ mass @ dq, energy, atol=1e-13)

    inverse_mass_jt = np.linalg.solve(mass, jacobian.T)
    gram = jacobian @ inverse_mass_jt
    acceleration_map = inverse_mass_jt.T
    velocity_projector = np.eye(3) - np.linalg.pinv(jacobian) @ jacobian
    torque_projector = np.eye(3) - jacobian.T @ np.linalg.solve(gram, acceleration_map)
    np.testing.assert_allclose(jacobian @ velocity_projector, 0, atol=1e-13)
    np.testing.assert_allclose(acceleration_map @ torque_projector, 0, atol=1e-13)
    np.testing.assert_allclose(torque_projector @ torque_projector, torque_projector, atol=1e-13)
    assert np.linalg.norm(torque_projector - torque_projector.T) > .1
    raw_torque = np.array([10., -20., 5.])
    exact = torque_projector @ raw_torque
    clipped = np.clip(exact, -.5, .5)
    wrong = velocity_projector @ raw_torque
    assert np.linalg.norm(acceleration_map @ wrong) > .1
    assert np.linalg.norm(acceleration_map @ clipped) > .1
    rows = []
    for damping in [0., .01, .03, .1, .3, 1.]:
        projector = np.eye(3) - jacobian.T @ np.linalg.solve(
            gram + damping**2 * np.eye(2), acceleration_map)
        torque = projector @ raw_torque
        rows.append({'damping_sqrt_inverse_kg': damping,
                     'torque_Nm': torque.tolist(),
                     'incremental_task_acceleration_norm_m_s2': float(np.linalg.norm(acceleration_map @ torque)),
                     'idempotence_error': float(np.linalg.norm(projector @ projector - projector))})
    cases = {'correct_dynamic_projection': exact,
             'velocity_projector_used_on_torque': wrong,
             'correct_projection_then_component_clipping': clipped}
    results = {name: {'torque_Nm': value.tolist(),
                     'incremental_task_acceleration_m_s2': (acceleration_map @ value).tolist(),
                     'incremental_task_acceleration_norm_m_s2': float(np.linalg.norm(acceleration_map @ value))}
               for name, value in cases.items()}
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.8), constrained_layout=True)
        labels = ['Dynamic\nprojection', 'Velocity projector\non torque', 'Dynamic +\ncomponent clipping']
        values = [x['incremental_task_acceleration_norm_m_s2'] for x in results.values()]
        bars = axes[0].bar(labels, values, color=['#83b7a2', '#bd9fd3', '#e9b679'], width=.65)
        for bar, value in zip(bars, values):
            axes[0].annotate(f'{value:.3g}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                             xytext=(0,5), textcoords='offset points', ha='center')
        axes[0].set(ylabel='Task acceleration change [m/s²]', title='Where secondary torque reaches the task')
        axes[0].set_ylim(0, max(values)*1.2)
        axes[1].plot([r['damping_sqrt_inverse_kg'] for r in rows[1:]],
                     [r['incremental_task_acceleration_norm_m_s2'] for r in rows[1:]], 'o-', color='#497eaa')
        axes[1].set(xscale='log', yscale='log', xlabel='Damping λ [kg⁻¹/²]',
                    ylabel='Task acceleration change [m/s²]', title='Regularization introduces task leakage')
        for ax in axes: ax.grid(axis='y', alpha=.2)
        args.figure.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.figure,dpi=160);plt.close(fig)
    report = {'numpy': np.__version__, 'q_rad': q.tolist(), 'joint_mass_matrix_kg_m2': mass.tolist(),
              'task_jacobian_m': jacobian.tolist(), 'raw_secondary_torque_Nm': raw_torque.tolist(),
              'component_limit_Nm': .5, 'cases': results, 'damping_sweep': rows,
              'checks': ['finite-difference geometry', 'independent kinetic energy',
                         'mass positive definite', 'full task row rank', 'velocity kernel',
                         'dynamic torque kernel', 'projector idempotence'],
              'scope': 'Planar 3R, 2D position task; frozen q and qdot. Values are incremental accelerations due only to added torque, not full accelerations including gravity or Jdot*qdot.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
