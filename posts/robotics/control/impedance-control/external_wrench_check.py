"""Check the sign of environment-on-robot wrench using inverse dynamics.

NumPy and Pinocchio 4.1.0; a fixed-base sample arm, no robot commands.
Wrench order is [force, moment], expressed at the offset tool frame.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('external-wrench-results.json'))
    args = parser.parse_args()
    model = pin.buildSampleModelManipulator()
    parent = model.njoints-1
    placement = pin.SE3(pin.exp3(np.array([.2, -.1, .3])), np.array([.12, -.05, .18]))
    frame = model.addFrame(pin.Frame('offset_tool', parent, 0, placement, pin.FrameType.OP_FRAME))
    q = np.array([.35, -.45, .6, -.7, .4, .25])
    force = pin.Force(np.array([3., -2., 5., .4, -.3, .2]))
    external = pin.StdVec_Force()
    for _ in range(model.njoints): external.append(pin.Force.Zero())
    # rnea expects external forces at joint origins in their local coordinates.
    external[parent] = placement.act(force)
    data = model.createData()
    jacobian = pin.computeFrameJacobian(model, data, q, frame, pin.ReferenceFrame.LOCAL)
    expected = jacobian.T @ force.vector
    rows = []
    for name, velocity, acceleration in [
        ('static', np.zeros(model.nv), np.zeros(model.nv)),
        ('moving', np.array([.2,-.1,.15,.3,-.25,.12]), np.array([.1,.2,-.1,.05,.15,-.2])),
    ]:
        no_contact = pin.rnea(model, data, q, velocity, acceleration).copy()
        with_contact = pin.rnea(model, data, q, velocity, acceleration, external).copy()
        estimated = no_contact-with_contact
        np.testing.assert_allclose(estimated, expected, atol=1e-12, rtol=0)
        if name == 'static':
            gravity = pin.computeGeneralizedGravity(model, data, q).copy()
            np.testing.assert_allclose(no_contact, gravity, atol=1e-12, rtol=0)
        recovered_acceleration = pin.aba(model, data, q, velocity, with_contact, external).copy()
        np.testing.assert_allclose(recovered_acceleration, acceleration, atol=1e-12, rtol=0)
        rows.append({'case': name, 'no_contact_inverse_dynamics_nm': no_contact.tolist(),
                     'actuator_torque_with_external_wrench_nm': with_contact.tolist(),
                     'model_minus_actuator_nm': estimated.tolist(),
                     'jacobian_transpose_wrench_nm': expected.tolist(),
                     'torque_identity_max_error_nm': float(np.max(np.abs(estimated-expected))),
                     'forward_dynamics_max_error_rad_s2': float(np.max(np.abs(recovered_acceleration-acceleration)))})
    report = {'pinocchio': pin.__version__, 'numpy': np.__version__,
              'tool_wrench_n_nm': force.vector.tolist(), 'cases': rows,
              'equation': 'M*qdd + h = actuator_torque + J.T*environment_on_robot_wrench',
              'scope': 'Exact model consistency, including a rotated and offset tool frame. '
                       'No sensor, actuator tracking, friction, acceleration estimation or observer validation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
