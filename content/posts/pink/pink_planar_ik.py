"""Offline planar 2R differential IK with Pink, Pinocchio and quadprog.

Units: metres, radians and seconds. No URDF downloads, GUI or robot commands.
"""
import json
from importlib.metadata import version

import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from qpsolvers import available_solvers


def planar_model():
    model = pin.Model()
    first = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "shoulder")
    model.addJointFrame(first)
    second = model.addJoint(first, pin.JointModelRZ(),
                            pin.SE3(np.eye(3), np.array([1., 0., 0.])), "elbow")
    parent_frame = model.addJointFrame(second)
    model.addFrame(pin.Frame("tool", second, parent_frame,
                              pin.SE3(np.eye(3), np.array([.6, 0., 0.])),
                              pin.FrameType.OP_FRAME))
    model.lowerPositionLimit[:] = -2.6
    model.upperPositionLimit[:] = 2.6
    model.velocityLimit[:] = 1.2
    return model


def main():
    if "quadprog" not in available_solvers:
        raise RuntimeError("Install quadprog in the same environment as pin-pink")
    model = planar_model()
    configuration = pink.Configuration(model, model.createData(), np.array([.4, .6]))
    target = pin.SE3(np.eye(3), np.array([1.1, .55, 0.]))
    position = FrameTask("tool", position_cost=1., orientation_cost=0., gain=.5)
    position.set_target(target)
    posture = PostureTask(cost=1e-3)
    posture.set_target(configuration.q.copy())
    tasks = [position, posture]
    dt = .01
    peak_velocity = 0.
    for step in range(400):
        velocity = pink.solve_ik(configuration, tasks, dt, solver="quadprog")
        assert velocity.shape == (model.nv,) and np.isfinite(velocity).all()
        assert np.all(np.abs(velocity) <= model.velocityLimit + 1e-8)
        peak_velocity = max(peak_velocity, float(np.max(np.abs(velocity))))
        configuration.integrate_inplace(velocity, dt)
        assert np.all(configuration.q >= model.lowerPositionLimit - 1e-8)
        assert np.all(configuration.q <= model.upperPositionLimit + 1e-8)
        actual = configuration.get_transform_frame_to_world("tool").translation
        error = float(np.linalg.norm(actual - target.translation))
        if error < 1e-5:
            break
    assert error < 1e-5, f"Target not reached: {error} m"
    print(json.dumps({
        "pink": version("pin-pink"), "pinocchio": pin.__version__, "solver": "quadprog",
        "iterations": step + 1, "dt_s": dt, "simulated_time_s": (step + 1) * dt,
        "final_position_error_m": error, "peak_velocity_rad_s": peak_velocity,
        "final_q_rad": configuration.q.tolist(), "joint_limits_respected": True,
        "orientation_constrained": False, "collision_checked": False,
    }, indent=2))


if __name__ == "__main__":
    main()
