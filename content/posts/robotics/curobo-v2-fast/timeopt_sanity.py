"""Independent scalar checks for the cuRoboV2-Fast explanation.

This file does not import cuRobo, solve TimeOpt, check collisions, or reproduce
paper benchmarks. It studies one interior path segment at positive progress
speed; it does not implement the planner's start/stop boundary conditions.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import json
import math


def path(s: float) -> tuple[float, float, float, float]:
    """Return q, q_s, q_ss, q_sss for a constructed cubic in radians."""
    return (.2 + .6*s + .3*s*s - .1*s*s*s,
            .6 + .6*s - .3*s*s, .6-.6*s, -.6)


def squared_speed(s: float) -> tuple[float, float, float]:
    """Return x=ds/dt squared, x_s and x_ss, with x strictly positive."""
    return .25 + (s-.5)**2, 2*s-1, 2.


def joint_state(s: float) -> tuple[float, float, float, float]:
    q, qs, qss, qsss = path(s)
    x, xp, xpp = squared_speed(s)
    root = math.sqrt(x)
    return q, qs*root, qss*x+.5*qs*xp, root*(qsss*x+1.5*qss*xp+.5*qs*xpp)


def inverse_dynamics(q: float, velocity: float, acceleration: float,
                     payload: float = 0.) -> float:
    """Frictionless one-link pendulum: base inertia plus a point payload.

    Base rotational inertia is 0.1 kg m^2, base gravitational mass is 0.3 kg,
    payload radius is 0.5 m, and q=0 is the downward equilibrium.
    """
    del velocity  # A single fixed-inertia pendulum has no Coriolis term.
    if not math.isfinite(payload) or payload < 0:
        raise ValueError("payload must be finite and non-negative")
    inertia = .1 + payload*.5**2
    return inertia*acceleration + (.3+payload)*9.81*.5*math.sin(q)


def coefficients(s: float, payload: float = 0.) -> tuple[float, float, float]:
    """Recover A,D,g from the three inverse-dynamics calls in the text."""
    q, qs, qss, _ = path(s)
    gravity = inverse_dynamics(q, 0., 0., payload)
    a = inverse_dynamics(q, 0., qs, payload) - gravity
    d = inverse_dynamics(q, qs, qss, payload) - gravity
    return a, d, gravity


def scalar_example() -> dict:
    s = .5
    q, velocity, acceleration, jerk = joint_state(s)
    uniform_duration = 1/math.sqrt(.25)
    local_duration = 2*math.asinh(1)
    a, d, gravity = coefficients(s, payload=2.)
    x, xp, _ = squared_speed(s)
    return {
        "scope": "constructed scalar path, positive-speed interior; not a planner or hardware benchmark",
        "at_s_0_5": {"q_radians": q, "velocity_rad_s": velocity,
                       "acceleration_rad_s2": acceleration, "jerk_rad_s3": jerk,
                       "payload_kg": 2., "A": a, "D": d, "gravity_torque_Nm": gravity,
                       "affine_torque_Nm": .5*a*xp+d*x+gravity},
        "timing_only_comparison": {"constant_x_0_25_seconds": uniform_duration,
                                   "variable_x_seconds": local_duration,
                                   "duration_reduction_percent": 100*(1-local_duration/uniform_duration),
                                   "feasibility_note": "no actuator limits or start/stop constraints imposed on this comparison"},
        "required_rest_pose": {"q_radians": math.pi/2, "payload_kg": 2.,
                               "gravity_torque_Nm": inverse_dynamics(math.pi/2, 0., 0., 2.),
                               "upper_limit_Nm": 5., "static_feasible": False},
    }


def check() -> None:
    """Compare the chain rule with high-precision derivatives of q(s(t))."""
    # Near s=1/2, t-t_center=u and s=1/2+sinh(u)/2 are exact inverses
    # of the chosen timing integral. Differentiate the composition independently.
    with localcontext() as context:
        context.prec = 80
        D = Decimal
        def composed(u: Decimal) -> Decimal:
            s = D('.5') + (u.exp()-(-u).exp())/4
            return D('.2')+D('.6')*s+D('.3')*s*s-D('.1')*s*s*s
        h = D('1e-8')
        fm2, fm1, f0, fp1, fp2 = [composed(k*h) for k in (-2, -1, 0, 1, 2)]
        reference = ((fp1-fm1)/(2*h), (fp1-2*f0+fm1)/(h*h),
                     (fp2-2*fp1+2*fm1-fm2)/(2*h*h*h))
        actual = joint_state(.5)[1:]
        assert all(abs(D(str(v))-r) < D('1e-12') for v, r in zip(actual, reference))
    for payload in (0., .1, 2., 3.):
        for index in range(101):
            s = index/100
            q, velocity, acceleration, _ = joint_state(s)
            a, d, gravity = coefficients(s, payload)
            x, xp, _ = squared_speed(s)
            expected = inverse_dynamics(q, velocity, acceleration, payload)
            assert math.isclose(.5*a*xp+d*x+gravity, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(joint_state(.5)[1], .4125, abs_tol=1e-15)
    assert math.isclose(joint_state(.5)[2], .075, abs_tol=1e-15)
    assert math.isclose(joint_state(.5)[3], .3375, abs_tol=1e-15)
    # Composite Simpson quadrature independently checks the closed-form duration.
    count = 10000
    values = [1/math.sqrt(squared_speed(i/count)[0]) for i in range(count+1)]
    numerical = (values[0]+values[-1]+4*sum(values[1:-1:2])+2*sum(values[2:-1:2]))/(3*count)
    assert math.isclose(numerical, 2*math.asinh(1), rel_tol=1e-12)
    assert inverse_dynamics(math.pi/2, 0., 0., 2.) > 5
    for payload in (-1., math.nan, math.inf):
        try:
            inverse_dynamics(0., 0., 0., payload)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid payload accepted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    print(json.dumps(scalar_example(), ensure_ascii=False, indent=2, allow_nan=False))
