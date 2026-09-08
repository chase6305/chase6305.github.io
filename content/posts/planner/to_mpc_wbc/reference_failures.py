"""Probe WholeBodyX reference lifecycle; no plant execution or robot hardware.

Install the local WholeBodyX source as described in the article, then run this file.
The injected error tests manager behavior; it does not measure solver reliability.
"""
import json

import numpy as np
from wholebodyx import JointMPC, MPCReferenceManager, RobotState
from wholebodyx.model import PlanarDualArm
from wholebodyx.mpc import PlanResult
from wholebodyx.qp import QPResult


class FailOnce:
    def __init__(self, planner):
        self.planner = planner
        self.fail_next = False

    def solve(self, state, goal):
        if self.fail_next:
            self.fail_next = False
            return PlanResult(QPResult("error", detail="deliberately injected failure"))
        return self.planner.solve(state, goal)

    def reset(self):
        self.fail_next = False
        self.planner.reset()


def main():
    limits = PlanarDualArm().limits
    planner = FailOnce(JointMPC(limits))
    manager = MPCReferenceManager(planner, limits)
    goal, events = np.full(6, .2), []

    def update(stamp):
        result = manager.update(RobotState(np.zeros(6), np.zeros(6), stamp), goal, .02)
        events.append(dict(stamp=stamp, success=result.success, reason=result.reason,
                           plan_present=manager.plan is not None))
        return result

    assert update(0.).success
    old_plan = manager.plan
    try:
        old_plan.sample(old_plan.valid_until)
    except ValueError:
        expired_rejected = True
    else:
        raise AssertionError("End-of-plan timestamp must not be sampled")

    # The old plan still covers this interval; a changed goal forces a new solve.
    assert .02 + .02 < old_plan.valid_until
    goal = np.full(6, .21)
    planner.fail_next = True
    failed = update(.02)
    assert not failed.success and manager.plan is None
    assert failed.position is None and failed.velocity is None
    assert failed.reason == "goal_changed"
    assert update(.04).success  # A later retry is allowed after this injected error.
    assert not update(.03).success  # Clock rollback latches an invalid-clock state.
    assert not update(.06).success  # Advancing the clock alone does not repair it.
    manager.reset()
    assert update(.06).success
    print(json.dumps(dict(expired_sample_rejected=expired_rejected, events=events), indent=2))


if __name__ == "__main__":
    main()
