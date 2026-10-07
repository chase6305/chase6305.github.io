"""Scalar PID with filtered measurement derivative and conditional integration.

Python 3.10+, standard library only. This is an offline example, not a driver.
The integral state already includes Ki and has the same unit as the output.
"""
from dataclasses import dataclass, field
from math import isfinite


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    limit: float
    tau: float = 0.02
    anti_windup: bool = True
    integral: float = field(default=0.0, init=False)
    previous_y: float | None = field(default=None, init=False)
    filtered_dy: float = field(default=0.0, init=False)
    last_raw: float = field(default=0.0, init=False)
    last_output: float = field(default=0.0, init=False)

    def __post_init__(self):
        if (not all(isfinite(v) for v in (self.kp, self.ki, self.kd,
                                         self.limit, self.tau))
                or self.limit <= 0 or self.tau < 0
                or not isinstance(self.anti_windup, bool)):
            raise ValueError("finite gains, positive limit and nonnegative tau required")

    def reset(self, measured=None, integral_output=0.0):
        """Reset state; resetting to zero alone does not ensure a bumpless transfer."""
        if not isfinite(integral_output) or (measured is not None and not isfinite(measured)):
            raise ValueError("reset state must be finite")
        self.integral = float(integral_output)
        self.previous_y = None if measured is None else float(measured)
        self.filtered_dy = self.last_raw = self.last_output = 0.0

    def update(self, target, measured, dt):
        if not all(isfinite(v) for v in (target, measured, dt)) or dt <= 0:
            raise ValueError("inputs must be finite and dt positive")
        if (not isfinite(self.integral) or not isfinite(self.filtered_dy)
                or (self.previous_y is not None and not isfinite(self.previous_y))):
            raise ValueError("controller state is nonfinite; reset before reuse")

        # Compute first and commit only after validation; a rejected sample must
        # not poison the derivative history or integrator.
        error = target - measured
        dy = 0.0 if self.previous_y is None else (measured - self.previous_y) / dt
        alpha = 1.0 / (1.0 + self.tau / dt)
        filtered = self.filtered_dy + alpha * (dy - self.filtered_dy)
        increment = self.ki * error * dt
        candidate = self.integral + increment
        base = self.kp * error - self.kd * filtered
        proposed_raw = base + candidate
        if not all(isfinite(v) for v in (error, dy, filtered, increment,
                                         candidate, base, proposed_raw)):
            raise ValueError("PID arithmetic overflow; state was not changed")

        integrate = (not self.anti_windup or abs(proposed_raw) <= self.limit
                     or (proposed_raw > self.limit and increment < 0)
                     or (proposed_raw < -self.limit and increment > 0))
        integral = candidate if integrate else self.integral
        raw = base + integral
        if not isfinite(raw):
            raise ValueError("PID output overflow; state was not changed")
        output = max(-self.limit, min(self.limit, raw))
        self.previous_y, self.filtered_dy = float(measured), filtered
        self.integral, self.last_raw, self.last_output = integral, raw, output
        return output


if __name__ == "__main__":
    controller = PID(kp=2.0, ki=0.5, kd=0.1, limit=3.0)
    print(controller.update(target=1.0, measured=0.0, dt=0.01))
