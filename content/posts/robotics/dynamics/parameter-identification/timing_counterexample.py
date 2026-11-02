"""Exact single-frequency force-lag counterexample; standard library only.

A translating mass obeys F=m*a+b*v. All states and forces are analytic,
without sensor noise, filtering, interpolation or hardware measurements.
The validation reference is aligned; a separate metric compares delayed logs.
"""
import argparse
import json
import math
import random

MASS = 2.
DAMPING = .4
AMPLITUDE = .03
RATE = 100.
TRAIN_SAMPLES = 400
VALIDATION_SAMPLES = 300


def valid_number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    try:
        valid = math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    return float(value)


def state(time, frequency):
    omega = 2 * math.pi * frequency
    return (AMPLITUDE * omega * math.cos(omega * time),
            -AMPLITUDE * omega**2 * math.sin(omega * time))


def force(time, frequency, delay=0.):
    velocity, acceleration = state(time - delay, frequency)
    return MASS * acceleration + DAMPING * velocity


def run(delay_ms=20., train_hz=.7, validation_hz=1.1, correct_training_delay=False):
    delay_ms = valid_number(delay_ms, -50., 50., "delay_ms")
    train_hz = valid_number(train_hz, .1, 2., "train_hz")
    validation_hz = valid_number(validation_hz, .1, 2., "validation_hz")
    if type(correct_training_delay) is not bool:
        raise ValueError("correct_training_delay must be a boolean")
    delay = delay_ms / 1000
    omega = 2 * math.pi * train_hz
    phase = 0. if correct_training_delay else omega * delay
    # Expand sin(w(t-delay)) and cos(w(t-delay)) into the two state columns.
    mass = MASS * math.cos(phase) - DAMPING / omega * math.sin(phase)
    damping = DAMPING * math.cos(phase) + MASS * omega * math.sin(phase)

    def prediction(time, frequency):
        velocity, acceleration = state(time, frequency)
        return mass * acceleration + damping * velocity

    def rms(errors):
        return math.sqrt(math.fsum(error * error for error in errors) / len(errors))

    train_errors = [prediction(index / RATE, train_hz) - force(
        index / RATE + (delay if correct_training_delay else 0.), train_hz, delay)
        for index in range(TRAIN_SAMPLES)]
    validation_errors = [prediction(index / RATE, validation_hz) - force(index / RATE, validation_hz)
                         for index in range(VALIDATION_SAMPLES)]
    logged_errors = [prediction(index / RATE, validation_hz) - force(index / RATE, validation_hz, delay)
                     for index in range(VALIDATION_SAMPLES)]
    return {"delay_ms": delay_ms, "train_hz": train_hz, "validation_hz": validation_hz,
        "correct_training_delay": correct_training_delay,
        "mass_kg": mass, "damping_n_s_per_m": damping,
        "train_rmse_n": rms(train_errors), "validation_reference_rmse_n": rms(validation_errors),
        "validation_logged_rmse_n": rms(logged_errors),
        "true_mass_kg": MASS, "true_damping_n_s_per_m": DAMPING,
        "amplitude_m": AMPLITUDE, "rate_hz": RATE,
        "train_samples": TRAIN_SAMPLES, "validation_samples": VALIDATION_SAMPLES,
        "scope": "Analytic noiseless single-frequency signals; validation reference is aligned. Known-lag correction evaluates the analytic force, without sampled interpolation or boundary extrapolation."}


def self_check():
    result = run()
    for key, expected in (("mass_kg", 1.984277531322008),
                          ("damping_n_s_per_m", 1.1712329299973558),
                          ("validation_reference_rmse_n", .11428275596923788),
                          ("validation_logged_rmse_n", .1656119059028341)):
        if not math.isclose(result[key], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"Default counterexample changed: {key}")
    rng = random.Random(20261030)
    for _ in range(300):
        delay, training, validation = rng.uniform(-50, 50), rng.uniform(.1, 2), rng.uniform(.1, 2)
        for corrected in (False, True):
            trial = run(delay, training, validation, corrected)
            if trial["train_rmse_n"] > 1e-12:
                raise AssertionError("Single-frequency lag stopped fitting exactly")
            if corrected and trial["validation_reference_rmse_n"] > 1e-12:
                raise AssertionError("Known analytic correction lost the physical coefficients")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay-ms", type=float, default=20.)
    parser.add_argument("--train-hz", type=float, default=.7)
    parser.add_argument("--validation-hz", type=float, default=1.1)
    parser.add_argument("--correct-training-delay", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if args.check:
            self_check()
        result = run(args.delay_ms, args.train_hz, args.validation_hz, args.correct_training_delay)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
