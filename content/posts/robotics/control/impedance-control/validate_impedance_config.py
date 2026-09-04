"""Validate the example impedance-controller YAML configuration."""

import argparse
import math
from pathlib import Path
from typing import Any

import yaml


VECTOR6_PATHS = (
    ("control", "stiffness"),
    ("control", "damping_ratio"),
    ("control", "virtual_mass"),
    ("control", "integral_gain"),
    ("safety", "wrench_abs_limit"),
)
VECTOR3_PATHS = (
    ("safety", "translation_error_m"),
    ("safety", "rotation_error_rad"),
)
COMPENSATION_MODES = {"none", "gravity", "nonlinear"}
COMMAND_MODES = {"raw_joint_torque", "compensated_joint_torque"}
FORMULATIONS = {"direct_wrench", "inertia_shaped"}
REFERENCE_FRAMES = {"local", "world", "local_world_aligned"}


def value_at(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing required field: {'.'.join(path)}")
        value = value[key]
    return value


def require_finite_vector(
    config: dict[str, Any], path: tuple[str, ...], length: int
) -> list[float]:
    value = value_at(config, path)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{'.'.join(path)} must contain {length} values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value
    ):
        raise ValueError(f"{'.'.join(path)} must contain only finite numbers")
    return [float(item) for item in value]


def require_positive(config: dict[str, Any], path: tuple[str, ...]) -> float:
    value = value_at(config, path)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{'.'.join(path)} must be finite and greater than zero")
    return float(value)


def require_choice(
    config: dict[str, Any], path: tuple[str, ...], choices: set[str]
) -> str:
    value = value_at(config, path)
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{'.'.join(path)} must be one of: {allowed}")
    return value


def require_nonempty_string(config: dict[str, Any], path: tuple[str, ...]) -> str:
    value = value_at(config, path)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{'.'.join(path)} must be a non-empty string")
    return value


def validate(config: dict[str, Any]) -> None:
    if value_at(config, ("schema_version",)) != 1:
        raise ValueError("schema_version must be 1")
    require_nonempty_string(config, ("config_id",))
    require_nonempty_string(config, ("model_hash",))
    require_nonempty_string(config, ("frames", "base"))
    require_nonempty_string(config, ("frames", "controlled"))
    if value_at(config, ("frames", "spatial_order")) != ["linear", "angular"]:
        raise ValueError("frames.spatial_order must be [linear, angular]")
    require_choice(config, ("frames", "reference"), REFERENCE_FRAMES)
    command_mode = require_choice(config, ("interface", "command"), COMMAND_MODES)
    require_choice(config, ("control", "formulation"), FORMULATIONS)

    for path in VECTOR6_PATHS:
        require_finite_vector(config, path, 6)
    for path in VECTOR3_PATHS:
        require_finite_vector(config, path, 3)

    if any(value < 0 for value in value_at(config, ("control", "stiffness"))):
        raise ValueError("control.stiffness cannot contain negative values")
    if any(value <= 0 for value in value_at(config, ("control", "virtual_mass"))):
        raise ValueError("control.virtual_mass must be strictly positive")
    if any(value < 0 for value in value_at(config, ("control", "damping_ratio"))):
        raise ValueError("control.damping_ratio cannot contain negative values")
    if any(value < 0 for value in value_at(config, ("control", "integral_gain"))):
        raise ValueError("control.integral_gain cannot contain negative values")
    for path in VECTOR3_PATHS + (("safety", "wrench_abs_limit"),):
        if any(value <= 0 for value in value_at(config, path)):
            raise ValueError(f"{'.'.join(path)} must be strictly positive")

    period_s = require_positive(config, ("control", "period_s"))
    nyquist_hz = 0.5 / period_s
    for name in ("joint_velocity_cutoff_hz", "external_wrench_cutoff_hz"):
        cutoff_hz = require_positive(config, ("filter", name))
        if cutoff_hz >= nyquist_hz:
            raise ValueError(f"filter.{name} must be below Nyquist ({nyquist_hz:g} Hz)")

    driver_mode = require_choice(
        config, ("interface", "driver_compensation"), COMPENSATION_MODES
    )
    model_mode = require_choice(
        config, ("control", "model_compensation"), COMPENSATION_MODES
    )
    if command_mode == "raw_joint_torque" and driver_mode != "none":
        raise ValueError(
            "raw_joint_torque requires interface.driver_compensation: none"
        )
    if command_mode == "compensated_joint_torque" and driver_mode == "none":
        raise ValueError(
            "compensated_joint_torque requires a declared driver compensation"
        )
    if driver_mode != "none" and model_mode != "none":
        raise ValueError(
            "driver and controller compensation overlap; exactly one layer must own it"
        )

    nullspace_enabled = value_at(config, ("nullspace", "enabled"))
    if not isinstance(nullspace_enabled, bool):
        raise ValueError("nullspace.enabled must be true or false")
    nullspace_stiffness = value_at(config, ("nullspace", "stiffness"))
    nullspace_damping = value_at(config, ("nullspace", "damping_ratio"))
    for path, value in (
        (("nullspace", "stiffness"), nullspace_stiffness),
        (("nullspace", "damping_ratio"), nullspace_damping),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{'.'.join(path)} must be finite and non-negative")

    require_positive(config, ("safety", "torque_rate_nm_s"))
    state_timeout_s = require_positive(config, ("safety", "state_timeout_s"))
    if state_timeout_s < period_s:
        raise ValueError("safety.state_timeout_s cannot be shorter than control.period_s")
    misses = value_at(config, ("safety", "consecutive_deadline_misses"))
    if not isinstance(misses, int) or isinstance(misses, bool) or misses <= 0:
        raise ValueError("safety.consecutive_deadline_misses must be a positive integer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML configuration to validate")
    args = parser.parse_args()

    try:
        with args.config.open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
        if not isinstance(config, dict):
            raise ValueError("the YAML root must be a mapping")
        validate(config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(f"invalid: {error}") from None
    print(f"valid: {args.config}")


if __name__ == "__main__":
    main()
