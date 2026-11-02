"""Numerical/data helpers shared by the two offline identification examples."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def format_and_save_report(report, output=None, protected=()):
    """Protect model/data inputs even if --output names an alias of one."""
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output is not None:
        output = Path(output)
        for source in (Path(value) for value in protected if value is not None):
            if output.resolve() == source.resolve() or (
                    output.exists() and source.exists() and output.samefile(source)):
                raise ValueError("Report output must not replace a model or input data file")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return text


def joint_names(names):
    try:
        if isinstance(names, (str, bytes)):
            raise ValueError("Joint names need a sequence of nonempty strings")
        names = tuple(names)
    except TypeError as error:
        raise ValueError("Joint names need a sequence of nonempty strings") from error
    if (not names or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Model joint names must be nonempty and unique")
    return names


def _load_signal_csv(path, names, fields):
    names = joint_names(names)
    required = ["time_s"] + [f"{field}/{name}" for field in fields for name in names]
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or any(name not in header for name in required):
            raise ValueError("CSV needs unique time_s and " + "/".join(fields) + " columns for each model joint")
        signal_fields = {name for name in header if name.split("/", 1)[0] in fields}
        if signal_fields != set(required[1:]):
            raise ValueError("CSV signal fields and joint names must exactly match the requested schema")
        rows = []
        for row in reader:
            if None in row:
                raise ValueError(f"CSV line {reader.line_num}: extra unlabelled cells")
            try:
                rows.append([float(row[name]) for name in required])
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV line {reader.line_num}: missing or nonnumeric value") from error
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("CSV needs at least two complete finite samples")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("time_s must be strictly increasing")
    width = len(names)
    return values[:, 0], *(values[:, 1 + k * width:1 + (k + 1) * width] for k in range(len(fields)))


def load_processed_csv(path, names):
    """Already-aligned q/dq/ddq/joint-side torque; no differentiation here."""
    return _load_signal_csv(path, names, ("q", "dq", "ddq", "tau"))


def load_raw_csv(path, names):
    """Position and joint-side torque on a common timestamp axis."""
    return _load_signal_csv(path, names, ("q", "tau"))


def _write_signal_csv(path, names, fields, times, values):
    names = joint_names(names)
    if any(np.iscomplexobj(value) for value in (times, *values)):
        raise ValueError("CSV output signals must be real")
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("CSV output needs at least two finite strictly increasing times")
    arrays = [np.asarray(value, dtype=float) for value in values]
    arrays = [value[:, None] if value.ndim == 1 and len(names) == 1 else value for value in arrays]
    if (len(arrays) != len(fields) or any(value.shape != (len(times), len(names))
            or not np.isfinite(value).all() for value in arrays)):
        raise ValueError("Each output signal needs finite sample-by-joint values matching the header")
    rows = np.column_stack((times, *arrays))
    header = ["time_s"] + [f"{field}/{name}" for field in fields for name in names]
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def write_processed_csv(path, names, times, q, v, a, torque):
    _write_signal_csv(path, names, ("q", "dq", "ddq", "tau"), times, (q, v, a, torque))


def write_raw_csv(path, names, times, q, torque):
    _write_signal_csv(path, names, ("q", "tau"), times, (q, torque))


def scaled_svd_fit(matrix, target, rcond=1e-10):
    """Fit only retained directions of a column-scaled design matrix.

    This is a data-dependent basis, not a proof of structural base parameters.
    The returned full vector minimizes norm in the scaled coordinates.
    It is not a uniquely recovered set of physical link inertias.
    """
    if np.iscomplexobj(matrix) or np.iscomplexobj(target):
        raise ValueError("Regression data must be real")
    matrix, target = np.asarray(matrix, float), np.asarray(target, float)
    if matrix.ndim != 2 or target.shape != (len(matrix),) or not matrix.size:
        raise ValueError("Expected a nonempty 2D matrix and an aligned target vector")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Regression data must be finite")
    if isinstance(rcond, bool) or not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("rcond must lie in (0, 1)")
    # hypot avoids squaring overflow/underflow for otherwise finite columns.
    with np.errstate(over="ignore", under="ignore"):
        scale = np.hypot.reduce(matrix, axis=0)
    if not np.isfinite(scale).all():
        raise ValueError("Column norms exceed floating-point range")
    raw_column_norms = scale.copy()
    scale[scale == 0] = 1.
    normalized = matrix / scale
    u, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    if singular[0] == 0:
        raise ValueError("Design matrix has no observable direction")
    rank = int(np.sum(singular > rcond * singular[0]))
    basis = vt[:rank].T
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        coordinates = (u[:, :rank].T @ target) / singular[:rank]
        parameters = (basis @ coordinates) / scale
    if not np.isfinite(coordinates).all() or not np.isfinite(parameters).all():
        raise ValueError("Fitted coordinates or parameters exceed floating-point range")
    return {"parameters": parameters, "scale": scale, "basis": basis,
            "raw_column_l2_norms": raw_column_norms,
            "coordinates": coordinates, "singular_values": singular, "rank": rank,
            "retained_condition": float(singular[0] / singular[rank - 1]), "rcond": rcond}


def rmse(predicted, observed):
    """Aggregate matching real samples; never let broadcasting invent pairs."""
    if np.iscomplexobj(predicted) or np.iscomplexobj(observed):
        raise ValueError("RMSE needs real samples")
    try:
        predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    except (TypeError, OverflowError) as error:
        raise ValueError("RMSE needs finite real samples") from error
    if (predicted.shape != observed.shape or not predicted.size
            or not np.isfinite(predicted).all() or not np.isfinite(observed).all()):
        raise ValueError("RMSE needs matching nonempty shapes and finite samples")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        residual = predicted - observed
        if not np.isfinite(residual).all():
            raise ValueError("RMSE residuals exceed floating-point range")
        scale = float(np.max(np.abs(residual)))
        return scale * float(np.sqrt(np.mean((residual / scale)**2))) if scale else 0.


def unseen_fraction(matrix, fit):
    normalized = matrix / fit["scale"]
    residual = normalized - (normalized @ fit["basis"]) @ fit["basis"].T
    denominator = np.linalg.norm(normalized)
    return float(np.linalg.norm(residual) / denominator) if denominator else 0.
