"""Build/check a deterministic, independently runnable two-article download."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER = ROOT/"content/posts/ai/muon-optimizer"
BATCH = ROOT/"content/posts/ai/muon-learning-rate-batch-size"
ARCHIVE = OPTIMIZER/"muon-examples.zip"
FILES = [
    ("README.txt", OPTIMIZER/"README.txt"),
    ("muon-download.json", OPTIMIZER/"muon-download.json"),
    *[(name, OPTIMIZER/name) for name in ("requirements-numpy.txt", "requirements-torch.txt", "requirements-plots.txt")],
    *[("optimizer/"+name, OPTIMIZER/name) for name in (
        "muon_math.py", "matrix_update.py", "toy_training.py", "make_figures.py",
        "assets/math-reference.json", "assets/matrix-ns-reference.json",
        "assets/matrix-polar-reference.json", "assets/toy-training-reference.json")],
    *[("batch/"+name, BATCH/name) for name in (
        "batch_math.py", "mean_field_probe.py", "make_figures.py", "assets/math-reference.json",
        "assets/batch-interactive-default.json", "assets/mean-field-probe-reference.json")],
    ("checks/test_muon_articles.py", ROOT/"scripts/test_muon_articles.py"),
    ("checks/test_muon_training.py", ROOT/"scripts/test_muon_training.py"),
]


def equivalent(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equivalent(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isfinite(actual) and abs(actual-expected) <= max(1e-13, abs(expected)*1e-10)
    return actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--numpy", action="store_true")
    parser.add_argument("--torch", action="store_true")
    args = parser.parse_args()
    if args.check:
        with zipfile.ZipFile(ARCHIVE) as archive:
            if archive.namelist() != [name for name, _ in FILES]:
                raise SystemExit("Muon download member list is stale")
            for name, source in FILES:
                if archive.read(name) != source.read_bytes():
                    raise SystemExit("Stale Muon download member: " + name)
    else:
        with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, source in FILES:
                info = zipfile.ZipInfo(name, date_time=(2026, 11, 3, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source.read_bytes())
    if args.test:
        with tempfile.TemporaryDirectory(prefix="muon-download-") as parent:
            folder = Path(parent)/"download"
            folder.mkdir()
            # A surrounding checkout must never satisfy a missing extracted file.
            decoy = Path(parent)/"content/posts/ai/muon-optimizer/muon_math.py"
            decoy.parent.mkdir(parents=True)
            decoy.write_text('raise RuntimeError("Imported the parent checkout")\n')
            with zipfile.ZipFile(ARCHIVE) as archive:
                archive.extractall(folder)
            command = [sys.executable, "-B", "checks/test_muon_articles.py"] + (["--numpy"] if args.numpy else [])
            subprocess.run(command, cwd=folder, check=True, timeout=45)
            if args.torch:
                subprocess.run([sys.executable, "-B", "checks/test_muon_training.py"], cwd=folder, check=True, timeout=90)
            checks = [("optimizer/muon_math.py", "optimizer/assets/math-reference.json", []),
                      ("batch/batch_math.py", "batch/assets/math-reference.json", [])]
            if args.numpy:
                checks.extend(("optimizer/matrix_update.py", f"optimizer/assets/matrix-{kind}-reference.json", ["--transform", kind])
                              for kind in ("ns", "polar"))
            for script, reference, flags in checks:
                actual = json.loads(subprocess.check_output([sys.executable, "-B", script]+flags, cwd=folder, timeout=30))
                if not equivalent(actual, json.loads((folder/reference).read_text())):
                    raise SystemExit("Muon numerical reference is stale: " + reference)
            for relative in ("optimizer/muon_math.py", "optimizer/matrix_update.py", "optimizer/toy_training.py", "batch/batch_math.py", "batch/mean_field_probe.py"):
                source = folder/relative
                saved = source.read_bytes()
                fallback = Path(parent)/"fallback"
                fallback.mkdir(exist_ok=True)
                (fallback/source.name).write_text('raise RuntimeError("Imported PYTHONPATH fallback")\n')
                try:
                    source.unlink()
                    result = subprocess.run(command, cwd=folder, env={**os.environ, "PYTHONPATH": str(fallback)},
                                            capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 or "Missing local Muon example source" not in result.stderr:
                        raise SystemExit("Incomplete download did not fail locally: " + relative)
                finally:
                    source.write_bytes(saved)
    print("Muon download matches source" + ("; extracted checks passed" if args.test else ""))


if __name__ == "__main__":
    main()
