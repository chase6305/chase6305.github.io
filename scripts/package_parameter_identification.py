"""Build/check the self-contained Pinocchio and Drake identification download."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/robotics/dynamics/parameter-identification"
ARCHIVE = BUNDLE / "parameter-identification-lab.zip"
NAMES = ("README.txt", "identification_common.py", "pinocchio_identification.py", "drake_identification.py",
         "preprocess_identification.py", "timing_counterexample.py", "assets/timing-reference.json",
         "preprocessing_study.py", "assets/preprocessing-study.json",
         "make_identification_figure.py", "assets/identification-held-out.png",
         "two_link_id.urdf", "pendulum_id.urdf", "requirements-pinocchio.txt", "requirements-drake.txt",
         "requirements-linux-cp312.txt", "assets/pinocchio-reference.json", "assets/drake-reference.json",
         "check_examples.py")


def source(name):
    return ROOT / "scripts/test_parameter_identification.py" if name == "check_examples.py" else BUNDLE / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--test", action="store_true", help="Needs both backends in this Python environment")
    args = parser.parse_args()
    if (BUNDLE / "requirements-linux-cp312.txt").read_bytes() != (ROOT / "scripts/requirements-identification.txt").read_bytes():
        raise SystemExit("Published and CI identification wheel locks differ")
    if args.check:
        with zipfile.ZipFile(ARCHIVE) as archive:
            if archive.namelist() != list(NAMES):
                raise SystemExit("Identification ZIP member list is stale")
            for name in NAMES:
                if archive.read(name) != source(name).read_bytes():
                    raise SystemExit(f"Stale identification ZIP member: {name}")
    else:
        with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in NAMES:
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source(name).read_bytes())
    if args.test:
        with tempfile.TemporaryDirectory(prefix="identification-lab-") as directory:
            with zipfile.ZipFile(ARCHIVE) as archive:
                archive.extractall(directory)
            subprocess.run([sys.executable, "-B", "check_examples.py"], cwd=directory, check=True, timeout=60)
            for backend in ("pinocchio", "drake"):
                result = subprocess.run([sys.executable, "-B", backend + "_identification.py",
                    "--output", "result.json"], cwd=directory, check=True, capture_output=True, timeout=60)
                actual = json.loads(result.stdout)
                if actual != json.loads(Path(directory, "result.json").read_text()):
                    raise SystemExit("Console and saved result differ")
                expected = json.loads(Path(directory, "assets", backend + "-reference.json").read_text())
                for key in ("urdf_sha256", "algorithm_sha256", "helper_sha256", "joint_names", "regressor_shape", "retained_rank", "data_kind"):
                    if actual[key] != expected[key]:
                        raise SystemExit(f"{backend} reference differs: {key}")
                for key in ("retained_scaled_condition", "train_measured_rmse_nm", "validation_measured_rmse_nm",
                            "validation_noise_free_rmse_nm"):
                    if not math.isclose(actual[key], expected[key], rel_tol=1e-8, abs_tol=1e-10):
                        raise SystemExit(f"{backend} numerical reference differs: {key}")
            study = subprocess.run([sys.executable, "-B", "preprocessing_study.py", "--summary",
                "--output", "window-study.json"], cwd=directory, check=True, capture_output=True, timeout=60)
            actual = json.loads(Path(directory, "window-study.json").read_text())
            expected = json.loads(Path(directory, "assets/preprocessing-study.json").read_text())
            for key in ("algorithm_sha256", "helper_sha256", "preprocessor_sha256", "dynamics_script_sha256", "urdf_sha256", "common_support"):
                if actual[key] != expected[key]:
                    raise SystemExit(f"Window-study reference differs: {key}")
            if len(actual["results"]) != len(expected["results"]):
                raise SystemExit("Window-study reference has different settings")
            for row, reference in zip(actual["results"], expected["results"]):
                for key, value in row.items():
                    if not math.isclose(value, reference[key], rel_tol=1e-8, abs_tol=1e-10):
                        raise SystemExit(f"Window-study numerical reference differs: {key}")
            if b"321/221" not in study.stdout or b"Noise-free RMSE" not in study.stdout:
                raise SystemExit("Window-study console summary omitted its data/target contract")
    print("Identification archive matches source" + ("; extracted examples and references passed" if args.test else ""))


if __name__ == "__main__":
    main()
