#!/usr/bin/env python3
"""Build/check the portable OneJev client examples; offline smoke tests only."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "content/posts/ai/jev-decision-layer"
MEMBERS = (
    ("onejev-readme.txt", "onejev-readme.txt"),
    ("onejev_examples.py", "onejev_examples.py"),
    ("assets/onejev-settings-screen.png", "onejev-settings-screen.png"),
    ("assets/onejev-math-results.json", "onejev-math-results.json"),
    ("assets/onejev-response-example.json", "onejev-response-example.json"),
)
PREFIX = "onejev-examples/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    archive = SOURCE / "onejev-examples.zip"
    if not args.check:
        with zipfile.ZipFile(archive, "w") as output:
            for source, name in MEMBERS:
                entry = zipfile.ZipInfo(PREFIX + name, date_time=(2026, 11, 2, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                output.writestr(entry, (SOURCE / source).read_bytes())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist() != [PREFIX + name for _, name in MEMBERS]:
            raise SystemExit("Unexpected archive members; rebuild")
        for source, name in MEMBERS:
            if bundle.read(PREFIX + name) != (SOURCE / source).read_bytes():
                raise SystemExit("Stale archive member: " + name)
        if args.test:
            with tempfile.TemporaryDirectory(prefix="onejev-download-") as temp:
                bundle.extractall(temp)  # Exact safe member names were checked above.
                work = Path(temp) / PREFIX
                command = [sys.executable, "-B", "onejev_examples.py"]
                p = subprocess.run(command + ["math"], cwd=work, capture_output=True, text=True, check=True)
                result = json.loads(p.stdout)
                if not math.isclose(result["loss_with_brier_weight_1"], 0.2831435513142097, rel_tol=1e-13):
                    raise SystemExit("Unexpected offline arithmetic")
                p = subprocess.run(command + ["request", "--image", "onejev-settings-screen.png"],
                                   cwd=work, capture_output=True, text=True, check=True)
                body = json.loads(p.stdout)
                if body["state"]["screen"] != "<image:1>" or len(body["media"]) != 1:
                    raise SystemExit("Unexpected offline image request")
                p = subprocess.run(command + ["read-response", "onejev-response-example.json"],
                                   cwd=work, capture_output=True, text=True, check=True)
                view = json.loads(p.stdout)
                if (view["next_choice"] != "enable" or view["next_choice_probability"] != 0.8
                        or view["next_confidence"] != 0.733333 or view["progress_score"] != 1.0
                        or view["progress_confidence"] != 0.85):
                    raise SystemExit("Unexpected offline response view")
    print(f"{archive.name}: {len(MEMBERS)} members match source")


if __name__ == "__main__":
    main()
