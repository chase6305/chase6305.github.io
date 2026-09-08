#!/usr/bin/env python3
"""Build the article download, or verify and test its extracted contents.

python scripts/package_rl_lab.py                 # rebuild the ZIP
python scripts/package_rl_lab.py --check         # stdlib-only equality check
python scripts/package_rl_lab.py --check --test  # requires installed PyTorch
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

BUNDLE = Path(__file__).resolve().parents[1] / "content/posts/ai/ppo-dpo-grpo"
FILES = ("README.txt", "requirements.txt", "rl_lab.py", "ppo_chain.py",
         "token_objectives.py", "test_rl_lab.py", "plot_results.py")


def check_archive():
    with zipfile.ZipFile(BUNDLE / "rl-lab.zip") as archive:
        names = archive.namelist()
        if len(names) != len(FILES) or set(names) != set(FILES):
            raise SystemExit("ZIP file list differs from FILES; rebuild with scripts/package_rl_lab.py")
        for name in FILES:
            if archive.read(name) != (BUNDLE / name).read_bytes():
                raise SystemExit(f"Stale ZIP member: {name}; rebuild with scripts/package_rl_lab.py")
    print(f"Verified {len(FILES)} ZIP members against article sources.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing the ZIP")
    parser.add_argument("--test", action="store_true", help="Implies --check; run extracted tests and CLIs")
    args = parser.parse_args()
    if not (args.check or args.test):
        with zipfile.ZipFile(BUNDLE / "rl-lab.zip", "w") as archive:
            for name in FILES:
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 8, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (BUNDLE / name).read_bytes())
    check_archive()
    if args.test:
        with tempfile.TemporaryDirectory(prefix="chase-rl-download-") as directory:
            with zipfile.ZipFile(BUNDLE / "rl-lab.zip") as archive:
                archive.extractall(directory)  # Exact flat member allowlist checked above.
            commands = (
                ["-m", "unittest", "-v", "test_rl_lab.py"],
                ["rl_lab.py", "--algorithm", "atoms"],
                ["rl_lab.py", "--algorithm", "all", "--steps", "3", "--output", "results"],
                ["rl_lab.py", "--algorithm", "dpo", "--preference-flips", "3",
                 "--steps", "3", "--output", "results-noisy-dpo"],
                ["ppo_chain.py", "--steps", "3", "--output", "results-chain"],
                ["token_objectives.py", "--output", "results-tokens.json"],
            )
            for command in commands:
                print("Running extracted:", " ".join(command), flush=True)
                subprocess.run([sys.executable, "-B", *command], cwd=directory, check=True)


if __name__ == "__main__":
    main()
