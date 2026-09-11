#!/usr/bin/env python3
"""Build/check reproducible article lab archives; test only extracted teaching code."""

import argparse
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LABS = {
    "x-vla": ["action_generation_lab.py", "rotation_lab.py", "timing_lab.py"],
    "real-time-chunking": ["rtc_lab.py"],
}


def archive_bytes(source, prefix, members):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in members:
            entry = zipfile.ZipInfo(prefix + name, date_time=(2026, 9, 12, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (source / name).read_bytes())
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="reject missing/stale archives without writing")
    parser.add_argument("--test", action="store_true", help="run each lab from a temporary extraction")
    args = parser.parse_args()
    for slug, programs in LABS.items():
        source = ROOT / "content/posts/ai" / slug
        members = ["README.txt", "requirements.txt", *programs]
        prefix = slug + "-lab/"
        output = source / (slug + "-lab.zip")
        if args.check:
            if not output.exists():
                raise SystemExit(f"{output.name}: missing; rebuild the archives")
        else:
            output.write_bytes(archive_bytes(source, prefix, members))
        with zipfile.ZipFile(output) as archive:
            # Extraction is limited to fixed filenames produced from local sources.
            if archive.namelist() != [prefix + name for name in members]:
                raise SystemExit(f"{output.name}: unexpected archive members")
            # Compare extracted bytes, not compressed bytes: zlib versions may
            # encode identical files differently across local and CI machines.
            for name in members:
                if archive.read(prefix + name) != (source / name).read_bytes():
                    raise SystemExit(f"{output.name}: stale member {name}; rebuild the archives")
            if args.test:
                with tempfile.TemporaryDirectory(prefix="xvla-rtc-lab-") as temporary:
                    archive.extractall(temporary)
                    directory = Path(temporary) / (slug + "-lab")
                    for program in programs:
                        subprocess.run([sys.executable, "-B", program], cwd=directory,
                                       check=True, timeout=60)
        print(f"{output.name}: {len(members)} files match source", flush=True)


if __name__ == "__main__":
    main()
