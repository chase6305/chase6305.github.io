#!/usr/bin/env python3
"""Build or verify a deterministic, standard-library-only article lab archive."""
import argparse
import io
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/ai/llm-training-metrics"
FILES = ("metrics_lab.py", "paired_eval.py", "qwen2.5-7b-budget.json")
README = """LLM metrics lab
================
Python 3.8+; standard library only; no network access or model weights needed.

1. Reproduce arithmetic and the pinned Qwen configuration audit:
   python -B metrics_lab.py
2. Reproduce the synthetic 100-question paired comparison:
   python -B paired_eval.py
3. Compare already-scored real results:
   python -B paired_eval.py --baseline baseline.jsonl --candidate candidate.jsonl --ids ids.json

Each JSONL row: {"id":"q001","correct":true,"status":"ok"}
The ID manifest is the complete expected list, e.g. ["q001"].
Status may be "ok" or "error"; an error must have correct=false and remains
in the denominator. Missing/extra/duplicate IDs are rejected.

These tools do not score text, train a model, or measure model quality.
Demo predictions are synthetic; Qwen values are a configuration/index audit.
Paired bootstrap assumes independent questions and two fixed checkpoints;
it does not account for clustered questions or training-seed variability.
--seed and --repeats control statistical resampling only. They do not train
checkpoints, generate new answers, or isolate decoding variability.

Article: https://chase6305.github.io/posts/ai/llm-training-metrics/
"""


def archive_bytes():
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        sources = [(name, (BUNDLE / name).read_bytes()) for name in FILES]
        sources.append(("README.txt", README.encode("utf-8")))
        for name, data in sources:
            info = zipfile.ZipInfo("llm-metrics-lab/" + name, date_time=(2026, 9, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return result.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the archive is stale")
    args = parser.parse_args()
    output = BUNDLE / "llm-metrics-lab.zip"
    content = archive_bytes()
    if args.check:
        if not output.is_file() or output.read_bytes() != content:
            parser.error("archive is missing or stale; rerun without --check")
        print("Archive matches the current article lab sources")
    else:
        output.write_bytes(content)
        print(f"Wrote {output.relative_to(ROOT)} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
