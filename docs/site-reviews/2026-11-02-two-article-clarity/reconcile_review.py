"""Reconcile this review; completion also requires the requested elapsed time."""
import argparse
import gzip
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = "0d6f32a910b912d8d9f1d6af4b25a0a829803664"
START = datetime(2026, 11, 2, 6, 21, 9, tzinfo=timezone.utc)
EXPECTED = {
    "content/posts/ai/supermimic/index.md",
    "content/posts/ai/rlhf-rlvr-rlcd/index.md",
    "content/posts/ai/rlhf-rlvr-rlcd/assets/post-training-feedback-routes.svg",
    "docs/blog-editorial-review.json",
    "scripts/check_overflow_focus.cjs",
}


def read(name):
    file = HERE / name
    if file.exists():
        return json.loads(file.read_text())
    return json.loads(gzip.decompress(file.with_suffix(file.suffix + ".gz").read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True)
    assert git("rev-parse", "HEAD").strip() == BASE
    changed = set(git("diff", "--name-only").splitlines())
    assert changed == EXPECTED, changed
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    assert all(p.startswith("docs/site-reviews/2026-11-02-two-article-clarity/") or p in {
        "docs/two-article-clarity-review-20261102.json",
        "docs/image-generation/post-training-feedback-routes-20261102.json",
    } for p in untracked), untracked
    manifest = read("source-manifest.json")
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    original = json.loads(git("show", BASE + ":docs/blog-editorial-review.json"))
    current = json.loads((ROOT / "docs/blog-editorial-review.json").read_text())
    assert len(original["posts"]) == len(current["posts"]) == 96
    assert [a["id"] for a, b in zip(original["posts"], current["posts"]) if a != b] == [83, 91]
    anchors = read("baseline-anchors.json")
    preserved = {}
    for slug, record in anchors.items():
        file = args.public / "posts/ai" / slug / "index.html"
        html = file.read_text()
        ids = {a or b for a, b in re.findall(r'\bid=(?:"([^"]+)"|([^\s>]+))', html)}
        assert set(record["ids"]) <= ids
        preserved[slug] = {"prior_ids_retained": record["count"], "current_ids": len(ids),
                           "html_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    strict = read("strict-build.json")
    assert not strict["errors"] and not strict["python_fragment_warnings"] and not strict["unrendered_math_warnings"]
    reading = read("reading.json")
    assert len(reading["layouts"]) == 48 and len(reading["anchors"]) == 40
    assert all(not r["issues"] for r in reading["layouts"])
    assert all(r["exists"] and r["top"] >= 55 and r["bottom"] <= 1000 for r in reading["anchors"])
    semantic = read("semantics.json")
    assert len(semantic["cases"]) == 16
    assert all(not r["violations"] and not r["missingReferences"] for r in semantic["cases"])
    search = read("search.json")
    assert len(search["cases"]) == 48
    assert all(all(m["exists"] for m in r["matches"]) for r in search["cases"])
    ordinary, large = read("print.json"), read("print-large.json")
    assert len(ordinary["layouts"]) == 576 and all(r["passed"] for r in ordinary["layouts"])
    assert len(large["layouts"]) == 12 and all(r["passed"] for r in large["layouts"])
    units = read("native-pdf-units.json")
    assert units["cases"] == 36 and units["failures"] == 0
    exports = read("native-exports.json")
    assert len(exports["pdfs"]) == 8 and sum(r["pages"] for r in exports["pdfs"]) == 337
    assert all(r["tagged"] and r["restored"]["focus"] and r["restored"]["disclosures"]
               and r["restored"]["scrollError"] <= 1 for r in exports["pdfs"])
    numbers = read("published-numbers.json")
    for name, digest in numbers["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert numbers["torch_snippet_unchanged_since_cpu_execution"]
    assert read("fresh-build-equivalence.json")["byte_differences"] == []
    for slug in anchors:
        assert not list((ROOT / "content/posts/ai" / slug).rglob("*.pyc"))
    now = datetime.now(timezone.utc)
    elapsed = (now - START).total_seconds()
    report = {"status": "preflight" if args.preflight else "complete", "checked_at_utc": now.isoformat(),
              "window_start_utc": START.isoformat(), "elapsed_seconds": elapsed,
              "minimum_elapsed_seconds": 10800, "elapsed_requirement_met": elapsed >= 10800,
              "baseline_revision": BASE, "tracked_files_changed": sorted(changed),
              "source_hashes_match": True, "acceptance_records_changed": [83, 91],
              "preserved_page_ids": preserved,
              "checks": {"strict_html_pages": strict["html_pages"], "reading_layouts": 48,
                         "chapter_loads": 40, "semantic_contexts": 16, "search_cases": 48,
                         "ordinary_print_layouts": 576, "large_print_layouts_in_scope": 12,
                         "native_exports": 8, "native_pages": 337, "short_pdf_units": 36},
              "commit_push_or_deploy": False,
              "known_outside_scope": "The earlier extra whole-site 200% print run failed on 42 other unchanged articles; this review does not claim a full-site 200% print pass."}
    if not args.preflight:
        assert elapsed >= 10800, "The requested three-hour window has not finished"
    output = HERE / ("preflight.json" if args.preflight else "final-acceptance.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "elapsed_requirement_met", "source_hashes_match")}))


if __name__ == "__main__":
    main()
