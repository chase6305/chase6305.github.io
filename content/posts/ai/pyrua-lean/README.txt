PyRUA-Lean article examples
==========================

Python >= 3.10, standard library only. These programs do not run a model,
simulator, segmentation server or physical robot.

1. Paper bookkeeping and constructed examples:
   python3 -B interface_sanity.py --check
   python3 -B interface_sanity.py

   The default output is saved in reference-output.json. The checks compare
   cumulative input accounting against direct prompt replay, reconstruct the
   rounded main-result counts, distinguish RoboDojo macro/pooled aggregation,
   check a rigid-translation placement example, and exhaust all 256 binary
   groups of eight against a Decimal oracle. Per-task RoboDojo counts and
   asymmetric clipping are checked independently. Geometry does not check contact,
   reachability, collision, rotation, perception or hardware.

2. Official runtime, with synthetic recording objects:
   git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git
   git -C PyRUA-Lean checkout e754cf43c822b91f4728777a78071fa4e012c371
   python3 -B probe_runtime.py --source PyRUA-Lean --output runtime-probe.json

   This needs Git and a local checkout. The probe itself does not access the
   network. It verifies HEAD and SHA-256 of seven modules before importing any
   official code; package initialization is skipped and third-party imports
   are disabled. It checks persistence after an exception, non-rollback of an
   already executed recording step, deferred show versus materialized image
   bytes, cell versus primitive budgets, default feedback, and stopped actions.
   The recording object is not a robot simulator. Results do not reproduce the
   paper's benchmark, VLA behavior, or training. Existing output is never
   overwritten. Temporary workspaces are cleaned automatically.

The browser accountant fixes illustrative initial prefixes at 2000 (tool) and
2400 (code). It preserves the same constructed primitive events, groups at
most `group` events per code cell, and rounds retained feedback up per group.
No cache discount, compaction, tokenizer, image encoding, latency, or success
model is included. Last-cell output is not counted as earlier input.

Paper: arXiv:2610.01939v2, October 13, 2026, Sections 3-5 and Appendices A-K.
Source: https://github.com/DAGroup-PKU/PyRUA-Lean
Pinned code: e754cf43c822b91f4728777a78071fa4e012c371, inspected 2026-11-18.
The repository's later releases and leaderboard snapshots are separate from
the paper's October results. No official source or model weights are bundled.
