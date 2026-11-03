# Jev / OneJev response and training review, 2026-11-03

This review continues the existing article after the figure pass. The handwritten
request and downloadable client now use the same Noul, Choice and Score questions.
A new response walkthrough maps each displayed value to its API field, distinguishes
four-option confidence 0.733333 from the earlier three-option 0.7, and works through
Score confidence 0.85. The screenshot example is connected to an observe, select,
execute, observe-again cycle. Startup/request diagnosis, candidate-ID training
labels and upstream `--plan-only` complete the practical explanation.

The client adds `read-response FILE`. It reads this example's three answers and
0–2 progress range from a saved JSON file. It does not issue another HTTP request,
execute an action, validate the complete response distribution or replace the
existing execution gate. `ask` retains the server's returned fields. Existing
`math`, `request`, `ask`, image bytes and probability arithmetic remain intact.

The fifth download member, `onejev-response-example.json`, is a **constructed
response excerpt**, with an explicit constructed model name and no `usage`.
`upstream-response-probe.json` compares it with actual pinned Flide 0.1.7 answer
assembly from constructed logits. Actual text and image request planning passes;
using the same constructed outputs for both plans checks format compatibility,
not predictions for the already-enabled screenshot. No model forward is executed.

Eight standard-library tests compare arithmetic with an independent Decimal
oracle; check the published request and response snippets; retain request/media
contracts; preserve original HTTP responses; and read a saved response through
the CLI on a path containing Chinese text and spaces. Reading issues no second
request or file write. Invalid/missing fields and non-finite values produce
visible errors. The five-member archive passes extraction and offline smoke
checks. The installed Flide/qev help exposes the published command flags.
`gguf-availability.json` records the observed Hub revision and available model/
projector files; no GGUF download or llama.cpp inference is claimed. Its inference arithmetic and existing images are unchanged.

`training-target-probe.json` checks the actual pinned `train.common` mapping with
24 candidate permutations: the one-hot target follows `enable`, including when
it leaves the first slot. The source maps missing candidate targets to zero.
The article separates singular training `question` from inference `questions`
and makes the target's annotation assumption explicit.

`training-plan-probe.json` and `plan-only-cpu.log` record an actual, unmodified
upstream `train.sft --plan-only --limit 100` run with `CUDA_VISIBLE_DEVICES` empty.
It renders the public 100-row sample, with no rows above the configured 16,384
limit, and reaches `PLAN_OK`. For this local run, fixed OneJev 0.8B metadata,
tokenizer and processor replace the configuration's default Qwen3.5-0.8B model
name; the input/media paths and runs directory are also local overrides. The
article explains both the default and the `--model` override. The run loads no
model weights, executes no optimizer step and measures no training peak memory.
The four ranks and two steps are a batch plan, not distributed training results.
One missing training dependency was installed into a task-specific target
folder; the previous inference environment was left intact.

The fresh production build passes strict validation: 320 HTML pages, 32,651 local
references, 634 code blocks and 560 acceptance checks, without errors or snippet/
math warnings. Acceptance record 81 has 22 self-checks; the other records remain
unchanged. TOC fixtures and article filtering pass. All 74 previous rendered IDs
remain; two new heading IDs bring the count to 76. The eight existing figures and
rendered routing lab are unchanged. A new troubleshooting fragment was moved to
a normal heading after its initial disclosure fragment landed under the header.

Compressed browser reports cover the accepted build: 24 reading layouts,
72 fresh fragment loads, 220 actual chapter activations, 16 exact asynchronous
code copies and 16 table contexts. Fourteen overflowing table contexts scroll
with actual arrow keys; two fitting contexts retain the expected enhanced/native
focus behavior. The shared code-reading suite also passes its 15 layouts and
copy/disclosure/focus checks. No shared JavaScript, CSS, shortcode, theme or
workflow changes are made in this review.

The expanded search check covers 40 cases across both language entrances and
320/1440 widths, including `read-response`, `PLAN_OK` and `OneJev 训练`. The very
broad query `训练 目标` fills the ordinary 20-result cap with other articles; it is
not treated as an indexing failure or a promise to rank every matching article.
No shared search code is changed. Eight independent axe contexts have no
violations; closed-popup ARIA results remain incomplete, with ID references
resolving. Color contrast is excluded from that engine check.

All eight figures pass 32 image-viewer and 64 geometry/caption contexts. Ordinary
and 200% print checks each cover six contexts. Four tagged native PDF exports
contain 43/114/42/118 pages; these are export counts, not manual-review counts.
New response and plan-only sections are present in each PDF. The actual reviewed
A4 pages and retained phone screenshots are listed in `visual-review.json`.
Capture helpers use the real theme control so class-only changes cannot produce
an artificial mixed code palette.

`preservation.json` checks the 194 files already dirty at the start. Only this
article, its client/README/archive, two relevant scripts and editorial record 81
are permitted changes to that baseline. Earlier figure and article work stays
intact. `article-changes.patch` and `task-scope.json` isolate this hour's revisions;
`source-manifest.json` ties accepted output and review artifacts to source hashes.
The elapsed-window receipt is `../../jev-onejev-response-review-20261103.json`.

Browser helpers take a fresh loopback base URL, private CDP port and output
directory. Run them serially. The semantic helper additionally takes the pinned
external axe path. Native PDF export requires Poppler. No complete inference,
training reproduction, hardware evaluation, commit, push or deployment is claimed.
