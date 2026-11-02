# Pion article review, seven-hour window

The seven-hour review ran from 2026-10-30 14:54:41 UTC to
2026-10-30 21:55:06 UTC (7h 0m 25s elapsed). See
[the completed window record](../../pion-optimization-20261030-sevenhour.json)
for phase history. This directory records constructed numerical checks and
reading checks; it does not contain a model-training reproduction.

The article uses arXiv:2605.19282v1 and official Pion commit
`87965014b360bde604632cba0aa12e423c6c63c6`. The local paper's SHA-256 is
`2016d07cca61aaa9be936f5a5a701ff033a89e6022f03a9795f639c4d1a2428f`.

## Numerical evidence

- `rotated-precision.json.gz`: 1,008 ordinary-scale CPU cases, with seven
  rectangular/square shapes, twelve generated orthogonal bases per shape,
  four constructed spectra, and three schedules. Float64 matrix polynomials
  are compared with independent NumPy SVD calculations, then with the pinned
  VLA BF16 helper. The observed maximum Float64 entry error is
  `2.220446049250313e-15`; the maximum BF16 singular value is
  `1.012932418663255`. These are observations for these matrices, not universal
  rounding bounds or estimates of gradient noise.
- `per-head-steps.json.gz`: nine layouts, 108 configurations, 432 actual
  `PerHeadPionAdamW.step()` calls. Rank-one blocks isolate Q/K/V row splitting,
  O column splitting, query/KV head counts, head-local dimension factors, and
  shape-incompatible fallback. The traces include EMA/Nesterov, a zero
  gradient, a missing gradient, and weight decay. Parameters and momentum
  use Float64; the official polynomial uses BF16. The maximum weight
  difference is `2.7755575615628914e-17`, with zero buffer difference.
- `helper-input-mutation.json`: six CPU calls on FP32/FP64/BF16 tensors.
  The pinned VLA `fast_newtonschulz` changes its BF16 input during
  normalization; `high_pass_ns` preserves the tested inputs. This motivates
  recording raw gradients before optimizer execution.
- The article bundle separately includes `assets/official-steps.json`:
  twelve whole-matrix configurations and 48 calls on rank-one observations.
  It is independent of the per-head record above.

From the repository root, use an isolated NumPy/PyTorch environment and the
matching checked-out official file:

```bash
python -B docs/site-reviews/2026-10-30-pion-sevenhour/probe_rotated_precision.py \
  --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py \
  --output /tmp/rotated-precision.json
python -B docs/site-reviews/2026-10-30-pion-sevenhour/probe_per_head_steps.py \
  --official-file /path/to/Pion/RL/verl/verl/utils/muon.py \
  --output /tmp/per-head-steps.json
```

Both probes verify the official source hash before importing it. Their reports
include library versions and reference-source hashes. These two CPU probes load
no GPU, model weights, rollout workers or distributed shards. The per-head inputs verify
block geometry; they do not reproduce Qwen attention or its head distribution.

## VLA receive-buffer ordering

The optional bundled `probe_official_vla_buffers.py` isolates the pinned
`DefaultPion` loop with two tiny parameters. A missing gradient uses and clears
the receive-buffer view before the previous asynchronous gather is consumed.
For same-numel adjacent nonzero/None parameters, two deterministic CPU completion
orders respectively lose the previous gradient update or apply its old update
to the missing-gradient parameter. A real one-rank CPU/Gloo run also observed
the latter outcome. CPU allocation redirection, replayed completion, and the
storage-sharing flat Gloo output view are disclosed separately.

The original class was then called with its original CUDA allocator and
unwrapped one-rank NCCL on an RTX 5090 D v2, PyTorch 2.8.0+cu128 and NCCL 2.27.3.
Five repetitions of eight layouts/patterns produced 35 normal controls. All five
same-group nonzero/None cases lost the preceding gradient contribution: its
first weight was `0.999000` instead of the independent `0.9961715728752538`.
No model or multi-rank training was run.

The current article bundle's `assets/vla-buffer-reference.json` retains 64
cases: 16 CPU replays, eight real Gloo calls, and 40 original CUDA/NCCL calls.
There are 56 normal controls and eight observed independent-update mismatches;
successful probe execution does **not** mean 64 correct optimizer updates.
Real backend timing and the selected device/software limit the observation.
`vla-buffer-cpu-initial.json.gz` preserves the earlier 24-case CPU-only record.

```bash
python -B content/posts/ai/pion-muon-vla-rlvr/probe_official_vla_buffers.py \
  --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py --gloo
# With an available CUDA GPU and NCCL backend:
python -B content/posts/ai/pion-muon-vla-rlvr/probe_official_vla_buffers.py \
  --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py --gloo --nccl
```

## Attention architecture check

`attention-architecture.json` records the original paper's standard-attention
assumption, Qwen3's QK-Norm, the Transformers 4.51.3 reference implementation,
and an immutable official Qwen3-4B configuration. The article now distinguishes
the appendix's norm/temperature motivation from the actual normalized Q/K
forward pass. It also derives projection/block shapes from the explicit
`head_dim=128`, query-head count 32 and KV-head count 8; dividing hidden_size 2560
by query-head count 32 would give the wrong head dimension for this model.

`source-audit.json` records the final 20-point reconciliation with the supplied
43-page paper and pinned files: exact versus finite-precision formulas, SNR
assumptions, experimental budgets/denominators, the reported baseline mismatch,
single-parameter policy versus buffer execution, and limitations. It is a
manual editorial audit, not an extra numerical-test count.

`source-line-links.json` verifies the eleven Pion source ranges against the
retained raw files. The Qwen3 link was additionally corrected to raw lines
195–218, which include both norm construction and the projected Q/K forward
calls. Web-extracted line numbering had omitted blank lines. The architecture
record now uses raw lines 62–76, 195–196 and 217–219.

`probe_qk_norm_scaling.py` and `qk-norm-scaling.json` independently check the
positive-scale statement on 14 constructed activation combinations. The probe
hash-checks the Transformers source and executes only its original RMSNorm
class on CPU, comparing with an 80-digit Decimal reference and fixed nonuniform
learned norm weights. Mathematical cancellation holds at epsilon zero; fixed
positive epsilon affects small activations, while official FP32 normalization
still rounds. The maximum observed norm/probability reference errors were
`2.290272025362583e-7` / `4.5808187354445806e-8`. There is no model, RoPE, mask,
GQA, dropout, attention module or training in this check. It is separate from
the 19 bundled example tests.

```bash
python -B docs/site-reviews/2026-10-30-pion-sevenhour/probe_qk_norm_scaling.py \
  --official-file /path/to/transformers/src/transformers/models/qwen3/modeling_qwen3.py \
  --output /tmp/qk-norm-scaling.json
```

## Article and download contracts

The bundled scalar reference keeps the original default report and deterministic
plot unchanged. Optional `--trace` output accepts an already normalized scalar;
it does not renormalize a scalar into one. Independent 90-digit Decimal
calculations check each intermediate stage.

The stateful reference distinguishes a zero observation from `gradient=None`.
The missing-gradient `skip` and `decay` policies explain the two pinned
single-parameter branches; they do not validate the multi-parameter VLA
communication loop. `gradient_update_rms` measures the gradient contribution only,
excluding weight decay. Its implementation scales before taking a norm so
finite coordinate RMS values remain finite when a direct intermediate norm
would overflow.

Download checks execute extracted neighboring files, including when a parent
directory contains a deliberately failing repository decoy. An incomplete
extraction must fail locally instead of importing the parent checkout.
The served 18-member ZIP matches repository bytes and passes 19 extracted
NumPy-inclusive checks in CPython 3.12.14 / NumPy 2.2.6. Its SHA-256 is
`f709b6a3b14e822bb5e1b3a7f8cf083a3f9e6d001448601278a681dacbf05856`.
The current CI configuration includes these checks in its isolated numerical
environment; remote CI has not been executed by this review.

## Search completeness

The new QK-Norm paragraph exposed an existing multi-term search failure. In the
pinned FlexSearch library, chapter appends could place the same page ID in
different relevance buckets. Duplicate IDs could consume a result limit or
cause multi-term intersections to omit matching pages. The adapter now uses a
shared single-bucket descriptor for page content; title, heading and section
scoring remain separate, and indexing still yields between chapter batches.
The vendored library and its license/hash are unchanged.

`search-append.json.gz` compares all 96 public search entries on 20 targeted
queries from the final CUDA-reviewed build against a whole-text oracle. Eleven queries were incomplete before the
fix; the current body matches equal the oracle and contain no repeated pages.
This is a targeted comparison, not an exhaustive enumeration of all queries.
The independent unit fixtures also check cross-chapter terms, the twenty-result
limit, and 400 generated Latin/CJK/prefix queries with shuffled chapters. The
old descriptor fails the new fixtures. Browser checks cover the default and
Chinese entrances, keyboard editing, network retry and native fallbacks.
`search-append-initial.json.gz` retains the earlier 16-query/nine-incomplete
observation. `search-responsiveness.json` compares old/fixed adapters with the
same earlier search-data bytes at CPU-emulation rates 1 and 6: trusted keyboard
input was accepted during indexing and focus was retained in all four cases.
These loopback observations do not establish a statistical speedup or physical
device performance.

```bash
node scripts/test_site_search.cjs
node docs/site-reviews/2026-10-30-pion-sevenhour/check_search_append.cjs \
  --public /path/to/fresh-production-build --output /tmp/search-append.json
```

## Reading evidence

The final frozen source and PDF snapshot is retained under
`/tmp/chase-pion-sevenhour-20261030/source-linked/`. Its 38 source hashes are
saved in `final-source-checkpoint.json.gz`. The current production build is
`/tmp/chase-pion-sevenhour-20261030/production-complete`; its 2,213 emitted files
are byte-identical to the validated `production-source-linked` build, as
recorded in `final-build.json`. The source-linked build passes strict
blog/snippet validation and pinned-asset checks. See `validation.json` for
counts, source hashes and the distinction between current and earlier checks.

Four native PDFs contain 255 pages: scripts/light/100% (35),
scripts/dark/200% (94), native/warm/100% (34), and native/warm/200% (92).
`final-pdf-raster-report.json.gz` records their hashes and the separate
all-page raster scan: no empty pages or painted ink in the outer 12-pixel band
at 1,500-pixel render height. Font-wide extraction boxes are not painted-ink
bounds.

`final-page-review-queue.json.gz` gives content-review evidence for every page.
All 255 final pages exactly match the fully reviewed CUDA snapshot after the
source-link-only correction. In that preceding snapshot, seventy pages were
viewed directly, 183 matched an earlier reviewed full-page PNG, and two matched
a directly reviewed current page. Its queue is retained as
`cuda-reviewed-page-review-queue.json.gz`, alongside
`cuda-reviewed-source-checkpoint.json.gz` and
`cuda-reviewed-pdf-raster-report.json.gz`. This is not a claim of 255 newly viewed pages or a
substitute of automated edge scanning for content review. Formula, code, figure
caption and table content remains complete. An existing decorative breadcrumb
can be clipped at 200% text size, and some FAQ answers continue on the next page.
Print CSS reveals disclosure bodies even when their DOM summaries/arrows retain
the closed state.

`build-difference.json` verifies that the final source-link correction changed
only that URL fragment in the article HTML and seven related RSS feeds;
2,205 other emitted files are byte-identical. `link-and-pixel-check.json`
records the exact painted-page matches and verifies the corrected URI in all
four new native PDFs, with no obsolete URI. Code-copy, forced-color print,
buffer-table and ZIP checks from before this URL-only change remain identified
by their original report hashes.

The final article passes 96 reading layouts, 172 chapter activations, 378 actual
keyboard scroll checks, 32 exact code copies in four contexts, and six forced-color
print layouts with 5,697 text samples. Mobile buffer-table screenshots show its
left edge and actual keyboard-scrolled rightmost column, including native
200% text. Shared styles and templates were not changed in this window.

The final 21 interactive states have zero confirmed axe violations and zero
runtime exceptions. Ninety-six repeated manual-incomplete node records concern
eight unique closed-popup controls; their referenced IDs exist and open popup
states were checked separately. They are not reported as zero incompletes.
Two search contexts each exercised 17 article queries, including NCCL and the
new buffer explanation.

`browser-evidence.json.gz` retains the full current reports and clearly labels
earlier shared-component runs. The earlier all-site native audit covered 317
routes/634 cases; it predates the final paragraph and is not called a fresh
final all-site run. Search, fallbacks, image viewers, asynchronous copying and
print restoration were checked serially against the agent's loopback CDP
browser. No deployment or commit was performed.

`initial-work-preservation.json` verifies that 114 of the initial 126 dirty files
remain byte-identical, the other 12 changed files are the intended set, the
other 95 editorial records are preserved, and all 63 original article HTML IDs
remain. Original publication date, draft state and URL are stable.

The flow illustration's green Promotion and purple Suppression nodes now match
the quantitative figure. The original asset is retained. Its full edit prompt,
source hash and output hash are in
[the built-in Imagegen record](../../image-generation/pion-flow-20261031-colors.json).
