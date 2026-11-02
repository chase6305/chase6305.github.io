# Pion article continuation, 2026-11-02

Status: complete. The seven-hour window started at
`2026-11-01T17:06:33Z` and finished at `2026-11-02T00:07:29.773478+00:00`
(25256.773 elapsed seconds). Final reconciliation passes; see
`docs/pion-optimization-20261102-sevenhour.json` and `final-acceptance.json`.
Changes remain local; this review does not commit, push or deploy.

## Result

Three optional explanations and runnable standard-library helpers address
specific interpretation errors:

| Explanation | Construction and independent checks |
| --- | --- |
| Action dimension, layer-gradient rank and momentum rank | A scalar-output ReLU network has a full-rank hidden gradient. Loss finite differences, exact elimination/eigenvalues, NumPy backpropagation/SVD and CPU autograd check the construction. Three orthogonal rank-one gradients give EMA ranks one, two and three. |
| Correlated observations and independent sampling units | Copying four independent observations increases log rows without reducing the mean's variance. Exact enumeration checks row-wise correction bias and four-group correction. An unequal-variance, negatively correlated construction checks the general covariance identity. |
| GRPO group signal and retention | Zero-sum detached advantages can yield a nonzero policy gradient. Binary-outcome enumeration, finite differences, a closed second moment and CPU autograd check complete-group and conditional moments. Retention probability and proposal cost remain separate. |

The Bernoulli coefficient agrees with supplied PDF Appendix C, equation A16.
The article follows the score decomposition in A13–A20 while explaining why its
zero-residual scalar example differs from the paper's residual-dominant low-SNR
approximation. The examples are constructed distributions, not model measurements.

Implementation notes now distinguish actual parameter-name routing from the
module-level optimizer table. For a four-dimensional `(2,3,5,7)` gradient, the
pinned VLA class transforms `(2,105)` but scales using the original last two
dimensions: `gamma*sqrt(7)`. Its `view` also requires compatible gradient and
momentum strides. The article's matrix/RMS derivation is explicitly two-dimensional.

Private Decimal contexts fix inherited rounding, exponent-range, flag and trap
behavior in the new GRPO helper and existing flat-spectrum helper. Default saved
JSON remains unchanged. Twenty parameter cases check total variance; 288 perturbed
contexts additionally check the flat-spectrum fix.

Explicit Bash continuations make optional commands reusable from print. Eleven
before/after pairs give identical argument lists in 22 fake-interpreter executions;
no installer or probe runs during this comparison. Local keep-together groups fix
three shell-command splits, one short Python split, two conditional-definition
splits and two GRPO worked-example splits across the five native exports. Concise
expectation-table headers avoid an orphan final character at 200% text sizing.
No shared CSS, template, JavaScript or theme-submodule change is needed.

## Numerical and source evidence

- `numeric-checks.json`: 35 methods pass under CPython 3.10.0 / NumPy 1.26.4 and
  CPython 3.12.14 / NumPy 2.2.6; CPython 3.11.15 passes 25 standard-library methods.
- `wrap-download-stdlib.json` and `wrap-download-numpy.json`: the current served
  archive extracts into paths containing Chinese characters and spaces. Both
  dependency modes pass, including all seven default reference comparisons. The
  standard-library environment has no NumPy, PyTorch or Matplotlib.
- Current archive: 32 members, 105,687 bytes, SHA-256
  `cfb2f7a0f2fb65135411a7b363ecf521506a2ae941d0ee18f320c9b5a8b7606e`.
- `torch-constructs.json`: four hidden networks, 1,072 realized groups and 12 group
  distributions checked with PyTorch 2.8.0+cpu; zero optimizer steps.
- `score-residual-audit.json`: an independent finite-policy audit enumerates 4,092
  groups and checks 1,368 finite differences, maximum error about 1.254e-10. It
  separates reward-conditional means from residual score variance, with zero
  optimizer steps. This is not every regularity assumption of the paper.
- `parameter-layout-audit.json`: six routing cases and two adapted CPU optimizer
  steps. `parameter-stride-audit.json`: eight attempts, four complete and four
  expected `view` rejections. CUDA allocation is redirected to CPU and one-rank
  communication is a synchronous fixture; these do not validate a backend.
- `live-pinned-sources.json`: five immutable official source files match cached
  hashes at commit `87965014b360bde604632cba0aa12e423c6c63c6`. Downloaded source
  is not executed by that comparison.

The supplied 43-page PDF SHA-256 is
`2016d07cca61aaa9be936f5a5a701ff033a89e6022f03a9795f639c4d1a2428f`.
No GPU execution or model training occurs in this window. Existing article GPU
records describe prior work and remain unchanged.

`editorial-cross-check.json` records the final pass through the complete article,
28 self-check answers and helper contracts. `evidence-format-check.json` records
parsing of the durable JSON/gzip records and review-only Python sources. These
inspections add no runtime-test counts.

## Reading, accessibility and print

`grpoexample-source-manifest.json` freezes 51 source files for the final build.
Article HTML SHA-256 is
`1535e0490513e52859dce45f0a959c19b619e3502ed9afcaaf45872eeaa7410a`.
`strict-grpoexample.json` reports 319 HTML pages, 32,458 local references and no
errors or snippet/math warnings. Asset integrity and TOC fixtures also pass.
Production and draft-inclusive previews use loopback ports 13559 and 13561.

Browser suites run serially on dedicated headless Chrome CDP 9240, without using
the user's browser. `grpoexample-browser-batch.json` records eight passing final
purposes: 64 exact keyboard copies, generic math, draft-inclusive print, 24 new
worked-example layouts with six actual arrow-scroll cases, ordinary/forced-color
large-text print, 48 content-search cases across both language entrances, and
native PDF export. Earlier checks cover images, reading, navigation, semantics
and site-wide readability; shared implementation files remain unchanged.

The five tagged native PDFs have **46 / 126 / 45 / 124 / 131 pages, totaling 472**.
Exports preserve source content, print palette, disclosures, focus and reading
position. These are exported-page counts, not visual-review counts.

| Actual PDF pagination checks | Passing cases |
| --- | ---: |
| Short shell commands / Python unit | 25 / 5 |
| Disclosure heading with first content | 45 |
| Numerical table rows | 60 |
| Figure 3 with its full caption | 5 |
| Conditional-moment definitions with formulas | 5 |
| Two-response GRPO example with its result | 5 |

All 46 pages of standard A4 are accepted: two direct current views and 44 inherited
only through exact PNG matches to previously viewed pages. Separately, the selected
raster ledger accepts 153 required rows: 83 direct views and 70 exact-PNG inherited
passes. Its other 92 rows are not claimed manually reviewed. All 245 selected
rasters (211 unique PNG hashes) have geometric edge/empty checks. See
`grpoexample-full-standard-review.json`, `grpoexample-visual-statistics.json` and
`grpoexample-visual-review.json.gz`; curated before/after and final images are in
`visuals/`. A metadata-only last page in large A4 is explicitly accepted; it does
not imply missing body text.

Review-setup failures remain in the evidence: a copy check omitted a disclosure
opener; class-only theme changes left `color-scheme` inconsistent; selecting the
native theme control moved print-reading focus; forced colors exposed both schemes;
and a table assertion ran before resize-observer settling. Corrected preconditions
pass. These are not reported as production defects.

## Preservation and evidence scope

The initial snapshot contains 302 already-dirty files. Exactly seven intentional
edits affect that set; the other 295 remain byte-identical, including completed
prior reviews and unrelated TAMP/identification/site work. Only acceptance record
95 changes. Three new helper/reference pairs and this review/audit are added.
Original publication date, draft state, URL, all 78 prior page IDs and six figure
assets remain intact. No initially clean tracked file changes and no article
bytecode files are present.

`this-window-change-summary.json` and `this-window-text.diff.gz` isolate the
seven modified files and six added helper/reference files against the initial
dirty-workspace snapshot. Review evidence itself is excluded. The binary archive
is represented by before/after hashes; the text diff is a review aid rather than
a complete binary patch.

Evidence uses `/tmp/chase-pion-sevenhour-20261102` for bulky builds, PDFs, logs and
rasters. Durable summaries, helper sources and selected screenshots live here.
Historical phase files deliberately retain their original hashes, archive sizes
and page counts; the **grpoexample** records and archive hash above identify the
final candidate. The final reconciliation verifies source hashes, archive, initial-file
preservation and the actual seven-hour elapsed-time requirement.
`evidence-manifest.json` hashes this review and its audit, excluding itself.
