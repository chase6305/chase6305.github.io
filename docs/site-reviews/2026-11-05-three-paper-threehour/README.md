# Three-paper continuous editorial review

Started: 2026-11-05 11:28 UTC. Requested review window: three hours.
Completed: 2026-11-05 14:57:12 UTC.
Status: complete; final source hashes, production output and validation totals are recorded below.

Scope: LAST v1, OmniCalib v2 and the supplied Machine Learning for Robotics review. Existing publication dates, published URLs, draft states and chapter anchors are retained. Earlier Ultralytics/cuRobo work remains in the shared working tree.

## Source interpretation

- LAST: distinguish shared world-frame changes from tool/TCP changes. The codebook calculator freezes candidates in original coordinates and applies one covariance to all distances; it is not a trained tokenizer. The sample-dependent second-moment decomposition assumes vector outer products within one whitening group, positive regularization and normalized soft weights. It does not resolve the paper's unspecified control-point/statistics axes or establish an unseen implementation's behavior.
- OmniCalib: retain hardware versus offline injection, Base-subtracted recovery, model-conditional local uncertainties and same-robot held-out sequences. The reported Hessian includes the knee-prior precision. The extended calculator separates an independent data direction from changing prior precision; a QR oracle checks its means and covariance.
- Robotics ML review: define PPO's probability ratio and advantage, expand the five conceptual acronyms, and derive a bounded-disturbance held-input condition for one scalar model. The generic expression does not certify a robot controller. The SureSim probe imports only one hash-verified statistical source and uses fixed constructed arrays; no experiment or confidence-coverage reproduction is claimed.

## Independent arithmetic

```bash
python3 -B content/posts/ai/last-action-tokenizer/geometry_sanity.py --check
python3 -B content/posts/calibration/omnicalib/observability_sanity.py --check
python3 -B content/posts/ai/machine-learning-for-robotics/deployment_sanity.py --check
node scripts/test_last_whitening.cjs
node scripts/test_omnicalib_prior.cjs
```

`probe_numerical.py` additionally compares the small standard-library examples with NumPy/SciPy matrix exponentials/logarithms, SVD least squares/covariances, direct quadratic forms and interval projection. Run these optional dependencies in a separate environment.

`probe_suresim.py` runs the downloadable article script. It requires a checkout at `e9e6cb567f8792630044e5c1662ed6e1db9c7480`; `ppi.py` must hash to `e5d8025a524f6bab9060965c1bd30d07d19ddc4e43cf6069a7619b4b41b4c691`. NumPy 2.4.6 and Matplotlib 3.11.1 were used under Python 3.11.15. With `--check-endpoints`, the probe also independently inverts the scalar prefix-wealth predicate for all 24 raw endpoints; normalized errors stay below 0.000002, consistent with the source search tolerance. This checks arithmetic rather than confidence coverage. The sparse-checkout instructions were executed in a new directory, and the two downloaded scripts were tested together outside the site repository.

## Browser checks

Run serially on a private loopback Chrome CDP endpoint. `check_whitening.cjs` and `check_prior.cjs` cover real native keyboard input, presets, invalid inputs, stale-result clearing, contrast, three themes and static fallbacks after disabled scripts, blocked scripts or initial SVG failure. `check_touch.cjs` uses Chrome-emulated touch taps and scrolling, not physical-device tests.

The baseline article reading/figure/zoom and large-text print drivers remain in `../2026-11-05-last-omnicalib/`. The updated `check_native_print.cjs` also checks LAST result labels and values stay together. Inspect native PDFs, not just emulated print screenshots.

## Native print canvas correction

Native `Page.printToPDF` from a dark forced-color screen retained a black paper margin, despite white computed body/root print styles. A one-page fixture reproduced this. Explicit white `@page` paint fixes the native PDF; setting only `print-color-adjust` on the root did not. The repository-owned print CSS now sets the page background, and `check_print_canvas.cjs` checks actual rasterized paper-edge colors on A4/Letter for the three articles in three themes and both forced-color palettes. Its independent root-only negative-control document must retain the old dark margin.

Poppler's Type-3 Chinese heading font boxes can extend above the page while the actual glyphs remain inset. `check_pdf_paint.py` records these metric candidates and checks raster ink at their affected paper edges. It supplements horizontal text bounds and manual glyph review; it is not PDF/UA certification. Initial forced-color PDFs exposed the dark canvas in this check, rather than clipped heading glyphs.

The review's evaluation section additionally checks [NeME v1](https://arxiv.org/html/2507.06404v1), tables 2–3 and section 6, and [StepEval v1](https://arxiv.org/abs/2509.19524v1). It separates classifier F1, policy meta-F1 and physical success, and retains the original offline-replay limitation and blueprint scope.

## Minimal dependency check

The pinned CPU probe was also run with `-I -B` in a separate CPython 3.11 venv containing only the 11 runtime distributions in the downloadable requirements file (plus venv seed tools). SciPy and PyTorch were absent. PyPI wheel transfers timed out, so this isolated check reused hash-verified installed distribution files rather than claiming a successful clean network installation. `minimal-dependency-manifest.json` records versions, file counts and aggregate hashes. The isolated run produced the same 12 cases and endpoint checks; `python -I -m pip check` passed. Isolated mode matters on this machine because external ROS packages can otherwise enter through PYTHONPATH.

## Validation record

The recorded [validation-results.json](validation-results.json) separates the complete frozen-source browser run from a subsequent OmniCalib wording correction. That correction removes stream alignment from the optimized parameter list; the final 162-layout reading matrix and all 14 OmniCalib native print exports were run again. [source-results.json](source-results.json) records the final published-source hashes and verifies the original dates, draft states and existing explicit chapter anchors against the initial backup.

- The two browser calculators match independent Python calculations over 840 input cases, 4704 numerical values and 672 code-choice comparisons, with 22 rejected argument cases. The NumPy/SciPy review adds 1969 independent small-matrix/control comparisons.
- The complete serial browser run passes 13 drivers: reading/zoom, calculator inputs and fallback, touch, chapter navigation, formula focus/reflow, ordinary/forced large-text print, default and edited-state native PDF export, paper canvas and later reader actions. It includes 416 chapter activations and 18 formula focus/reflow cases.
- Ten axe audits report no violations. Three page-level incomplete `aria-valid-attr-value` results involve closed popup controls; the initial acceptance manually checked their IDs and native keyboard opening. Neither calculator has an incomplete result in its scoped normal/error audits.
- The native print set contains 42 default exports, 32 edited/error-state exports and 30 paper-canvas exports. The raster diagnostic checks 104 PDFs and 3649 pages, with no horizontal text overflow or ink at the affected paper edges. Poppler reports 400 pages with heading font-box overshoot; those metrics are retained separately from painted ink. The corrected OmniCalib supplement adds 14 checked exports / 457 pages, again with no affected-edge ink.
- Shared ordinary and forced-color print checks each pass 618 layouts. Article-scoped 200% print checks pass another 60 layouts. The earlier optional legacy-site 200% stress issues remain outside these three articles; a whole-site 200% pass is not claimed.
- The final whole-site deep-reading check passes 1132 cases, including 720 figure-anchor cases, with no exceptions. Eight selected native PDF pages at 200% text size, all seven mobile figure/caption captures and both mobile large-text calculator captures were also inspected directly; [manual-visual-results.json](manual-visual-results.json) records the page/image hashes and observations, and confirms that all 24 final published-source hashes still match.
- A fresh production build passes strict formula, Python/Bash/structured snippet, inventory/filter and resource integrity checks: 103 reviewed posts, 100 published articles, 678 acceptance checks, 335 HTML pages, 35713 local references and 33 integrity assets. No Python bytecode is published.

The full browser suite can be repeated serially with:

```bash
python3 -B docs/site-reviews/2026-11-05-three-paper-threehour/run_browser_checks.py \
  --base http://127.0.0.1:13297 --port 9337 \
  --output /tmp/three-paper-browser-review \
  --axe /tmp/chase-a11y/node_modules/axe-core/axe.min.js
```

Use a fresh draft-inclusive build, a private loopback Chrome CDP endpoint, Node, Python with Pillow, and Poppler. The runner writes each command's log and checks that published sources remain unchanged during execution. Its native canvas negative-control document is an intentional reproduction of the old margin behavior on the recorded Chrome version.

The native and browser checks are Chrome tests, with emulated touch and forced palettes; they do not certify physical devices or PDF/UA conformance. Arithmetic checks do not implement LAST training, estimate actual OmniCalib parameters, reproduce robotics benchmarks or prove statistical coverage.
