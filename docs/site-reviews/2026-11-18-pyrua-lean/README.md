# PyRUA-Lean article review

The article uses the user-provided 29-page arXiv v2 PDF and the immutable official
source revision in [source-notes.json](source-notes.json). It does not run the
paper's models, training, benchmark episodes or hardware.

Run standard-library arithmetic and the browser-independent oracle:

```bash
python3 -B content/posts/ai/pyrua-lean/interface_sanity.py --check
python3 -B scripts/render_pyrua_figures.py --check
node scripts/test_pyrua_accounting.cjs
```

Rendering `main-outcomes.svg` additionally needs an isolated Matplotlib environment:

```bash
python -B scripts/render_pyrua_figures.py --output content/posts/ai/pyrua-lean/assets
```

The optional runtime probe requires the pinned official checkout, verifies its
commit and seven file hashes before imports, and uses recording objects without
third-party numerical packages, simulator hosts, network or hardware. Its source
and output rejection can be rechecked with:

```bash
python3 -B docs/site-reviews/2026-11-18-pyrua-lean/check_probe_contracts.py \
  --source /path/to/pinned/PyRUA-Lean --output /tmp/pyrua-probe-contracts.json
```

Serve a fresh production build over loopback. Use a private existing Chrome CDP
endpoint and run browser suites serially under the shared heavy-work lock:

```bash
node docs/site-reviews/2026-11-18-pyrua-lean/check_article.cjs \
  http://127.0.0.1:13681 9498 /tmp/pyrua-reading \
  --screen-only --axe /tmp/chase-a11y/node_modules/axe-core/axe.min.js
node docs/site-reviews/2026-11-18-pyrua-lean/check_article.cjs \
  http://127.0.0.1:13681 9498 /tmp/pyrua-behavior \
  --behavior-only --axe /tmp/chase-a11y/node_modules/axe-core/axe.min.js
node docs/site-reviews/2026-11-18-pyrua-lean/check_print_geometry.cjs \
  http://127.0.0.1:13681 9498 /tmp/pyrua-print-geometry
node docs/site-reviews/2026-11-18-pyrua-lean/check_article.cjs \
  http://127.0.0.1:13681 9498 /tmp/pyrua-print --print-only
```

The screen driver covers 96 combinations of width, theme, font size, text spacing
and script availability; it checks native chapter/figure fragments and independent
semantic rules. Behavior checks include real number-control arrows, Tab/Enter/Space,
invalid recovery, scrolling regions, image viewing, blocked/late lab enhancement,
downloads, and search from both language entrances. The test-only axe-core version
and SHA-256 are pinned to the repository's 4.14.0 engine.

The print driver exports 24 tagged A4/Letter PDFs directly from screen media at
100%/200% root text size, in three themes, with and without page scripts. Poppler
and Python inspect label/value word geometry on the same page; plain extracted
reading order can interleave a wrapped label and its adjacent value. Inputs,
results, focus and reading position must survive printing. Inspect rasterized
pages as well. Add `--forced-colors` to both print drivers for forced palettes;
`--letter-only` narrows native exports when only that paper size is needed.

The export check focuses a calculator input with scripts and the nearby native
chapter heading without scripts. A separate late-reading diagnostic is saved by
`check_native_reanchor.cjs`: Chromium can reposition a no-script document while
printing and retain that position on returning. This
also occurs on the existing Muon guide; exact late-position restoration is a
scripted enhancement, not a guarantee of the native fallback.

Temporary evidence stays outside the repository; [review.json](review.json) records
the completed stages, findings, fixes and practical limits.
