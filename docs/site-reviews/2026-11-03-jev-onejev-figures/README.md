# Jev / OneJev figure review, 2026-11-03

Three figures now introduce the existing OneJev chapter: hosted Jev versus local
OneJev and the Flide engine; candidate-slot logits, softmax and typed Choice
fields; and a shared state prefix with separate question branches. The five
previous figures remain, with their existing IDs and assets, and are numbered
4–8. The article's original publication date, draft state and all 71 prior
rendered IDs remain. Three new figure IDs bring that count to 74.

Two diagrams use the built-in Imagegen tool. Exact prompts, original generated
files, asset hashes, dimensions and label/arrow review are in
`../../image-generation/jev-onejev-figures-20261103.json`. The numerical figure is
produced by `content/posts/ai/jev-decision-layer/make_onejev_readout_figure.py`,
which reads the existing standard-library arithmetic helper. A fresh run in the
recorded local plotting environment produces a byte-identical PNG. Its K = 3, T = 1 values are constructed, not measured model
predictions or robot success rates. The later API example has four candidates;
the adjacent prose now states that difference explicitly.

The explanatory scope follows the article's existing pinned OneJev and Flide
sources. Hosted Jev has a text/structured-state input in this snapshot. Flide
loads OneJev weights; API compatibility does not identify the two model weights.
The shared-prefix figure shows logical input branches, without answer transfer
between questions. The prose distinguishes input isolation from statistical
independence and explains runtime dispatch/fallback and new observations.
No model forward, GPU prediction, training or hardware experiment is added.

All figures use the existing image and caption components. New captions are
shortened after reviewing 320-pixel views at 200% text sizing; detailed explanation
remains in selectable adjacent prose. Figure 2's raw HTML caption uses plain
K/T notation so Markdown math delimiters do not appear literally in HTML or PDF.
`caption-reflow-comparison.json` records the resulting figure-height reduction.
`image-delivery.json` records the original PNGs and emitted responsive WebP
candidates. Three mobile device-pixel ratios select and decode WebP candidates
while retaining the PNG original-image links. Keyboard zoom remains available.

Adding image shortcodes exposed an existing ordinal-dependent ID in the routing
lab. The repository-owned `jev-routing-lab` shortcode now accepts an optional
`id` argument while preserving its old ordinal default. This article pins its
previous `jev-routing-lab-4` value. The rendered lab section is byte-identical to
the baseline, including its four old control/status IDs. No shared JavaScript,
CSS, theme submodule or workflow is changed by this review.

`article-changes.patch` and `task-scope.json` isolate this hour's revisions from
the earlier working-tree edits. `source-manifest.json` identifies the final fresh
production build. It passes
strict site, Python/structured-snippet and math validation: 320 HTML pages and
32,641 local references, without errors or warnings. The existing six OneJev
standard-library tests and four-member archive check pass. Article filtering
passes for all 93 published articles. `preservation.json` verifies that the
142 files already dirty at the start remain unchanged except this article and
acceptance record 81; the other editorial records are unchanged.

Compressed browser reports cover the final article: 24 reading layouts, 64 fresh
fragment loads, 64 figure geometry/caption contexts, 32 image-viewer contexts,
24 searches, 212 actual chapter activations, eight table-arrow cases and eight
exact code copies. Eight independent axe contexts have no violations; the closed
popup `aria-valid-attr-value` results remain incomplete, with all referenced IDs
resolving. Color contrast is excluded from that engine check. Figure checks cover
320/1440 pixels, 100%/200% text, page scripts on/off, reserved aspect ratios and
selected-figure position after image decode. Following an explicit text resize,
the figure check scrolls the chosen target into view; native initial fragment
positioning is covered separately by the reading check.

Print layout checks pass six ordinary and six 200% contexts in three themes,
with page scripts on/off. Four tagged native PDF exports contain 39/107/39/111
pages. Those are export counts, not manual-review counts. `visual-review.json`
lists the actual screenshots and A4 pages reviewed; the three new figures and
full captions are intact. Focus, disclosures and scroll position restore after
export. The PDF captions contain no raw K/T math delimiters.

The existing lab-loading suite passes 89 cases for the filtering and routing
labs, including delayed/failing data and retries. That suite ran after the ID
fix on the earlier production-v3 build; later changes only shorten and clarify
article prose/captions. The final lab HTML remains byte-identical to the baseline.
Full command/build scopes are retained in the reports rather than treating the
lab-loading run as a new final-build measurement.

Browser helpers accept a fresh loopback URL, a private CDP port and an output
directory. Run them serially. `check_semantics.cjs` additionally takes the pinned
external axe path; `check_print.cjs --large-text` checks 200% root text; native
PDF export needs Poppler. `capture_figures.cjs` records complete figure/caption
screenshots with separate viewport and element dimensions.

This work leaves the revisions local. The elapsed-window receipt is stored in
`../../jev-onejev-figures-review-20261103.json`.
