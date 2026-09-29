# Jev article and release review — 2026-09-29

The user requested two further hours of Jev improvement followed by commit, push and deployment. Work began at 14:37:04 UTC; the earliest authorized push time for this review is 16:37:04 UTC (00:37:04 on September 30 in Asia/Shanghai). The release scope also includes the preceding completed InternVL 3.5, GEN-1.5 and VLA improvements described in [the initial review](article-detail-review-20260929.md).

## Content

- Separate Jev's public interface, provider training claims, paper results, community development runs and independent integration examples. Preserve the unverified status of the supplied NavJev numbers.
- Explain state information loss, candidate coverage, coordinate/combination constraints, observation age, request age, network round-trip latency, submission and ambiguous outcomes.
- Add asynchronous behavior-tree timing, goal-version invalidation, progress-based handoff and the distinction between offline replay and closed-loop evaluation.
- Preserve Jev-Mobile's success-conditioned cost/time denominators and five percentage-point success reduction against the step-wise VLM baseline. Keep the community microwave failure and old/new protocol distinction visible.
- Expand the community drawer accounting into 12 planner and 63 Jev requests, showing full model cost and measured network timing separately from provider latency reports.
- Add validation-only threshold selection, subgroup calibration, Brier arithmetic and explicit synthetic risk/cost counterexamples. The interactive explorer never estimates final task success or invokes a model.
- Define navigation SR/SPL separately from elapsed time, and compare typed selection with SayCan's language and skill-affordance evidence. Neither addition validates the unverified NavJev figures.

Evidence and claim boundaries are recorded in [the source review](jev-source-review-20260929.json). Source snapshots and HTTP checks remain in the local audit directory rather than being republished with the article.

## Reproducible material

`content/posts/ai/jev-decision-layer/decision_gate.py` contains 13 standard-library test groups. The caller owns immutable request metadata and uses a common monotonic clock; the script only returns routing strings. Its strict rejection of a selected ID below the highest reported probability is an explicit example policy, not silent action substitution.

`routing_lab.py` contains eight test groups and constructs 200 records with unique IDs. It selects threshold 0.8 only on the 40-record validation group. The held-out and shifted groups contain 80 records each. These are constructed examples, not sampled robot trials or Jev measurements. A separate binary-probability fixture demonstrates aggregate versus subgroup calibration; a hypothetical Wilson calculation does not claim a confidence interval for the constructed records.

The eight-file `jev-decision-lab.zip` contains both scripts, README, JSON/CSV records, results and the numerical plot. `scripts/package_jev_lab.py --check --test` verifies archive contents and runs the extracted examples. Core calculations require only Python 3.10+ standard library; optional plotting used Matplotlib 3.10.9.

The widget loads a fingerprinted data resource and scopes its CSS to the article. `scripts/check_jev_routing.cjs` compares 909 group/threshold/cost combinations with Python, checks the default/reset threshold against the Python-selected value, and exercises real keyboard input, mobile/desktop themes, missing-data fallback and disabled-JavaScript fallback. It also checks for raw inline TeX left in prose by escaped delimiters.

## Figures

Four active conceptual diagrams use the built-in Imagegen tool; a fifth figure is a deterministic Matplotlib plot. Two concepts were added in this review, and the original decision-loop illustration was edited to include the confidence field on the Jev-to-gate edge. The earlier image remains preserved alongside the revised asset.

The numerical plot uses distinct markers and line styles in addition to color. Its arithmetic remains identical to the published full JSON, and the archive includes the revised plotting code and figure.

Exact final asset paths, source images, prompts, edit prompts and checksums are in [the diagram manifest](image-generation/jev-diagrams-20260929.json). Labels and arrows were inspected against the text. Published captions describe the mechanism and evidence scope without image-production commentary.

## Validation and release boundary

Audit directory: `/tmp/chase-jev-two-hours-20260929-yi29elxw`.

The completed checks include a fresh production build, strict math and snippet validation, learning-path/tag/editorial consistency, article filters, TOC fixtures, offline examples, widget parity and a full 82-page browser review. The article retains the existing heading anchors and includes 12 acceptance checks. New figures were checked at 320, 390 and 1440 CSS pixels across light, dark and warm themes, including five zoom interactions.

The final candidate was built from the 62 staged files in an isolated checkout, using Hugo 0.165.0 and an archive of the tracked Hextra commit `38d18a5a25d9700dc88888b6e906f1bccf4631b0`. The production check covered 270 HTML pages, 20,590 local references, 79 public article schemas and 536 code blocks with no errors or unrendered math warnings. All 27 original Jev heading anchors remain; the revised article has 43 headings. The source review contains 19 scoped claims.

A cold browser visit initially exceeded the interaction check's ten-second page-load wait while the existing KaTeX font CDN was loading. A diagnostic visit showed the widget already initialized, fonts then loaded, and no failed requests or JavaScript exceptions. The check now retains full-page readiness, allows thirty seconds, and includes page/font status when it times out. The production rerun passed all 909 arithmetic comparisons and twelve widget layouts.

The publication uses the repository's normal `main` push and GitHub Pages workflow. No workflow, configuration or theme-submodule changes are included. Generated `public/`, private task-planning drafts and three unrelated pre-existing avatar-generation records are excluded. No Jev API experiments or physical robot actions were performed.
