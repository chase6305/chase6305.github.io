# Article detail review — 2026-09-29

Scope: deepen InternVL 3.5, GEN-1.5 and VLA explanations; add a separate Jev article requested during the work. At the end of this initial review, all changes were local and no commit, push, Jev/robot API experiment or deployment had been performed. The user subsequently authorized a further two-hour Jev review followed by deployment; that follow-up is documented in [Jev review](jev-review-20260929.md).

## Content and reproducible examples

- `content/posts/ai/internvl-3-5/index.md`: causal label alignment, teacher forcing, gradients through a frozen language model, training-stage examples and token weighting. `supervision_check.py` checks seven properties on a tiny synthetic causal model; it does not load InternVL weights.
- `content/posts/ai/gen-1-5/index.md`: demonstration records, task ambiguity, Bayesian information gain, memory/cost and matched evaluation. `prompt_information.py` uses the standard library. The downloadable lab archive contains the updated source and results.
- `content/posts/ai/vla-evolution/index.md`: conditional velocity derivation, batch tensors, concrete arithmetic, time sampling, Euler/Heun budgets, velocity/action/noise parameterization and evaluation. `flow_matching_lab.py` actually trains three fixed-seed, 4,609-parameter networks on a synthetic conditional Gaussian mixture. Results separate sample-label loss, field approximation error and numerical integration error. CPU PyTorch 2.13.0 and 2.8.0 reproduced all 30 reported sample-quality rows identically in these environments.
- `content/posts/ai/jev-decision-layer/index.md`: typed interfaces, candidate grounding, observation freshness, confidence routing, Jev-Mobile statistics, community simulation evidence, behavior-tree integration and controlled ablations. `decision_gate.py` has eight standard-library test groups and synthetic statistical examples. NavJev is explicitly left unverified pending a primary source.

The Jev article preserves the lower success rate versus the step-wise VLM baseline, success-conditioned latency/cost denominators, community microwave failure, retries and lack of physical-robot validation. The community repository is pinned to `f08de2e4e20d6cd69fea9c57ac1062c3ef510f1e`.

Existing URLs, publication dates and draft states were preserved. All existing section IDs remain: InternVL 58 → 61, GEN-1.5 22 → 28, VLA 75 → 84 (headings carrying `data-hextra-search-id`, excluding the page title). At that initial checkpoint, the new Jev article was ready for a normal production build and had not yet been published remotely.

## Assets and prompts

Five conceptual diagrams used the built-in Imagegen tool. Final images are stored inside their article bundles, with selected source paths, full prompts, dimensions and checksums recorded in:

- [Three-article diagram manifest](image-generation/three-article-mechanisms-20260929.json)
- [Jev diagram manifest](image-generation/jev-diagrams-20260929.json)

The manifests link each exact saved asset and prompt. Labels and arrows were visually inspected before acceptance. The Flow Matching learning-results chart uses deterministic Matplotlib plotting from the executed experiment, separately from the conceptual illustrations. No generation commentary was added to published article prose or captions.

## Layout finding

A narrow-screen check exposed horizontal document overflow from positioned formula descendants inside a scrollable table. A repository-owned CSS change sets `position: relative` on `.content table`, keeping positioned mathematical/accessibility descendants inside its scroll region. The stronger site-wide check also caught four intended display equations in the existing reinforcement-learning article written with single-dollar delimiters. Those eight delimiter lines were corrected to double-dollar display math; equations and heading IDs are unchanged. The theme submodule was not modified.

The browser checker now compares document scroll width with `documentElement.clientWidth`. Comparing with `innerWidth` could miss the problem when the mobile layout viewport expanded along with the overflow. Four-article checks at 320, 390 and 1440 CSS pixels, across light/dark/warm themes, passed after this fix, including six new-image zoom checks and four mobile TOC navigations.

## Validation artifacts

Local audit directory: `/tmp/chase-three-articles-20260929-s95k7c1g`.

- Production Hugo build, strict rendered-math validation, Python/structured snippet syntax, local references, learning paths, tag classification and editorial metadata.
- 82 article records, 79 public articles, 243 acceptance checks, 161 classified tags; no static validation errors or math warnings.
- Filter tests, heading/TOC fixtures, GEN lab archive verification, existing VLA self-tests and new example checks.
- `flow-cross-version-comparison.json`, `heading-continuity-final.json`, `final-validation.json` and browser screenshots preserve the numerical and rendering checks.

Private task-planning drafts, existing avatar assets and generated-output tracking were not changed. Unrelated pre-existing avatar-generation records remain untouched.

Final validation: the full 82-page browser suite passed, including mobile navigation, image zoom, three themes and 24 additional article-layout cases, with no JavaScript exceptions. Production output was rebuilt in a fresh `production-complete/` directory and the ordinary ignored `public/` output was refreshed. The work began at 13:01:56 UTC and continued beyond the requested one-hour window while completing the additional Jev article and browser regressions.
