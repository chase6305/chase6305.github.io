# LAST and OmniCalib article review

The shared browser drivers also cover the subsequently requested Machine Learning for Robotics review. Its claim notes are in [the neighboring review directory](../2026-11-05-robotics-ml-review/README.md).

Final counts and article hashes are recorded in [validation-results.json](validation-results.json).

The later three-hour revision is recorded in [the follow-up review](../2026-11-05-three-paper-threehour/README.md). This directory retains the initial evidence; use the follow-up report for current article hashes, added calculators and pinned-source statistical checks.

## Fixed inputs

| Input | SHA-256 |
| --- | --- |
| `/home/ubuntu/Documents/2606.11221v1.pdf` | `01e34c8010795400f1069af07a9c64b0246cfb2a2f68e003375aa2ee9f3b1ca0` |
| `/home/ubuntu/Documents/2609.19582v2.pdf` | `fe5543d20efe67ecac48b48fd5839a73ce259805388c20cda3fbea4707d201ff` |

The articles use these supplied versions. They do not claim a reproduction of either training or a robot calibration run.

## Claim review

LAST:

- PDF page 3, section 3.2 explicitly takes a GW perspective without optimizing a GW objective. The article does not invent a transport solver or coupling loss.
- Pages 4–5, equations 8–17: block-start-relative SE(3) log, fixed spline controls, soft schema, whitening, residual code, reconstruction and commitment.
- Appendix local BCH analysis, pages 12–13: additivity has a local second-order error. The article separates this from global Euclidean coordinates and checks the block-relative, rather than merely inter-frame, rotation range.
- Page 6 and appendix A.3, pages 15–16: frozen tokenizer labels and shared hidden-state conditioning. No token-to-DiT edge is asserted where the equations give only `h`.
- Page 7, tables 2–3: LIBERO 95.8% average and SimplerEnv 57.3% use their reported protocols. Improvements are percentage points, not universally comparable records.
- Page 8, tables 4–5: the MLP's PlaceObj result exceeds the DiT result; continuous refinement does not improve every task.
- Pages 19–20: degree 4 for pose and degree 0 for gripper; `Tc=16` is not temporal compression for `T=8`; ridge is not a derivative penalty or hard endpoint constraint.
- Metric arguments, sample-dependent covariance decoding, covariance axes and soft/hard reconstruction are discussed as assumptions or implementation questions. The rank-one single-sample whitening limit is an independent conditional derivation, not an assertion about unseen official code.

OmniCalib:

- Pages 3–4, section III: whitening, manifold corrections, staged arm estimation, contact closure, knee prior, time-varying tf projection and head rotation-only scope.
- Page 5, equations 11–13: fixed baselines add information beyond relative rotations. The example additionally checks the unobservable common rotation about one baseline.
- Pages 5–6, experiment protocol: arm offsets are injected before hardware data acquisition; leg offsets are injected offline into recorded data.
- Table I: ICP max joint-change recovery error 0.0060 degrees; ArUco's tiny recovery residual is not absolute encoder accuracy.
- Table II: `Delta = Est - Base - Inject`; displayed columns are rounded. RMS of printed residuals is 0.062880839 degrees. Hip-yaw errors exceed 0.1 degrees. Right-knee local sigma is 0.0089 degrees.
- Pages 7–8: fixed-base MuJoCo reaching is distinct from real static held-out geometry and walking-odometry replay.
- Table III: C is development, A/B are held out on the same physical robot. Best-sequence mean 0.775 degrees is distinct from three-sequence mean 1.061 degrees and iKalibr's 0.902 degrees on a separate rich-motion record.
- Rig-relative repeatability, reference agreement, local Hessian sigma and injected-change accuracy remain separate metrics.

## Repeatable numerical checks

```bash
python3 -B content/posts/ai/last-action-tokenizer/geometry_sanity.py --check
python3 -B content/posts/calibration/omnicalib/observability_sanity.py --check
python3 -B scripts/render_last_omnicalib_figures.py --check
node scripts/test_omnicalib_prior.cjs
```

The last command compares 378 browser-side numbers across 63 cases against the independent Python covariance interface, checks rejected inputs and the prior-strength invariant. The renderer verifies exact SVG bytes as well as the numerical constructions.

Optional NumPy/SciPy oracles, in a separate environment:

```bash
python -B docs/site-reviews/2026-11-05-last-omnicalib/probe_numerical.py --output /tmp/last-omni-numerical.json
```

The recorded [numerical-results.json](numerical-results.json) uses Python 3.11.15, NumPy 2.4.6 and SciPy 1.17.1. Its 684 comparisons use matrix exponential/logarithm, Rotation, independent symmetric eigendecomposition, SciPy spline design matrices, augmented SVD ridge least squares and nuisance least squares. They do not validate high-dimensional training, learned covariance statistics or sensor accuracy.

## Browser checks

Serve a fresh draft-inclusive build at a loopback URL. Use an immutable output directory for the final checks; do not rely on a watcher noticing newly created bundles. Run drivers serially on a private CDP endpoint.

```bash
node docs/site-reviews/2026-11-05-last-omnicalib/check_articles.cjs <baseURL> <CDP-port> <output> /tmp/chase-a11y/node_modules/axe-core/axe.min.js
node docs/site-reviews/2026-11-05-last-omnicalib/check_prior.cjs <baseURL> <CDP-port> <output> /tmp/chase-a11y/node_modules/axe-core/axe.min.js
node docs/site-reviews/2026-11-05-last-omnicalib/check_print_geometry.cjs <baseURL> <CDP-port> <output>
node docs/site-reviews/2026-11-05-last-omnicalib/check_print_geometry.cjs <baseURL> <CDP-port> <output> --forced-colors
node docs/site-reviews/2026-11-05-last-omnicalib/check_native_print.cjs <baseURL> <CDP-port> <output>
```

The main article suite covers 162 reading layouts across three widths, three themes, ordinary/large/spaced text, scripts on/off; six keyboard table scroll cases; seven figure viewers; six semantic audits; downloads; anchors; and all four deterministic SVGs' text bounds. Mobile overflow compares `scrollWidth` with layout `clientWidth`, so Chrome's expanded visual viewport cannot mask an oversized inline formula. The interactive suite covers native keyboard input, presets, invalid and recovered values, preserved focus, contrast, and no-script/blocked-script/initial-plot-failure fallbacks.

The semantic audits' only incomplete category was `aria-valid-attr-value` on closed appearance menus, search and image dialogs. Manual review confirmed each ID reference resolves and the corresponding popup opens through keyboard activation on a fresh article page. The prior experiment had neither violations nor incomplete rules in valid and invalid states.

The 200% print geometry driver also checks the earlier Ultralytics and cuRobo additions after their long equations were split. Native export supports `--last-only`, `--omnicalib-only` and `--review-only`, A4 plus selected Letter cases, tagged PDFs, article tail presence, text bounds and label/value pagination. Poppler is required.

Screenshots and PDF evidence stay in `/tmp`; small final summaries can be recorded alongside this document. Inspect the figures and captions, not just the bounds assertions.

## Practical limits and incidental findings

- No LAST training, official tokenizer execution, hardware calibration or medical deployment was performed.
- Browser checks are Chrome emulation, not physical-device validation.
- Native JavaScript-free printing is controlled by Chrome's scroll anchoring. The driver requires the same screen geometry and visible reading target, records any scroll drift, and only asserts exact restoration when the page's restoration scripts execute. Final LAST exports have zero native drift after long inline math was made breakable.
- An optional whole-site 200% print scan exposed long equations in pre-existing articles. The five articles added in this conversation were fixed and checked separately; other legacy editorial changes are outside this addition. The ordinary-size whole-site suite is recorded separately from that optional stress scan.
