# RMA/UMI/RTC, residual RL and contact-calibration articles

Status: complete. The requested continuous review window is 2026-11-05 15:54:51–22:54:51 UTC (China: November 5 23:54:51–November 6 06:54:51). The full window and required acceptance checks are complete.

Three requested articles are integrated into the Hugo inventory:

- `content/posts/ai/rma-umi-rtc-deployment/index.md`: RMA v1, UMI v3 and RTC v2, organized around dynamics context, demonstration coordinates and asynchronous execution.
- `content/posts/ai/residual-reinforcement-learning/index.md`: residual RL v2, fixed-controller/action bookkeeping, TD3 and the original assembly experiments.
- `content/posts/calibration/interactive-contact-calibration/index.md`: contact-calibration v1, SDF weights, active exploration, observability and stopping criteria.

The supplied PDFs are the version authority; `work-log.json` records their paths and SHA-256 hashes. `claim-ledger.json` maps 41 selected high-impact claims to PDF pages or pinned source files, with explicit qualifications. `source-page-map.json`, `source-interpretation.json` and `source-checkouts.json` retain source details. The integrated article makes no claim that the three original papers jointly validated one system. Existing posts, publication dates, older draft states and earlier uncommitted work are preserved. No deployment or commit is part of this task.

## Source findings that affect the explanation

The RTC v2 transition weight includes **both factors**: `c_i * (exp(c_i)-1)/(e-1)`. Direct inspection of PDF page 5, arXiv HTML and pinned official code caught a missing `c_i` in the initial new draft/helper/widget/figure. Earlier arithmetic agreement had checked a shared mistaken formula. Those artifacts were corrected, and the final numerical checks use independent Decimal references plus a pinned-source comparison. At `c=1/2`, the correct weight is approximately `0.1887703344`, versus `0.3775406688` for the exponential-only expression. The existing dedicated RTC article already had the correct factor.

Contact-calibration v1 Eq. 7 prints `2*sigma_P` in the exponent denominator but `sigma_P^2` in the prefactor; the surrounding variance terminology also differs. PDF and HTML agree. The article retains the literal expression, separately defines a consistent standard-deviation Gaussian, and checks both normalized-weight conventions. It does not infer an unseen implementation. Eq. 12's one-sided contact/noncontact stopping branches are also retained. The primary lab entry was checked for further implementation links; `contact-primary-source-check.json` records the limited scope of that search.

Contact VII-B also summarizes table calibration as about 30 real actions versus 20 simulated actions, while Table II gives 17.2±6.5 and 21.8±5.8. Direct PDF-raster inspection confirms the mismatch. The article retains the table values and flags the inconsistent prose rather than deriving a trial-count improvement ratio.

RMA's project code currently leads to a 2023 CMS extension, not a frozen original v1 implementation. Its stated simulation time step also conflicts with its stated policy frequency. The article uses the supported deployment frequencies without inventing a resolution. UMI's 2026 source snapshot is explicitly distinguished from the 2024 paper experiments; `relative`, legacy `rel`, component deltas, strict timestamp filtering, actuator lead subtraction and observation-buffer endpoint holding are checked separately.

## Numerical and source checks

The default checks require only Python's standard library and Node:

```bash
python3 -B content/posts/ai/rma-umi-rtc-deployment/deployment_contracts.py --check
python3 -B content/posts/ai/residual-reinforcement-learning/residual_control_sanity.py --check
python3 -B content/posts/calibration/interactive-contact-calibration/contact_calibration_sanity.py --check
python3 -B scripts/render_deployment_figures.py --check
node scripts/test_deployment_labs.cjs
```

The browser arithmetic test covers 2992 parameter cases and 126310 values, including 60-digit Decimal mask values, 80-digit scalar recurrences, finite-grid posteriors and 14 invalid-input cases. Standard-library helpers separately check action/frame contracts, guidance/actor derivatives, clipping, disturbance bounds, a bounded tanh residual with locally unstable zero and independently computed nonzero equilibrium, literal/std kernels, geometric rank, resampling, correlated measurements and stopping boundaries.

Optional checks use an isolated CPython 3.11 environment:

- `check_math_independent.py`: NumPy/SciPy matrix-exponential derivatives, gradients, singular values, Gaussian density/posterior and covariance calculations. It does not import a paper implementation.
- `probe_rtc_mask.py --source-root <pinned-RTC-checkout>`: verifies commit/source hash, extracts the official pure function into a NumPy Float64 adapter and compares 3384 profiles with independent 70-digit Decimal values. This does not execute JAX or a policy.
- The article's NumPy-only `umi_source_probe.py`: 600 pose-matrix comparisons and 16 recorded waypoint calls, including the distinction between raw target timestamps and scheduling times after lead subtraction.
- The article's NumPy/SciPy `umi_observation_probe.py`: 120 synthetic buffers, 560 nearest-frame checks, 360 low-dimensional comparisons, and seven rotation/clamping queries. It imports only a verified pure interpolation module and extracts `get_obs` into recording stubs.
- The article's `umi_waypoint_probe.py`: verifies four immutable Git-object source hashes, loads only the pure trajectory interpolator and replays 21 scheduling-branch cases with two clock origins; 19 independent scalar samples check a constructed finite-speed retiming example. No hardware controller module is imported.
- `check_umi_probe_contracts.py` with `--kind action` or `--kind observation`, plus `--source-root <checkout> --numpy-python <isolated-interpreter>`: wrong revision/hash rejection, verification before third-party imports, exclusive output preservation and an unchanged source checkout.

All three UMI probes ran in fresh environments containing only their pinned numerical runtime dependencies plus pip/setuptools. `-I -B` avoids inherited ROS paths and article bytecode. Preserve a virtual environment interpreter's symlink path when invoking it; resolving that symlink would bypass the environment. The source checks neither import a robot/camera hardware module nor open a device connection. The waypoint probe has a separate `check_umi_waypoint_contracts.py --source-root <checkout> --numpy-python <isolated-interpreter>` driver, which tests rejection before site packages load and output/source preservation. None of these constructed calculations reproduces policy training, paper success rates, physical tracking or hardware timing.

## Figures and article links

The three articles contain nine numbered figures: four conceptual overviews and five deterministic SVGs. The integrated article has four figures; its original RTC `#figure-2` anchor is retained while the new RMA and UMI figures use descriptive IDs and the visible captions run from 1 to 4. The timing preset now references visible figure 4.

`scripts/render_deployment_figures.py` checks and renders exact RTC weights, the UMI reanchoring example, residual headroom, point-plane rank and signed sampling gaps. The UMI example reconstructs the same world pose from both anchors; copying the old relative command instead reconstructs a different pose. Diagram labels, axes and arrows are manually inspected alongside the math.

Conceptual assets and built-in Imagegen prompts/edits are recorded in `docs/image-generation/deployment-sevenhour-2026-11-06.json`. The accepted RMA training diagram was corrected before integration: the estimated context feeds both MSE and the frozen policy, while simulated rollouts return to the history buffer. The rejected initial candidate remains only in the generation record.

## Browser and build review

Run browser drivers serially on a private loopback CDP endpoint. They accept a base URL, CDP port and output directory. `check_articles.cjs`, `check_labs.cjs` and `check_closed_popups.cjs` additionally accept the pinned axe path as their fourth positional argument. `build-snapshots.json` binds frozen build directories to source hashes. An immutable snapshot avoids image-fetch interruptions during live Hugo rebuilds.

The latest normal-config production build is `production9`, recorded in `build-snapshots.json`; its preview is frozen at `http://127.0.0.1:44643`. It passes strict content/math/snippet validation: 103 public article schemas, 344 HTML pages, 37535 local references, 736 editorial acceptance checks and no reported errors or math/snippet warnings. The filter has exactly 103 public articles and 11 learning paths. Asset validation checks 35 emitted-integrity assets, 20 local formula fonts and two manifest icons. The 18 original article assets/downloads match source, build and HTTP bytes.

| Scope | Passed evidence | Source relationship |
| --- | --- | --- |
| Article reading | 162 layouts, 9 image zooms, 5 SVG label-bound checks, 6 keyboard table checks | Production7 full suite; the later residual target-noise paragraph and contact action-count qualification each pass 54 targeted layouts and two semantic checks |
| Formula/table/chart print geometry | 18 ordinary and 18 forced-color layouts at 200% text | Production6; all formulas, tables, figures and widgets unchanged in production9 |
| Default native PDFs | 42 exports, 1566 pages, 126 intact result label/value pairs | Production5; all widgets unchanged afterward; later integrated/residual prose receives targeted native rechecks |
| Final integrated-article native PDFs | 14 exports, 661 pages, 42 intact result pairs; tagged PDFs, article tails and screen restoration pass | Production7, including final caption, source-method label and experiment counts; integrated source unchanged in production8 |
| Final residual-article native PDFs | 14 exports, 429 pages, 42 intact result pairs; tagged PDFs, tails and screen restoration pass | Production8, including the target-policy smoothing clarification |
| Final contact-article native PDFs | 14 exports, 497 pages, 42 intact result pairs; tagged PDFs, tails and screen restoration pass | Production9, including the original prose/TableII discrepancy |
| Edited/invalid/unschedulable lab PDFs | 48 exports, 2546 pages, 216 intact result pairs, 16 print palettes | Production5; widget code, styling and markup unchanged afterward |
| Mandatory shared print suite | 636 draft-inclusive layouts, 502212 text samples; zero failures/exceptions | Same article source as production5 plus three unchanged older drafts |
| All article tables | 24 narrow page contexts, 72 table instances, 64 real ArrowRight scrolling checks | Both selected readable-table fixes included; table source unchanged afterward |
| New reading links and search | 8 contexts, 16 real keyboard fragment activations, 4 populated searches | Both local waypoint section and cross-article TD3 links included; destinations unchanged |
| Full chapter navigation | 106 posts, 11188 chapter activations; modifiers, legacy anchors, Back/Forward and subsequent Tab input | Earlier frozen draft-inclusive source; later added section checked separately above |
| Whole-site image viewers | 1456 cases, 76 articles with images, six image formats, zero runtime exceptions | All nine current figures included; image assets unchanged afterward |
| Whole-site reading/reflow | 1172 reading cases including 748 figure deep links; 2106 reflow layouts | Earlier source; the two later figures are covered by the targeted article suite |
| Whole-site readability | 336 contrast layouts, 276621 text samples and 18 print layouts | Earlier source; article/widget contrast and print palettes also checked separately |
| Popup state review | 36 keyboard-opened controls, resolved references, Escape focus restoration | Populated popup scopes have no axe violations/incomplete results; closed states reviewed separately |
| Empty search review | Six queries with no options; populated searches find each new article | Empty-list review is retained separately from populated results |

The native drivers cover A4/Letter, ordinary/large text, scripts enabled/disabled, themes, forced colors, edited/invalid/unschedulable inputs, result pagination and screen/focus restoration. The PDF raster audits distinguish Type 3 font-metric overhang from actual ink: default production5, edited production5, final integrated production7 and final residual production8 and final contact production9 have no horizontal text overflow or affected paper edges with painted ink. The final integrated audit records 30 pages with vertical font-metric overhang, not clipped glyphs. Six selected 200%-text PDF pages were inspected directly for the corrected RMA caption, RMA trial-count qualification, UMI unequal sample counts shortened source-method link TD3 noise distinction and contact action-count discrepancy. Formula/chart geometry is an additional check; it does not establish scientific correctness.

Manual 320px/200% table review prompted selective horizontal reading for the UMI trajectory-error and RRL initial-condition tables. The compact two-column numerical cost table remains compact. The table driver checks actual keyboard scrolling and records long-cell widths. No shared reading style was changed for an individual table.

Further retained evidence covers widget arithmetic/fallback/recovery, touch scrolling, painted focus outlines, native print position restoration and canvas paper colors. The canvas driver has a negative control. Whole-site axe results have no violations or missing IDs; closed popup heuristics and empty search states receive the state-aware reviews above. An empty violation list is not treated as resolution of every incomplete result. Color contrast is checked independently.

The published filtered-clone/sparse-file command sequence was replayed in a fresh public checkout. All three UMI probes pass and the source worktree stays clean; `umi-published-command-replay.json` records the stages and hashes. The optional source probes execute pure numerical functions or extracted branches with recording stubs. They do not execute hardware queues, servo loops, camera acquisition, a policy or training.

The requested seven-hour review is complete. Temporary PDFs/screenshots are outside published bundles; compact evidence, source hashes and limitations are retained in this directory. No article edit is justified solely by a closed-popup audit heuristic or a Type 3 font-metric box.

`validation-summary.json` binds the current manuscripts to the relevant source-stable reports. Their combined native evidence is 42 exports / 1587 pages / 126 intact result pairs; these are three targeted source-bound groups, not a claim that every earlier whole-site report used the final prose.

All 58 reader self-check questions now have an answer-location and evidence-scope record in `acceptance-coverage.json`; every mapped anchor resolves in production9. The final source-parity check matches all 26 recorded build inputs and all five supplied PDF fingerprints. These editorial mappings do not turn constructed examples into paper-experiment reproduction.
