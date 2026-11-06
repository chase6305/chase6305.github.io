# Deployment article followup

Status: complete. This round follows the completed seven-hour review in `../2026-11-05-deployment-sevenhour/`; its historical source hashes and results are preserved.

Three additions connect the existing explanations to concrete diagnostics:

- The integrated article adds `#action-metric`: affine decoding, normalized-coordinate error and physical-tolerance error, including the chain-rule gradient and consistent m/mm conversion. The two-coordinate fixture uses fixed local rotation coordinates, not a global pose distance.
- The residual-RL article calculates the mean after total-action clipping: nominal 0.9 with equal-probability ±0.2 exploration yields executed actions 0.7 and 1.0, mean 0.85 and mean correction −0.05. This is a constructed two-point distribution, not the original robot noise experiment.
- The contact-calibration article adds `#multimodal-mean`: two equally weighted point-contact hypotheses at opposite walls satisfy the event, while their mean lies in the gap. The 1D construction separates a mean-pose consistency test from compatibility of the full candidate distribution.

The existing standard-library helpers now check 200 additional metric derivatives, 100 unit conversions, 126 exact rational clipped-noise expectations and 30 Decimal contact factors. The existing browser arithmetic comparison still passes 2992 cases / 126310 values / 14 invalid inputs. No browser widget, shared style or figure was changed.

`validation.json` binds the article/helper/editorial-metadata source hashes to the fresh production build. `content.json`, `filter.json`, `assets.json` and `artifact-integrity.json` retain strict content/math/snippet validation, exact public inventory, pinned assets and 18 source/build/HTTP byte comparisons. `preservation.json` checks that all prior anchor IDs, publication dates and URLs survive. `acceptance-coverage.json` adds three mapped questions to the preceding 58; the article inventories now contain 23, 17 and 21 questions respectively.

Current preview: `http://127.0.0.1:46087`, serving `/tmp/chase-deployment-followup-production-ryofdnr8` (quiet HTTP server session63070). Browser drivers run serially on private CDP9337. The focused chapter driver covers both native fragment behavior and the mobile enhancement. Its initial assertion incorrectly expected desktop links to apply the mobile focus policy; that review-driver assumption was corrected without changing site navigation.

All numerical examples are constructed calculations. They do not execute a trained policy, hardware queues or servo loops, and do not reproduce the papers' task success rates. The previous paper-version qualifications, including contact Eq7 and TableII discrepancies, remain intact.

Final checks pass: 103 public articles, 739 site acceptance entries, 61 self-check question mappings, zero strict content/math/snippet errors, 162 reading layouts plus 54 final contact layouts, 24 new-chapter keyboard cases and 18 ordinary plus 18 forced-color print layouts at 200% text. Current-source native evidence combines 42 exports / 1624 pages / 126 intact result pairs. Raster diagnostics find no horizontal outside text or affected edges with painted ink; Type3 vertical metric overhang is recorded separately. Three selected additions were also inspected directly in large-text PDFs. Source/build/HTTP matches all 18 article originals/downloads.
