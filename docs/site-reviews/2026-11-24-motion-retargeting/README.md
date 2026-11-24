# Motion retargeting article checks

The checks cover the GMR, UMR and OTRetarget overview and its OmniRetarget and HOI-Retarget companion.

## Arithmetic and site output

```sh
python3 -B content/posts/robotics/motion-retargeting/audit_motion_time.py check
python3 -B content/posts/robotics/contact-preserving-retargeting/contact_window.py --check
python3 -B content/posts/robotics/contact-preserving-retargeting/contact_audit.py --check
python3 -B scripts/test_hoi_release_pairs.py
node scripts/test_contact_window.cjs
python3 -B scripts/render_retargeting_figures.py --check
```

The Python window example enumerates feasible position histories to check its closed-form result. The Node check independently propagates reachable sets backward from the terminal position and compares the browser calculation and the static default fixture. `python3 -B scripts/render_retargeting_figures.py --render` regenerates the four WebP images with Matplotlib and Pillow.

`contact_audit.py` compares distance, surface shift, and active anchor error on a local plane. Its checks cover tangential drift, late release, penetration, rotated and translated object frames, empty supports, and duplicate timestamps.

`audit_motion_time.py` checks source indices and spans for the GMR, UMR and FRDV examples. Forty synthetic SMPL-X inputs matched the timestamp mapping against the verified adapter's interpolated position coordinates. The two original UMR Character NPZ exports recovered 8 and 78 source timestamps from their recorded indices. The FRDV export recovered 60 source timestamps from its BVH and the frame range recorded in its config. The paired HOI CLI test checks sign reversal, quaternion sign equivalence, motion drift after the first frame, and rejection of incompatible or truncated rows.

The browser driver uses a fresh production build served over loopback and a private Chrome CDP instance. Run it serially with other browser checks. It needs Node with native WebSocket support and Poppler's `pdfinfo`, `pdftotext` and `pdfimages`.

```sh
node docs/site-reviews/2026-11-24-motion-retargeting/check_articles.cjs \
  <baseURL> <CDP-port> <output-directory> [/path/to/axe.min.js]
```

The driver checks narrow reading and enlarged text, three themes, native and enhanced modes, discovery through both language entrances, keyboard chapter and figure navigation, the calculator transition from its static defaults, and A4 and Letter PDF text and state. The optional argument enables the repository's pinned axe-core 4.14.0 audit. PDF checks verify that each figure stays with its caption and each calculator label stays with its value.

Chrome touch emulation covered tapping and editing the numeric input, submitting by touch, and dragging an overflowing article table at 390 pixels and at 320 pixels with 200% text. The article suite passed at the root and at `/study/`, with 44 reading profiles and five tagged PDFs in each location. The four figures had a unique dialog target, and the open dialogs exposed the accessible name `图片查看`. Escape returned focus to each opener. The two new links from CARD and Video to Data landed on visible chapter headings at 320 pixels. A raster comparison isolated the removal of the active input's caret from the printed A4 form. The root release validation covered 363 HTML pages and all six downloads matched the bundle source bytes. An earlier full-site prefix scan found a root-relative stylesheet path in a standalone lab.

After adding the timing and paired-data examples, the root and prefix article checks passed again. At 320 pixels and 200% text, the six in-body entry links reached visible chapters with scripts disabled. Each build preserved the previous publication dates and heading IDs, and all seven downloads matched the bundle bytes. The root validator covered 41,526 local references without snippet or formula errors. Manual A4 inspection covered the FRDV commands and the paired alignment table.

Firefox 157.0 was also checked at 320, 768 and 1280 CSS pixels for both articles, including enlarged text and keyboard calculator input. Its A4 export displayed the Chinese labels and values. On this Linux environment, Firefox's Cairo PDF export lost some Chinese text on extraction with the installed Noto CJK fonts. A standalone Chinese paragraph produced the same extraction issue. Chromium's tagged PDF text checks covered both articles.

## Pinned upstream probes

The article downloads contain the expected digests for their source files. Use the fixed Git revisions linked in the articles. The probes accept the roots of checkouts with those files and create temporary synthetic fixtures.

The local Git trees resolved 76 distinct fixed-revision links to GMR, UMR, lucidrains/OTRetarget, the HOI-Retarget code and Holosoma.

| Probe | Upstream revision | Coverage |
| --- | --- | --- |
| `probe_motion_sources.py --gmr` | `25f56db79ac411258b377b6d4ccba0c8ff3372ac` | Real CPU Mink/MuJoCo single-joint solve, G1 XML name mapping, synthetic SMPL-X timing and a recorded export path |
| `probe_motion_sources.py --umr` | `d1cdaedaf67a827198da71c5fa831f48b2e51f91` | NumPy cache metadata and compatibility rules |
| `probe_motion_sources.py --umr … --check-umr-training` | Same UMR revision | Two short CPU training runs on synthetic meshes with PyTorch, including batch-size modes, neighbor counts and slot output |
| `probe_motion_sources.py --umr … --check-umr-character` | Same UMR revision | Character frame selection and a 78-frame FK comparison against MuJoCo with literal hinges and rotation-vector groups |
| `probe_motion_sources.py --ot` | `e6e87b85435439b2fa330f2fa0878e8c9dd81ca8` | CPU PyTorch transport shape, marginals and underflow, plus a SciPy SLSQP callback for fixed, movable and height-limited box targets |
| `probe_contact_sources.py --holosoma` | `c9882fcefe195d8ec0549bd44c2f2853f8084735` | NumPy/SciPy/trimesh helpers for stance, tetrahedral adjacency and rigid transforms |
| `probe_contact_sources.py --hoi` | `b341dadde3fc745ef5ea606eaacbde9a0cbb916f` | Standard-library window planning and failure replay using a recording solver, plus the difference stencil |
| `probe_contact_sources.py --hoi … --check-hoi-solver` | Same HOI-Retarget revision | CasADi 3.7.2 and IPOPT run the extracted window on a synthetic slider, checked against an analytic soft-contact solution |

The GMR probe was run with Mink 1.3.0, MuJoCo 3.13.0, DAQP 0.9.1, qpsolvers 4.13.0, NumPy 2.2.6 and SciPy 1.15.3. The OTRetarget example was checked with CPU PyTorch at lucidrains revision `e6e87b85435439b2fa330f2fa0878e8c9dd81ca8`, including the four upstream tests. These checks exercise individual implementation paths with small inputs. Published benchmark numbers, full dataset pipelines and robot performance are sourced to the papers.

A separate GMR BVH replay used the pinned parser, G1 task configuration and solver with the first 60 frames of FRDV's sample at `72298bc095b2ed5fe0fe94933c795668e5bf88af`. The temporary MJCF retained the kinematic tree and inertials after mesh elements were removed. A recording viewer received 60 calls. The CLI wrote the first 59, including 29 finite joint values per frame and unit quaternions in `xyzw` order.

The pinned FRDV CLI ran on the same BVH interval with its default LAFAN1/G1 configuration, one worker, and MuJoCo Menagerie `eea4c7431b63da8278b0ff31638e0b0e941cec0e`. The resolved G1 XML and all 35 mesh files matched Git blob hashes. With CPython 3.10, MuJoCo 3.13.0, NumPy 2.2.6 and the native GCC 12.3 solver, the run saved a `(60, 36)` pose matrix at 30.000300003 FPS, 16 tracked keypoints, and 30 joint names including the floating base. The first and last sample times are 0 and 1.966647 s. The fixed repository release contains the kinematic pipeline, and the paper describes a separate dynamics-validation stage.

Verified source from SBTO `cbed4ede` was replayed on a constructed NPZ and two free-joint vectors. Its loader read 29.97 FPS as 29, moving the timestamp of sample 299 by 0.333702 s. The extracted field-swap function preserved all three translations and all four quaternion components in the target order. This replay covers preprocessing of a synthetic reference and does not run the trajectory optimizer.

UMR's original Character CLI ran with the downloadable configuration in an isolated copy of the fixed source and mesh files. The run used Python 3.12, CPU execution of PyTorch 2.13.0+cu130, NumPy 2.5.1, SciPy 1.18.0, MuJoCo 3.3.7 and Clarabel 0.11.1. The 8-frame output had 36 position columns and matched source indices 12 through 26 at stride 2. Changing the config to stride 1 on the full sequence caused a cache hit until `--force-retarget` was supplied. The subsequent output covered all 78 frames with the same correspondence file hash. Both outputs had finite joint positions and unit base quaternions. Their matched-index maximum translation difference was 0.0932 m in this small CPU configuration.

## Public HOI-Retarget output

The article's Dataset Viewer requests read two OMOMO rows matching `omomo/sub10_clothesstand_000` and the first CoRoleHOI G1 pair from dataset revision `b5ac8125e497e6735d6314e63a0e1eedf6229336`. Each request returned the complete trajectory cells. The downloadable `inspect_hoi_release.py` reads one row at a time and optionally evaluates labelled link-to-anchor errors with its corresponding URDF. G1 and H2 model SHA-256 digests are `3afed5cf263ce3afb7ae2efdd50483093a70af0a65198dda8fb338db5f85a53f` and `6e6809e37b49462e0199ba10f00b5faae110f1129c1d8ae60f615b81a9edbd0e`.

Pinocchio 4.1.0 and MuJoCo 3.13.0 reproduced the same contact-distance summaries. The MuJoCo run kept the URDF kinematic tree and fixed body frames, computed link positions in the base frame, and composed them with the separately decoded floating-base pose. Each embodiment was evaluated on 100 labelled right-palm samples. The other links had empty support and yield `null`. G1's mean of 0.965 mm rose to 76.805 mm when the published anchor was scaled again, and to 188.779 mm when the unadapted object position was used. CoRoleHOI's checked G1 pair differs by a constant translation to within 0.1 µm per frame, with opposite stored vertical offsets and identical rotations up to quaternion sign.

The same CoRoleHOI event has reciprocal H2 rows at indices 456 and 457. The `--partner` comparison recovered a fixed translation from the first sample and checked it against the other 189 samples. The world translation from the second row to the first is approximately `(0.516, 2.959, 21.568)` mm for H2 and `(0.395, 2.265, 16.508)` mm for G1. The maximum positional residuals are 0.124 µm and 0.085 µm. The object rotations agree up to quaternion sign. A separate Pinocchio/SciPy replay measured 254 active link samples per robot. Translating their bases and objects together preserved every object-frame position to numerical precision. Translating the object alone shifted those positions by 21.776 mm for H2 and 16.667 mm for G1.

PyArrow projected eight metadata columns from the two original CoRoleHOI Parquet files at the same release revision. HTTP range reads transferred under 170,000 bytes and skipped trajectory and video columns. The 456 rows per robot formed 228 reciprocal pairs. Every flag agreed with `abs(partner_z_offset_cm) <= 3`. Both embodiments passed that height test in 120 events, only G1 in 14, and both exceeded the threshold in 94. This count uses the published height field.

Pass `--object-mesh clothesstand.obj --mesh-scale 0.83` on the G1 row or `--object-mesh clothesstand.obj` on the H2 row to check distances to the published triangles. The OBJ SHA-256 is `60a20e223309f3f85bdcbb270a6eaaef657fbe90a83339795059395b7ee8a46d`. Its URDF gives identity geometry transforms. The mesh is open, so the check uses unsigned nearest-triangle distances. The palm means are 7.1965 and 8.7972 mm. Each variant gave the same count on the palm positions and the labelled anchor positions when tested against 20 mm, with 11 frames on G1 and 13 on H2.
