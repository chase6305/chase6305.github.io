# cuRoboV2-Fast article review

Paper: arXiv `2611.06913v1`, 2026-11-04, CC BY 4.0. Source API and implementation: NVlabs/curobo commit `b29a8c2ee6869585bb05b056b9ab53b9c0753106`. The older local checkout (`8e734f3...`) was not used as evidence for the Fast API. Its untracked `franka_custom.yml` was preserved.

## Checks

- `python3 -B content/posts/robotics/curobo-v2-fast/timeopt_sanity.py --check`: independent Decimal differentiation of the analytic inverse time map, three-call torque coefficients, Simpson integration and a static payload counterexample. No third-party packages required.
- `python3 -B scripts/render_curobo_timeopt_figures.py --check`: verify the deterministic SVGs remain identical to the constructed example.
- GPU probe: `python -B docs/site-reviews/2026-11-05-curobo-fast/probe_gpu.py --source-root <pinned-checkout> --output <fresh-directory>`. Uses an existing CUDA environment and isolated output/cache. It has a main guard, keeps CUDA graphs enabled and explicitly releases solver resources.
- GPU environment: RTX 5090 D v2, driver 595.99.02, Python 3.11.15, PyTorch 2.13.0+cu130, Warp 1.16.0, cuda-core 1.1.1, cuda-bindings 13.3.1, NumPy 2.4.6, SciPy 1.17.1. This is one working smoke configuration, not a dependency lock or a paper benchmark reproduction.
- `gpu-source-results.json` records the single-query result. Seven actuated joints become nine output columns after the locked finger joints are restored. Output joint names match the returned columns. Solver-reported event time and synchronized first-query wall time are deliberately kept separate.
- Browser review: 320/1440 px, light/dark, 100%/200% root text, all image loads, captions, formulas, image dialog open/close and native fallback passed. Every label in both numerical SVGs fits its viewBox; labels and curves were also visually reviewed.

## Scope

The scalar example is a positive-speed interior segment without start/stop or actuator constraints; it is not a TimeOpt implementation. The GPU query exercises the official Franka scene with RRT and TimeOpt configured. It does not compare a full benchmark, test payload accuracy, prove continuous-time collision freedom or connect to a physical robot. The two model-derived SVGs contain constructed data; paper performance numbers are separately attributed in the article.

Native print review completed six A4 exports across light/dark/warm themes at 100%/200% root text. All exported text stayed inside PDF page bounds. Production validation, filter inventory and TOC fixtures passed with the new article included.
