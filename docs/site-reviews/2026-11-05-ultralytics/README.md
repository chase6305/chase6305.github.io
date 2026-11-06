# Ultralytics article review

Requested review window: 2026-11-05 06:26:03–09:26:03 UTC (14:26–17:26 China Standard Time).

Source: `/home/ubuntu/workspace/3rdparty/ultralytics`, version `8.4.202`, commit `1a767da5e58467ab8fbc2407b600b7632033dbc5`. Preserve the article URL, date and published state. All review environments and model/dataset outputs live outside the site repository.

## First audit

- The default `nms=None` selects the ordinary head in predictor, validator and most export paths; `nms=False` selects an available end-to-end head. Verify the effective backend rather than reading only the architecture YAML.
- `v8DetectionLoss.loss_names` explicitly selects `l1_loss` when `reg_max<=1`; the earlier prose about a lingering DFL log label needs correction.
- CUDA inference may prefetch image/video batches. `stream=True` controls result materialization, while source buffering and prefetching have separate behavior.
- `resume=True` on a non-resumable checkpoint warns and starts a new run in this checkout. Explain the fallback so readers cannot mistake it for a restored optimizer history.
- Detection fitness in this commit weights `mAP50-95` alone. It is worth stating the concrete rule rather than referring only to generic fitness.

## Validation boundaries

The initial CPU smoke checks used random initialization, not pretrained accuracy evaluation. They checked model construction, result shapes, centered LetterBox geometry and NMS/head selection. They do not measure training quality, GPU speed, multi-rank training or device deployment. Later runs must identify their data, initialization, backend and scope explicitly.

## Reproducible checks added during review

- `python3 -B content/posts/ai/ultralytics/detection_geometry.py --check` checks fixed examples, round trips, integer padding, IoU and NMS with no third-party packages.
- `python3 -B scripts/render_ultralytics_geometry.py --check` checks the two quantitative SVGs against the published example.
- `node scripts/test_ultralytics_geometry.cjs` compares 372 configurations with independent Python arithmetic, including thin images and half-to-even rounding. Run outside a sandbox that cannot support the Node→Python child process.
- `check_geometry.cjs <baseURL> <CDP-port> <output> [axe.min.js]` checks native controls, invalid-input clearing, focus, contrast, three themes, larger text/spacing, forced colors and native/setup fallbacks. Axe covers valid and invalid states.
- `check_print.cjs <baseURL> <CDP-port> <output>` exports native A4/Letter PDFs in three themes and two text sizes, verifies all 96 result label/value pairs stay together, and checks a forced-colors-independent print palette.
- `probe_cpu.py --source-root <pinned-checkout> --output <new-directory> [--training] [--onnx --weights <existing-yolo26n.pt>]` reproduces the geometry, gradient, fresh-process checkpoint recovery and CPU ONNX probes. It does not download weights or install packages. See `source-results.json` for the exercised dependency versions and numerical scope.

The CPU model probe was rerun through the committed driver in a fresh output directory. Both ordinary and end-to-end outputs matched within the documented tolerances on the shipped image. Synthetic training deliberately checks state transitions rather than accuracy. Root Python installation metadata may name a different installed Ultralytics package; the driver validates the checkout revision and imports that checkout explicitly.

A native PDF review found inherited definition-list indentation inside result cards; the scoped styles now reset the card values' padding and margins. A4/Letter exports passed after that correction. Diagrams remain vector shapes in the native PDF.

## Final integration status

The repository-wide `check_site_reflow.cjs --all-articles` run completed 1,980 layouts with zero failures and zero runtime exceptions on the draft-inclusive preview. It started before the additional cuRobo article was registered; that new article received its own mobile/desktop, large-text, formula/image and native-PDF checks.

The final fresh production build and `validate_blog.py --python-snippets --structured-snippets` pass with both new articles registered (100 reviewed entries, 97 published article schemas). Filtering and TOC fixture checks pass. No shared template, image viewer or theme submodule was modified.
