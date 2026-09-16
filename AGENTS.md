# Repository Guidelines

## Project Structure & Module Organization

This repository is a Hugo site currently using the Hextra theme, as configured in `config.yaml`. Blog content lives under `content/`; most posts use a page bundle such as `content/posts/python/tqdm/index.md`, with related images stored beside the article. Reusable templates and overrides belong in `layouts/`, custom styles in `assets/css/`, translations in `i18n/`, and new-content defaults in `archetypes/default.md`. Theme directories are Git submodules; avoid editing them directly when an override in `layouts/` or `assets/` will work. Hugo generates `public/`, which is ignored and must not be committed.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` fetches the registered theme submodules after cloning.
- `hugo server -D` starts a local preview and includes draft posts.
- `hugo --minify` performs the production build used by GitHub Actions.
- `hugo --minify --baseURL "http://localhost/"` is useful for checking production-style output locally.
- `./deploy.sh` builds and pushes `public/` to `gh-pages`; use it only when intentionally performing a manual deployment. Normal pushes to `main` trigger GitHub Actions.

There is no unit-test suite. Treat a clean Hugo production build as the minimum validation for every change, and inspect changed pages in the local server.

## Coding Style & Naming Conventions

Use two-space indentation in YAML and preserve the existing front matter format. Write Markdown with descriptive headings and fenced code blocks that specify a language. Keep post bundles lowercase and topic-oriented; use hyphens for multiword names, for example `content/posts/python/new-feature/index.md`. Store post-specific media beside `index.md` and reference it relatively. Put theme customizations in repository-owned overrides rather than changing the submodule.

## Article Illustrations and Captions

- Use `content/posts/ai/transformer-attention/assets/` as the visual reference for new technical diagrams. Inspect relevant examples before creating assets, especially `training-lifecycle.webp`, `transformer-block-overview.webp`, and `training-vs-kv-cache-inference.webp`.
- Favor white backgrounds, thin rounded outlines, restrained pastel blue/purple/orange/green fills, clear arrows, generous spacing, and legible sans-serif labels. Keep color meanings consistent within an article; do not rely on color alone to convey meaning.
- Prefer Imagegen for new conceptual illustrations, architecture diagrams, and flowcharts in this style. Use deterministic plotting tools for measured data, quantitative charts, and exact numerical comparisons. Preserve useful existing assets unless a replacement improves the explanation.
- Use short labels inside diagrams; English labels are appropriate when clearer, with Chinese explanations in the surrounding article. Keep essential explanations in selectable text so the article remains readable on narrow screens and accessible without images.
- Article prose, captions, alt text, and reader-facing metadata should explain the subject directly. Do not add production commentary such as “Imagegen 生成”, “AI 生成图”, “教学图”, “教学示意图”, “原创概念图”, or boilerplate such as “不是论文原图”. Do not describe which other article supplied the visual style.
- Preserve meaningful scientific context: cite borrowed figures, identify measured versus illustrative values when relevant, and state assumptions, omitted branches, units, and simplifications. Removing production commentary must not turn an illustration into a claimed experimental result.
- Keep generation prompts and asset provenance in `docs/image-generation/`, outside published page bundles. Do not insert them into article text or captions.
- Store article images in the page bundle's `assets/` directory. Prefer the existing `post-image` shortcode and `article-figure` markup with numbered captions; use descriptive alt text, responsive image processing, and image zoom for dense diagrams. Do not edit shared CSS solely to style one figure when the existing components suffice.
- Before accepting a diagram, inspect every label and arrow against the explanation or source code. Check training/inference branches, tensor shapes, time alignment, units, numeric examples, legends, and figure references. Regenerate or edit misleading connections rather than explaining away an incorrect diagram.
- Validate image loading, zoom, captions, formulas, navigation, and horizontal overflow on desktop and mobile in both light and dark themes. Run the production Hugo build after integration.

## Testing Guidelines

When adding an article, update `data/blog_topics.json` and `docs/blog-editorial-review.json` so the learning path, neighboring-post links, filter inventory, and acceptance checks include it. Preserve existing records and follow the metadata format used by recent additions.

Before submitting, run `hugo --minify` and confirm there are no broken shortcode, front matter, or template errors. For layout or CSS changes, check desktop and narrow mobile widths, light and dark themes, navigation, code blocks, and image loading. Verify new drafts with `hugo server -D` before changing `draft` to `false`.

For release validation, build into a fresh temporary directory with `hugo --minify --destination <fresh-directory>`, then run `python -B scripts/validate_blog.py --public <fresh-directory> --python-snippets --structured-snippets`. A reused `public/` can contain stale draft or preview pages and produce misleading validation failures. Use `python -B` when importing article scripts so bytecode is not added to published page bundles.

## Commit & Pull Request Guidelines

History uses short, imperative summaries such as `add some docs` and `add calibration posts`. Follow that pattern, but make the scope specific: `add A* introduction diagrams` or `fix mobile TOC spacing`. Keep generated output out of commits. Pull requests should summarize the affected pages, note the local build result, link relevant issues, and include before/after screenshots for visual changes. Call out configuration, workflow, or submodule updates explicitly.
