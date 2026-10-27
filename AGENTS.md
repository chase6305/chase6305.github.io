# Repository Guidelines

## Project Structure & Module Organization

This repository is a Hugo site currently using the Hextra theme, as configured in `config.yaml`. Blog content lives under `content/`; most posts use a page bundle such as `content/posts/python/tqdm/index.md`, with related images stored beside the article. Reusable templates and overrides belong in `layouts/`, custom styles in `assets/css/`, translations in `i18n/`, and new-content defaults in `archetypes/default.md`. Theme directories are Git submodules; avoid editing them directly when an override in `layouts/` or `assets/` will work. Hugo generates `public/`, which is ignored and must not be committed.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` fetches the registered theme submodules after cloning.
- `hugo server -D` starts a local preview and includes draft posts.
- `hugo --minify` performs the production build used by GitHub Actions.
- `hugo --minify --baseURL "http://localhost/"` is useful for checking production-style output locally.
- `./deploy.sh` builds and pushes `public/` to `gh-pages`; use it only when intentionally performing a manual deployment. Normal pushes to `main` trigger GitHub Actions.

Treat a clean Hugo production build as the minimum validation for every change, and inspect changed pages in the local server. `scripts/validate_blog.py` checks site output and snippet syntax; `scripts/test_blog_filter.cjs` checks article filtering. Topic-specific example checks live in `scripts/test_blog_*.py` and article bundles. Run the checks relevant to changed code; optional robotics and ML dependencies should be installed in a separate environment. Browser checks use an existing loopback Chrome CDP endpoint and should run serially because keyboard focus is shared.

For heading or table-of-contents changes, run `python3 -B scripts/test_blog_toc.py` (use `--hugo /path/to/hugo` if needed). It builds disposable fixtures covering formulas, escaped dollar signs, code spans, linked labels and Unicode anchors. TOC formulas must reuse Markdown's rendered headings rather than guessing from dollar signs in plain text.

For chapter navigation behavior, run `node scripts/check_chapter_focus.cjs <baseURL> <CDP-port> <output-directory>`. It checks immediate focus, later keyboard input, modifier clicks, legacy span anchors and native fragment Back/Forward behavior at narrow widths and large text. Add `--all-posts --drafts` on a draft-inclusive preview to activate every article chapter link at 320/768 pixels and 100%/200% font sizes.

For changes to the shared image partial, run `python3 -B scripts/test_blog_images.py --hugo /path/to/hugo`. Article images should reserve their aspect ratio before lazy loading; preserve small images' intrinsic width and read SVG dimensions from `viewBox` when available. For reading-layout changes, `node scripts/check_blog_reading.cjs <baseURL> <CDP-port> <output-directory> --drafts` checks a representative chapter deep link in every article at 320/1440 pixels, with and without JavaScript. Add `--figures` to cover all numbered figure anchors and their image/table stability. Use a draft-inclusive build with `--drafts`; omit the flag for a production build. Keep published URLs, existing chapter anchors, original publication dates and draft states stable during editorial revisions.

For navigation, appearance or image zoom changes, run `node scripts/check_site_accessibility.cjs <baseURL> <CDP-port> <output-directory>` serially with other browser checks. It covers five viewport widths, three themes, mobile menu focus, theme keyboard controls, blocked storage, keyboard zoom and reduced motion. `python3 -B scripts/test_site_assets.py --public <fresh-directory>` verifies pinned browser assets, local formula fonts and emitted integrity. Keep vendored versions, licenses and `docs/vendor-assets.json` in sync; formula fonts must match Hugo's embedded KaTeX version.

For table or formula overflow behavior, run `node scripts/check_overflow_focus.cjs <baseURL> <CDP-port> <output-directory>` and `node scripts/check_blog_math.cjs <baseURL> <CDP-port> <output-directory>`. Reflow must preserve an active region's focus, hide obsolete scroll hints immediately, remove a fitting region's temporary tab stop after blur, and restore keyboard access if it overflows again.

For repository-owned keyboard outline styles, run `node scripts/check_site_focus.cjs <baseURL> <CDP-port> <output-directory> --forced-colors`. It measures painted outlines against their surrounding surfaces in three themes, normal colors and light/dark forced-color palettes. Gradient or image backgrounds still need visual review.

For image viewer changes, run `node scripts/check_image_viewer.cjs <fresh-public-directory> <CDP-port> <output-directory>` serially with other browser suites. It serves controlled delayed and failed originals to verify immediate previews, cancellation, retries, modal focus, small images, short screens, localized labels and printing during downloads. Keep native original-image links available when JavaScript or dialog support is unavailable. `node scripts/check_blog_image_viewers.cjs <baseURL> <CDP-port> <output-directory> --drafts` covers every article image at desktop/mobile widths in light and dark themes, including focus and scroll restoration.

For search changes, run `node scripts/test_site_search.cjs` and `node scripts/check_site_search.cjs <baseURL> <CDP-port> <output-directory>`. Both language entrances share the canonical blog collection; verify the Chinese home cards and search as well as the default-language pages. Keep theme override provenance and its published license in `docs/theme-overrides.json` in sync with theme upgrades.

For text sizing, spacing or sticky navigation changes, run `node scripts/check_site_reflow.cjs <baseURL> <CDP-port> <output-directory> --all-articles` against a draft-inclusive build. It checks 200% root text sizing, increased text spacing and their combination with and without JavaScript, plus actual arrow-key scrolling of enlarged tables. `node scripts/check_site_readability.cjs <baseURL> <CDP-port> <output-directory> --all-articles` checks text and placeholder contrast, print disclosure restoration and native canvas PDF colors. These browser suites also run serially.

For code-copy or code-block changes, run `node scripts/check_code_reading.cjs <baseURL> <CDP-port> <output-directory>` to check overflow focus after text sizing and disclosure changes, exact asynchronous keyboard copying and focus after tabbing away.

Copy failure checks must also preserve an existing selection and background-tab focus; a short manually selected block keeps focus until blur, then drops its temporary tab stop. Keep pending controls focusable and reject duplicate activations without native disabling.

For print styles or long-formula revisions, run `node scripts/check_blog_print.cjs <baseURL> <CDP-port> <output-directory> --drafts` on the draft-inclusive build. It checks A4 content width, code wrapping, formula/table bounds, printed text contrast, disclosures and screen-state restoration in three themes, with and without JavaScript. Keep canvas print palettes independent of screen themes and restore them after printing.

Repeat print checks with `--forced-colors` after changing print palettes. `node scripts/check_filter_colors.cjs <baseURL> <CDP-port> <output-directory>` checks actual canvas text/curve colors, line patterns, selected views, media-only redraws and print restoration for the filtering lab.

For portable C++17, CMake, Modbus protocol and loopback TCP/UDP examples, run `python3 -B scripts/test_blog_examples.py --only cpp cmake modbus network`. These groups use temporary build directories, a simulated serial port and loopback-only sockets; they do not require NumPy, Matplotlib or Pinocchio. The optional `python` group still needs its robotics/plotting environment.

For the tqdm article, run `python -B scripts/test_tqdm_examples.py` in a dedicated environment with `tqdm`. CI installs the wheel pinned by version and hash in `scripts/requirements-progress.txt`; the checks execute published snippets on one-shot iterators, failures, non-TTY streams, partial byte chunks and optimized Python.

## Coding Style & Naming Conventions

Use two-space indentation in YAML and preserve the existing front matter format. Write Markdown with descriptive headings and fenced code blocks that specify a language. Keep post bundles lowercase and topic-oriented; use hyphens for multiword names, for example `content/posts/python/new-feature/index.md`. Store post-specific media beside `index.md` and reference it relatively. Put theme customizations in repository-owned overrides rather than changing the submodule.

Check explanatory tables at 320 pixels wide. If long sentences collapse into near-vertical text, add `{.table-readable}` immediately after that Markdown table to use the existing minimum column widths and horizontal scrolling. Keep compact numerical tables compact; select tables based on their rendered content. Verify that an overflowing table can receive keyboard focus and scroll with the arrow keys.

## Article Illustrations and Captions

- Use `content/posts/ai/transformer-attention/assets/` as the visual reference for new technical diagrams. Inspect relevant examples before creating assets, especially `training-lifecycle.webp`, `transformer-block-overview.webp`, and `training-vs-kv-cache-inference.webp`.
- Favor white backgrounds, thin rounded outlines, restrained pastel blue/purple/orange/green fills, clear arrows, generous spacing, and legible sans-serif labels. Keep color meanings consistent within an article; do not rely on color alone to convey meaning.
- Prefer Imagegen for new conceptual illustrations, architecture diagrams, and flowcharts in this style. Use deterministic plotting tools for measured data, quantitative charts, and exact numerical comparisons. Preserve useful existing assets unless a replacement improves the explanation.
- Use short labels inside diagrams; English labels are appropriate when clearer, with Chinese explanations in the surrounding article. Keep essential explanations in selectable text so the article remains readable on narrow screens and accessible without images.
- Article prose, captions, alt text, and reader-facing metadata should explain the subject directly. Do not add production commentary such as “Imagegen 生成”, “AI 生成图”, “教学图”, “教学示意图”, “原创概念图”, or boilerplate such as “不是论文原图”. Do not describe which other article supplied the visual style.
- Preserve meaningful scientific context: cite borrowed figures, identify measured versus illustrative values when relevant, and state assumptions, omitted branches, units, and simplifications. Removing production commentary must not turn an illustration into a claimed experimental result.
- Keep generation prompts and asset provenance in `docs/image-generation/`, outside published page bundles. Do not insert them into article text or captions.
- Store article images in the page bundle's `assets/` directory. Prefer the existing `post-image` shortcode and `article-figure` markup with numbered captions; use descriptive alt text, responsive image processing, and image zoom for dense diagrams. Do not edit shared CSS solely to style one figure when the existing components suffice.
- Inside `article-figure`, wrap the caption number in `span.article-figure__number` and its explanation in `span.article-figure__text`. The caption uses a two-column grid; bare caption text can be clipped even when the page itself has no horizontal overflow. Inspect the full caption on a narrow viewport.
- Keep figure numbers unique within each article. When inserting a figure, update later captions and textual references; `scripts/validate_blog.py` rejects duplicate rendered figure labels.
- Before accepting a diagram, inspect every label and arrow against the explanation or source code. Check training/inference branches, tensor shapes, time alignment, units, numeric examples, legends, and figure references. Regenerate or edit misleading connections rather than explaining away an incorrect diagram.
- Validate image loading, zoom, captions, formulas, navigation, and horizontal overflow on desktop and mobile in both light and dark themes. Run the production Hugo build after integration.

## Testing Guidelines

When adding an article, update `data/blog_topics.json` and `docs/blog-editorial-review.json` so the learning path, neighboring-post links, filter inventory, and acceptance checks include it. Preserve existing records and follow the metadata format used by recent additions.

When adding tags, assign each exact tag spelling to one field in `data/tag_groups.yaml`. Keep existing tag URLs stable; the public tag index should contain every active tag exactly once, with no unclassified group or empty sections.

Before submitting, run `hugo --minify` and confirm there are no broken shortcode, front matter, or template errors. For layout or CSS changes, check desktop and narrow mobile widths, light and dark themes, navigation, code blocks, and image loading. Verify new drafts with `hugo server -D` before changing `draft` to `false`.

For release validation, build into a fresh temporary directory with `hugo --minify --destination <fresh-directory>`, then run `python -B scripts/validate_blog.py --public <fresh-directory> --python-snippets --structured-snippets`. A reused `public/` can contain stale draft or preview pages and produce misleading validation failures. Use `python -B` when importing article scripts so bytecode is not added to published page bundles.

## Commit & Pull Request Guidelines

History uses short, imperative summaries such as `add some docs` and `add calibration posts`. Follow that pattern, but make the scope specific: `add A* introduction diagrams` or `fix mobile TOC spacing`. Keep generated output out of commits. Pull requests should summarize the affected pages, note the local build result, link relevant issues, and include before/after screenshots for visual changes. Call out configuration, workflow, or submodule updates explicitly.
