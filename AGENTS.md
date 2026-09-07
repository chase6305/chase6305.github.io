# Repository Guidelines

## Project Structure & Module Organization

This repository is a Hugo site using the PaperMod theme. Blog content lives under `content/`; most posts use a page bundle such as `content/posts/python/tqdm/index.md`, with related images stored beside the article. Site-wide configuration is in `config.yaml`. Reusable templates and overrides belong in `layouts/`, custom styles in `assets/css/`, translations in `i18n/`, and new-content defaults in `archetypes/default.md`. `themes/PaperMod/` is a Git submodule; avoid editing it directly when an override in `layouts/` or `assets/` will work. Hugo generates `public/`, which is ignored and must not be committed.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` fetches the PaperMod theme after cloning.
- `hugo server -D` starts a local preview and includes draft posts.
- `hugo --minify` performs the production build used by GitHub Actions.
- `hugo --minify --baseURL "http://localhost/"` is useful for checking production-style output locally.
- `./deploy.sh` builds and pushes `public/` to `gh-pages`; use it only when intentionally performing a manual deployment. Normal pushes to `main` trigger GitHub Actions.

There is no unit-test suite. Treat a clean Hugo production build as the minimum validation for every change, and inspect changed pages in the local server.

## Coding Style & Naming Conventions

Use two-space indentation in YAML and preserve the existing front matter format. Write Markdown with descriptive headings and fenced code blocks that specify a language. Keep post bundles lowercase and topic-oriented; use hyphens for multiword names, for example `content/posts/python/new-feature/index.md`. Store post-specific media beside `index.md` and reference it relatively. Put theme customizations in repository-owned overrides rather than changing the submodule.

## Testing Guidelines

Before submitting, run `hugo --minify` and confirm there are no broken shortcode, front matter, or template errors. For layout or CSS changes, check desktop and narrow mobile widths, light and dark themes, navigation, code blocks, and image loading. Verify new drafts with `hugo server -D` before changing `draft` to `false`.

## Commit & Pull Request Guidelines

History uses short, imperative summaries such as `add some docs` and `add calibration posts`. Follow that pattern, but make the scope specific: `add A* introduction diagrams` or `fix mobile TOC spacing`. Keep generated output out of commits. Pull requests should summarize the affected pages, note the local build result, link relevant issues, and include before/after screenshots for visual changes. Call out configuration, workflow, or submodule updates explicitly.
