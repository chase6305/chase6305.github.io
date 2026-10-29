# Verify a published build

Run the normal production and article checks first. Once the Pages workflow for the intended commit has succeeded, compare its fresh build with the published HTTP responses:

```bash
python3 -B scripts/check_site_release.py \
  --public /tmp/chase-production \
  --revision <full-commit-sha> \
  --output /tmp/chase-release.json
```

Build `/tmp/chase-production` from that same commit, with the production base URL, and finish the build before starting the comparison. Keep the directory unchanged until verification finishes. The checker snapshots file hashes before HTTP requests and reads article metadata using `git show` at the resolved commit. Editing the working metadata afterward cannot change that release's expected article or draft inventory. The revision field identifies the supplied source revision; it does not independently prove which sources produced an arbitrary existing build directory.

The default inventory covers every HTML page, JavaScript and CSS file, WOFF2 formula fonts, local downloads linked from HTML, available sitemap/feeds and robots.txt. Known draft routes and two missing-page probes must return the site's 404 body with status 404. It refuses a build that contains a known draft page or omits a published article or linked download.

Add selected assets or all generated images when needed:

```bash
python3 -B scripts/check_site_release.py \
  --public /tmp/chase-production --revision <full-commit-sha> \
  --include posts/planner/task-and-motion-planning/assets/search-feedback.webp \
  --include posts/planner/task-and-motion-planning/tamp-lab.zip \
  --output /tmp/chase-release.json

python3 -B scripts/check_site_release.py \
  --public /tmp/chase-production --revision <full-commit-sha> \
  --images --output /tmp/chase-release-with-images.json
```

The ZIP is already included when linked by an article; `--include` also supports unlinked build-relative files. `--images` can transfer substantially more data. `--base-url` accepts a production origin or a project-site prefix, for example `https://example.github.io/project/`.

Each result records expected status, SHA-256 and byte size, plus every HTTP attempt. Gzip bodies are decompressed before comparison. Redirects are failures; they must not disguise a missing route as a successful page. Stale 200 responses, HTTP errors and connection failures are retried within the configured limit. Defaults are four concurrent requests, a 20-second socket timeout and two additional attempts; `--workers`, `--timeout` and `--retries` set bounded alternatives. The timeout applies to socket operations, not to the entire run. The command honors the environment's HTTP proxy settings.

Exit status 0 means every requested response matched; 1 means at least one HTTP result failed; 2 indicates invalid configuration, metadata or local output. An exhausted connection retry remains a failure in the report. Inspect its attempts before deciding whether to rerun; a TLS interruption is different evidence from a confirmed 404 or persistent hash mismatch. Rerunning produces a new report, so preserve the earlier report when investigating transient failures.

This read-only command does not push, deploy, poll workflow status, validate MIME semantics or open a browser. Byte equality complements the existing HTML, formula, navigation, accessibility, scientific-example and browser checks. It cannot certify robot-model assumptions or replace review of a diagram's labels and arrows.

The regression suite has no third-party dependencies and makes requests only to an ephemeral loopback HTTP server:

```bash
python3 -B scripts/test_site_release.py -v
```

It covers immutable Git metadata and hash snapshots, escaped and case-sensitive paths, project prefixes, missing downloads, draft leakage, external-path rejection, gzip, custom 404 bodies, stale responses, 503 recovery, truncated bodies, redirect rejection, timeouts and ordered concurrent results. CI runs these local fixtures; live production verification remains a separate release step.
