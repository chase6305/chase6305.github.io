#!/usr/bin/env bash
# Optional manual gh-pages publication. Normal main pushes use GitHub Actions.
set -euo pipefail

if (( $# > 1 )); then
  printf '用法: %s [提交说明]\n' "$0" >&2
  exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
hugo_bin="${HUGO_BIN:-hugo}"
commit_msg="${1:-update site $(date '+%Y-%m-%d %H:%M:%S')}"
deploy_temp="$(mktemp -d "${TMPDIR:-/tmp}/chase-site-deploy.XXXXXX")"
trap 'rm -rf -- "$deploy_temp"' EXIT
build_dir="$deploy_temp/build"
checkout_dir="$deploy_temp/checkout"

printf '从全新目录构建生产网站...\n'
"$hugo_bin" --source "$repo_dir" --destination "$build_dir" \
  --minify --environment production --buildDrafts=false
python3 -B "$repo_dir/scripts/validate_blog.py" --public "$build_dir" \
  --python-snippets --structured-snippets --strict-math
python3 -B "$repo_dir/scripts/test_site_assets.py" --public "$build_dir"

# Only contact the publication remote after the complete build is validated.
deploy_remote="${DEPLOY_REMOTE:-$(git -C "$repo_dir" remote get-url origin)}"
git init --quiet --initial-branch=gh-pages "$checkout_dir"
git -C "$checkout_dir" remote add origin "$deploy_remote"
if git -C "$checkout_dir" ls-remote --exit-code --heads origin refs/heads/gh-pages > "$deploy_temp/remote-ref"; then
  git -C "$checkout_dir" fetch --quiet --depth=1 origin refs/heads/gh-pages
  git -C "$checkout_dir" checkout --quiet -B gh-pages FETCH_HEAD
else
  remote_status=$?
  if (( remote_status != 2 )); then
    exit "$remote_status"
  fi
fi

# The private checkout retains branch history but no stale build files.
git -C "$checkout_dir" rm --quiet -r --ignore-unmatch -- .
python3 -B - "$build_dir" "$checkout_dir" <<'PY'
import shutil
import sys
shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('.git'))
PY
touch "$checkout_dir/.nojekyll"
# The validated artifact must not inherit a developer's global ignore patterns.
git -C "$checkout_dir" add --all --force
if git -C "$checkout_dir" diff --cached --quiet; then
  printf '内容未变化，跳过提交和推送。\n'
  exit 0
fi

# Keep repository-local identity settings available in the temporary checkout.
for identity_key in user.name user.email; do
  if identity_value="$(git -C "$repo_dir" config --get "$identity_key")"; then
    git -C "$checkout_dir" config "$identity_key" "$identity_value"
  fi
done
git -C "$checkout_dir" commit --quiet -m "$commit_msg"
git -C "$checkout_dir" push origin HEAD:refs/heads/gh-pages
printf '手动发布分支已推送；实际站点状态请在 GitHub Pages 中核对。\n'
