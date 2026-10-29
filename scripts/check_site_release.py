#!/usr/bin/env python3
"""Compare a fresh production build with deployed HTTP bytes; never deploy.

Article metadata comes from a Git commit, independently of later working-tree
edits. Browser behavior, HTML validity and scientific examples have separate
checks. Run only after the intended Pages workflow has completed successfully.
"""
import argparse
import concurrent.futures
from dataclasses import dataclass
import gzip
import hashlib
from http.client import HTTPException
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import zlib
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urljoin, urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = {'.py', '.cpp', '.h', '.txt', '.json', '.zip', '.csv', '.urdf', '.cmake'}
IMAGES = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.svg', '.avif'}


@dataclass(frozen=True)
class Target:
    route: str
    status: int
    sha256: str
    size: int


class References(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.urls = []
        self.feed(source)

    def handle_starttag(self, tag, pairs):
        attrs = dict(pairs)
        if tag == 'a' and attrs.get('href'):
            self.urls.append(attrs['href'])


def commit_metadata(repository, revision):
    """Resolve once, then read only that immutable tree, never the working file."""
    sha = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', '--verify',
        '--end-of-options', revision+'^{commit}'], text=True, timeout=15).strip()
    source = subprocess.check_output(['git', '-C', str(repository), 'show',
        sha+':docs/blog-editorial-review.json'], text=True, timeout=15)
    return sha, json.loads(source)


def normalize_base(value):
    parts = urlsplit(value)
    if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError('Use an HTTP(S) site URL without credentials, query or fragment')
    parts.port  # Validate a supplied port before making any requests.
    if any(piece in ('.', '..') for piece in unquote(parts.path).split('/')):
        raise ValueError('Site URL must not contain dot path segments')
    return value.rstrip('/')+'/'


def article_route(post):
    path = post['path']
    if not path.startswith('content/') or not path.endswith('.md') or '..' in path.split('/'):
        raise ValueError('Invalid article path: '+str(path))
    if type(post['draft']) is not bool:
        raise ValueError('Article draft state must be a boolean')
    return path.removeprefix('content/').removesuffix('/index.md').removesuffix('.md').lower()+'/'


def inventory(public, base, metadata, *, images=False, include=()):
    """Snapshot expected hashes before requests; reject known draft output."""
    public = public.resolve()
    if not (public/'index.html').is_file() or not (public/'404.html').is_file():
        raise ValueError('Production directory needs index.html and 404.html')
    paths = {}
    for post in metadata['posts']:
        relative = article_route(post)
        local = public/relative/'index.html'
        if post['draft'] and local.exists():
            raise ValueError('Production output contains a draft: '+relative)
        if not post['draft'] and not local.is_file():
            raise ValueError('Published article is missing from output: '+relative)
        paths[urljoin(base, quote(relative, safe='/+-._'))] = (404 if post['draft'] else 200, public/'404.html' if post['draft'] else local)

    def add(file):
        if not file.resolve().is_relative_to(public):
            raise ValueError('Asset is outside the production build: '+str(file))
        relative = file.relative_to(public).as_posix()
        route = relative.removesuffix('index.html') if file.name == 'index.html' else relative
        paths[urljoin(base, quote(route, safe='/+-._'))] = (200, file)

    for file in sorted(public.rglob('*')):
        if not file.is_file():
            continue
        if file.suffix.lower() in {'.html', '.js', '.css', '.woff2'} or (images and file.suffix.lower() in IMAGES):
            add(file)
        if file.suffix.lower() == '.html':
            relative = file.relative_to(public).as_posix()
            page_url = urljoin(base, quote(relative, safe='/+-._'))
            for reference in References(file.read_text(encoding='utf-8')).urls:
                parts = urlsplit(urljoin(page_url, reference))
                site = urlsplit(base)
                if (parts.scheme, parts.netloc) != (site.scheme, site.netloc):
                    continue
                prefix = unquote(site.path)
                path = unquote(parts.path)
                if not path.startswith(prefix):
                    continue
                local = (public/path[len(prefix):]).resolve()
                if local.suffix.lower() in DOWNLOADS:
                    if not local.is_relative_to(public) or not local.is_file():
                        raise ValueError('Linked download missing: '+path)
                    add(local)
    for relative in ('robots.txt', 'sitemap.xml', 'index.xml', 'zh-cn/index.xml'):
        if (public/relative).is_file():
            add(public/relative)
    for relative in include:
        parts = urlsplit(relative)
        if parts.scheme or parts.netloc or parts.query or parts.fragment:
            raise ValueError('--include requires a build-relative file path')
        local = (public/unquote(parts.path).lstrip('/')).resolve()
        if not local.is_relative_to(public) or not local.is_file():
            raise ValueError('Included file is outside or absent from the build')
        add(local)
    for relative in ('site-release-check-missing/', 'zh-cn/site-release-check-missing/'):
        if (public/relative).exists():
            raise ValueError('Missing-page fixture conflicts with build output')
        paths[urljoin(base, relative)] = (404, public/'404.html')
    # Decode and encode once so a Unicode route and its HTML-escaped spelling
    # share the same target; keep URL paths case-sensitive for static assets.
    result = {}
    for url, (status, file) in paths.items():
        route = quote(unquote(urlsplit(url).path), safe='/+-._')
        body = file.read_bytes()
        target = Target(route, status, hashlib.sha256(body).hexdigest(), len(body))
        if route in result and result[route] != target:
            raise ValueError('Conflicting expected route: '+route)
        result[route] = target
    return [result[route] for route in sorted(result)]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


_worker = threading.local()


def check_target(target, origin, *, timeout=20., retries=2, delay=.5):
    if not hasattr(_worker, 'opener'):
        _worker.opener = build_opener(NoRedirect())
    attempts = []
    for index in range(retries+1):
        attempt = {'number': index+1}
        try:
            request = Request(origin+target.route, headers={
                'User-Agent': 'Chase site release verification (read-only)',
                'Cache-Control': 'no-cache', 'Accept-Encoding': 'gzip'})
            try:
                response = _worker.opener.open(request, timeout=timeout)
            except HTTPError as error:
                response = error
            with response:
                body = response.read()
                encoding = response.headers.get('Content-Encoding', '').lower().strip()
                if encoding == 'gzip':
                    body = gzip.decompress(body)
                elif encoding not in ('', 'identity'):
                    raise ValueError('Unsupported content encoding: '+encoding)
                actual_hash = hashlib.sha256(body).hexdigest()
                attempt.update(status=response.code, sha256=actual_hash, bytes=len(body),
                    content_type=response.headers.get('Content-Type'),
                    passed=response.code == target.status and actual_hash == target.sha256)
        except (OSError, ValueError, EOFError, HTTPException, zlib.error) as error:
            attempt.update(error=type(error).__name__+': '+str(error), passed=False)
        attempts.append(attempt)
        if attempt['passed']:
            break
        if index < retries:
            time.sleep(delay*(index+1))
    return {'route': target.route, 'expected_status': target.status,
            'expected_sha256': target.sha256, 'expected_bytes': target.size,
            'passed': attempts[-1]['passed'], 'attempts': attempts}


def audit(targets, base, *, workers=4, timeout=20., retries=2, delay=.5, progress=None):
    parts = urlsplit(base)
    origin = parts.scheme+'://'+parts.netloc
    rows = [None]*len(targets)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_target, target, origin, timeout=timeout,
            retries=retries, delay=delay): index for index, target in enumerate(targets)}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            rows[futures[future]] = future.result()
            if progress and (completed % 25 == 0 or completed == len(targets)):
                progress(completed, len(targets))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--revision', default='HEAD', help='Git revision used for the production build')
    parser.add_argument('--base-url', default='https://chase6305.github.io/')
    parser.add_argument('--images', action='store_true', help='Also compare every emitted image file')
    parser.add_argument('--include', action='append', default=[], help='Extra build-relative file; repeatable')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--retries', type=int, default=2, help='Additional attempts per target')
    parser.add_argument('--output', type=Path, help='Write a detailed JSON report')
    args = parser.parse_args()
    if (not 1 <= args.workers <= 8 or not 0 < args.timeout <= 120 or not 0 <= args.retries <= 5):
        parser.error('Require workers=1..8, timeout in (0,120], and retries=0..5')
    try:
        base = normalize_base(args.base_url)
        revision, metadata = commit_metadata(ROOT, args.revision)
        targets = inventory(args.public, base, metadata, images=args.images, include=args.include)
        rows = audit(targets, base, workers=args.workers, timeout=args.timeout, retries=args.retries,
            progress=lambda done, total: print(f'Checked {done}/{total}', file=sys.stderr, flush=True))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, str(error)+'\n')
    summary = {'urls': len(rows), 'failures': sum(not row['passed'] for row in rows),
               'retried_urls': sum(len(row['attempts']) > 1 for row in rows)}
    report = {'revision': revision, 'public': str(args.public.resolve()), 'base_url': base,
              'images_included': args.images, 'summary': summary, 'results': rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'revision': revision, **summary}, ensure_ascii=False))
    return 1 if summary['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
