#!/usr/bin/env python3
"""Validate pinned browser assets, local formula fonts and emitted integrity."""
import argparse
import base64
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Assets(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.links = []
        self.feed(source)

    def handle_starttag(self, tag, pairs):
        attrs = dict(pairs)
        if tag == 'script' and attrs.get('src'):
            self.links.append((attrs['src'], attrs.get('integrity')))
        elif tag == 'link' and attrs.get('rel') == 'stylesheet':
            self.links.append((attrs['href'], attrs.get('integrity')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, default=ROOT / 'public')
    args = parser.parse_args()
    public = args.public.resolve()
    manifest = json.loads((ROOT / 'docs/vendor-assets.json').read_text())
    pinned_files = 0
    for package in manifest:
        assert package['source'].endswith(f"/{package['package']}-{package['version']}.tgz")
        license_path = ROOT / 'assets/vendor' / package['package'] / 'LICENSE'
        assert license_path.is_file(), f'Missing license: {package["package"]}'
        for record in package['files']:
            file = ROOT / record['path']
            assert file.is_file(), f'Missing pinned asset: {file}'
            assert hashlib.sha256(file.read_bytes()).hexdigest() == record['sha256'], f'Changed pinned asset: {file}'
            if file.is_relative_to(ROOT / 'static/licenses'):
                published = public / file.relative_to(ROOT / 'static')
                assert published.is_file() and published.read_bytes() == file.read_bytes(), f'Missing published license: {published}'
            pinned_files += 1

    styles = list((public / 'css').glob('katex*.css'))
    assert styles, 'No built KaTeX stylesheet'
    font_paths = set()
    for stylesheet in styles:
        urls = re.findall(r'url\(([^)]+)\)', stylesheet.read_text())
        assert len(urls) == 20, f'Expected twenty WOFF2 faces: {stylesheet}'
        for url in urls:
            url = url.strip('\"\'')
            assert not urlsplit(url).scheme and not url.startswith('//'), f'External math font: {url}'
            font = (stylesheet.parent / unquote(url)).resolve()
            assert font.is_relative_to(public) and font.suffix == '.woff2', f'Invalid font path: {url}'
            assert font.is_file() and font.read_bytes()[:4] == b'wOF2', f'Missing or invalid font: {url}'
            font_paths.add(font)
    assert len(font_paths) == 20

    checked = set()
    for html in public.rglob('index.html'):
        for href, integrity in Assets(html.read_text()).links:
            parts = urlsplit(href)
            assert not parts.scheme and not parts.netloc, f'External script/style: {html}: {href}'
            target = public / unquote(parts.path).lstrip('/')
            assert target.is_file(), f'Missing script/style: {html}: {href}'
            if integrity and (href, integrity) not in checked:
                for token in integrity.split():
                    algorithm, expected = token.split('-', 1)
                    actual = base64.b64encode(hashlib.new(algorithm, target.read_bytes()).digest()).decode()
                    assert actual == expected, f'Invalid integrity: {href}'
                checked.add((href, integrity))
    print(json.dumps({'pinned_packages': len(manifest), 'pinned_files': pinned_files,
                      'local_math_fonts': len(font_paths), 'verified_integrity_assets': len(checked)}, indent=2))


if __name__ == '__main__':
    main()
