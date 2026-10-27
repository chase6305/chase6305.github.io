#!/usr/bin/env python3
"""Validate pinned browser assets, local formula fonts and emitted integrity."""
import argparse
import base64
import hashlib
import json
import re
import subprocess
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

    overrides = json.loads((ROOT / 'docs/theme-overrides.json').read_text())
    revision = subprocess.check_output(['git', '-C', str(ROOT / 'themes/hextra'), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
    assert revision == overrides['source_commit'], 'Review the theme override manifest after upgrading Hextra'
    theme_license = ROOT / overrides['published_license']
    assert hashlib.sha256(theme_license.read_bytes()).hexdigest() == overrides['license_sha256']
    assert (public / theme_license.relative_to(ROOT / 'static')).read_bytes() == theme_license.read_bytes()
    for override in overrides['overrides']:
        assert (ROOT / override['local']).is_file(), override
        assert (ROOT / 'themes/hextra' / override['upstream']).is_file(), override

    renderer = json.loads((ROOT / 'docs/render-overrides.json').read_text())
    version = re.search(r'HUGO_VERSION:\s*["\']?([0-9.]+)',
                        (ROOT / '.github/workflows/hugo.yml').read_text()).group(1)
    assert renderer['source_version'] == 'v' + version, 'Review table rendering after upgrading Hugo'
    assert (ROOT / renderer['local']).is_file()
    renderer_license = ROOT / renderer['published_license']
    assert hashlib.sha256(renderer_license.read_bytes()).hexdigest() == renderer['license_sha256']
    assert (public / renderer_license.relative_to(ROOT / 'static')).read_bytes() == renderer_license.read_bytes()

    checked = set()
    viewer_scripts = set()
    for html in public.rglob('index.html'):
        for href, integrity in Assets(html.read_text()).links:
            parts = urlsplit(href)
            assert not parts.scheme and not parts.netloc, f'External script/style: {html}: {href}'
            target = public / unquote(parts.path).lstrip('/')
            assert target.is_file(), f'Missing script/style: {html}: {href}'
            assert 'medium-zoom' not in href, f'Retired image dependency is still loaded: {href}'
            if '/js/image-viewer' in href:
                assert integrity, 'Image viewer must have integrity metadata'
                viewer_scripts.add(href)
            if integrity and (href, integrity) not in checked:
                for token in integrity.split():
                    algorithm, expected = token.split('-', 1)
                    actual = base64.b64encode(hashlib.new(algorithm, target.read_bytes()).digest()).decode()
                    assert actual == expected, f'Invalid integrity: {href}'
                checked.add((href, integrity))
    assert len(viewer_scripts) == 1, 'Expected one shared native image-viewer asset'
    print(json.dumps({'pinned_packages': len(manifest), 'pinned_files': pinned_files,
                      'active_pinned_packages': sum(item.get('runtime', True) for item in manifest),
                      'native_image_viewers': len(viewer_scripts),
                      'theme_overrides': len(overrides['overrides']),
                      'core_renderer_overrides': 1,
                      'local_math_fonts': len(font_paths), 'verified_integrity_assets': len(checked)}, indent=2))


if __name__ == '__main__':
    main()
