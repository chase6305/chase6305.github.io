"""Disposable HTML/manifest asset checks; no Hugo, dependencies or network."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_site_assets import check_asset_references
from site_asset_references import resolve_reference_url


def integrity(body):
    return 'sha256-' + base64.b64encode(hashlib.sha256(body).digest()).decode()


class AssetReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='site-asset-references-')
        self.addCleanup(self.temp.cleanup)
        self.public = Path(self.temp.name).resolve()
        self.css = b'body { color: black; }'
        self.script = b'window.loaded = true;'
        self.write('css/main.css', self.css)
        self.write('js/main.js', self.script)
        self.head = (f'<link rel="stylesheet" href="/css/main.css" integrity="{integrity(self.css)}">'
                     f'<script src="/js/main.js" integrity="{integrity(self.script)}"></script>')

    def write(self, name, data):
        path = self.public / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return path

    def check(self, base='https://example.test/'):
        return check_asset_references(self.public, base)

    def test_both_recovery_pages_check_real_styles_and_scripts(self):
        for name in ('index.html', '404.html', 'zh-cn/404.html'):
            self.write(name, self.head)
        result = self.check()
        self.assertEqual(result['html_pages_checked'], 3)
        self.assertEqual(result['verified_integrity_assets'], 2)
        self.write('404.html', self.head.replace('/css/main.css', '/css/missing.css'))
        with self.assertRaisesRegex(AssertionError, 'Missing script/style:.*404.html'):
            self.check()

    def test_recovery_sri_failure_for_css_and_javascript(self):
        for body in (self.css, self.script):
            with self.subTest(asset=body):
                self.write('zh-cn/404.html', self.head.replace(integrity(body), integrity(b'incorrect')))
                with self.assertRaisesRegex(AssertionError, 'Invalid integrity:.*zh-cn/404.html'):
                    self.check()

    def test_relative_script_names_are_checked_per_page(self):
        self.write('one/index.html', f'<script src="main.js" integrity="{integrity(self.script)}"></script>')
        self.write('two/index.html', f'<script src="main.js" integrity="{integrity(self.script)}"></script>')
        self.write('one/main.js', self.script)
        self.write('two/main.js', b'different bytes')
        with self.assertRaisesRegex(AssertionError, 'Invalid integrity:.*two/index.html'):
            self.check()

    def test_manifest_relative_root_query_and_project_prefix(self):
        self.write('index.html', '<link rel="MANIFEST" href="config/site.webmanifest?v=2">')
        self.write('config/site.webmanifest', json.dumps({'name': 'Optional fields', 'icons': [
            {'src': '../icons/relative.png?v=3'}, {'src': '/project/icons/root.png#icon'},
            {'src': 'https://elsewhere.invalid/not-fetched.png'},
            {'src': 'data:image/png;base64,not-fetched'}, {'src': '/outside-project/not-fetched.png'},
            {}, None, {'src': 17}, {'src': ''},
        ]}))
        self.write('icons/relative.png', b'relative icon')
        self.write('icons/root.png', b'root icon')
        result = self.check('https://example.test/project/')
        self.assertEqual(result['manifest_icons_checked'], 2)
        (self.public / 'icons/relative.png').unlink()
        with self.assertRaisesRegex(ValueError, r'Manifest icon missing:.*project/icons/relative.png\?v=3'):
            self.check('https://example.test/project/')

    def test_optional_icons_and_external_manifest_remain_optional(self):
        self.write('index.html', '<link rel="manifest" href="site.webmanifest">'
                   '<link rel="manifest" href="https://elsewhere.invalid/manifest.webmanifest">')
        for data in ({'name': 'Without icons'}, {'icons': 'unsupported value'}, {'icons': [None, {}]}):
            with self.subTest(data=data):
                self.write('site.webmanifest', json.dumps(data))
                self.assertEqual(self.check()['manifest_icons_checked'], 0)

    def test_manifest_icon_url_cleanup_preserves_filename_spaces_and_case(self):
        self.write('index.html', '<link rel="manifest" href="config/site.webmanifest">')
        # Expected file mappings were independently checked with Node's URL API.
        cases = [
            (' \t../icons/plain.png \r\n', 'icons/plain.png'),
            ('\x00\x01../icons/plain.png\x1e\x1f', 'icons/plain.png'),
            ('../icons/pl\nain.\tpng', 'icons/plain.png'),
            ('..\\icons\\plain.png', 'icons/plain.png'),
            ('../icons/Case.png', 'icons/Case.png'),
            ('../icons/%20name.png%20', 'icons/ name.png '),
            ('%20', 'config/ '),
            ('\u00a0', 'config/\u00a0'),
        ]
        for reference, filename in cases:
            with self.subTest(reference=reference):
                self.write(filename, b'icon')
                self.write('config/site.webmanifest', json.dumps({'icons': [{'src': reference}]}))
                self.assertEqual(self.check()['manifest_icons_checked'], 1)

    def test_manifest_same_origin_case_and_default_ports_cannot_hide_missing_icons(self):
        self.write('index.html', '<link rel="manifest" href="config/site.webmanifest">')
        cases = [
            ('https://example.test/project/', 'https://EXAMPLE.TEST/project/icons/icon.png'),
            ('https://example.test/project/', 'https://example.test:443/project/icons/icon.png'),
            ('http://example.test/project/', 'http://EXAMPLE.TEST:80/project/icons/icon.png'),
        ]
        for base, reference in cases:
            with self.subTest(reference=reference):
                icon = self.write('icons/icon.png', b'icon')
                self.write('config/site.webmanifest', json.dumps({'icons': [{'src': reference}]}))
                self.assertEqual(self.check(base)['manifest_icons_checked'], 1)
                icon.unlink()
                with self.assertRaisesRegex(ValueError, 'Manifest icon missing'):
                    self.check(base)

    def test_manifest_href_is_cleaned_before_resolution(self):
        self.write('icons/icon.png', b'icon')
        self.write('config/site.webmanifest', '{"icons":[{"src":"../icons/icon.png"}]}')
        for reference in (' \tconfig\\site.webmanifest?v=2 \r\n',
                          'HTTPS://EXAMPLE.TEST:443/project/config/site.webmanifest?v=2'):
            with self.subTest(reference=reference):
                self.write('index.html', f'<link rel="manifest" href="{reference}">')
                self.assertEqual(self.check('https://example.test/project/')['manifest_icons_checked'], 1)

    def test_external_backslash_authority_is_skipped_and_query_backslashes_are_kept(self):
        base = 'https://example.test/project/config/site.webmanifest'
        self.assertEqual(resolve_reference_url(base, '..\\icons\\icon.png?mode=a\\b#part\\c'),
                         'https://example.test/project/icons/icon.png?mode=a\\b#part\\c')
        self.write('index.html', '<link rel="manifest" href="config/site.webmanifest">')
        self.write('config/site.webmanifest', json.dumps({'icons': [
            {'src': '\\\\elsewhere.test\\project\\icons\\missing.png'},
            {'src': 'https://example.test:444/project/icons/missing.png'},
        ]}))
        self.assertEqual(self.check('https://example.test/project/')['manifest_icons_checked'], 0)

    def test_manifest_icon_cannot_escape_build_through_symlink(self):
        with tempfile.TemporaryDirectory(prefix='site-asset-outside-') as directory:
            outside = Path(directory) / 'icon.png'
            outside.write_bytes(b'outside')
            self.write('index.html', '<link rel="manifest" href="site.webmanifest">')
            self.write('site.webmanifest', '{"icons":[{"src":"escape.png"}]}')
            (self.public / 'escape.png').symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'outside the production build'):
                self.check()


if __name__ == '__main__':
    unittest.main(verbosity=2)
