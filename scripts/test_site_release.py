"""Local, deterministic regression tests for immutable release verification."""
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest

import check_site_release as release


def target(route, body=b"correct", status=200):
    return release.Target(route, status, hashlib.sha256(body).hexdigest(), len(body))


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="release-inventory-")
        self.root = Path(self.temp.name)
        self.public = self.root / "public"
        self.public.mkdir()
        self.write("index.html", b'<a href="posts/example/example.py">download</a>')
        self.write("404.html", b"not found")
        self.write("posts/example/index.html", b"article")
        self.write("posts/example/example.py", b"print(42)\n")
        self.metadata = {"posts": [
            {"path": "content/posts/example/index.md", "draft": False},
            {"path": "content/posts/hidden/index.md", "draft": True}]}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, body):
        path = self.public / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return path

    def inventory(self, **kwargs):
        return release.inventory(self.public, "https://example.test/project/", self.metadata, **kwargs)

    def test_downloads_fonts_and_immutable_hashes(self):
        self.write("fonts/math.woff2", b"font")
        rows = self.inventory()
        by_route = {row.route: row for row in rows}
        self.assertEqual(by_route["/project/posts/hidden/"].status, 404)
        self.assertIn("/project/fonts/math.woff2", by_route)
        expected = target("/project/posts/example/example.py", b"print(42)\n")
        self.assertEqual(by_route[expected.route], expected)
        self.write("posts/example/example.py", b"changed after inventory")
        self.assertEqual(by_route[expected.route], expected)
        self.assertEqual(sum(row.status == 404 for row in rows), 3)

    def test_draft_leak_missing_published_page_and_download(self):
        draft = self.write("posts/hidden/index.html", b"draft")
        with self.assertRaisesRegex(ValueError, "contains a draft"):
            self.inventory()
        draft.unlink()
        article = self.public / "posts/example/index.html"
        article.unlink()
        with self.assertRaisesRegex(ValueError, "Published article is missing"):
            self.inventory()
        self.write("posts/example/index.html", b"article")
        (self.public / "posts/example/example.py").unlink()
        with self.assertRaisesRegex(ValueError, "Linked download missing"):
            self.inventory()

    def test_escaped_paths_case_images_and_extra_files(self):
        self.write("My 图%/Run File.py", b"source")
        self.write("figure.PNG", b"image")
        self.write("assets/Extra.JS", b"script")
        self.write("data.bin", b"binary")
        self.write("index.html", '<a href="My%20%E5%9B%BE%25/Run%20File.py?download=1">file</a>'.encode())
        normal = {row.route for row in self.inventory()}
        self.assertIn("/project/My%20%E5%9B%BE%25/Run%20File.py", normal)
        self.assertIn("/project/assets/Extra.JS", normal)
        self.assertNotIn("/project/figure.PNG", normal)
        expanded = {row.route for row in self.inventory(images=True, include=["data.bin"])}
        self.assertTrue({"/project/figure.PNG", "/project/data.bin"} <= expanded)

    def test_linked_config_ui_feed_and_extensionless_downloads(self):
        names = ["example.yaml", "robot-dialog.ui", "posts/index.xml", "LICENSE", "extra.bin"]
        for name in names:
            self.write(name, b"download")
        self.write("index.html", "".join(f'<a href="{name}">download</a>' for name in names).encode())
        routes = {row.route for row in self.inventory()}
        self.assertTrue({"/project/" + name for name in names} <= routes)
        (self.public / "extra.bin").unlink()
        with self.assertRaisesRegex(ValueError, "Linked download missing"):
            self.inventory()

    def test_browser_icons_manifest_and_shared_search_indexes(self):
        self.write("favicon.ico", b"light icon")
        self.write("favicon-dark.ico", b"dark icon")
        self.write("touch.png", b"touch icon")
        self.write("site.webmanifest", b"{}")
        self.write("en.search-data.min.abc.json", b"[]")
        self.write("zh-cn.search-data.min.def.json", b"[]")
        self.write("assets/widget-data.abc.json", b"[]")
        self.write("index.html", b'<link rel><link rel="ICON" href="favicon.ico?v=2" '
                   b'data-favicon-dark="favicon-dark.ico?v=2">'
                   b'<link rel="apple-touch-icon" href="touch.png">'
                   b'<link rel="manifest" href="site.webmanifest">')
        routes = {row.route for row in self.inventory()}
        expected = {"favicon.ico", "favicon-dark.ico", "touch.png", "site.webmanifest",
                    "en.search-data.min.abc.json", "zh-cn.search-data.min.def.json",
                    "assets/widget-data.abc.json"}
        self.assertTrue({"/project/" + name for name in expected} <= routes)
        (self.public / "favicon-dark.ico").unlink()
        with self.assertRaisesRegex(ValueError, "Linked download missing"):
            self.inventory()

    def test_unlinked_nested_sitemaps_and_taxonomy_feeds(self):
        files = {
            "sitemap.xml": b"<sitemapindex/>",
            "en/sitemap.xml": b"<urlset><url>English</url></urlset>",
            "zh-cn/sitemap.xml": b"<urlset><url>Chinese</url></urlset>",
            "tags/cpp/index.xml": b"<rss><channel>cpp</channel></rss>",
            "categories/robotics/index.xml": b"<rss><channel>robotics</channel></rss>",
        }
        for name, body in files.items():
            self.write(name, body)
        # None of the XML files are linked from the fixture's HTML pages.
        rows = {row.route: row for row in self.inventory()}
        for name, body in files.items():
            route = "/project/" + name
            self.assertEqual(rows[route], target(route, body))

    def test_manifest_declared_icons_are_default_frozen_targets(self):
        self.write('index.html', b'<link rel="manifest" href="config/site.webmanifest?v=1">')
        self.write('config/site.webmanifest', json.dumps({'icons': [
            {'src': '../icons/install.png?v=2'}, {'src': '/project/icons/root.png'},
            {'src': 'https://elsewhere.invalid/external.png'},
            {'src': '/outside-project/outside.png'}, None, {'src': ''},
        ]}).encode())
        self.write('icons/install.png', b'install icon')
        self.write('icons/root.png', b'root icon')
        rows = {row.route: row for row in self.inventory()}
        expected = target('/project/icons/install.png', b'install icon')
        self.assertEqual(rows[expected.route], expected)
        self.assertEqual(rows['/project/icons/root.png'], target('/project/icons/root.png', b'root icon'))
        self.write('icons/install.png', b'changed after snapshot')
        self.assertEqual(rows[expected.route], expected)
        (self.public / 'icons/root.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Manifest icon missing'):
            self.inventory()

    def test_manifest_icon_internal_symlink_keeps_its_declared_route(self):
        self.write('index.html', b'<link rel="manifest" href="site.webmanifest">')
        self.write('site.webmanifest', b'{"icons":[{"src":"alias.png"}]}')
        actual = self.write('actual.png', b'icon bytes')
        (self.public / 'alias.png').symlink_to(actual)
        rows = {row.route: row for row in self.inventory()}
        self.assertEqual(rows['/project/alias.png'], target('/project/alias.png', b'icon bytes'))
        (self.public / 'manifest-alias.webmanifest').symlink_to(self.public / 'site.webmanifest')
        self.write('index.html', b'<link rel="manifest" href="manifest-alias.webmanifest">')
        rows = {row.route: row for row in self.inventory()}
        self.assertEqual(rows['/project/manifest-alias.webmanifest'],
                         target('/project/manifest-alias.webmanifest', b'{"icons":[{"src":"alias.png"}]}'))

    def test_manifest_href_input_origin_and_missing_icon(self):
        body = b'{"icons":[{"src":"../icons/Case.png?v=3"}]}'
        self.write('config/site.webmanifest', body)
        for reference in (' \tconfig\\site.webmanifest?v=2 \r\n',
                          'HTTPS://EXAMPLE.TEST:443/project/config/site.webmanifest?v=2'):
            with self.subTest(reference=reference):
                self.write('index.html', f'<link rel="manifest" href="{reference}">'.encode())
                icon = self.write('icons/Case.png', b'icon bytes')
                rows = {row.route: row for row in self.inventory()}
                self.assertEqual(rows['/project/config/site.webmanifest'],
                                 target('/project/config/site.webmanifest', body))
                self.assertEqual(rows['/project/icons/Case.png'],
                                 target('/project/icons/Case.png', b'icon bytes'))
                icon.unlink()
                with self.assertRaisesRegex(ValueError, 'Manifest icon missing'):
                    self.inventory()

    def test_outside_paths_and_conflicting_fixture(self):
        outside = self.root / "outside.py"
        outside.write_bytes(b"outside")
        for extra in ["../outside.py", "https://example.test/a.py", "data.bin?q=x"]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.inventory(include=[extra])
        (self.public / "escape.js").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.inventory()
        (self.public / "escape.js").unlink()
        (self.public / "site-release-check-missing").mkdir()
        with self.assertRaisesRegex(ValueError, "fixture conflicts"):
            self.inventory()

    def test_commit_metadata_ignores_working_tree_changes(self):
        repository = self.root / "repo"
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        docs = repository / "docs"
        docs.mkdir()
        path = docs / "blog-editorial-review.json"
        path.write_text(json.dumps(self.metadata))
        subprocess.run(["git", "-C", str(repository), "add", "docs"], check=True)
        subprocess.run(["git", "-C", str(repository), "-c", "user.name=Release test",
            "-c", "user.email=release@example.test", "commit", "-qm", "fixture"], check=True)
        sha, original = release.commit_metadata(repository, "HEAD")
        path.write_text('{"posts": []}')
        frozen_sha, frozen = release.commit_metadata(repository, sha)
        self.assertEqual((frozen_sha, frozen), (sha, original))
        self.assertEqual(frozen, self.metadata)

    def test_base_url_validation(self):
        for base in ["file:///tmp/site", "https://user:secret@example.test/",
                     "https://example.test/?x=1", "https://example.test/#anchor",
                     "https://example.test:bad/", "https://example.test/a/%2E%2E/b/"]:
            with self.subTest(base=base), self.assertRaises(ValueError):
                release.normalize_base(base)
        self.assertEqual(release.normalize_base("https://example.test/project"),
                         "https://example.test/project/")


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = {}
        cls.lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                with cls.lock:
                    count = cls.counts.get(self.path, 0) + 1
                    cls.counts[self.path] = count
                status, body, headers = 200, b"correct", {}
                if self.path == "/gzip":
                    body = gzip.compress(body)
                    headers["Content-Encoding"] = "gzip"
                elif self.path == "/bad-gzip":
                    headers["Content-Encoding"] = "gzip"
                elif self.path == "/missing":
                    status, body = 404, b"not found"
                elif self.path == "/redirect":
                    status, headers = 302, {"Location": "/redirect-destination"}
                elif self.path == "/503-once" and count == 1:
                    status = 503
                elif self.path == "/stale-once" and count == 1:
                    body = b"stale but status 200"
                elif self.path == "/always-stale":
                    body = b"wrong"
                elif self.path == "/project/en/sitemap.xml":
                    body = b"<urlset><url>current</url></urlset>"
                elif self.path == "/project/zh-cn/sitemap.xml":
                    body = b"<urlset><url>stale</url></urlset>"
                elif self.path == "/project/tags/cpp/index.xml":
                    status, body = 404, b"not found"
                elif self.path == '/project/icons/install.png':
                    body = b'current icon'
                elif self.path == '/project/icons/stale.png':
                    body = b'old icon'
                elif self.path == '/project/icons/missing.png':
                    status, body = 404, b'not found'
                elif self.path == "/slow":
                    time.sleep(.12)
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body) + (10 if self.path == "/truncated-once" and count == 1 else 0)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def check(self, row, **kwargs):
        return release.check_target(row, self.base, delay=0, **kwargs)

    def test_gzip_and_expected_custom_404(self):
        for row in [target("/gzip"), target("/missing", b"not found", 404)]:
            self.assertTrue(self.check(row, retries=0)["passed"])
        self.assertFalse(self.check(target("/missing"), retries=0)["passed"])

    def test_transient_status_stale_bytes_and_truncated_body_retry(self):
        for path in ["/503-once", "/stale-once", "/truncated-once"]:
            with self.subTest(path=path):
                row = self.check(target(path), retries=1)
                self.assertTrue(row["passed"])
                self.assertEqual(len(row["attempts"]), 2)
                self.assertFalse(row["attempts"][0]["passed"])
                if path == "/truncated-once":
                    self.assertIn("IncompleteRead", row["attempts"][0]["error"])

    def test_wrong_bytes_and_bad_gzip_fail(self):
        for path in ["/always-stale", "/bad-gzip"]:
            row = self.check(target(path), retries=1)
            self.assertFalse(row["passed"])
            self.assertEqual(len(row["attempts"]), 2)

    def test_redirect_is_not_followed(self):
        row = self.check(target("/redirect"), retries=0)
        self.assertFalse(row["passed"])
        self.assertEqual(row["attempts"][0]["status"], 302)
        self.assertNotIn("/redirect-destination", self.counts)

    def test_timeout_reports_failure(self):
        row = self.check(target("/slow"), timeout=.02, retries=0)
        self.assertFalse(row["passed"])
        self.assertIn("timed out", row["attempts"][0]["error"])

    def test_parallel_audit_preserves_input_order(self):
        expected = [target("/last"), target("/first"), target("/missing", b"not found", 404)]
        rows = release.audit(expected, self.base, workers=3, retries=0)
        self.assertEqual([row["route"] for row in rows], [row.route for row in expected])
        self.assertTrue(all(row["passed"] for row in rows))

    def test_unlinked_xml_inventory_detects_stale_and_missing_responses(self):
        with tempfile.TemporaryDirectory(prefix="release-xml-http-") as directory:
            public = Path(directory)
            (public / "index.html").write_bytes(b"home without XML links")
            (public / "404.html").write_bytes(b"not found")
            body = b"<urlset><url>current</url></urlset>"
            names = ("en/sitemap.xml", "zh-cn/sitemap.xml", "tags/cpp/index.xml")
            for name in names:
                path = public / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            base = self.base + "/project/"
            targets = [row for row in release.inventory(public, base, {"posts": []})
                       if row.route.endswith(".xml")]
            self.assertEqual({row.route for row in targets},
                             {"/project/" + name for name in names})
            rows = {row["route"]: row for row in release.audit(targets, base, retries=0)}
            self.assertTrue(rows["/project/en/sitemap.xml"]["passed"])
            stale = rows["/project/zh-cn/sitemap.xml"]
            self.assertFalse(stale["passed"])
            self.assertEqual(stale["attempts"][0]["status"], 200)
            self.assertNotEqual(stale["attempts"][0]["sha256"], stale["expected_sha256"])
            missing = rows["/project/tags/cpp/index.xml"]
            self.assertFalse(missing["passed"])
            self.assertEqual(missing["attempts"][0]["status"], 404)

    def test_manifest_icon_targets_detect_stale_and_missing_http_bytes(self):
        with tempfile.TemporaryDirectory(prefix='release-manifest-http-') as directory:
            public = Path(directory)
            (public / 'index.html').write_bytes(b'<link rel="manifest" href="site.webmanifest">')
            (public / '404.html').write_bytes(b'not found')
            names = ('install.png', 'stale.png', 'missing.png')
            (public / 'site.webmanifest').write_text(json.dumps({'icons': [
                {'src': 'icons/' + name} for name in names]}))
            (public / 'icons').mkdir()
            for name in names:
                (public / 'icons' / name).write_bytes(b'current icon')
            base = self.base + '/project/'
            targets = [row for row in release.inventory(public, base, {'posts': []})
                       if row.route.endswith('.png')]
            self.assertEqual({row.route for row in targets}, {'/project/icons/' + name for name in names})
            rows = {row['route']: row for row in release.audit(targets, base, retries=0)}
            self.assertTrue(rows['/project/icons/install.png']['passed'])
            self.assertFalse(rows['/project/icons/stale.png']['passed'])
            self.assertEqual(rows['/project/icons/stale.png']['attempts'][0]['status'], 200)
            self.assertFalse(rows['/project/icons/missing.png']['passed'])
            self.assertEqual(rows['/project/icons/missing.png']['attempts'][0]['status'], 404)


if __name__ == "__main__":
    unittest.main()
