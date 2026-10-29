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


if __name__ == "__main__":
    unittest.main()
