#!/usr/bin/env python3
"""Exercise shared HTML validation with disposable pages; no Hugo or network."""
from pathlib import Path
import tempfile
import unittest

from validate_blog import Page, check_html_pages


class HTMLValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="blog-html-validation-")
        self.addCleanup(self.temp.cleanup)
        self.public = Path(self.temp.name).resolve()

    def write(self, name, text):
        path = self.public / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def check(self, **options):
        pages = {path: Page(path.read_text(encoding="utf-8"))
                 for path in self.public.rglob("*.html")}
        return check_html_pages(self.public, pages, **options)

    def test_missing_image_on_both_language_homes(self):
        for name in ("index.html", "zh-cn/index.html"):
            self.write(name, '<img src="/orca-avatar.webp" alt="Orca">')
        result = self.check()
        self.assertEqual(result["html_pages_checked"], 2)
        self.assertEqual(result["local_references_checked"], 2)
        self.assertCountEqual(result["errors"], [
            "Missing local target: index.html -> /orca-avatar.webp",
            "Missing local target: zh-cn/index.html -> /orca-avatar.webp",
        ])
        self.write("orca-avatar.webp", "image fixture")
        self.assertEqual(self.check()["errors"], [])

    def test_non_article_anchors_and_valid_unicode_fragment(self):
        self.write("projects/index.html", '<a href="/zh-cn/projects/#%E5%AE%9E%E9%AA%8C">valid</a>'
                   '<a href="/labs/demo/#missing">missing</a>')
        self.write("zh-cn/projects/index.html", '<h1 id="实验">Project</h1>')
        self.write("labs/demo/index.html", '<h1 id="experiment">Lab</h1>')
        self.assertEqual(self.check()["errors"], [
            "Missing anchor: projects/index.html -> /labs/demo/#missing"])

    def test_duplicate_ids_and_figure_labels_outside_collections(self):
        self.write("index.html", '<h1 id="same">Home</h1><span id="same">Again</span>')
        self.write("labs/demo/index.html", '<span class="article-figure__number">图 1</span>'
                   '<span class="article-figure__number">图<span>1</span></span>')
        self.assertCountEqual(self.check()["errors"], [
            "Duplicate HTML ids: index.html: ['same']",
            "Duplicate figure labels: labs/demo/index.html: ['图1']",
        ])

    def test_missing_alt_on_recovery_page_but_allow_decorative_image(self):
        self.write("404.html", '<img src="/icon.svg"><img src="/icon.svg" alt="">')
        self.write("icon.svg", '<svg xmlns="http://www.w3.org/2000/svg"/>')
        self.assertEqual(self.check()["errors"], ["Missing image alt: 404.html -> /icon.svg"])

    def test_math_warning_and_global_or_selected_strictness(self):
        self.write("projects/index.html", '<p>$$</p>')
        self.write("labs/demo/index.html", '<p>$$</p>')
        result = self.check()
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["unrendered_math_warnings"]), 2)
        self.assertCountEqual(self.check(strict_math=[])["errors"], result["unrendered_math_warnings"])
        selected = self.check(strict_math=["projects/index.html"])
        self.assertEqual(selected["errors"], ["Unrendered display math: projects/index.html: ['$$']"])
        self.assertEqual(len(selected["unrendered_math_warnings"]), 2)

    def test_external_links_and_literal_markup_are_not_local_failures(self):
        self.write("index.html", '<a href="https://example.invalid/missing#absent">external</a>'
                   '<a href="mailto:reader@example.invalid">email</a>'
                   '<a href="/projects/#present">project</a>'
                   '<pre><code>&lt;a href="/missing/#absent"&gt;Example&lt;/a&gt;\n$$</code></pre>'
                   '<code>$$</code><script>const literal="$$"</script>'
                   '<math><semantics><annotation>$$</annotation></semantics></math>')
        self.write("projects/index.html", '<h1 id="present">Project</h1>')
        result = self.check(strict_math=[])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["unrendered_math_warnings"], [])
        self.assertEqual(result["local_references_checked"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
