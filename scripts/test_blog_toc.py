#!/usr/bin/env python3
"""Build disposable pages to check math, literal dollars and code in both TOCs."""
import argparse
import json
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = r"""---
title: TOC rendering fixture
type: blog
draft: true
math: true
reading_focus: Temporary regression fixture
reading_prerequisites: Markdown
---

## Dollar math $G_t$ and $a < b$ {#dollar-math}

## Parentheses math \(\dot q\) {#parentheses-math}

## Literal prices \$5 and \$10 {#literal-prices}

## Code `$HOME` and `$PATH` {#literal-code}

## Mixed **bold** and `q[0]` with $\alpha$ {#mixed-markup}

## [Reference](https://example.com/) and $x$ {#linked-label}

## 中文公式 $q_1$ {#中文公式}
"""


class TocPage(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.current = None
        self.links = {"desktop": {}, "mobile": {}}
        self.ids = set()
        self.id_tags = {}
        self.html_language = None
        self.nav_language = None
        self.back_to_top_language = None
        self.archive_languages = {"title": [], "date": [], "draft": []}
        self.title_languages = []
        self.title_language_attributes = []
        self.content_languages = []
        self.content_language_attributes = []
        self.generic_body_language = None
        self.search_trigger_languages = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        language = attrs.get("lang", self.stack[-1][2] if self.stack else None)
        if tag == "html":
            self.html_language = language
        if tag == "h1":
            self.title_languages.append(language)
            self.title_language_attributes.append(attrs.get("lang"))
        if attrs.get("id") == "generic-body-copy":
            self.generic_body_language = language
        if "data-search-open" in attrs:
            self.search_trigger_languages.append(language)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, ("duplicate id", attrs["id"])
            self.ids.add(attrs["id"])
            self.id_tags[attrs["id"]] = tag
        group = next((item[1] for item in reversed(self.stack) if item[1]), None)
        classes = attrs.get("class", "").split()
        if "content" in classes:
            self.content_languages.append(language)
            self.content_language_attributes.append(attrs.get("lang"))
        if "hextra-toc" in classes:
            group = "desktop"
            self.nav_language = language
        elif "blog-mobile-toc" in classes:
            group = "mobile"
        elif "archive-year__posts" in classes:
            group = "archive"
        if attrs.get("id") == "backToTop":
            self.back_to_top_language = language
        if group == "archive":
            if tag == "time":
                self.archive_languages["date"].append(language)
            elif tag == "span":
                self.archive_languages["title"].append(language)
            elif tag == "small" and "blog-draft-label" in classes:
                self.archive_languages["draft"].append(language)
        if tag == "a" and group in self.links and attrs.get("href", "").startswith("#"):
            assert self.current is None, "nested TOC link"
            self.current = {"group": group, "href": attrs["href"],
                            "math": 0, "text": [], "language": language}
        if self.current and "katex" in classes:
            self.current["math"] += 1
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append((tag, group, language))

    def handle_data(self, text):
        if self.current:
            self.current["text"].append(text)

    def handle_endtag(self, tag):
        if tag == "a" and self.current:
            item = self.current
            self.links[item["group"]][unquote(item["href"][1:])] = {
                "math": item["math"], "text": "".join(item["text"]),
                "language": item["language"]}
            self.current = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hugo", default="hugo")
    args = parser.parse_args()
    expected = {"dollar-math": 2, "parentheses-math": 1, "literal-prices": 0,
                "literal-code": 0, "mixed-markup": 1, "linked-label": 1,
                "中文公式": 1}
    with tempfile.TemporaryDirectory(prefix="chase-toc-regression-") as folder:
        work = Path(folder)
        content = work / "content"
        article = content / "posts/toc-fixture/index.md"
        article.parent.mkdir(parents=True)
        article.write_text(FIXTURE)
        (content / "archives.md").write_text("---\ntitle: Archive\nlayout: archives\n---\n")
        (content / "generic-zh.md").write_text(
            '---\ntitle: 中文通用页\nlayout: default\ncontentLanguage: zh-CN\n---\n\n'
            '<p id="generic-body-copy">中文介绍。</p>\n')
        (content / "generic-en.md").write_text(
            '---\ntitle: English generic page\nlayout: default\n---\n\n'
            '<p id="generic-body-copy">An English introduction.</p>\n')
        shutil.copy2(ROOT / "content/posts/_index.md", content / "posts/_index.md")
        command = [args.hugo, "--minify", "-D", "--source", str(ROOT),
                   "--contentDir", str(content), "--destination", str(work / "public")]
        for declared_language in (None, "zh-CN"):
            for anchor_at_start in (None, False, True):
                options = "" if declared_language is None else "\ncontentLanguage: " + declared_language
                if anchor_at_start is not None:
                    options += "\nheadingAnchorAtStart: " + str(anchor_at_start).lower()
                article.write_text(FIXTURE.replace("math: true", "math: true" + options))
                subprocess.run(command, check=True, capture_output=True, text=True)
                page = TocPage((work / "public/posts/toc-fixture/index.html").read_text())
                language = declared_language or "en-US"
                for group, links in page.links.items():
                    assert set(links) == set(expected), (group, links)
                    for key, count in expected.items():
                        assert links[key]["math"] == count, (group, key, links[key])
                        assert links[key]["language"] == language, (group, key, links[key])
                        assert key in page.ids
                        assert page.id_tags[key] == ("span" if anchor_at_start is False else "h2")
                    assert links["literal-prices"]["text"] == "Literal prices $5 and $10"
                    assert links["literal-code"]["text"] == "Code $HOME and $PATH"
                assert page.nav_language == page.back_to_top_language == page.html_language == "en"
                archive = TocPage((work / "public/archives/index.html").read_text())
                assert archive.archive_languages == {"title": [language], "date": ["en"], "draft": ["en"]}
        for name, language, attribute in (("generic-zh", "zh-CN", "zh-CN"),
                                          ("generic-en", "en", None)):
            generic = TocPage((work / f"public/{name}/index.html").read_text())
            assert generic.title_languages == generic.content_languages == [language]
            assert generic.title_language_attributes == generic.content_language_attributes == [attribute]
            assert generic.generic_body_language == language
            assert generic.html_language == "en"
            assert generic.search_trigger_languages and all(
                value == "en" for value in generic.search_trigger_languages)
        # Auto-generated sections have no source File. Topic navigation must be optional.
        (content / "posts/_index.md").unlink()
        subprocess.run(command, check=True, capture_output=True, text=True)
    print(json.dumps({"toc_variants": 2, "anchor_modes": 3, "headings_per_variant": len(expected),
                      "page_languages": ["en-US", "zh-CN"],
                      "chapter_and_archive_languages": "passed; UI/date/draft labels retain English",
                      "generic_page_languages": "explicit Chinese title/body; undeclared English and UI inheritance preserved",
                      "literal_dollars": "preserved", "both_math_delimiters": "passed",
                      "nested_links": "absent", "automatic_section": "passed"}, indent=2))


if __name__ == "__main__":
    main()
