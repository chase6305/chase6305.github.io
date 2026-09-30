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
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, ("duplicate id", attrs["id"])
            self.ids.add(attrs["id"])
            self.id_tags[attrs["id"]] = tag
        group = next((kind for _, kind in reversed(self.stack) if kind), None)
        classes = attrs.get("class", "").split()
        if "hextra-toc" in classes:
            group = "desktop"
        elif "blog-mobile-toc" in classes:
            group = "mobile"
        if tag == "a" and group and attrs.get("href", "").startswith("#"):
            assert self.current is None, "nested TOC link"
            self.current = {"group": group, "href": attrs["href"],
                            "math": 0, "text": []}
        if self.current and "katex" in classes:
            self.current["math"] += 1
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append((tag, group))

    def handle_data(self, text):
        if self.current:
            self.current["text"].append(text)

    def handle_endtag(self, tag):
        if tag == "a" and self.current:
            item = self.current
            self.links[item["group"]][unquote(item["href"][1:])] = {
                "math": item["math"], "text": "".join(item["text"])}
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
        shutil.copy2(ROOT / "content/posts/_index.md", content / "posts/_index.md")
        command = [args.hugo, "--minify", "-D", "--source", str(ROOT),
                   "--contentDir", str(content), "--destination", str(work / "public")]
        for anchor_at_start in (False, True):
            fixture = FIXTURE.replace("math: true", "math: true\nheadingAnchorAtStart: " +
                                      str(anchor_at_start).lower())
            article.write_text(fixture)
            subprocess.run(command, check=True, capture_output=True, text=True)
            page = TocPage((work / "public/posts/toc-fixture/index.html").read_text())
            for group, links in page.links.items():
                assert set(links) == set(expected), (group, links)
                for key, count in expected.items():
                    assert links[key]["math"] == count, (group, key, links[key])
                    assert key in page.ids
                    assert page.id_tags[key] == ("h2" if anchor_at_start else "span")
                assert links["literal-prices"]["text"] == "Literal prices $5 and $10"
                assert links["literal-code"]["text"] == "Code $HOME and $PATH"
        # Auto-generated sections have no source File. Topic navigation must be optional.
        (content / "posts/_index.md").unlink()
        subprocess.run(command, check=True, capture_output=True, text=True)
    print(json.dumps({"toc_variants": 2, "anchor_modes": 2, "headings_per_variant": len(expected),
                      "literal_dollars": "preserved", "both_math_delimiters": "passed",
                      "nested_links": "absent", "automatic_section": "passed"}, indent=2))


if __name__ == "__main__":
    main()
