#!/usr/bin/env python3
"""Build disposable full-content feeds and inspect their actual XML/HTML URLs."""
import argparse
from datetime import date, timedelta
from html.entities import html5 as HTML5_ENTITIES
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BODY = '''
## Fixture heading {#heading}

Full article body after its summary.

<a id="download" href="file.zip?x=a&amp;y=b#part">download</a>
<a id="sibling" title='literal href="untouched.txt"' href='../sibling/?x=a&amp;amp;y=b#part'>sibling</a>
<a id="chapter" href="#heading">chapter</a>
<a id="query" href="?x=a+b&amp;y=%2F&amp;z=%26#part">query</a>
<a id="root" href="/root/file.zip">root</a>
<a id="cdn" href="//cdn.example.org/image.png">cdn</a>
<a id="external" href="https://other.example.org/x?a=1&amp;b=2">external</a>
<a id="email" href="mailto:reader@example.org">email</a>
<a id="data" href="data:text/plain,hello%20world">data</a>
<a id="empty" href="">empty</a>
<a id="padded-fragment" href=" \t#heading \r\n">padded fragment</a>
<a id="padded-path" href=" \tfile.zip \n">padded path</a>
<a id="newlines" href="fi\nle.\tzip">embedded whitespace</a>
<a id="encoded-space" href="%20file.zip%20">encoded spaces</a>
<a id="nbsp" href="&nbsp;file.zip&nbsp;">non-ASCII spaces</a>
<a id="literal-copy" href="file.zip?x=1&copy=2">literal copy parameter</a>
<a id="literal-not" href="file.zip?x=1&notit=2">literal entity prefix</a>
<a id="literal-amp" href="file.zip?x=1&amp=2">literal amp parameter</a>
<span id="literal-attribute" title='<a href="untouched.txt">literal</a>'>label</span>
<A ID="upper" CLASS="link" HREF=bare.txt>unquoted</A>
<map><area id="map" href="map.html#target" alt="Map"></map>
<img id="image" src="assets/image.png" srcset="assets/a,b.png 1x, data:image/svg+xml,%3Csvg%3E,%3C/svg%3E 2x, ../image.png 3x" alt="image">
<picture><source id="source" srcset='small.png, ../wide.png 2x'></picture>
<video id="video" src="movie.mp4" poster="poster.png"></video>
<audio id="audio" src="sound.mp3"></audio><track id="track" src="captions.vtt">
<svg><defs><path id="symbol" d="M0 0"/></defs><use id="use" href="#symbol" xlink:href="#symbol"/></svg>
<svg><text><![CDATA[<a href="file.zip">literal SVG text</a>]]></text></svg>
<script-note><a id="script-note-link" href="script-note.zip">download</a><script>const note = 1;</script></script-note>
<code-note><a id="code-note-link" href="code-note.zip">download</a><code>note</code></code-note>
<!-- <a href="file.zip">comment</a> -->
<script>const example = '<a href="file.zip">code</a>'; const other='href="fake.txt"';</script>
<style>.icon { background: url(#symbol); }</style>
<pre><code>&lt;a href="file.zip"&gt;encoded&lt;/a&gt;</code></pre>
<textarea><a href="file.zip">textarea</a></textarea>
'''


def attribute_cases():
    """Independent HTML5 entity-table oracle, without parsing attributes as text.

    Python HTMLParser applies text-context unescaping to its attribute values,
    so it cannot supply expected DOM values for the ambiguous cases below.
    The standard-library table distinguishes semicolon and legacy spellings.
    """
    cases = []
    for name, value in sorted(HTML5_ENTITIES.items()):
        if name.endswith(";"):
            cases.append({"id": name, "input": "prefix:&" + name + "=tail",
                          "expected": "prefix:" + value + "=tail"})
        else:
            for suffix, expected in (("", value), ("=2", "&" + name + "=2"),
                                     ("Zzz", "&" + name + "Zzz")):
                cases.append({"id": name + ":" + suffix, "input": "&" + name + suffix,
                              "expected": expected})
    for raw, expected in (
        ("file.zip?x=1&copy=2&notit=3&amp;ok=4", "file.zip?x=1&copy=2&notit=3&ok=4"),
        ("&amp;copy=2", "&copy=2"),
        ("&notit;", "&notit;"),
        ("&#38;", "&"), ("&#x26;", "&"), ("&#X26;", "&"),
        ("&#x1F600;", "😀"), ("&#65", "A"), ("&#65=2", "A=2"),
    ):
        cases.append({"id": raw, "input": raw, "expected": expected})
    return cases


class Elements(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.elements = {}
        self.feed(source)

    def handle_starttag(self, tag, pairs):
        attrs = dict(pairs)
        if attrs.get("id"):
            self.elements[attrs["id"]] = (tag, attrs)


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(hugo, project, destination):
    result = subprocess.run([hugo, "--minify", "--source", str(project),
                             "--destination", str(destination)],
                            capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)


def feed(path):
    root = ET.parse(path).getroot()
    assert root.tag == "rss"
    return root.find("channel")


def check(project, old, new, base):
    decoder_cases = attribute_cases()
    actual_decoder = json.loads((new / "index.html").read_text())
    assert len(actual_decoder) == len(decoder_cases)
    for case, actual in zip(decoder_cases, actual_decoder):
        assert actual == case["expected"], ("HTML5 attribute", case["id"], actual, case["expected"])
    old_feed = feed(old / "tags/fixture/index.xml")
    new_feed = feed(new / "tags/fixture/index.xml")
    old_items, new_items = old_feed.findall("item"), new_feed.findall("item")
    assert len(old_items) == len(new_items) == 50, "Retain the upstream fifty-item limit"
    for before, after in zip(old_items, new_items):
        for key in ("title", "link", "guid", "pubDate", "author"):
            assert before.findtext(key) == after.findtext(key), (key, before.findtext(key))
    assert not any("Draft" in item.findtext("title", "") for item in new_items)
    assert new_feed.findtext("link") == old_feed.findtext("link")
    assert (old / "posts/index.xml").read_bytes() == (new / "posts/index.xml").read_bytes(), "Keep the summary feed unchanged"
    assert len(feed(new / "posts/index.xml").findall("item")) == 3, "Retain summary-feed configuration"
    assert (old / "index.xml").exists() and (new / "index.xml").exists()
    assert [item.findtext("link") for item in feed(old / "index.xml").findall("item")] == [
        item.findtext("link") for item in feed(new / "index.xml").findall("item")]
    for path in old.rglob("*.html"):
        assert path.read_bytes() == (new / path.relative_to(old)).read_bytes(), path
    article = base + "posts/article/"
    before = next(item for item in old_items if item.findtext("link") == article)
    after = next(item for item in new_items if item.findtext("link") == article)
    original = before.findtext("description", "")
    content = after.findtext("description", "")
    assert "Full article body after its summary." in content
    for block in (
        '<!-- <a href="file.zip">comment</a> -->',
        '<script>const example = \'<a href="file.zip">code</a>\'; const other=\'href="fake.txt"\';</script>',
        '<style>.icon { background: url(#symbol); }</style>',
        '<pre><code>&lt;a href="file.zip"&gt;encoded&lt;/a&gt;</code></pre>',
        '<textarea><a href="file.zip">textarea</a></textarea>',
        '<svg><text><![CDATA[<a href="file.zip">literal SVG text</a>]]></text></svg>',
    ):
        assert block in original and block in content, block
    elements = Elements(content).elements
    expected = {
        "download": ("href", article + "file.zip?x=a&y=b#part"),
        "sibling": ("href", base + "posts/sibling/?x=a&amp;y=b#part"),
        "chapter": ("href", article + "#heading"),
        "query": ("href", article + "?x=a+b&y=%2F&z=%26#part"),
        "root": ("href", "https://example.org/root/file.zip"),
        "cdn": ("href", "https://cdn.example.org/image.png"),
        "external": ("href", "https://other.example.org/x?a=1&b=2"),
        "email": ("href", "mailto:reader@example.org"),
        "data": ("href", "data:text/plain,hello%20world"),
        "empty": ("href", article),
        "padded-fragment": ("href", article + "#heading"),
        "padded-path": ("href", article + "file.zip"),
        "newlines": ("href", article + "file.zip"),
        "encoded-space": ("href", article + "%20file.zip%20"),
        "nbsp": ("href", article + "%C2%A0file.zip%C2%A0"),
        "literal-copy": ("href", article + "file.zip?x=1&copy=2"),
        "literal-not": ("href", article + "file.zip?x=1&notit=2"),
        "literal-amp": ("href", article + "file.zip?x=1&amp=2"),
        "script-note-link": ("href", article + "script-note.zip"),
        "code-note-link": ("href", article + "code-note.zip"),
        "upper": ("href", article + "bare.txt"),
        "map": ("href", article + "map.html#target"),
        "image": ("src", article + "assets/image.png"),
        "video": ("src", article + "movie.mp4"),
        "audio": ("src", article + "sound.mp3"),
        "track": ("src", article + "captions.vtt"),
        "use": ("href", "#symbol"),
    }
    for element, (attribute, value) in expected.items():
        assert elements[element][1].get(attribute) == value, (element, elements[element])
    # WHATWG URL parsing is independent of Hugo's Go URL implementation. In
    # particular, legal HTML URL whitespace must not become filename spaces.
    # Parse the original HTML fragment before Hugo's RSS root-URL
    # postprocessor or HTML minifier can normalize those references.
    originals = Elements(BODY).elements
    references = {element: originals[element][1][attribute]
                  for element, (attribute, _) in expected.items() if element != "use"}
    # Attribute-context values independently established by the HTML5 named-
    # reference rule; HTMLParser's context-free unescape would corrupt these.
    references.update({
        "literal-copy": "file.zip?x=1&copy=2",
        "literal-not": "file.zip?x=1&notit=2",
        "literal-amp": "file.zip?x=1&amp=2",
    })
    oracle = subprocess.run(["node", "-e", """
      const fs = require('node:fs');
      const {base, references} = JSON.parse(fs.readFileSync(0, 'utf8'));
      process.stdout.write(JSON.stringify(Object.fromEntries(Object.entries(references)
        .map(([key, value]) => [key, new URL(value, base).href]))));
    """], input=json.dumps({"base": article, "references": references}),
        capture_output=True, text=True, check=True)
    for element, value in json.loads(oracle.stdout).items():
        attribute = expected[element][0]
        assert elements[element][1][attribute] == value, ("WHATWG URL", element, value)
    assert elements["sibling"][1]["title"] == 'literal href="untouched.txt"'
    assert elements["use"][1]["xlink:href"] == "#symbol"
    assert elements["literal-attribute"][1]["title"] == '<a href="untouched.txt">literal</a>'
    assert elements["video"][1]["poster"] == article + "poster.png"
    assert elements["image"][1]["srcset"] == (
        article + "assets/a,b.png 1x, data:image/svg+xml,%3Csvg%3E,%3C/svg%3E 2x, "
        + base + "posts/image.png 3x")
    assert elements["source"][1]["srcset"] == article + "small.png, " + base + "posts/wide.png 2x"
    assert 'href="file.zip?x=a&amp;y=b#part"' in original, "The fixture must reproduce the broken relative download"
    return len(expected) + 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hugo", default="hugo")
    args = parser.parse_args()
    assertions = 0
    with tempfile.TemporaryDirectory(prefix="chase-rss-regression-") as directory:
        work = Path(directory)
        project = work / "project"
        # Use the real templates/partials, with tiny HTML templates and no theme assets.
        for name in ("list.rss.xml", "blog/list.rss.xml", "_partials/blog/rss-content.html", "_partials/blog/rss-attribute.html",
                     "_partials/blog/rss-url.html", "_partials/blog/rss-srcset.html"):
            target = project / "layouts" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "layouts" / name, target)
        compat = "_partials/utils/hugo-compat/language-locale.html"
        write(project, "layouts/" + compat, (ROOT / "themes/hextra/layouts" / compat).read_text())
        for name in ("home.html", "single.html", "list.html"):
            write(project, "layouts/" + name, "<!doctype html><html><body>{{ .Content }}</body></html>")
        # Render the real attribute partial in Hugo against an independent
        # complete HTML5 named-entity table, without optional parser packages.
        write(project, "data/attribute_cases.json", json.dumps(attribute_cases()))
        write(project, "layouts/home.html", '{{- $values := slice -}}{{- range .Site.Data.attribute_cases -}}'
              '{{- $values = $values | append (partial "blog/rss-attribute.html" .input) -}}'
              '{{- end -}}{{ $values | jsonify | safeHTML }}')
        write(project, "content/posts/_index.md", "---\ntitle: Posts\ntype: blog\n---\n")
        write(project, "content/posts/article/index.md", "---\ntitle: RSS fixture\ndate: 2026-03-01\ntype: blog\ntags: [fixture]\nsummary: Fixture summary only\n---\n" + BODY)
        for i in range(50):
            day = date(2026, 1, 1) + timedelta(days=i)
            write(project, f"content/posts/entry-{i}/index.md", f"---\ntitle: Entry {i}\ndate: {day}\ntype: blog\ntags: [fixture]\n---\nEntry body {i}.\n")
        write(project, "content/posts/draft/index.md", "---\ntitle: Draft fixture\ndate: 2026-03-02\ntype: blog\ntags: [fixture]\ndraft: true\n---\nDraft body.\n")
        original_template = (ROOT / "themes/hextra/layouts/list.rss.xml").read_text()
        fixed_template = (ROOT / "layouts/list.rss.xml").read_text()
        for prefix in ("", "project/"):
            base = "https://example.org/" + prefix
            write(project, "hugo.yaml", f"baseURL: {base}\ntitle: RSS regression\noutputs:\n  home: [html, rss]\n  section: [html, rss]\nservices:\n  rss:\n    limit: 3\nmarkup:\n  goldmark:\n    renderer:\n      unsafe: true\nminify:\n  disableXML: true\n")
            old, new = work / ("old-" + ("prefix" if prefix else "root")), work / ("new-" + ("prefix" if prefix else "root"))
            write(project, "layouts/list.rss.xml", original_template)
            build(args.hugo, project, old)
            write(project, "layouts/list.rss.xml", fixed_template)
            build(args.hugo, project, new)
            assertions += check(project, old, new, base)
    print(json.dumps({"base_url_variants": 2, "url_and_preservation_checks": assertions,
                      "original_url_and_preservation_checks_retained": 54,
                      "attribute_entity_cases_per_base_url": len(attribute_cases()),
                      "attribute_oracle": "Python standard-library HTML5 named-reference table and attribute-context cases",
                      "upstream_full_feed_limit": 50, "configured_summary_feed_limit": 3,
                      "xml_and_html_parsed": True, "url_oracle": "Node WHATWG URL", "article_html_and_summary_feed": "unchanged",
                      "protected_blocks_and_svg_local_references": "unchanged"}, indent=2))


if __name__ == "__main__":
    main()
