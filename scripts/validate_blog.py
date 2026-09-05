#!/usr/bin/env python3
"""Read-only checks for reviewed posts and Hugo's generated local links."""
import argparse
import ast
import json
import os
import re
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def code_blocks(text):
    """Yield language, content, and opening line for fenced code (including lists)."""
    fence = None
    lines = []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence is None:
            if re.match(r"^\s*(?:\\`){3,}", line):
                raise ValueError(f"Escaped code fence at line {number}; this renders as prose")
            if match:
                fence, info = match.groups()
                language = info.strip().split()[0] if info.strip() else ""
                opening = number
                lines = []
        elif match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
            yield language, textwrap.dedent("\n".join(lines)) + "\n", opening
            fence = None
        else:
            lines.append(line)
    if fence:
        raise ValueError(f"Unclosed fence at line {opening}")


def parse_jsonc(code):
    """Accept comments/trailing commas without corrupting quoted URLs or text."""
    tokens = r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/'
    code = re.sub(tokens, lambda m: m[0] if m[0].startswith('"') else
                  re.sub(r"[^\n]", " ", m[0]), code)
    code = re.sub(r'("(?:\\.|[^"\\])*")|,\s*(?=[}\]])',
                  lambda m: m[1] or "", code)
    return json.loads(code)


def check_structured_snippet(language, code):
    """Syntax checks only: never execute article shell commands."""
    if language in ("bash", "sh"):
        env = dict(os.environ, BASH_ENV="", ENV="")
        command = ["bash", "--noprofile", "--norc", "-n"] if language == "bash" else ["sh", "-n"]
        result = subprocess.run(command, input=code, text=True, capture_output=True,
                                timeout=10, env=env)
        if result.returncode:
            raise ValueError(result.stderr.strip())
    elif language == "json":
        json.loads(code)
    elif language == "jsonc":
        parse_jsonc(code)
    elif language == "xml":
        ET.fromstring(code)
    else:
        return False
    return True


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.duplicate_ids = set()
        self.links = []
        self.images = []
        self.schemas = []
        self.filter_data = None
        self.has_draft_notice = False
        self.topic_links = []
        self._in_topic_nav = False
        self._schema_text = None
        self._filter_text = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "nav":
            self._in_topic_nav = "blog-topic-nav" in attrs.get("class", "").split()
        if tag == "a" and self._in_topic_nav:
            self.topic_links.append(attrs)
        if tag == "script" and attrs.get("type") == "application/ld+json":
            self._schema_text = ""
        if tag == "script" and attrs.get("id") == "blog-filter-data":
            self._filter_text = ""
        if "blog-draft-notice" in attrs.get("class", "").split():
            self.has_draft_notice = True
        if attrs.get("id"):
            if attrs["id"] in self.ids:
                self.duplicate_ids.add(attrs["id"])
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in ("img", "script", "source") and attrs.get("src"):
            self.links.append(attrs["src"])
        if tag == "img":
            self.images.append(attrs)
            for candidate in attrs.get("srcset", "").split(","):
                if candidate.strip():
                    self.links.append(candidate.strip().split()[0])

    def handle_data(self, data):
        if self._schema_text is not None:
            self._schema_text += data
        if self._filter_text is not None:
            self._filter_text += data

    def handle_endtag(self, tag):
        if tag == "nav":
            self._in_topic_nav = False
        if tag == "script" and self._schema_text is not None:
            try:
                self.schemas.append(json.loads(self._schema_text))
            except json.JSONDecodeError:
                self.schemas.append({"invalid_json": True})
            self._schema_text = None
        if tag == "script" and self._filter_text is not None:
            try:
                self.filter_data = json.loads(self._filter_text)
            except json.JSONDecodeError:
                self.filter_data = "invalid JSON"
            self._filter_text = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, default=ROOT / "public")
    parser.add_argument("--python-snippets", action="store_true")
    parser.add_argument("--structured-snippets", action="store_true",
                        help="Parse Bash/sh, JSON/JSONC and XML; do not execute commands")
    parser.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args()
    review = json.loads((ROOT / "docs/blog-editorial-review.json").read_text())
    errors = []
    snippets = []
    for artifact in (ROOT / "content/posts").rglob("*.pyc"):
        errors.append(f"Python bytecode must not be published as a blog resource: {artifact}")
    actual = set((ROOT / "content/posts").rglob("*.md"))
    actual = {path for path in actual if path.name != "_index.md"}
    expected = {ROOT / post["path"] for post in review["posts"]}
    records = {ROOT / post["path"]: post for post in review["posts"]}
    second = json.loads((ROOT / "docs/blog-second-pass.json").read_text())
    self_checks = {ROOT / p["path"]: p["acceptance_checks"] for p in second["posts"]}
    if set(self_checks) != expected:
        errors.append("Second-pass coverage differs from the article inventory")
    if actual != expected:
        errors.append(f"Review coverage differs: {actual ^ expected}")
    topics = json.loads((ROOT / "data/blog_topics.json").read_text())
    topic_posts = [ROOT / "content" / path for topic in topics for path in topic["posts"]]
    if set(topic_posts) != actual or len(topic_posts) != len(actual):
        errors.append("Learning paths must cover every article exactly once")
    topic_ids = [topic["id"] for topic in topics]
    if len(set(topic_ids)) != len(topic_ids) or any(not re.fullmatch(r"[a-z0-9-]+", value) for value in topic_ids):
        errors.append("Invalid or duplicated learning-path anchor")
    block_count = 0
    syntax_counts = Counter()
    for path in sorted(actual):
        text = path.read_text(encoding="utf-8")
        if text.count("## 阅读自测与验收") != 1 or any(
            check not in text for check in self_checks.get(path, [])
        ):
            errors.append(f"Missing or duplicated acceptance section: {path}")
        if not re.match(r"^---\n[\s\S]*?\n---\n", text):
            errors.append(f"Missing YAML front matter: {path}")
        else:
            front = text.split("---", 2)[1]
            record = records.get(path)
            if record:
                for key, expected_value in [
                    ("summary", record["summary"]),
                    ("description", record["summary"]),
                    ("reading_prerequisites", record["prerequisites"]),
                    ("reading_focus", record["focus"]),
                    ("contentLanguage", "zh-CN"),
                ]:
                    field = re.search(rf"^{key}: (.+)$", front, re.MULTILINE)
                    try:
                        actual_value = json.loads(field[1]) if field else None
                    except json.JSONDecodeError:
                        actual_value = None
                    if actual_value != expected_value:
                        errors.append(f"Editorial metadata mismatch: {path}: {key}")
                draft = bool(re.search(r"^draft: true$", front, re.MULTILINE))
                if draft != record["draft"]:
                    errors.append(f"Draft state changed: {path}")
                route = path.relative_to(ROOT / "content").as_posix()
                route = re.sub(r"/index\.md$|\.md$", "", route).lower()
                built = (args.public / route / "index.html").exists()
                if built != (not draft or args.include_drafts):
                    errors.append(f"Unexpected published/draft output: {route}")
        try:
            for language, code, line in code_blocks(text):
                block_count += 1
                if not language:
                    errors.append(f"Unlabelled code fence: {path.relative_to(ROOT)}:{line}")
                if args.python_snippets and language in ("python", "py"):
                    try:
                        ast.parse(code)
                        syntax_counts["python"] += 1
                    except SyntaxError as error:
                        snippets.append(f"{path.relative_to(ROOT)}:{line}: {error.msg} (block line {error.lineno})")
                if args.structured_snippets:
                    try:
                        if check_structured_snippet(language, code):
                            syntax_counts[language] += 1
                    except (ValueError, ET.ParseError, OSError, subprocess.TimeoutExpired) as error:
                        errors.append(f"Invalid {language} snippet: {path.relative_to(ROOT)}:{line}: {error}")
        except ValueError as error:
            errors.append(f"{path}: {error}")
    images = json.loads((ROOT / "docs/blog-image-prompts.json").read_text())["images"]
    for image in images:
        path = ROOT / image["path"]
        post = path.parent.parent / "index.md"
        if not path.is_file() or not post.is_file() or f"assets/{path.name}" not in post.read_text():
            errors.append(f"Generated image missing or unused: {path}")
    public = args.public.resolve()
    html = {path: Page(path.read_text(encoding="utf-8")) for path in public.rglob("*.html")}
    if not html:
        errors.append("No Hugo output; run hugo --minify first")
    learning = html.get(public / "learning-paths/index.html")
    if learning is None:
        errors.append("Missing learning-path index")
    index_page = html.get(public / "posts/index.html")
    if index_page is None or index_page.filter_data is None:
        errors.append("Missing full-article filter inventory")
    def article_route(path):
        value = path.relative_to(ROOT / "content").as_posix()
        return "/" + re.sub(r"/index\.md$|\.md$", "", value).lower() + "/"
    for topic in topics:
        paths = [ROOT / "content" / path for path in topic["posts"]]
        visible = [path for path in paths if args.include_drafts or not records[path]["draft"]]
        if learning is not None:
            if topic["id"] not in learning.ids:
                errors.append(f"Missing topic anchor: {topic['id']}")
            for path in paths:
                count = learning.links.count(article_route(path))
                if count != (1 if path in visible else 0):
                    errors.append(f"Incorrect topic listing or draft leak: {path}")
        for index, path in enumerate(visible):
            page = html.get(public / article_route(path).lstrip("/") / "index.html")
            if page is None:
                continue
            expected_links = [("", "/learning-paths/#" + topic["id"])]
            if index > 0:
                expected_links.append(("prev", article_route(visible[index-1])))
            if index + 1 < len(visible):
                expected_links.append(("next", article_route(visible[index+1])))
            actual_links = [(a.get("rel", ""), a.get("href")) for a in page.topic_links]
            if actual_links != expected_links:
                errors.append(f"Incorrect topic neighbors: {path}: {actual_links}")
    schema_count = 0
    for path, record in records.items():
        if record["draft"] and not args.include_drafts:
            continue
        route = path.relative_to(ROOT / "content").as_posix()
        route = re.sub(r"/index\.md$|\.md$", "", route).lower()
        page = html.get(public / route / "index.html")
        if page is None:
            continue  # Missing output is already reported above.
        if page.has_draft_notice != record["draft"]:
            errors.append(f"Incorrect draft preview notice: {route}")
        schemas = [s for s in page.schemas if isinstance(s, dict)
                   and s.get("@type") == "BlogPosting"]
        if (len(schemas) != 1 or schemas[0].get("inLanguage") != "zh-CN"
                or not schemas[0].get("author")
                or not schemas[0].get("datePublished")
                or not schemas[0].get("dateModified")):
            errors.append(f"Missing or invalid BlogPosting metadata: {route}")
        else:
            schema_count += 1
    checked = 0
    collection_pages = 0
    filter_indexes = 0
    for path, page in html.items():
        if not (any(path.is_relative_to(public / section) for section in
                    ("posts", "tags", "categories", "series", "archives")) or
                path == public / "learning-paths/index.html"):
            continue
        collection_pages += 1
        relative = path.relative_to(public).as_posix()
        if page.filter_data is not None:
            filter_indexes += 1
            data = page.filter_data
            visible = {article_route(p): record for p, record in records.items()
                       if args.include_drafts or not record["draft"]}
            if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
                errors.append(f"Invalid filter data: {relative}")
            elif len(data) != len(visible) or {row.get("url") for row in data} != set(visible):
                errors.append(f"Filter coverage or draft leak: {relative}")
            else:
                for row in data:
                    if row.get("draft") != visible[row["url"]]["draft"]:
                        errors.append(f"Incorrect filter draft flag: {relative}: {row['url']}")
                    page.links.append(row["url"])
                    page.links.extend(tag["url"] for tag in row.get("tags", []))
        if page.duplicate_ids:
            errors.append(f"Duplicate HTML ids: {relative}: {sorted(page.duplicate_ids)}")
        base = "https://chase6305.github.io/" + relative
        for destination in page.links:
            url = urlsplit(urljoin(base, destination))
            if url.scheme not in ("http", "https") or url.netloc != "chase6305.github.io":
                continue
            target = public / unquote(url.path).lstrip("/")
            if target.is_dir():
                target = target / "index.html"
            checked += 1
            if not target.exists():
                errors.append(f"Missing local target: {relative} -> {destination}")
            elif url.fragment and target in html and unquote(url.fragment) not in html[target].ids:
                errors.append(f"Missing anchor: {relative} -> {destination}")
        for image in page.images:
            if "alt" not in image:
                errors.append(f"Missing image alt: {relative} -> {image.get('src')}")
    print(json.dumps({"reviewed_posts": len(expected), "learning_paths": len(topics), "acceptance_checks": sum(map(len, self_checks.values())),
                      "blog_posting_schemas": schema_count, "code_blocks": block_count,
                      "syntax_checked_by_language": dict(sorted(syntax_counts.items())),
                      "html_pages": len(html), "local_references_checked": checked,
                      "collection_pages_checked": collection_pages, "filter_indexes_checked": filter_indexes,
                      "errors": sorted(set(errors)), "python_fragment_warnings": snippets},
                     ensure_ascii=False, indent=2))
    return bool(errors or snippets)


if __name__ == "__main__":
    raise SystemExit(main())
