#!/usr/bin/env python3
"""Build native table/math focus and labeled task checks with disposable content."""
import argparse
import json
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = r'''---
title: Native reading fixture
type: blog
draft: true
math: true
reading_focus: Temporary fixture
reading_prerequisites: Markdown
---

## Native content

Inline $q_1$ remains inline.

$$
A = \begin{bmatrix}1&2\\3&4\end{bmatrix}
$$

| Meaning | Value |
| :--- | ---: |
| **Matrix** | $q_1$ |
{#native-table .table-readable}

| Author setting | Value |
| --- | --- |
| Explicit | 1 |
{#author-table tabindex="-1"}

| Author zero | Value |
| --- | --- |
| Explicit | 0 |
{#zero-table tabindex="0"}

- {{< task-check >}}Pending `x < 1` and $q_1$.{{< /task-check >}}
- {{< task-check checked=true >}}Finished **bold** with [a link](https://example.com/).{{< /task-check >}}
'''


class ReadingPage(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.tables = []
        self.displays = []
        self.checks = []
        self.labels = 0
        self.alignments = []
        self.feed(source)

    def handle_starttag(self, tag, pairs):
        attrs = dict(pairs)
        assert len(pairs) == len(attrs), (tag, 'duplicate attributes', pairs)
        if tag == 'label':
            self.labels += 1
        if tag == 'table':
            self.tables.append(attrs)
        if tag == 'span' and 'katex-display' in attrs.get('class', '').split():
            self.displays.append(attrs)
        if tag in ('th', 'td') and attrs.get('style'):
            self.alignments.append(attrs['style'].replace(' ', ''))
        if tag == 'input' and attrs.get('type') == 'checkbox':
            self.checks.append({'in_label': self.labels > 0, **attrs})

    def handle_endtag(self, tag):
        if tag == 'label':
            self.labels -= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hugo', default='hugo')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='chase-native-reading-') as name:
        folder = Path(name)
        content = folder / 'content'
        post = content / 'posts/native-fixture/index.md'
        post.parent.mkdir(parents=True)
        post.write_text(FIXTURE)
        shutil.copy2(ROOT / 'content/posts/_index.md', content / 'posts/_index.md')
        subprocess.run([args.hugo, '--minify', '-D', '--source', str(ROOT),
                        '--contentDir', str(content), '--destination', str(folder / 'public')],
                       check=True, text=True, capture_output=True)
        source = (folder / 'public/posts/native-fixture/index.html').read_text()
        page = ReadingPage(source)
        tables = {table['id']: table for table in page.tables}
        assert tables['native-table']['tabindex'] == '0'
        assert 'table-readable' in tables['native-table']['class']
        assert tables['author-table']['tabindex'] == '-1'
        assert tables['zero-table']['tabindex'] == '0'
        assert {'text-align:left', 'text-align:right'} <= set(page.alignments)
        assert len(page.displays) == 1 and page.displays[0]['tabindex'] == '0'
        assert len(page.checks) == 2 and all(item['in_label'] and 'disabled' in item for item in page.checks)
        assert 'checked' not in page.checks[0] and 'checked' in page.checks[1]
        assert 'Pending' in source and 'Finished' in source and '<code>x &lt; 1</code>' in source
        assert '<strong>bold</strong>' in source and 'href=https://example.com/' in source
    print(json.dumps({'tables': 3, 'alignment_and_attributes': 'preserved',
                      'display_math_focus': True, 'task_states': ['unchecked', 'checked'],
                      'implicit_labels_and_inline_markup': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
