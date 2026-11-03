"""Check label/value pairs in native PDFs, including wrapped Chinese labels."""

import json
import sys
import xml.etree.ElementTree as ET


def compact(value):
    return "".join(value.split())


def matches(words, text):
    """Follow adjacent words on a line, or the next wrapped line."""
    text = compact(text)

    def walk(group, remaining):
        if not remaining:
            yield {key: reducer(w[key] for w in group) for key, reducer in (
                ("xMin", min), ("yMin", min), ("xMax", max), ("yMax", max))}
            return
        if len(group) >= 48:
            return
        previous = group[-1] if group else None
        for word in words:
            part = compact(word["text"])
            if not part or not remaining.startswith(part) or word in group:
                continue
            if previous:
                height = max(word["yMax"]-word["yMin"], previous["yMax"]-previous["yMin"])
                dy = (word["yMin"]+word["yMax"]-previous["yMin"]-previous["yMax"])/2
                same_line = abs(dy) < height*.3 and -1 <= word["xMin"]-previous["xMax"] < height*.6
                next_line = height*.3 <= dy < height*1.8 and abs(word["xMin"]-group[0]["xMin"]) < 3
                if not (same_line or next_line):
                    continue
            yield from walk(group+[word], remaining[len(part):])
    yield from walk([], text)


def inspect(xml, cards):
    root = ET.fromstring(xml)
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    pages = [[dict(text=w.text or "", **{key: float(value) for key, value in w.attrib.items()})
              for w in page.findall(".//x:word", ns)] for page in root.findall(".//x:page", ns)]
    rows = []
    for card in cards:
        locations = []
        for number, words in enumerate(pages, 1):
            labels, values = list(matches(words, card["label"])), list(matches(words, card["value"]))
            for label in labels:
                for value in values:
                    # The component prints each definition as a two-cell row.
                    # Different label/value font sizes have different ascent
                    # bounds even when their CSS table cells share a top edge.
                    if value["xMin"] > label["xMin"]+20 and abs(value["yMin"]-label["yMin"]) < max(3, .2*(label["yMax"]-label["yMin"])):
                        locations.append({"page": number, "label": label, "value": value})
        rows.append({**card, "passed": bool(locations), "locations": locations})
    return {"pages": len(pages), "cards": rows, "passed": all(row["passed"] for row in rows)}


if __name__ == "__main__":
    print(json.dumps(inspect(sys.stdin.buffer.read(), json.loads(sys.argv[1])), ensure_ascii=False))
