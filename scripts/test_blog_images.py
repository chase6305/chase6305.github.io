#!/usr/bin/env python3
"""Build image fixtures for stable article geometry and explicit opt-outs."""
import argparse
import json
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Images(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.images = {}
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            keys = [key for key, _ in attrs]
            assert len(keys) == len(set(keys)), ("duplicate attribute", attrs)
            item = dict(attrs)
            self.images[item.get("alt")] = item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hugo", default="hugo")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="chase-image-regression-") as directory:
        work = Path(directory)
        content = work / "content"
        fixtures = [("posts/default", None, True), ("posts/opt-out", False, False),
                    ("posts/opt-in", True, True), ("notes/default", None, False)]
        for relative, option, reserve in fixtures:
            bundle = content / relative
            bundle.mkdir(parents=True)
            shutil.copyfile(ROOT / "content/orca-avatar-64.webp", bundle / "small.webp")
            (bundle / "points.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="72pt" height="36pt" '
                'viewBox="0 0 72 36"><rect width="72" height="36"/></svg>')
            (bundle / "ratio.svg").write_text(
                "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0, 0, 125.5, 50.2'>"
                "<rect width='125.5' height='50.2'/></svg>")
            (bundle / "scientific.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1e-2\n+2e-3\t1.25e2 5.02e1">'
                '<rect width="125" height="50.2"/></svg>')
            setting = "" if option is None else f"reserveFigureSpace: {str(option).lower()}\n"
            (bundle / "index.md").write_text(
                "---\ntitle: Image fixture\ndraft: true\n" + setting + "---\n\n"
                '![raster](small.webp "Small raster")\n\n'
                "![points](points.svg)\n\n![ratio](ratio.svg)\n\n![scientific](scientific.svg)\n\n"
                '{{< post-image src="/external.svg" alt="explicit" width="240" height="120" >}}\n\n'
                '![unknown](https://example.com/unknown.webp)\n')
        subprocess.run([args.hugo, "--minify", "-D", "--source", str(ROOT),
                        "--contentDir", str(content), "--destination", str(work / "public")],
                       check=True, capture_output=True, text=True)
        for relative, _, reserve in fixtures:
            images = Images((work / "public" / relative / "index.html").read_text()).images
            assert images["raster"]["width"] == "64"
            assert images["raster"]["height"] == "64"
            assert "width" not in images["unknown"]
            assert "aspect-ratio" not in images["unknown"].get("style", "")
            for name in ("raster", "points", "ratio", "scientific", "explicit"):
                assert int(images[name]["width"]) > 0
                assert int(images[name]["height"]) > 0
                style = images[name].get("style", "")
                assert ("aspect-ratio" in style) == reserve, (relative, name, style)
            if reserve:
                assert "64px" in images["raster"]["style"]  # Do not enlarge icons.
                assert "96px" in images["points"]["style"]  # SVG points are not pixels.
                assert "125.5/50.2" in images["ratio"]["style"].replace(" ", "")
                assert "125/50.2" in images["scientific"]["style"].replace(" ", "")
                assert "240px" in images["explicit"]["style"]
    print(json.dumps({"page_modes": len(fixtures), "images_per_mode": 6,
                      "svg_points_fractional_scientific_multiline_viewbox": "passed",
                      "small_image_cap_and_opt_out": "passed"}, indent=2))


if __name__ == "__main__":
    main()
