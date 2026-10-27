#!/usr/bin/env python3
"""Build image fixtures for stable article geometry and explicit opt-outs."""
import argparse
import json
import shutil
import struct
import subprocess
import tempfile
import zlib
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wide_png():
    """A small, deterministic RGB fixture independent of article illustrations."""
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    header = struct.pack(">IIBBBBB", 640, 320, 8, 2, 0, 0, 0)
    pixels = (b"\0" + b"\x28\x55\xaa" * 640) * 320
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


class Images(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.images = {}
        self.meta = {}
        self.zoom_links = []
        self.link_depth = 0
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            item = dict(attrs)
            key = item.get("property", item.get("name"))
            self.meta[key] = item.get("content")
        if tag == "a":
            self.link_depth += 1
            assert self.link_depth == 1, ("nested image links", attrs)
            item = dict(attrs)
            if "post-image-link" in item.get("class", "").split():
                self.zoom_links.append(item)
        if tag == "img":
            keys = [key for key, _ in attrs]
            assert len(keys) == len(set(keys)), ("duplicate attribute", attrs)
            item = dict(attrs)
            self.images[item.get("alt")] = item

    def handle_endtag(self, tag):
        if tag == "a":
            self.link_depth -= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hugo", default="hugo")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="chase-image-regression-") as directory:
        work = Path(directory)
        content = work / "content"
        fixtures = [("posts/default", None, True, True), ("posts/opt-out", False, False, True),
                    ("posts/opt-in", True, True, True), ("notes/default", None, False, True),
                    ("posts/no-zoom", None, True, False)]
        for relative, option, reserve, zoom in fixtures:
            bundle = content / relative
            bundle.mkdir(parents=True)
            shutil.copyfile(ROOT / "content/orca-avatar-64.webp", bundle / "small.webp")
            (bundle / "wide.png").write_bytes(wide_png())
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
            if not zoom:
                setting += "imageZoom: false\n"
            (bundle / "index.md").write_text(
                "---\ntitle: Image fixture\ndraft: true\nimages: [small.webp]\n" + setting + "---\n\n"
                '![raster](small.webp "Small raster")\n\n'
                "![points](points.svg)\n\n![ratio](ratio.svg)\n\n![scientific](scientific.svg)\n\n"
                '{{< post-image src="/external.svg" alt="explicit" width="240" height="120" >}}\n\n'
                '![unknown](https://example.com/unknown.webp)\n\n'
                '[![linked](small.webp)](https://example.com/paper)\n')
            with (bundle / "index.md").open("a") as stream:
                stream.write('\n![responsive](wide.png)\n')
        subprocess.run([args.hugo, "--minify", "-D", "--source", str(ROOT),
                        "--contentDir", str(content), "--destination", str(work / "public")],
                       check=True, capture_output=True, text=True)
        for relative, _, reserve, zoom in fixtures:
            rendered = (work / "public" / relative / "index.html").read_text()
            assert "ZgotmplZ" not in rendered, "Unsafe or discarded image sizing"
            parsed = Images(rendered)
            if relative.startswith("posts/"):
                assert parsed.meta["og:image:width"] == "64"
                assert parsed.meta["og:image:height"] == "64"
                assert parsed.meta["og:image:type"] == "image/webp"
                # Reusing a resource in a later link must keep its first description.
                assert parsed.meta["og:image:alt"] == "raster"
                assert parsed.meta["twitter:image:alt"] == "raster"
            images = parsed.images
            assert len(parsed.zoom_links) == (7 if zoom else 0)
            assert all(link.get("target") == "_blank" and link.get("aria-label") for link in parsed.zoom_links)
            assert {link["href"] for link in parsed.zoom_links} == ({images[name]["src"] for name in
                ("raster", "points", "ratio", "scientific", "explicit", "unknown", "responsive")} if zoom else set())
            assert images["responsive"]["src"].endswith("/wide.png")
            candidates = [part.strip().split() for part in images["responsive"]["srcset"].split(",")]
            assert all(url.endswith(".webp") for url, _ in candidates)
            assert candidates[-1][1] == images["responsive"]["width"] + "w"
            assert images["responsive"]["sizes"]
            if zoom:
                assert images["responsive"]["data-zoom-src"] == images["responsive"]["src"]
            else:
                assert "data-zoom-src" not in images["responsive"]
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
    print(json.dumps({"page_modes": len(fixtures), "images_per_mode": 8,
                      "full_width_png_webp_and_original_links": "passed",
                      "responsive_zoom_uses_original": "passed",
                      "svg_points_fractional_scientific_multiline_viewbox": "passed",
                      "small_image_cap_and_opt_out": "passed",
                      "native_zoom_links_and_author_links": "passed"}, indent=2))


if __name__ == "__main__":
    main()
