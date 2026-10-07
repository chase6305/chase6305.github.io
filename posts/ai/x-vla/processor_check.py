#!/usr/bin/env python3
"""Compare X-VLA's augmentation-free image transform with a local CLIP config.

Optional diagnostic, separate from the NumPy-only downloadable labs.
Requires torch, torchvision, transformers, numpy and Pillow; no model weights.
Run: python processor_check.py --config preprocessor_config.json --output check.json
The four synthetic RGB fixtures exercise resizing and normalization, not policy quality.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPImageProcessor


def fixtures():
    yield "black", np.zeros((320, 480, 3), dtype=np.uint8)
    yield "white", np.full((480, 320, 3), 255, dtype=np.uint8)
    y, x = np.indices((317, 479))
    yield "rgb_ramps", np.stack((x % 256, y % 256, (3 * x + 5 * y) % 256), axis=-1).astype(np.uint8)
    checks = (((x // 4 + y // 4) % 2) * 255).astype(np.uint8)
    yield "checkerboard", np.stack((checks, 255 - checks, checks), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and nonnegative")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config.get("image_processor_type") != "CLIPImageProcessor":
        parser.error("this comparison is only defined for CLIPImageProcessor")
    processor = CLIPImageProcessor.from_dict(config)
    training = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225), inplace=True),
    ])
    rows = []
    for name, array in fixtures():
        image = Image.fromarray(array)
        actual = processor(images=image, return_tensors="pt")["pixel_values"][0]
        expected = training(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=args.tolerance)
        assert actual.shape == (3, 224, 224)
        error = (actual - expected).abs()
        rows.append({"fixture": name, "input_shape": list(array.shape),
                     "output_shape": list(actual.shape),
                     "max_absolute_error": float(error.max()),
                     "mean_absolute_error": float(error.mean())})
    report = {
        "scope": "CPU image preprocessing only; no XVLAProcessor tokenizer, HTTP transport, weights or robot",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": config,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "torchvision", "transformers", "numpy", "Pillow")},
        "absolute_tolerance": args.tolerance,
        "results": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
