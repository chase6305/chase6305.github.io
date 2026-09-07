#!/usr/bin/env python3
"""Download only pinned config/tokenizer/processor files, NEVER model weights.

Requires the model's documented Transformers version and a local PyTorch runtime.
Checks input construction on CPU; this is not model inference or an evaluation.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("rynnbrain", "internvl-3-5"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--exercise-cli", action="store_true",
                        help="Also check prepare-only CLI with model loading forbidden")
    args = parser.parse_args()
    script = ROOT / "content/posts/ai" / args.model / "infer_image.py"
    spec = importlib.util.spec_from_file_location("blog_inference", script)
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)

    import transformers
    from PIL import Image
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    expected_version = "5.2.0" if args.model == "rynnbrain" else "4.55.0"
    if transformers.__version__ != expected_version:
        parser.error(f"Use transformers=={expected_version} for this pinned interface check")
    common = dict(revision=inference.REVISION, trust_remote_code=False,
                  local_files_only=args.local_files_only)
    config = AutoConfig.from_pretrained(inference.MODEL_ID, **common)
    # Resolve native implementation without constructing a model or its parameters.
    implementation = AutoModelForImageTextToText._model_mapping[type(config)]
    processor = AutoProcessor.from_pretrained(inference.MODEL_ID, **common)
    image_path = ROOT / "content/posts/ai/transformer-attention/assets/transformer-block-overview.webp"
    with Image.open(image_path) as source:
        picture = source.convert("RGB")
    inputs = inference.build_inputs(processor, picture, "Describe the diagram.")
    if args.model == "rynnbrain":
        rendered = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        assert rendered.endswith("<think>\n\n</think>\n\n"), "Non-thinking prefix not applied"
    else:
        assert (inputs["input_ids"] == config.image_token_id).sum().item() == (
            inputs["pixel_values"].shape[0] * config.image_seq_length)
    assert inputs["input_ids"].shape[0] == 1
    assert "pixel_values" in inputs
    assert inputs["pixel_values"].numel() > 0
    for name, value in inputs.items():
        if value.is_floating_point():
            assert value.isfinite().all(), name
    cli_checks = None
    if args.exercise_cli:
        command = ["--image", str(image_path), "--question", "Describe the diagram.",
                   "--prepare-only", "--local-files-only"]
        output = io.StringIO()
        with patch.object(AutoModelForImageTextToText, "from_pretrained",
                          side_effect=AssertionError("Prepare-only tried to load model weights")) as loader:
            with patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(output):
                inference.main(command)
            cli_checks = json.loads(output.getvalue())
            assert cli_checks["input_tokens"] == inputs["input_ids"].shape[1]
            assert not cli_checks["model_weights_loaded"] and not cli_checks["model_forward_executed"]
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    inference.main(command + ["--max-input-tokens", "1"])
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError("Excessive input budget was not rejected")
            loader.assert_not_called()
        cli_checks["budget_rejected_before_weight_loading"] = True
        with patch.object(AutoModelForImageTextToText, "from_pretrained",
                          side_effect=AssertionError("Invalid input tried to load weights")), \
             patch.object(AutoProcessor, "from_pretrained",
                          side_effect=AssertionError("Invalid input reached the Processor loader")):
            failures = [
                (command + ["--image", str(script.with_name("index.md"))], contextlib.nullcontext(), "Cannot decode image"),
                (command, patch("PIL.Image.Image.getexif", return_value={274: 6}), "EXIF orientation"),
                (command, patch("transformers.__version__", "0.0.0"), "requires transformers"),
                ([arg for arg in command if arg != "--prepare-only"], patch("torch.cuda.is_available", return_value=False), "CUDA GPU"),
            ]
            for bad_command, context, expected_message in failures:
                stderr = io.StringIO()
                with context, contextlib.redirect_stderr(stderr):
                    try:
                        inference.main(bad_command)
                    except SystemExit as error:
                        assert error.code == 2
                        assert expected_message in stderr.getvalue()
                    else:
                        raise AssertionError(f"Expected rejection: {expected_message}")
        cli_checks["early_error_guards"] = ["invalid_image", "exif_orientation", "library_version", "missing_cuda"]
    print(json.dumps({
        "model": inference.MODEL_ID, "revision": inference.REVISION,
        "transformers": transformers.__version__, "implementation": implementation.__name__,
        "processor": type(processor).__name__,
        "inference_input_builder_tested": True,
        "prepare_only_cli": cli_checks,
        "inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in inputs.items()},
        "model_weights_downloaded": False, "model_forward_executed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
