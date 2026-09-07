"""Minimal RynnBrain 1.1 image inference; never sends robot commands.

Weights are downloaded only after CLI validation and the CUDA check.
Full-weight inference has not been executed as part of the blog's CPU tests.
"""

import argparse
import json
from pathlib import Path

MODEL_ID = "Alibaba-DAMO-Academy/RynnBrain1.1-2B"
REVISION = "b4663299019d15468301f64435539632daed7985"


def build_inputs(processor, picture, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": picture},
        {"type": "text", "text": question},
    ]}]
    # Keep template-only options out of the image processor's kwargs in 5.2.0.
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False,
    )
    return processor(text=[prompt], images=[picture], return_tensors="pt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", default="Describe the objects and their spatial relationships.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--prepare-only", action="store_true",
                        help="CPU input inspection only; never load model weights")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Use cached checkpoint files only; do not download missing files")
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error("--image must be an existing local file")
    if not args.question.strip() or not 1 <= args.max_new_tokens <= 4096:
        parser.error("Use a nonempty question and max-new-tokens in [1,4096]")
    if not 1 <= args.max_input_tokens <= 16384:
        parser.error("This minimal example caps input tokens at 16384")

    import torch
    import transformers
    from PIL import Image, UnidentifiedImageError
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != "5.2.0":
        parser.error("This pinned example requires transformers==5.2.0")
    if not args.prepare_only and not torch.cuda.is_available():
        parser.error("A compatible CUDA GPU is required for this weight-inference example")
    try:
        with Image.open(args.image) as source:
            # Some Processor loading paths apply EXIF rotation. Refuse ambiguity.
            if source.getexif().get(274, 1) != 1:
                parser.error("EXIF orientation must be absent or 1; normalize the image "
                             "and its coordinate annotations/intrinsics together first")
            picture = source.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        parser.error(f"Cannot decode image: {error}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION,
                                              trust_remote_code=False,
                                              local_files_only=args.local_files_only)
    inputs = build_inputs(processor, picture, args.question)
    prompt_length = inputs["input_ids"].shape[1]
    if prompt_length > args.max_input_tokens:
        parser.error(f"Input has {prompt_length} tokens; reduce image size or question length")
    if any(value.is_floating_point() and not value.isfinite().all() for value in inputs.values()):
        parser.error("Processor produced non-finite input values")
    if args.prepare_only:
        print(json.dumps({
            "mode": "prepare-only", "model": MODEL_ID, "revision": REVISION,
            "transformers": transformers.__version__, "processor": type(processor).__name__,
            "image_size": list(picture.size), "image_frame": "stored_pixels_without_exif_rotation",
            "input_tokens": prompt_length, "max_input_tokens": args.max_input_tokens,
            "reserved_output_tokens": args.max_new_tokens,
            "inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       for name, value in inputs.items()},
            "model_weights_loaded": False, "model_forward_executed": False,
        }, indent=2))
        return
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=dtype, trust_remote_code=False,
        local_files_only=args.local_files_only,
    ).eval().to("cuda")
    inputs = {name: value.to(device="cuda", dtype=dtype if value.is_floating_point() else value.dtype)
              for name, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    answer_ids = output[0, prompt_length:]
    print(f"model={MODEL_ID}; revision={REVISION}; input_tokens={prompt_length}")
    print(f"generated_tokens={answer_ids.numel()}")
    print(processor.decode(answer_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
