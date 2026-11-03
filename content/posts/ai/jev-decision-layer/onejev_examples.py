#!/usr/bin/env python3
"""Small OneJev examples: probability arithmetic, request construction and HTTP calls.

The arithmetic uses constructed logits, not model predictions. Request/ask use
the System One wire format. This file does not execute the selected action.
Only the Python standard library is required; inference needs a separate server.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("logits must be nonempty and finite")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # Subtract before dividing, so a common large offset cannot overflow.
    peak = max(logits)
    weights = [math.exp((x - peak) / temperature) for x in logits]
    total = sum(weights)
    return [w / total for w in weights]


def arithmetic() -> dict:
    logits = [math.log(8), 0.0, 0.0]
    p = softmax(logits)
    warm = softmax(logits, 2.0)
    uniform = 1 / len(p)
    confidence = (max(p) - uniform) / (1 - uniform)
    ce = -math.log(p[0])
    brier = sum((pi - yi) ** 2 for pi, yi in zip(p, [1.0, 0.0, 0.0]))
    score_p = [0.1, 0.6, 0.3]
    score = sum(i * pi for i, pi in enumerate(score_p))
    distance = sum(pi * abs(i - 1) for i, pi in enumerate(score_p))
    uniform_mad = 2 / 3
    return {
        "source": "constructed logits and probabilities; not model inference",
        "logits": logits,
        "probabilities_T1": p,
        "choice_confidence": confidence,
        "probabilities_T2": warm,
        "cross_entropy": ce,
        "brier_sum": brier,
        "loss_with_brier_weight_1": ce + brier,
        "score_distribution": score_p,
        "expected_score": score,
        "score_confidence": max(0.0, 1 - distance / uniform_mad),
    }


def build_request(image: Path | None = None, model: str = "OneJev1.6-0.8B") -> dict:
    state = {
        "task": "Enable desktop notifications.",
        "observation": {
            "page": "Settings",
            "notifications_enabled": False,
            "toggle_visible": True,
        },
    }
    body = {
        "model": model,
        "state": state,
        "questions": {
            "done": {"type": "noul", "instructions": "Notifications are enabled."},
            "next": {
                "type": "choice",
                "instructions": "What should the agent do next to complete the task?",
                "criteria": {
                    "enable": "Click the Notifications toggle if notifications are disabled.",
                    "stop": "Stop because notifications are already enabled.",
                    "reobserve": "Capture a new screen because the current state is unreadable.",
                    "handoff": "Ask the planner for help because the setting is unavailable.",
                },
            },
            "progress": {
                "type": "score",
                "instructions": "Which level describes the current notification setting?",
                "criteria": [
                    "Notifications toggle is not available on the screen.",
                    "Notifications toggle is available but notifications are disabled.",
                    "Notifications are enabled.",
                ],
            },
        },
    }
    if image is not None:
        media_type = {
            ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp",
        }.get(image.suffix.lower())
        if media_type is None:
            raise ValueError("image must be a PNG, JPEG or WebP file")
        data = base64.b64encode(image.read_bytes()).decode("ascii")
        body["state"] = {"task": state["task"], "screen": "<image:1>"}
        body["media"] = [{"type": "image", "data": f"data:{media_type};base64,{data}"}]
    return body


def ask(body: dict, base_url: str, timeout: float) -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def read_response(response: dict) -> dict:
    """Read this example's three answers; this is not an action-execution gate.

    The server's model/usage/debug fields and the original response are preserved
    by `ask`. This separate view only extracts the fields explained in the article.
    """
    def number(value, name, low=0.0, high=1.0):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not low <= value <= high or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
        return value

    try:
        model = response["model"]
        answers = response["answers"]
        done, next_answer, progress = (answers[k] for k in ("done", "next", "progress"))
        if (not isinstance(model, str) or not model
                or [done["type"], next_answer["type"], progress["type"]] != ["noul", "choice", "score"]):
            raise ValueError("expected model and done/next/progress answers from this example")
        choice = next_answer["choice"]
        probabilities = next_answer["probabilities"]
        if not isinstance(choice, str) or not isinstance(probabilities, dict) or choice not in probabilities:
            raise ValueError("next.choice must name an entry in next.probabilities")
        return {
            "model": model,
            "next_choice": choice,
            "next_choice_probability": number(probabilities[choice], "selected probability"),
            "next_confidence": number(next_answer["confidence"], "Choice confidence"),
            "done_probability": number(done["noul"], "Noul probability"),
            "progress_score": number(progress["score"], "Score", high=2.0),
            "progress_confidence": number(progress["confidence"], "Score confidence"),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("expected this example's model and done/next/progress answer fields") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("math", help="recompute constructed probability and loss examples")
    reader = commands.add_parser("read-response", help="read the three example answers from a saved JSON response")
    reader.add_argument("file", type=Path)
    for name in ["request", "ask"]:
        sub = commands.add_parser(name, help="print a request" if name == "request" else "call a local inference server")
        sub.add_argument("--image", type=Path, help="use this screenshot instead of the text observation")
        sub.add_argument("--model", default="OneJev1.6-0.8B", help="served model name, as reported by /v1/models")
        if name == "ask":
            sub.add_argument("--url", default="http://127.0.0.1:8000")
            sub.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        if args.command == "math":
            result = arithmetic()
        elif args.command == "read-response":
            result = read_response(json.loads(args.file.read_text(encoding="utf-8")))
        else:
            result = build_request(args.image, args.model)
            if args.command == "ask":
                result = ask(result, args.url, args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read(4096).decode('utf-8', errors='replace')}", file=sys.stderr)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
