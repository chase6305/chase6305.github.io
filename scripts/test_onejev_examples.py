#!/usr/bin/env python3
"""Independent arithmetic and loopback-only wire checks; no inference model needed."""
from __future__ import annotations

import base64
import contextlib
import copy
from decimal import Decimal, localcontext
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "content/posts/ai/jev-decision-layer/onejev_examples.py"
spec = importlib.util.spec_from_file_location("onejev_examples", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@contextlib.contextmanager
def server(status=200, payload=None):
    seen = []
    response = payload if payload is not None else {"model": "fixture", "answers": {}, "usage": {}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append({"path": self.path, "type": self.headers["Content-Type"], "body": json.loads(raw)})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response if isinstance(response, bytes) else json.dumps(response).encode())

    http = HTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", seen, response
    finally:
        http.shutdown()
        worker.join(timeout=5)
        http.server_close()
        assert not worker.is_alive()


def cli(*args):
    return subprocess.run([sys.executable, "-B", str(EXAMPLE), *args], capture_output=True, text=True, timeout=10)


class OneJevExamples(unittest.TestCase):
    def test_response_example_and_read_command(self):
        fixture = json.loads((EXAMPLE.parent / "assets/onejev-response-example.json").read_text())
        body = example.build_request()
        answers = fixture["answers"]
        self.assertEqual(set(answers), set(body["questions"]))
        self.assertEqual(list(answers["next"]["probabilities"]), list(body["questions"]["next"]["criteria"]))
        self.assertEqual(list(answers["progress"]["legend"].values()), body["questions"]["progress"]["criteria"])
        with localcontext() as ctx:
            ctx.prec = 80
            expected = (Decimal("0.8") - Decimal("0.25")) / Decimal("0.75")
            self.assertEqual(answers["next"]["confidence"], float(round(expected, 6)))
        expected_view = {
            "model": "constructed-example-not-model-inference",
            "next_choice": "enable", "next_choice_probability": 0.8,
            "next_confidence": 0.733333, "done_probability": 0.05,
            "progress_score": 1.0, "progress_confidence": 0.85,
        }
        section = EXAMPLE.with_name("index.md").read_text().split("{#onejev-response-example}", 1)[1]
        published = re.search(r"```json\n([\s\S]*?)```", section)
        self.assertIsNotNone(published)
        self.assertEqual(json.loads(published.group(1)), expected_view)
        original = copy.deepcopy(fixture)
        self.assertEqual(example.read_response(fixture), expected_view)
        self.assertEqual(fixture, original)
        with tempfile.TemporaryDirectory() as folder, server(payload=fixture) as (url, seen, _):
            p = cli("ask", "--url", url)
            self.assertEqual(p.returncode, 0, p.stderr)
            file = Path(folder) / "响应 示例.json"
            file.write_text(p.stdout)
            p = cli("read-response", str(file))
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout), expected_view)
            self.assertEqual(len(seen), 1, "reading a saved response must not issue another request")
            self.assertEqual(json.loads(file.read_text()), fixture)

    def test_response_view_errors_and_extra_metadata(self):
        fixture = json.loads((EXAMPLE.parent / "assets/onejev-response-example.json").read_text())
        augmented = copy.deepcopy(fixture)
        augmented.update(usage={"input_tokens": 123, "output_tokens": 3}, debug={"fixture": True})
        self.assertEqual(example.read_response(augmented), example.read_response(fixture))
        for response in [None, [], {}, {"model": "fixture", "answers": {}}]:
            with self.assertRaises(ValueError):
                example.read_response(response)
        for field, value in [("choice", "missing"), ("confidence", True), ("confidence", math.nan),
                             ("confidence", math.inf), ("confidence", -0.1), ("confidence", 10 ** 1000)]:
            response = copy.deepcopy(fixture)
            response["answers"]["next"][field] = value
            with self.assertRaises(ValueError):
                example.read_response(response)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "invalid.json"
            for raw in ['[]', '{"model":"fixture","answers":{}}', 'not JSON']:
                file.write_text(raw)
                p = cli("read-response", str(file))
                self.assertEqual(p.returncode, 1)
                self.assertEqual(p.stdout, "")
                self.assertNotIn("Traceback", p.stderr)

    def test_softmax_against_decimal_and_extreme_offsets(self):
        rng = random.Random(20261102)
        with localcontext() as ctx:
            ctx.prec = 80
            for _ in range(100):
                values = [rng.uniform(-12, 12) for _ in range(rng.randint(2, 12))]
                t = rng.uniform(0.25, 4)
                weights = [(Decimal.from_float(v) / Decimal.from_float(t)).exp() for v in values]
                total = sum(weights)
                expected = [float(w / total) for w in weights]
                actual = example.softmax(values, t)
                self.assertLess(max(abs(a - b) for a, b in zip(actual, expected)), 2e-15)
                self.assertAlmostEqual(sum(actual), 1)
        self.assertEqual(example.softmax([10000, 10000]), [0.5, 0.5])
        self.assertEqual(example.softmax([-1e308, 1e308]), [0.0, 1.0])
        for values, t in [([], 1), ([math.nan], 1), ([math.inf], 1), ([0], 0), ([0], -1), ([0], math.nan)]:
            with self.assertRaises(ValueError):
                example.softmax(values, t)

    def test_published_arithmetic(self):
        d = example.arithmetic()
        for actual, expected in zip(d["probabilities_T1"], [0.8, 0.1, 0.1]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(d["choice_confidence"], 0.7)
        self.assertAlmostEqual(d["probabilities_T2"][0], 2 - math.sqrt(2))
        with localcontext() as ctx:
            ctx.prec = 80
            self.assertAlmostEqual(d["cross_entropy"], float(-(Decimal(4) / Decimal(5)).ln()))
            self.assertAlmostEqual(d["loss_with_brier_weight_1"], float(-(Decimal(4) / Decimal(5)).ln() + Decimal("0.06")))
        self.assertAlmostEqual(d["expected_score"], 1.2)
        self.assertAlmostEqual(d["score_confidence"], 0.4)
        p = cli("math")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout), d)

    def test_request_contract_and_media_bytes(self):
        body = example.build_request()
        article = EXAMPLE.with_name("index.md").read_text()
        section = article.split("{#onejev-quickstart}", 1)[1]
        published = re.search(r"```json\n([\s\S]*?)```", section)
        self.assertIsNotNone(published)
        self.assertEqual(json.loads(published.group(1)), body)
        self.assertEqual(body["model"], "OneJev1.6-0.8B")
        self.assertFalse(body["state"]["observation"]["notifications_enabled"])
        self.assertNotIn("media", body)
        self.assertEqual({k: q["type"] for k, q in body["questions"].items()}, {"done": "noul", "next": "choice", "progress": "score"})
        self.assertEqual(list(body["questions"]["next"]["criteria"]), ["enable", "stop", "reobserve", "handoff"])
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "截图 示例.PNG"
            image = (EXAMPLE.parent / "assets/onejev-settings-screen.png").read_bytes()
            file.write_bytes(image)
            other = example.build_request(file, "local-demo")
            self.assertEqual(other["model"], "local-demo")
            self.assertEqual(other["state"], {"task": body["state"]["task"], "screen": "<image:1>"})
            self.assertEqual(other["questions"], body["questions"])
            uri = other["media"][0]["data"]
            self.assertTrue(uri.startswith("data:image/png;base64,"))
            self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), image)
            self.assertEqual(file.read_bytes(), image)
            p = cli("request", "--image", str(file), "--model", "local-demo")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout), other)

    def test_loopback_request_and_unmodified_response(self):
        payload = {"model": "fixture", "answers": {"next": {"type": "choice", "choice": "enable", "probabilities": {"enable": 0.7, "stop": 0.1, "reobserve": 0.1, "handoff": 0.1}, "confidence": 0.6}}, "usage": {"input_tokens": 12, "output_tokens": 1}}
        with server(payload=payload) as (url, seen, response):
            p = cli("ask", "--url", url + "/", "--model", "fixture")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout), response)
            self.assertEqual(seen, [{"path": "/v1/systemone", "type": "application/json", "body": example.build_request(model="fixture")}])

    def test_server_errors_are_visible(self):
        with server(status=422, payload={"detail": "fixture invalid request"}) as (url, seen, _):
            p = cli("ask", "--url", url)
            self.assertEqual(p.returncode, 1)
            self.assertIn("HTTP 422", p.stderr)
            self.assertIn("fixture invalid request", p.stderr)
            self.assertEqual(p.stdout, "")
            self.assertEqual(len(seen), 1)
        with server(payload=b"not JSON") as (url, _, _):
            p = cli("ask", "--url", url)
            self.assertEqual(p.returncode, 1)
            self.assertNotIn("Traceback", p.stderr)

    def test_invalid_local_settings_do_not_call_server(self):
        with server() as (url, seen, _):
            for timeout in ["0", "-1", "nan", "inf"]:
                p = cli("ask", "--url", url, "--timeout", timeout)
                self.assertEqual(p.returncode, 1)
                self.assertIn("timeout", p.stderr)
            self.assertEqual(seen, [])
        p = cli("request", "--image", "missing-file.png")
        self.assertEqual(p.returncode, 1)
        self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
