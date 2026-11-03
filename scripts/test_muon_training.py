"""Optional CPU integration checks; needs the article's PyTorch 2.14.1 environment.

No model or dataset downloads. The source hash pins the implementation used by
the article; it does not certify GPU arithmetic or full training reproducibility.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT/"optimizer/toy_training.py" if (ROOT/"muon-download.json").is_file()
          else ROOT/"content/posts/ai/muon-optimizer/toy_training.py")
if not SOURCE.is_file():
    raise SystemExit("Missing local Muon training source: " + str(SOURCE))
spec = importlib.util.spec_from_file_location("toy_training", SOURCE)
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)
torch = toy.torch_runtime()


class NativeMuonTrainingTests(unittest.TestCase):
    def test_pinned_source(self):
        self.assertEqual(torch.__version__.split("+")[0], "2.14.1")
        source = Path(importlib.util.find_spec("torch.optim._muon").origin)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         "c17abbb8af4c935d125b189c38d3cbaec295f1cfd232d2712bdb75e13184b972")

    def test_semantic_groups_and_shared_role_rejection(self):
        model = toy.make_model(torch)
        muon, adamw = toy.optimizers(torch, model)
        self.assertEqual([id(p) for p in muon.param_groups[0]["params"]], [id(model.hidden.weight)])
        all_params = muon.param_groups[0]["params"]+adamw.param_groups[0]["params"]
        self.assertEqual(len(all_params), len({id(p) for p in all_params}))
        self.assertEqual({id(p) for p in all_params}, {id(p) for p in model.parameters()})
        self.assertIn(id(model.head.weight), {id(p) for p in adamw.param_groups[0]["params"]})
        model.head.weight = model.hidden.weight
        with self.assertRaises(ValueError):
            toy.optimizers(torch, model)

    def test_uneven_microbatch_against_linear_derivative(self):
        model = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        x = torch.tensor([[1., 2.], [2., -1.], [-1., 0.], [3., 1.], [0., 2.]])
        y = torch.tensor([[1., 0., -1.], [0., 1., 2.], [-2., 1., 0.], [1., 3., 1.], [0., 2., 1.]])
        toy.backward_batch(model, (x, y), torch.arange(5), 2)  # 2 + 2 + 1 rows.
        expected = -2/(5*3)*(y.T@x)
        torch.testing.assert_close(model.weight.grad, expected, rtol=2e-7, atol=2e-7)

    def test_nonlinear_gradients_before_ns(self):
        torch.manual_seed(19)
        full = toy.make_model(torch)
        accumulated = toy.make_model(torch)
        accumulated.load_state_dict(full.state_dict())
        train, _ = toy.make_data(torch)
        indices = torch.arange(32)
        toy.backward_batch(full, train, indices, 32)
        toy.backward_batch(accumulated, train, indices, 13)
        for first, second in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(first.grad, second.grad, rtol=2e-5, atol=2e-8)

    def test_checkpoint_round_trip_and_missing_momentum(self):
        result = toy.run(steps=160, batch=32, microbatch=13)
        self.assertEqual(result["resume_max_parameter_difference"], 0)
        self.assertGreater(result["missing_muon_momentum_max_difference"], 1e-4)
        self.assertEqual(result["processed_training_rows"], 5120)
        for split in ("train", "validation"):
            self.assertTrue(math.isfinite(result["final_mse"][split]))
            self.assertLess(result["final_mse"][split], result["initial_mse"][split])

    def test_missing_gradient_and_explicit_zero(self):
        parameter = torch.nn.Parameter(torch.eye(2))
        optimizer = torch.optim.Muon([parameter], lr=0.02, weight_decay=0.1, momentum=0.5)
        parameter.grad = torch.eye(2)
        optimizer.step()
        before = parameter.detach().clone()
        buffer = optimizer.state[parameter]["momentum_buffer"].clone()
        parameter.grad = None
        optimizer.step()
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], buffer, rtol=0, atol=0)
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        self.assertGreater(float((parameter.detach()-before).abs().max()), 0)
        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], buffer*0.5, rtol=0, atol=0)

    def test_real_process_resume_and_file_preservation(self):
        with tempfile.TemporaryDirectory(prefix="muon-process-test-") as temporary:
            checkpoint = Path(temporary)/"run.pt"
            saved = toy.save_training(checkpoint, steps=80)
            before = checkpoint.read_bytes()
            with self.assertRaises(FileExistsError):
                toy.save_training(checkpoint, steps=1)
            self.assertEqual(checkpoint.read_bytes(), before)
            completed = json.loads(subprocess.check_output(
                [sys.executable, "-B", str(SOURCE), "--resume", str(checkpoint), "--steps", "160"], timeout=45))
            continuous = toy.run()
            self.assertEqual(completed["continued_steps"], 80)
            self.assertEqual(completed["parameter_sha256"], continuous["parameter_sha256"])
            self.assertNotEqual(completed["parameter_sha256"], saved["parameter_sha256"])
            self.assertEqual(checkpoint.read_bytes(), before)
            with self.assertRaises(ValueError):
                toy.resume_training(checkpoint, steps=79)
            with self.assertRaises(ValueError):
                toy.resume_training(checkpoint, batch=16)

    def test_cli_rejects_zero_instead_of_using_defaults(self):
        for flag in ("--batch", "--microbatch"):
            result = subprocess.run([sys.executable, "-B", str(SOURCE), flag, "0"], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("require steps", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
