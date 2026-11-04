"""Optional CPU capture checks plus a truly standard-library-only record replay."""

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = (ROOT/"optimizer" if (ROOT/"muon-download.json").is_file()
          else ROOT/"content/posts/ai/muon-optimizer")
SOURCE = FOLDER/"gradient_noise.py"
if not SOURCE.is_file():
    raise SystemExit("Missing local Muon gradient source: " + str(SOURCE))
spec = importlib.util.spec_from_file_location("gradient_noise", SOURCE)
noise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(noise)
toy = noise.local_module("noise_test_toy", FOLDER/"toy_training.py")
torch = toy.torch_runtime()


class GradientNoiseTests(unittest.TestCase):
    def test_sample_mean_against_independent_linear_derivative(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.25, -0.5]]))
        x = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[-1.,2.]])
        y = torch.tensor([[1.],[-1.],[2.],[0.]])
        gradients, batches = noise.collect_gradients(torch,model,model.weight,(x,y),batches=8,batch=3,seed=71)
        weight = model.weight.detach().reshape(-1)
        for actual, indices in zip(gradients,batches):
            expected = torch.zeros(2)
            for i in indices:
                expected += 2/3*(weight.dot(x[i])-y[i,0])*x[i]
            torch.testing.assert_close(torch.tensor(actual),expected,atol=2e-7,rtol=2e-7)

    def test_parameters_grads_and_rng_are_preserved(self):
        torch.manual_seed(19)
        model = toy.make_model(torch);data,_ = toy.make_data(torch)
        parameters = [p.detach().clone() for p in model.parameters()]
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        grads = [p.grad.clone() for p in model.parameters()]
        rng = torch.random.get_rng_state().clone()
        first = noise.collect_gradients(torch,model,model.hidden.weight,data,batches=8,batch=16,seed=11)
        second = noise.collect_gradients(torch,model,model.hidden.weight,data,batches=8,batch=16,seed=11)
        self.assertEqual(first,second)
        for p, before, grad in zip(model.parameters(),parameters,grads):
            torch.testing.assert_close(p,before,atol=0,rtol=0)
            torch.testing.assert_close(p.grad,grad,atol=0,rtol=0)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)
        self.assertTrue(model.training)

    def test_default_capture_and_recorded_reference(self):
        result = noise.summarize(noise.capture())
        expected = json.loads((FOLDER/"assets/gradient-noise-reference.json").read_text())
        for key in ("observed_batch_gradient_variance","observed_mean_gradient_power"):
            self.assertTrue(math.isclose(result[key],expected[key],rel_tol=5e-5,abs_tol=1e-8))
        for key in result["estimate"]:
            self.assertTrue(math.isclose(result["estimate"][key],expected["estimate"][key],rel_tol=5e-5,abs_tol=1e-8))
        self.assertEqual(result["parameter_shape"],[12,12])
        self.assertEqual(result["batch_count"],64)
        self.assertEqual(result["batch_size"],16)
        self.assertEqual(result["sampling"],"iid-with-replacement")

    def test_replay_without_any_site_packages_and_file_preservation(self):
        record = noise.capture(batches=8,batch=3,seed=72)
        with tempfile.TemporaryDirectory(prefix="muon-noise-test-") as folder:
            path = Path(folder)/"gradients.json"
            noise.save_records(path,record)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                noise.save_records(path,record)
            # -I -S excludes PYTHONPATH and all third-party packages, including
            # PyTorch and NumPy; file loading is explicit and local.
            actual = json.loads(subprocess.check_output(
                [sys.executable,"-I","-S","-B",str(SOURCE),"--replay",str(path)],timeout=20))
            self.assertEqual(actual,noise.summarize(record))
            self.assertEqual(path.read_bytes(),before)
            failure = subprocess.run([sys.executable,"-I","-S","-B",str(SOURCE),
                                      "--replay",str(path),"--batch","4"],
                                     capture_output=True,text=True,timeout=20)
            self.assertNotEqual(failure.returncode,0)
            self.assertIn("does not resample",failure.stderr)
            self.assertEqual(path.read_bytes(),before)

    def test_malformed_records_and_invalid_capture_settings(self):
        record = noise.capture(batches=2,batch=2,seed=13)
        changes = [
            {"sampling":"without-replacement"}, {"batch_count":1},
            {"batch_size":0}, {"parameter_shape":[12]},
            {"parameter_shape":[2,2]}, {"parameter_sha256_after":"0"*64},
            {"sampled_indices":[[-1,-1],[-1,-1]]}, {"layer":None},
            {"gradients":[[float("nan")]*144]*2},
        ]
        for change in changes:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    noise.validate_record({**record,**change})
        model = toy.make_model(torch);data,_ = toy.make_data(torch)
        for change in ({"batches":1},{"batch":0},{"batch":True},{"seed":-1}):
            with self.assertRaises(ValueError):
                noise.collect_gradients(torch,model,model.hidden.weight,data,**change)


if __name__ == "__main__":
    unittest.main(verbosity=2)
