"""Compare pinned RL Pion optimizer steps on rank-one CPU gradients.

No model is loaded. Rank-one observations make the BF16 spectral response
exactly signed unit entries, so momentum, decay and missing-gradient behavior
can be checked separately from general-matrix rounding. See the other probe
for the latter. The reference itself uses Float64.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-file", type=Path, required=True)
    source = parser.parse_args().official_file
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        parser.error(str(error))
    if digest != "15c1331453ef3db4b565a514d1e43f74692e2e910eb8cb939894d54371dd402d":
        parser.error("Official RL source does not match pinned Pion commit 8796501")
    import numpy as np
    import torch
    from pion_step import step

    spec = importlib.util.spec_from_file_location("pinned_pion_rl", source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    torch.set_num_threads(1)
    observations = [np.array([[.8, 0., 0.], [0., 0., 0.]]),
                    np.zeros((2, 3)), None,
                    np.array([[-1.6, 0., 0.], [0., 0., 0.]])]
    cases = []
    for beta in (.5, .95):
        for nesterov in (False, True):
            for kp in (0, 1, 5):
                weight = np.array([[1., 2., 3.], [4., 5., 6.]])
                buffer = np.zeros_like(weight)
                parameter = torch.nn.Parameter(torch.from_numpy(weight.copy()))
                optimizer = official.DefaultPionAdamW(
                    [parameter], lr=.01, weight_decay=.1, momentum=beta,
                    nesterov=nesterov, promotion_steps=kp, ns_steps=5, scale_factor=.2)
                rows = []
                for index, gradient in enumerate(observations, 1):
                    parameter.grad = None if gradient is None else torch.from_numpy(gradient.copy())
                    optimizer.step()
                    weight,buffer,trace = step(weight,gradient,buffer,beta=beta,nesterov=nesterov,
                                               promotion_steps=kp,epsilon=1e-7)
                    actual_weight = parameter.detach().numpy()
                    actual_buffer = optimizer.state[parameter]["momentum_buffer"].numpy()
                    np.testing.assert_allclose(actual_weight,weight,atol=4e-15,rtol=0)
                    np.testing.assert_allclose(actual_buffer,buffer,atol=2e-16,rtol=0)
                    rows.append({"step": index, "status": trace["status"],
                                 "max_weight_difference": float(np.max(np.abs(actual_weight-weight))),
                                 "max_buffer_difference": float(np.max(np.abs(actual_buffer-buffer))),
                                 "official_observed_steps": optimizer.state[parameter]["step"]})
                assert optimizer.state[parameter]["step"] == 3
                cases.append({"beta": beta, "nesterov": nesterov, "promotion_steps": kp, "steps": rows})
    result = {"scope": "Rank-one CPU optimizer steps; no model, GPU, VLA communication or general-matrix BF16 agreement",
              "official_commit": "87965014b360bde604632cba0aa12e423c6c63c6",
              "official_source_sha256": digest, "torch": torch.__version__, "numpy": np.__version__,
              "device": "cpu", "parameter_dtype": "float64", "official_internal_dtype": "bfloat16",
              "lr": .01, "weight_decay": .1, "scale_factor": .2, "epsilon": 1e-7, "cases": cases}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
