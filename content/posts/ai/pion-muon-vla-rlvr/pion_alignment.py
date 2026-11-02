"""Two labeled diagonal examples: spectral strength does not identify signal.

The clean matrix is fixed as diag(1, 0, 0, 0). Each perturbation is constructed,
not sampled from training data. Alignment is a Frobenius cosine, not task
accuracy or an estimate of the stochastic gradient SNR in the paper.
Momentum equals the constructed gradient; no momentum history is accumulated.
"""
import json
import math
from pion_spectral import evaluate_spectrum


def report():
    signal = [1., 0., 0., 0.]
    rows = []
    for name, noise in [("weak_tail", [0., .2, .1, .05]),
                        ("dominant_perturbation", [0., 5., 0., 0.])]:
        gradient = [clean+perturbation for clean, perturbation in zip(signal, noise)]
        spectrum = evaluate_spectrum(gradient, 1, 0.)
        updates = {"normalized_gradient": spectrum["normalized"],
                   "muon": spectrum["muon"], "pion": spectrum["pion"]}
        # With this fixed rank-one signal, <S,U>_F / (||S||_F ||U||_F)
        # is U[0,0] / ||U||_F. All updates here have nonzero norm.
        alignment = {key: values[0]/math.hypot(*values) for key, values in updates.items()}
        rows.append({"name": name, "signal_diagonal": signal.copy(),
                     "perturbation_diagonal": noise, "gradient_diagonal": gradient,
                     "update_diagonals": updates, "cosine_with_clean_signal": alignment})
    return {"scope": "Constructed fixed diagonal perturbations, not training or statistical SNR",
            "promotion_steps": 1, "suppression_steps": 4, "epsilon": 0.,
            "momentum_equals_gradient": True, "examples": rows}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
