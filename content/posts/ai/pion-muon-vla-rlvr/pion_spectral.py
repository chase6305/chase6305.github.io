"""Scalar spectral maps from Pion, arXiv:2605.19282v1 (standard library).

These calculations describe normalized singular values, not training accuracy.
Finite iterations are smooth filters; a printed floating-point zero can be
underflow and is not a proof of exact rank reduction.
"""
import argparse
import json
import math

PROMOTION = (1.875, -1.25, .375)
SUPPRESSION = (0., 2.5, -1.5)
MUON = (3.4445, -4.7750, 2.0315)
TOTAL_STEPS = 5


def check_steps(promotion_steps):
    if type(promotion_steps) is not int or not 0 <= promotion_steps <= TOTAL_STEPS:
        raise ValueError("promotion_steps must be an integer from 0 to 5")
    return promotion_steps


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite, real and nonnegative")
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite, real and nonnegative")
    return float(value)


def check_sigma(sigma):
    sigma = finite_nonnegative(sigma, "sigma")
    if sigma > 1:
        raise ValueError("sigma must be in [0, 1]")
    return sigma


def polynomial(sigma, coefficients):
    a, b, c = coefficients
    square = sigma * sigma
    return sigma * (a + square * (b + c * square))


def pion(sigma, promotion_steps=1):
    sigma = check_sigma(sigma)
    promotion_steps = check_steps(promotion_steps)
    for _ in range(promotion_steps):
        sigma = polynomial(sigma, PROMOTION)
    for _ in range(TOTAL_STEPS - promotion_steps):
        sigma = polynomial(sigma, SUPPRESSION)
    return sigma


def muon(sigma):
    sigma = check_sigma(sigma)
    for _ in range(TOTAL_STEPS):
        sigma = polynomial(sigma, MUON)
    return sigma


def response_trace(sigma, promotion_steps=1):
    """List the normalized input and every finite polynomial output."""
    sigma = check_sigma(sigma)
    promotion_steps = check_steps(promotion_steps)
    pion_values, muon_values = [sigma], [sigma]
    stages = []
    for index in range(TOTAL_STEPS):
        promoting = index < promotion_steps
        stages.append("promotion" if promoting else "suppression")
        pion_values.append(polynomial(pion_values[-1], PROMOTION if promoting else SUPPRESSION))
        muon_values.append(polynomial(muon_values[-1], MUON))
    return {"input_is_normalized": True, "promotion_steps": promotion_steps,
            "suppression_steps": TOTAL_STEPS-promotion_steps,
            "pion_stages": stages, "pion": pion_values, "muon": muon_values}


def half_response(promotion_steps):
    """Input yielding output 0.5 after exactly five steps; not a hard cutoff."""
    check_steps(promotion_steps)
    low, high = 0., 1.
    for _ in range(70):
        middle = (low + high) / 2
        if pion(middle, promotion_steps) < .5:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def effective_rank(spectrum):
    """Entropy rank of nonnegative amplitudes (not squared values); zero: 0."""
    values = tuple(finite_nonnegative(value, "Spectrum value") for value in spectrum)
    if not values:
        raise ValueError("spectrum must not be empty")
    largest = max(values)
    if largest == 0:
        return 0.
    scaled = [value/largest for value in values]
    total = math.fsum(scaled)
    probabilities = [value/total for value in scaled if value > 0]
    # A probability can underflow to zero; its limiting p*log(p) term is zero.
    return math.exp(-math.fsum(value * math.log(value) for value in probabilities if value > 0))


def normalize_spectrum(spectrum, epsilon=0.):
    """Frobenius normalization, including finite epsilon, without norm overflow."""
    values = tuple(finite_nonnegative(value, "Spectrum value") for value in spectrum)
    epsilon = finite_nonnegative(epsilon, "epsilon")
    if not values:
        raise ValueError("spectrum must not be empty")
    largest = max(values)
    if largest == 0:
        return [0.]*len(values)
    scaled = [value/largest for value in values]
    norm = math.hypot(*scaled)
    if epsilon > largest:
        ratio = largest/epsilon
        return [(value*ratio)/(1+norm*ratio) for value in scaled]
    return [value/(norm+epsilon/largest) for value in scaled]


def evaluate_spectrum(spectrum, promotion_steps=1, epsilon=0.):
    raw = tuple(finite_nonnegative(value, "Spectrum value") for value in spectrum)
    promotion_steps = check_steps(promotion_steps)
    epsilon = finite_nonnegative(epsilon, "epsilon")
    normalized = normalize_spectrum(raw, epsilon)
    muon_values = [muon(value) for value in normalized]
    pion_values = [pion(value, promotion_steps) for value in normalized]
    return {"raw": list(raw), "normalized": normalized, "muon": muon_values,
            "pion": pion_values, "input_erank": effective_rank(raw),
            "muon_erank": effective_rank(muon_values), "pion_erank": effective_rank(pion_values),
            "muon_norm": math.hypot(*muon_values), "pion_norm": math.hypot(*pion_values),
            "promotion_steps": promotion_steps, "suppression_steps": TOTAL_STEPS-promotion_steps,
            "epsilon": epsilon, "all_zero": not any(raw)}


def report():
    raw = (.8, .4, .2, .02)
    norm = math.hypot(*raw)
    normalized = [value / norm for value in raw]
    return {
        "paper": "arXiv:2605.19282v1",
        "scope": "Deterministic scalar arithmetic, not model training or noise identification",
        "steps": TOTAL_STEPS,
        "suppression_unstable_fixed_point": math.sqrt(2 / 3),
        "half_response_inputs": [half_response(kp) for kp in range(6)],
        "example_raw_singular_values": list(raw),
        "example_normalized_singular_values": normalized,
        "example_muon": [muon(value) for value in normalized],
        "example_pion_kp1": [pion(value, 1) for value in normalized],
        "example_input_erank": effective_rank(normalized),
        "interactive_default": evaluate_spectrum(raw, 1, 0.),
        "isotropic_four_direction_sigma": .5,
        "isotropic_four_direction_pion_kp0": pion(.5, 0),
        "real_robot_success_counts_inferred_from_30_trials": {
            "AdamW": [12, 10, 6], "Muon": [17, 10, 8], "Pion": [28, 25, 24]},
        "vl_adapter_suite_averages_percent": {
            "AdamW": (32.2 + 97.0 + 89.2 + 69.6) / 4,
            "Muon": (97.0 + 99.0 + 95.8 + 88.0) / 4,
            "Pion": (100.0 + 99.4 + 97.2 + 92.4) / 4},
        "vlanext_pion_minus_muon_percentage_points": {
            "LIBERO": 96.35 - 93.65, "LIBERO-Plus": 75.93 - 72.34,
            "Language": 86.93 - 77.53, "Noise": 76.09 - 69.98,
            "Robot": 63.18 - 57.36},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--sigma", type=float)
    choice.add_argument("--spectrum", nargs="+", type=float, help="Raw nonnegative singular amplitudes")
    parser.add_argument("--kp", type=int, default=1)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--trace", action="store_true", help="Show every iteration; requires --sigma")
    args = parser.parse_args()
    try:
        check_steps(args.kp)
        if args.epsilon is not None and args.spectrum is None:
            raise ValueError("--epsilon requires --spectrum (raw amplitudes)")
        if args.trace and args.sigma is None:
            raise ValueError("--trace requires --sigma (already normalized input)")
        if args.spectrum is not None:
            result = evaluate_spectrum(args.spectrum, args.kp, args.epsilon if args.epsilon is not None else 0.)
        elif args.sigma is None:
            if args.kp != 1:
                raise ValueError("Choose --sigma or --spectrum when changing --kp")
            result = report()
        else:
            result = {
                "sigma": args.sigma, "promotion_steps": args.kp,
                "suppression_steps": TOTAL_STEPS-args.kp,
                "muon_five_steps": muon(args.sigma), "pion_five_steps": pion(args.sigma, args.kp)}
            if args.trace:
                result["trace"] = response_trace(args.sigma, args.kp)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
