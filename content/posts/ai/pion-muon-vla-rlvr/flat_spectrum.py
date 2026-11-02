"""Flat identity spectra: dimension, update RMS and finite representation.

The default study uses only the standard library and allocates no matrices.
An optional hash-checked official CPU/BF16 probe allocates identities up to128.
All values describe constructed spectra, not gradient statistics or training.
"""
import argparse
from decimal import (Context, Decimal, DivisionByZero, InvalidOperation,
                     Overflow, ROUND_HALF_EVEN, localcontext)
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from pion_spectral import check_steps, muon, pion


RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 65536)
BF16_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)
SOURCE_SHA256 = '8e7ca8d6f40ebc4fb066acb15387e618d515896cca5d1eb0ad234e5310e9157e'


def rank_response(rank, promotion_steps):
    """One equal singular coefficient; RMS/(eta*gamma) for full-rank I_r."""
    if type(rank) is not int or not 1 <= rank <= 65536:
        raise ValueError('rank must be an integer from 1 to 65536')
    check_steps(promotion_steps)
    private = Context(prec=100, rounding=ROUND_HALF_EVEN,
                      Emin=-999999, Emax=999999, capitals=1, clamp=0,
                      flags=[], traps=[InvalidOperation, DivisionByZero, Overflow])
    with localcontext(private):
        value = Decimal(1) / Decimal(rank).sqrt()
        for index in range(5):
            coefficients = ('1.875', '-1.25', '.375') if index < promotion_steps else ('0', '2.5', '-1.5')
            a, b, c = map(Decimal, coefficients)
            value = a * value + b * value**3 + c * value**5
        if value <= 0:
            raise RuntimeError('Positive flat input unexpectedly lost its high-precision response')
        logarithm = float(value.log10())
        decimal_text = str(value)
    normalized = 1. / math.sqrt(rank)
    coefficient = pion(normalized, promotion_steps)
    return {
        'promotion_steps': promotion_steps, 'suppression_steps': 5 - promotion_steps,
        'normalized_input_float64': normalized,
        'coefficient_decimal': decimal_text,
        'log10_coefficient_decimal': logarithm,
        'coefficient_float64': coefficient,
        'float64_underflowed': coefficient == 0.,
        'float64_entropy_rank': rank if coefficient != 0. else 0,
    }


def report():
    return {
        'construction': 'M=I_r, full-rank square identity; epsilon=0; five whole-matrix polynomial iterations',
        'scope': 'Constructed flat spectra, not measured model gradients. No matrices are allocated in the standard-library study.',
        'decimal_precision': 100,
        'ratio_definition': 'In exact arithmetic for this construction, RMS(gradient update)/(eta*gamma)=abs(coefficient); the ideal full-rank polar factor gives 1. Float64 coefficients are recorded before learning-rate/dimension multiplication, whose rounding is not modeled here. Weight decay is excluded.',
        'entropy_convention': 'Every positive equal coefficient gives mathematical entropy rank r; a stored all-zero spectrum has the separate example convention 0.',
        'rows': [
            {'rank': rank, 'mathematical_entropy_rank': rank,
             'muon_float64_coefficient': muon(1. / math.sqrt(rank)),
             'pion': {str(kp): rank_response(rank, kp) for kp in (0, 1, 2)}}
            for rank in RANKS
        ],
    }


def probe_bf16(source):
    """Observe the original helper at epsilon 0 and its 1e-7 default."""
    source = Path(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Official VLA source does not match pinned Pion commit 8796501')
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('pinned_flat_spectrum_probe', source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rows = []
    for rank in BF16_RANKS:
        for kp in (0, 1, 2):
            for epsilon in (0., 1e-7):
                result = official.high_pass_ns(torch.eye(rank, dtype=torch.float64), kp, 5 - kp, epsilon=epsilon)
                diagonal = result.diag().to(torch.float64)
                off_diagonal = float((result.to(torch.float64) - torch.diag(diagonal)).abs().max())
                if not bool(torch.isfinite(result).all()) or not bool((diagonal == diagonal[0]).all()) or off_diagonal != 0.:
                    raise RuntimeError('The tested identity did not retain a finite equal diagonal')
                coefficient = float(diagonal[0])
                rows.append({'rank': rank, 'promotion_steps': kp, 'epsilon': epsilon,
                             'coefficient': coefficient, 'stored_entropy_rank': rank if coefficient != 0. else 0,
                             'off_diagonal_max': off_diagonal})
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Official source file changed during the probe')
    return {
        'official_commit': '87965014b360bde604632cba0aa12e423c6c63c6',
        'source_sha256': digest, 'source_modified': False,
        'torch': torch.__version__, 'device': 'cpu',
        'input_dtype': 'float64', 'internal_and_output_dtype': 'bfloat16',
        'scope': 'Only small identity matrices in the official helper. No model, GPU, optimizer state or distributed execution. Rounding and underflow are device/kernel observations, not universal thresholds.',
        'cases': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-file', type=Path, help='Add the optional PyTorch CPU/BF16 observations')
    args = parser.parse_args()
    result = report()
    if args.official_file is not None:
        try:
            result['official_bf16'] = probe_bf16(args.official_file)
        except (OSError, ValueError, ImportError) as error:
            parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
