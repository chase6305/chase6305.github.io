"""Constructed scalar-output gradient rank and EMA-history examples.

The network is y_i=v^T ReLU(W*x_i), evaluated at W=I and trainable head v=1,
with mean absolute regression loss to target0. Every preactivation is nonzero and every y_i=1,
so this point is differentiable despite the ReLU and absolute value.
These small networks are not a VLA architecture or training measurement.
"""
from fractions import Fraction
import json

from pion_spectral import effective_rank


def hidden_example(dimension):
    """Exact full-rank hidden-gradient certificate for one scalar output."""
    if type(dimension) is not int or not 2 <= dimension <= 32:
        raise ValueError('dimension must be an integer in [2,32]')
    size = dimension
    a = Fraction(1, 2*size)
    inputs = [[Fraction(1) if i == j else -a for j in range(size)]
              for i in range(size)]
    gradient = [[value/size for value in row] for row in inputs]
    parallel = (1-a*(size-1))/size
    orthogonal = (1+a)/size
    return {
        'scalar_output_dimension': 1,
        'shared_weight_positions': size,
        'hidden_shape': [size, size],
        'hidden_weights': [[int(i == j) for j in range(size)] for i in range(size)],
        'inputs': [[float(value) for value in row] for row in inputs],
        'output_weights': [[1 for _ in range(size)]],
        'targets': [0 for _ in range(size)],
        'mean_absolute_loss': 1,
        'hidden_gradient': [[float(value) for value in row] for row in gradient],
        'hidden_gradient_exact': [[str(value) for value in row] for row in gradient],
        'output_head_gradient': [[float(Fraction(1, size)) for _ in range(size)]],
        'hidden_rank': size, 'output_head_rank': 1,
        'positive_eigenvalue_certificate': {
            'parallel_to_ones': str(parallel),
            'orthogonal_to_ones': str(orthogonal),
            'orthogonal_multiplicity': size-1,
            'determinant': str(parallel*orthogonal**(size-1)),
        },
    }


def momentum_example():
    """Three independent diagonal directions, one rank-one gradient per step."""
    beta = Fraction(1, 2)
    diagonal = [Fraction(0)]*3
    rows = []
    for step in range(3):
        diagonal = [beta*value for value in diagonal]
        diagonal[step] += 1-beta
        rows.append({
            'step': step+1, 'gradient_rank': 1,
            'buffer_diagonal': [float(value) for value in diagonal],
            'buffer_diagonal_exact': [str(value) for value in diagonal],
            'buffer_rank': sum(value != 0 for value in diagonal),
            'buffer_effective_rank': effective_rank([float(value) for value in diagonal]),
        })
    return {'beta': str(beta), 'initial_buffer': 'zero',
            'update': 'B_t=beta*B_(t-1)+(1-beta)*G_t', 'steps': rows}


def report():
    return {
        'construction': 'Scalar-output ReLU network, identity hidden weights, '
                        'ones output head, nonzero preactivations, positive '
                        'residuals, mean absolute regression loss.',
        'scope': 'Constructed differentiable points and exact rank '
                 'certificates; not measured VLA gradients or model training. '
                 'The EMA example uses separate rank-one gradient directions.',
        'hidden_examples': [hidden_example(size) for size in (2, 3, 8)],
        'momentum': momentum_example(),
    }


if __name__ == '__main__':
    print(json.dumps(report(), indent=2, allow_nan=False))
