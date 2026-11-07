"""Small algebra checks for the Weight-Space Programming article.

Standard library only. These constructed examples do not run an LLM, rebuild
the released Qwen edit, or reproduce any reported game/benchmark experiment.
"""

import argparse
import itertools
import math


def relu(value):
    return max(0.0, value)


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("Vector dimensions differ")
    return math.fsum(a * b for a, b in zip(left, right))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def mlp(first, bias, second, value):
    preactivation = matvec(first, value)
    if len(preactivation) != len(bias):
        raise ValueError("Hidden bias dimension differs")
    hidden = [relu(v + b) for v, b in zip(preactivation, bias)]
    return matvec(second, hidden)


def remove_column_direction(matrix, direction, selected):
    if not matrix or not matrix[0] or len(matrix) != len(direction):
        raise ValueError("Matrix and direction dimensions differ")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("Matrix must be rectangular")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("Matrix must be finite")
    if any(not math.isfinite(value) for value in direction):
        raise ValueError("Direction must be finite")
    selected = tuple(selected)
    if any(type(index) is not int or not 0 <= index < width for index in selected):
        raise ValueError("Selected column is outside the matrix")
    if not math.isclose(dot(direction, direction), 1.0, abs_tol=1e-12):
        raise ValueError("Direction must have unit norm")
    result = [row[:] for row in matrix]
    for column in selected:
        coefficient = dot(direction, [row[column] for row in matrix])
        for row, component in zip(result, direction):
            row[column] -= component * coefficient
    return result


def check():
    truth_rows = []
    for a, b in itertools.product((0, 1), repeat=2):
        conjunction = relu(a + b - 1)
        exclusive = relu(a - b) + relu(b - a)
        if conjunction != (a & b) or exclusive != (a ^ b):
            raise AssertionError("Boolean construction failed")
        truth_rows.append((a, b, int(conjunction), int(exclusive)))
    for value in (-10.0, -0.25, 0.0, 0.125, 15.0):
        if relu(value) - relu(-value) != value:
            raise AssertionError("Signed ReLU transport failed")

    first = [[1.0, -2.0], [-3.0, 0.5], [0.25, 2.0]]
    bias = [0.5, -0.5, 1.0]
    second = [[2.0, -1.0, 3.0], [-4.0, 0.5, 1.0]]
    permutation = [2, 0, 1]
    perm_first = [first[i] for i in permutation]
    perm_bias = [bias[i] for i in permutation]
    perm_second = [[row[i] for i in permutation] for row in second]
    for value in itertools.product((-2.0, -0.5, 0.0, 1.0, 3.0), repeat=2):
        if mlp(first, bias, second, value) != mlp(
            perm_first, perm_bias, perm_second, value
        ):
            raise AssertionError("Permutation changed the network function")

    # Columns 0 and 2 write into the first coordinate (the chosen direction).
    # Editing column 0 alone leaves column 2's contribution untouched.
    matrix = [[3.0, 2.0, -1.0], [4.0, -2.0, 5.0]]
    unit = [1.0, 0.0]
    edited = remove_column_direction(matrix, unit, {0})
    if edited != [[0.0, 2.0, -1.0], [4.0, -2.0, 5.0]]:
        raise AssertionError("Wrong selected-column edit")
    for hidden in itertools.product((0.0, 0.5, 2.0), repeat=3):
        before, after = matvec(matrix, hidden), matvec(edited, hidden)
        expected = [-3.0 * hidden[0], 0.0]
        if [a - b for a, b in zip(after, before)] != expected:
            raise AssertionError("Output change differs from the derived formula")
    # A sparse edit can affect an unbounded set of inputs when its neuron fires.
    if matvec(matrix, [2.0, 0.0, 0.0]) == matvec(edited, [2.0, 0.0, 0.0]):
        raise AssertionError("Expected an active-neuron intervention")
    try:
        remove_column_direction(matrix, [2.0, 0.0], {0})
    except ValueError:
        pass
    else:
        raise AssertionError("Non-unit direction was silently accepted")
    # A non-axis direction checks that subtraction is a projection, not merely
    # zeroing one coordinate. In 2D the surviving vector is its perpendicular
    # component, which supplies an independent expected value.
    direction, perpendicular = (0.6, 0.8), (-0.8, 0.6)
    nonaxis = remove_column_direction(matrix, direction, iter((0, 2)))
    for column in (0, 2):
        original = [row[column] for row in matrix]
        expected = [v * dot(perpendicular, original) for v in perpendicular]
        for row, value in zip(nonaxis, expected):
            assert math.isclose(row[column], value, abs_tol=1e-12)
    assert [row[1] for row in nonaxis] == [row[1] for row in matrix]
    assert matrix == [[3.0, 2.0, -1.0], [4.0, -2.0, 5.0]]
    for bad_matrix, bad_direction, bad_columns in (
        ([[1.0, 2.0], [3.0]], (1.0, 0.0), (0,)),
        (matrix, (1.0,), (0,)),
        (matrix, (float("nan"), 0.0), (0,)),
        ([[float("inf")]], (1.0,), (0,)),
        (matrix, (1.0, 0.0), (-1,)),
        (matrix, (1.0, 0.0), (3,)),
        (matrix, (1.0, 0.0), (True,)),
    ):
        try:
            remove_column_direction(bad_matrix, bad_direction, bad_columns)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed projection input was accepted")
    touched = 2 * 64 * 1024
    if touched != 131072 or not math.isclose(touched / 800000000 * 100, 0.016384):
        raise AssertionError("Nominal edit count is inconsistent")
    print("Boolean inputs (a, b, AND, XOR):", truth_rows)
    print("25 hidden-permutation cases and 27 column-edit cases passed")
    print("Touched positions: 131072; nominal fraction: 0.016384%")
    assert 65260 + 64928 == 130188
    assert math.isclose(64 / 3584, 1 / 56)
    print("Non-axis projection, untouched column, finite shape and index checks passed")
    print("Algebra only; no published experiment was reproduced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run the constructed checks")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        parser.print_help()
