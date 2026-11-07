"""Two point agents on a four-node graph: fixed-path retiming versus yielding.

This is an exhaustive finite-graph example, not cuArms, PIBT, LaCAM, arm
collision checking, manipulation skills, or a continuous-robot completeness test.
"""

import argparse
from collections import deque
from itertools import permutations, product


GRAPH = {"L": ("C",), "C": ("L", "R", "Y"), "R": ("C",), "Y": ("C",)}
POSITIONS = {"L": (-1, 0), "C": (0, 0), "R": (1, 0), "Y": (0, 1)}


def transition_valid(before, after):
    if before[0] == before[1] or after[0] == after[1]:
        return False
    if any(b != a and b not in GRAPH[a] for a, b in zip(before, after)):
        return False
    return not (before[0] == after[1] and before[1] == after[0])


def breadth_first(start, goal, successors):
    parents = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return tuple(reversed(path))
        for candidate in successors(current):
            if candidate not in parents:
                parents[candidate] = current
                queue.append(candidate)
    return None


def coordinated(start, goal):
    def successors(state):
        choices = [(v,) + GRAPH[v] for v in state]
        return (n for n in product(*choices) if transition_valid(state, n))

    return breadth_first(start, goal, successors)


def fixed_path_retiming(paths):
    goal = tuple(len(p) - 1 for p in paths)

    def successors(progress):
        before = tuple(p[i] for p, i in zip(paths, progress))
        for increment in product((0, 1), repeat=2):
            candidate = tuple(i + step for i, step in zip(progress, increment))
            if any(i > limit for i, limit in zip(candidate, goal)):
                continue
            after = tuple(p[i] for p, i in zip(paths, candidate))
            if transition_valid(before, after):
                yield candidate

    return breadth_first((0, 0), goal, successors)


def minimum_pair_distance_squared(before, after):
    # Independent exact closest approach of simultaneous linear point motions.
    p0, p1 = (POSITIONS[v] for v in before)
    q0, q1 = (POSITIONS[v] for v in after)
    relative = [a - b for a, b in zip(p0, p1)]
    velocity = [(c - a) - (d - b) for a, b, c, d in zip(p0, p1, q0, q1)]
    vv = sum(v * v for v in velocity)
    fraction = max(0.0, min(1.0, -sum(p * v for p, v in zip(relative, velocity)) / vv)) if vv else 0.0
    return sum((p + fraction * v) ** 2 for p, v in zip(relative, velocity))


def check():
    states = tuple(permutations(GRAPH, 2))
    for before, after in product(states, repeat=2):
        adjacent = all(b == a or b in GRAPH[a] for a, b in zip(before, after))
        oracle = adjacent and minimum_pair_distance_squared(before, after) > 1e-12
        assert transition_valid(before, after) == oracle
    for start, goal in product(states, repeat=2):
        route = coordinated(start, goal)
        assert route is not None and route[0] == start and route[-1] == goal
        assert all(transition_valid(a, b) for a, b in zip(route, route[1:]))
    assert fixed_path_retiming((("L", "C", "R"), ("R", "C", "L"))) is None
    route = coordinated(("L", "R"), ("R", "L"))
    # Exchange requires one point to enter and leave leaf Y, giving >=4 moves
    # for that point. The returned four-step route meets this independent bound.
    assert len(route) - 1 == 4
    assert any("Y" in state for state in route)
    moves = sum(a != b for old, new in zip(route, route[1:]) for a, b in zip(old, new))
    assert moves == 6
    print("Fixed shortest paths + arbitrary waits: no solution in the finite progress graph")
    print("Joint search with side node Y:")
    for step, state in enumerate(route):
        print(f"  {step}: A={state[0]} B={state[1]}")
    print("Makespan: 4 abstract steps; total edge moves: 6")
    print("144 transition comparisons and 144 start/goal queries passed; point agents only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else parser.print_help()
