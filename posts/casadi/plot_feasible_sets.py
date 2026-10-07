#!/usr/bin/env python3
"""Plot and verify the article's two constrained problems.

python plot_feasible_sets.py --output /tmp/casadi-figures
Requires CasADi, NumPy and Matplotlib. Uses the same solver settings,
constraints and acceptance tolerances as the article.
"""
import argparse
import json
from pathlib import Path
import casadi as ca

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def solve_example(equal_xy=False):
    z = ca.SX.sym("z", 2)
    x, y = z[0], z[1]
    objective = (x - 1)**2 + (y - 2)**2
    constraints = ca.vertcat(x + y - 1, x - y) if equal_xy else x + y - 1
    problem = {"x": z, "f": objective, "g": constraints}
    solver = ca.nlpsol(
        "two_equalities" if equal_xy else "one_equality",
        "ipopt", problem,
        {"ipopt.print_level": 0, "print_time": False, "ipopt.tol": 1e-10},
    )
    lower = np.array([0.0, -np.inf])
    upper = np.array([np.inf, 1.0])
    count = int(constraints.numel())
    solution = solver(
        x0=[0.25, 0.75],
        lbx=lower, ubx=upper,
        lbg=np.zeros(count), ubg=np.zeros(count),
    )
    status = solver.stats()
    if not status.get("success", False):
        raise RuntimeError(status.get("return_status", "unknown solver failure"))

    values = np.asarray(solution["x"]).ravel()
    residual = np.asarray(solution["g"]).ravel()
    if not np.isfinite(values).all() or not np.isfinite(residual).all():
        raise RuntimeError("nonfinite solver output")
    tolerance = 1e-7
    if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
        raise RuntimeError("variable bound violated")
    if np.max(np.abs(residual)) > tolerance:
        raise RuntimeError("equality constraint violated")

    expected = [0.5, 0.5] if equal_xy else [0.0, 1.0]
    # 边界解会受内点法容差影响，因此比较数值误差，而不是浮点严格相等。
    np.testing.assert_allclose(values, expected, atol=3e-4, rtol=0)
    expected_cost = 2.5 if equal_xy else 2.0
    np.testing.assert_allclose(float(solution["f"]), expected_cost, atol=1e-6)
    return values, float(solution["f"]), status["return_status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/casadi-figures'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [solve_example(equal_xy) for equal_xy in (False, True)]
    records = []
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    xx, yy = np.meshgrid(np.linspace(-.5, 2.5, 400), np.linspace(-1.5, 2.5, 400))
    objective = (xx - 1)**2 + (yy - 2)**2
    x = np.linspace(-.5, 2.5, 400)
    for index, (ax, result) in enumerate(zip(axes, results)):
        solution, cost, status = result
        records.append({'equal_xy': bool(index), 'solution': solution.tolist(),
                        'objective': cost, 'status': status})
        ax.add_patch(Rectangle((0, -1.5), 2.5, 2.5, color='#e7f3ed', zorder=0,
                               label='Variable bounds: x >= 0, y <= 1'))
        contours = ax.contour(xx, yy, objective, levels=[.5, 1, 2, 2.5, 4, 6, 10],
                              colors='#a5b4c7', linewidths=.9)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%g')
        ax.plot(x, 1 - x, '--', color='#4d83bc', lw=1.5, label='Equality: x + y = 1')
        if index == 0:
            feasible_x = np.linspace(0, 2.5, 250)
            ax.plot(feasible_x, 1 - feasible_x, color='#13856d', lw=4,
                    label='Feasible ray (continues beyond view)')
            ax.annotate('', xy=(2.47, -1.47), xytext=(2.17, -1.17),
                        arrowprops={'arrowstyle': '->', 'color': '#13856d', 'lw': 3})
        else:
            ax.plot(x, x, '--', color='#9b73b8', lw=1.8, label='Extra equality: x = y')
            ax.scatter([.5], [.5], s=230, facecolors='none', edgecolors='#13856d',
                       linewidths=3, zorder=5, label='Only feasible point')
        ax.scatter([1], [2], marker='x', color='#606c80', s=75, zorder=5,
                   label='Unconstrained minimum (infeasible)')
        ax.scatter(solution[0], solution[1], s=70, color='#db863d', edgecolor='white',
                   zorder=6, label='IPOPT solution')
        ax.annotate(f'({solution[0]:.1f}, {solution[1]:.1f})\nf = {cost:.1f}',
                    xy=solution, xytext=(.35, -.1) if index == 0 else (1.35, .45),
                    arrowprops={'arrowstyle': '-', 'color': '#db863d'},
                    fontsize=12, color='#875329')
        ax.set(xlim=(-.5, 2.5), ylim=(-1.5, 2.5), xlabel='x', ylabel='y',
               title='One equality: a feasible ray' if index == 0 else 'Two equalities: a single point')
        ax.set_aspect('equal')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), frameon=False, fontsize=8)
    fig.suptitle(r'Minimize $(x-1)^2+(y-2)^2$ subject to all stated constraints', fontsize=15)
    fig.savefig(args.output/'feasible-sets.png', dpi=170)
    plt.close(fig)
    payload = {'casadi_version': ca.__version__, 'problems': records}
    (args.output/'feasible-sets-results.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
