---
title: "CasADi: 数值优化和自动微分库"
date: 2025-04-03
lastmod: 2026-09-05
draft: false
tags: ["CasADi", "Optimization", "Automatic Differentiation"]
categories: ["机器人技术"]
authors: ["chase"]
summary: "使用 CasADi 建立上下界和等式约束，区分变量边界与约束边界，并通过可手算例子验证求解状态和结果。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "使用 CasADi 建立上下界和等式约束，区分变量边界与约束边界，并通过可手算例子验证求解状态和结果。"
contentLanguage: "zh-CN"
reading_prerequisites: "Python、微分与约束优化"
reading_focus: "先核对 x、g 及边界维度，再比较数值解和解析预期。"
related_posts:
  - "/posts/pink"
  - "/posts/planner/to_mpc_wbc"
math: true
---

## 1. 先区分建模工具和求解器

CasADi 用符号计算图表达目标、约束及其导数；IPOPT 等求解器负责执行数值优化。自动微分不会把一般非凸问题变成凸问题，也不会让不可行约束自动获得解。

本文只做两件事：建立变量边界与等式约束，再用能手算的小问题检验结果。安装与符号类型可查 [CasADi 官方文档](https://web.casadi.org/docs/)。

```bash
python -m pip install casadi numpy
```

| 对象 | 含义 | 本例维度 |
| --- | --- | --- |
| `x` | 待求的变量向量 | 2 |
| `f` | 标量目标函数 | 1 |
| `g` | 约束表达式向量 | 1 或 2 |
| `lbx / ubx` | 变量下界 / 上界 | 与 x 一致 |
| `lbg / ubg` | 约束值下界 / 上界 | 与 g 一致 |
| `x0` | 求解器初值，不是最终解 | 与 x 一致 |

`SX` 和 `MX` 用于符号表达，`DM` 用于数值矩阵。不要直接用普通 NumPy 函数处理尚未求值的 CasADi 符号，也不要把有限差分近似与自动微分混为一谈。

## 2. 两个问题的解析预期

共同目标与变量边界：

$$
\min_{x,y}\;(x-1)^2+(y-2)^2,\qquad x\ge0,\quad y\le1.
$$

第一个问题增加 `x+y=1`。代入 `y=1-x` 后，目标是 `2x²+2`，所以最优点为 `(0,1)`，目标值为 `2`。

第二个问题再增加 `x=y`。两个等式共同确定唯一可行点 `(0.5,0.5)`，目标值为 `2.5`。这两个问题可用于验证建模代码，但不能据此声称通用非凸 NLP 有全局最优保证。

## 3. 一份可独立运行的代码

保存为 `casadi_bounds.py`。同一函数求解两种约束，分别检查求解状态、边界、约束残差和解析预期。绘图不参与正确性判断。

```python
import casadi as ca
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


if __name__ == "__main__":
    print("CasADi:", ca.__version__)
    for equal_xy in (False, True):
        print(equal_xy, solve_example(equal_xy))
```

```bash
python casadi_bounds.py
```

第一个解的 x 可能是很小的正数，而不是打印为精确的零。接受标准应来自问题尺度和容差，不应依靠四舍五入后的输出是否“看起来一样”。

## 4. 如何读图

![目标函数等高线、x+y=1 以及变量边界](casadi_1.png)

![在共同边界下同时加入 x+y=1 与 x=y](casadi_2.png)

图中直线表示等式或不等式边界，不代表整张平面都是可行域；需要同时满足全部条件。以上是保留的历史绘图，不是当前精简代码的自动输出。

若需要重画，单独安装 Matplotlib，用网格计算目标的等高线，再绘制约束直线和 `solve_example` 返回的点。不要用“图上有一个点”代替约束验收。

## 5. 修改模型时的排查顺序

1. 检查 `x / g` 的维度与对应上下界，等式必须使用相等的 `lbg / ubg`。
2. 用已知可行点直接计算约束，排除符号、单位和边界写反。
3. 检查初值、变量尺度、求解状态和残差；不可行问题不能只靠增加迭代次数解决。
4. 一般非凸问题需要多初值或更合适的建模策略，成功状态不等于全局最优。
5. 非光滑表达式在切换点需要专门处理，自动微分只对给定计算图按其规则求导。

插件是否可用取决于 CasADi 安装包及平台。遇到 IPOPT 加载失败，记录 CasADi 版本、解释器路径和原始错误；不要随意替换系统共享库。


## 阅读自测与验收

- 把变量边界和 g 的边界分别打印：第一例只有一个等式，第二例有两个；维度匹配不等于约束含义正确。
- 故意添加互相矛盾的约束，确认程序能报告失败，而不是继续把 sol['x'] 当作可用答案。
