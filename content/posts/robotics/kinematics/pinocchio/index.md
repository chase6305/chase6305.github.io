---
title: Pinocchio 机械臂逆运动学迭代数值解
date: 2024-08-18
lastmod: 2026-09-28
draft: false
tags: ["Kinematics", "Pinocchio", "Python"]
categories: ["机器人技术"]
authors: ["chase"]
summary: "使用 Pinocchio frame、SE(3) 残差与阻尼最小二乘求 IK，解释单位尺度、限位和失败状态，并用独立 FK 验证。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "使用 Pinocchio frame、SE(3) 残差与阻尼最小二乘求 IK，解释单位尺度、限位和失败状态，并用独立 FK 验证。"
contentLanguage: "zh-CN"
reading_prerequisites: "Pinocchio、Jacobian 与李群基础"
reading_focus: "本例限固定基标量关节，检查 frame 名称和误差坐标，不把失败候选当成功返回。"
related_posts:
  - "/posts/robotics/kinematics/jacobian"
  - "/posts/robotics/kinematics/pytorch"
math: true
---

Pinocchio 提供正运动学、雅可比与李群运算，数值逆解需要在这些工具之上定义误差、迭代策略和失败条件。本文使用**末端 frame**作为目标，避免把 URDF 的 link 名称当作 joint 名称再减一。

![数值逆运动学中的目标、正运动学、位姿误差、阻尼求解和限位检查](assets/numerical-ik-loop.webp "先统一误差与雅可比的参考系，再迭代更新关节。只有末端残差与关节限位都通过检查，才能报告成功。")

## 1. 模型和坐标约定

本例适用于固定基座、每个关节以一个标量表示的有界转动或移动关节。浮动基、球关节及使用二维配置表示的连续转动关节，不能直接使用逐元素限位裁剪。

目标 `target` 和当前末端 `current` 都是基座中的位姿。定义：

$$
E = T_{\mathrm{current}}^{-1}T_{\mathrm{target}},
\qquad e=\log(E).
$$

误差与 `LOCAL` frame Jacobian 配合使用，并通过 `Jlog6` 构造误差的雅可比。阻尼最小二乘求一个局部更新；它不保证从任意初值收敛，也不包含碰撞检测。[Pinocchio 官方逆运动学示例](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b_examples_d_inverse_kinematics.html)

## 2. 可复用的求解函数

完整文件可下载 [frame_ik.py](frame_ik.py)，包含下方函数和第 3 节的命令行入口；安装 NumPy 与 `pin` 后即可运行。

```python
from numbers import Integral

import numpy as np
import pinocchio as pin


def solve_ik(model, frame_name, target, q0, max_iter=500,
             position_tol=1e-4, rotation_tol=1e-4, damping=1e-3,
             position_scale=1.0, rotation_scale=1.0):
    if model.nv == 0 or any(j.nq != 1 or j.nv != 1 for j in model.joints[1:]):
        raise ValueError("Only fixed-base scalar joints are supported")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    settings = np.array([position_tol, rotation_tol, damping,
                         position_scale, rotation_scale], dtype=float)
    if not np.isfinite(settings).all() or np.any(settings <= 0):
        raise ValueError("Tolerances, damping and scales must be finite and positive")
    row_scale = 1.0 / np.array([position_scale] * 3 + [rotation_scale] * 3)
    if not np.isfinite(row_scale).all():
        raise ValueError("Residual scales are too small")
    frame_id = model.getFrameId(frame_name)
    if frame_id >= len(model.frames):
        raise ValueError(f"Unknown end-effector frame: {frame_name}")

    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()
            and np.all(lower < upper)):
        raise ValueError("Finite, ordered joint bounds are required")
    q = np.asarray(q0, dtype=float).copy()
    if q.shape != (model.nq,) or not np.isfinite(q).all():
        raise ValueError("q0 must be a finite vector of size model.nq")
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("q0 is outside joint limits")
    if not (np.isfinite(target.homogeneous).all()
            and np.allclose(target.rotation.T @ target.rotation,
                            np.eye(3), atol=1e-6, rtol=0)
            and np.isclose(np.linalg.det(target.rotation), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("target must contain a valid rigid rotation")

    data = model.createData()

    def evaluate(configuration):
        pin.forwardKinematics(model, data, configuration)
        pin.updateFramePlacements(model, data)
        current = data.oMf[frame_id]
        relative = current.actInv(target)
        error = pin.log6(relative).vector
        position_error = np.linalg.norm(current.translation - target.translation)
        rotation_error = np.linalg.norm(pin.log3(relative.rotation))
        return relative, error, position_error, rotation_error

    status = "iteration_limit"
    for iteration in range(max_iter):
        relative, error, pos_err, rot_err = evaluate(q)
        if pos_err <= position_tol and rot_err <= rotation_tol:
            status = "converged"
            break

        jacobian = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL
        )
        error_jacobian = -pin.Jlog6(relative.inverse()) @ jacobian
        error_jacobian = row_scale[:, None] * error_jacobian
        weighted_error = row_scale * error
        delta = -error_jacobian.T @ np.linalg.solve(
            error_jacobian @ error_jacobian.T + damping**2 * np.eye(6),
            weighted_error,
        )
        if not np.isfinite(delta).all():
            status = "nonfinite_step"
            break

        cost = weighted_error @ weighted_error
        accepted = False
        for step in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = np.clip(pin.integrate(model, q, step * delta),
                                lower, upper)
            _, candidate_error, _, _ = evaluate(candidate)
            candidate_error = row_scale * candidate_error
            if candidate_error @ candidate_error < cost:
                q = candidate
                accepted = True
                break
        if not accepted:
            status = "stagnation"
            break

    _, _, pos_err, rot_err = evaluate(q)
    success = bool(pos_err <= position_tol and rot_err <= rotation_tol
                   and np.all(q >= lower) and np.all(q <= upper))
    return {
        "success": success,
        "status": "converged" if success else status,
        "q": q,
        "iterations": iteration + 1,
        "position_error": float(pos_err),
        "rotation_error": float(rot_err),
    }
```

旋转误差单位为弧度，位置误差使用 URDF 的长度单位（通常为米）。默认两类 `scale` 都为 1，保留未加权六维残差的数值约定。若指定 `position_scale=0.1`、`rotation_scale=0.5`，则分别用 0.1 个长度单位与 0.5 rad 归一化；误差与 Jacobian 的对应行一起缩放。它们定义优化代价的相对尺度，并不改变 `position_tol` 和 `rotation_tol` 的独立验收条件。

## 3. 用正解生成可达目标

若从代码块手动保存，把下面代码接在同一个 `frame_ik.py` 文件末尾；下载版已经包含。目标由一组已知关节角生成，便于区分求解器问题和目标不可达问题：

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("urdf")
    parser.add_argument("frame")
    args = parser.parse_args()

    model = pin.buildModelFromUrdf(args.urdf)
    frame_id = model.getFrameId(args.frame)
    if frame_id >= len(model.frames):
        raise ValueError(f"Unknown frame: {args.frame}")
    q0 = np.clip(pin.neutral(model),
                 model.lowerPositionLimit, model.upperPositionLimit)
    known_q = np.clip(q0 + 0.1,
                     model.lowerPositionLimit, model.upperPositionLimit)
    data = model.createData()
    pin.forwardKinematics(model, data, known_q)
    pin.updateFramePlacements(model, data)
    target = data.oMf[frame_id].copy()

    result = solve_ik(model, args.frame, target, q0)
    print(result)
    if not result["success"]:
        raise SystemExit("IK failed: inspect residuals and seed")
```

```bash
python frame_ik.py /path/to/robot.urdf ee_link
```

URDF 路径和 frame 名称必须替换为实际模型中的值。测试近目标、远目标、关节限位附近与不可达目标，并记录返回状态。

## 4. 限位处理的边界

只在收敛后把角度加减 $2\pi$，不能保证获得合法解；任意裁剪也可能改变末端位姿。本例在每次候选更新时检查限位，最后重新计算 FK 残差。

裁剪式更新仍可能在边界停滞。更复杂的项目可使用带限位的优化问题、多初值搜索或任务层约束求解，但依然需要明确失败返回，不能把未收敛的关节角直接交给控制器。


## 5. 尺度、容差与收敛状态应分别验证 {#ik-scaling-validation}

令 $S=\operatorname{diag}(s_p^{-1}I_3,s_R^{-1}I_3)$，函数实际使用 $\tilde e=Se$、$\tilde J=SJ_e$，求解

$$
\min_{\Delta q}\;\|\tilde J\Delta q+\tilde e\|_2^2
+\lambda^2\|\Delta q\|_2^2.
$$

只给误差乘权重、却不缩放 Jacobian，并不是同一个目标函数。若把几何模型的米改成毫米，位姿平移、$s_p$ 和位置容差都应乘 1000，旋转与关节角约定不变；此时归一化的局部优化问题保持一致。这里仅讨论长度单位换算，不代表可以随意缩放质量与惯量。

还应区分三个量：

- `position_scale`、`rotation_scale` 决定代价中的相对尺度；
- `damping` 控制局部更新的正则化；本文使用 $\lambda^2$，不同代码中叫作 `damp` 的变量未必采用同一约定；
- `position_tol`、`rotation_tol` 分别验收真实 FK 的位置距离与旋转角。SE(3) 对数的前三项与旋转耦合，并不总是简单的平移差。

`Jlog6` 的符号和左右乘次序可以对照 [Pinocchio 4.1.0 的 IK 示例](https://github.com/stack-of-tasks/pinocchio/blob/v4.1.0/examples/inverse-kinematics.py)，并用[独立残差差分检查]({{< relref "/posts/robotics/kinematics/jacobian#reference-point-duality" >}})验证；不应只凭一次求解收敛判断公式正确。

下载 [ik_validation.py](ik_validation.py)，与 `frame_ik.py` 放在同一目录后运行：

```bash
python -B ik_validation.py --output ik-validation.json
```

在 Pinocchio 4.1.0 的内置六轴模型上，固定种子的 60 个“已知可达目标、近初值”案例全部通过独立 FK 检查，最大位置误差约 $9.38\times10^{-5}$ m，最大姿态误差约 $6.10\times10^{-5}$ rad。米与毫米的对照均在第 3 次循环确认收敛，关节解最大差约 $3.33\times10^{-16}$ rad。[完整记录](assets/ik-validation.json)还包括 12 种无效输入的拒绝结果，以及一个报告 `stagnation` 的不可达目标。

这些结果只覆盖写明的初值和模型，不能证明任意目标都收敛。`stagnation` 也不是不可达性的数学证明：不合适的初值、限位裁剪和回溯步长都可能使可达目标停滞。对外接口应同时保留状态、末端残差与候选关节角，让调用方决定重试或失败处理。

## 阅读自测与验收

- 用已知 q 的 FK 构造目标，记录位置与旋转残差，再逐渐增大初值扰动；近初值成功不能推广到任意初值。
- 故意输入缺失 frame、越界 q0 和不可达目标，确认返回或异常语义明确；没有成功标记时不能直接下发关节值。
