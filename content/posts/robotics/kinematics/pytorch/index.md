---
title: pytorch 机械臂逆运动学迭代数值解
date: 2025-02-26
lastmod: 2026-09-05
draft: false
tags: ["Kinematics", "PyTorch", "Inverse Kinematics"]
categories: ["机器人技术"]
authors: ["chase"]
summary: "用 pytorch_kinematics 构建独立 FK–IK–FK 检查，明确批量维度、关节顺序、重试筛选、坐标转换和可微边界。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "用 pytorch_kinematics 构建独立 FK–IK–FK 检查，明确批量维度、关节顺序、重试筛选、坐标转换和可微边界。"
contentLanguage: "zh-CN"
reading_prerequisites: "PyTorch 张量与机器人 FK"
reading_focus: "先在 CPU 上验证已知可达目标，筛出收敛且合法的重试后再比较性能。"
related_posts:
  - "/posts/robotics/kinematics/pinocchio"
  - "/posts/cuda/warp"
math: true
---

## 先用小接口验证模型，再封装求解器

[pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) 提供基于 PyTorch 的 FK、Jacobian 与批量迭代 IK。原笔记中的大封装依赖未提供的基类和成员变量，不能直接运行；这里改成独立的 FK → IK → FK 检查脚本。

适用范围是具有明确关节限位的串联链。自由浮动基座、mimic 关节、连续关节周期选择、碰撞以及工具偏置需要额外建模，不能自动等同于“任意 URDF 都能解”。

## 数据约定

| 数据 | 形状/约定 |
| --- | --- |
| 关节输入 | `(B, n)`，顺序取自 `get_joint_parameter_names()` |
| FK 矩阵 | `(B, 4, 4)`，串联链根坐标系下的末端 link 位姿 |
| Jacobian | `(B, 6, n)`，核对末端参考点与表达坐标系 |
| IK 结果 | `(目标数, 重试数, n)`，每次重试有独立收敛标记 |

旋转关节通常使用弧度，移动关节使用米。所有张量与 chain 必须在同一 device 和 dtype；批量大小为 1 时也不要随意删除 batch 维度。

## 完整检查脚本

在已安装匹配 PyTorch 和 `pytorch-kinematics` 的环境中保存为 `check_ik.py`，执行 `python check_ik.py robot.urdf end_link`。目标由同一模型的 FK 生成，只验证接口闭环，不证明真实机器人的标定精度。

暂时没有机器人模型时，可下载本文的 [二连杆测试 URDF](planar2r.urdf)，与脚本放在同一目录后执行 `python check_ik.py planar2r.urdf tip`。模型包含两个有界旋转关节和一个固定工具变换，不依赖网格文件；关节名称应打印为 `shoulder`、`elbow`。这是运动学测试夹具，不是带惯量和碰撞模型的仿真机器人。

```python
import argparse
from pathlib import Path

import torch
import pytorch_kinematics as pk

parser = argparse.ArgumentParser()
parser.add_argument("urdf")
parser.add_argument("end_link")
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device("cpu")  # 先在 CPU 验证，再比较 CUDA 批量性能
dtype = torch.float64
chain = pk.build_serial_chain_from_urdf(
    Path(args.urdf).read_text(encoding="utf-8"), args.end_link
).to(dtype=dtype, device=device)
names = chain.get_joint_parameter_names()
limits = torch.tensor(chain.get_joint_limits(), dtype=dtype, device=device).T
if limits.shape != (len(names), 2):
    raise ValueError("Unexpected joint-limit shape")
if not torch.isfinite(limits).all() or not (limits[:, 0] < limits[:, 1]).all():
    raise ValueError("This example requires finite, ordered joint limits")

lower, upper = limits[:, 0], limits[:, 1]
q_known = (lower + 0.45 * (upper - lower)).unsqueeze(0)
target = chain.forward_kinematics(q_known)
target_matrix = target.get_matrix().detach()

solver = pk.PseudoInverseIK(
    chain,
    joint_limits=limits,
    num_retries=20,
    max_iterations=200,
    pos_tolerance=1e-4,
    rot_tolerance=1e-4,
    early_stopping_any_converged=True,
    lr=0.2,
)
result = solver.solve(target)
candidates = result.solutions[0][result.converged[0]]
if candidates.numel() == 0:
    raise RuntimeError("No converged retry; inspect residuals and initializations")
valid = (
    torch.isfinite(candidates).all(dim=-1)
    & (candidates >= lower - 1e-8).all(dim=-1)
    & (candidates <= upper + 1e-8).all(dim=-1)
)
candidates = candidates[valid]
if candidates.numel() == 0:
    raise RuntimeError("Converged retries failed joint-limit validation")

# 在有限关节区间内选最接近参考姿态的一组；连续关节需另做周期距离。
distances = torch.linalg.vector_norm(candidates - q_known, dim=-1)
q_solution = candidates[distances.argmin()].unsqueeze(0)
actual = chain.forward_kinematics(q_solution).get_matrix()
position_error = torch.linalg.vector_norm(
    actual[0, :3, 3] - target_matrix[0, :3, 3]
)
relative_rotation = actual[0, :3, :3].T @ target_matrix[0, :3, :3]
cosine = ((torch.trace(relative_rotation) - 1) / 2).clamp(-1, 1)
rotation_error = torch.acos(cosine)
print("joints:", names)
print("q:", q_solution)
print("position error (m):", position_error.item())
print("rotation error (rad):", rotation_error.item())
if position_error > 2e-4 or rotation_error > 2e-4:
    raise RuntimeError("Independent FK check failed")
```

不能因为 `converged_any` 为真就返回 `solutions[:, 0, :]`：第 0 次重试未必是成功的那一次。应先按每次重试的收敛标记过滤，再验证限位与 FK 残差。

## 工具与世界坐标的换算

如果目标描述的是 TCP，而 chain 的末端是 flange，应先用已知工具外参把目标换算到 flange。世界坐标目标也需转换到 chain 根坐标系：

$$
{}^rT_f=({}^wT_r)^{-1}\,{}^wT_{\mathrm{tcp}}\,
({}^fT_{\mathrm{tcp}})^{-1}.
$$

本文使用列向量，$ {}^aT_b$ 把 b 系坐标变到 a 系。不要把别的图形库的行向量变换直接复制进来。

## 可微与性能的边界

FK 支持自动微分，不意味着任意带重试、早停、候选选择和裁剪的 IK 调用都是端到端可微映射。训练时需明确梯度经过哪些操作，避免 `detach`、转 NumPy 或重建 tensor 意外断开计算图。

GPU 优势取决于 batch 大小、迭代次数与数据传输。计时时区分首次初始化、预热和稳定执行；CUDA 异步执行还需同步后计时。失败目标、奇异附近目标和实际控制周期都应单独测试。


## 阅读自测与验收

- 打印 batch、重试维度、关节顺序和 device/dtype，先用 CPU 与已知目标完成一次 FK→IK→FK 对照。
- 只选择已收敛且满足限位的候选，再按连续性等任务标准选择解；返回张量的存在不等于每个重试都成功。
