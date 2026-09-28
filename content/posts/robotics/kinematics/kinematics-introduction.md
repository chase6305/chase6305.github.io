---
title: "机器人运动学基础"
date: 2024-04-01
lastmod: 2026-09-28
draft: true
authors: ["chase"]
tags: ["Robotics", "Kinematics", "D-H Parameters"]
categories: ["机器人技术"]
series: ["机器人运动学教程"]
series_weight: 1
summary: "梳理机器人正逆运动学、坐标约定、Jacobian 与数值验证的基础路线，连接六轴、七轴及批量求解专题。"
description: "梳理机器人正逆运动学、坐标约定、Jacobian 与数值验证的基础路线，连接六轴、七轴及批量求解专题。"
contentLanguage: "zh-CN"
reading_prerequisites: "线性代数、三角函数与 Python"
reading_focus: "从同一模型的 FK 闭环验证开始，再增加限位、奇异和碰撞条件。"
related_posts:
  - "/posts/robotics/kinematics/jacobian"
  - "/posts/robotics/kinematics/pinocchio"
math: true
---

## 先统一模型、坐标与单位

运动学描述关节配置与机器人几何状态的关系，不讨论产生运动所需的力矩。本文先用平面二连杆把 FK、IK 和 Jacobian 接成一个能手算、能运行的闭环，再连接到六轴、七轴和数值求解。

采用列向量约定，$ {}^aT_b$ 把 b 坐标系中的点变换到 a 坐标系；连乘顺序必须让相邻坐标系衔接。角度、长度、关节顺序、零位与工具偏置都属于模型的一部分。

齐次变换的平移表示子坐标系原点在父坐标系中的位置。求逆不仅是对平移取负，而是 $R^{-1}=R^\top$、$t^{-1}=-R^\top t$。旋转和平移的作用顺序通常不能交换，可参考 [Modern Robotics 的齐次变换说明](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/)。

## 三个核心问题

| 问题 | 输入 | 输出 | 常见误区 |
| --- | --- | --- | --- |
| 正运动学 FK | 关节配置 | 连杆或末端位姿 | 混用标准 DH 与改进 DH |
| 逆运动学 IK | 目标位姿与初值/约束 | 一组或多组关节候选 | 把数值停滞当成目标不可达证明 |
| Jacobian | 当前配置 | 速度的局部映射 | 混用参考点、表达坐标系和 twist 顺序 |

## 平面二连杆的 FK

令两根连杆长度为 $l_1=1$ m、$l_2=0.5$ m。$q_1$ 是第一连杆相对世界 x 轴的角度，$q_2$ 是第二连杆相对第一连杆的角度，均以逆时针为正。

$$
x=l_1\cos q_1+l_2\cos(q_1+q_2),\qquad
y=l_1\sin q_1+l_2\sin(q_1+q_2).
$$

当 $(q_1,q_2)=(0,\pi/2)$ 时，末端为 $(1,0.5)$ m。若把第二连杆方向错写成 $q_2$，在这个特例上可能碰巧正确；再检查 $(\pi/2,0)$，正确末端应为 $(0,1.5)$ m。选择多个不同姿态，比只看零位更容易发现坐标错误。

## IK 为什么可能有两组解，或根本无解

只给定平面位置 $(x,y)$，由余弦定理可得：

$$
c_2=\frac{x^2+y^2-l_1^2-l_2^2}{2l_1l_2},\qquad
q_2=\operatorname{atan2}\left(\pm\sqrt{1-c_2^2},c_2\right),
$$

$$
q_1=\operatorname{atan2}(y,x)
-\operatorname{atan2}(l_2\sin q_2,l_1+l_2\cos q_2).
$$

若 $|c_2|>1$ 且超出浮点容差，目标不在这个模型的位置工作空间中。不能直接把明显越界的数裁剪到 $[-1,1]$，再返回一个貌似有效的边界解。

对 $(1,0.5)$ m，两个位置解约为 $(0,1.5708)$ rad 和 $(0.9273,-1.5708)$ rad。末端位置相同，但末端方向 $q_1+q_2$ 不同。如果同时要求某个末端朝向，就增加了任务条件，不能继续把这两组都算作完整位姿解。

### 完整 NumPy 检查

下面的函数固定使用上述不等长模型，不处理关节限位和碰撞。无解返回空列表；工作空间边界处重合的分支会合并。

~~~python
import numpy as np

L1, L2 = 1.0, 0.5

def fk(q):
    q1, q2 = np.asarray(q, dtype=float)
    return np.array([L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
                     L1 * np.sin(q1) + L2 * np.sin(q1 + q2)])

def ik_position(target):
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target must contain two finite coordinates")
    x, y = target
    c2 = (x*x + y*y - L1*L1 - L2*L2) / (2 * L1 * L2)
    if abs(c2) > 1 + 1e-12:
        return []
    c2 = np.clip(c2, -1, 1)   # 只吸收边界附近的浮点误差
    angle = np.arccos(c2)
    branches = [angle] if abs(np.sin(angle)) < 1e-12 else [angle, -angle]
    solutions = []
    for q2 in branches:
        q1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(q2), L1+L2*np.cos(q2))
        q = np.array([q1, q2])
        np.testing.assert_allclose(fk(q), target, atol=1e-10)
        solutions.append(q)
    return solutions

np.testing.assert_allclose(fk([0, np.pi/2]), [1, .5], atol=1e-12)
np.testing.assert_allclose(fk([np.pi/2, 0]), [0, 1.5], atol=1e-12)
solutions = ik_position([1, .5])
assert len(solutions) == 2
assert len(ik_position([1.5, 0])) == 1
assert len(ik_position([.5, 0])) == 1
assert ik_position([2, 0]) == []
assert ik_position([0, 0]) == []
print("position IK candidates [rad]:", solutions)
~~~

几何候选生成后，还要检查角度等价表示、关节限位、碰撞和与当前姿态的距离。对于连续关节，比较 $\pi-\epsilon$ 与 $-\pi+\epsilon$ 时尤其需要处理角度环绕；它们在角度数值上相差接近 $2\pi$，在几何上却很接近。

## Jacobian 是当前位置附近的速度关系

对上面的 FK 求导，得到位置 Jacobian：

$$
\begin{bmatrix}\dot x\\\dot y\end{bmatrix}=
\underbrace{\begin{bmatrix}
-l_1\sin q_1-l_2\sin(q_1+q_2)&-l_2\sin(q_1+q_2)\\
l_1\cos q_1+l_2\cos(q_1+q_2)&l_2\cos(q_1+q_2)
\end{bmatrix}}_{J(q)}
\begin{bmatrix}\dot q_1\\\dot q_2\end{bmatrix}.
$$

它不是把一个很远的末端目标一次性转换成关节角的公式。数值 IK 可以利用它构造小步更新，但每步都应重新计算非线性 FK 误差。

当 $q_2=0$ 或 $\pi$ 时，$\det J=l_1l_2\sin q_2=0$，位置速度映射失去一个独立方向。阻尼可以限制数值更新，不能让机器人在该位形瞬间获得缺失的运动方向。进一步的参考系、奇异值和有限差分验证见 [Jacobian 专题]({{< relref "/posts/robotics/kinematics/jacobian" >}})。

## 最小验证闭环

1. 在有效关节范围内选择配置 q。
2. 用 FK 生成一个确定可达的末端目标。
3. 从另一个初值求 IK，并检查求解状态与限位。
4. 对候选重新做 FK，分别报告位置和姿态误差。

冗余机器人可能返回不同关节角但相同末端位姿；这不是错误。反过来，角度接近原值也不能代替末端残差检查。

先用固定基标量关节模型学习，再增加连续关节、浮动基、任务冗余和碰撞。配置维数 nq 与速度维数 nv 不总相同，含四元数时不能直接逐元素相加更新。

| 接下来想解决的问题 | 可继续阅读 |
| --- | --- |
| 六轴机械臂的坐标建模与解分支 | [六自由度正逆运动学]({{< relref "/posts/robotics/kinematics/six-dof-kinematics" >}}) |
| 冗余七轴如何选取解 | [七轴 SRS 几何逆解]({{< relref "/posts/robotics/kinematics/seven-dof-kinematics" >}}) |
| 使用真实模型做数值 IK | [Pinocchio 实现]({{< relref "/posts/robotics/kinematics/pinocchio" >}}) |
| 多任务与限位如何组合 | [Pink 微分 IK]({{< relref "/posts/pink" >}}) |
| 为什么随机点云不能代表全部能力 | [位置工作空间估计]({{< relref "/posts/robotics/workspace/whole-workspace" >}}) |


## 阅读自测与验收

- 给一个两关节平面臂手算 FK，再用数值扰动解释雅可比，最后讨论目标不可达或逆解多解。
- 在学习笔记中把几何解、数值解、限位和碰撞分开；草稿路线不意味着已实现通用机器人求解器。
