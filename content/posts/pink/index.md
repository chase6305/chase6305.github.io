---
title: 'Pink: 一个高效易用的机器人逆运动学库'
date: 2026-02-14
lastmod: 2026-02-14
draft: false
tags: ["Pink"]
categories: ["编程技术"]
authors: ["chase"]
summary: "Pink: 一个高效易用的机器人逆运动学库"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

参考: https://stephane-caron.github.io/pink/ 和 https://github.com/stephane-caron/pink?tab=readme-ov-file

在机器人运动规划与控制中，逆运动学（Inverse Kinematics, IK） 是一个核心问题：给定机器人期望的末端位置（比如机械臂要抓取的物体），如何计算出各个关节应该转动的角度？当机器人需要同时满足多个目标（如“伸手拿杯子”的同时还要“保持身体平衡”），并且受到关节限位等约束时，问题就变得更加复杂。

今天，我们来探索一个实测非常好用的 Python 库——Pink。它基于二次规划（Quadratic Programming, QP）来求解带约束的逆运动学问题，并由法国学者 Stéphane Caron 开发维护，兼具扎实的数学基础和优秀的工程实现。

### 什么是 QP-based IK？

传统的逆运动学方法在处理多个冲突的目标时往往力不从心。**QP-based IK** 通过“加权任务”的方式优雅地解决了这个问题。

其核心思想是：
1.  **定义任务**：每个任务由一个**残差函数** $e(q)$ 定义，目标是将其驱零。例如，将脚移动到目标位置 $p_{foot}^{\star}$ 的任务可定义为：$$e(q) = p_{foot}^{\star} - p_{foot}(q) $$
2.  **一阶近似**：我们希望找到一个关节速度 $v$，使得任务沿着梯度下降方向优化，即满足 $J_e(q) v = -\alpha e(q)$。这里的 $J_e(q)$ 是任务雅可比矩阵，$\alpha$是任务增益。
3.  **构建优化问题**：当有多个任务时，我们无法同时完美满足所有任务。因此，将所有任务的目标统一到一个框架下，并通过**加权**来解决冲突，最终形成一个”二次规划（QP）“问题：
$$
    \min_{v} \sum_{\text{task } e} \| J_e(q) v + \alpha e(q) \|^2_{W_e} \quad \text{s.t.} \quad v_{\text{min}}(q) \leq v \leq v_{\text{max}}(q)
$$
    这里，$W_e$ 就是任务的权重，权重越高，优先级越高。

### Pink 库：让复杂的 QP-based IK 变得简单

Pink 正是基于上述理论，并依托强大的机器人动力学库 **Pinocchio** 构建的。它将复杂的 QP 问题封装在简洁的 API 背后，让用户能专注于机器人本身的任务定义。

#### Pink 是如何工作的？

Pink 在求解时，会将连续的 QP 问题离散化，将优化变量由速度 $v$ 转换为关节空间的增量 $\Delta q = v \Delta t$。

对于一个单独的任务，其贡献给总优化目标的函数是：
$$
\| J \Delta q + \alpha e \|^2_W + \mu \|\Delta q\|^2
$$
*   $J, e$：由具体任务计算出的雅可比矩阵和误差。
*   $\alpha$：**任务增益**（gain），控制任务收敛的速度。
*   $W$：**权重矩阵**（cost），定义任务内不同维度（如位置 vs 姿态）的重要性。
*   $\mu$：**Levenberg-Marquardt 阻尼**（lm_damping），用于保证数值求解的稳定性。

PINK 会将所有任务（Task）、障碍物约束（Barrier）和关节限制（Limit）的目标和约束，统一转换为标准 QP 形式 $\frac{1}{2} \Delta q^T H \Delta q + c^T \Delta q$ 的矩阵 $H$ 和向量 $c$，然后调用其自带的 `qpsolvers` 库（支持 OSQP、DAQP 等多种求解器）进行高效求解。

### 核心任务详解：FrameTask 与 PostureTask

Pink 预置了多种任务类型，其中最常用的是以下两个：

#### 1. 末端位姿跟踪任务（FrameTask）
**目标**：驱动机器人的某个“框架”（如夹爪、脚底）到达目标位置和姿态。

这是 IK 中最核心的任务。它的难点在于，机器人的位姿变化是定义在李群上的，而误差和雅可比的求导不能简单地在欧式空间进行。很多 IK 库在这里会出错。

Pink 的作者 Stéphane Caron 在其[博客](https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html)中特别澄清了这一点，并在 Pink 中给出了正确实现。简单来说，它通过在**李代数**（切空间）上定义误差 $e$ 和雅可比 $J$，保证了数学上的精确性，从而让末端运动更加精准稳定。

#### 2. 关节角任务（PostureTask）
**目标**：让机器人的关节角度趋向于一个给定的“舒适”姿态 $q^*$。
这个任务通常用作**正则化项**。当主任务（如 FrameTask）因为奇异或不可达而失效时，PostureTask 可以发挥作用，防止关节速度失控发散，驱动机器人回到一个安全的姿态。
PostureTask 的 QP 目标函数为：

$$H_{task}=J^TWJ+μI$$
$$c_{task}= -\alpha e^T W J$$

它的计算非常简单：
*   误差：$e = q^* - q$
*   雅可比：$J = I$（单位阵）
通过调整其权重 $W$，可以控制关节回正的“力度”。

#### 注意事项
1. PostureTask 不会影响浮动基座的自由度，只作用于实际的关节角。
2. 由于其雅可比恒为满秩，PostureTask 常用于正则化，防止主任务奇异时解发散
3.  任务权重$W$可以用来调节不同关节的优先级或灵敏度
### 与Pinocchio+CasADi的区别
1. Pink: 专用 IK 求解器. 专注于解决多任务、带约束的微分逆运动学问题。它是一个“开箱即用”的工具。
2. Pinocchio+CasADi: 通用开发框架. Pinocchio 提供高效的运动学和动力学算法 ，CasADi 提供符号计算和数值优化。两者结合，用于构建自定义的、复杂的优化问题，如轨迹优化、MPC等。

