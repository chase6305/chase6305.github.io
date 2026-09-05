---
title: 'Pink: 一个高效易用的机器人逆运动学库'
math: true
date: 2026-02-14
lastmod: 2026-09-05
draft: false
tags: ["Pink", "Inverse Kinematics", "Optimization"]
categories: ["机器人技术"]
authors: ["chase"]
summary: "用一致的残差与 Jacobian 解释 Pink 微分 IK，区分任务权重和严格优先级，说明限位、积分与求解失败处理。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "用一致的残差与 Jacobian 解释 Pink 微分 IK，区分任务权重和严格优先级，说明限位、积分与求解失败处理。"
contentLanguage: "zh-CN"
reading_prerequisites: "Jacobian、QP 与 Pinocchio"
reading_focus: "把一个控制周期的局部解和全局 IK 分开，权重不能代替缺失的硬约束。"
related_posts:
  - "/posts/robotics/kinematics/jacobian"
  - "/posts/robotics/kinematics/pinocchio"
---

## Pink 求的是带约束的微分 IK

Pink 依托 Pinocchio 计算运动学，把多个任务的一阶局部近似组合成 QP。它适合在每个控制周期求一个关节速度/切空间增量，再更新机器人配置；不是一次调用就保证找到任意目标的全局 IK。

[官方文档](https://stephane-caron.github.io/pink/)和[源码仓库](https://github.com/stephane-caron/pink)是接口与版本的核对入口。具体求解器由 qpsolvers 及已安装的后端提供，不是所有后端都默认可用。

## 残差、Jacobian 与符号必须成对使用

设任务残差为 $e(q)$，其对切空间增量的导数为 $J_e$：

$$
e(q\oplus\Delta q)\approx e(q)+J_e(q)\Delta q.
$$

一个教学形式的目标是：

$$
\min_{\Delta q}\frac12\|J_e\Delta q+\alpha e\|_W^2
+\frac{\lambda}{2}\|\Delta q\|^2,
\quad A\Delta q\le b.
$$

其展开为 $H=J_e^\top WJ_e+\lambda I$、$c=\alpha J_e^\top We$。若选择 $e=q^*-q$，欧氏空间中的导数是 $-I$；不能一边使用“目标减当前”的残差，一边直接把其导数写成 $+I$ 而不改变方程符号。

这是解释 QP 的简化模型，不是逐项复刻 Pink 的任务成本和 LM 阻尼实现。实际调用应使用同一个 Task 提供的 error 与 Jacobian，避免混用其他库的约定。

## FrameTask 与 PostureTask

| 任务 | 用途 | 边界 |
| --- | --- | --- |
| FrameTask | 跟踪指定 frame 的位置与姿态 | SE(3) 误差与几何 Jacobian 不可随意互换 |
| PostureTask | 将受控关节拉向参考姿态，提供正则化 | 不直接约束浮动基座，也不自动保证参考姿态无碰撞 |
| 约束/Barrier | 限位、速度或显式配置的几何条件 | 没添加的约束不会因使用 QP 自动存在 |

位置与姿态有不同单位，应明确 cost 的缩放。任务权重越高表示违反该任务的代价越大，**不构成严格优先级保证**；多个任务冲突时仍会折中。需要严格层级时，应采用相应的层级求解方案或硬约束，而不只是无限增大权重。

## 一个控制周期如何接起来

1. 用当前配置更新 FK，设置目标和必要任务。
2. 给出实际时间步长 `dt`，求解局部 QP。
3. 检查求解状态、速度上限、当前配置是否越界。
4. 使用配置流形上的 integrate 更新，不能对含四元数的配置直接 `q += v * dt`。
5. 重新计算非线性任务误差并决定继续、减速、重规划或安全停止。

离散任务增益与连续速度反馈增益的单位不同，修改 `dt` 后不能假设同一数值仍产生相同的闭环响应。姿态任务和阻尼有助于改善数值行为，但不能保证不可达目标下的全局收敛或硬件安全。

## 与 Pinocchio + CasADi 的分工

Pink 提供现成的微分 IK 任务框架。Pinocchio + CasADi 则用于构建更自由的非线性优化问题，可扩展到轨迹优化或 MPC，但需要自行选择残差、约束、初始化与求解流程。

选择标准应是问题结构：单周期、多任务微分控制可先验证 Pink；跨时间约束、动力学或复杂非线性目标则需要更完整的优化建模。

参考：[Pink 逆运动学接口](https://stephane-caron.github.io/pink/inverse-kinematics.html)、[任务定义](https://stephane-caron.github.io/pink/tasks.html)。


## 阅读自测与验收

- 对每个任务检查误差、雅可比、参考系和权重的单位；修改残差符号时对应雅可比也必须一致。
- 制造相互冲突的任务，检查加权折中与严格优先级的差别；QP 可解不代表结果已经满足碰撞和所有硬件限制。
