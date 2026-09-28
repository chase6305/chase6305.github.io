---
title: 蒙特卡罗估计机器人位置工作空间：采样、覆盖与灵巧度
date: 2025-02-27
lastmod: 2026-09-28
draft: false
tags: ["Robotics", "Workspace Analysis", "Monte Carlo"]
categories: ["机器人技术"]
authors: ["chase"]
summary: "通过关节随机采样与 FK 估计机器人位置工作空间，给出可复现二连杆示例，并区分覆盖、密度和可达性。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "通过关节随机采样与 FK 估计机器人位置工作空间，给出可复现二连杆示例，并区分覆盖、密度和可达性。"
contentLanguage: "zh-CN"
reading_prerequisites: "FK、NumPy 与随机采样"
reading_focus: "点云是有限采样估计，不包含所有姿态能力或碰撞后的可行集合。"
related_posts:
  - "/posts/robotics/kinematics/jacobian"
  - "/posts/robotics/kinematics/pinocchio"
math: true
---


## 蒙特卡罗方法简介

蒙特卡罗方法（Monte Carlo Method）是一种通过随机采样来解决数学问题的数值计算方法。它广泛应用于各种领域，包括物理学、金融、工程和计算机科学。在机械臂的运动学和控制中，蒙特卡罗方法可以用于路径规划、逆运动学求解、碰撞检测等问题。

## 估计范围与限制

本文估计的是位置可达工作空间，有限随机采样不能穷尽“全工作空间”，也不能证明空白区域一定不可达。位置点云不包含每点可实现的姿态范围；加入碰撞和任务约束后，可行集合还会缩小。

关节空间均匀采样不会产生工作空间均匀点云，局部点密度也不直接等于灵巧度。应固定随机种子，比较不同样本量下的覆盖变化，并在边界处增加定向采样或 IK 验证。

### 采样流程 {#制作流程}

- 定义机械臂模型：确定机械臂的关节数、关节类型（旋转或平移）、关节角度范围等参数。
- 随机采样关节配置：在关节角度范围内随机生成大量的关节配置。
- 正向运动学计算：对于每个随机生成的关节配置，计算末端执行器的位置和姿态。
- 记录可达位置：将所有计算得到的末端执行器位置记录下来，形成机械臂的可达空间的估计。
- 可视化可达空间：将记录的可达位置进行可视化，展示机械臂的工作范围。

## 可复现的二连杆模型 {#案例代码}

下面是平面二连杆教学模型，长度单位为米、角度为弧度，没有障碍物或自碰撞检测。无关节限位时，其位置到原点的距离应落在 `abs(L1 - L2)` 到 `L1 + L2` 之间，可作为 FK 与采样结果的基本检查。

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义机器人的参数
L1 = 1.0  # 第一个连杆的长度
L2 = 0.6  # 第二个连杆的长度；不等长才能清楚看到内部不可达圆孔
num_samples = 10000  # 随机采样的数量

def forward_kinematics(theta1, theta2):
    """
    计算正向运动学，得到末端执行器的位置
    :param theta1: 第一个关节的角度
    :param theta2: 第二个关节的角度
    :return: 末端执行器的位置 (x, y)
    """
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y

# 随机采样关节配置
rng = np.random.default_rng(42)
theta1_samples = rng.uniform(0, 2*np.pi, num_samples)
theta2_samples = rng.uniform(0, 2*np.pi, num_samples)

# 计算末端执行器的位置
x, y = forward_kinematics(theta1_samples, theta2_samples)
positions = np.column_stack((x, y))  # NumPy 批量计算，不逐点进入 Python 循环
radii = np.linalg.norm(positions, axis=1)
assert positions.shape == (num_samples, 2)
assert np.isfinite(positions).all()
assert np.all(radii >= abs(L1 - L2) - 1e-12)
assert np.all(radii <= L1 + L2 + 1e-12)
# 两个确定性边界姿态，不能依靠随机采样恰好命中边界。
np.testing.assert_allclose(forward_kinematics(0.0, 0.0), [1.6, 0.0], atol=1e-12)
np.testing.assert_allclose(forward_kinematics(0.0, np.pi), [0.4, 0.0], atol=1e-12)
print("samples:", num_samples, "observed radius range:", radii.min(), radii.max())

# 绘制可达空间
plt.figure(figsize=(8, 8))
plt.plot(positions[:, 0], positions[:, 1], 'b.', markersize=1)
plt.title('Monte Carlo Simulation of Robot Workspace')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.axis('equal')
plt.grid(True)
plt.show()
```


## 为什么不能直接取点云凸包

本例的解析集合是内半径 0.4 m、外半径 1.6 m 的圆环，面积为 $\pi(1.6^2-0.4^2)$。点云凸包会填满中间不可达圆孔，因而不能作为可达性的判据。

比较覆盖率时，应固定体素或栅格分辨率，再改变样本数和 seed。分辨率、样本量、关节限制与碰撞筛选条件应一并记录；只报告“点云看起来很密”无法复现实验。

## 点云更密，为什么反而可能更接近奇异位形

对上述二连杆，末端到基座的距离满足：

$$
r^2=L_1^2+L_2^2+2L_1L_2\cos q_2.
$$

当 $q_2$ 在 $[0,2\pi)$ 均匀分布时，$r$ 的分布既不均匀，也不是圆环面积的均匀分布。对 $|L_1-L_2|\le r\le L_1+L_2$，其累积分布为：

$$
F_{\mathrm{joint}}(r)=1-\frac1\pi\arccos\left(\frac{r^2-L_1^2-L_2^2}{2L_1L_2}\right).
$$

相反，如果在解析圆环内按面积均匀采样，则：

$$
F_{\mathrm{area}}(r)=\frac{r^2-r_{\min}^2}{r_{\max}^2-r_{\min}^2},\qquad
r=\sqrt{U(r_{\min}^2,r_{\max}^2)}.
$$

直接均匀抽取半径也不对，因为外侧同宽圆环的面积更大。这里的面积采样只适用于已知解析集合的这个二连杆案例，不能原样推广到带障碍的六轴机械臂。

<span id="制作案例"></span>

<figure class="article-figure">
  {{< post-image src="assets/workspace-sampling.png" alt="二连杆关节均匀采样在圆环边界附近更密，面积均匀采样密度更均衡；经验径向分布与解析曲线吻合" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">每种分布采样 20,000 个点，seed 为 42。两种点云具有相同解析可达集合，局部密度不同；右图用累积分布核对采样实现。</span></figcaption>
</figure>

在 $r\ge1.5$ m 的外侧薄环里，关节均匀采样的解析占比约 **23.40%**，面积均匀采样仅约 **12.92%**；本次固定种子运行分别得到 **23.38%** 和 **12.81%**。因此“外圈采到的点更多”主要反映采样映射，不能单独用于比较两台机器人在那里工作是否灵巧。

这个模型的位置 Jacobian 还满足：

$$
\det J=L_1L_2\sin q_2.
$$

完全伸直或折叠时，末端落在外边界或内边界，$\det J=0$，恰好是位置速度映射退化的位形。对这个 $2\times2$ Jacobian，$|\det J|$ 等于两个奇异值的乘积；高维或混合平移、旋转任务应先明确任务空间和尺度，再谈可操作度。可结合 [Modern Robotics 的工作空间说明](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-5-task-space-and-workspace/)与[可操作度椭球](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-4-manipulability/)理解集合边界和局部运动能力的区别。

### 复现图与数字

[下载采样与绘图脚本](workspace_sampling.py)，安装 NumPy、Matplotlib 后执行：

```bash
python workspace_sampling.py --output-dir workspace-sampling
```

脚本写出图像与 JSON 结果，检查半径边界，并将经验 CDF 与解析 CDF 比较。当前运行的最大差约为 0.00833。[本次结果文件](assets/workspace-sampling.json)记录种子、样本数和区间占比，可用于核对复现结果。

最终报告至少分开三项：**可达集合的估计边界、指定分辨率下的采样覆盖、选定任务 Jacobian 的局部运动能力**。这三项回答不同问题，不应压缩成一个“点云密度分数”。

## 阅读自测与验收

- 用两连杆平面模型的已知内外半径检查采样结果，区分采样覆盖范围与解析可达集合。
- 改变采样数和 seed，观察边界是否稳定；随机未采到的区域不一定不可达，凸包内部也不一定全部可达。
