---
title: "机器人动力学参数辨识：从数据采集到安全验证"
date: 2026-09-03
lastmod: 2026-10-30
draft: false
authors: ["chase"]
math: true
tags: ["Robotics", "Dynamics", "Parameter Identification", "Pinocchio", "RLS", "EKF"]
categories: ["机器人技术"]
summary: "从数据与可辨识性建立动力学回归，对比 Pinocchio、Drake、RBDL、Robotics Toolbox 与 iDynTree，提供两套离线辨识示例，并通过交互反例说明时间延迟与参数偏差。"
showToc: true
TocOpen: true
comments: false
images: ["assets/dynamics-regression-principle-v2.png"]
description: "从数据与可辨识性建立动力学回归，对比 Pinocchio、Drake、RBDL、Robotics Toolbox 与 iDynTree，提供两套离线辨识示例，并通过交互反例说明时间延迟与参数偏差。"
contentLanguage: "zh-CN"
reading_prerequisites: "刚体动力学、回归与数据处理"
reading_focus: "先完成可回滚的离线辨识闭环，训练残差小不代表参数可以直接上机。"
related_posts:
  - "/posts/robotics/control/impedance-control"
  - "/posts/calibration/kinematics"
---

动力学参数辨识的目标，是从关节位置、速度、加速度和力矩观测中估计质量、质心、惯量及摩擦等参数。它服务于重力补偿、逆动力学、阻抗控制和仿真，但辨识结果首先是“候选参数”，不能未经审查就覆盖真机配置。

辨识完成后，质量、质心、惯量和摩擦参数如何进入重力补偿、操作空间动力学与柔顺交互，可继续阅读[机器人阻抗控制专题]({{< relref "/posts/robotics/control/impedance-control" >}})。

> **安全提示**：求解器正常退出、训练误差很小或回归矩阵满秩，都不代表参数可以直接进入实时控制器。候选结果必须经过独立数据验证、物理一致性检查、人工审批和可回滚的低速真机验证。辨识程序不得在线热更新正在使用的力矩模型。

> 本文只讨论动力学参数辨识。关节零位、连杆几何、关节轴、Base/Tool 外参等属于[运动学参数标定]({{< relref "/posts/calibration/kinematics" >}})，需要使用对应的观测与参数模型。

## 阅读指南

- 先选工具并运行示例：[五方案对比](#dynamics-library-comparison)、[Pinocchio 开箱示例](#pinocchio-quickstart)、[Drake 符号示例](#drake-quickstart)；两套下载示例都不依赖 RobotServer；
- 第一次搭建离线辨识流程：先读第 1、2、3、4、6、7 节；
- 只估计末端工具负载：重点读 6.3.1 和第 7 节；
- 研究摩擦与传动：重点读 2.3、3.1 和 6.3.2；
- 需要在线更新：重点读 4.4–4.7，并先完成离线基准；
- 准备部署：直接检查第 7、11 节；
- 日志只有位置和力矩：使用[预处理入口](#raw-log-preprocessing)和[窗口对照](#preprocess-window-study)，核对时间、导数与关节侧力矩；
- 想直观看时间偏差：操作[延迟反例计算器](#timing-lab)，比较训练拟合和两种验证力通道；
- 已有控制工程，需要接入数据和审核流程：阅读第 8、9 节。

动力学参数辨识内部还可以继续细分：

| 子问题 | 主要参数 | 最低数据需求 | 常用方法 |
| --- | --- | --- | --- |
| 静态重力/负载辨识 | 工具质量、质心、重力项、bias | 多个静止姿态的 $q,\tau$ | 静态 LS/WLS |
| 完整刚体动力学辨识 | 质量、质心一阶矩、惯量 | 动态 $q,\dot q,\ddot q,\tau$ | 基参数 + WLS/TLS/IV |
| 物理一致辨识 | 同上，但保证惯量可实现 | 动态轨迹与物理边界 | SDP、Cholesky/NLS |
| 摩擦辨识 | Coulomb、viscous、Stribeck | 正反向低速/匀速数据 | LS、鲁棒或非线性拟合 |
| 传动辨识 | 转子惯量、效率、减速比误差、摩擦 | 电机侧与关节侧同步数据 | 残差拟合、联合优化 |
| 柔性系统辨识 | 刚度、阻尼、回差、模态 | 力矩、双编码器或振动数据 | 时域/频域 system ID |
| 在线自适应辨识 | 工具负载、摩擦变化、bias 漂移 | 实时状态和力矩 | FF-RLS、EKF/UKF、MHE |
{.table-readable}

如果目标是改善刚体逆动力学和力矩前馈，主线通常是“静态重力/负载 → 摩擦 → 完整刚体参数”。柔性与传动模型需要额外传感器和不同激励，应作为后续独立阶段，而不是无条件塞入同一个回归矩阵。

### 第一次实践的最小可行路线

如果暂时不需要在线自适应、柔性模型和浮动基辨识，可以先完成下面这个最小闭环：

1. 固定 URDF、工具、减速比、力矩来源和关节顺序；
2. 录制一组训练多谐波轨迹和一组不同系数的验证轨迹；
3. 保留 `raw.csv`，离线生成带 $q,\dot q,\ddot q,\tau$ 的 `processed.csv`；
4. 用 Pinocchio 生成 $Y$，通过 QR/SVD 检查秩和条件数；
5. 使用 WLS 或 Huber 得到 candidate；
6. 检查独立验证误差、参数不确定性和质量矩阵正定性；
7. 保存 review 与 rollback，先仿真再低速上机。

只有这七步全部打通后，再考虑 FF-RLS、Dual EKF、MHE 或更复杂的摩擦/柔性模型。这样可以避免同时调试数据、模型、求解器和实时更新逻辑。

## 1. 动力学回归模型

### 1.1 动力学参数辨识原理

先看信号、回归矩阵和参数之间的关系：

<figure class="article-figure" id="fig-identification-principle">
  {{< post-image src="assets/dynamics-regression-principle-v2.png" alt="训练状态生成回归矩阵，与训练力矩共同拟合候选参数；同一参数进入留出回归矩阵，预测力矩与独立观测计算残差" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">训练矩阵与训练力矩共同进入求解器，得到的同一候选参数用于独立验证。惯性回归坐标使用质量、一阶矩和原点惯量；质心坐标本身不作为独立线性参数。摩擦与 bias 可增加对应回归列，留出数据不参与拟合。</span></figcaption>
</figure>

对固定基刚体系统，逆动力学可写成关于动力学参数线性的形式：

$$\boldsymbol{\tau}=Y(\boldsymbol q,\dot{\boldsymbol q},\ddot{\boldsymbol q})\boldsymbol\theta+\boldsymbol\epsilon$$

其中 $Y$ 是由运动状态计算得到的回归矩阵；$\boldsymbol\theta$ 包含质量、质心一阶矩、转动惯量，也可以追加黏性摩擦、库仑摩擦和力矩偏置；$\boldsymbol\epsilon$ 是测量噪声与未建模动力学。采集 $N$ 个样本后，将各个 $Y_i$ 和 $\tau_i$ 叠加即可用最小二乘求解：

$$
\underbrace{\begin{bmatrix}\tau_1\\\tau_2\\\vdots\\\tau_N\end{bmatrix}}_{\bar\tau} = \underbrace{\begin{bmatrix}Y_1\\Y_2\\\vdots\\Y_N\end{bmatrix}}_{\bar Y}
\theta+\bar\epsilon
$$

在所选参数空间中，若 $\bar Y$ 满列秩、$W$ 正定，WLS 的形式解为：

$$\hat{\theta}_{WLS}=(\bar Y^TW\bar Y)^{-1}\bar Y^TW\bar\tau$$

实际实现不应显式求逆，应使用 QR、SVD 或稳定的线性求解器。完整惯性参数往往存在冗余，秩亏时不能使用这个逆矩阵表达式，应先确定基参数或用 SVD 明确保留子空间。带 ridge 的目标为：

$$\hat\theta_{ridge}=\arg\min_\theta\left(\|W^{1/2}(\bar Y\theta-\bar\tau)\|_2^2+\lambda_r\|L\theta\|_2^2\right)$$

辨识的核心就是寻找一组 $\hat\theta$，使预测力矩 $\hat\tau=Y\hat\theta$ 尽可能接近实测力矩。`RobotServer` 的 `RegressorBackend` 将“如何生成 $Y_i$”与通用数据处理、求解器解耦；`robot_identification_pinocchio` 使用 Pinocchio 刚体动力学回归器生成参数列。

### 符号、参考侧与单位

| 量 | 推荐单位 | 说明 |
| --- | --- | --- |
| $q$ | rad | 转动关节统一使用弧度；移动关节使用 m |
| $\dot q$ | rad/s | 必须与模型关节方向一致 |
| $\ddot q$ | rad/s² | 记录求导/滤波方法 |
| $\tau$ | N·m | 统一换算到模型定义的关节侧 |
| $m$ | kg | 质量必须为正 |
| $mc_x,mc_y,mc_z$ | kg·m | 质心一阶矩，不是质心坐标本身 |
| $I_{xx},\ldots,I_{zz}$ | kg·m² | 明确是在质心还是连杆原点表达 |
| $f_v$ | N·m·s/rad | 黏性摩擦系数 |
| $f_c$ | N·m | 库仑摩擦幅值 |
| $J_m$ | kg·m² | 明确是电机侧惯量还是已反射到关节侧 |
{.table-readable}

数据、回归器和最终配置必须使用同一参考侧。电机侧惯量反射到理想关节侧时通常与减速比平方有关；若模型后端已经包含传动映射，就不能再次换算。对移动关节，力矩和相关摩擦单位应相应改为 N，而不是 N·m。

对每个刚体，常见的 10 维惯性参数由质量 $m$、质心一阶矩 $m c_x,m c_y,m c_z$ 和六个对称惯量分量组成。注意辨识通常直接得到一阶矩，而不是单独的质心；需要在 $m>0$ 时通过 $c_x=(m c_x)/m$ 等关系恢复质心坐标。摩擦和驱动惯量属于额外扩展列，不应与刚体的 10 维参数混为一谈。

### 完整参数不等于可辨识参数

<figure class="article-figure" id="fig-base-parameter-spaces">
  {{< post-image src="assets/base-parameter-spaces.webp" alt="完整回归矩阵乘完整参数，与基回归矩阵乘参数组合，得到相同维度的预测力矩；基参数由完整参数线性映射得到" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text">两种参数空间表达相同的力矩预测。M 是堆叠方程的行数，P 是完整参数数目，r 是保留的独立组合数；基参数通常是多个物理参数的组合。结构性依赖与有限数据激励不足造成的秩下降仍需区分。</span></figcaption>
</figure>

将每个连杆的 10 个惯性参数全部放入 $\theta$，并不意味着它们都能被独立估计。若某个参数方向在所有允许运动下都不改变关节力矩，它就是结构不可辨识方向；去除这些方向后得到的最小独立组合称为**结构基参数**。工程上可以用符号化简，或对丰富状态取样后做 QR/SVD 来构造数值候选基，再在其他轨迹上核对。

单组有限数据的 SVD 还可能因为缺乏激励而少保留一些方向，因此“当前数据的数值保留子空间”与“结构基参数”需要区分。QR/SVD 是秩分析工具，采样到一个低秩矩阵本身并不证明所有允许运动都具有相同依赖关系。

对堆叠回归矩阵进行奇异值分解：

$$Y=U\Sigma V^T$$

大于数值阈值的奇异值数量给出有效秩 $r$；接近零的奇异值对应无法由当前数据区分的参数方向。阈值必须结合矩阵缩放、噪声和奇异值谱设置，不能机械使用一个对所有机器人通用的常数。

因此需要区分：

- **完整参数**：URDF 中每个刚体的质量、质心和惯量描述；
- **结构基参数**：在模型的允许运动中对关节力矩具有独立作用的最小参数组合；
- **可观测程度**：即使属于基参数，不同方向也可能因激励不足而具有很大不确定性。

`rank < parameter_count` 时，不应把完整参数的普通最小二乘解直接解释成真实物理值。可以在明确映射的基参数空间求解，或引入可信 CAD 先验和物理一致性约束；无论哪一种，都要保存参数空间转换关系。

从传统关节动力学角度，同一个模型也可以写为：

$$\tau=M(q)\ddot q+C(q,\dot q)\dot q+g(q)+\tau_f(\dot q)+\tau_b$$

<figure class="article-figure" id="fig-torque-decomposition">
  {{< post-image src="assets/dynamics-torque-decomposition-v2.png" alt="惯性力矩、速度相关项、重力、摩擦与偏置分别进入同一个加和节点，输出各关节力矩向量" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text">v 是关节速度，a 是关节加速度。各项按符号相加，可以增强或抵消；相对大小取决于运动、姿态、负载与摩擦。图中列出固定基回归模型的分解项，外部接触力需另行建模。</span></figcaption>
</figure>

其中 $M$ 是质量矩阵，$C\dot q$ 是科氏力和离心力，$g$ 是重力项，$\tau_f$ 是摩擦，$\tau_b$ 是力矩 bias。$M$、$C$ 和 $g$ 看起来是非线性函数，但对质量、质心一阶矩和惯量参数仍然保持线性，因此可以整理成 $Y\theta$。

### 参数结果与实际用途

| 辨识结果 | 通常写入 | 实际作用 | 使用前检查 |
| --- | --- | --- | --- |
| 连杆质量、质心、惯量 | URDF 或动力学模型 | 重力补偿、RNEA、质量矩阵、仿真 | 质量为正、惯量物理一致、坐标系正确 |
| 工具质量、质心、惯量 | tool profile | 更换末端工具后的补偿 | 与当前工具 ID 和安装方向绑定 |
| 黏性/库仑摩擦 | controller profile | 低速跟踪和力矩前馈 | 正反向残差、温度依赖、死区 |
| 力矩 bias | 驱动或控制器标定文件 | 消除传感器/电流零偏 | 静止复测，防止掩盖外力 |
| 驱动反射惯量 | transmission/actuator profile | 加速段前馈 | 减速比和电机侧/关节侧单位 |
{.table-readable}

不要把所有结果直接覆盖到 URDF：刚体参数、工具参数和控制器经验参数应分别进入自己的配置层，并保持共同的版本号和审核记录。

### 1.2 一页式全流程

先只看主线，不必先理解公式。图 4 概览从 `0` 到 `9` 的十阶段流程，展示验证通过、人工批准和启用后异常回滚的路径；其他返工分支见后面的表格。

<figure class="article-figure" id="fig-identification-workflow" style="max-width: 40rem; margin-inline: auto;">
  {{< post-image src="assets/identification-workflow-v3.png" alt="从冻结模型、设计激励、预检、采集、预处理、回归、求解、验证、评审到分阶段启用，异常时回滚到已批准快照" >}}
  <figcaption><span class="article-figure__number">图 4</span><span class="article-figure__text">先保存不可变原始记录，再生成处理数据与候选结果。通过独立误差和物理检查后进入评审，批准后才分阶段启用；启用异常时恢复先前批准的参数快照。图中省略的返工路径由下表说明。</span></figcaption>
</figure>

其中 `raw.csv` 永远不修改，`processed.csv` 可以重新生成，`result.json` 永远先标记为 candidate。可以把一次辨识看成下面十个阶段。只有当前阶段的检查通过，才进入下一步：

| 阶段 | 要做的事 | 输入 | 必须留下的结果 | 不通过时 |
| --- | --- | --- | --- | --- |
| 0. 冻结版本 | 固定 URDF、减速比、工具和控制周期 | 机器人 profile | `session.json`、版本哈希 | 版本变化后重新开始 |
| 1. 设计激励 | 生成多谐波轨迹，优化信息量 | 关节限制、模型、轨迹系数 | `excitation.yaml` | 调整频率、幅值或相位 |
| 2. 安全预检 | 检查限位、速度、jerk、碰撞和驱动能力 | 候选轨迹、机器人 profile | 预检报告 | 降低幅值或重设计 |
| 3. 录制原始数据 | 执行已批准轨迹，记录状态与力矩 | $t,q,\tau$ 及上下文 | 不可变 `raw.csv` | 隔离受影响区间，严重异常重录 |
| 4. 清洗求导 | 对齐信号、剔除异常、得到 $\dot q,\ddot q$ | `raw.csv` | `processed.csv`、排除区间 | 回查时钟和滤波设置 |
| 5. 构造回归 | 由模型计算每个样本的 $Y_i$ | processed 数据、URDF | 矩阵摘要、参数名列表 | 查关节顺序和坐标系 |
| 6. 数值求解 | WLS/ridge/Huber 求 $\hat\theta$ | $Y,\tau$ | `result.json`（candidate） | rank/条件数不合格则返工 |
| 7. 离线验证 | holdout、残差、物理一致性 | 独立数据和工作空间网格 | 验证报告 | 重新激励或修正模型 |
| 8. 人工评审 | 检查量纲、来源、风险和回滚 | candidate + 报告 | `review.json`、`rollback.json` | 拒绝并保留审查记录 |
| 9. 分阶段导入 | 仿真→低速位置模式→力矩/阻抗 | 已批准版本 | 发布版本和运行日志 | 立即恢复 rollback |
{.table-readable}

这张表中的“结果”不是临时终端输出，而是后续阶段的输入。这样即使换机器、换工具或更换力矩传感器，也能定位差异来自哪一阶段。

### 每个阶段只回答一个问题

0. **版本冻结**：我现在辨识的到底是哪台机器人、哪套工具和哪版模型？
1. **轨迹设计**：所有待估参数是否都被充分激励？
2. **安全预检**：这条轨迹在位置、速度、加速度和 jerk 限制内吗？
3. **数据录制**：传感器是否在同一时间轴上提供了可信的 $q$ 和 $\tau$？
4. **预处理**：求导和滤波是否引入了可接受的噪声与延迟？
5. **构造回归**：关节顺序、坐标系与 $Y$ 的参数列是否完全对应？
6. **数值求解**：从 $Y\theta\approx\tau$ 得到的参数是否数值可辨识？
7. **独立验证**：参数能否预测另一段未参与拟合的数据？
8. **人工评审**：参数的单位、符号、物理范围和风险是否有人负责确认？
9. **实际使用**：新模型是否能在可观测、可停止、可回滚的条件下改善控制？

## 2. 先定义可靠的数据契约

处理后的 CSV 表头为：

```text
time_s,q/<joint>...,dq/<joint>...,ddq/<joint>...,tau/<joint>...
```

关节名称必须非空、唯一且顺序稳定；时间戳必须有限并严格递增；所有状态和力矩值必须有限。原始 CSV 应保持不可变，并记录力矩来源（驱动电流估算、关节传感器或外部传感器）、URDF/模型摘要、软件版本和被排除的接触、饱和、丢帧区间。

ROS 2 只负责采集和会话编排：录制器读取具名 `JointState.position/effort`，按稳定关节名重排，并使用进程单调时钟打时间戳；它不发送运动命令。ROS 2 在这里是可选的中间件，不是辨识算法的依赖；如果系统使用 EtherCAT、串口、TCP、厂商 SDK 或控制器日志，只要导出相同的数据契约，就可以直接进入后续离线流程。这样算法库可以在没有 ROS graph 的环境下测试。

### 2.1 录制后立即检查

在机器人离开实验状态前，先检查采样间隔 $\Delta t$ 的最小值、最大值、中位数和抖动，确认没有时间回退；统计缺失、重复、非有限和饱和样本；画出每个关节的 $q$ 与 $\tau$ 时序，并核对运动方向和力矩方向。若发现关节错序、单位错误、持续饱和或大段丢帧，应重新录制，而不是留到求解阶段修补。

### 2.2 时间对齐与延迟

位置和力矩即使拥有相同消息时间戳，也可能经过不同的驱动、滤波和通信链路。先固定延迟的正方向：令 $\tau_{true}(t)$ 表示时刻 $t$ 的真实力矩，记录通道相对状态滞后 $\Delta>0$，则记录值为：

$$
\begin{aligned}
\tau_{log}(t)&=\tau_{true}(t-\Delta)+\epsilon(t),\\
\tau_{true}(t)&=Y(q(t),\dot q(t),\ddot q(t))\theta.
\end{aligned}
$$

直接把同一个时间戳下的记录力矩与状态配对，通常会产生系统误差。在这个约定下，理想的离线校正是将 $\tau_{log}(t+\Delta)$ 与时刻 $t$ 的状态配对，而不是把已滞后的力矩再向后取一次；同时裁去没有共同覆盖的边界，并记录插值方法。固定延迟模型也无法代表任意变化的抖动。

一个无噪声反例是单轴移动系统 $F=m\ddot q+b\dot q$。取 $m=2$ kg、$b=0.4$ N·s/m、$q=0.03\sin(2\pi\cdot0.7t)$ m，仅让力记录滞后 20 ms。以 100 Hz 取 400 点，不校正延迟的最小二乘会得到约 $\hat m=1.98428$ kg、$\hat b=1.17123$ N·s/m，训练残差仍在浮点舍入量级。因为单一频率下的相移仍可由正弦和余弦组合表示，错误参数能够吸收延迟。

令训练角频率为 $\omega$、相移为 $\delta=\omega\Delta$，展开滞后力中的正弦与余弦，就得到本例的等效系数：

$$
\begin{aligned}
\hat m&=m\cos\delta-\frac{b}{\omega}\sin\delta,\\
\hat b&=b\cos\delta+m\omega\sin\delta.
\end{aligned}
$$

换成 1.1 Hz、同样按 100 Hz 取 300 点的另一条轨迹，沿用这组参数，相对**同样滞后 20 ms 的记录力**，预测 RMSE 约为 0.166 N；相对**已对齐的无延迟参考力**，约为 0.114 N。这是两个不同的验证目标，不能混写成一个“验证误差”。这个构造反例没有测量噪声，已经足以说明训练拟合不能单独验证时间对齐。

{{< identification-timing id="timing-lab" >}}

页内计算器通过两列 QR 最小二乘求解，不调用动力学库。下载包中的 [timing_counterexample.py](timing_counterexample.py)使用上面的闭式系数独立复算；运行 `python3 -B timing_counterexample.py --check` 即可核对[默认结果](assets/timing-reference.json)。它只依赖 Python 标准库，不是采样日志的插值精度测试，也不代表一般多关节系统都能用这两个系数吸收延迟。

应使用同一单调时钟记录各信号的采集时刻，并保存设备时间戳与接收时间戳。可以通过已知激励下的互相关或小范围延迟扫描估计固定延迟，但估计值必须在独立数据上复核。禁止分别平移训练集和验证集来追求各自最小误差，否则延迟参数会吸收模型错误。

零相位离线滤波不会产生固定相位延迟，但会使用未来样本，因此只能用于离线辨识；实时控制器中的因果滤波器仍有延迟，部署时不能假设两者等价。

### 2.3 力矩从哪里来

回归目标应是与模型关节坐标一致的关节侧力矩。常见来源并不等价：

| 来源 | 优点 | 主要风险 |
| --- | --- | --- |
| 关节力矩传感器 | 直接观测关节侧力矩 | 零偏、温漂、带宽和交叉耦合 |
| 电机电流估算 | 易获取、采样频率高 | 力矩常数、减速比、效率和电流环偏差 |
| 应变片/六维力传感器映射 | 可观察外力或末端载荷 | 坐标变换、雅可比奇异和接触力混入 |
| 控制器指令力矩 | 无需额外传感器 | 不等于实际输出，可能忽略饱和和驱动动态 |

若由电机电流估算，理想换算可写成：

$$\tau_j=s_j\,\eta_j\,N_j\,K_{t,j}(i_j-i_{0,j})$$

其中 $s_j$ 是方向符号，$\eta_j$ 是传动效率，$N_j$ 是减速比，$K_{t,j}$ 是力矩常数，$i_{0,j}$ 是电流零偏。必须明确减速比定义是电机侧/关节侧还是相反，并确认驱动器报告的是相电流、$q$ 轴电流还是已经换算的力矩。

不要把“指令力矩”默认当作“测量力矩”。如果只能获取指令值，应在结果中明确 `torque_source`，并把电流环动态、限幅和力矩变化率限制当作模型误差来源。

## 3. 激励轨迹与预处理

单一正弦通常无法同时激励所有参数。可使用多谐波轨迹：

$$q_j(t)=q_{0j}+\sum_{k=1}^{K}(a_{jk}\sin k\omega t+b_{jk}\cos k\omega t)$$

设计阶段的参考速度、加速度与 jerk 可以由解析导数得到，用于检查候选轨迹。构造辨识回归矩阵时，需要的是**实际发生的运动**；不能因为参考轨迹可解析求导，就把指令导数直接当成实测导数。跟踪误差、限幅和驱动延迟都会让两者不同。

执行前应检查各关节位置、速度、加速度和 jerk 的全程限制。有限时间网格可能漏掉采样点之间的峰值；可结合导数极值、保守包络或自适应加密检查，明确剩余误差。关于离散网格与连续峰值的区别，可对照 [TOPPRA 算例]({{< relref "/posts/trajectory/toppra" >}})。轨迹限位检查之外，仍需考虑碰撞与执行系统状态。

轨迹设计同时面对两个不同目标：

- **安全可执行**：整个周期满足位置、速度、加速度、jerk、碰撞距离和力矩限制；
- **信息充分**：堆叠回归矩阵的有效秩足够，较小奇异值不过度接近零，条件数可接受。

仅满足限位并不代表轨迹适合辨识。所有关节使用相同频率、相同相位和相近幅值时，回归矩阵可能高度相关。可为不同关节配置不同的正余弦系数和谐波组合，再通过 SVD、条件数或 Fisher 信息矩阵指标筛选候选轨迹。轨迹优化不能绕过碰撞与驱动能力检查。

训练轨迹与验证轨迹应在姿态、速度或谐波组合上存在明确差异。例如训练集使用一组多谐波系数，验证集使用另一组较低幅值系数；两者都覆盖目标工作空间，但验证数据不参与轨迹参数优化、滤波参数选择和延迟调参。

对真实数据，保留原始记录，另存预处理结果。对称窗口与非均匀时间三点差分是一种离线基线；窗口长度、边界处理和求导顺序需要验证。也可以使用样条、局部多项式或状态平滑器得到运动估计，不能把某一种差分方法当作所有传感器的默认答案。

如果对回归方程滤波，应对 $Y$ 的各列和 $\tau$ 使用一致的线性处理、时间边界与有效样本区间；参数恒定时才有 $F(Y\theta)=(FY)\theta$。先平滑 $q$ 再计算非线性回归矩阵，通常不等于直接平滑已计算的 $Y$。只平滑某一侧、只对齐曲线峰值，可能改变原来的动力学关系。降噪还会引入时间相关性，下面的不确定性分析必须考虑这一点。

### 3.1 摩擦参数需要单独关注

最简单的关节摩擦模型为：

$$\tau_f=f_v\dot q+f_c\operatorname{sign}(\dot q)$$

其中 $f_v$ 是黏性摩擦系数，$f_c$ 是库仑摩擦系数。零速附近的静摩擦、Stribeck 效应、齿槽转矩和减速器回差并不能被这个模型完整描述；直接使用 `sign(0)` 还会让零速样本产生歧义。

采集摩擦数据时，应分别包含稳定的正向和反向速度平台，并记录关节/驱动温度。换向瞬间和极低速爬行区可以单独标记，不与稳态摩擦样本混用。若摩擦残差随温度或速度明显变化，可以建立分段模型，但不能为了降低训练误差无限增加参数。

恒速段的总力矩仍可能包含重力、负载和科氏／离心项。摩擦回归应使用扣除可信已知项后的残差，或联合建模这些动力学项；正反速度样本尽量覆盖相同姿态，避免把姿态变化误当作摩擦变化。

工程上通常先用低加速度恒速段估计摩擦和 bias，再用多关节动态轨迹估计惯性参数，最后联合求解检查参数耦合。这样更容易区分“惯量估计错误”和“摩擦模型不完整”。

## 4. 求解与诊断

除了普通最小二乘，还可以使用加权最小二乘、ridge 或 Huber IRLS：

```cpp
auto dataset = PreprocessPositionAndTorque(names, time, q, tau, options);
auto problem = BuildRegressionProblem(dataset, backend);
SolverOptions solver;
solver.ridge = 1e-10;              // 例示正则强度；需结合参数尺度验证偏差
auto result = Solve(problem, solver);
```

`RobotServer` 示例中的两关节后端声明参数名和数量，构造回归问题后输出 `parameters`、`rmse`、`rank` 和 `condition_number`。真实项目至少检查：

1. 在所声明的基参数空间中检查 `rank == parameter_count`。若在完整参数空间引入先验或约束，应分别报告数据秩和先验贡献，不能靠 ridge 后矩阵可逆就宣称参数已被数据辨识。
2. 训练集与独立 holdout 集的 RMSE、逐关节峰值误差。
3. 条件数、协方差或等价不确定性指标。
4. Huber 权重是否大量下降；大量异常点通常意味着接触、通信或力矩映射问题。
5. 在目标工作空间采样验证质量矩阵对称正定，并检查质量、质心和摩擦参数的量纲与物理范围。

### 4.1 物理一致性不是只检查质量为正

候选刚体参数至少满足：

- 质量 $m>0$，并与 CAD、称重或部件规格处于合理范围；
- 质心位于连杆几何附近，且使用了正确的连杆坐标系；
- 质心处惯量张量对称正定；
- 主惯量满足三角不等式，例如 $I_x\le I_y+I_z$；
- 对目标工作空间中的采样姿态，质量矩阵 $M(q)$ 对称正定。

惯量张量可写为：

$$
I_C=\begin{bmatrix}
I_{xx}&I_{xy}&I_{xz}\\
I_{xy}&I_{yy}&I_{yz}\\
I_{xz}&I_{yz}&I_{zz}
\end{bmatrix}
$$

还要确认辨识参数对应的是连杆坐标原点处惯量还是质心处惯量；两者需要通过平行轴定理转换。坐标系或参考点混淆，可能得到数值残差很小但无法正确写入 URDF 的参数。

最后，分别在静态姿态、低速轨迹和动态轨迹上检查重力项、摩擦项和惯性项。若只有联合总力矩拟合良好，而单项出现不合理的大数并相互抵消，说明参数缺乏物理可解释性。

### 4.2 如何阅读不确定性和残差

先假定回归矩阵可视为确定、在所选参数空间满列秩，误差零均值，且没有 ridge 或额外约束。若误差协方差的模型是 $\Sigma=\sigma^2W^{-1}$，用残差估计整体噪声尺度后，常见近似为：

$$\operatorname{Cov}(\hat\theta)\approx\hat\sigma^2(\bar Y^TW\bar Y)^{-1}$$

其中 $W$ 是**相对精度矩阵**；如果已经直接取 $W=\Sigma^{-1}$、包含完整的绝对噪声尺度，就不应再重复乘同一个 $\sigma^2$。对角项反映单个参数的不确定性，非对角项反映参数之间的相关性。

若仍使用给定的 $W$ 求点估计，而真实误差协方差为一般的 $\Sigma$，应从线性估计器推导：

$$
A=(\bar Y^TW\bar Y)^{-1}\bar Y^TW,\qquad
\operatorname{Cov}(\hat\theta)=A\Sigma A^T.
$$

这同时包含不同样本噪声大小和样本之间的相关性。广义最小二乘使用完整误差协方差进行白化，可参考 [GLS 的协方差与白化接口](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLS.html)。若 $Y$ 也含噪声、残差与回归量相关，或者已经加入正则化、投影、物理约束，上述简单公式不再直接给出完整的不确定性结论。

Huber 损失在小残差区保持二次项，在大残差区转为线性项：

$$
\rho_\delta(r)=
\begin{cases}
\frac{1}{2}r^2,&|r|\le\delta\\
\delta\left(|r|-\frac{1}{2}\delta\right),&|r|>\delta
\end{cases}
$$

这会降低少量异常点的影响，但 $\delta$ 应依据力矩噪声尺度选择，并报告降权样本比例。

建议为每个参数同时报告：估计值、标准差、相对标准差以及主要相关参数。若参数接近零，相对标准差会失去解释意义，此时应同时观察绝对标准差和对力矩预测的影响。

残差 $r=\tau-Y\hat\theta$ 至少按以下方式绘制：

- 残差随时间：发现接触、丢帧、升温和模式切换；
- 残差随关节位置：发现重力项或坐标系错误；
- 残差随速度：发现摩擦模型不完整；
- 残差随加速度：发现惯量或时间对齐错误；
- 残差直方图与关节间相关性：发现系统性偏差和耦合遗漏。

较低的训练 RMSE 只能说明模型解释了训练力矩，不能证明每个参数都准确。最终判断应结合独立验证误差、参数不确定性、物理约束和实际控制表现。

#### 列缩放后的条件数不代表激励强度 {#scaled-condition-excitation}

以第 2.2 节的单轴系统为例，固定频率和采样时刻，只把位移幅值乘以 $s>0$，速度与加速度也乘以 $s$。回归矩阵整体缩小，但逐列归一化后的矩阵不变：

$$
Y_s=sY,\qquad \widetilde Y_s=\widetilde Y.
$$

所以数据秩与归一化后的条件数都可以完全不变。在回归量确定、满列秩且观测噪声协方差相同的假设下，参数估计协方差却放大为原来的 $s^{-2}$ 倍。**幅值缩小 100 倍，参数标准差可增大 100 倍。** 这不是一般机器人回归器随关节幅值变化的统一缩放定律，只是这个线性单轴系统的明确反例。

因此，还应保留原始列范数、噪声尺度和参数不确定性。列范数带有各自的单位，不能直接按不同列的数值大小排列“信息量”。第 10 节的两套示例会保存原始列范数，零列记为 0；求解时为避免除零使用的缩放值 1 与它分开记录。

#### 相关噪声的反例：同样 200 行数据，区间覆盖率可能差很多 {#correlated-uncertainty}

考虑单轴移动系统 $F=m\ddot q+b\dot q$，真实参数为 $m=2$ kg、$b=0.4$ N·s/m。给定精确运动 $q(t)=0.03\sin(2\pi\cdot0.7t)$ m，以 100 Hz 取 200 点。只对力观测加入零均值、稳态标准差 0.05 N 的高斯 AR(1) 噪声，相邻相关系数为 0.9，即 $\Sigma_{ij}=0.05^2\,0.9^{|i-j|}$。

同一条轨迹重复生成 2,500 组独立噪声记录，每组都做**相同的普通最小二乘**。只改变不确定性的计算方式，不更换点估计器：

| 参数 | 忽略时间相关性的标准差 | 使用完整 $\Sigma$ 的标准差 | 2,500 次估计的经验标准差 |
| --- | ---: | ---: | ---: |
| 质量／kg | 0.00841 | 0.03373 | 0.03357 |
| 阻尼／N·s/m | 0.03896 | 0.15123 | 0.14917 |
{.table-readable}

<figure class="article-figure" id="fig-correlated-uncertainty">
  {{< post-image src="assets/correlated-uncertainty.png" alt="质量与阻尼的重复估计直方图吻合考虑相关噪声的宽分布，错误独立假设给出的分布明显过窄" >}}
  <figcaption><span class="article-figure__number">图 5</span><span class="article-figure__text">直方图来自 2,500 次独立噪声记录；蓝线使用完整协方差，橙色虚线错误地忽略同一记录内的相关性。曲线描述重复实验中的估计量分布。这里使用无约束 OLS，个别阻尼估计可能为负；加入物理约束后应重新分析其分布。</span></figcaption>
</figure>

用“估计值 ±1.96×标准差”构造名义 95% 区间，错误独立假设下，两项实际覆盖率只有 **37.84%／39.64%**；使用已知完整协方差后为 **95.20%／95.12%**。这里故意给两种方法相同且正确的边际噪声方差，差异来自遗漏时间相关性。各参数的区间是分别计算的，不是两个参数同时覆盖的联合 95% 区域。

下载 [correlated_uncertainty.py](correlated_uncertainty.py)，只依赖 NumPy：

```bash
python -B correlated_uncertainty.py --output correlated-uncertainty.json
```

需要重新绘图时，安装 Matplotlib 并在同一命令后加 `--figure correlated-uncertainty.png`。

[结果文件](assets/correlated-uncertainty.json)记录了固定种子、协方差、版本和覆盖率。脚本通过 AR(1) 递推产生噪声，再与 $A\Sigma A^T$ 的解析结果对照。真实实验通常不知道完整 $\Sigma$，还需要估计残差相关结构、比较不同采集批次，并验证区间是否可信；不能把这组理想覆盖率当作实机标定精度。滤波如何改变样本相关性，可继续对照[机器人滤波专题]({{< relref "/posts/robotics/control/robot-filters" >}})。

### 4.3 更有效的辨识方法

普通最小二乘可以作为基线，但真实机器人通常需要组合方法：

| 方法 | 主要解决的问题 | 使用建议 |
| --- | --- | --- |
| 基参数辨识 | 完整参数列线性相关、解不唯一 | 先用 QR/SVD 找独立参数组合，再求解 |
| 最优激励轨迹 | 回归矩阵病态、参数方差大 | 优化最小奇异值、条件数或 Fisher 信息量 |
| 加权最小二乘（WLS） | 不同关节或样本噪声不同 | 权重来自传感器噪声估计，不能凭拟合效果随意调 |
| Huber/Tukey 鲁棒回归 | 少量无法完全剔除的异常点 | 仍需先剔除接触、饱和和通信故障区间 |
| 总最小二乘（TLS） | $Y$ 和 $\tau$ 都含测量误差 | 需要匹配误差尺度与结构；普通 TLS 不保证适合非线性构造的回归矩阵 |
| 工具变量（IV） | 回归量噪声与残差相关，OLS 有偏 | 需要与噪声不相关、又与运动相关的有效工具变量 |
| 物理约束优化 | 出现负质量或非物理惯量 | 对质量、惯量和质量矩阵施加一致性约束 |
| MAP/贝叶斯方法 | 数据不足但有可信 CAD 先验 | 输出后验不确定性，并明确先验对结果的影响 |
{.table-readable}

工程上推荐的高性价比路线是：

1. 用 CAD/称重结果初始化并确定合理物理范围；
2. 通过 QR/SVD 得到基参数或至少识别不可观测方向；
3. 优化多谐波激励，使 $Y$ 的较小奇异值增大；
4. 先清洗数据，再用 WLS 或 Huber 求候选解；
5. 若回归量噪声造成明显偏差，再考虑 TLS 或 IV；
6. 用物理一致性约束恢复可部署的完整参数；
7. 使用完全独立的轨迹验证，并与旧模型做逐关节对比。

没有一种高级求解器能够补救错误的力矩换算、时间不同步或缺乏激励。提升效果时，通常应先改善数据和轨迹，再升级求解器。

### 4.4 递推最小二乘与遗忘因子（FF-RLS）

<figure class="article-figure" id="fig-offline-online-workflow">
  {{< post-image src="assets/offline-online-workflow.webp" alt="离线先设计激励再采集、回归和独立验证；验证结果初始化在线候选估计，异常数据或激励不足时保持估计" >}}
  <figcaption><span class="article-figure__number">图 6</span><span class="article-figure__text">离线建立基准，在线跟踪选定参数并保存候选历史。图中把参数估计与控制器启用分开；RLS、EKF 的噪声尺度、初值和约束仍需要各自配置。</span></figcaption>
</figure>

当数据持续到达，或工具负载、摩擦、力矩 bias 随时间缓慢变化时，可以使用递推最小二乘。对单个时刻的回归 $\tau_k=Y_k\theta+\epsilon_k$，带遗忘因子的更新为：

$$K_k=P_{k-1}Y_k^T\left(\lambda I+Y_kP_{k-1}Y_k^T\right)^{-1}$$

$$\hat\theta_k=\hat\theta_{k-1}+K_k\left(\tau_k-Y_k\hat\theta_{k-1}\right)$$

$$P_k=\frac{1}{\lambda}\left(I-K_kY_k\right)P_{k-1}$$

其中 $P$ 是递推的逆加权信息矩阵，不能无条件解释成真实参数协方差；$K$ 是增益，$0<\lambda\le1$ 是遗忘因子。上式使用单位观测权重：各行噪声不同，需要先白化 $Y_k$ 与 $\tau_k$，或在增益分母中用 $\lambda R_k$ 替换 $\lambda I$ 并让 $P$ 的初值与该噪声约定一致。仅给矩阵起名 `covariance`，不会自动满足这些假设。

$\lambda=1$ 对应不遗忘的 RLS，仍包含初始化所代表的先验；$\lambda<1$ 会逐渐降低旧数据权重，能够跟踪时变参数，但也会放大噪声并增加漂移风险。有效记忆长度可粗略理解为 $1/(1-\lambda)$ 个更新块，实际仍要结合采样率、块大小、激励和噪声验证。投影、冻结与遗忘还会改变矩阵的统计解释，不能只看 $P$ 变小就声称估计更准确。

| 方法 | 适合场景 | 主要风险 |
| --- | --- | --- |
| RLS（$\lambda=1$） | 固定参数、数据逐步到达、避免反复批量求解 | 不可观测方向的信息不增长，信息矩阵的数值条件可能恶化 |
| FF-RLS（$\lambda<1$） | 温度相关摩擦、工具负载或 bias 缓慢变化 | 噪声放大、参数漂移、错误吸收接触力 |
| 变遗忘因子 RLS | 工况突变时快速适应、稳定时降低波动 | 切换逻辑和阈值需要独立验证 |
| 投影/约束 RLS | 需要保持质量、摩擦或负载参数在物理范围 | 约束空间必须与参数化一致 |
{.table-readable}

FF-RLS 的实践建议：

1. 用离线批量辨识结果初始化 $\hat\theta_0$；初始化 $P_0$ 前，核对其与观测白化、参数单位和先验定义的尺度是否一致；
2. 只在线更新少量明确会变化的参数，如工具质量、摩擦或 bias；刚体惯量默认冻结；
3. 只有在回归矩阵满足持续激励条件时更新，否则保持参数不变；
4. 接触、碰撞、饱和、急停和控制模式切换时暂停更新；
5. 对参数范围、单次更新量和变化率加约束，并持续记录创新项 $\tau_k-Y_k\hat\theta_{k-1}$；
6. 在线估计结果仍作为 candidate，不能直接无审查写入实时力矩控制模型。

对于“拿起未知物体后估计负载”，可以只递推工具质量和质心相关参数；对于“整机所有连杆参数”，仍优先使用离线最优激励和批量物理约束辨识。这样既利用 FF-RLS 的适应能力，又避免控制过程中的参数漂移破坏稳定性。

### 4.5 联合参数—状态估计：Augmented EKF 与 Dual EKF

RLS 通常假设回归量 $q,\dot q,\ddot q$ 已经可用。当速度、加速度不可直接可靠测量，或需要同时估计传感器 bias、外部扰动和缓慢变化参数时，可以把动力学写成状态空间模型：

$$x_{k+1}=f(x_k,u_k,\theta_k)+w_k,\qquad y_k=h(x_k,\theta_k)+v_k$$

$$\theta_{k+1}=\theta_k+w_{\theta,k}$$

其中 $x$ 可以包含 $q,\dot q$，$\theta$ 包含待估动力学参数，$w,v$ 分别为过程噪声和测量噪声。

**联合 EKF（Joint/Augmented EKF）**将状态和参数拼成增广状态 $z=[x^T,\theta^T]^T$，用一个滤波器同时更新。它形式统一并能保留状态—参数交叉协方差，但维度较大、线性化和协方差计算成本高，也更容易因尺度差异和不可观测性造成数值问题。

**双 EKF（Dual EKF，DEKF）**使用两个相互耦合的滤波器：状态滤波器在当前参数下估计 $x$，参数滤波器再用当前状态更新 $\theta$。它更容易为状态和参数设置不同的噪声模型、更新频率与约束，但两个滤波器互相依赖，调参不当会产生滞后或相互强化的估计偏差。

| 方法 | 优势 | 更适合 | 主要风险 |
| --- | --- | --- | --- |
| FF-RLS | 实现简单、计算量小、线性参数更新直接 | $q,\dot q,\ddot q$ 可信，少量参数缓慢变化 | 回归量噪声导致偏差 |
| 增广 EKF | 联合保留状态—参数相关性 | 状态与参数强耦合、系统维度可控 | 增广维度大、线性化误差、协方差病态 |
| Dual EKF | 状态/参数可分频率和噪声建模 | 状态快速、参数缓慢的多时间尺度系统 | 双滤波器耦合、调参复杂 |
{.table-readable}

使用 EKF 类方法时需要特别注意：

1. 先做局部可观测性或灵敏度分析；滤波器收敛不等于真实参数可辨识；
2. 参数过程噪声 $Q_\theta$ 决定参数变化速度，过大导致漂移，过小则无法跟踪；
3. 状态、质量、惯量和摩擦参数量纲差异很大，应进行尺度归一化；
4. 使用 Joseph 形式或平方根滤波等数值稳定实现，监控协方差正定性和创新一致性；
5. 接触力未建模时，滤波器可能把外力吸收到质量、摩擦或 bias 中；应显式估计扰动或停止参数更新；
6. 对参数采用投影、重参数化或约束更新，避免产生负质量和非物理惯量。

实用策略仍是“离线辨识完整模型，在线只估计少量变化量”：先用批量方法确定刚体基参数，再由 FF-RLS、Dual EKF 或增广 EKF 跟踪工具负载、摩擦、零偏或外部扰动。在线估计应有冻结条件、范围限制和回滚版本。

### 4.6 方法选择指南

可以按下面的顺序选择，而不是先决定使用哪一种“高级算法”：

```text
数据能否离线完整获得？
├─ 是 → 状态 q、dq、ddq 是否可信？
│       ├─ 是 → 基参数 + WLS/Huber + 物理约束
│       └─ 否 → TLS/IV，或先进行状态平滑
└─ 否，需要在线更新 → 状态是否可信？
        ├─ 是 → RLS / FF-RLS / 约束 RLS
        └─ 否 → 参数与状态是否强耦合？
                ├─ 维度较小 → 增广 EKF/UKF
                ├─ 多时间尺度 → Dual EKF
                └─ 强约束、算力充足 → MHE
```

| 需求 | 首选方法 | 原因 |
| --- | --- | --- |
| 辨识整机刚体参数 | 批量基参数 + 物理约束 | 能使用全数据、全局诊断和独立验证 |
| 在线估计未知工具质量 | 约束 FF-RLS 或 Dual EKF | 参数少、变化慢，可设置冻结条件 |
| 编码器速度噪声较大 | 状态平滑 + 批量法，或 EKF | 避免直接使用噪声导数 |
| 模型非线性明显且雅可比困难 | UKF | 无需求解析雅可比，但计算量更大 |
| 需要同时满足边界和滑动窗口约束 | MHE | 可显式加入状态/参数约束和鲁棒代价 |
| 存在未知接触扰动 | 扰动观测器 + 冻结参数更新 | 避免把外力错误辨识成模型参数 |

MHE（移动窗估计）在最近一段数据窗口上反复求解带约束优化问题，能够显式处理参数范围、输入饱和和鲁棒损失；计算成本取决于状态维度、窗口、约束与求解器，需要按目标周期测量。UKF 可以减轻 EKF 的一阶线性化误差，但同样不能解决不可观测性。无论采用哪种在线方法，都应与离线基准结果并行比较，并保留冻结和回滚机制。

### 4.7 FF-RLS 更新示例

下面代码展示单个采样块的核心更新，`Y` 的行数等于本次观测的关节数：

```cpp
struct FfRls {
    Eigen::VectorXd theta;
    Eigen::MatrixXd covariance;
    double forgetting_factor{0.999};

    void Update(const Eigen::MatrixXd& Y,
                const Eigen::VectorXd& measured_torque,
                bool excitation_ok,
                bool contact_or_fault) {
        if (!excitation_ok || contact_or_fault) return;  // 冻结更新

        const Eigen::MatrixXd innovation_cov =
            forgetting_factor * Eigen::MatrixXd::Identity(Y.rows(), Y.rows()) +
            Y * covariance * Y.transpose();
        const Eigen::MatrixXd gain =
            covariance * Y.transpose() * innovation_cov.ldlt().solve(
                Eigen::MatrixXd::Identity(Y.rows(), Y.rows()));

        const Eigen::VectorXd innovation = measured_torque - Y * theta;
        theta += gain * innovation;
        covariance = (Eigen::MatrixXd::Identity(theta.size(), theta.size()) -
                      gain * Y) * covariance / forgetting_factor;

        ProjectToAllowedRange(theta);       // 参数范围约束
        SymmetrizeAndFloor(covariance);     // 数值稳定保护
        RecordInnovation(innovation);       // 持续诊断
    }
};
```

这是算法骨架，不是可直接进入实时控制器的完整实现。生产实现还应避免控制周期内动态分配，预先分配矩阵，检查 LDLT 分解状态，限制单步参数变化，并将参数更新与控制输出解耦。若创新持续超限，应冻结估计器并报告数据/模型异常，而不是继续增大协方差让滤波器追随故障信号。

## 5. 推荐实施顺序与安全边界

推荐的迁移顺序是：工具质量/质心和力矩 bias → 单关节黏性与库仑摩擦 → Pinocchio 刚体回归器 → 可观测性优化和独立轨迹 → 可选物理一致性约束。

求解器输出应保存为 candidate artifact，绑定原始 CSV 的 SHA-256、处理参数、模型摘要和代码版本。人工 review 另存独立文件，批准条件包括在声明的可辨识参数空间达到预期秩、数值条件可接受、质量矩阵检查通过和完整回滚清单。任何 URDF、传动、固件、工具、采样频率或力矩来源变化都要求重新评审。

这一章定义“什么时候可以继续”，第 7 章再说明参数具体写入哪些配置。真机验证从位置模式、低速度和低增益开始，逐步检查重力补偿、力矩模式、导纳与阻抗控制。辨识模块绝不在线热更新控制模型；发现异常时恢复旧参数，而不是继续尝试提高控制增益。

## 6. 一次完整辨识实验的执行清单

下面给出一套可以直接转成实验 SOP 的流程。每一步都应留下输入、输出和操作者签名，避免“跑出一组数字”却无法复现。

### 6.1 实验前准备

先冻结机器人版本：记录 URDF、关节顺序、减速比、编码器方向、额定电流/力矩常数、控制周期、末端工具质量与质心。确认急停、限位、碰撞检测和 watchdog 可用，并在仿真或吊装状态下检查轨迹包络。辨识通常分为无负载和带工具两次实验；若要估计工具参数，工具必须固定且在整个数据集期间不更换。

### 6.2 需要采集什么

原始阶段至少记录时间、位置、力矩与实验上下文。速度和加速度可以来自可信估计器，也可以在离线处理时由实测位置补充；最终回归数据与对应记录应覆盖下列字段：

| 数据 | 用途 | 注意事项 |
| --- | --- | --- |
| 单调时间戳 $t$ | 求导、对齐和频率检查 | 不能用消息序号代替；检查丢帧和回退 |
| 关节位置 $q$ | 运动状态和重力项 | 弧度、统一零位和方向 |
| 关节速度 $\dot q$ | 科氏力和黏性摩擦 | 优先使用高质量估计值，或由平滑位置求导 |
| 关节加速度 $\ddot q$ | 惯性项 | 速度求导一次或位置求导两次；先控制噪声放大 |
| 关节力矩 $\tau$ | 回归目标 | 标明电流换算、传感器或外部估计来源 |
| 控制模式/使能状态 | 样本筛选 | 排除切换、饱和和故障区间 |
| 温度、电压（可选） | 解释摩擦和电机常数变化 | 用于分段建模或复现实验条件 |

位置、速度、加速度、力矩必须同一关节顺序、同一时间基准。记录原始信号，不要在 recorder 内覆盖滤波后的值。对于电流换算力矩，还要保存电流—力矩比例、减速比、效率假设以及电流零偏。

### 6.3 执行激励并划分数据

轨迹通常包含几秒的静止段（估计重力和 bias）、多谐波往返段（激励惯量与耦合项）以及不同速度的小幅段（摩擦）。让各关节在允许工作区间内覆盖不同位置和正、反向速度，避开机械限位、线缆拉紧和明显碰撞。重复运行至少两遍：一遍用于训练，另一遍作为 holdout 验证，不能随机打散时间序列后“伪造”独立集。

#### 6.3.1 入门实验：静态工具负载辨识

如果当前目标只是识别末端工具质量与质心，不必一开始就执行高速多谐波轨迹。机器人在静止状态下满足近似关系：

$$\tau(q)=g_{robot}(q)+Y_{tool}(q)\theta_{tool}+\tau_b$$

先固定同一个工具，在覆盖不同重力方向的多个安全姿态静止采样。每个姿态等待振动衰减后，取一段力矩均值和方差；方差异常的姿态不进入正常样本。若本体模型可信，也可以用“带工具力矩减去无工具力矩”的差分削弱本体模型误差，但两组实验必须使用相同姿态、温度和控制状态。

最小执行步骤为：

1. 记录无工具或已知工具的基准数据；
2. 安装待辨识工具，记录工具 ID、安装变换和大致质量范围；
3. 选择使工具重力在不同关节上产生明显力矩的多个姿态；
4. 每个姿态采集 $q,\tau$、温度和稳定状态标记；
5. 求解工具质量、三个质心一阶矩及必要的关节 bias；
6. 用未参与求解的静态姿态验证力矩预测误差；
7. 将结果写入 tool profile，而不是修改机器人本体连杆参数。

输出至少包含工具质量、法兰坐标系下质心、参数不确定性、训练/验证逐关节误差及适用的工具安装变换。若工具姿态覆盖不足，质量可能可估而质心某些方向不可观测，应明确保留为未确认状态。

#### 6.3.2 推荐的分阶段实验矩阵

不要直接用一条复杂轨迹同时估计所有参数。更容易诊断的安排如下：

| 实验 | 轨迹特征 | 主要信号 | 目标参数 | 关键验收 |
| --- | --- | --- | --- | --- |
| A. 静态多姿态 | 每个姿态充分静止 | $q,\tau$、温度 | 重力、工具质量/质心、bias | 未见姿态上的静态残差 |
| B. 正反向恒速 | 多个稳定速度平台，低加速度 | $q,\dot q,\tau$ | 黏性/库仑摩擦 | 正反向残差、温度重复性 |
| C. 单关节扫频 | 其他关节保持安全姿态 | $q,\dot q,\ddot q,\tau$ | 驱动惯量、局部摩擦/柔性线索 | 频率响应和重复试验一致 |
| D. 多关节多谐波 | 关节联合运动、充分耦合 | 全部状态、力矩和上下文 | 刚体基参数 | rank、条件数和 holdout 指标 |
| E. 独立验证 | 不同系数、姿态与速度组合 | 与 D 相同 | 不重新拟合 | 新模型相对旧模型稳定改善 |
| F. 在线跟踪 | 正常任务中的安全片段 | 实时状态、力矩、接触标记 | 工具/摩擦/bias 的慢变化 | 有界更新、可冻结和可回滚 |
{.table-readable}

每完成一阶段，都应固定已通过验证的结果，再决定下一阶段允许更新哪些参数。例如实验 B 不应同时放开所有连杆惯量；实验 F 默认不应更新机器人本体惯量。若后一阶段导致前一阶段指标明显恶化，应检查参数耦合和模型结构，而不是只接受新的总体 RMSE。

### 6.4 清洗、求解和产物

先标记时间戳异常、丢帧、使能切换、力矩饱和、接触和急停恢复区间，按连续有效区间分段处理。平滑与求导窗口应留在同一有效段内，避免跨越被剔除的模式切换或接触区间。对实际测量位置做离线平滑，再通过拟合曲线求导、非均匀差分或状态平滑器估计速度与加速度。随后由机器人模型生成 $Y$，按样本堆叠成 $\mathbf Y\boldsymbol\theta=\boldsymbol\tau$，求解并输出以下结果：

- 参数向量及稳定的参数名（质量、质心、一阶/二阶惯量、黏性摩擦、库仑摩擦、bias）；
- `rank`、奇异值、条件数和 ridge/Huber 配置；
- 训练集与 holdout 的总体 RMSE、逐关节 RMSE/峰值误差和残差曲线；
- 协方差、参数标准差或置信区间，以及被 Huber 降权的样本比例；
- 质量矩阵正定性、待估刚体质量为正、质心范围和摩擦符号检查；
- 原始数据 SHA-256、处理配置、模型摘要、软件 commit 和排除区间。

这些内容共同构成 candidate artifact；只有参数数组而没有元数据的结果不可审查、不可回滚。

### 6.5 一个两关节示例

假设采集到 `axis_a`、`axis_b` 两个关节，模型后端声明四个参数：`axis_a` 的位置项和黏性摩擦项，以及 `axis_b` 的惯量项和 bias。示例程序用合成轨迹生成 240 个样本，先预处理位置，再根据已知参数合成力矩，最后重新预处理并求解。理想输出类似：

```text
parameters=2.000000 0.350000 1.400000 -0.200000
rmse=0.000000 rank=4 condition=...
```

这里的四个数字能否“用到机器人上”，还要看 `rank=4`、独立轨迹误差、单位和参数物理意义。示例验证的是接口链路，不代表任何真实机械臂的质量或摩擦参数；真实数据必须替换合成力矩，并完成上一节列出的全部闸门。

### 6.6 如何判断结果是否可信

在选定的可辨识子空间达到预期秩，只是必要条件。还要确认 holdout 误差与训练误差接近，逐关节残差没有随位置、速度或温度呈明显趋势，参数不确定性没有异常放大。将辨识模型与原模型分别计算重力力矩和质量矩阵，在目标姿态网格上比较；若某些姿态误差突然变大，通常是激励不足、关节顺序错误、工具质量错误或力矩来源不匹配，而不是继续调大正则化参数。

## 7. 结果如何落到实际控制系统

参数导入应是显式、可回滚的发布流程：

1. 在离线工具中把 candidate 转换为机器人 profile/URDF 所需格式，保留单位转换和参数名映射表。
2. 保存当前线上参数快照，并把新文件、review、数据摘要和回滚清单绑定在同一版本中。
3. 先在仿真中验证重力补偿、逆动力学前馈和质量矩阵；再在无负载、位置模式、低速低增益下上机。
4. 逐步开启重力补偿、力矩前馈、导纳/阻抗控制，限制最大力矩和力矩变化率，实时监视跟踪误差、电流、温度和 watchdog。
5. 任何异常立即切回旧参数；通过分阶段验证和稳定运行后，再将已审核候选版本设为默认配置，保留原始 candidate 与审批记录。

辨识结果并不一定全部写入一个文件：刚体惯量通常进入 URDF 或动力学模型，摩擦和力矩 bias 进入控制器 profile，工具参数进入 tool profile。不同来源、不同温度或不同负载下的参数可以保留为带条件的版本，但不能静默覆盖默认模型。

| 辨识结果 | 推荐落点 | 实际用途 | 导入前必须检查 |
| --- | --- | --- | --- |
| 连杆质量、质心、惯量 | URDF `<inertial>` 或模型参数文件 | $M(q)$、$C(q,\dot q)$、$g(q)$ | 单位、坐标系、惯量正定 |
| 工具质量与质心 | tool profile | 工具重力补偿、负载模型 | 工具 ID、安装方向和法兰坐标系 |
| 黏性/库仑摩擦 | controller profile | 低速补偿和逆动力学前馈 | 正负速度残差、死区和温度影响 |
| 力矩 bias | 标定/驱动 profile | 传感器或电流估算零偏补偿 | 静止姿态、温漂和上电重复性 |
| 驱动反射惯量 | actuator/transmission profile | 关节等效惯性模型 | 减速比定义、是否已包含在刚体模型中 |
{.table-readable}

### 导入前后应比较什么

不要只比较总体 RMSE。至少对旧模型和新模型分别记录：

- 静止姿态下的重力补偿残差；
- 相同轨迹下的逐关节 RMS 和峰值力矩预测误差；
- 跟踪误差、最大电流、力矩变化率和温升；
- 不同位置、速度方向及工具负载下的残差分布；
- 质量矩阵最小特征值和数值条件。

只有新模型在独立数据和分阶段真机验证中稳定优于旧模型，并且没有引入新的安全风险，才应批准替换。

### 配置落地示例

刚体参数通常转换为 URDF 的 `<inertial>`：

```xml
<inertial>
  <origin xyz="0.012 -0.004 0.083" rpy="0 0 0"/>
  <mass value="2.430"/>
  <inertia ixx="0.018" ixy="0.0002" ixz="-0.0004"
           iyy="0.021" iyz="0.0001" izz="0.009"/>
</inertial>
```

工具负载和控制器参数更适合放在独立 profile 中：

```yaml
tool_dynamics:
  tool_id: inspection_camera_v2
  mass_kg: 1.27
  center_of_mass_m: [0.018, -0.006, 0.094]

joint_dynamics:
  joint_1:
    viscous_friction_nm_s_per_rad: 0.25
    coulomb_friction_nm: 0.41
    torque_bias_nm: -0.06
```

这些数值只用于展示格式，不是可直接使用的机器人参数。转换工具应按稳定参数名写入目标字段，检查 kg、m、rad、N·m 等单位，并保存生成前后的 diff。不要通过正则表达式修改正在运行的 YAML；应生成新的版本化文件，经过审核后再切换 profile。

## 8. 推荐的实验目录与命令

建议每次实验使用独立目录，避免后处理覆盖原始记录：

```text
session-2026-09-03-arm-tool/
├── raw.csv                 # 只读原始记录
├── processed.csv           # 滤波、求导后的数据
├── session.json            # 关节顺序、力矩来源、模型摘要
├── excitation.yaml         # 轨迹系数与限位检查结果
├── result.json             # status=candidate 的参数结果
├── review.json             # 人工审核结论
└── rollback.json           # 导入前的旧参数快照
```

本节命令依赖包含 `robot_identification` 组件的 `RobotServer` 工程，不是安装 Pinocchio 后就会自动出现的标准命令。没有该工程时，可使用第 10.3、10.4 节的下载包；只希望复算本文统计例子时，可运行上面的独立 NumPy 脚本。在对应工程根目录中，可以运行独立示例验证工具链：

这个独立目标需要 C++17、CMake 3.15+ 和 Eigen 3.3+；没有要求先启动 ROS 或连接机器人。

```bash
./scripts/run_identification_example.sh
```

也可以直接构建 `robot_identification`，运行测试和合成数据示例：

```bash
cmake -S robot_identification -B /tmp/robot-identification \
  -DBUILD_TESTING=ON -DROBOT_IDENTIFICATION_BUILD_EXAMPLES=ON
cmake --build /tmp/robot-identification --parallel 2
ctest --test-dir /tmp/robot-identification --output-on-failure
/tmp/robot-identification/generic_regressor_example
```

这个示例只验证 API、预处理、矩阵组装和诊断流程，不能当作任何真实机器的参数。

### session 与 candidate 示例

`session.json` 描述“这批数据从哪里来”：

```json
{
  "schema_version": 1,
  "kind": "robot_identification_raw_joint_state",
  "session_id": "arm-no-tool-001",
  "time_source": "process_monotonic",
  "joint_names": ["joint_1", "joint_2"],
  "sample_count": 12000,
  "rejected_count": 38,
  "torque_source": "calibrated_joint_torque",
  "dataset_sha256": "<raw.csv SHA-256>",
  "model_digest": "<URDF/model digest>"
}
```

`result.json` 描述“用哪批数据、哪个模型得到了什么候选结果”。这个格式例子只估计一个关节的黏性摩擦，其余动力学作为已知项扣除；若从 12000 个原始时刻中排除 38 个，每个有效时刻参与两条关节标量方程，则共有 23924 行：

```json
{
  "schema_version": 1,
  "kind": "robot_identification_result",
  "status": "candidate",
  "dataset_sha256": "<same raw.csv SHA-256>",
  "model_digest": "<same model digest>",
  "software_revision": "<git commit>",
  "joint_names": ["joint_1", "joint_2"],
  "parameter_names": ["actuator/joint_1/viscous_friction"],
  "parameters": [0.25],
  "diagnostics": {
    "row_count": 23924,
    "rank": 1,
    "condition_number": 1.0,
    "train_rmse": 0.1,
    "validation_rmse": 0.2,
    "max_absolute_error": 0.5,
    "mass_matrix_passed": true
  }
}
```

`dataset_sha256` 和 `model_digest` 必须在两份文件中一致。`status` 保持 `candidate`；人工批准写入独立的 review artifact，而不是修改求解结果来伪装审批状态。

本例只有一个非零回归列，所以其二范数条件数为 1；不能把示例中的单列诊断当作整机惯性模型的条件数。`mass_matrix_passed` 在这里针对固定的名义刚体模型，不代表已经辨识了新的惯量。实际产物还需关联独立验证会话和数据哈希，不能只留下一个验证 RMSE 数字。

## 9. 排查与运行示例

### 9.1 常见失败模式

| 现象 | 优先排查 |
| --- | --- |
| rank 小于参数数目 | 激励轨迹过于单一、关节未运动、参数化存在冗余 |
| 训练误差小而 holdout 很大 | 过拟合、训练/验证工况不同、时间泄漏 |
| 所有关节残差有固定偏置 | 力矩零偏、重力方向或坐标系错误 |
| 高速段误差随速度增大 | 减速比/电机常数错误，或未建模摩擦与时延 |
| 质量矩阵非正定 | 惯量参数不物理、单位错误或模型坐标系不一致 |
| Huber 大量降权 | 接触、丢帧、饱和或力矩传感器异常 |

排查时应先修复数据和模型，再重新采集；不要用更强 ridge、删除异常参数或手动修改结果来“改善”指标。

### 9.2 可运行示例

`RobotServer` 中与本文流程对应的组件如下：

| 组件 | 对应阶段 | 职责 |
| --- | --- | --- |
| `robot_identification` | 预处理、求解、诊断、验证 | 中间件无关核心，支持任意自由度和可插拔回归器 |
| `robot_identification_pinocchio` | 生成回归矩阵 $Y$ | 使用 Pinocchio 固定基刚体回归器生成参数列 |
| `robot_identification_ros2` | 原始数据录制、会话编排 | 可选 ROS 2 中间件层，不发送机器人运动命令 |
| `robot_server_tools` | 数据集离线检查 | 检查数据契约与会话元数据，不修改运行配置 |
| `schema/` | candidate、review、rollback | 约束可追溯产物和审批边界 |

实际移植到非 ROS 2 系统时，只需替换数据录制入口；核心预处理、回归求解、诊断和 artifact 结构仍可保留。

#### 9.2.1 示例 A：先用合成数据跑通整条链路

在 `RobotServer` 根目录执行：

```bash
./scripts/run_identification_example.sh
```

该命令会构建并运行 `generic_regressor_example`。本次核对的示例先生成两关节正弦位置，**用预处理函数得到位置、速度与加速度，再用这些量合成力矩**，随后组装回归矩阵并求解；它并非用独立的实测运动或解析导数验证求导器。终端会打印参数、RMSE、秩和条件数。这样适合确认编译环境、数据接口与回归链路，但由于合成和拟合共用运动处理，接近零的残差不能证明预处理对真实编码器噪声同样准确。

#### 9.2.2 示例 B：检查一份处理后 CSV

处理后 CSV 的最小样例（两关节、两行）如下：

```csv
time_s,q/axis_a,q/axis_b,dq/axis_a,dq/axis_b,ddq/axis_a,ddq/axis_b,tau/axis_a,tau/axis_b
0.000,0.10,-0.20,0.00,0.10,0.20,0.00,1.23,-0.40
0.010,0.10,-0.199,0.02,0.10,0.19,0.01,1.25,-0.39
```

检查重点是：时间严格递增；`q/dq/ddq/tau` 的关节集合完全相同；没有空值、`NaN` 或 `Inf`；列顺序与模型后端声明一致。真实项目可先用 `robot_server_tools` 的离线检查工具验证 CSV，再进入回归求解。

#### 9.2.3 示例 C：真实实验的伪代码骨架

```cpp
// 1) 从 recorder 读取 raw.csv，并明确稳定关节顺序
auto raw = LoadRawCsv("raw.csv", {"joint_1", "joint_2", "joint_3"});

// 2) 剔除切换、饱和、接触和丢帧区间，并分成连续段
auto segments = ExcludeAndSplitIntervals(raw, excluded_intervals);

// 3) 每段独立平滑、求导与裁边，之后才拼接回归样本
std::vector<ProcessedDataset> processed_segments;
for (const auto& segment : segments) {
    processed_segments.push_back(PreprocessPositionAndTorque(
        segment.names, segment.time, segment.position, segment.torque,
        preprocessing_options));
}
auto dataset = ConcatenateRegressionRows(processed_segments);

// 4) PinocchioRegressor 或自定义后端生成 Y
PinocchioRegressor backend(model);
auto problem = BuildRegressionProblem(dataset, backend);

// 5) 求解后先保存 candidate，不写运行配置
SolverOptions options;
options.ridge = 1e-10;
options.huber_delta = 1.5;
auto candidate = Solve(problem, options);
SaveCandidate("result.json", candidate, sha256("raw.csv"));
```

伪代码中的函数名用于说明数据流；具体版本以 `robot_identification` 头文件和安装包 API 为准。关键约束是第 5 步只生成 candidate，审核和低速验证完成前不得自动覆盖控制器参数。

## 10. 动力学辨识工具与开源项目

### 10.1 五种动力学库怎么选 {#dynamics-library-comparison}

先区分两层：**动力学库负责根据模型和状态计算方程、回归器或导数；辨识流程还需要数据同步、激励设计、求解、可辨识性分析和验证。** 读入 URDF 并得到逆动力学，不等于已经得到一套可交付的参数辨识系统。

| 方案 | 动力学与模型入口 | 辨识时最有用的能力 | 适合的起点与仍需补齐的部分 |
| --- | --- | --- | --- |
| **[Pinocchio](https://github.com/stack-of-tasks/pinocchio)** | C++ 核心、Python 接口；URDF；RNEA、CRBA、ABA | `computeJointTorqueRegressor` 直接生成惯性参数回归矩阵；`computeRNEADerivatives` 提供逆动力学对状态的解析导数 | 已有机械臂控制或优化工程，需要高效生成 $Y$；自行组织数据、求解与物理约束。本文主要推荐这一入口 |
| **[Drake](https://drake.mit.edu/)** | C++/Python；URDF/SDF → `MultibodyPlant`；支持数值、AutoDiff 与符号标量 | 将模型转换为符号系统，提取 lumped parameters；与优化、仿真和控制设计配合 | 希望检查方程结构、联动优化或仿真；需显式选择待估参数、设置符号 Context，并处理不可辨识组合 |
| **[RBDL](https://rbdl.github.io/)** | 以 C++ 刚体算法为核心；RNEA、CRBA、ABA；URDF 通过对应 addon 载入 | 高效的正/逆动力学、约束算法；另有 CasADi 数学后端 | 已经在 C++ 控制器中使用 RBDL，想复用模型；回归器构造、数据和求解流程需按选定构建配置补齐 |
| **[Robotics Toolbox for Python](https://petercorke.github.io/robotics-toolbox-python/arm_dh.html)** | Peter Corke 等维护；DH 等模型；逆动力学、惯量、重力和摩擦计算 | 在 Python 中检查小型模型与动力学项；`rne_python` 支持符号模型参数 | 课程、原型与手算对照；先确认模型确实带有质量和惯量，再扩展参数回归与约束求解 |
| **[iDynTree](https://gbionics.github.io/idyntree/classiDynTree_1_1KinDynComputations.html)** | 多体动力学、URDF 模型；固定基与浮动基状态和空间量 | 惯性参数回归器、浮动基逆动力学，适合与控制和状态估计共用模型 | 人形、浮动基、带基座力/力矩观测的系统；需统一空间速度、参考坐标系、接触力和参数顺序 |
{.table-readable}

Pinocchio 面向高效刚体计算，并提供解析导数，适合放进控制与辨识 pipeline；但这里不做跨库速度排名。一次计算的延时还取决于模型规模、算法、Python/C++ 边界和构建选项，应在自己的模型上测量。尤其不要混淆下面两个接口：

$$
\begin{aligned}
Y(q,v,a)&=\frac{\partial\tau}{\partial\theta},\\
\text{状态导数}&:\quad
\frac{\partial\tau}{\partial q},\quad
\frac{\partial\tau}{\partial v},\quad
\frac{\partial\tau}{\partial a}.
\end{aligned}
$$

前者针对选定的线性惯性参数 $\theta$，用于线性辨识；后者针对状态，用于优化、灵敏度分析和滤波。拥有后者不会自动替你定义前者的参数向量、列顺序和物理约束。

Drake 起源于 MIT CSAIL，目前核心开发由 Toyota Research Institute 主导。URDF 解析默认建立数值模型；要得到含未知质量、质心和惯量的符号方程，还要调用 `ToSymbolic()`，并在符号 Context 中替换相关参数。[MIT 的系统辨识章节](https://underactuated.csail.mit.edu/sysid.html)展示了从 URDF、符号逆动力学到 lumped-parameter 最小二乘的做法。下面的第二个例子沿着这条路线，保留独立验证。

另外两点会直接影响结果：Robotics Toolbox 的动力学方法需要模型提供动力学参数，不能把仅含几何的模型当作已经标定的机器人；iDynTree 的 [`InverseDynamicsInertialParametersRegressor`](https://gbionics.github.io/idyntree/group__iDynTreeModel.html)覆盖基座 wrench 与关节力矩，不能直接忽略其参考系和基座六维方程，只取一个形状相似的矩阵使用。

### 10.2 辨识流程与研究项目

如果需要现成的激励、基参数和求解流程，再看这一层的项目；它们与上面的动力学后端可以组合使用。

| 项目 | 适合用途 | 特点 |
| --- | --- | --- |
| [FloBaRoID](https://github.com/kjyv/FloBaRoID) | 人形/浮动基与固定基研究 | 覆盖激励优化、基参数、摩擦和物理一致性流程 |
| [dynamic_parameter_identification](https://github.com/lbr-stack/dynamic_parameter_identification) | 固定基机械臂实验参考 | 面向质量、质心、惯量和摩擦估计 |
| [Physically-Consistent System Identification](https://github.com/ShahramKhorshidi/system_identification) | 研究物理一致参数化 | 包含 LMI/SDP 与 log-Cholesky 非线性参数化思路 |
| [BIRDy](https://github.com/TUM-ICS/BIRDy) | 比较不同辨识算法 | MATLAB 研究框架，覆盖 LS、TLS、IV、滤波与在线方法 |
| [SymPyBotics](https://github.com/cdsousa/SymPyBotics) | 学习和生成符号回归模型 | 可生成动力学项和带摩擦的回归矩阵 |
| [SYMORO Python 3](https://github.com/xu-yang16/symoro-python3) | 符号模型与基参数研究 | 适合理解 identification model 和最小参数结构 |
{.table-readable}

对当前 `RobotServer`，推荐继续以 Pinocchio 回归器作为模型后端，以 `robot_identification` 承担数据、求解、诊断和 artifact 管理。若后续扩展人形或浮动基数据契约，再重点借鉴 FloBaRoID；若加入物理约束求解器，可参考 SDP 或 log-Cholesky 路线，但保持其为可选后端。

### 10.3 主要示例一：Pinocchio 从 URDF 到留出验证 {#pinocchio-quickstart}

这个入口适合“我有 URDF 和处理后的关节日志，先想把 $Y\theta\approx\tau$ 跑通”。下载 [两个辨识示例及模型](parameter-identification-lab.zip)，解压后进入目录。包内有两连杆与单摆 URDF、两个独立示例、公共 CSV/SVD 工具、参考结果和说明。下面命令按 Linux x86_64、Python 3.12 的独立环境复现；安装名为 `pin`，代码中的导入名为 `pinocchio`。

```bash
python3.12 -m venv .venv-pin
.venv-pin/bin/python -m pip install "pin==4.1.0" "numpy==2.2.6"
.venv-pin/bin/python -B pinocchio_identification.py --output pin-result.json
```

默认实验读取 `two_link_id.urdf` 作为名义模型，保持几何不变，另外构造质量、质心和惯量有偏差的真实模型。它生成 **401 个训练样本和另一条轨迹的 301 个验证样本**；位置、速度和加速度由多谐波轨迹解析给出，用真实模型的 RNEA 生成参考力矩，再分别叠加标准差为 $0.02\,\mathrm{N\cdot m}$ 的独立高斯噪声。求解器只读取名义模型生成的回归矩阵和含噪力矩，构造真值仅用于生成数据与事后核对。没有驱动机器人，也没有检验编码器求导误差、摩擦、接触或通信时延。

核心回归步骤如下，在解压目录中可以直接运行：

```python
import numpy as np
import pinocchio as pin
from identification_common import rmse, scaled_svd_fit
from pinocchio_identification import regressors, synthetic_data

model = pin.buildModelFromUrdf("two_link_id.urdf")
data = model.createData()
_, q, v, a, torque, _ = synthetic_data(model)
Y = np.vstack([
    pin.computeJointTorqueRegressor(model, data, qi, vi, ai).copy()
    for qi, vi, ai in zip(q, v, a)
])
fit = scaled_svd_fit(Y, torque.ravel())
_, qv, vv, av, torque_v, _ = synthetic_data(model, validation=True)
Yv = regressors(model, qv, vv, av)
print("retained rank:", fit["rank"], "/", Y.shape[1])
print("validation RMSE:", rmse(Yv @ fit["parameters"], torque_v.ravel()))
```

`data` 会被后续算法调用复用，因此每个样本的矩阵要 `.copy()`。堆叠顺序为“样本 1 的所有关节、样本 2 的所有关节……”，目标力矩也使用同样的 `ravel()` 顺序。公共 `rmse` 要求预测和观测形状相同；不能让 NumPy 把一维向量与列向量广播成所有样本两两组合。每个关节体惯量的参数顺序由 Pinocchio 的 [`Inertia.toDynamicParameters()` 定义](https://github.com/stack-of-tasks/pinocchio/blob/v4.1.0/include/pinocchio/src/spatial/inertia.hxx#L561)：

$$
\theta_i=\begin{bmatrix}
m & mc_x & mc_y & mc_z & I_{xx}^{O} & I_{xy}^{O} & I_{yy}^{O} & I_{xz}^{O} & I_{yz}^{O} & I_{zz}^{O}
\end{bmatrix}^{T}.
$$

这里的惯量关于模型中对应关节体的坐标原点 $O$；URDF 的 `inertial/origin` 与惯量则描述惯性坐标系及质心惯量。固定连接的连杆可能已合并到同一关节体，不能默认每个 URDF `link` 都对应独立十列。两种惯量表示不能按字段名逐项抄回，质心坐标也不能直接当作一阶矩。

还要区分刚体回归器和完整算法调用的附加项。在本例无外力的约定下，[Pinocchio 4.1.0 的 RNEA](https://raw.githubusercontent.com/stack-of-tasks/pinocchio/v4.1.0/include/pinocchio/src/algorithm/rnea.hxx)另加关节侧 `armature` 惯量项，而 `computeJointTorqueRegressor` 的刚体十参数不含该项：

$$
\tau_{RNEA}=Y\theta+\operatorname{diag}(J_a)\ddot q.
$$

默认构造模型的 `armature` 为零，所以两者一致。若自己的模型设了非零附加惯量，比较前应明确扣除已知项，或为未知项另建回归列并重新分析秩。扣除已知项后，独立验证也应使用相同的目标定义，并保留补偿模型与参数。设置 `model.friction`、`model.damping` 也不表示这版 RNEA 已自动加入相应摩擦力矩。本文适配器只预测刚体惯性块，结果 JSON 会记录模型项、未纳入的力矩项和名义 `armature`；CSV 预处理工具不会自动补偿它们。

本例先按训练列的二范数缩放，再做 SVD，保留相对阈值 $10^{-10}$ 以上的方向。这个缩放用于数值条件调整，不等于按噪声协方差做 WLS。返回向量在**缩放后的参数坐标**中取最小范数，仅用于预测；当前数据得到的保留子空间，也不等于已解析证明的结构基参数。

实现中以稳定的 `hypot` 累加列范数，避免直接平方让很大或很小的有限列溢出或消失；不可表示的范数或参数会明确报错。这个数值处理能保持可计算的缩放关系，不能恢复轨迹没有激励的方向。

结果中的 `training_column_l2_norms` 保留未归一化的列范数，包括真实零列；应结合[激励强度与不确定性](#scaled-condition-excitation)解释，不能只把较好的归一化条件数当作辨识充分的证据。

输出中的 `validation_unseen_scaled_direction_fraction` 衡量验证矩阵的行在训练保留子空间之外所占的相对范数。默认实验接近浮点舍入量级；若验证运动依赖训练尚未约束的方向，不能仅依据较低训练残差接受候选结果。这个几何诊断量不是置信度，仍需结合噪声、独立误差和参数不确定性判断。

| 本地复现项 | 参考结果 |
| --- | --- |
| 堆叠矩阵 | $802\times20$ |
| 保留秩 / 未观测方向数 | 6 / 14 |
| 保留奇异值的条件数，列缩放后 | 约 5.27 |
| 名义 URDF 在验证含噪力矩上的 RMSE | 约 $0.1848\,\mathrm{N\cdot m}$ |
| 验证集对含噪力矩的 RMSE | 约 $0.0202\,\mathrm{N\cdot m}$ |
| 验证集对无噪参考力矩的 RMSE | 约 $0.00260\,\mathrm{N\cdot m}$ |
{style="break-inside: avoid;"}

这些数值来自包内固定模型、轨迹和随机种子，最后几位可能随平台变化，**不是机器人实验精度**。两套示例的结果都会记录后端版本、模型哈希和算法／公共工具哈希；使用自己的 CSV 时另记两份输入哈希，便于核对数据与实现是否一致。

Pinocchio 完整结果同时保存名义 URDF 的验证误差，以及辨识后逐关节 RMSE、平均残差和峰值残差，残差统一为“观测减预测”，避免只看混合后的总分。它还检查 $Y\theta_{true}$ 与构造真值模型 RNEA 的一致性，以及名义模型的解析状态导数与有限差分的一致性。前者验证回归器与模型的约定，后者验证状态导数；二者都不能证明真实机器人模型正确。

<figure class="article-figure" id="fig-identification-held-out" style="max-width: 40rem; margin-inline: auto;">
  {{< post-image src="assets/identification-held-out.png" alt="两连杆构造实验的列缩放奇异值谱保留六个方向；独立验证中，两个关节的力矩 RMSE 从名义模型的约 0.0678 和 0.2524 降至约 0.0191 和 0.0212 牛米" >}}
  <figcaption><span class="article-figure__number">图 7</span><span class="article-figure__text">上图显示训练矩阵的相对奇异值与阈值，不高于显示下限的值用向下三角标在下限处；下图比较同一条留出轨迹上两个关节的含噪力矩误差。名义模型与构造真值具有相同几何、不同惯量，噪声标准差为 0.02 N·m。预测改善没有消除未观测的参数方向。</span></figcaption>
</figure>

#### 换成自己的处理后数据

先用包内数据验证 CSV 通道，再换成真实文件：

```bash
.venv-pin/bin/python -B pinocchio_identification.py --csv-dir example-csv
.venv-pin/bin/python -B pinocchio_identification.py \
  --train example-csv/train.csv --validation example-csv/validation.csv \
  --output csv-replay.json

.venv-pin/bin/python -B pinocchio_identification.py \
  --urdf robot.urdf --train train.csv --validation validation.csv \
  --output candidate-prediction.json
```

CSV 沿用第 2、9 节的 `time_s,q/<joint>,dq/<joint>,ddq/<joint>,tau/<joint>` 格式，每个字段都覆盖所有模型关节。使用 UTF-8 编码，入口也接受带 BOM 的 UTF-8 文件。脚本根据模型关节名取列，检查有限值、完整样本和严格递增的时间；输入字节哈希相同的训练、验证文件会被拒绝。不过，文件不同并不证明统计独立，仍需由采集过程保证轨迹独立。

这个适配器处理固定基、每关节一个有界标量坐标的转动关节模型，力矩统一为 N·m。移动关节需要另外处理力的单位；连续转动关节、球关节和浮动基需要显式的配置表示转换，不能只通过 `nq == nv` 猜测兼容性。脚本不从位置自动生成速度和加速度，不估计摩擦或接触，也不把预测参数向量写回 URDF。真实数据残差明显偏大时，先按第 2、3、6 节检查测量链路和模型项。

入口默认模型根坐标系中的重力为 $(0,0,-9.81)$ m/s²。若底座相对这一坐标约定倾斜，应先换算重力，可用 `--gravity GX GY GZ` 传入；也可以在 URDF 中明确相应的固定安装变换。两种表示要保持一致，不能对同一个安装旋转重复换算。报告中的 `gravity_m_s2` 需要与实际安装一致。

例如，若已确认根坐标系中重力沿 $-X$，可在前面的真实 CSV 命令后增加 `--gravity -9.81 0 0`；这不是所有侧装机器人的通用方向，需要由自己的安装坐标确定。该选项只属于 Pinocchio 固定基适配器，Drake 单摆仍按第 10.4 节的固定几何约定运行。

#### 只有位置和力矩日志时 {#raw-log-preprocessing}

下载包中的 `preprocess_identification.py` 可以把 `time_s,q/<joint>...,tau/<joint>...` 转为上述处理后格式；这个入口始终根据位置重算导数，不使用日志中已有的速度／加速度列。先用两份构造记录跑通入口：位置噪声标准差为 $10^{-5}$ rad，力矩噪声标准差为 0.02 N·m，已知延迟为零。

```bash
.venv-pin/bin/python -B preprocess_identification.py --make-demo raw-demo
for dataset_split in train validation; do
  .venv-pin/bin/python -B preprocess_identification.py \
    --input "raw-demo/${dataset_split}-raw.csv" --output "processed/${dataset_split}.csv" \
    --joints joint1 joint2 --window 21 --degree 5 --max-gap 0.03
done
.venv-pin/bin/python -B pinocchio_identification.py \
  --train processed/train.csv --validation processed/validation.csv \
  --output raw-pipeline.json
```

工具在每个居中窗口内，按**实际时间戳**建立局部多项式的 [NumPy 最小二乘问题](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.lstsq.html)，再由拟合系数计算位置、速度与加速度，窗口内的时间间隔可以不完全一致。`--window 21 --degree 5` 是这组构造信号的设置，需要结合真实运动带宽、噪声和独立数据调整。这里两批数据分别约为 50 Hz 和 42.9 Hz，相同点数对应不同时间跨度，元数据会分别记录。`--max-gap 0.03` 表示拒绝超过 30 ms 的间隔，也是本例的采样约束。

默认裁去两端各 10 个样本，得到 381 个训练样本和 281 个验证样本；本地复现的留出力矩 RMSE 约为 0.0203 N·m。每个输出 CSV 旁保存 `.meta.json`，包含输入／输出哈希、算法哈希、窗口跨度、局部条件数和裁剪数量。使用新的输出路径，原始文件保持不变。

若固定力矩延迟已通过独立校准确定，可增加 `--torque-delay 0.020`。它按照第 2.2 节的符号约定，在时刻 $t+0.020$ 对记录力矩做线性插值，与时刻 $t$ 的状态配对，并裁去无重叠边界；负值表示力矩通道超前。此选项对全部关节使用同一个延迟，通道延迟不同应先分别对齐。工具不估计延迟、不做端点外推，也不负责去接触、角度解缠或自动分割故障区间，应先提供清洗后的连续日志段。

这是局部状态估计基线，不等于对 $Y$ 和 $\tau$ 两侧施加相同滤波；平滑与力矩插值还会改变误差和相关性。应继续按第 3、4 节检查窗口偏差、残差和不确定性，不能把构造数据的结果当作真实传感器的保证。

#### 比较窗口宽度与多项式阶数 {#preprocess-window-study}

用同一组构造记录比较五个固定设置，可以观察噪声与平滑偏差的取舍。在解压目录、已安装 Pinocchio 的环境中运行[窗口对照脚本](preprocessing_study.py)：

```bash
.venv-pin/bin/python -B preprocessing_study.py --summary --output window-study.json
```

所有设置都只拟合**同一组中央 321 个训练时刻**，并在另一条轨迹的**同一组中央 221 个验证时刻**上评价，避免不同裁边数量改变比较范围。原始位置噪声仍为 $10^{-5}$ rad，力矩噪声仍为 0.02 N·m；解析加速度和无噪力矩只用于事后诊断，不参与回归或自动选择设置。

| 点数 / 阶数 | 验证加速度 RMSE / rad·s⁻² | 验证力矩 RMSE / N·m | 对无噪力矩 RMSE / N·m |
| --- | ---: | ---: | ---: |
| 11 / 5 | 0.004767 | 0.021186 | 0.004417 |
| 21 / 3 | 0.036819 | 0.021129 | 0.003998 |
| 21 / 5 | 0.000992 | 0.021152 | 0.003686 |
| 41 / 5 | 0.005606 | 0.021075 | 0.003514 |
| 81 / 5 | 0.071685 | 0.021572 | 0.006658 |
{.table-readable}

三个误差都在共同验证区间聚合两个关节。21 点 / 3 阶的力矩 RMSE 略低于 21 点 / 5 阶，但加速度误差约大 37 倍；81 点 / 5 阶的加速度误差进一步升高。这里的力矩噪声底还掩盖了部分预测差异。**力矩拟合良好不能单独证明求导正确，也不能把某个窗口当作通用默认值。**

脚本还用无噪位置单独复算导数，帮助区分噪声与平滑偏差：11 点 / 5 阶的加速度 RMSE 降至约 0.000026 rad·s⁻²，81 点 / 5 阶仍约为 0.071681 rad·s⁻²。后一种设置即使没有位置噪声也存在明显偏差。这组无噪位置只用于诊断，没有替换参与辨识的含噪位置。

点数还要换算成实际跨度。例如 21 点在训练数据中覆盖 0.4 s，在验证数据中约为 0.467 s；完整[窗口对照记录](assets/preprocessing-study.json)同时保存两种跨度、共同时间范围、随机种子与源文件哈希。这个固定对照没有用验证误差搜索窗口；真实日志应结合传感器带宽、运动频率、导数诊断与独立验证选择预处理方法。

### 10.4 主要示例二：Drake 生成符号方程并估计参数组合 {#drake-quickstart}

这个入口适合“希望看清方程如何依赖质量、质心和惯量，再联动仿真或优化”。使用同一个下载包，在另一个独立环境运行：

```bash
python3.12 -m venv .venv-drake
.venv-drake/bin/python -m pip install "drake==1.58.0" "numpy==2.2.6"
.venv-drake/bin/python -B drake_identification.py --output drake-result.json
```

此处的版本是本文复现环境；其他平台先对照 [Drake 的 wheel 安装与支持配置](https://drake.mit.edu/pip.html)。本例只需要多体、符号和 NumPy 最小二乘功能，不调用商业优化求解器。

`pendulum_id.urdf` 定义一个绕 $+Y$ 轴转动的刚体单摆，基座焊接到世界，$q=0$ 时质心在关节下方，重力为世界坐标的 $-Z$ 方向。令质心距离为 $\ell>0$，绕质心 $Y$ 轴的惯量为 $J$，外加黏性阻尼 $b\dot q$，则需要施加的关节力矩为：

$$
\tau=(J+m\ell^2)\ddot q+b\dot q+mg\ell\sin q.
$$

脚本从数值 URDF 创建 `MultibodyPlant`，调用 `ToSymbolic()`，在符号 Context 中设置未知质量、质心和惯量，再调用逆动力学。实现中以 $k=J/m$ 表示质心单位惯量，并先用平行轴定理移到刚体原点；`SpatialInertia` 构造器的单位惯量参数关于原点，不能直接填质心惯量。重力通过 `CalcForceElementsContribution` 传给 `CalcInverseDynamics`，避免漏掉重力；URDF 阻尼为零，$b\dot q$ 只额外加入一次。

[`DecomposeLumpedParameters`](https://drake.mit.edu/pydrake/pydrake.symbolic.html#pydrake.symbolic.DecomposeLumpedParameters)将方程分离为状态函数、参数组合与已知偏置：

$$
\begin{aligned}
\tau&=W(q,\dot q,\ddot q)\alpha+w_0,\\
W&=\begin{bmatrix}\ddot q & \dot q & g\sin q\end{bmatrix},\\
\alpha&=\begin{bmatrix}J+m\ell^2 & b & m\ell\end{bmatrix}^{T},\qquad w_0=0.
\end{aligned}
$$

Drake 不承诺分解后的列顺序，脚本先核对再排列成上面的顺序。这个 API 也不保证给出最少且独立的组合，仍需检查数据矩阵的秩。在解压目录可用下面的完整小段检查生成结果：

```python
from drake_identification import build_plant, regressors, symbolic_regressor, synthetic_data
from identification_common import rmse, scaled_svd_fit

plant = build_plant()
symbols = symbolic_regressor(plant)
print("generated equation:", symbols["torque"])
print("lumped parameters:", symbols["alpha"])
_, q, v, a, torque, _ = synthetic_data(plant)
W = regressors(symbols, q, v, a)
fit = scaled_svd_fit(W, torque.ravel())
_, qv, vv, av, torque_v, _ = synthetic_data(plant, validation=True)
Wv = regressors(symbols, qv, vv, av)
print("estimated [A, b, h]:", fit["parameters"])
print("validation RMSE:", rmse(Wv @ fit["parameters"], torque_v.ravel()))
```

默认同样使用 401 个训练样本、301 个独立轨迹验证样本和 $0.02\,\mathrm{N\cdot m}$ 的力矩噪声。构造模型的 $m=2$ kg、$\ell=0.5$ m、$J=0.17$ kg·m²、$b=0.08$ N·m·s/rad，参考结果为：

| 估计量 | 构造真值 | 含噪估计，约 |
| --- | --- | --- |
| $A=J+m\ell^2$，kg·m² | 0.67 | 0.67020 |
| $b$，N·m·s/rad | 0.08 | 0.08237 |
| $h=m\ell$，kg·m | 1.00 | 1.00018 |
{style="break-inside: avoid;"}

数据矩阵为 $401\times3$、保留秩为 3；验证含噪力矩 RMSE 约为 $0.0198\,\mathrm{N\cdot m}$，对无噪参考约为 $0.00219\,\mathrm{N\cdot m}$。脚本还比较符号逆动力学、数值 URDF 逆动力学和分解式，误差处于浮点舍入量级。这验证的是该单摆的方程与实现约定。

**得到三个参数组合，仍不能同时唯一恢复未知的 $m,\ell,J,b$。** 例如真值下，$(m,\ell,J)=(2,0.5,0.17)$ 和 $(2.5,0.4,0.27)$ 都给出 $A=0.67,h=1$。如果质心距离 $\ell$ 已由独立测量固定，才可以进一步计算 $m=h/\ell$ 和 $J=A-h\ell$，并继续做物理一致性检查。把 URDF 中尚待辨识的质心距离直接当作已知量，会额外引入一个假设。

若确实已有这项独立测量，可直接运行：

```bash
.venv-drake/bin/python -B drake_identification.py \
  --known-com-distance 0.5 --output pendulum-axis-parameters.json
```

本例得到质量约 $2.00036\,\mathrm{kg}$、质心 $Y$ 轴惯量约 $0.17011\,\mathrm{kg\cdot m^2}$。脚本保留该已知距离、数据秩和必要的正质量／正惯量／非负阻尼检查；若估计为负，会原样报告并标记检查失败。这只恢复所假设单摆模型的参数，完整三维惯量及测量距离的不确定性仍需另外处理。

脚本可用 `--train train.csv --validation validation.csv` 读取同一单摆约定下的处理后数据；它没有通用 `--urdf` 参数。换成多关节机器人，需要重新声明参数化、外力、驱动映射和可辨识组合，不能只替换模型文件名。若目标是通用固定基机械臂的首个离线闭环，优先从上一个 Pinocchio 示例开始。

### 10.5 接入真实项目时保留哪些接口

不论选择哪个库，都建议保留以下边界：模型后端输入已经对齐的 $(q,v,a)$，输出带参数顺序说明的 $Y$ 或预测方程；数据层负责时间、单位和排除区间；求解层负责缩放、噪声权重、基参数与物理约束；验证层只读取独立数据和 candidate。换库时，先用同一 URDF 的无噪逆动力学核对关节顺序、重力、惯量参考点、摩擦和外力，再比较辨识结果。

例如，[Robotics Toolbox 的 `rne`](https://petercorke.github.io/robotics-toolbox-python/arm_dh.html#roboticstoolbox.robot.DHRobot.DHRobot.rne)会计入转子惯量和关节摩擦，而本文 Pinocchio 示例的 $Y$ 只包含刚体惯性参数。把前者的总力矩作为后者的回归目标前，需要先统一这些动力学项，明确哪些已补偿、哪些另加参数列；否则误差差异未必来自动力学算法。

两个示例都保存版本、模型哈希、秩、条件数和验证误差。它们提供可复现的离线起点；真实系统仍应继续完成本文的数据处理、不确定性和物理一致性步骤。

## 11. 上线前最后检查

- {{< task-check >}}原始数据仍然存在且 SHA-256 与 `session.json` 一致。{{< /task-check >}}
- {{< task-check >}}关节顺序、单位、编码器方向和力矩换算已由第二人复核。{{< /task-check >}}
- {{< task-check >}}激励轨迹没有触发限位、碰撞、饱和或 watchdog。{{< /task-check >}}
- {{< task-check >}}选定的可辨识回归子空间达到预期秩，条件数和参数不确定性在可接受范围。{{< /task-check >}}
- {{< task-check >}}holdout 误差、逐关节峰值误差和残差趋势均已记录。{{< /task-check >}}
- {{< task-check >}}质量矩阵在目标工作空间保持正定，质量和质心通过物理范围检查。{{< /task-check >}}
- {{< task-check >}}`review.json` 明确批准范围，`rollback.json` 可以独立恢复旧版本。{{< /task-check >}}
- {{< task-check >}}已完成仿真、低速位置模式和逐步提高负载的验证，并保存运行日志。{{< /task-check >}}

## 12. 常见问题

### 术语速查

| 缩写/术语 | 含义 |
| --- | --- |
| COM | Center of Mass，质心 |
| OLS / LS | 普通最小二乘 / 最小二乘 |
| WLS | 加权最小二乘 |
| TLS | 总最小二乘，同时考虑回归量与观测误差 |
| IV | 工具变量法，用于减小回归量噪声引起的偏差 |
| IRLS | 迭代重加权最小二乘，常用于实现鲁棒损失 |
| RLS / FF-RLS | 递推最小二乘 / 遗忘因子递推最小二乘 |
| EKF / UKF | 扩展卡尔曼滤波 / 无迹卡尔曼滤波 |
| DEKF | Dual EKF，使用状态与参数两个耦合滤波器 |
| MHE | Moving Horizon Estimation，移动窗估计 |
| Base parameters | 对观测力矩具有独立作用的最小动力学参数组合 |
| bias | 传感器、电流换算或模型中的近似恒定偏置 |
| candidate | 尚未获准进入运行配置的候选参数结果 |

### 采样频率越高越好吗？

不是。采样频率需要覆盖轨迹主要动态并留有抗混叠余量，但更高频率也会记录更多编码器量化噪声和通信抖动。应根据轨迹最高谐波、传感器带宽和驱动日志能力选择，并报告真实时间戳，而不是假设固定周期。

### 需要采多少数据？

不能只按样本行数判断。关键是回归矩阵是否达到期望秩、奇异值分布是否合理，以及独立验证误差是否稳定。大量重复姿态或相同速度的数据，信息量可能远小于一段设计良好的多谐波轨迹。

### 必须直接测量 $\dot q$ 和 $\ddot q$ 吗？

不一定。可以对实际编码器运动做平滑求导，或使用状态估计器；必须保存生成方法和参数，避免对含噪信号直接做连续差分。构造实验可以使用解析轨迹的真值导数，真机日志则不能直接用指令轨迹的解析导数替代实际运动：跟踪误差和驱动延迟会改变它们。

### 可以使用控制器指令力矩吗？

可以作为受限条件下的近似，但必须明确它不是实际输出力矩。驱动饱和、电流环、力矩变化率限制和通信延迟都会引入偏差；能够获得校准后的关节力矩或电机电流时，应优先使用实际测量链路。

### 辨识结果可以直接在线更新吗？

不可以默认这样做。在线估计器应只更新经过批准的参数子集，并具有持续激励、无接触、无故障等更新条件，以及范围、变化率、冻结和回滚保护。机器人本体惯量通常使用离线批准版本。

### 为什么预测力矩很好，参数却不可信？

多个参数可能高度相关并互相补偿，使 $Y\hat\theta$ 拟合良好，但单个物理参数不唯一。应同时检查基参数、秩、协方差、独立验证和物理一致性，而不是只看总体 RMSE。

### 固定基方法能直接用于人形或移动机器人吗？

不能直接套用。固定基机械臂将基座视为不动，关节状态通常足以构造对应的关节力矩回归矩阵；浮动基系统还需要描述基座的六维运动，并处理接触约束和外力：

$$M(q)\dot v+h(q,v)=S^T\tau+J_c^Tf_c$$

其中 $v$ 包含基座与关节广义速度，$S$ 选择驱动关节，$J_c^Tf_c$ 表示足底、手部等接触力。辨识时通常还需要 IMU、基座位姿/速度、接触状态以及足底或末端力传感器。接触力未知或接触切换未建模时，参数估计会把外力错误吸收到惯量和 bias 中。

当前 `robot_identification_pinocchio` 明确面向 `nq == nv` 的固定基模型。扩展到浮动基时，应版本化扩展数据契约、回归器接口和验证方法，而不是仅修改 URDF 根关节类型。人形机器人可进一步参考 FloBaRoID 一类针对 floating-base 的框架。

## 13. 小结

最终交付的不是一串参数，而是“数据 + 模型 + 诊断 + 审核 + 回滚 + 运行证据”的完整包。只有这个包能够被另一位工程师复现和审查，辨识结果才适合进入实际控制系统。


## 阅读自测与验收

- 检查回归矩阵的秩、条件数和原始列范数，结合噪声解释激励强度，并将训练与验证轨迹分开。
- 确认质量、惯量等估计参数的物理一致性，再比较留出轨迹上的预测力矩残差；较低训练误差不是唯一标准。
- 区分动力学后端与完整辨识流程，并能根据回归器、符号建模、浮动基需求选择五种工具之一。
- 运行 Pinocchio 示例，解释为什么 20 列回归矩阵只保留 6 个方向，以及为什么预测向量不能直接写回 URDF。
- 运行 Drake 示例，说明三个参数组合与四个物理未知量的区别，并用另一条轨迹验证力矩预测。
- 从位置／力矩日志生成处理后 CSV，检查窗口跨度、延迟正方向与边界裁剪；比较窗口时使用共同时间段，并保留输入哈希。
- 在延迟计算器中分别比较滞后记录与已对齐参考，解释为什么训练残差接近零仍可能得到错误参数。
