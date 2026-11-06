---
title: "cuRoboV2-Fast 详解：B 样条、力矩约束与 GPU 时间优化"
date: 2026-11-05
draft: false
tags: ["cuRobo", "Motion Planning", "Trajectory Optimization", "Robotics", "CUDA", "Dynamics", "Paper Notes"]
categories: ["机器人"]
authors: ["chase"]
summary: "从路径与时间的区别出发，推导 cuRoboV2-Fast 的 B 样条、TimeOpt 链式法则与负载力矩约束，再核对 GPU RRT、计算复用、实验口径和 V2 Python 接口，并提供可运行的标量算例。"
description: "从路径与时间的区别出发，推导 cuRoboV2-Fast 的 B 样条、TimeOpt 链式法则与负载力矩约束，再核对 GPU RRT、计算复用、实验口径和 V2 Python 接口，并提供可运行的标量算例。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
images: ["assets/planning-pipeline.webp"]
reading_prerequisites: "关节运动学、链式法则与刚体动力学基础；理解路径参数化和数值优化会更容易阅读 TimeOpt"
reading_focus: "先分清路径、时间律和执行轨迹，再沿平方路径速度推导约束；最后将论文的不同计时口径与固定源码的规划接口对齐。"
related_posts:
  - "/posts/trajectory/toppra"
  - "/posts/trajectory/ruckig"
  - "/posts/planner/task-and-motion-planning"
  - "/posts/robotics/dynamics/parameter-identification"
  - "/posts/cuda/warp"
---

让机械臂尽快绕开障碍，需要同时回答两个问题：**沿哪条路径运动，以及沿这条路径怎样分配时间。** 一个几何上更短的路径，可能经过重力负担大、力矩余量小的姿态，执行起来反而更慢。拿起物体后，碰撞几何和惯性都发生变化，空手时可行的运动也未必还能执行。

cuRoboV2-Fast 将路径优化、负载动力学和时间分配放在同一条 GPU 规划流水线中。理解它的重点，是把“计算一条轨迹花多久”和“机器人执行这条轨迹花多久”分开，再看两种时间分别由哪些设计改善。

本文依据 Balakumar Sundaralingam、Adithyavairavan Murali 和 Stan Birchfield 的 [《cuRoboV2-Fast: Fast Torque-Limited Collision-Free Motion Planning》arXiv v1](https://arxiv.org/html/2611.06913v1)，发布于 **2026-11-04**。实现部分核对 [NVlabs/curobo](https://github.com/NVlabs/curobo) 的固定提交 [`b29a8c2ee6869585bb05b056b9ab53b9c0753106`](https://github.com/NVlabs/curobo/tree/b29a8c2ee6869585bb05b056b9ab53b9c0753106)。论文描述的方法、当前源码的默认设置与本文构造的数学例子，会在对应章节分别说明。

官方仓库已经进入 V2 API：本文使用 `MotionPlanner`，不沿用 V1 的 `MotionGen` 调用方式。`cuRoboV2-Fast` 是论文的方法名称，本文所核对的实现位于上述仓库提交中，不是需要另找同名 pip 包才能使用的接口。

## 阅读路线

第一次阅读，建议按[问题定义](#problem) → [整体流程](#pipeline) → [B 样条](#spline) → [TimeOpt](#timeopt)建立主线。关注原理时继续读[力矩系数](#torque)与[带状系统](#banded)；关注工程接入时可转到[源码地图](#source)、[最小运行例子](#usage)和[执行边界](#limits)。[标准库算例](#scalar-example)不需要 GPU，可独立复算链式法则与力矩关系。

## 1. 规划问题：路径、时间律与执行轨迹 {#problem}

设机器人有 $d$ 个关节，关节配置为 $q\in\mathbb R^d$。路径 $q(s)$ 描述关节空间中的几何曲线，$s\in[0,1]$ 表示沿路径前进的进度。时间律 $s(t)$ 决定什么时候到达哪个进度，最后得到：

$$
q_{\mathrm{exec}}(t)=q(s(t)).
$$

同一条 $q(s)$ 可以对应许多条执行轨迹。缓慢通过整条路径、只在困难区段减速、快速通过宽松区段，空间路线可以相同，速度、加速度、jerk 和力矩却不同。

| 对象 | 决定什么 | 典型检查 |
| --- | --- | --- |
| 几何路径 $q(s)$ | 经过哪些关节配置 | 关节范围、机器人自碰撞、环境碰撞 |
| 时间律 $s(t)$ | 沿路径怎样前进 | 速度、加速度、jerk、力矩 |
| 执行轨迹 $q(s(t))$ | 关节随真实时间怎样变化 | 采样、控制接口、跟踪误差与实测限制 |
{.table-readable}

在静态场景、刚性几何模型和相同路径的前提下，重新分配时间不会改变路径上经过的空间姿态。但对于移动障碍物，机器人与障碍物在什么时刻相遇也参与碰撞判断；此时不能直接沿用“几何相同所以碰撞情况相同”的结论。

### 1.1 统一缩放时间为什么不够

假设将原轨迹的时间全部乘以 $\gamma>1$，也就是整体放慢。速度、加速度和 jerk 分别按 $\gamma^{-1}$、$\gamma^{-2}$、$\gamma^{-3}$ 缩小。这有助于满足运动学限制，但它让每一段一起变慢，宽松区段也受到最困难区段的牵制。

动力学还有一个额外问题：重力项不会随统一降速一起消失。刚体模型常写成：

$$
\tau=M(q)\ddot q+C(q,\dot q)\dot q+g(q).
$$

减速可以减小惯性与速度相关项，却不能保证某个必须保持的姿态在给定负载下可静止支撑。后面会给出一个明确的单摆反例。

## 2. 整体流程：优化几何，再改善时间分配 {#pipeline}

论文的方法先用直接种子进行轨迹优化，必要时再调用 GPU RRT 提供绕障路线。轨迹优化用 B 样条改进路径与统一时间尺度，TimeOpt 随后固定几何、优化局部前进速度，最后检查输出轨迹。

<figure class="article-figure" id="figure-1">
  {{< post-image src="assets/planning-pipeline.webp" alt="cuRoboV2-Fast 规划流程：起点与目标、机器人与负载、碰撞场景共同参与规划，IK 和直接种子进入 B 样条轨迹优化，必要时由 GPU RRT 产生替代种子，随后进行固定路径 TimeOpt 和采样验证，控制器执行是后续阶段" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">路径、动力学与时间优化的职责。RRT 提供种子，TimeOpt 沿固定几何调速；示意的是论文主线，源码也允许显式选择先运行图规划等配置。</span></figcaption>
</figure>

这里需要区分三个层次的可行性：

1. IK 找到了满足末端目标的关节配置，并不代表存在到达它的无碰撞路径。
2. RRT 找到了通过当前离散碰撞检查的路线，并不代表这条路线已经满足平滑、力矩和执行采样要求。
3. 优化与最终检查通过，表示满足相应模型与采样契约；控制器跟踪、感知误差与接触过程仍是执行系统的职责。

这也解释了为何不能把论文中的 RRT 时间直接当作完整动作生成时间。

## 3. GPU RRT：把树扩展与验证拆开 {#rrt}

RRT 从起点树和目标树向中间生长。每轮可以并行生成多个候选：采样关节配置，寻找树中最近节点，沿采样方向推进有限步长，再检查关节范围、自碰撞与环境碰撞。

这一实现的关键，是将候选扩展、沿边采样验证和树节点写入组织成独立步骤。对于一条后半段被挡住的边，可以保留其有效前缀；对已接受的扩展，用并行扫描为端点分配不同存储位置，再写入树中。两棵树连通后，通过父节点索引恢复路线，接着进行捷径缩短和样条拟合。

这样设计有两方面价值：计算上，可以把许多状态的验证合并执行；工程上，增加验证内容时，不必同时重写树生长内核。缓冲区复用和 CUDA graph 则降低了反复发射小计算的开销。

搜索仍然受到节点容量、扩展轮数、随机采样和验证分辨率的限制。耗尽当前预算只能说明这次没有找到路径，不能证明整个连续配置空间不存在解。与[TAMP 的可行性证据](../../planner/task-and-motion-planning/)一样，应明确失败究竟来自碰撞见证、数值问题还是搜索预算。

## 4. B 样条：让采样与优化变量解耦 {#spline}

### 4.1 每个采样只依赖少量控制点

将关节轨迹表示为三次 B 样条，每一段只依赖四个相邻控制点。设段内坐标为 $\alpha\in[0,1]$、结点间隔为 $h$，对于位置及前三阶时间导数，可以概括为：

$$
q^{(r)}(t)=h^{-r}\sum_{j=0}^{3}
N_j^{(r)}(\alpha)\,\widetilde u_{k+j}.
$$

这里 $r=0,1,2,3$ 分别对应位置及前三阶时间导数。

$\widetilde u$ 表示包含边界构造在内的控制序列，$N_j$ 是当前段的三次基函数。增加轨迹检查的采样数，会增加并行评估工作，却不必同等增加优化变量数。修改某个控制点，也只会影响它的局部支撑范围。

要注意“控制点”“样条结点”和“输出采样点”的区别：控制点是优化变量，结点划分基函数作用区间，采样点用于计算代价、约束或交给控制器。把三个数量写成同一个“轨迹长度”，会混淆计算量与运动分辨率。

### 4.2 反向传播为什么用 gather

若每个采样点把梯度散写到其四个控制点，多个采样可能同时更新同一个控制点，需要处理写冲突。另一种组织方式是：让一个控制点主动收集支撑区内各采样的梯度，再一次写回自己的结果。

用线性映射写采样导数：

$$
q_i^{(r)}=A_i^{(r)}U+b_i^{(r)},
$$

则控制点梯度为：

$$
\nabla_U\mathcal L=
\sum_i\sum_{r=0}^{3}
\bigl(A_i^{(r)}\bigr)^{\mathsf T}
\nabla_{q_i^{(r)}}\mathcal L.
$$

实现可以直接应用局部基函数权重，不需要显式生成大矩阵 $A$。这一组织方式还使位置、速度、加速度与 jerk 的贡献能在同一个控制点归约中合并。

三次样条也不自动意味着所有高阶量处处连续：在通常的简单内部结点处，它可保持位置、速度、加速度连续，而三阶导数可以发生跳变。因此检查 jerk 时需要关注结点两侧，不能只看一组恰好避开结点的均匀采样。

## 5. TimeOpt：用平方路径速度重新分配时间 {#timeopt}

这一阶段固定已经选定的几何路径 $q(s)$，引入平方路径速度：

$$
x(s)=\dot s^{\,2}>0.
$$

再用另一组三次 B 样条表示 $x(s)$。几何控制点与时间控制点是两组不同变量；改变时间控制点会改变行进速度，不会重新绕开一个障碍物。

### 5.1 从路径导数得到真实时间导数

以下用撇号表示对 $s$ 求导，用点表示对真实时间求导。因为 $\dot s=\sqrt{x}$：

$$
\ddot s=\frac12x',\qquad
\dddot s=\frac12\sqrt{x}\,x''.
$$

因此关节的速度、加速度和 jerk 分别是：

$$
\dot q=q'\sqrt{x},
$$

$$
\ddot q=q''x+\frac12q'x',
$$

$$
\dddot q=\sqrt{x}\left(
q'''x+\frac32q''x'+\frac12q'x''
\right).
$$

这些式子说明：局部调速不仅改变速度，也通过 $x'$、$x''$ 改变加速度与 jerk。若只调整一个“局部速度上限”而不检查导数，可能让速度看似合规，jerk 却超限。

### 5.2 时长目标与时间映射

正向沿路径行进时：

$$
\begin{aligned}
T&=\int_0^1\frac{1}{\sqrt{x(s)}}\,ds,\\
t(s)&=\int_0^s\frac{1}{\sqrt{x(u)}}\,du.
\end{aligned}
$$

优化目标是在速度、加速度、jerk 和力矩限制下缩短这个时间。数值求解使用有限采样近似目标和约束，时间映射还需要积分与反演，才能按控制周期输出 $q(s(t))$。不能把沿路径均匀采样的数组直接当作等时间间隔的执行指令。

<figure class="article-figure" id="figure-2">
  {{< post-image src="assets/fixed-path-timing.svg" alt="同一固定路径上的两种时间律：常数平方路径速度 0.25 对应 2 秒，构造的变速函数 0.25 加 s 减 0.5 的平方对应约 1.76275 秒；只比较时间积分，没有施加执行约束" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text">构造的固定路径时间分配算例。虚线为常数速度律，实线在两端更快；11.86% 的时长差仅来自这两个函数的积分，不是论文实验，也没有验证力矩或起停约束。</span></figcaption>
</figure>

图 2 的路径区段以正速度经过端点，便于检查公式。完整规划器还要处理规定的起始状态、边界构造和静止轨迹；这里没有用这个简化算例替代其端点处理。

正的路径进度速度也不必然意味着关节端点在运动：若边界几何满足 $q'=0$，则 $\dot q=q'\sqrt{x}=0$；加速度还要同时考虑 $q''$ 等边界条件。因此，不能简单把时间系数设为零来替代完整起停构造，否则积分和求解器的正值条件都可能受到影响。

### 5.3 TimeOpt 改善什么，不能改变什么

TimeOpt 的优化范围是当前路径上的时间分配。它可以在紧约束区域减速，在宽松区域加速；如果固定路径本身存在无法满足的限制，重新调速不一定能修复它。

若负载改变后必须改变关节姿态或绕行方式，就需要重新做带动力学的几何优化，必要时重新选择 IK 解或路径种子。先优化几何与后优化时间相互补充，并不意味着固定路径时间优化能够解决所有负载问题。

## 6. 力矩约束：三次逆动力学，复用多次时间更新 {#torque}

### 6.1 固定路径上的仿射关系

把 TimeOpt 的链式法则代入刚体动力学。在通常的刚体模型下，科里奥利／离心项对速度是二次的，于是可将固定路径上的力矩写成：

$$
\tau(s)=\frac12A(s)x'(s)+D(s)x(s)+g(s).
$$

其中：

$$
A(s)=M(q(s))q'(s),
$$

$$
D(s)=M(q(s))q''(s)
+C(q(s),q'(s))q'(s).
$$

这里的 $A,D,g$ 都是每个关节一项的向量，几何路径固定后就可以预计算。注意该关系依赖所采用的动力学模型；额外的速度线性摩擦、接触力或其他执行器模型，不能不加处理就塞进同一个二次速度假设。

### 6.2 如何用现有逆动力学获得系数

记逆动力学接口为 $\operatorname{ID}(q,v,a)$。每个路径采样位置计算三次：

$$
g=\operatorname{ID}(q,0,0),
$$

$$
A=\operatorname{ID}(q,0,q')-g,
$$

$$
D=\operatorname{ID}(q,q',q'')-g.
$$

随后每次更新时间律，只需计算 $\frac12Ax'+Dx+g$，无需重新运行完整逆动力学。这里第二、第三次调用中的输入是构造系数的代数探针，不是在声称 $q'$ 本身就是当前真实关节速度或加速度。

这是与 [TOPPRA](../../trajectory/toppra/) 相联系的地方：二者都利用固定路径后的动力学结构。TimeOpt 进一步围绕平方速度样条的局部支撑组织 GPU 优化，并在其问题中处理采样的 jerk 约束；不能仅凭都在做时间参数化，就认为算法、约束与保证完全相同。

### 6.3 负载为何会迫使路径变化

以单摆的构造模型为例，负载位于转轴外 0.5 m，基础等效重力质量 0.3 kg。在水平姿态 $q=\pi/2$ 且必须静止时，2 kg 负载需要：

$$
\tau_g=(0.3+2)\times9.81\times0.5
=11.2815\ \mathrm{N\,m}.
$$

若力矩上限为 5 N·m，这个规定的静止姿态不可行。把到达过程放慢，不会改变该静止姿态所需的重力力矩。对于冗余机械臂，改变关节配置可能改变力臂与各关节负担；但这属于几何／IK 选择的变化，而不是沿原路径调速。

<figure class="article-figure" id="figure-3">
  {{< post-image src="assets/payload-static-limit.svg" alt="单摆水平静止姿态的重力力矩随负载质量线性增加，2 千克负载需要 11.2815 牛米，超过构造的 5 牛米上限；时间重分配不能改变这一静态需求" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text">规定静止姿态下的负载反例。实线来自明确给定的单摆参数，虚线为构造的力矩上限；它不表示某台实际机械臂的负载能力。</span></figcaption>
</figure>

## 7. 局部支撑为什么产生带状 Newton 系统 {#banded}

三次时间样条的一个采样只依赖四个相邻系数，所以对应目标项或约束的梯度也只涉及这四项。把这些局部贡献组合起来，Newton 系统的主要耦合集中在矩阵对角线附近。

当前 CUDA 求解器按这个结构存储带状因子，而不是把时间控制点当作完全稠密的通用优化问题。预测方向与校正方向复用同一次 Cholesky 分解，约束采样与局部贡献计算则可以并行执行。

求解中的 primal–dual predictor–corrector 可以这样理解：先预测一个更接近约束边界和互补条件的方向，再校正不同约束的间隙与乘子，随后用回溯步长保持可行的内部状态。求解器保留找到的较短可行解，最终仍要检查输出。

局部结构带来的高效数值求解，与全局最优性是两回事。尤其 jerk 中含有平方根与导数乘积，整个约束问题不能因为力矩对 $x,x'$ 仿射，就被简单称为“已获得全局最优的凸规划”。

固定源码的独立 `TrajectoryTimeOptCfg` 使用 **32 个时间控制系数**，默认 511 个均匀区间，并额外考虑路径结点两侧与时间结点等约束位置。它要求 CUDA FP32 模型，当前独立 TimeOpt 接口检查 1–64 个自由度；这些是该实现的接口限制，不是 B 样条理论只能使用 32 个系数。

## 8. GPU 计算：少算、复用、直接求损失梯度 {#gpu}

### 8.1 运动学与 Jacobian

机器人拓扑可以在初始化时准备好：每个关节影响哪些后代连杆，哪些关节对某个工具位姿有贡献。前向计算得到连杆与碰撞球的位置，后向先把输出的梯度合并成连杆上的等效 wrench，再按关节收集受其影响的连杆贡献。

需要显式 Jacobian 的 IK，可按列并行计算；只需要标量损失梯度的轨迹优化，则可以直接计算向量—Jacobian 乘积，不必先生成完整 Jacobian 再相乘。两种接口服务不同的优化算法。

### 8.2 碰撞粗筛与重复查询复用

用许多球表示机器人碰撞体时，两两测试数量会随球数增长。对明显分离的连杆，先用各连杆球集合的 AABB 做粗筛，可以跳过后续球对检测。环境扫掠查询也可以先排除距离较远的障碍物，并复用相邻半段在公共位置的距离结果。

这类粗筛是在相同几何与检查分辨率下减少不必要工作，不应被解读成把球模型自动升级成精确网格连续碰撞证明。机器人球近似、激活距离、环境表达和采样间距仍然影响结果。

### 8.3 可微逆动力学与运行时负载

递归 Newton–Euler 算法沿运动链向外传播速度与加速度，再向根部累积力，得到关节力矩。GPU 实现可以并行处理独立链，在依赖交汇处合并贡献，并用紧凑空间运算减少中间存储。

轨迹优化真正需要的往往是损失关于 $q,\dot q,\ddot q$ 的梯度。直接把力矩损失反传到这些量，可以避免先生成完整动力学导数矩阵再收缩；随后再沿样条采样关系反传到控制点。

质量、质心与惯性作为运行时数据更新，约简后的拓扑不变时可以继续复用调度。这与“任意换一个机器人都无需编译”不同：自由度、维度和执行选项仍可能参与内核专化。

## 9. 抓取集成：几何附着与惯性更新分别处理 {#grasp}

抓取任务通常给出多个候选物体抓取位姿。IK 可以先筛选并排序目标关节配置，再为选中的目标规划到达预抓取位姿、沿接近方向进入抓取位姿、抬起物体等阶段；转移到放置点是后续独立规划调用。

抬起物体后，至少有两种状态变化：

- **碰撞几何变化。** 持有物体成为随工具一起移动的碰撞体，原先环境中的物体表示也要正确更新。
- **动力学变化。** 物体的质量、质心位置与惯性改变力矩需求；只有碰撞形状正确，不代表力矩计算也正确。

接触阶段还需要允许预期接触，同时避免把整只机械臂与环境的碰撞都关闭。对于多段轨迹，应保留各段末端状态与下一段起点的一致性，并明确段间时间如何拼接。

规划器选出一个 IK 可达抓取，不保证从当前状态能无碰撞到达，更不直接证明抓取稳定、无滑动或接触过程成功。高层任务排序、抓取生成、视觉估计与低层控制需要独立的接口与反馈。

## 10. 实验怎样读：三个计时尺度 {#experiments}

论文报告的核心结果见下表；这些是作者的实验，不是本文算例或本机测试。

| 实验口径 | 作者报告的结果 | 阅读时固定的条件 |
| --- | --- | --- |
| 完整运动规划 | 3 kg 负载时，RTX 5090 平均 4.43 ms | 对比 cuRoboV2 为 14.7 倍；不是单独 RRT 时间 |
| 嵌入式规划 | 同一负载，AGX Orin 平均 25.99 ms | 设备不同，不能将其当作同一硬件的简单替换 |
| 真机完成周期 | YAM 平均 5.93 s，比 V2 缩短 34.6% | 使用 RTX 6000 Ada，包含运动、夹爪与其他处理 |
{.table-readable}

出处为[论文实验章节 IV-C 与 IV-G](https://arxiv.org/html/2611.06913v1#S4)。完整规划、局部模块和真机周期使用不同的输入集合、统计方式与硬件，不能把其中最小的一个数字作为“cuRobo 一次动作耗时”。

### 10.1 少迭代与单次更快共同生效

路径优化时间可以粗略理解为“迭代次数 × 单次迭代成本”，还要加上初始化、IK、几何搜索、检查和其他调度开销。B 样条表示与求解策略影响前者，计算复用与内核组织影响后者。

论文中的收敛消融，与完整计时配置也不能混用：较少初始迭代用于分析收敛，完整运行采用自己的初始与细化预算。当前源码的 `outer_iterations` 又是初始求解／检查轮次，而不是直接等于论文里的 L-BFGS 步数，见后面的接口说明。

### 10.2 周期变快主要来自哪里

真机完成周期中，规划已经只占约 1%。继续缩短规划延迟当然有价值，但单独把这一小部分消除，也不会让整个周期按相同比例大幅缩短。更快且可跟踪的运动、夹爪动作以及其他处理，决定了剩余时间。

这也是文章开头要区分两种时间的原因：更快算出一条慢轨迹，与更快算出一条执行时间短的轨迹，产生的系统收益不同。

比较成功率时还要看分母、负载与共享成功查询集。只在所有方法都成功的查询上比较轨迹时长，与在完整问题集上统计规划成功率，回答的是不同问题。

## 11. 源码地图：对应到固定提交 {#source}

以下路径均相对于仓库根目录，建议结合 [固定提交](https://github.com/NVlabs/curobo/tree/b29a8c2ee6869585bb05b056b9ab53b9c0753106)阅读。

| 路径 | 作用 |
| --- | --- |
| `curobo/motion_planner.py` | 公开规划入口与轮次契约 |
| `curobo/_src/motion/motion_planner_cfg.py` | IK、图规划、TrajOpt 与 TimeOpt 组合配置 |
| `curobo/_src/motion/motion_planner.py` | 重试、直接种子与图种子的规划编排 |
| `curobo/_src/graph_planner/rrt/` | 双向 RRT 扩展、验证、树与路径处理 |
| `curobo/_src/curobolib/kernels/trajectory/bspline/` | 样条采样、边界与梯度内核 |
| `curobo/_src/motion/motion_trajectory_timeopt.py` | 固定路径数据、动力学系数与结果构造 |
| `curobo/_src/trajectory/timeopt/` | 时间样条基与 GPU 求解器组织 |
| `curobo/_src/curobolib/kernels/trajectory/timeopt/` | TimeOpt 的 CUDA 计算 |
| `curobo/examples/getting_started/motion_planning.py` | 官方姿态规划与抓取示例 |
{.table-readable}

一个容易忽略的默认值是：当前 `MotionPlannerCfg.create()` 的 `graph_planner_algorithm` 默认仍为 `"prm"`，而官方入门示例显式设置 `"rrt"`；`enable_timeopt` 默认是 `True`。因此，阅读 Fast 论文后直接用所有默认配置，不能自动推断本次执行的每个模块都与论文评估配置一致。

`outer_iterations` 控制每次轨迹优化中的初始求解／可行性检查轮次，默认每轮运行 50 个优化步骤；`finetune_attempts` 控制额外的细化轮次，默认每轮 25 步；`max_attempts` 则是规划重试预算。直接种子和图种子可各自触发优化，所以不能把一个局部轮次参数当作整次调用的总迭代上限。

## 12. 安装与最小调用 {#usage}

### 12.1 固定源码与依赖组合

下面按当前固定提交的安装说明，在支持的 Linux／NVIDIA 环境中创建独立环境。CUDA 12 与 CUDA 13 的依赖组分开选择，不要同时安装：

```bash
git clone https://github.com/NVlabs/curobo.git
cd curobo
git checkout --detach b29a8c2ee6869585bb05b056b9ab53b9c0753106
uv venv --python 3.11
source .venv/bin/activate

# 新环境，选择 CUDA 13 的依赖组（包含 PyTorch）
uv pip install '.[cu13-torch]'

python -c 'import curobo; print(curobo.__version__); print(curobo.__file__)'
```

若使用相应的 CUDA 12 环境，将安装组替换为 `'.[cu12-torch]'`；已有匹配的 PyTorch 时，仓库也提供不重复声明 torch 的 `cu12`／`cu13` 组。项目元数据要求 Python 3.10+，具体驱动、GPU 与软件栈仍需按该提交的[安装说明](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/docs/getting-started/installation.rst)核对。

首次构建与运行会涉及编译、缓存、内存分配和 CUDA graph 预热。应把这些准备成本与后续稳定调用分开记录；`nvidia-smi` 显示的 CUDA 支持版本也不能独自证明整个 PyTorch／运行时／编译器组合已经匹配。

### 12.2 运行官方示例

```bash
python -m curobo.examples.getting_started.motion_planning --mode pose
python -m curobo.examples.getting_started.motion_planning --mode grasp
```

需要浏览器交互查看时，可按示例说明加入 `--visualize`。这类可视化与仿真示例不等于已经连接真实机械臂的控制程序。

### 12.3 一个 V2 风格的姿态规划骨架

下面从默认关节状态构造一个小幅改变后的关节目标，再通过 FK 转成末端目标。这能明确示例的目标来自哪里，避免写入一组不知参考坐标系的任意末端坐标。

```python
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState


def main():
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
        graph_planner_algorithm="rrt",
        enable_timeopt=True,
    )
    with MotionPlanner(config) as planner:
        planner.warmup(enable_graph=True, num_warmup_iterations=2)
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )
        goal_joints = start.clone()
        goal_joints.position[:, 0] += 0.2
        goal_pose = planner.compute_kinematics(goal_joints).tool_poses.as_goal()
        result = planner.plan_pose(
            goal_pose, start,
            max_attempts=1,
            enable_graph_attempt=0,
        )
        if result is None or result.success is None or not bool(result.success.any()):
            raise RuntimeError("本次规划没有通过结果检查")
        trajectory = result.get_interpolated_plan()
        print(trajectory.position.shape)
        print(trajectory.joint_names)
        print(trajectory.dt)


if __name__ == "__main__":
    main()
```

`enable_graph_attempt=0` 在这个示例中显式允许首轮进入图规划；它不是图 1“先直接优化、失败再启用 RRT”策略的必选写法。主入口保护也应保留，避免编译子进程重新执行顶层规划代码。

本文在 RTX 5090 D v2、PyTorch 2.13.0／CUDA 13.0 环境中，使用固定源码跑通了这个单查询流程。所检查配置开启 RRT 与 TimeOpt，返回通过验证的结果；活动机械臂关节有 7 个，而插值输出包含两根手指，共 9 列。这是接口冒烟检查，既不是论文 RTX 5090 的性能复现，也没有连接真实机器人。

消费结果时应检查成功掩码、实际有效轨迹长度、采样周期和关节名称。规划器的活动关节与输出中可能补回的锁定关节未必具有相同数量，不能按自己猜测的数组顺序发指令。需要跨下一次求解保存的数据，也应核对结果是否复用内部缓冲，必要时复制。

## 13. 独立可运行的数学算例 {#scalar-example}

下载 [timeopt_sanity.py](timeopt_sanity.py)，只使用 Python 标准库即可运行：

```bash
python3 -B timeopt_sanity.py --check
```

它选择一个构造的单关节路径：

$$
\begin{aligned}
q(s)&=0.2+0.6s+0.3s^2-0.1s^3,\\
x(s)&=0.25+(s-0.5)^2.
\end{aligned}
$$

在 $s=0.5$ 处，链式法则给出：

| 量 | 数值 |
| --- | --- |
| 关节位置 | 0.5625 rad |
| 关节速度 | 0.4125 rad/s |
| 关节加速度 | 0.075 rad/s² |
| 关节 jerk | 0.3375 rad/s³ |

检查器使用反演时间映射后的 $q(s(t))$ 做高精度数值差分，独立核对这三个导数；同时检查四种负载、101 个路径位置上的直接逆动力学与 $\frac12Ax'+Dx+g$ 是否一致，并用数值积分核对图 2 的闭式时间。

这个文件不实现 GPU RRT、碰撞检查或 primal–dual 求解器，也没有起停约束和真实控制器。它能验证正文中的代数、单位与反例，不能证明某条机器人轨迹在硬件上可执行。

## 14. 接入时需要保留的边界 {#limits}

| 容易混淆的结论 | 应当检查的实际条件 |
| --- | --- |
| 轨迹通过采样检查，所以连续时间绝对无碰撞 | 检查分辨率、扫掠模型、球近似与环境表达 |
| 只要放慢就满足负载要求 | 静止姿态的重力力矩、加减速与执行器限制 |
| 更新了物体碰撞体，就完成了负载更新 | 质量、质心、惯性与几何附着分别核对 |
| TimeOpt 能修复任意坏路径 | 固定路径上的可行速度律是否存在 |
| GPU 求解快，所以控制跟踪一定正确 | 控制周期、时延、关节映射、跟踪误差与硬件反馈 |
| 论文的亚毫秒模块时间等于动作周期 | 初始化、IK、优化、执行、夹爪和其他处理的计时范围 |
{.table-readable}

仓库使用 Apache-2.0，示例机器人资源还有各自的许可证，具体见仓库的 `LICENSE` 与 `LICENSE_ASSETS`。独立数值算例和一次成功的 GPU 调用，都不替代自己的场景、负载与控制系统验证。

## 阅读自测与验收

- 路径 q(s)、时间律 s(t) 与执行轨迹 q(s(t)) 分别决定什么？
- 为什么统一减速不能解决所有力矩问题？
- RRT 的粗检查路线为什么还需要拟合、优化与最终检查？
- 三次 B 样条的四点局部支撑怎样同时帮助前向、反向和带状求解？
- 控制点、样条结点与输出采样点为什么不能混称为轨迹长度？
- 从 x(s)=ṡ² 怎样推导关节速度、加速度和 jerk？
- 三次逆动力学调用怎样得到 A、D、g，依赖哪些模型假设？
- 为什么 TimeOpt 的局部高效求解不意味着全局最优？
- 物体几何附着与负载惯性更新各自改变什么？
- 规划时间、计划执行时长和真机周期为什么不能直接比较？
- 当前 V2 的 RRT 选择、TimeOpt 开关和求解轮次怎样显式配置？
- 为什么需要按输出自身的关节名称、有效长度与时间戳接入控制器？

## 参考资料

- [cuRoboV2-Fast 论文 v1](https://arxiv.org/html/2611.06913v1)：方法、实验设置与报告结果；论文以 CC BY 4.0 发布。
- [NVlabs/curobo 固定源码](https://github.com/NVlabs/curobo/tree/b29a8c2ee6869585bb05b056b9ab53b9c0753106)：本文 API 与实现核对的依据。
- [TimeOpt 配置与接口](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/_src/motion/motion_trajectory_timeopt.py)：固定路径、采样可行性与缓冲复用约定。
- [官方运动规划示例](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/examples/getting_started/motion_planning.py)：姿态规划、抓取阶段与可视化。
