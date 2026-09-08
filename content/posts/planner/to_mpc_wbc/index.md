---
title: TO、MPC与WBC在机器人控制中的作用与区别对比文档
date: 2025-05-06
lastmod: 2026-09-08
draft: false
tags: ["Motion Planning", "MPC", "Whole-Body Control"]
categories: ["机器人技术"]
authors: ["chase"]
summary: "梳理轨迹优化、MPC 与全身控制的职责、反馈链路和接口，说明它们如何组合以及不能相互替代的约束。"
math: true
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "梳理轨迹优化、MPC 与全身控制的职责、反馈链路和接口，说明它们如何组合以及不能相互替代的约束。"
contentLanguage: "zh-CN"
reading_prerequisites: "机器人状态、优化与反馈控制"
reading_focus: "沿目标、状态估计、规划和执行追踪数据流，不用运行频率定义算法类别。"
related_posts:
  - "/posts/casadi"
  - "/posts/robotics/control/impedance-control"
---

TO、MPC 和 WBC 不是三种互斥算法：TO 描述如何优化一段轨迹，MPC 描述如何滚动求解并利用反馈执行，WBC 描述如何协调全身多个任务。一个 MPC 内部可以求解 TO 问题，WBC 也可以采用不同形式的控制器。

![轨迹优化生成参考、MPC 滚动规划、WBC 协调全身任务，并由状态估计形成反馈的关系](assets/planning-control-loop.webp "一种常见架构：TO 提供参考，MPC 根据状态重新规划，WBC 把运动与接触目标转成关节指令。具体系统可以合并或省略某一层。")

## 1. 从输入与输出理解职责

| 模块 | 典型输入 | 典型输出 | 主要问题 |
| --- | --- | --- | --- |
| TO：轨迹优化 | 初末状态、动力学、约束、代价 | 一段状态与控制轨迹 | 这段运动是否可行，代价有多大？ |
| MPC：模型预测控制 | 当前状态、参考、预测模型 | 当前控制量及预测轨迹 | 执行后状态变了，接下来如何调整？ |
| WBC：全身控制 | 任务目标、接触状态、机器人模型 | 关节力矩、速度或加速度等指令 | 多个任务如何共享关节和接触能力？ |

TO 可以处理动力学、接触和时变约束，也可以在线求解。机器人 TO 往往是非凸问题，求解器得到的通常是局部解，不能笼统保证全局最优。

MPC 在每个周期更新状态，求解有限时域问题，执行当前一小段控制，再滚动向前。它可以采用简化模型，也可以使用更完整的模型。

WBC 常使用加权 QP 或任务优先级优化，但 WBC 本身不等于 QP；基于零空间投影等方法也可以实现全身协调。

## 2. 为什么需要不同时间尺度

长时域规划需要预见未来运动；关节层还需及时响应测量和接触变化。把问题分层，能让每层使用合适的模型与计算预算。

各层频率取决于问题规模、求解器、机器人硬件和稳定性要求。“TO 秒级、MPC 毫秒级、WBC 更快”只能作为某类系统的例子，不能作为定义。MPC 与 WBC 也不一定需要 GPU。

## 3. 以足式机器人为例

1. 规划层产生落足、质心或末端参考。
2. MPC 根据当前状态与接触计划，预测未来运动并调整控制目标。
3. WBC 综合平衡、姿态、足端与操作任务，求解满足约束的关节指令。
4. 状态估计器融合编码器、IMU 等信息，为下一轮规划和控制提供反馈。

对动力学 WBC，常见决策变量包括广义加速度、关节力矩和接触力。应同时检查浮动基动力学、摩擦约束、接触运动约束和执行器限制。高层目标不可行时，底层不能凭空保证精确跟踪。

## 4. 接口比模块名称更值得检查

| 接口问题 | 常见后果 | 检查方式 |
| --- | --- | --- |
| 坐标系或力的正负号不一致 | 运动方向或接触力错误 | 单轴、静态场景验证 |
| 接触计划与实际接触不同 | 约束不成立、力矩突变 | 对比计划接触与传感器反馈 |
| 高层输出超过执行器能力 | WBC 不可行或长期饱和 | 记录约束余量与求解状态 |
| 状态估计或控制消息过期 | 跟踪抖动、稳定性下降 | 测量时间戳与完整控制延迟 |

## 5. 延伸阅读与工具

- [MIT Underactuated Robotics：Trajectory Optimization](https://underactuated.mit.edu/trajopt.html)：理解直接配点、约束和非凸优化。
- [OCS2](https://leggedrobotics.github.io/ocs2/)：用于最优控制与 MPC 的工具箱，需要用户定义或接入系统模型。
- [TinyMPC](https://tinympc.org/)：关注资源受限平台上的 MPC 求解。
- [OpenLoong Dynamics Control](https://github.com/loongOpen/Openloong-dyn-control)：查看具体人形机器人中的 MPC 与 WBC 组合。

评估实现时，记录求解耗时分布、最坏控制周期、约束违反和失败恢复；只报告平均频率无法说明控制链路是否可靠。


## 6. MPC：优化未来序列，闭环执行当前一段 {#mpc}

从当前估计状态 $\hat x_t$ 出发，通用有限时域问题可以写成：

$$
\begin{aligned}
\min_{x_{0:N},u_{0:N-1}}\;&\sum_{k=0}^{N-1}\ell(x_k,u_k)+\ell_f(x_N)\\
\text{s.t.}\;&x_0=\hat x_t,\quad x_{k+1}=f_d(x_k,u_k,\sigma_k),\\
&x_k\in\mathcal X(\sigma_k),\quad u_k\in\mathcal U(\sigma_k).
\end{aligned}
$$

$k$ 是预测节点索引，N 是预测步数；$x_k\in\mathbb R^{n_x}$、$u_k\in\mathbb R^{n_u}$ 分别为状态与控制，$\sigma_k$ 表示接触模式。这里先把接触日程视为已知；若把接触时刻或接触集合一起优化，问题结构和求解难度也会改变。

代价表示偏好，约束表示允许范围。跟踪误差小的解如果要求地面提供向下拉脚的法向力，仍然不满足单侧接触条件。求解后采用当前第一项或第一小段，获得新测量后重新预测；旧轨迹可以用于 warm start，但不应脱离有效期与状态偏差检查一直播放。[MIT 轨迹优化与 MPC](https://underactuated.mit.edu/trajopt.html)

| 预测模型 | 常见决策量或输出 | 需要继续核对 |
| --- | --- | --- |
| 质心/动量或单刚体模型 | 质心运动、接触力、动量变化 | 是否覆盖关节限位、运动学可达性和执行器力矩 |
| 带落脚变量的降阶模型 | 落脚点、接触日程、受力参考 | 接触日程是给定还是优化变量 |
| 全身模型 | 全身状态、力矩、接触力、反馈增益 | 与下游 WBC 的职责是否重叠 |
| WholeBodyX 当前关节积分模型 | 关节位置与速度参考 | 没有浮动基、接触力或惯性预测 |

状态估计与目标命令是两个入口：估计告诉控制器“现在在哪里”，目标说明“希望去哪里”。当前状态还应直接进入 WBC；低层伺服和真实执行器位于 WBC 输出之后，优化结果不能等同于已经实现的动作。

## 7. WBC：当前拍协调任务与物理约束 {#wbc}

### 7.1 动力学 WBC 的决策变量

常见浮动基动力学方程为：

$$
M(q)\dot v+h(q,v)=S^T\tau+J_c(q)^T\lambda.
$$

$q$ 是构型，$v\in\mathbb R^{n_v}$ 是广义速度，$\dot v$ 是广义加速度；$M\in\mathbb R^{n_v\times n_v}$、$h\in\mathbb R^{n_v}$ 分别为质量矩阵和偏置项。$S\in\mathbb R^{n_a\times n_v}$ 选择执行器自由度，$\tau\in\mathbb R^{n_a}$ 为关节力矩。$J_c\in\mathbb R^{n_c\times n_v}$，$\lambda\in\mathbb R^{n_c}$ 是环境施加于机器人的接触力或 wrench，二者必须使用一致的坐标与排列。

浮动基的六个速度自由度没有直接电机驱动；若姿态用四元数存储，构型维数 $n_q$ 与速度维数 $n_v$ 还可能不同，不能直接用逐元素相加更新构型。单点接触常用三维力，面接触可用六维 wrench 或多个接触点表达。

以 $z=[\dot v;\lambda;\tau]$ 为决策变量，可以构造当前拍的加权 QP：

$$
\begin{aligned}
\min_z\;&\frac12\sum_i\|A_i z-b_i\|_{W_i}^2+\frac\rho2\|z\|^2\\
\text{s.t.}\;&Ez=e,\qquad l\le Cz\le u.
\end{aligned}
$$

这里 $\|r\|_W^2=r^TWr$，$W_i\succeq0$，$\rho\ge0$。任务残差可以来自摆动脚、躯干或姿态的期望加速度；对普通运动学任务有 $\ddot x_i=J_i\dot v+\dot J_i v$。姿态误差需要使用旋转的局部表示，不能直接相减四元数。

保持静止的刚性支撑接触常写成 $J_c\dot v+\dot J_c v=0$；动力学、单侧法向力、摩擦边界和执行器限位分别进入等式或不等式约束。线性化摩擦锥可得到 QP；保留非线性锥或接触互补关系时，求解问题可能不再是同一种 QP。[TSID 官方实现](https://github.com/stack-of-tasks/tsid)

### 7.2 权重、层级与求解失败

有限权重允许任务之间折中，不能把“大权重”称为严格优先级。层级逆动力学或级联 QP 会约束低层不能破坏高层已经达到的最优结果。两者都还要处理不可行、数值误差与执行延迟。[Herzog 等：层级逆动力学与动量控制](https://arxiv.org/abs/1410.7284)

右脚离地时，如果它已经不再接触地面，相应接触力应被移除或约束为零；左脚仍承担支撑。若目标超出摩擦或力矩范围，应调整软任务、上层参考或接触计划。只有优化器报告成功还不够：错误的接触判断、质量惯量、状态估计或伺服带宽，都会使“模型可行”与“物理可行”分离。

## 8. WholeBodyX：把概念逐项映射到源码 {#wholebodyx}

以下结论来自 2026-09-08 检查的本地 WholeBodyX 工作树。该目录没有可解析的 Git HEAD，本仓库在 `docs/wholebodyx-blog-reference.json` 保存了所读文件的 SHA-256，用于识别具体源码快照。它目前是**固定基座运动学验证框架**，不能把其中的 H1 展示解释为动态行走或平衡验证。

| 源文件与接口 | 实际职责 | 输出/限制 |
| --- | --- | --- |
| `mpc.py: JointMPC.solve` | 有限时域关节速度 QP | `JointPlan` 包含 N+1 个位置、N 个速度、起始时间及步长 |
| `runtime.py: MPCReferenceManager.update` | 校验参考并决定重规划或复用 | 当前参考位置、速度、原因；失败清除旧计划 |
| `tasks.py: PostureTask.linearize` | 位置误差反馈与速度前馈 | 实际是速度目标：`gain*(qref-q)+vref`，不是加速度任务 |
| `wbc.py: KinematicWBC.step` | 加权速度级最小二乘与硬关节边界 | `VelocityCommand`，不输出力矩或接触力 |
| `types.py: JointLimits.velocity_bounds` | 组合速度、下一步位置及速度变化率限制 | 在积分模型与离散步长下约束指令 |
| `simulation.py: JointIntegrator` | 执行速度积分 | 无质量、惯量、碰撞和接触动力学 |

WholeBodyX 的预测模型为：

$$
q_{k+1}=q_k+\Delta t\,u_k,\qquad u_k=\dot q_k.
$$

代价对未来 $q_{1:N}$ 跟踪关节目标，并惩罚速度；终端节点使用 terminal weight。约束包括关节位置、速度，以及可选的 $|u_k-u_{k-1}|\le a_{\max}\Delta t$。首个速度差相对当前测量速度计算。速度变化率限制不等于浮动基动力学或力矩可行性。

速度级 WBC 的任务为 $J_i(q)u\approx b_i$，加权残差与正则、平滑项进入目标。它同时限制：

$$
\max\left(-v_{\max},\frac{q_{\min}-q}{\Delta t},v-a_{\max}\Delta t\right)
\le u\le
\min\left(v_{\max},\frac{q_{\max}-q}{\Delta t},v+a_{\max}\Delta t\right).
$$

若没有配置加速度限值，去掉对应两项。这里使用逐元素上下界，只对当前固定基座积分模型成立。它与第 7 节中同时求 $\dot v,\lambda,\tau$ 的动力学 WBC 是不同层级的模型。

参考管理器会检查起点、时间戳、有效期、积分关系和限位。默认每 0.1 s 周期重规划，目标改变或关节位置预测偏差超过 0.03 rad 可提前触发；控制步长由调用方提供。刷新失败时返回失败并清除旧参考，而不是自动继续执行旧计划；真机停机/降级仍由硬件运行层负责。该同步调度没有硬实时保证。

## 9. 可运行 Python 对照：相同 WBC，不同参考 {#run-control}

下载 [wholebodyx_demo.py](wholebodyx_demo.py)。两组都使用同样的六关节模型、初始状态、目标、`PostureTask` 和 WBC；一组直接跟踪目标，另一组跟踪 MPC 管理器提供的位置与速度参考。这避免把不同任务配置的差异误算成 MPC 收益。

先安装你自己的 WholeBodyX 源码，以下路径是本次参考目录，应按实际位置替换：

```bash
python3 -m venv .venv-control
source .venv-control/bin/activate
python -m pip install /home/ubuntu/workspace/chase/WholeBodyX
python -B wholebodyx_demo.py --output results-control
```

安装后，示例不需要 PyTorch、预训练模型、URDF 或可视化组件；核心依赖是 NumPy、SciPy 和 OSQP。本文实际使用 WholeBodyX 已有环境运行，版本为 NumPy 2.3.5、SciPy 1.17.1、OSQP 1.1.3；新建环境若解析到其他依赖版本，应记录后再比较。

250 个 0.02 s 控制步骤后，两组关节误差二范数分别约为 $7.18\times10^{-10}$ rad 与 $5.03\times10^{-16}$ rad，初始均为 0.52915 rad。MPC 组实际规划 50 次、复用参考 200 次。可下载[运行报告](assets/report.json)、[直接 WBC CSV](assets/wbc.csv) 与 [MPC→WBC CSV](assets/mpc-wbc.csv)。

脚本逐步检查积分关系和执行速度的组合边界，违约容差为 $10^{-6}$；运行中的最大速度边界违约分别约为 $3.07\times10^{-11}$ 和 $1.11\times10^{-9}$ rad/s。这是数值容差内的运动学验证，不是两种方案速度、鲁棒性或真机安全性的排行榜。简单可达的姿态目标本来就不要求 MPC 才能完成。

如果使用仓库现有 `h1_mpc_wbc.py`，还需要相应 URDF 与可选依赖。其固定骨盆原地踏步、运动学支撑脚对齐和虚拟物体附着，不等于浮动基动力学、刚性抓取接触或物理碰撞仿真。

## 10. 与策略学习怎样组合 {#learning-control}

PPO、DPO、GRPO 是训练目标或训练方法，MPC/WBC 是在线规划控制模块，两者不是互斥选项。策略可以输出速度命令、目标姿态、任务参考或有界残差，再由控制器处理当前模型中的约束；也可以用优化控制器产生示范，训练策略近似其行为。

若策略动作先经过 WBC/QP 修正，训练时应保存**原始采样动作及其概率**，同时单独记录实际执行命令；不能拿修正后的命令去匹配原动作分布的 old logp。把控制器作为环境的一部分时，回报应来自整个组合闭环。训练与部署的控制器、限幅和观测协议发生变化，就可能引入分布偏移。更详细的训练衔接见 [PPO、DPO 与 GRPO 的控制章节](../../ai/ppo-dpo-grpo/#mpc-wbc)。

概念结构亦参考读者提供的[《从 MPC 到 WBC：人形机器人如何规划并执行全身运动？》](https://zhuanlan.zhihu.com/p/2080588693757224044)正文；具体 WholeBodyX 能力以所核对源码为准。进一步研究全身最优控制与反馈增益，可阅读 [Crocoddyl 官方项目](https://github.com/loco-3d/crocoddyl)。

## 阅读自测与验收

- 给每一条模块接口写清传递的是状态、轨迹、接触计划还是关节命令，并注明时间戳和坐标系。
- 模拟估计延迟、优化超时或接触变化，确认下层如何处理过期参考；模块频率高不等于闭环一定稳定。

<details>
<summary>展开核对：控制接口与验证边界</summary>

- MPC 参考应说明状态/控制含义、时间戳、有效期和坐标系；WBC 输出还需匹配低层伺服接口。
- 过期参考、求解失败和接触变化必须有运行层处理；平均求解速度无法代替时延与失败恢复检查。
- WholeBodyX 本文示例只验证固定基座积分关系和关节速度边界；动态平衡、接触摩擦和力矩可行性不在该模型中。

</details>
