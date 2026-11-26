---
title: "NeuralActuator 详解：可微仿真中的等效力矩、外力估计与数据复核"
date: 2026-11-25
lastmod: 2026-11-26
draft: false
authors: ["chase"]
math: true
tags: ["Robotics", "Dynamics", "NeuralActuator", "Force Estimation", "Simulation", "Transformer", "Teleoperation"]
categories: ["机器人技术"]
summary: "从论文 v2 和公开代码出发，拆解 NeuralActuator 的九帧输入、等效力矩、接触门控与可微训练，复核 NAD 数据划分、时间对齐和跨平台指标。"
description: "从论文 v2 和公开代码出发，拆解 NeuralActuator 的九帧输入、等效力矩、接触门控与可微训练，复核 NAD 数据划分、时间对齐和跨平台指标。"
showToc: true
TocOpen: true
comments: false
images: ["assets/contact-cover.webp"]
contentLanguage: "zh-CN"
reading_prerequisites: "刚体动力学、平移 Jacobian 与监督学习。正文从离散状态递推解释可微仿真"
reading_focus: "先区分物理关节力矩、仿真等效输入和外力，再沿训练、日志回放与在线推理核对信号来源和评估范围。"
related_posts:
  - "/posts/robotics/dynamics/parameter-identification"
  - "/posts/robotics/control/impedance-control"
  - "/posts/robotics/kinematics/jacobian"
  - "/posts/ai/transformer-attention"
---

同一台低成本机械臂在空载转动关节、夹住砝码或推动测力计时，舵机电流都可能增大。电流只反映末端受力吗？还会反映姿态、惯性、减速器摩擦、夹爪与机构阻力，以及内部位置环在做什么。NeuralActuator 用九帧位置命令、状态和舵机遥测预测能推进仿真的等效广义输入，同时估计环境在末端参考点施加的三维合力。前者从位姿轨迹反传学习，后者要用外力标签监督。[论文 v2，§III][paper]

默认的隐式耦合让等效输入独自推进仿真，因而可以吸收负载和模型误差。如果再把力头估计的同一负载施加到仿真，外力的作用就可能被计算两次。读实验时也要分清输入来源。日志回放会提供未来各步已记录的遥测，真机虚拟力传感器每次只看到当前及过去的测量，Franka 的离线实验还用到了五帧后的实测姿态。[论文 v2，§III-D、§IV-F 与附录 B][paper]

<figure class="article-figure" id="fig-contact-illustration">
  {{< post-image src="assets/contact-cover.webp" alt="桌面机械臂的两指夹爪接触固定测力台，关节电机和线缆可见" >}}
  <figcaption><span class="article-figure__text" style="grid-column: 1 / -1">机械臂推压场景（ImageGen 示意）。</span></figcaption>
</figure>

[单关节算例](#coupling-example)解释末端外力怎样产生关节力矩。[复现入口](#reproduction)给出运行命令，[数据划分复核](#split-audit)和[Franka 裁剪对照](#franka-mjx)解释样本与时间窗。[命令力矩对照](#franka-torque-probe)单独检查冻结力头对输入的响应。

## 阅读范围与版本 {#scope}

本文所用的 26 页 PDF 对应 **arXiv:2607.11734v2，2026-07-17**。下文的公式与罗马数字表号沿用该版，RSS 会议入口另列。代码和 Git 数据树固定在 [Dynamics-Modeling 的 `541a93a` 提交][commit]。README 的阿拉伯数字表号以及新增后端指标单独注明来源。[论文 v2][paper] · [RSS 2026 会议条目][rss] · [作者项目页][project]

Dynamics-Modeling 是作者的研究合集。这里跟踪 `NeuralActuator/` 下的数据、权重接口、训练和评估，并用仓库其他项目解释它在整体研究路线中的位置。

| 入口 | 负责的问题 |
| --- | --- |
| `NeuralActuator/` | 从命令、本体状态与电机遥测估计驱动和外力，刚体状态由仿真器推进 |
| `RigidFormer/`、`RigidFormer-v2/` | 学习刚体状态转移或将其用作机器人世界模型 |
| `HandDiffSim/`、`Hand_Robot_Control/` | 分别提供灵巧手的可微仿真，以及通信和关节控制示例 |
| `Penetration_Solving/`、`SimReady_Gate/` | 处理刚体场景的初始穿透和仿真就绪验证 |
| `DSAD/` | 用动态各向异性图元做视频降阶表达 |
{.table-readable}

NeuralActuator 保留质量矩阵、刚体几何和积分器，学习的是进入这些结构的控制量。与之相邻的[机器人动力学参数辨识]({{< relref "/posts/robotics/dynamics/parameter-identification" >}})从回归矩阵估计惯量、摩擦等有物理含义的参数，因此两种方法的验收对象不同。[固定版本仓库目录][repo]

## 为什么位姿能监督推进，却辨不出物理力矩 {#effort-identifiability}

令 $f_{\mathrm{ext}}$ 为**环境对机器人**的力，在基座系表达。定义 $J_v$ 为同一坐标系、同一力参考点的平移 Jacobian。对手臂关节，论文式 (4) 可写为

$$
\tau_{\mathrm{ID}}=M(q)\ddot q+C(q,\dot q)\dot q+g_{\mathrm{grav}}(q)
$$

$$
\tau_{\mathrm{ID}}=\tau_{\mathrm{act}}+J_v(q)^\top f_{\mathrm{ext}} .
$$

第二式来自虚功等价。末端速度为 $J_v\dot q$，力的功率为 $f_{\mathrm{ext}}^\top J_v\dot q$，对应关节广义力就是 $J_v^\top f_{\mathrm{ext}}$。固定砝码对机械臂向下施力，传感器若定义为“机器人对环境”，在转换到基座系的同时要换符号。若存在绕参考点的自由力偶，还应加 $J_\omega^\top m$。末端单点的三维合力模型把该项省略。两个大小相等、方向相反且不共线的接触力可以合成零力和非零力偶，因此三维输出无法表示那种接触。[论文 v2，§III-B][paper]

### 传统残差为什么会把摩擦当成接触 {#classical-residual}

传统方法先通过逆动力学或动量观测器求外部关节力矩，再从 $J_v^\top$ 反解末端力。若把关节侧驱动近似成 $K_t i+b$，力矩残差就会同时包含逆动力学误差、滤波/求导误差和电流—力矩拟合误差。SO-101 发布配方甚至使用带符号的负载寄存器，它的计数值不能直接代入安培单位的公式。

误差还会经过姿态相关的放大。设 $J_v^\top$ 的三个奇异值为 0.30、0.10、0.01 m。在对应奇异方向上，同样的 0.10 N·m 关节残差会分别变成约 0.33、1 和 10 N 的力误差。阻尼伪逆能压低小奇异值方向的放大，也会衰减该方向的真实力。神经网络力头避开每帧显式求伪逆，仍受训练数据的激励和传感信号可区分性约束。关于参考点和平移 Jacobian，可对照[机械臂 Jacobian]({{< relref "/posts/robotics/kinematics/jacobian" >}})。[论文 v2，§IV-D][paper]

### 隐式和显式耦合的单关节算例 {#coupling-example}

对任意三维扰动 $\Delta f$，若将力与关节驱动改为

$$
f'=f+\Delta f,\qquad
\tau'=\tau-J_v^\top\Delta f,
$$

同一构型下的总广义输入保持不变。即使有完美位姿和已知刚体惯量，位姿损失也只能约束总效果。带标签的力头能约束 $f$，但默认的隐式仿真不从等效输入里扣掉这个 $J_v^\top f$ 分量。

<figure class="article-figure" id="fig-joint-load-illustration">
  {{< post-image src="assets/single-joint-load-illustration.webp" alt="水平杆左端铰接在支架，右端通过竖直细绳悬挂砝码" >}}
  <figcaption><span class="article-figure__text" style="grid-column: 1 / -1">单关节杆端载荷（ImageGen 示意）。</span></figcaption>
</figure>

取图示水平构型作为起点，逆时针转动为正。设等效惯量为 0.20 kg·m²，连杆自身的重力和偏置已从方程两侧扣除。关节驱动为 3 N·m，算例把杆端竖直力固定为 −4 N。该姿态下竖直 Jacobian 分量为 0.25 m/rad，因此外力贡献 −1 N·m。以静止为起点，用 20 ms 半隐式 Euler 步推进，先更新速度，再用更新后的速度更新角度。下表另用 MuJoCo 3.11.0 的单轴模型复算，点力到关节的映射与 Jacobian 转置结果一致。

| 模式 | 送入驱动端 | 单独注入力 | 总力矩 | 一步角位移 |
| --- | ---: | ---: | ---: | ---: |
| 隐式拟合 | 2 N·m | 0 | 2 N·m | 0.004 rad |
| 显式分账 | 3 N·m | −4 N | 2 N·m | 0.004 rad |
| 给隐式输入再加力 | 2 N·m | −4 N | 1 N·m | 0.002 rad |
{.table-readable}

假如显式仿真把力臂错写成 0.20 m，真实的 −1 N·m 会被建模成 −0.8 N·m。等效头可以输出 2.8 N·m 补偿这 0.2 N·m 的建模差，同时重现角位移。即使显式模型有精确的力标签，低位姿误差也仍容许这种错配。将力从抓点 $p$ 移到刚体质心 $c$ 时，应补上力矩 $(p-c)\times f$ 来保持对关节的作用。固定 OMX 模型定义了独立的 `end_effector_target`，惯性原点就在该施力参考点。将实现移植到其他 MJCF 时，应检查该约定是否仍成立。[论文 v2，§III-B、§IV-L][paper] · [MJX 施力分支][omx-coupling] · [OMX 末端参考体][omx-xml] · [MuJoCo 力与力矩接口][apply-ft]

同一问题可在公开 OMX 模型上复算。取 +X 推力测试的零起始第 250 行，时间约 4.270 s，力向量的 X/Z 分量分别为 −2.411/−0.854 N。在 `end_effector_target` 施这组力时，J2 的广义外力约 +0.217 N·m。若原样把力移到 `link5` 质心，J2 会变为 −0.024 N·m。向质心补上力臂叉乘项后，MuJoCo 的 `mj_applyFT` 与抓点的 Jacobian 转置给出同一组关节广义力。这个对照固定了姿态和标签，只改变施力点。[复算用 CSV][omx-force-csv] · [OMX 末链接和参考点][omx-xml] · [MuJoCo 力映射接口][apply-ft]

### 直接预测和残差预测是另一条设计轴 {#parameterization}

直接预测把网络输出当作等效控制量。残差预测先计算基线 $K_t i$，再由网络加修正。两个参数化都可以选隐式或显式耦合。OMX 公开实现只在四个转动关节使用残差基线，先把 mA 除以 1000 换成 A，再乘 $1.3\ \mathrm{N\,m/A}$。夹爪的仿真通道仍直接预测滑块广义力。这个固定基线可以帮助起步，但当基线的偏差随方向、负载或温度变化时，网络需要连同偏差一并修正。论文表 XIV 的收益应限定在比较的任务与配方内。[论文 v2，§III-B、§IV-K][paper] · [残差与限幅实现][omx-coupling]

## 从九帧输入到四项输出 {#network}

编码器窗口包含过去八帧与当前帧。60 Hz 下，最旧到当前样本的时间差约为 $8/60=133$ ms。注意力可以在这九个已收集的帧之间双向聚合，只要每帧特征在当时已知，整次预测就能在时刻 $t$ 因果完成。Franka 离线构造目标时使用了未来状态，下文单独核对。显式时间窗也决定了模型对长期状态的依赖方式。慢变化温度可以作为当前遥测进入，超过历史窗的滞回则需靠现有状态或其他信号间接判断。

| 平台 | 每帧输入 | 用作 effort 的字段 | 动力学头送到哪里 |
| --- | --- | --- | --- |
| OpenManipulator-X | 36 维 | `current1–5`，mA | 四路转动轴加一路滑动夹爪，单位分别为 N·m 和 N |
| SO-101 | 42 维 | `load1–6`，有符号寄存器计数 | 六路转动轴，包含转动夹爪 |
| Franka Panda | 52 维 | `tau_d1–7`，控制器命令力矩 | 原版仿真位置执行器的控制口，夹爪宽度回放 |
{.table-readable}

**OMX。** 36 维按源码顺序是 5 个目标角、4 个臂关节角、1 个夹爪坐标、各 5 个电流/速度/电压/温度、1 个目标夹爪坐标，以及 4 个手臂和 1 个夹爪跟踪误差。CSV 还记录 PWM，网络输入未选这一列。[数据字段说明][dataset] · [OMX 特征构造][omx-features]

**SO-101。** 六个关节各有七类量，夹爪和其余关节统一处理。输入速度沿用编码器步数每秒，仿真得到弧度每秒后乘每弧度的步数才送进网络。电压也按原始十分之一伏记录，例如 122 表示 12.2 V。[SO-101 特征构造][so-features] · [SO-101 仿真状态回写][so-sim-features]

**Franka。** 52 维含连杆侧与电机侧状态。输入选择控制器命令力矩 `tau_d`，CSV 里的实测关节力矩 `tau` 与外力矩估计 `tau_ext` 未进入公开配方。发布权重也没有 OMX 的输入均值方差。Panda XML 将第一个臂关节的位置误差乘 4,500、关节速度乘 −450，相加后限力。代码中名叫 `tau` 的预测量进入这个位置控制口。遥测输入 `tau_d`、模型输出的仿真控制量与 XML 最终计算的驱动力矩各有自己的单位和用途。[Franka 特征构造][franka-features] · [Franka 执行器 XML][franka-xml]

用同一 MJCF 做原地前向计算。J1 在零位静止时，给 `ctrl=0.01` rad，执行器力矩为 +45 N·m。保持同一命令，关节若静止于 0.02 rad，力矩变为 −45 N·m。回到零位并把命令改成 0.10 rad，线性项为 450 N·m，限幅后为 +87 N·m。三组数都取自仿真的 `qfrc_actuator`，读发布模型的动力学输出时要连同位置误差、速度与限幅一起看。[写入 Panda 控制量的代码][franka-eval-control] · [Panda 执行器 XML][franka-xml]

四层预归一化编码器的隐藏宽度为 192，每层四个注意力头，各头宽 48，FFN 中间宽度 384。注意力层将 Query 投影扩成 384 维，拆成 Query 和同形状的门控 logits。聚合的每个输出分量先乘一个 sigmoid，再经输出投影。编码器的 FFN 使用 GELU，末层归一化后对九帧取平均，输出头用 SiLU。更一般的投影和缩放推导可参照[Transformer 注意力详解]({{< relref "/posts/ai/transformer-attention" >}})。[注意力实现][attention]

以 OMX 为例，输入张量为“批 × 9 × 36”，四头的注意力分数为“批 × 4 × 9 × 9”。每帧投影后加位置向量。类注释仍写着正弦位置编码，前向实际调用 `LearnablePositionalEncoding`，从 16 个参数槽取前九个。历史由调用者维护，每次 Transformer 前向重新计算这九帧。[编码与输出头][transformer]

<figure class="article-figure" id="fig-actuator-flow">
  {{< post-image src="assets/neuralactuator-flow.webp" alt="九帧命令、状态与遥测进入 Transformer，等效驱动推进下一步位姿，独立的外力与门控相乘，机构分数单独输出" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">默认隐式耦合中，等效驱动推进仿真，外力可以送给上层控制。</span></figcaption>
</figure>

输出张量可以分成四项。

- **等效驱动**。一个向量为每个执行器通道供值，之后由仿真适配器解释、限幅或转换。
- **原始外力**。三个基座系分量，单位 N。
- **接触门控**。一个标量 $g$，最终输出 $\hat f=g\hat f_{\mathrm{raw}}$。它与上文注意力逐分量的门、重力项都不同。训练使用标签范数大于 0.01 N 的二分类目标，这个值用来区分标签零与非零，实验检测下限另测。
- **机构状态**。每个通道一个 sigmoid 分数。OMX 受限实验只监督 J3，正常标签为 1，橡皮筋增阻标签为 0。报告受限类时读 $1-c_3$。其他通道有数值输出，缺少该实验下的监督及概率校准。

力回归的梯度经乘积同时走向 raw force 与 gate。当 $g$ 接近零时，raw force 收到的回归梯度会被缩小。只凭乘积，较大的原始力与较小的门值可以给出类似最终力。门的 BCE 单独约束接触状态。隐式耦合下，位姿损失仍会更新共享编码器，但在单次前向的计算图里，不会沿“力头输出 → 施力 → 下一状态”的直接路径走。[论文 v2，式 (6)–(10)、(16)–(19)][paper] · [共享力特征和门控读出][transformer]

## 可微训练与三种时间协议 {#training-and-causality}

### 一次训练步到底用了哪一行 CSV {#frame-alignment}

以 OMX 的一个训练窗为例。用训练窗起点的实测位置初始化仿真，历史从同一轨迹的前八帧填充，缺帧在归一化特征空间补零。每一步保留第 $t$ 行的命令、电流、电压和温度，用仿真的位置、速度与夹爪坐标覆盖状态字段，再重新算跟踪误差。网络输出限幅后的控制，经过多个物理子步得到下一状态。训练以第 $t+1$ 行的位置作为目标，力、门和机构状态标签也移了一行。窗口抽样按轨迹边界进行，避免把前一条轨迹尾部拼给后一条开头。[批次切片][omx-batches] · [特征覆盖和仿真步][omx-coupling]

| 时序 | 来自当前 $t$ | 参考标签 | 怎么解释 |
| --- | --- | --- | --- |
| 训练 rollout | 日志命令/遥测与仿真状态 | $q_{t+1}$ 与 $f_{t+1}$ 等 | 用当前读数推进下一状态。辅助力标签也偏移一帧 |
| 离线动力学评估 | 日志命令/遥测与仿真状态 | 状态按下一步对齐，力按 $f_t$ 计 | 重放这段已执行轨迹的遥测，允许观察漂移 |
| 虚拟力传感器 | 日志实测状态，或现场已到达的状态与遥测 | 离线可对 $f_t$，现场看当前输出 | 跳过物理步进，用观测推进历史缓冲 |
{.table-readable}

OMX 配置步长 17 ms，故训练与独立评估的力标签相差约一采样步。单独在接触沿附近检查错位比依赖整段 MAE 更有意义，持续数秒的静态保持对一帧偏差并不敏感。它是脚本索引差异，不能拿来推断传感器具有固定 17 ms 的物理延迟。[论文 v2，附录 B][paper] · [训练标签切片][omx-batches] · [离线评估对齐][omx-eval] · [`--force_only` 路径][infer-force]

OMX 与 SO-101 用检查点的均值和标准差对每帧做标准化，再按分量限到 ±10。这个步骤发生在仿真状态覆盖和误差重算之后。序列开始时用零填历史，零是标准化坐标里的值。投影偏置和位置向量随后仍会作用，填充帧也会参与注意力与平均池化，所以启动的前八帧应单独统计，尤其是在带载上电或读取窗口刚恢复时。[预处理和历史缓冲][infer-force] · [位置编码与池化][transformer]

训练初始化还会影响复现。OMX 从窗起点的日志位置与前向差分速度起步，速度的分母是该条轨迹的平均采样间隔。历史使用同一条轨迹中严格早于起点的帧。采样器默认先等概率抽合法轨迹，再在该轨迹的合法区间内抽起点，所以长轨迹按帧摊到的采样概率较低。运行评估会从轨迹头初始化，速度置零。比较两类损失时需记录初始速度和历史，而不能只比较积分步数。[训练窗抽样][omx-sampler] · [训练窗初始化][omx-init] · [评估初始化][omx-eval-init]

固定提交还存在一处**验证初始化的索引错配**。OMX 和 SO-101 的训练器从验证数组切出命令、遥测和目标，`rollout_single` 却总用训练数组取起始位置、速度和历史。例如 OMX 六方向配置中，验证第一文件的第 100 行若被抽为起点，同一索引在训练数组的 J4 与该验证行相差约 2.46°，随机扰动尚未叠加。此问题影响训练循环内的验证损失及基于它的选权重。独立的 `evaluate_actuator*.py` 有自己的初始化路径，不能用这一发现直接改写论文数值。[验证批次][omx-val-batch] · [训练数组绑定][omx-arrays] · [OMX 初始化][omx-init] · [SO-101 初始化][so-init]

### 多步反传需要把历史也放进状态 {#bptt}

把仿真状态和历史缓冲合写为 $z_t$，日志提供的外生输入写成 $u_t$，一步递推为 $z_{t+1}=G_\theta(z_t,u_t)$。链式法则给出

$$
D_{t+1}=\partial_zG_\theta\,D_t+\partial_\theta G_\theta,
\quad D_t=\frac{\mathrm{d}z_t}{\mathrm{d}\theta}.
$$

在此形态下，位姿损失可从未来几步反传到之前的驱动预测。等效控制当步若命中硬限幅，控制沿已饱和分量的局部梯度会消失，后面的状态和其他样本仍能产生梯度。因此要和 MAE 一起看饱和频度与控制范围。OMX 的四个臂关节在训练代码里限到 ±5 N·m，夹爪用独立阈值。它们是仿真输入的数值限制，不能替代真机的电流、速度、温度和机械限制。[论文 v2，式 (11)–(13)、表 II][paper] · [限幅、施力和子步][omx-coupling]

主线训练使用 JAX/Flax、MuJoCo MJX 和 AdamW。OMX/SO-101 的主要长窗配方从 128 步增至 256、320 步，Franka 固定 128 步。以配置的采样周期和子步数量算，三个平台的内部步长分别约 4.25、3.21、2.00 ms。训练窗、Transformer 九帧感受野、评估 100/300/500/600 步是三件事。论文表 II 的参数梯度余弦相对 128 步至少为 0.96，输入梯度范数从 64 步的 $2.73\times10^{-2}$ 降到 500 步的 $9.37\times10^{-3}$，这是该设置下的观察值。未分块的整段长时域优化仍需检查积分、接触切换和梯度稳定性。[论文 v2，§III-D、表 II 与附录 B][paper] · [主线配置][config-dir]

令 $r_k$ 为门控力第 $k$ 维与配对标签之差，$v_k$ 表示该力分量是否有效。启用掩码时，一步的力损失是

$$
\ell_f=\frac{w}{3}\sum_{k=1}^3v_k\,\ell_\beta(r_k) .
$$

Smooth L1 在残差绝对值小于阈值时用 $r_k^2/(2\beta)$，达到阈值后用 $|r_k|-\beta/2$。$w$ 是按标签是否非零选取的固定样本权重。OMX 定向力配方用 $\beta=0.15$，非零标签权重为 3，非掩码模式相当于三个 $v_k$ 全为 1。例如仅 $z$ 有效且其 Smooth L1 代价为 0.9，乘样本权重前，三维均值项是 0.3。按有效通道数归一化会得到 0.9，需要重新校准损失权重。SO-101 的 gate 仍会将无有效力的零填充帧当作无接触标签。Franka 的训练分支独立监督 $z$ 轴，指标仍可按三分量统计。[论文 v2，式 (16)–(19)][paper] · [OMX 损失][omx-loss] · [定向力配置][omx-config] · [SO-101 训练器][so-train] · [Franka 训练器][franka-train]

### 日志回放、因果估计和反事实规划 {#counterfactual}

读到“600 步约 10 秒”时，要问这 10 秒内每一步的电流从哪里来。公开评估用这一条真实轨迹已测到的电流或负载，仿真负责产生下一状态。换一个候选命令时，未来驱动板的电流、限流、摩擦与接触响应会跟着变。要规划多条候选控制序列，需要另建因果的命令—遥测响应模型，或换一个只依赖规划时已知变量的动力学函数。复制原轨迹的遥测只给出了条件回放。[论文 v2，§III-D、§IV-A、§V][paper]

真机力估计可以按“读当前状态和遥测 → 加入历史 → 计算 $\hat f_t$ → 上层生成下一控制指令”的顺序运行。这里测得的是**刚执行的控制所引起的遥测**，还需记录发送时钟、传感器采样时钟与主机接收时钟。论文的 Franka 离线目标用

$$
\tilde q_t^{\mathrm{cmd}}
=1.03\,q^{\mathrm{rec}}_{\min(t+5,\,T-1)} .
$$

五帧在 62.5 Hz 下是 80 ms。公开函数先按比例裁剪轨迹，再将末尾索引限制在裁剪后最后一帧。现场接口应提供截至 $t$ 已知的控制器目标。直接把 CSV 稀疏的 `cmd_pos` 代入已按代理目标训练的权重，会同时改变目标通道与误差通道，应单独量化输入分布变化。[论文 v2，§IV-F 与附录 B][paper] · [Franka 裁剪和未来代理][franka-load]

## NAD 的标签、单位和划分 {#nad}

### 先分清观测到了什么 {#labels}

NAD 采集用同构 leader/follower 双臂。人操纵 leader，follower 跟随目标并和环境接触。OMX 的三轴力标签源于固定设施上的六轴传感器，匹配轨迹与传感器时间后把力转到机器人基座系。三个力矩通道不进入模型监督。相邻的无接触参考保持类似运动，帮助模型区分运动耗力和外部负载。力计推压是另一套评估，标量读数只映射到已知接触轴，其余两个参考分量设为零。[论文 v2，§III-C、§IV-B 与附录 A][paper] · [采集说明][teleop]

<figure class="article-figure" id="fig-teleoperation-illustration">
  {{< post-image src="assets/nad-teleoperation-illustration.webp" alt="左侧一只手引导主臂，右侧同构从臂的夹爪压在固定测力台上" >}}
  <figcaption><span class="article-figure__text" style="grid-column: 1 / -1">主从双臂采集场景（ImageGen 示意）。</span></figcaption>
</figure>

**OMX 的公开采集入口。** 主循环用 ROS 时间给电机状态记相对时间。六轴传感器另有读取线程，去零后的三路力保存在加锁的最新值里，入行时先交换 X/Y 再将三轴取负。缓存中只有力值，因此仅凭这个采集入口生成的行时间，无法算出该行传感器样本的实际采样年龄。自己复做推压时，可对力帧单独记录采样时间，并用已知方向确认基座轴和符号。[电机状态与行时间][omx-teleop-row] · [力缓存与轴变换][omx-teleop-force]

以 CSV 里的带符号分量判定力的方向。在六条定向推力测试记录中，每条取力范数最大的约 20% 行，再逐轴取中位数。各记录绝对值最大的分量都落在文件名所指的轴上，其中 `force_x_plus` 的 X 分量中位数为 −3.16 N，`force_z_plus` 的 Z 分量中位数为 −5.22 N。这些记录还有侧向分量，应保留三轴标签来评分。[六方向测试文件][omx-force-tree]

**SO-101 的公开采集入口。** 同一寄存器用总线同步读各电机，不同寄存器仍分开请求。脚本在传感器和位置读完后统一给整行记主机时间。某类读失败时，代码会打印告警并为该组填零，CSV 的零值本身不区分正常读数与这类失败。力列初始全为 `-999`。抓放的标注脚本按手选时间段填常量，静态保持的标注入口可以整列填常量。[寄存器读取][so-teleop-registers] · [行时间与数据列][so-teleop-row] · [SO-101 标注流程][so-teleop-labels]

已知重量标签采用基座系向下的 $-mg$。以 200 g 为例，参考约为 −1.96 N。静态持有时该值代表名义外力，起升加速时实际竖直反力还包含 $ma_z$。例如物体随夹爪向上加速 2 m/s² 时，物体受到机器人向上约 2.36 N，机器人受到相反的约 −2.36 N，标签少了约 0.40 N 的幅值。无侧向传感器时，不能用这个标签检验真实侧向力。[论文 v2，§III-C][paper]

`-999` 是数据哨兵，不能当作 −999 N。定向推力 CSV 的三个轴都可以有效。空载参考可能三轴都记哨兵，而载荷型 CSV 常只有夹持期间的 $z$ 有效。未启用掩码的配方会将哨兵转成零并参与训练。启用掩码的配方只让有效分量进入力回归，门控分类另走自己的标签路径。要把“缺测去掉”和“无接触的已知零”在应用自己的数据时明确区分。[数据字段与哨兵][dataset] · [OMX loader 的有效掩码][omx-loader] · [力损失及门控][omx-loss]

### 夹爪角度、开口和仿真滑块 {#gripper}

论文式 (20) 从对称曲柄滑块机构做几何投影。曲柄沿滑轴的分量是 $r\cos\alpha$，垂直分量是 $r\sin\alpha$。长度为 $l$ 的连杆再贡献滑轴投影。双指是两倍单侧位移，减去两侧的内表面偏置。

$$
a(\alpha)
=2\bigl(\sqrt{l^2-r^2\sin^2\alpha}+r\cos\alpha-\delta\bigr).
$$

在理想准静态机构中，每个卡爪力的幅值随几何变化。也可以从虚功得到 $|\tau_m|=F_{\mathrm{jaw}}\,|\mathrm da/\mathrm d\alpha|$，其中 $F_{\mathrm{jaw}}$ 是每指对称夹持的力。接近死点时导数小，理想传动增益很高，实际摩擦、游隙和弹性会限制它。这个力与三维外力头所输出的末端合力含义不同。

公开数据文档把 `aperture` 写成开口宽度。训练的夹爪位置项和评估的 `Grip` 都使用已标定的单指滑块坐标。单指坐标可带零位偏移并取负值。核对 CSV 时应查看实际数值和 XML 滑动关节范围。把字段再除以二、换符号或和两指开口直接比较都会改变监督。Franka 独立使用归一化的 `gripper_width` 乘 0.04 m 映射到仿真指端行程。[论文 v2，§III-E、§IV-A][paper] · [OMX 特征与夹爪][omx-features] · [Franka 夹爪变换][franka-features]

### 公布的划分能支持多强的泛化结论 {#split-audit}

论文附录 A 的 OMX 和 SO-101 划分按任务保存八条训练、一条验证和一条测试。几种工况仍是同命令族内的留出重复，不能代替整类动作、换设备、换传动或更换负载范围的留出。表 A16 是论文任务分配口径，包含 100 g 的两种动作。固定 Git 树的 OMX `weight/` 则把两行留给无负载参考，带负载文件覆盖 200–500 g。引用统计时须指明用的是论文任务份数、仓库路径数还是实际独立采集数。[论文 v2，附录 A][paper] · [发布目录说明][dataset]

本文逐条核对固定 Git 树。OMX 350 条 CSV 路径分布在 10 个空载、12 个定向力与匹配参考、9 个负载和 4 个机构状态目录。SO-101 有 100 条路径，Franka 有 35 条。OMX 空载抓放的十个文件在 `force_unlabeled/` 与 `motor_condition/` 各有一份，对应文件处于相同划分且字节相同。[仓库数据树][data-tree]

论文还列出正常机构的 200 g 任务复用。相应的 `weight/` 与 `motor_condition/` 文件哈希不同。我用五个位置和五个电流分量定位，再逐列比较，发现十个短文件都完整落在相应长文件的连续区间，时间原点统一后其余量仅有舍入差。扣除这两组已确认的复用，OMX 与 SO-101 的 450 条路径对应 430 条轨迹。如下图的测试对，697 行短文件对应长文件的零起始索引 480–1176，时间统一加约 8.205 s 后重合。[完整对齐结果](nad-audit.json)

<figure class="article-figure" id="fig-nad-overlap">
  {{< post-image src="assets/nad-overlap.webp" alt="OMX 的两个 200 g 测试文件在较长文件第 480 至 1176 行的 J3 电流完全重合，长文件还含前置和后置片段" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text">200 g 抓放测试记录的 J3 电流。时间对齐后的 697 行重合。</span></figcaption>
</figure>

**Franka 有一对跨划分的完全重复。** 按固定版本下载测试 `200_006.csv` 和训练 `200_007.csv`，两份均为 1,020,944 字节、929 个采样行，Git blob 和 SHA-256 相同，逐字节比较也通过。三个公开训练配置都把后者纳入训练，把前者作为 200 g 测试。为了检查改名或截取后留下的重复，我进一步比较了 30 条训练与 5 条测试记录的全部 150 对文件。在七路关节位置、七路命令力矩和七路电机位置上，要求数值相同并至少连续九行。满足条件的仍只有这一对，从行 0 起完整重合 929 行。该判据会漏过有数值扰动或重采样的片段。[测试文件][franka-duplicate-test] · [训练文件][franka-duplicate-train] · [全对比结果](nad-audit.json)

**验证用哪些轨迹，和权重怎样选，是另一项检查。** Franka 三阶段配置的 `val_datasets` 都逐项重复全部 30 条训练路径，每个测试重量都出现在训练中。两次微调还默认从上一阶段的 `_best_test.pkl` 起步。训练器周期性跑测试路径并按全轨迹最大关节 MAE 保存该文件，测试轨迹因此参与过选权重。[Franka 原始配方][franka-config] · [第一阶段微调][franka-ft1] · [第二阶段微调][franka-ft2] · [Franka 选权重代码][franka-best]

OMX 的 `weight_all` 两个微调配置也采用前一阶段的 `best_test`。后文另报去掉 200 g 后的公开评估均值，仍只描述这个固定检查点。设计独立验证时，应按原始采集分组，再划出从未参与选权重的整条盲测记录。[OMX 微调配方][weight-ft2]

### 从四个 CSV 样本看标签覆盖与时钟 {#log-samples}

下表按原始 CSV 行计数。有效力分量指原始值不等于 `-999`。采样间隔取相邻行的时间差，九帧跨度取最旧到当前之间八个间隔的总时间。两项都用日志的时间戳，仿真积分使用配置的固定时间步。[原始时间复核](nad-audit.json)

| 固定版本样本 | 行数 | 采样间隔中位数 | 九帧跨度中位数 | X / Y / Z 有效行数 |
| --- | ---: | ---: | ---: | ---: |
| [OMX 的 +X 推力][omx-force-csv] | 542 | 17.00 ms | 136.01 ms | 542 / 542 / 542 |
| [OMX 同命令无接触参考][omx-reference-csv] | 542 | 16.01 ms | 136.00 ms | 0 / 0 / 0 |
| [SO-101 的 300 g 抓放][so-csv] | 539 | 16.01 ms | 128.46 ms | 0 / 0 / 187 |
| [Franka 的 300 g 抬升][franka-csv]，裁剪前 | 941 | 15.98 ms | 127.96 ms | 0 / 0 / 941 |
{.table-readable}

OMX 的两条 +X 记录按行比较五个命令角，最大差在浮点舍入量级。无接触参考的单步中位数是 16.01 ms，实际九帧跨度中位数却是 136.00 ms。做重采样或叠加瞬态时，应使用各自的时间戳直接计算窗口。推力记录的最小三轴力范数约 0.045 N，已经高于 0.01 N 的训练门阈值。这段样本按标签阈值全为正类，实际接触切换和检测下限需独立参考。[推力和参考数据][omx-force-csv] · [门标签实现][omx-loss]

SO-101 的 300 g 记录在零起始索引 186–372 给出 −2.94 N。按训练的 $t+1$ 标签切片，第一帧正类会配给索引 185 的输入，最后一帧正类会配给索引 371。无接触参考取零的评估协议则在 186–372 比较对应的力输出。这种偏移很容易被长时间窗均值掩盖，检查力反馈延迟时应单列开始和解除两侧的误差。[SO-101 样本][so-csv] · [SO-101 配对源码][so-batches]

对这条记录，用 `so101_weight.pkl` 和 `so101_weight.yaml` 跑 `infer_actuator.py --robot so101 --force_only`，固定 538 个预测，只把参考从 $f_t$ 改成 $f_{t+1}$。第 185 行的竖直绝对误差从 2.896 N 变为 0.044 N，第 372 行从 0.597 N 变为 2.343 N。其余 536 个参考值相同，整段竖直 MAE 只从 1.026 N 变为 1.024 N。按绝对值的三角不等式，两处各 2.94 N 的标签变化对全段 MAE 的合计影响至多约 0.011 N。这项重算比较的是同一预测的标签索引，两个边界仍由载荷标签定义。[单条推理源码][infer-main] · [SO-101 300 g 记录][so-csv]

Franka 的 300 g 文件按 30% 开头、25% 末尾裁成 423 行。裁剪后计算限尾五帧代理，实测前视时间中位数约 79.98 ms，最后一行的前视为零。裁剪前 `cmd_pos1–7` 共有七组不同的向量，裁剪后只剩三组。这个原始字段与平滑代理代表不同的目标输入协议。[Franka 样本][franka-csv] · [代理变量生成][franka-load]

**发布配方的可用步数与表 IX 的两种时域也有区别。** 五个公开测试 CSV 裁剪后保留 418–423 行，下一状态评估可用 417–422 步。脚本遇到超出可用长度的窗口会跳过。因此直接用随附配置运行，就不会输出 500 步一列。引用论文 500 步结果时应保留论文口径。复现时还需要能生成相同长度的裁剪设置、时间起点或相应记录。[Franka 测试列表][franka-ft2] · [裁剪实现][franka-load] · [评估窗口跳过规则][franka-window]

## 论文实验的指标与证据 {#experiments}

### 位姿、力与帧率的分母 {#metrics}

表 III、IV、V 的关节 MAE 在每个关节单独统计，转动轴为度，OMX 的 `Grip` 为单指位移的毫米。以表 III 的 600 步为例，十个空载任务在 J1–J4 的均值为 3.1°、2.8°、3.2°、3.1°，夹爪列 0.2 mm。这些值汇总了从起点到截断时刻的误差，并未给出第 600 步的末帧误差。600 步仅约 10 秒，不能外推到更长窗口或未见任务。[论文 v2，§IV-A 与表 III][paper] · [OMX 轨迹前缀评估][omx-eval]

表中的三分量力 MAE 对每帧的绝对坐标误差取均值，然后对帧和轨迹聚合。误差若仅在 $z$ 上，$z$ 轴差 0.60 N 对表中 `F` 的贡献为 0.20 N。若另有横向误差，不能靠简单乘三得到竖直误差。三分量 MAE、轴向力 MAE、力向量模长偏差和接触沿延迟应分别报告。[论文 v2，§IV-A、表 IX][paper]

| 论文 v2 | 公布结果 | 测量范围 |
| --- | --- | --- |
| 表 IV，OMX 六方向传感器 | 实际接触轨迹的 500 步 F 为 0.38–0.57 N。无接触参考约 0.00–0.02 N | 六方向各自比较，F 为三分量平均。含无接触参考的总平均为 0.23 N |
| 表 V，OMX 已知负载 | 100/300/600 步的任务平均 F 为 0.12/0.08/0.11 N | 名义重力参照，含夹持和空中运动阶段 |
| 表 VI，OMX 力计 | 100/600 步的任务平均 F 为 0.08/0.10 N | 正面、顶部共六个位置/方向组合。标量力计映射至一条已知轴 |
| 表 VII，OMX 传统方法 | ID-Linear 1.41、ID-Friction 1.23、GMO 0.66、NeuralActuator 0.12 N | 同一负载基准的 100 步。基线逐步读取实测状态，本文方法使用仿真状态 |
| 表 IX，Franka | 五种负载平均 F 在 100/500 步为 0.34/0.28 N | 五帧前视代理。只有竖直维受监督，横向以零参考计入三维 MAE |
| 表 X，SO-101 | 500 步、六个任务的 F 为 0.47–0.73 N，最大关节格约 9.8° | 单独训练的 SO-101 模型，并行比较六个 300–500 g 任务 |
{.table-readable}

表 VII 的 5.5 倍来自 $0.66/0.12$，比较对象是论文在低成本电流协议下改写的 GMO。作者也指出，约 50 g（约 0.5 N）及以下接近这套低成本平台的噪声底。逆动力学基线和 GMO 都使用校准的电流—力矩近似，因此该结果反映整条协议的效果，无法单独归因于 Transformer，或用来给带关节力矩传感器的观测器排名。[论文 v2，§IV-D 与表 VII][paper]

### 消融先核对统计对象 {#ablations}

论文表 XIV 的**参数化**比较用完整轨迹。OMX 无夹爪任务从残差的平均/最差关节 0.49°/0.64° 降到直接预测的 0.30°/0.39°。SO-101 六任务的最差按关节汇总值几乎不变（1.53° 到 1.52°），F 从 0.38 N 到 0.22 N。这里的“最差关节”先按任务聚合再在关节之间取最大。

同一表的**耦合**比较则取最差的单个任务×关节格。OMX 隐式和显式分别为 0.88° 和 1.33°，F 为 0.06 和 0.08 N，均来自三种子实验。SO-101 为 2.50° 和 6.9°，F 为 0.22 和 0.33 N。SO-101 的隐式行采用一个发布权重，显式行是三次从头训练的汇总。该差异值得检查力臂、惯量、标定和学习难度，同时保留种子统计的不对称。SO-101 的 500 步表 X 与发布检查点的全程表都应按各自来源引用，不能只替换步数解释两表的数值差异。[论文 v2，§IV-K、§IV-L 与表 XIV][paper] · [公开 SO-101 结果][repo-so] · [公开耦合消融][repo-coupling]

在表 XV 的约 1.4M 参数架构比较中，Transformer 在六列中四列最优或并列，F 为 0.23 N。GRU 在 J2 和 J3 分别得到 2.08°、1.68°，优于 Transformer 的 3.31°、2.01°。表 A20 中 MLP 为约 1.14M，GRU/LSTM/Transformer 在 1.43–1.44M 左右，学习率也有差别。模型的选型证据来自整个配方和指标，位置精度优先且预算紧时可以保留 GRU 作为对照。[论文 v2，表 XV 与附录 D][paper]

### 适配、机构状态、模仿学习与延迟 {#downstream}

- **采集后适配。** 作者用 12 条新轨迹、7,810 帧微调 70 个 epoch，共 356 秒。图 11 的约 44% 是该进度曲线自己的聚合。表 VIII 单看 500 步的四个平均关节 MAE，先取四列均值则从约 2.13° 到 1.27°，下降约 41%。这是采集后更新上一版权重的时间，不应写成每次控制周期内在线反传。[论文 v2，§IV-E、图 11、表 VIII][paper]
- **J3 机构状态。** 表 XI 的准确率 91.0%、召回率 96.2%、AUC 0.95，针对同一测试组上设橡皮筋增阻与正常的二分类。网络按帧打分，手工特征基线每 64 步做一次、50% 重叠，训练轨迹数量也不同（32 对 16）。这能说明该设定下区分增阻的能力，故障类型与寿命估计仍需独立标签和测试。[论文 v2，§IV-G、表 XI][paper]
- **真实行为克隆。** 同样的八步控制器输入由关节与开口历史增至加入三维力后的历史，冻结 NeuralActuator，仅训练命令策略。按表注明的 40 次折算，抓放从 32/40 到 37/40，抬升保持从 34/40 到 38/40。400 g 未出现在 BC 示教负载中，但在力估计预训练范围内。500 g 在抓放示教中已有。观察到的收益是这些物体与任务上的点估计，论文未给配对成功/失败列联表。[论文 v2，§IV-I、表 XIII、附录 F][paper]
- **推理延迟。** 表 XII 的 0.25 ms 均值、0.31 ms P95 是 JIT 后 GPU 模型推理，1.44M 个 FP32 参数约 5.50 MiB。60 Hz 控制周期约 16.7 ms，预算还包括舵机总线、时间对齐、主机调度和控制输出。README 另报告热启动后的 CPU 前向约 1.6–2.6 ms，自己的控制器仍应端到端计时。[论文 v2，§IV-H、表 XII][paper] · [CPU 虚拟力传感器说明][repo-force]

发布版机构分类另有一套结果。用 `omx_motor_condition.pkl` 和 `motor_condition.yaml` 运行同一个 CPU MJX 评估器，J3 的四条测试轨迹共 5,062 个逐帧判定，其中正常 2,949、受限 2,113，均判对。这与仓库表 4 的 100% 一致。整条记录共用同一机构标签，5,062 帧只来自四条轨迹。代码还输出五个电机的统计，其中四个通道缺少机构状态监督，读结果时应取 `CLASSIFICATION_J3`。训练器的独立验证分支把所有验证 J3 标签初始化为正常，配置却含两条受限验证轨迹。独立评估有自己的标签逻辑，识别任务名中的 `degrade`。论文表 XI、训练时验证损失以及公开检查点的独立评估应按各自路径核对。[机构分类配置][condition-config] · [训练中验证标签][omx-val-labels] · [独立分类实现][omx-condition-eval] · [README 表 4][repo-condition-results] · [本次分类结果](omx-motor-eval.json)

论文的视觉监督实验用 VGGT 给出相机内参、SAM3 提取掩码，再联合微调外参和执行器参数。计算图从执行器经过仿真状态、连杆正运动学和可微渲染到轮廓损失。附录 A18 使用一条真实视频的六张非连续图像，初始、固定相机只优化机器人侧、联合优化的平均软 IoU 分别为 0.2589、0.2974、0.8515。机器人侧还可包含受二次项约束的逐帧修正。低轮廓误差允许外参与轨迹互相补偿。要证明执行器或外力估计受益，应保持独立外参和外力测量做验证。[论文 v2，§IV-J 与附录 C][paper]

## 从仓库入口读到实际复现 {#reproduction}

### 选择检查点、配置和状态来源 {#code-map}

| 目的 | 代码入口 | 先核对的条件 |
| --- | --- | --- |
| 训练 OMX / SO-101 / Franka | 对应的 `train_actuator_diffsim*.py` | CSV 单位、历史长度、关节顺序、训练与评估来源、恢复权重 |
| 评估论文式轨迹指标 | 对应的 `evaluate_actuator*.py` | 数据选择、100/300/500/600 或全轨迹、是否取 EMA |
| 单条轨迹滚动推进 | `infer_actuator.py` | 由仿真输出状态，命令和 effort 遥测来自 CSV |
| 虚拟力传感器回放 | `infer_actuator.py --force_only` | 状态来自 CSV，本地仅执行前向，保留门值和力 |
| 采集同构双臂数据 | `teleop/omx/` 和 `teleop/so101/` | 原始采样时间、基座变换、舵机单位、零载参考 |
{.table-readable}

开始复现时可以先选 OMX 的六方向模型，使用该模型官方配置和相应测试 CSV。下面在 `NeuralActuator/` 下调用作者脚本。它对 CSV 回放，不连接机器人。检查点是 Python pickle，使用作者发布的文件并校验固定版本的摘要。两种模式选择不同的输出文件名，方便并排比较。脚本的 `--use_ema` 会在检查点包含 EMA 时选择该权重。[发布权重][weights] · [配置][omx-config] · [参数与入口][infer-main]

```bash
python infer_actuator.py --robot omx \
  --checkpoint checkpoints/omx_force_sensor.pkl \
  --config configs/force_sensor.yaml \
  --csv data/force_sensor/force_x_plus/test/001.csv \
  --force_only --out outputs/force_x_plus_virtual.npz
```

去掉 `--force_only` 会使用仿真状态并多输出 `sim_q`，同时应给 `--out` 换名。这个命令行即使在 `--force_only` 下也加载 MJX 相关模块，并统一取前 `N−1` 行。做逐帧比较时要保留这个裁切。即使架构相同，也要保留与检查点对应的配置。相同机器人的配置还可能改变电流滤波、坐标范围、裁剪、力矩参数化或 rollout 时间步。Franka 还会改变代理前视等预处理。比较模型应把检查点的发布修订、权重键、配置文件、CSV 路径以及 MAE 的取样窗口一起记录。[推理实现][infer-main] · [Franka 配置目录][config-dir]

如果要复查显式耦合，训练器和评估入口需要配对。OMX 与 SO-101 的公开训练分支识别 `apply_external_force`，后者用抓点相对质心的叉乘项构造附加力矩。固定提交的常规 `evaluate_actuator*.py` 和 `infer_actuator.py` rollout 只写入控制向量，无对应的外力注入分支。即使把配置改成 true，这些入口仍按隐式路径推进。复核表 XIV 的显式行需要相同的施力点和评估实现，应先用已知外力对照实际的仿真更新。[SO-101 力矩平移][so-coupling] · [OMX 评估步][omx-eval-step] · [SO-101 评估步][so-eval-step]

### 用 MJX 复跑 OMX 和 SO-101 评估 {#cpu-mjx}

作者的 `evaluate_actuator.py` 可以在 JAX 的 CPU 后端运行。这里保留默认的 MJX 求解路径。脚本的 `--use_cpu` 则切到原生 MuJoCo 分支，该分支会拒绝自带特征归一化统计的检查点。[独立评估入口][omx-eval-main]

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=disable python -B evaluate_actuator.py \
  --model_path checkpoints/omx_force_sensor.pkl \
  --config configs/force_sensor.yaml \
  --output outputs/omx_force_sensor_cpu.json
```

我用原始权重、JAX 0.7.2 和 MuJoCo/MJX 3.11.0，在公开的六条定向接触和六条空载参考轨迹上复跑。完整轨迹的四关节 MAE 为 0.82°、1.00°、1.03°、0.95°，单指 MAE 为 0.09 mm。六条接触的三分量力 MAE 为 0.44–0.57 N，等权均值 0.494 N。四关节、夹爪和接触误差范围按两位小数与 README 表 2 一致。只取 500 步并对全部十二条等权时，三分量 MAE 为 0.244 N。与论文表 IV 的 0.23 N 一起引用时，应注明各自的权重和评估记录。[README 表 2][repo-results] · [本次评估输出](omx-mjx-eval.json)

同一条 +X 推力文件用单条推理和批量评估分别运行，541 步三分量 MAE 都约为 0.542 N。下一节直接读取实测状态，得到 0.523 N。两种结果对应不同的输入状态及其历史，比较两种推理模式应固定文件、检查点和时间范围。

在相同环境中，SO-101 用 `evaluate_actuator_so101.py`、`so101_weight.yaml` 和 `so101_weight.pkl` 复跑六个 300–500 g 任务。完整轨迹先按任务平均再在各关节取最大，结果为 1.52°。三分量力 MAE 的任务等权均值为 0.219 N，与 README 的 0.22 N 一致。单算 $z$ 轴为 0.315 N。同一权重的 500 步三分量均值为 0.226 N。论文表 X 的区间 0.47–0.73 N 来自论文评估，各处数字保留各自配方和聚合方式。[SO-101 评估配置][so-config] · [统计实现][so-metrics] · [本次 SO-101 结果](so101-mjx-eval.json)

### Franka 完整评估，先固定起点和分母 {#franka-mjx}

我在同一 JAX CPU 环境中用官方 `evaluate_actuator_franka.py` 复跑 Franka 五个公开任务，使用发布的第二阶段权重 `franka_lift_hold.pkl`。独立评估器读取 `use_ema_params`，而配置中的 `eval_ema_params` 给训练器使用。以下运行均保留独立评估默认的 `params` 权重、1.03 倍五帧前视、16 ms 外层步长和 2 ms 物理子步。[官方运行入口][franka-eval-run] · [选权重实现][franka-eval-weights] · [发布权重][franka-checkpoint]

1. **发布裁剪。** 去掉开头 30% 与末尾 25%，以保留段第一行的实测位置和零历史初始化。可评估 417–422 步。
2. **只延长终点。** 仅把 `trim_end` 改为零。起点和初始化方式相同，五帧代理在较晚的终点才限尾，可评估 650–658 步。
3. **保留整段。** 把两侧裁剪都设为零。仿真和前缀 MAE 从原始第 0 行开始，覆盖原配置排除的首尾段，共 928–940 步。

每次先按任务在时间窗内求 MAE，再对五个任务等权。下表的 $F$ 为三分量 MAE，$F_z$ 只比较竖直分量。500 步相当于仿真时间 8 s。官方 JSON 的 `AVERAGE.n_steps` 固定写成 1000，实际评估长度要取各任务的 `n_steps`。[单任务指标][franka-window] · [官方汇总代码][franka-aggregate] · [逐任务逐轴结果](franka-rollout-audit.json)

| 五条公开测试记录 | 100 步 $F$ | 500 步 $F$ | 全程 $F$ | 全程 $F_z$ |
| --- | ---: | ---: | ---: | ---: |
| 发布裁剪 | 0.487 N | — | 0.415 N | 0.465 N |
| 只延长终点 | 0.487 N | 0.578 N | 0.744 N | 1.281 N |
| 保留整段 | 1.297 N | 0.893 N | 0.901 N | 1.673 N |
{.table-readable}

发布裁剪的全程 $F$ 为 0.4146 N，保留两位小数与 README 的 0.41 N 一致。只延长终点时，直到短记录第一次触发五帧代理的限尾，两次运行的力和状态预测逐项相同，共 413–418 步。表 IX 报告 100/500 步 0.34/0.28 N。两种延长范围都给出了 500 步数据，仍与表 IX 有差距。只延长终点并改用同一检查点的 `ema_params` 后，100/500 步 $F$ 为 0.491/0.583 N。在这套裁剪和权重下，切换参数键对这两项的影响约为 0.004/0.005 N。[论文 v2，表 IX][paper] · [README 的 Franka 结果][repo-franka] · [逐窗口结果](franka-rollout-audit.json)

把开头也放回后，100 步的起点由原始约第 280 行移至第 0 行。将三组输出按原始行号重切到发布裁剪对应的区间后，排除重复的 200 g，从原始第 0 行推进的 $F_z$ 为 0.489 N，从裁剪起点重新初始化的 $F_z$ 为 0.492 N。下图可看到同一条记录中段的两条曲线很接近。评估初始化或仿真漂移的影响时，要在同一原始区间比较。[原始行区间对照](franka-rollout-audit.json)

<figure class="article-figure" id="fig-franka-crop-comparison">
  {{< post-image src="assets/franka-crop-comparison.webp" alt="Franka 300 g 记录的竖直估力。两条曲线在约 4.5 至 11.3 秒接近，剪掉的首尾有较大的名义重量偏差" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text">同一条 300 g 记录，阴影为发布裁剪区间。</span></figcaption>
</figure>

**用常量检查三分量排名。** 对等权的 30 条公开训练记录，名义 $f_z$ 的中位数 −3.924 N 最小化常量的训练 MAE。用 $[0,0,-3.924]$ N 作为每帧的固定输出，无需知道各条测试记录的质量。对 300–600 g 四条记录，用发布裁剪的相同帧与标签作比较。

| 四条等权，发布裁剪全程 | X 对零参照 | Y 对零参照 | 竖直 $F_z$ | 三分量 $F$ |
| --- | ---: | ---: | ---: | ---: |
| NeuralActuator + MJX | 0.229 N | 0.486 N | 0.492 N | 0.402 N |
| 400 g 常量 | 0 N | 0 N | 0.981 N | 0.327 N |
{.table-readable}

模型的竖直 MAE 约减半，三分量分数却是常量更低。常量在两个缺测的侧向轴直接输出零，单轴误差由原评估分母摊到三轴。按负载逐条看，裁剪段内 300、400、500、600 g 的平均竖直估力依次为 −2.859、−3.899、−4.954、−5.755 N，幅值随负载递增。把 600 g 的首尾放回后，平均值变为 −3.336 N。五条未裁剪记录的最小门值仍是 0.995，门控几乎全开。[力头与门值读出][franka-eval-gate] · [评估分母与逐轴结果](franka-rollout-audit.json)

五条测试文件的竖直标签都整段为 $-mg$。这些结果量化了相对额定重量的偏差，实际抬升中的惯性反力、释放时刻和侧向力仍需独立参考。[Franka 标签载入][franka-load]

从固定仓库的 `NeuralActuator/` 运行下列命令。这里的 `outputs` 保存本地评估，三次仿真写不同的结果和 NPZ。

```bash
mkdir -p outputs/franka-audit
sed 's/^trim_end: 0\.25$/trim_end: 0.0/' configs/franka_lift_hold.yaml \
  > outputs/franka-audit/extended.yaml
sed -e 's/^trim_start: 0\.30$/trim_start: 0.0/' \
    -e 's/^trim_end: 0\.25$/trim_end: 0.0/' configs/franka_lift_hold.yaml \
  > outputs/franka-audit/uncropped.yaml

JAX_PLATFORMS=cpu MUJOCO_GL=disable python -B evaluate_actuator_franka.py \
  --model_path checkpoints/franka_lift_hold.pkl \
  --config configs/franka_lift_hold.yaml \
  --output outputs/franka-audit/released.json \
  --dump_rollout outputs/franka-audit/released
```

另外两次分别替换 `--config`、`--output` 和 `--dump_rollout`，用 `extended`、`uncropped` 命名。把两个复核脚本放到同一目录，在 Python 3.11 及以上、已安装 NumPy 的环境中运行下列命令。评分脚本校验权重的 SHA-256 和 CSV 的 Git blob，用原始行重建下一帧位置和当前帧名义力，随后从 NPZ 重算逐轴误差并检查官方 JSON。Franka 的力标签在单条记录内恒定，力参考的当前帧约定来自评估器源码。[Franka 评分脚本](audit_franka_rollout.py) · [NAD 校验脚本](audit_nad.py) · [力和状态的配对代码][franka-eval-align]

```bash
python -B /path/to/audit_franka_rollout.py --source .. \
  --checkpoint checkpoints/franka_lift_hold.pkl \
  --released-eval outputs/franka-audit/released.json \
  --released-rollouts outputs/franka-audit/released \
  --extended-eval outputs/franka-audit/extended.json \
  --extended-rollouts outputs/franka-audit/extended \
  --uncropped-eval outputs/franka-audit/uncropped.json \
  --uncropped-rollouts outputs/franka-audit/uncropped
```

复核 EMA 时可复制 `extended.yaml` 并追加一行 `use_ema_params: true`。把该次评估的输出路径另用 `--ema-eval` 和 `--ema-rollouts` 交给评分脚本。

### CPU 复查，直接运行力头 {#cpu-replay}

我用发布的 OMX 力传感器和 Franka 抬升权重，在 JAX CPU 上运行作者的 CSV 预处理、Transformer 和 `--force_only` 扫描函数。两者都读取原始 `params` 键，OMX 带检查点的归一化统计，Franka 不带。逐条轨迹重置历史。另在首两帧、首次凑满九帧时及中段和末段独立拼装张量，核对滑窗、门值以及原始力乘门的结果。两个权重的参数数目分别为 1,441,838 和 1,445,490。[模型实现][transformer] · [权重镜像][weights] · [CPU 输出](force-replay.json)

**OMX。** 取前文的 +X 推力以及同命令的参考文件，各有 541 个推理输出。第三次重放保留推力轨迹的状态和标签，只把五路电流替换成参考轨迹对应行的记录。这个输入试验回答模型怎样使用遥测，替换后的信号组合并非在机械臂上重新执行得到。

| OMX 输入 | 三分量 MAE | X 轴 MAE | 平均门值 |
| --- | ---: | ---: | ---: |
| 推力原始遥测 | 0.523 N | 0.586 N | 0.9973 |
| 参考原始遥测，零力参照 | 0.00046 N | 0.00068 N | 0.0015 |
| 推力状态与参考电流 | 0.957 N | 1.894 N | 0.9779 |
{.table-readable}

门几乎仍开着，X 轴估力误差却增加了约 1.31 N。这个样本里，单看门是否触发会漏掉力幅值的变化。图中的推压段使用基座系带符号的 X 分量。参考段的小误差按零填充参照计，推力段按三轴传感器标签计。

<figure class="article-figure" id="fig-omx-force-replay">
  {{< post-image src="assets/force-replay.webp" alt="OMX 推力记录的 X 轴力。实测曲线在推压时转为负数，原遥测的预测随之下降，换入空载参考电流后估力多在零附近" >}}
  <figcaption><span class="article-figure__number">图 4</span><span class="article-figure__text">同一条推力记录，保留标签与时间轴。图中为基座系 X 分量。</span></figcaption>
</figure>

**Franka。** 每条轨迹分别运行三种目标输入。第一种是发布的五帧前视代理，第二种把前视减到零并保留 1.03 比例，第三种使用 CSV 的稀疏 `cmd_pos`。每次都用新目标重算跟踪误差，保持实测状态、`tau_d` 和标签。表格只比较受监督的竖直分量。

| 负载 | 推理行数 | 前视代理 | 当前位姿代理 | 原始稀疏命令 |
| --- | ---: | ---: | ---: | ---: |
| 200 g，含跨划分重复 | 417 | 0.362 N | 0.376 N | 0.678 N |
| 300 g | 422 | 0.465 N | 0.478 N | 0.820 N |
| 400 g | 420 | 0.544 N | 0.554 N | 0.862 N |
| 500 g | 417 | 0.542 N | 0.554 N | 0.775 N |
| 600 g | 418 | 0.426 N | 0.431 N | 0.498 N |
| 300–600 g 四条等权 | — | 0.494 N | 0.504 N | 0.739 N |
{.table-readable}

在这四条轨迹上，使用五帧前视代理时，实测状态重放的竖直 MAE 为 0.494 N，同一窗口的[仿真状态评估](#franka-mjx)为 0.492 N。两次都保留日志中的命令力矩和电机侧遥测。实测状态重放里，把五帧前视换为当前位姿代理增加约 0.010 N，换到稀疏命令则增加约 0.244 N。首次替换只改变前视索引，第二次还改变了跟踪目标的形态。要评价真机接口，需采集控制器当时可用的密集命令再做离线或在线验证。[Franka 代理实现][franka-load] · [CPU 输出](force-replay.json)

[CPU 复查脚本](replay_force_only.py) 会校验固定源文件、CSV 和两个发布权重的 SHA-256，只执行所需的特征与前向函数。使用独立 Python 3.12 CPU 环境。如需比对环境版本，结果 JSON 记录了 JAX、Flax、NumPy、pandas 等实际版本。可选的 `--trace` 把每组力、门值、时间和参照另写到新 NPZ，用于复画曲线。

```bash
python -B replay_force_only.py \
  --source /path/to/Dynamics-Modeling/NeuralActuator \
  --weights /path/to/checkpoints
```

### 只改变 Franka 的命令力矩输入 {#franka-torque-probe}

上面的目标通道试验保持 `tau_d` 不变。这里反过来，固定五帧代理、连杆和电机侧状态，只改七路 `tau_d`。用随仓库发布的 Panda MJCF，在 `hand` 局部 Z 轴前方 0.10 m 取一个假定作用点。根据每帧实测姿态计算该点的平移 Jacobian。模拟增加 100 g 的准静态补偿时，送入网络的命令力矩增加

$$
\Delta\tau_d=-J_v(q,p)^\top\Delta f,\qquad
\Delta f=[0,0,-0.981\ \mathrm{N}]^\top .
$$

负号来自驱动端抵消环境力的约定。减少 100 g 时取反向扰动。复查用 `mj_applyFT` 验证了每帧的关节投影。四条记录的最大单轴改变量约为 0.480 N·m。下表使用发布裁剪，去掉已知重复的 200 g。每行都用扰动后的输出减同一帧的原始输出，再取均值。完全追随所设载荷时，两列应分别为 −0.981 N 和 +0.981 N。[Panda 手部结构][franka-hand] · [命令力矩的特征索引][franka-features] · [逐任务结果](force-replay.json)

| 原始记录 | 增加 100 g 的平均竖直响应 | 减少 100 g 的平均竖直响应 |
| --- | ---: | ---: |
| 300 g | −1.098 N | +0.762 N |
| 400 g | −1.108 N | +1.113 N |
| 500 g | −0.766 N | +1.111 N |
| 600 g | −0.394 N | +0.749 N |
{.table-readable}

增加载荷时，98.3%–98.8% 的帧在预期方向上变得更负，减少载荷时四条记录的所有评估帧都在预期方向上变化。受到相同等效载荷增量时，600 g 记录的平均响应幅值小于其他三条，门值的逐轨迹均值仍在 0.9997 以上。冻结检查点对 `tau_d` 的响应方向大多符合构造，幅值随记录与扰动方向变化。四条记录的遥测和姿态也有差别，要判断是否存在负载饱和，需要在同姿态和同运动下测试更多负载。这里用的是固定原传感轨迹的离线干预，真实加装载荷后控制器、电机和运动轨迹会一起变化。

在 400 g 记录的保留段里，再只扰动索引 210 的 `tau_d`，行号从零起算。索引 210 到 218 的九个连续输出都发生变化，原时钟跨度约 125 ms。竖直估力的最大差为 0.420 N，出现在最后一帧。这个区间之外，三轴力和门值与基线逐项相同。这里量到的是单帧输入在前向回放中的影响范围，物理传感与反馈延迟需要另测。[单帧响应数组](force-replay.json)

**用日志里的 `tau_ext` 复核符号。** 这七列是额外的外力矩估计。沿用同一作用点，以基座 Z 轴单位向量 $e_z$ 取关节映射 $j_z=J_v^\top e_z$。假定只有该点的竖直力，一维最小二乘返回

$$
\hat f_z=\frac{j_z^\top\tau_{\rm ext}}{j_z^\top j_z}.
$$

300 g 记录直接投影的竖直估计均值为 +3.427 N，名义标签是 −2.943 N。四条记录等权的竖直 MAE 是 9.271 N，先将七路力矩取反后为 0.746 N。两种评分用了相同的姿态、原始行号和名义标签。四条记录中 $j_z$ 的最小范数约为 0.529 m/rad，一维拟合后关节力矩残差的平均范数仍约为 0.694 N·m。

这个符号复核使用了冻结力头之外的遥测。数据字段表只描述了估计量及单位，实机应通过独立加载确认作用点、载荷设置和符号。[Franka 数据字段][dataset] · [机器人状态字段定义][franka-state-doc] · [逐任务符号复核](force-replay.json)

要重跑这个对照，在上一条命令中加 `--franka-torque-probe`。环境还需要 MuJoCo 及固定仓库的 Panda 模型资产。输出 JSON 的 `franka_torque_probe` 节记录载荷、参考点、门值与响应，`cases` 仍保存原日志的 MAE。[CPU 复查脚本](replay_force_only.py) · [完整结果](force-replay.json)

### 几个后端能支持什么结论 {#backends}

论文主线用 MJX。论文 §IV-N 的 NVIDIA Warp 补充实验用 Featherstone 积分器，按 60 Hz 控制、每帧五个子步做可微 rollout。固定仓库 README 另列 Newton、MuJoCo Warp 和 SuperDex 的适配实现及训练结果。应跟着各自依赖、机器人描述和控制装配走，不能看见 Warp 就认为它们使用同一求解器。[论文 v2，§IV-N][paper] · [四后端实现说明][repo-backends]

| 固定版 README 的 A100 实验 | 训练步时间 | 相对 MuJoCo float64 的最大关节差 |
| --- | ---: | ---: |
| MuJoCo MJX（JAX） | 8 ms | 0.0037 rad |
| Newton（JAX / PyTorch） | 22 / 24 ms | 0.017 rad |
| MuJoCo Warp（PyTorch） | 17 ms | 0.0000042 rad |
| SuperDex（CPU 仿真、GPU 网络） | 29 ms | 0.023 rad |
{.table-readable}

训练步指批大小 16、128 步 rollout 的前向与反传。最右列来自另外一个 320 步、无接触且在关节限制内的扭矩回放对照。SuperDex 以 URDF 搭建模型，并行跑 CPU 仿真场景，其他引擎使用的接口和求解器有别。该表只测指定设置，不表示任意模型的速度排序或不同后端对接触冲击的精度。训练主线关闭碰撞接触，载荷通过等效驱动或显式施力进入，因此抓放视频支持在已有数据上的状态和力匹配，新的几何接触仍需要接触求解器与独立验证。[后端计时与参考仿真][repo-backend-timing] · [无接触训练说明][repo-contact]

### 复查数据，或接进自己的工程 {#audit-and-use}

本文的计数、裁剪对齐和跨划分比较可用 [NAD 复核脚本](audit_nad.py) 重跑。默认从固定 Git 版本读取目录树和 32 个指定 CSV，用 Git blob 校验输入字节。加 `--franka-overlaps` 会再比较全部 150 个 Franka 训练与测试文件对。完整本地克隆可直接读 Git 对象。若已有单独的数据目录，可以用 `--csv-root` 指向 `NeuralActuator/data`，同时保留 `--source` 以检查目录清单。这个检查无需模型依赖和权重。[包含全对比的复核结果](nad-audit.json)

```bash
python3 -B audit_nad.py --source /path/to/Dynamics-Modeling --franka-overlaps
```

自己采集时，可以用下面三个问题组织验收。

1. **仿真能否复现既有记录？** 保存命令、状态和 effort 的单位、关节顺序及源时间。每类遥测保留有效标志和接收时间，便于区分真实零值、读取失败与旧缓存。从同一初始状态计算误差随时间窗的变化，并单列仿真控制输入的限幅比例。
2. **当前遥测能否区分外力？** 用同一作用点的已知正反推力校验独立传感器的符号，再变更速度、温度、姿态和力方向。分别报告接触稳定段的逐轴误差和接触切换的误报、延迟，明确每个分量有效的时间窗。
3. **是否要把输出用于控制？** 固定并记录模型和规范化参数，以整条轨迹和设备做留出。在同一时间轴记录采样、估力可用和新命令下发的三个时刻，统计总延迟和超时次数。接入[阻抗控制]({{< relref "/posts/robotics/control/impedance-control" >}})时把估计作为受限反馈输入，独立的力矩、电流和速度限制仍由控制系统执行。

## 阅读自测与验收 {#self-check}

- **两个不同的力与驱动力组合怎么产生相同的位姿？** 广义外力变化由驱动端作等量反向补偿，两路输入之和保持，位姿只能看到总效果。
- **单关节隐式算例再施加 −4 N 会怎样？** 20 ms 内的角位移从 0.004 rad 降为 0.002 rad，外力贡献被记了两次。
- **为什么九帧的双向注意力仍可用于 OMX 在线估力？** 九帧是当前帧与已到达的历史帧，窗口里无需读下一帧的遥测。
- **为什么读 SO-101 的记录要选 `load`？** 发布模型使用带符号的负载寄存器作为 effort 特征，输入尺度和电流计数不同。
- **掩码之后只剩竖直标签时，力损失分母怎么变？** 发布 SO-101 配方在分量乘有效性之后仍按三项平均，门控另行监督。
- **一行的训练与离线力标签偏差能说明什么？** 它说明配对脚本的索引差，在接触沿或释放沿要单独观察。传感器实际延迟需要时钟和独立基准。
- **论文表中 F 为 0.20 N 可以对应怎样的竖直误差？** 若两个侧向误差都为零，竖直误差可为 0.60 N。
- **Franka 为什么有约 80 ms 的前视？** 裁剪后的目标通道从五帧后的实测姿态构造，62.5 Hz 下五帧约 80 ms，轨迹末尾限制到最后一帧。
- **OMX 200 g 的短文件为何哈希不同却仍然重复？** 它是较长机构记录的连续裁剪，内容按时间原点和行区间对齐后相符。
- **Franka 哪一项公开测试需要重新划分？** 200 g 的 `test/200_006.csv` 与 `train/200_007.csv` 逐字节相同，而且前期微调权重选择会读取测试集。
- **同一个命令—电流日志能否用于比较新的候选控制序列？** 它给出实际执行轨迹的遥测，新的候选序列需要新的因果 effort 响应模型。
- **打开 `apply_external_force` 后，如何核对评估脚本？** 先看该入口是否把预测力写进仿真受力字段，并验证施力参考点和力矩平移。
- **SO-101 表 XIV 的 1.52° 与 2.50° 为何都能成立？** 1.52° 先对六任务按关节平均再取最大，2.50° 是任意单个任务与关节格的最大值。
- **为什么 Franka 默认发布配置没有 500 步一列？** 五条公开测试记录裁剪后只够 417–422 步，独立评估器会跳过超出长度的窗口。
- **只延长裁剪终点和把记录开头放回，有什么区别？** 前者保留相同的仿真初始位置与历史，后者把评估起点移回原始第 0 行。
- **为什么 400 g 常量的三分量分数更低，竖直误差却翻倍？** 两个缺测的侧向分量在评估里以零为参照，常量在这两轴恰好取零。
- **给 Franka 的 `tau_d` 加上等效载荷能检验什么？** 固定原日志的其他通道，可以观察发布力头对命令力矩的响应方向与幅值。实际负载改变时还需重采遥测。
- **OMX 的接触门仍开着，能说明替换电流后的估力可靠吗？** 样本平均门值为 0.978，但 X 轴 MAE 从 0.586 N 上升到 1.894 N，仍需单独检查各轴输出。

## 原始材料 {#sources}

- Zhiyang Dou、John U. Onyemelukwe、Hangxing Zhang 等，*NeuralActuator: Neural Actuation Modeling for Robot Dynamics and External Force Perception*。[arXiv v2][paper] · [RSS 2026][rss] · [项目页][project]
- Dynamics-Modeling 源码、NAD、权重和版本互相独立。本文用固定源码版本与配方，不同后端或论文实验保留各自口径。[GitHub 提交][commit] · [NAD 镜像][nad-hf] · [权重镜像][weights]

[paper]: https://arxiv.org/abs/2607.11734v2
[rss]: https://roboticsproceedings.org/rss22/p159.html
[project]: https://frank-zy-dou.github.io/projects/NeuralActuator/index.html
[commit]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/commit/541a93aac4368e73883e86a51f31a7add6bde059
[repo]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/541a93aac4368e73883e86a51f31a7add6bde059
[data-tree]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/data
[omx-force-tree]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/data/force_sensor
[nad-hf]: https://huggingface.co/datasets/frankzydou/NAD/tree/88838bce278e381a830a71af83e27dfbb1dac414
[weights]: https://huggingface.co/frankzydou/NeuralActuator/tree/cd58d48d465450ffb734ed8ba22e48266ccdb2f2
[dataset]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/docs/dataset.md
[teleop]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/teleop/README.md
[omx-xml]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/robot/open_manipulator_x.xml#L40-L53
[omx-features]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator.py#L111-L207
[omx-loader]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim.py#L150-L195
[omx-coupling]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim.py#L1159-L1264
[omx-loss]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim.py#L1321-L1378
[omx-batches]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim.py#L1748-L1769
[omx-eval]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator.py#L610-L691
[omx-eval-step]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator.py#L304-L339
[omx-config]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/configs/force_sensor.yaml
[weight-ft2]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/configs/weight_all_ft2.yaml
[so-features]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_so101.py#L103-L193
[so-batches]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim_so101.py#L1303-L1318
[so-coupling]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim_so101.py#L934-L957
[so-eval-step]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_so101.py#L275-L309
[so-train]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim_so101.py#L981-L1061
[franka-train]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim_franka.py#L915-L960
[franka-best]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim_franka.py#L1546-L1625
[franka-load]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_franka.py#L105-L166
[franka-features]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_franka.py#L178-L229
[franka-config]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/configs/franka_lift_hold.yaml
[franka-ft1]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/configs/franka_lift_hold_ft1.yaml
[franka-ft2]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/configs/franka_lift_hold_ft2.yaml
[franka-eval-run]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/scripts/eval_franka_lift_hold.sh
[franka-eval-weights]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_franka.py#L59-L76
[franka-eval-align]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_franka.py#L470-L527
[franka-eval-gate]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_franka.py#L337-L346
[franka-eval-control]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_franka.py#L284-L300
[franka-aggregate]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_franka.py#L750-L776
[franka-checkpoint]: https://huggingface.co/frankzydou/NeuralActuator/blob/cd58d48d465450ffb734ed8ba22e48266ccdb2f2/franka_lift_hold.pkl
[transformer]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/models/transformer.py#L225-L327
[attention]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/models/transformer.py#L70-L134
[infer-force]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/infer_actuator.py#L297-L364
[infer-main]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/infer_actuator.py#L367-L520
[repo-force]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/README.md#virtual-force-sensor
[repo-so]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/README.md#so-101-lerobot-arm
[repo-coupling]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/README.md#implicit-vs-explicit-force-coupling-with-differentiable-simulation
[repo-backends]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/README.md#additional-differentiable-simulation-backends
[repo-backend-timing]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/README.md#L580-L597
[repo-contact]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/README.md#L977-L998
[franka-duplicate-train]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/data/franka/lift_hold/train/200_007.csv
[franka-duplicate-test]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/data/franka/lift_hold/test/200_006.csv
[omx-force-csv]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/data/force_sensor/force_x_plus/test/001.csv
[omx-reference-csv]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/data/force_sensor/force_x_plus_ref/test/001.csv
[so-csv]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/data/so101/pick_and_place/300g/test/001.csv
[franka-csv]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/data/franka/lift_hold/test/300_001.csv
[config-dir]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/configs
[apply-ft]: https://mujoco.readthedocs.io/en/3.11.0/APIreference/APIfunctions.html#mj-applyft
[omx-sampler]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim.py#L273-L371
[omx-init]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim.py#L1419-L1461
[omx-arrays]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim.py#L887-L907
[omx-val-batch]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim.py#L1927-L1965
[so-init]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim_so101.py#L1066-L1101
[omx-eval-init]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator.py#L365-L385
[franka-window]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_franka.py#L615-L639
[franka-xml]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/robot_franka/panda.xml#L264-L277
[franka-hand]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/robot_franka/panda.xml#L197-L235
[franka-state-doc]: https://frankarobotics.github.io/libfranka/0.15.0/structfranka_1_1RobotState.html
[omx-teleop-row]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/teleop/omx/dxl_arm.py#L1508-L1557
[omx-teleop-force]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/teleop/omx/dxl_arm.py#L1586-L1628
[so-teleop-registers]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/teleop/so101/record_teleop_data.py#L229-L250
[so-teleop-row]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/teleop/so101/record_teleop_data.py#L561-L592
[so-teleop-labels]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/teleop/so101/README.md#L87-L97
[omx-eval-main]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator.py#L1022-L1062
[repo-results]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/README.md#L394-L406
[repo-franka]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/README.md#L805-L856
[condition-config]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/configs/motor_condition.yaml#L92-L102
[omx-val-labels]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/train_actuator_diffsim.py#L719-L735
[omx-condition-eval]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator.py#L621-L639
[repo-condition-results]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/README.md#L423-L435
[so-config]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/configs/so101_weight.yaml#L37-L65
[so-metrics]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_so101.py#L586-L632
[so-sim-features]: https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/541a93aac4368e73883e86a51f31a7add6bde059/NeuralActuator/evaluate_actuator_so101.py#L254-L275
