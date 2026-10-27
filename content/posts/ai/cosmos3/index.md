---
title: "Cosmos 3 论文详解：把推理、视频、音频与动作接入同一个世界模型"
date: 2026-10-22
lastmod: 2026-10-27
draft: false
tags: ["Cosmos 3", "World Model", "World Action Model", "Multimodal", "Flow Matching", "Robot Learning", "Paper Notes"]
categories: ["人工智能"]
authors: ["chase"]
summary: "拆解 Cosmos 3 的双塔 MoT、AR／扩散 token 排列、几何伪动作、物理时间对齐与分阶段训练，分析 DROID 策略、跨域迁移、视频和音频评估、消融及推理成本，并区分统一架构、专用检查点和真实控制能力。"
description: "拆解 Cosmos 3 的双塔 MoT、AR／扩散 token 排列、几何伪动作、物理时间对齐与分阶段训练，分析 DROID 策略、跨域迁移、视频和音频评估、消融及推理成本，并区分统一架构、专用检查点和真实控制能力。"
contentLanguage: "zh-CN"
math: true
toc: true
images: ["assets/dual-tower-overview.webp"]
imageZoom: true
reading_prerequisites: "Transformer attention、向量与坐标变换、扩散或 Flow Matching 的基本概念；动作表示与评估指标会逐项解释"
reading_focus: "先看双塔怎样交互，再用具体任务理解三种动作条件、物理时间与固定条件采样；沿训练课程和 DROID 请求追踪复用，最后分别核对生成、视觉预测与闭环控制的评测协议。"
related_posts:
  - "/posts/ai/transformer-attention"
  - "/posts/ai/vla-world-model-data"
  - "/posts/ai/supermimic"
  - "/posts/ai/video-to-data"
---

让一个模型回答视频里的机器人在做什么、生成动作执行后的画面，再直接预测机器人应该执行的动作，过去往往需要不同系统。视觉语言模型擅长回答，视频模型擅长合成，动作策略擅长控制；它们的表示、时间尺度和训练目标却不容易连接。

**Cosmos 3 的核心设计，是把离散语言的自回归建模与连续视频、音频、动作的流匹配建模放进同一个双塔 Transformer。** 理解塔 Reasoner 提供上下文，生成塔 Generator 预测连续模态；改变已知条件与预测目标，就能复用这套架构完成理解、生成、前向动力学、逆动力学与策略建模。

本文依据 NVIDIA 的 [《Cosmos 3: Omnimodal World Models for Physical AI》v4](https://arxiv.org/abs/2606.02800v4)，即这份 139 页报告：arXiv 版本日期为 2026-06-23，PDF 封面日期为 2026-06-24。以下实验、模型规模和训练配置均按该版本分析；文中提到的榜单名次保留报告中的日期。作者的[项目页](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)和[代码入口](https://github.com/nvidia/cosmos)可用于继续查阅。

在机器人任务中，专门适配的策略检查点根据当前观测和语言条件联合预测动作与视觉未来，再把动作交给低层控制器。沿这条路径，本文分别检查**表示怎样对齐、预测能否对应、执行是否成功**。

## 阅读路线

**第一次阅读：** [双塔交互](#mot) → [动作模式](#action-modes) → [训练课程](#training) → [DROID 策略](#droid-policy) → [机器人结果](#robot-results)，先把架构到执行的主线连起来。

**准备复现：** 重点核对[动作坐标与维数](#action-representation)、[时间对齐](#temporal-alignment)、[采样与损失](#flow-objective)、[推理配置](#inference)和[实验对照](#limits)。

<details id="cosmos3-reading-questions">
<summary>展开按问题查阅的章节索引</summary>

| 想先回答的问题 | 阅读入口 |
| --- | --- |
| 一个模型怎样兼容语言和连续信号 | [模型家族](#model-family)、[编码与 token](#encoders)、[MoT 与注意力](#mot) |
| 机器人动作为什么可以和视频一起生成 | [几何动作表示](#action-representation)、[三种动作模式](#action-modes)、[放杯子例子](#action-conditioning-example) |
| 视频、音频、动作怎样在时间上对齐 | [物理时间坐标](#temporal-alignment)、[可运行核对](#contract-lab) |
| 能力是怎样分阶段训练出来的 | [流匹配目标](#flow-objective)、[训练课程](#training)、[数据工程](#data) |
| 模型真的能控制机器人吗 | [DROID 后训练](#droid-policy)、[请求与执行](#droid-request-cycle)、[闭环与迁移结果](#robot-results) |
| 哪些结果可靠，哪些仍有边界 | [生成评估](#generation-results)、[机制消融](#ablations)、[成本与局限](#limits) |
{.table-readable}

</details>

文中的 **token** 是模型序列中的一个表示单元，既可以表示离散文本，也可以承载连续特征；**latent** 指编码器压缩后的连续表示。**干净条件**指保留观测值、不加生成噪声的输入；**检查点**是一份训练后保存的模型权重。

## 1. 统一的是架构与表示，还需要区分检查点 {#model-family}

Cosmos 3 覆盖语言、图像、视频、音频和动作。它将多种任务写成一个共同形式：给定部分干净条件，预测文本，或者生成其余连续模态。

| 任务 | 主要条件 | 主要输出 |
| --- | --- | --- |
| 视觉语言理解 | 图像／视频与问题 | 文本答案、定位或推理结果 |
| 图像与视频生成 | 文本，可选初始图像／视频 | 图像或未来视频 |
| 音视频生成 | 文本及可选视觉条件 | 相互对齐的视频与声音 |
| 前向动力学 FD | 已观察画面与已知动作 | 动作后的视觉变化 |
| 逆动力学 ID | 已观察视觉变化 | 解释变化的动作 |
| Policy／world-action | 当前观测与任务 | 动作及其预期视觉结果 |
{.table-readable}

报告以中期训练后的 Cosmos3-Nano 和 Cosmos3-Super 为通用起点，再通过图像、视频或机器人策略后训练，得到各自独立的专用检查点。

| 变体 | 双塔模型规模 | 基础语言／视觉语言骨干 | 报告中的定位 |
| --- | ---: | --- | --- |
| Edge | 约 4B | 自训练的约 2B 稠密骨干 | 较小规模；报告中列为后续发布 |
| Nano | 约 16B | Qwen3-VL-8B | 通用模型与 DROID 策略基础 |
| Super | 约 64B | Qwen3-VL-32B | 更大规模通用模型及生成专用版本 |
{.table-readable}

Nano 有 36 层、隐藏宽度 4096；Super 有 64 层、隐藏宽度 5120。这里的 MoT 是 **Mixture-of-Transformers**：每层按自回归 AR／扩散 DM 路径使用两套参数。它与通过路由器挑选少数 FFN 专家的稀疏 MoE 不同，64B 描述的是双塔规模。[报告 Sections 2.3、2.5](https://arxiv.org/pdf/2606.02800v4#page=11)

## 2. 先把不同模态编码成 token {#encoders}

### 2.1 同一张图，可以承担两种不同职责

Cosmos 3 的视觉输入有两条编码路径：

- **理解路径使用 ViT。** 先按 $16\times16$ patch 编码，再由两层 MLP 合并 $2\times2$ token 并投影到 Transformer 空间。这条路径还沿用 Qwen3-VL 的 DeepStack 特征和交错视频时间戳。
- **生成路径使用视频 VAE。** 来自 Wan2.2-TI2V-5B 的 VAE 在时间上压缩约 4 倍；空间上先压缩 $16\times16$，再做 $2\times2$ patch merge，得到有效 $32\times32$ 压缩。

ViT 产生适合理解的视觉特征，VAE 产生能够解码回图像／视频的连续 latent。二者即使来自同一幅图，也不能直接视为同一种 token。理解 ViT 随骨干训练，生成用 VAE 在训练期间保持冻结。

### 2.2 音频与动作如何进入连续路径

音频 VAE 接收 48 kHz 双声道波形，每 1920 个采样点形成一个时间步：

$$
\mathrm{TPS}_{\mathrm{audio}}=\frac{48000}{1920}=25.
$$

因此音频 latent 每秒有 25 个时间位置。音频 VAE 也被冻结，其输出经线性层进入 Transformer 隐藏空间。

动作则使用按具身类型区分的线性输入／输出投影，后文会解释不同维数怎样连接同一骨干。所有非语言模态还加上可学习的模态嵌入，以区分输入种类。

从报告具体实现看，AR 子序列主要包含语言和 ViT 视觉 token，音频和动作主要位于连续生成路径。“全模态”描述的是整个系统覆盖的模态集合；它不意味着每种输入都经过完全对称的理解分支，也不能自动推出所有语音问答能力。

## 3. MoT：两套参数怎样通过注意力协作 {#mot}

### 3.1 先排列 AR，再排列扩散 token

每个样本的序列先放自回归 AR 子序列，再放扩散 DM 子序列。AR 末尾有结束与开始生成的特殊 token；DM 中区分干净条件和带噪目标，同一区段再按视觉、音频、动作排序。

这里的 DM 是 diffusion 的缩写，实际训练使用流匹配目标。语言通过 next-token prediction 输出，连续模态则通过多次去噪／流积分生成。

<figure class="article-figure" id="fig-cosmos3-mot">
  {{< post-image src="assets/dual-tower-overview.webp" alt="Cosmos 3 双塔结构：AR 语言和 ViT 视觉进入 Reasoner，VAE 视觉、音频和动作进入 Generator；Generator 读取 AR 上下文，Reasoner 不读取扩散目标" >}}
  <figcaption>
    <span class="article-figure__number">图 1</span>
    <span class="article-figure__text">每个 Transformer 层都有各自的 AR 与 DM 参数路径。Generator 查询同一样本的 AR 与 DM 上下文，Reasoner 保持 AR 内部因果注意力。图中省略残差、归一化、投影细节和输出解码器；连续 latent 需要多步采样才能形成输出。</span>
  </figcaption>
</figure>

### 3.2 共享 attention 操作，不等于共享全部权重

每层的两条路径分别拥有归一化、注意力投影和 MLP 参数；它们在联合注意力计算中交互。将两条路径的 query、key、value 写成 $Q_R,K_R,V_R$ 与 $Q_D,K_D,V_D$：

$$
O_R=\operatorname{Attn}_{\mathrm{causal}}(Q_R,K_R,V_R),
$$

$$
O_D=\operatorname{Attn}_{\mathrm{full}}
\bigl(Q_D,[K_R;K_D],[V_R;V_D]\bigr).
$$

AR query 只看自身因果前缀，DM query 则能看全部 AR 条件及 DM token。**信息依赖从 AR 流向 DM，AR 不依赖正在去噪的 DM 内容。** 这是报告 Equations 7、8 的核心。

这里的“读取 Reasoner”主要是读取各层隐藏表示，并不要求每次先输出一段可见的思维链。[第 10.1 节](#101-prompt-upsampling-是一个独立接口)中的提示词扩写是额外的请求处理流程；固定 AR 上下文的缓存则由上述单向依赖结构支持，详见[推理优化](#103-工程优化分别处理哪种瓶颈)。

### 3.3 五个 token 就能看懂 mask

取三个 AR token 和两个 DM token，依次排列为：

$$
[R_0,R_1,R_2,D_0,D_1].
$$

下面的掩码中，行是 query、列是 key，1 表示允许注意：

$$
M=\begin{bmatrix}
1&0&0&0&0\\
1&1&0&0&0\\
1&1&1&0&0\\
1&1&1&1&1\\
1&1&1&1&1
\end{bmatrix}.
$$

右上角全为 0，说明 AR 读不到 DM；下方两行全为 1，说明 DM 内部是双向注意力，还能读取完整 AR 上下文。把多个样本打包时，样本之间仍要隔离，不能因为 token 拼接了就互相可见。

DM 的双向注意力是在**一次去噪计算内部**联合约束所有目标位置，不代表部署机器人可以提前获得真实未来观测。[报告 Figure 5、Section 5.2.2](https://arxiv.org/pdf/2606.02800v4#page=11)

## 4. 几何伪动作：跨具身复用的连接点 {#action-representation}

### 4.1 为什么使用相对位姿

汽车、相机、人手和机器人拥有不同控制接口。Cosmos 3 的中期训练选择共同的几何结构，而不是统一电机电流或 PID 参数。

对于连续位姿 $T_{t-1},T_t\in SE(3)$，运动表示为：

$$
\Delta T_t=T_{t-1}^{-1}T_t.
$$

每个位姿变化由 3D 平移与连续 6D 旋转表示组成，共 9 个数。**6D 是旋转参数化的维数，三维刚体朝向仍只有三个自由度。** 这些从状态变化推得的量称为 pseudo-actions，本文译为几何伪动作。

相对变换还决定了平移的坐标系。若约定 $T_t$ 将局部坐标映射到世界坐标，写成旋转 $R_t$ 与位置 $p_t$，则：

$$
\Delta R_t=R_{t-1}^{\mathsf T}R_t,\qquad
\Delta p_t=R_{t-1}^{\mathsf T}(p_t-p_{t-1}).
$$

例如前一时刻绕世界 z 轴旋转了 90°，位置从 $(1,2,0)$ m 移到 $(1,3,0)$ m：世界位移是 $(0,1,0)$ m，在前一局部坐标系中却是 $(1,0,0)$ m。这个构造例子说明，输入单位、轴向与坐标变换约定必须配套；直接相减世界坐标不一定得到所需的局部动作。

它们让模型知道“相机或末端怎样移动”，但不包含产生这种移动所需的全部低层控制、接触力和执行器信息。根据视频估计出的位姿变化也不一定是独立施加的干预，因此“动作条件建模”与严格识别全部物理因果机制仍有距离。

### 4.2 Ego、effector 与 grasp 三类分量

报告使用三个几何组件：主体观察坐标系的 ego pose、末端的 effector pose，以及抓握状态 grasp state。前两者是相对位姿；抓握状态直接描述当前时刻，例如夹爪开合或指尖在腕坐标系中的位置。

| 类型 | 表示构成 | 维数 |
| --- | --- | ---: |
| 汽车／相机 | 一组 ego 相对位姿 | 9 |
| 单臂机器人 | 一组末端相对位姿＋夹爪状态 | 10 |
| 双臂机器人 | 两组末端相对位姿＋各自夹爪状态 | 20 |
| 带 ego 运动的双臂人形机器人 | ego 9＋两组末端 9＋两组夹爪 1 | 29 |
| 第一视角双手 | 头部 9＋左右腕各 9＋左右手各五个指尖 15 | 57 |

第一视角 57 维可直接手算为 $9+2\times9+2\times15$。人形机器人的 29 维则是这组几何组件的和，不能解释成 29 个关节位置；这与 [SuperMimic](../supermimic/)的 29 维关节动作含义不同。

### 4.3 输入输出投影按 domain 区分

归一化后的动作向量 $x\in\mathbb R^{d_k}$ 经对应具身域 $k$ 的投影进入模型。把输入与输出两侧的隐藏状态分开记：

$$
h_{\mathrm{in}}=W_{\mathrm{in}}^{(k)}x+b_{\mathrm{in}}^{(k)},
$$

$$
y=W_{\mathrm{out}}^{(k)}h_{\mathrm{out}}+b_{\mathrm{out}}^{(k)}.
$$

$h_{\mathrm{out}}$ 是经过多模态 MoT 计算后的隐藏状态，不能直接拿 $h_{\mathrm{in}}$ 代替；$y$ 表示投影回该域动作维数的输出。在[第 7 节](#flow-objective)的生成目标下，网络学习预测连续表示的流速度，再由采样器逐步恢复动作。

例如 Nano 的动作接口可写成“单臂 10 维 → 4096 维隐藏空间 → MoT → 10 维输出”，第一视角双手则是“57 → 4096 → MoT → 57”。不同维数通过域专用投影连接共享骨干。报告将预测的 6D 旋转通过 SVD 还原为 $SO(3)$ 矩阵。[报告 Section 2.1.3、Figure 3](https://arxiv.org/pdf/2606.02800v4#page=8)

## 5. FD、ID 与 Policy：改变哪些 token 是已知量 {#action-modes}

令 $v_{t-1}$ 为前一视觉状态，$a_t$ 描述从 $v_{t-1}$ 到 $v_t$ 的变化。三种模式的差别，是哪些视频／动作 token 保持干净，哪些加入噪声后作为生成目标。

<figure class="article-figure" id="fig-cosmos3-action-modes">
  {{< post-image src="assets/action-modes.webp" alt="Cosmos 3 三种动作配置：前向动力学由观测和已知动作生成未来视频，逆动力学由视觉转移推断动作，策略模式从当前观测和任务联合生成动作及未来视频" >}}
  <figcaption>
    <span class="article-figure__number">图 2</span>
    <span class="article-figure__text">蓝色为已知条件，橙色为生成目标。Policy 模式中的动作与未来视频联合建模；不是先得到真实未来再计算动作。动作 <i>a</i><sub>t</sub> 连接前一视觉状态 <i>v</i><sub>t−1</sub> 与当前状态 <i>v</i><sub>t</sub>；三种模式均可结合文本条件。图中压缩了模态编码、特殊 token 和多步流采样。</span>
  </figcaption>
</figure>

- **FD：给定动作，预测结果。** 条件包括已观察画面和干净动作，未来视频是带噪目标。它适合评估动作后果或构造视觉滚动。
- **ID：给定视觉变化，推断动作。** 视频为干净条件，动作加入噪声后由模型恢复。仅凭视频可能存在多解；生成分布不等于唯一真值控制命令。
- **Policy：联合预测动作和视觉结果。** 当前观测与任务为条件，未来视频和动作都是目标。视频流为动作提供可共同学习的世界变化结构。

### 5.1 用同一次“把杯子放到托盘里”区分三种任务 {#action-conditioning-example}

设机器人当前看到杯子和桌面，任务是“把杯子放到托盘里”。用 $C$ 表示当前观测与语言条件，$A$ 表示未来动作块，$V$ 表示对应的未来视觉序列。三个模式可以概括成：

$$
\begin{aligned}
\mathrm{FD}:&\quad p_\theta(V\mid C,A),\\
\mathrm{ID}:&\quad p_\theta(A\mid C,V),\\
\mathrm{Policy}:&\quad p_\theta(V,A\mid C).
\end{aligned}
$$

FD 的提问是“**执行这组候选动作会怎样**”，所以动作必须由外部提供；ID 是“**怎样的动作能解释这段已知视觉变化**”，所以它可以读取后续画面；Policy 是“**现在该怎样做，以及预期会发生什么**”，所以动作与未来画面都要生成。FD 本身并不包含搜索最优动作的规划器。

训练日志可以同时包含 $C,A,V$，但每种模式只把约定的部分作为干净条件。Policy 训练会对未来 $A,V$ 加噪并学习恢复；部署时，未知目标从噪声初始化，模型仅凭 $C$ 联合采样。**真实未来画面只在训练中充当监督目标，不能在策略推理中作为已知输入。**

联合采样完成后读取动作，可以理解为取得联合分布的动作边缘：$p_\theta(A\mid C)=\int p_\theta(V,A\mid C)\,\mathrm dV$。这里用积分说明概率关系，实际推理通过联合采样取得动作，不显式计算这个积分。

虽然论文用“policy”称呼第三种模式，将通用几何动作变成可执行机器人策略仍需要适配。报告对 DROID 做了专门后训练，重新定义动作输出、加入本体状态并压缩采样步数，[第 11 节](#droid-policy)会详细说明。

同一表示也支持 T2I、T2V、I2V、V2V，以及边缘、深度、分割或场景地图到 RGB 的 video transfer。I2V 把初始 latent 帧保持干净，V2V 提供更长的干净前缀；这些条件 token 不作为带噪重建目标。

## 6. 物理时间对齐：token 序号不等于秒 {#temporal-alignment}

视频、音频与动作的采样率不同。若简单让每种 token 的时间索引每次加 1，那么 25 个音频时间步、15 个动作时间步和 6 个视频 latent 时间步，会被误当成不同长度的物理过程。

Cosmos 3 使用三维 MRoPE：视觉 token 有时间、高、宽坐标；音频和动作只使用时间轴，空间坐标置零。AR 中的语言和 ViT 视觉沿用 Qwen3-VL 的位置分配；DM 时间坐标则按实际物理速率调制。

设每秒 token 时间步数为 TPS，以 24 FPS 视频经 4 倍时间压缩后的速率为基准：

$$
\mathrm{TPS}_{\mathrm{base}}=\frac{24}{4}=6,\qquad
\delta_t=\frac{\mathrm{TPS}_{\mathrm{base}}}{\mathrm{TPS}}.
$$

| 模态／数据 | TPS | 相邻时间步的位置增量 |
| --- | ---: | ---: |
| 24 FPS 视频 latent | 6 | 1 |
| 16 FPS 视频 latent | 4 | 1.5 |
| 30 FPS 视频 latent | 7.5 | 0.8 |
| 音频 latent | 25 | 0.24 |
| 15 Hz 动作 | 15 | 0.4 |

于是相同的一秒物理时间对应相同的 6 个位置单位。不同模态在打包序列中可以相距很远，但物理时间坐标仍然对齐；序列排列与时间含义是两种坐标。

TPS 统计的是**每秒时间步数**，空间网格上的 token 还要另计。例如 Table 5 的 $480\times832$ 视频，经有效 $32\times32$ 空间压缩后，每个 latent 时间步包含 $15\times26=390$ 个视觉 token。24 FPS 对应 6 个 latent 时间步／秒，按稳态速率估算约为 $390\times6=2340$ 个视觉 token／秒，而不是每秒只有 6 个序列 token。这个估算省略首帧特殊处理、片段边界、文本与其他模态。

报告还在 AR 与 DM 位置范围之间插入固定的 15000 偏移，缓解前几帧过饱和和棋盘伪影。**这是位置坐标间隔，不是额外插入 15000 个 token，也不是延迟 15000 帧。** 多模态时间对齐本身仍由 TPS 调制负责。[报告 Sections 2.4、E.2](https://arxiv.org/pdf/2606.02800v4#page=12)

### 可运行核对：mask、样本隔离与一秒对齐 {#contract-lab}

下载 [token_contract_lab.py](token_contract_lab.py)，在保存脚本的目录运行：

```bash
python3 -B token_contract_lab.py --self-test
```

脚本只使用 Python 标准库，构造上述 $5\times5$ 注意力 mask，检查 AR 不能读取 DM、打包样本互相隔离，以及不同采样率在一秒处映射到相同坐标。它核对的是表示与接口关系，不执行 Cosmos 3 网络或视频采样。

输出中的 `masked_flow_example` 还演示了[下一节](#condition-loss-masks)的两个标量：条件值保持 5，只有目标值参与损失并沿流方向更新。这个简化例子省略 CFG、noise shift 与模态损失权重。

## 7. 连续生成目标：预测流速度，而非直接回归整段视频 {#flow-objective}

对于某个模态的干净 latent $x_0$ 和高斯噪声 $\epsilon$，报告使用：

$$
x_\sigma=\sigma\epsilon+(1-\sigma)x_0,\qquad
v^*=\epsilon-x_0.
$$

网络预测 $v_\theta(x_\sigma,\sigma,c)$，其中 $c$ 包含文本、干净媒体和已知动作等条件。这里的 $v_\theta$ 是连续表示空间中的**流速度**，前文的 $v_t$ 则表示视觉状态；流速度也不等于机器人关节角速度。视频的物理时间决定哪一段运动被表示，$\sigma$ 决定这段表示当前加了多少噪声，二者是不同坐标。

训练损失是带目标 mask 的速度均方误差，可概括为：

$$
\mathcal L_{\mathrm{gen}}
=\sum_m\lambda_m\,
\operatorname{MSE}_{\mathrm{target},m}
\bigl(v_{\theta,m},\epsilon_m-x_{0,m}\bigr).
$$

各模态分别采样噪声水平，干净条件不进入重建损失；动作在中期训练中使用 10 倍损失权重，补偿归一化动作数值带来的较小逐元素 MSE。它不是把动作幅度乘 10。

这里 $\sigma=1$ 对应噪声端，$\sigma=0$ 对应干净端，因此采样沿噪声水平减小的方向积分。若取一维构造值 $x_0=2,\epsilon=-1$，则 $v^*=-3$；从 $\sigma=0.5$ 的 $x_\sigma=0.5$ 走到 0，要用 $\Delta\sigma=-0.5$，得到 $0.5+(-0.5)(-3)=2$。这个例子说明符号方向，不代表模型只需一步就能生成准确视频。

报告使用分辨率相关的 noise shift 调整噪声分布。用 $u\in[0,1]$ 表示 shift 前的参数，报告写作 $\bar t=1-t$，则重参数化为：

$$
\sigma=\frac{s u}{1+(s-1)u}.
$$

同取 $u=0.5$，$s=1$ 时 $\sigma=0.5$，$s=5$ 时 $\sigma=5/6$，更偏向高噪声端。这是噪声参数的变换，不会把视频的 FPS 乘 5。训练还使用约 10% 的文本 dropout，为 classifier-free guidance 提供条件丢弃训练。[报告 Section 4.2](https://arxiv.org/pdf/2606.02800v4#page=27)

### 7.1 三种约束：能看见什么、需要学什么、允许改什么 {#condition-loss-masks}

“条件保持干净”与“条件不计损失”分别约束输入和监督，不能只做其中一件事：

| 约束 | 解决的问题 | I2V 中的例子 |
| --- | --- | --- |
| Attention mask | 一个 query 可以读取哪些 key | DM 能读取首帧条件；AR 仍读不到 DM |
| Loss mask | 哪些输出进入目标 MSE | 首帧不计重建损失，只监督未来目标 |
| 采样时固定条件 | 哪些 latent 可以随去噪步改变 | 已知首帧保持原值，只更新未来 latent |
{.table-readable}

取两个标量位置：一个已知条件为 5，一个目标的干净值为 2；后者的噪声取 −1，$\sigma=0.5$。模型输入因此为 $[5,0.5]$，只有第二个位置需要生成，其真实速度为 −3。

假设模型对两个位置分别输出速度 $[100,-2]$。正确的目标损失为 $(-2-(-3))^2=1$；条件位置输出的 100 不计入损失。用 $\Delta\sigma=-0.5$ 做一次简化 Euler 更新，得到 **$[5,1.5]$**：条件仍是 5，目标从 0.5 走向 2，但因预测速度有误，还没有到达真值。小脚本会实际核对这组结果。

这个例子也说明，排除条件的重建损失，不等于条件对学习没有作用。目标 token 仍会通过注意力读取条件，目标损失也会训练模型利用这些条件的能力。[报告 Section 4.2](https://arxiv.org/pdf/2606.02800v4#page=27)

## 8. 训练课程：先理解，再生成，再引入动作 {#training}

### 8.1 Reasoner 先建立理解能力

Reasoner 先在约 2200 万样本上预训练，再用约 217 万样本做面向 Physical AI 的 SFT。预训练覆盖 OCR、问答、定位与推理；SFT 提高视频比重，视频－文本约占一半，强化机器人、驾驶和基础设施场景。

Reasoner 使用 next-token prediction；语言骨干、视觉投影器和 ViT 从开始就共同训练。报告没有另做一个长期冻结 VLM、只训练 projector 的对齐阶段。预训练序列上限为 16k token，其中每个样本最多使用 2048 个图像 token 或 8192 个视频 token。

SFT 还混入筛选后的预训练数据，预训练流与 SFT 流的采样预算为 1:4，并加入约 80 万条指令遵循样本。这里控制的是训练访问预算，不能把 1:4 直接当作原始数据集去重后的数量比。

### 8.2 训练好的 Reasoner 初始化 Generator

Generator 使用相同的 Transformer block 结构，从训练好的 Reasoner 权重初始化生成路径，迁移语义和世界知识。在 Generator 预训练阶段，**Reasoner 塔冻结，生成路径参数更新**。

这一步先继承理解权重，再学习连续生成；统一架构仍有明确的训练阶段边界。

### 8.3 预训练学习图像、视频和音频

生成器预训练使用约 7.67 亿张图像、3.477 亿段视频；其中约 1.389 亿视频具有可用音轨。视频与音视频数据有重叠，不能把这些数字直接相加成互不重复的视频总数。

训练覆盖 256p、480p、720p 三个档位和多种比例，固定打包预算为每序列 74,000 token。四类生成模式的正文采样比例为：

| 模式 | 干净视觉条件 | 总体采样比例 |
| --- | --- | ---: |
| T2I | 无视频前缀，图像视为单帧 | 20% |
| T2V | 无视觉前缀 | 56% |
| I2V | 第一张 latent 帧 | 16% |
| V2V | 前两张 latent 帧 | 8% |

在视频样本内部，这对应 T2V／I2V／V2V 为 70%／20%／10%。VAE 的首帧处理使前两张 latent 帧对应前五张原始帧，不能简单解释成两张普通 RGB 帧。

### 8.4 中期训练引入动作与空间控制

中期训练保留原有生成任务，同时提高 Physical AI 高价值数据的覆盖，并加入动作和 video transfer：

| 数据流 | 训练份额 | 新增或强化的能力 |
| --- | ---: | --- |
| 图像 | 10% | 图像生成 |
| 视频 | 32% | T2V、I2V、V2V |
| 视频＋音频 | 8% | 视听对齐 |
| 动作 | 25% | FD、ID、联合策略模式 |
| 通用 transfer | 20% | 边缘、模糊、深度、分割控制 |
| 驾驶 transfer | 5% | world-scenario-map 控制 |
{.table-readable}

Nano 的生成预训练处理 31.05T token，中期训练再处理 2.4T；Super 分别为 17.86T 和 1.9T。这里是训练处理的 token 数，不等于独立样本数，也不保证每个样本只访问一次。

报告中的“比例”还分属不同层次：上表是中期训练流的混合配置；T2V／I2V／V2V 比例是生成模式的选择；动作数据按小时统计的领域分布则描述数据池。它们分别约束**从哪条流取数据、把样本变成什么任务、池里有哪些轨迹**。实际 token 占比还取决于样本长度、打包预算和填充程度，不能把这几种比例互相替代。

<figure class="article-figure" id="fig-cosmos3-curriculum">
  {{< post-image src="assets/training-curriculum.webp" alt="Cosmos 3 原报告训练课程：图像、视频和音频先预训练，动作与 transfer 在中期加入，通用 Nano 和 Super 分别后训练出图像、视频和 DROID 策略版本" >}}
  <figcaption>
    <span class="article-figure__number">图 3</span>
    <span class="article-figure__text">来源：Cosmos 3 v4，Figure 8。数字为各阶段样本规模的概览；不同数据流可能重叠，也不是训练混合权重。右侧三个后训练版本分别形成专用检查点。DROID 原始数据集规模与这里的后训练样本统计属于不同处理阶段。</span>
  </figcaption>
</figure>

### 8.5 后训练产生专用版本

报告给出三条代表性分支：

- **Super-Text2Image：** 先做 20k 步图像 SFT，再对 47 万高质量图文样本做 2k 步精修，强化语义、文字渲染和审美。
- **Super-Image2Video：** 做 10k 次迭代，重点配置为 480p、189 帧、24 FPS，约八秒；还混入 20% T2I image token，保留语义对齐。
- **Nano-Policy-DROID：** 接入本体状态、专用动作输出与低步数采样，适配真实机器人控制。

因此引用某个分数时必须写清检查点。例如 UniGenBench 的 91.36 对应图像专用 Super，而不是所有 Cosmos 3 变体的共同表现。[报告 Sections 3、4](https://arxiv.org/pdf/2606.02800v4#page=14)

## 9. 数据工程怎样影响“物理能力” {#data}

### 9.1 规模之外，还要控制监督质量和覆盖

Reasoner 的语义去重以整个图文／视频对话为单位，再由模型评审完整性、正确性和证据忠实度。保留规则要求三个维度分别过最低阈值，不能用平均分掩盖某一维的严重失败。

生成数据也经历场景切分、去重、语义分类、分辨率与美学过滤、重标注和分片。视频中主要伪影会被拒绝；较轻的文字叠加、运动模糊等则在预训练阶段标注保留。进入中期训练后，质量筛选更严格，并向机器人、驾驶、人体运动和罕见交互倾斜。

动作中期训练覆盖约 **840 万 episodes、6.13 万小时**。按小时统计，第一视角运动约占 67.4%，驾驶 16.3%，机器人 8.7%，相机运动 7.5%。这个分布说明，中期动作先验并非主要来自机器人关节示教。

机器人来源保留成功与失败片段，也保留空闲操作，并在元数据中记录 idle-step 数量，方便采样平衡。这与 DROID 策略后训练中移除失败示教、过滤空闲帧的目标不同：前者需要看见动作后果，后者需要学习可执行的成功行为。

### 9.2 合成数据提供长尾覆盖，也会造成取舍

作者构造五类 SDG 数据：物理交互、机器人场景、自动驾驶、数字人和仓库操作。仿真能系统改变接触、运动、视角与场景要素，用来补充通用视频分布的稀疏区域。

但附录 C.7 的结果不是“加入合成数据后所有指标都提高”。混合 SDG-All 后，PAIBench-G T2V Overall 从 79.67 到 79.77，Robot 分项增加 0.31，Human 分项下降 0.47；各单独来源也有跨领域退步。

这组结果支持按目标领域验证混合方案；总体小幅提分可能同时包含局部收益和退步。

### 9.3 音频要与可见事件相对应

旁白、后期配乐和真实碰撞声同时出现在网页视频里，但它们与画面的因果关系不同。中期音频筛选会区分可见人脸的同步语音、环境声、乐器演奏和非场景背景音乐。

重要的一点是：**音轨经过人声分离或音乐移除后，描述也要重新生成。** 如果继续使用描述原始旁白的字幕，模型就会学到错误的声音条件。最终中期音频池约 1880 万片段，其中约 1280 万为非语音、600 万为同步语音。[报告 Sections 3.2、Appendix C](https://arxiv.org/pdf/2606.02800v4#page=20)

## 10. 推理请求：结构化提示、采样与缓存 {#inference}

### 10.1 Prompt upsampling 是一个独立接口

报告的生成训练使用结构化 JSON caption。推理时，短请求先被扩展为场景、主体、空间关系、时间进程、相机、声音及分辨率等字段，再进入生成器。

在 I2V 中，初始图像是第一帧的视觉证据；扩写必须保留已经看到的主体和布局。FPS、duration、resolution 等控制字段也要与实际采样配置相符。字段齐全并不保证世界描述正确，但使输入假设更容易检查。

评估中的 upsampler 也属于比较协议的一部分。报告包含三种需要分别记录的设置：

| 评估场景 | 提示词处理与采样协议 |
| --- | --- |
| Table 11 图像基准 | Claude Opus 4.7 将提示词改写为结构化格式 |
| 视频自动指标与人评 | Claude Opus 4.6 做结构化改写 |
| Artificial Analysis 图像榜单 | GPT-5.5 扩写与重写，Gemini 3.1 Pro 评审；最多两次重写，可提前停止，按评审分选取最佳图像 |
{.table-readable}

因此 Table 11 的 UniGenBench 分数与 Artificial Analysis 的排名对应不同推理流程。后者包含生成、评审和重写循环，不能将其排名理解成同等成本的单次出图结果。通用生成接口也支持使用 Cosmos 3 Reasoner 扩写提示词，但这不改变上述报告评测采用的配置。[报告 Section 6.2、Appendix B.7](https://arxiv.org/pdf/2606.02800v4#page=94)

### 10.2 不同模式使用不同采样配置

| 检查点／模式 | 去噪步数 | CFG guidance | shift |
| --- | ---: | ---: | ---: |
| Nano／Super 音视频 | 50 | 6 | 10 |
| Super-Text2Image | 50 | 4 | 3 |
| Super-Image2Video | 50 | 6 | 5 |
| FD／ID | 50 | 1 | 5 |
| Nano-Policy-DROID | 4 | 3 | 5 |

这是报告 Table 21 的默认值。Transfer 另区分文本 guidance=3 与 control guidance=1.5。负提示词同样按模式选择，策略和 FD／ID 默认使用空字符串。

CFG 的 scale 也不表示成功概率。以下采用 [Diffusers Cosmos3OmniPipeline 的单权重约定](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L1724-L1750)：记条件分支速度为 $v_{\mathrm{cond}}$，负提示／条件丢弃分支为 $v_{\mathrm{neg}}$，组合为：

$$
v_g=v_{\mathrm{neg}}+g\bigl(v_{\mathrm{cond}}-v_{\mathrm{neg}}\bigr).
$$

$g=1$ 时采用条件预测，$g>1$ 时沿两个分支之差外推。若构造值为 $v_{\mathrm{cond}}=-2$、$v_{\mathrm{neg}}=1$，取 $g=3$ 得到 −8；这不等于把条件速度直接乘 3。空负提示只是负分支的输入选择，也不会自动关闭 CFG。上表保留报告参数，跨框架移植时还须核对 scale 的定义；Transfer 的双权重控制另需按其三分支方案处理。

因此，**步数决定更新几次，CFG 决定怎样组合条件预测，shift 决定怎样安排噪声坐标**。DROID 的四步采样在双分支 CFG 下对应八次逻辑上的 Generator 分支计算；两张 GPU 可在每一步并行处理两个分支，各分支的固定 AR 上下文还能分别缓存。这个计数既不是八次串行的完整模型调用，也不等于动作执行的 15 Hz。[报告 Sections 5.3.1、6.3.1](https://arxiv.org/pdf/2606.02800v4#page=45)

### 10.3 工程优化分别处理哪种瓶颈

Reasoner 激活缓存利用 AR 不读 DM 的依赖结构，避免每个去噪步重复计算固定条件。CFG 并行则把有条件与无条件分支分到两张 GPU，在每步合并预测；它减少墙钟延迟，同时增加硬件占用。

缓存的边界也由依赖关系决定：**固定 AR 条件的各层表示可复用，正在去噪的 DM 表示则持续改变。** DM 每一步会更新整段目标，不能将整套 DM 的 K／V 直接当作语言逐 token 生成时的追加式缓存。即使不解码 RGB，联合预测所需的视频 latent 仍参与去噪计算。

Context parallelism 分摊长序列，批处理提高离线生成吞吐，VAE 解码决定 latent 怎样变成可显示的视频。机器人单请求延迟与大批量离线 tokens/s 需要分开比较。[报告 Sections 5.3、6.3](https://arxiv.org/pdf/2606.02800v4#page=45)

## 11. DROID 策略：从几何动作到可执行关节序列 {#droid-policy}

### 11.1 后训练改变了动作接口

DROID 平台使用 Franka Panda 7 自由度机械臂与 Robotiq 2F-85 夹爪。报告从中期训练的 Nano 出发，重新初始化动作 encoder、动作解码 MLP 和动作 embedding，并对动作相关参数使用 5 倍学习率倍率。

策略接收当前机器人本体状态、任务短指令和三路图像。腕部图像 $360\times640$ 放在上方，两路外部图像各 $180\times320$ 放在下方左右，构成 **$540\times640$ canvas**。

输出为未来 **32 个绝对关节位置动作**，训练中同时预测辅助 RGB 视频。执行频率为 15 Hz，动作块覆盖约 $32/15\approx2.13$ s。这是预测／执行区间，不能据此推算一次网络前向只需要 $1/15$ s。

尤其要注意，策略后训练的绝对关节输出，与中期训练的 10 维单臂几何伪动作不是同一个控制接口。复用知识通过后训练实现，不能把二者不经转换地直接对接。

### 11.2 部署时生成视频，不一定解码视频

推理使用四步采样、CFG=3、shift=5，并跳过 video-latent 到 RGB 的解码，以减少开销。跳过解码保留了联合 latent 生成过程，不能解释成训练和推理都完全移除了视频流。

报告将策略服务器部署在两张 RTX Pro 6000 上，由 Franky 关节位置控制器执行预测动作。实时控制表现取决于模型延迟、动作块执行和反馈调度的组合；单独的 15 Hz 描述不足以给出整套系统的推理吞吐率。[报告 Section 4.2.5](https://arxiv.org/pdf/2606.02800v4#page=31)

### 11.3 一次请求的观测、预测与执行顺序 {#droid-request-cycle}

以“把螺丝刀放到上层架子”为例，一次策略请求可以按下面四步阅读：

1. **取得当前条件。** 读取当前本体状态、三路相机画面和短任务指令，构成该次预测的条件输入。
2. **联合采样未来。** 在固定条件下用四步流采样，生成 32 个动作及辅助视频 latent；这一步得到的是预测。
3. **执行关节目标。** 把动作交给 Franky 控制器，以 15 Hz 执行；跳过 RGB 解码可省去显示视频的开销。
4. **用新观测检查后果。** 后续请求应依据执行后的真实相机和本体状态更新条件，形成闭环。

第 4 步描述闭环控制的反馈要求；报告没有在 Section 4.2.5 中完整给出请求间隔、并行调度或动作块重叠执行细节，因而不能据此画出精确的异步时序。模型预测的视频也不能替代这里的真实反馈。

对实验复现来说，应分别记录**一次请求的模型延迟、动作发送频率、每次执行的块长度，以及重新获取观测的时机**。这四项决定一组预测怎样真正连接到环境。

## 12. 机器人结果：成功率、迁移与视觉一致性分别看 {#robot-results}

### 12.1 RoboLab 的指令粒度会显著影响成功率

Table 19 在 RoboLab-120 上报告不同指令具体程度下的闭环任务成功率：

| 模型 | Vague | Default | Specific |
| --- | ---: | ---: | ---: |
| Cosmos3-Nano-Policy-DROID | 20.6% | 36.8% | 39.7% |
| Nano，直接从 PT-init 做策略后训练 | 16.7% | 28.1% | 30.2% |
| π0.5 | 15.2% | 28.0% | 28.1% |
| DreamZero | 14.9% | 25.7% | 23.9% |

因此 39.7% 是 **Specific 指令的总体成功率**，不是任意提示词条件下的成功率。相对 π0.5 的 Specific 结果高 11.6 个百分点；这与“相对提高 11.6%”是不同表达。

通用策略也没有在每个细分格子都领先。例如 Complex＋Vague 中，Cosmos 为 4.1%，PT-init 为 7.1%，π0.5 为 5.3%。总体优势与局部失败可以同时存在。

### 12.2 动作中期训练提高了新具身适应效率

作者比较 PT-init 与 MT-init：前者尚未经历动作中期训练，后者已经学习跨域动作；后续在相同任务上使用相同适配流程。

LIBERO-10 每个检查点评估 500 次 rollout，成功率如下：

| 初始化 | 500 步 | 1000 步 | 1500 步 | 2000 步 |
| --- | ---: | ---: | ---: | ---: |
| MT-init | 24.6% | 91.4% | 95.8% | 97.4% |
| PT-init | 0.0% | 73.8% | 93.4% | 95.2% |

<figure class="article-figure" id="fig-cosmos3-adaptation">
  {{< post-image src="assets/libero-adaptation.webp" alt="LIBERO-10 后训练成功率：MT-init 在五百步达到二十四点六百分比，一千步达到九十一点四百分比，比 PT-init 更早取得高成功率，后期差距缩小" >}}
  <figcaption>
    <span class="article-figure__number">图 4</span>
    <span class="article-figure__text">数据来自 Cosmos 3 v4，Table 20。横轴为后训练迭代数，纵轴为闭环成功率，每个检查点 500 次 rollout。连线仅连接四个报告点；原表未报告跨训练种子误差范围，因此不添加推断的误差条。</span>
  </figcaption>
</figure>

早期差距最大，后期缩小，更直接支持“更好的适配起点”。这不是零样本跨具身控制，也不能把 2000 步的 97.4% 当成全部机器人任务成功率。

### 12.3 真机与榜单结果保留时间和协议

报告展示了将螺丝刀放到架子上，以及将螺丝刀和手套放入紫色容器的真机任务。

<figure class="article-figure" id="fig-cosmos3-real-robot">
  {{< post-image src="assets/droid-real-robot.webp" alt="Cosmos 3 原报告真机动作序列：Franka 机械臂将螺丝刀放到上层架子，并按语言指令将螺丝刀和手套放进紫色容器" >}}
  <figcaption>
    <span class="article-figure__number">图 5</span>
    <span class="article-figure__text">来源：Cosmos 3 v4，Figure 25。两组连续帧说明专用策略能在物理 DROID 平台执行语言条件操作；画面本身不提供全部任务的重复试验成功率。</span>
  </figcaption>
</figure>

RoboArena 使用真机双盲 A/B 偏好比较，报告中的第一名对应 **2026-05-30** 的榜单快照；MolmoSpaces 的 All Combined 结果对应 **2026-06-20**，oracle 成功率为 39.0%。它们与 RoboLab 的模拟任务成功率属于不同评估协议，不宜混成一个统一分数。

### 12.4 预测画面是否与自己输出的动作一致

附录 E.5 取模型联合预测的动作块，在 RoboLab 从同一初始状态执行，再将仿真画面与预测视频比较。左外部相机 PSNR 为 23.19 dB，腕部相机为 17.33 dB。

<figure class="article-figure" id="fig-cosmos3-consistency">
  {{< post-image src="assets/video-action-consistency.webp" alt="Cosmos 3 原报告比较同一预测动作块对应的 RoboLab 仿真画面与联合预测视频，分别展示腕部和左侧相机的 Sim Env 与 Pred 两行" >}}
  <figcaption>
    <span class="article-figure__number">图 6</span>
    <span class="article-figure__text">来源：Cosmos 3 v4，Figure 37。Sim Env 是执行该动作块的仿真观测，Pred 是模型与动作联合预测的视频。这个配对控制了初始状态和动作，检查动作－视频一致性；它与仅挑选看起来合理的视频不同。</span>
  </figcaption>
</figure>

这是一个很重要的补充检查：联合输出不能只各自看起来合理，还应描述同一未来。不过 PSNR 仍主要衡量短时视觉一致性，不能直接替代力、接触、长期稳定性或任务完成评估。

这里还有一个有用的泛化条件：附录 E.5 说明策略只在 DROID 后训练，一致性测试则放到 RoboLab。它检查的是进入留出环境后，动作与预测画面能否继续描述相同的变化；并未把 RoboLab 的真实未来帧作为策略条件。

## 13. 生成与推理评估：不要用一个“SOTA”覆盖所有指标 {#generation-results}

### 13.1 理解能力有明显领域差异

报告的 Reasoner 评估分成通用、机器人、智能基础设施和驾驶四组。按概览表，Super 的组平均分别为 73.7、57.8、62.6、79.3，Nano 为 69.6、55.1、61.0、76.0。

这些数值是异质基准的组平均，不是同一数据集上的四类准确率。Super 的通用平均低于表中的 Gemini 3.1 Pro 77.5，机器人平均也略低于其 58.2；基础设施和驾驶组则更强。驾驶组包含两个内部事件分类基准，不能把驾驶视频问答得分直接解释成自动驾驶闭环安全性。

### 13.2 文生图：语义、英文和中文文字是不同能力

UniGenBench 扩展版包含原始 600 条提示和 570 条 Physical AI 提示。图像专用 Super 的总体得分为 91.36，通用 Super 为 87.33，Nano 为 84.61。这里统计的是评估条件满足度，不是机器人任务成功率。

图像专用模型的 CVTG-500L 英文 GNED 为 80.88，中文 CVTG-102ch GNED 为 32.02；相同表中，Qwen-Image-2512 的中文 GNED 为 46.33。报告将 GNED／PNED 定义为字符级准确性指标，越高越好。总体语义得分突出，并不意味着中文文字渲染同样领先。

### 13.3 视频：自动指标、人评和物理预测各有侧重

| 基准／协议 | 代表结果 | 衡量什么 |
| --- | --- | --- |
| PAIBench-G T2V／I2V | Super 80.0／82.8 | 视觉质量与领域条件正确性 |
| RBench I2V | Nano 58.4%、Super 58.1% | 机器人场景中的生成任务与视觉质量 |
| Physics-IQ I2V／V2V | Super 43.8／59.7 | 与真实物理延续的时空一致性 |
| Physics-IQ＋WMReward BoN | Super 48.9／63.4 | 增加候选生成与重排后的结果 |
| Cosmos HUE T2V／I2V | Super 89.3／89.6 | 原子问题的人类验证 |
| Human World Bench | Super 71.9 | 人体运动指令遵循与物理可行性 |
{.table-readable}

RBench 的“任务完成”是在评估**生成视频所描绘的任务**，不是让物理机器人执行后的成功率。Physics-IQ 则要求生成结果接近记录到的真实后续；V2V 看到三秒条件视频后预测后五秒，信息量明显多于 I2V 单帧条件，因此两种分数不宜混排。

BoN 是 best-of-N：生成候选再用奖励模型挑选，额外推理成本不能省略。比较时应使用相同条件模式，并区分直接采样和重排。

HUE 使用 Yes／No／Unclear 原子问题，两人独立标注并对分歧复核；Unclear 留在分母，按未满足标准处理。报告中真实视频参考的 T2V／I2V HUE 为 93.6／94.4，Super 为 89.3／89.6。与此同时 Veo-3.1 T2V 为 91.3，所以“最佳开放权重结果”也不等于超过所有闭源模型。具体的汇总方式与误差范围见[评分核对](#evaluation-aggregation)。[报告 Tables 11–14、Appendix F](https://arxiv.org/pdf/2606.02800v4#page=54)

### 13.4 音频语义对齐强，不代表音质各项都最好

Cosmos-SoundBench 在 144 条非语音提示上区分声音语义、视听事件对齐与感知音质，每条提示生成 5 个随机种子样本。Nano 的综合 AVQ 为 7.34，Super 为 7.31；报告中 Seedance-1.5-Pro 为 7.64、Veo-3.1 为 7.45。

语义视听 SAV 组合声音语义、视听对齐与关键视觉证据：

$$
\mathrm{SAV}=0.60\,\mathrm{SA}+0.30\,\mathrm{AVAlign}+0.10\,\mathrm{VisualSupport}.
$$

Production Quality（PQ）则另外衡量声音清晰度、保真度、动态范围等。Nano 的 SAV=8.35、AVAlign=8.16 更突出，PQ 为 6.32。语义与对齐表现较强，感知音质仍有提升空间。

<figure class="article-figure" id="fig-cosmos3-audio-alignment">
  {{< post-image src="assets/audio-event-alignment.webp" alt="Cosmos 3 报告中的锤击音视频对齐：红蓝绿橙四个接触画面由标记连接到频谱瞬态，灰色非接触画面对应两次锤击之间的区间" >}}
  <figcaption>
    <span class="article-figure__number">图 7</span>
    <span class="article-figure__text">来源：Cosmos 3 v4，Figure 20。上方是同一生成样本的选取帧，下方是所生成音频的频谱图；竖线把锤击接触与声学瞬态对应起来，灰色帧保留非接触时刻作比较。原图未标时间和频率刻度，因此这里只检查事件对齐的定性现象，不能读取精确延迟，也不代表完整测试集的统计结果。</span>
  </figcaption>
</figure>

这个例子把“场景里有锤击声”进一步分解为“声音是否跟接触事件同时发生”。前者检查声音种类，后者检查时序与声源归属，PQ 又评价声音本身的感知质量。[报告 Section 6.2.3、Table 15](https://arxiv.org/pdf/2606.02800v4#page=61)

### 13.5 动作条件视觉预测也要区分任务和单位

经过相同下游适配后，DROID 前向动力学的 Super 从 PT-init 的 22.69 dB 提高到 MT-init 的 26.04 dB；Nano 从 23.24 到 25.52 dB，比较基线 Ctrl-World 为 22.99 dB。

该实验给定初始图像与长度 16 的末端动作块，预测后续 16 帧。它使用的接口与 DROID policy 的 32 个绝对关节动作不同。相机控制使用 RRE／RTE／ATE 检查轨迹，驾驶 ID 使用车辆位姿误差；这些指标与 PSNR、闭环成功率各自回答不同问题。

PSNR 的定义为：

$$
\mathrm{PSNR}=10\log_{10}(L^2/\mathrm{MSE}).
$$

其中 $L$ 是像素动态范围：对归一化到 $[0,1]$ 的像素，$L=1$；对 8 bit 像素，$L=255$。因此 dB 差值属于对数尺度，不能把 26.04 dB 与 22.69 dB 相除，得到“物理能力提升倍数”。画面轻微错位、相机运动和物体纹理都会影响像素 MSE；比较还需保持视角、分辨率、归一化和时间对齐一致。

### 13.6 HUE 的分母与误差范围怎样核对 {#evaluation-aggregation}

附录 F 的模型总分按所有视频的问答汇总。令 $Y_v$ 是视频 $v$ 的 Yes 数，$N_v$ 是该视频的全部问题数，则：

$$
\mathrm{HUE}_{\mathrm{model}}
=100\%\times\frac{\sum_v Y_v}{\sum_v N_v}.
$$

No 与 Unclear 都进入 $N_v$，但不增加 $Y_v$。两名标注者的回答经一致性与复核流程形成一个最终答案，也不应被重复算成两个观测。用一组构造值说明汇总差异：

| 视频 | Yes | No | Unclear | 单视频 HUE |
| --- | ---: | ---: | ---: | ---: |
| 甲 | 4 | 1 | 5 | 40% |
| 乙 | 2 | 0 | 0 | 100% |

正确汇总是 $(4+2)/(10+2)=50\%$。直接平均两个视频会得到 70%，把 Unclear 删掉则会得到约 85.7%；这两种做法都改变了报告规定的计分方式。视频中的事件被遮挡、难以判断，或者根本没有发生时，模型不能靠删除难题提高分数。

报告每个模型使用 100 条提示、每条 5 个随机种子，共 500 段视频；正文及可靠性段落写每段最多 20 题，但附录 F 的标注协议段又写最多 16 题。**实际重算应使用逐视频题数，不能只按某个上限推断总题数。**

附录的约 ±0.6 个百分点区间，来自将每条二元问答视为独立 Bernoulli 观测的近似。同一提示词的多个种子、同一视频的多个问题可能相关，因此独立性假设需要另外检验。本文据此认为：仅凭表中 0.1 分的差距，还不足以确认稳定的模型优势；更完整的比较需要逐提示词结果，并保留这些分组关系来估计不确定性。[报告 Appendix F，Equations 12–13](https://arxiv.org/pdf/2606.02800v4#page=110)

下载 [evaluation_lab.py](evaluation_lab.py)，运行以下命令可复算这组分母与汇总例子，以及 [FPS 消融](#142-时间位置调制与文本-fps-控制互补)中的 Composite 例子。两者均使用构造值，只核对公式，不重新测量论文模型：

```bash
python3 -B evaluation_lab.py --self-test
```

## 14. 关键消融：哪些结论得到了更直接支持 {#ablations}

### 14.1 更好的 Reasoner 有助于领域生成

附录 E.1 在相同 Nano 规模下比较 Cosmos Reasoner 与 Qwen3-VL 理解塔，两个实验的 Generator 都从头训练。T2V 的 Domain 分数从 73.7 提高到 75.7，Robot 分项从 66.5 到 71.3；Quality 均为 73.0。

这组实验更直接支持“领域理解上下文改善领域生成”。它改变理解塔，并把两组 Generator 都从头训练，因此没有单独测量[第 8.2 节](#82-训练好的-reasoner-初始化-generator)中“复制 Reasoner 权重初始化 Generator”这一步的增益。要区分这两种贡献，还需要在相同理解塔、数据与预算下比较生成塔的复制初始化与随机初始化。

### 14.2 时间位置调制与文本 FPS 控制互补

附录 E.2 比较四种时间控制，在四个 FPS 档位上报告 Composite：

| 设置 | Composite |
| --- | ---: |
| 无控制 | 8.51 |
| 仅文本 FPS／duration | 9.28 |
| 仅 MRoPE FPS 调制 | 9.63 |
| 两者同时使用 | 9.81 |

感知画质的变化较小，主要收益来自运动保真度。这说明仅在提示词中写 FPS，与将物理时间编码进 token 位置，提供的是互补约束。

Table 29 的 Composite 是在 10、15、24、30 FPS 四个档位内先计算 $\mathrm{VQ}\times\mathrm{MF}$，再对档位求平均；VQ 是 DOVER 画质分，MF 是运动保真度。它通常不等于平均 VQ 与平均 MF 的乘积。若两个档位的构造值分别为 $(10,0.9)$ 与 $(20,0.5)$，正确结果是 $(9+10)/2=9.5$，而先分别平均再相乘会得到 $15\times0.7=10.5$。复算时需要档位级明细，不能只用表中的两个均值。

### 14.3 三种动作模式共享知识，也发生竞争

附录 E.4 在 PushT 上使用 Edge，比较各模式分别训练 2k 步，与联合 FD／ID／policy 训练 6k 步，即每种模式获得相同优化步数：

| 设置 | FD PSNR ↑ | ID MSE ↓ | Policy coverage ↑ |
| --- | ---: | ---: | ---: |
| 单模式 | 27.13 | $1.11\times10^{-3}$ | 74.1% |
| 联合模式 | 26.22 | $3.09\times10^{-4}$ | 77.3% |

联合训练明显改善 ID 和策略 coverage，却使 FD PSNR 降低 0.91 dB。Coverage 在这里是 T 形物块与目标区域的覆盖度，不是前面 RoboLab 的任务成功率。报告使用 50 个初始状态，每个初始状态运行 10 次 rollout，共 500 次，并聚合覆盖度；不能把这个连续数值直接解释成“500 次里成功了多少次”。

动作模式之间可以迁移，也会竞争；同样的问题还发生在不同具身域之间。[报告 Appendix E](https://arxiv.org/pdf/2606.02800v4#page=107)

### 14.4 跨域迁移矩阵有方向，也有预算条件 {#domain-transfer}

Figure 28 的**行是被评估的目标域，列是额外加入训练的域**，对角线是单域基线。非对角线使用两域各 50% 的混合，训练 4000 次迭代；单域基线训练 2000 次，以匹配目标域的训练暴露量。这控制了目标域预算，但混合实验使用了更多总迭代和额外来源数据。

选取其中 FD PSNR 的四个格子：

| 评估域（行） | 加入域（列） | 单域 → 混合 PSNR | 图内 Δ 标注 |
| --- | --- | ---: | ---: |
| Camera Motion | Autonomous Vehicle | 11.96 → 12.82 dB | +0.86 dB |
| Autonomous Vehicle | Camera Motion | 17.87 → 17.83 dB | −0.05 dB |
| WidowX-250 | Google Robot | 15.63 → 17.02 dB | +1.39 dB |
| Google Robot | WidowX-250 | 19.70 → 20.09 dB | +0.39 dB |
{.table-readable}

数值与 Δ 均照图抄录；图示 Δ 与已圆整分数相减有 0.01 dB 的差异，本文保留原标注。第一对域说明收益可以不对称：驾驶数据改善相机运动预测，反向混合却略降。第二对域在两个方向都有收益，幅度也不同，不能把一对域压缩成一个对称的“相似度”。

Figure 29 进一步比较第一视角人体运动与 AgiBot：人体数据混入后，AgiBot FD PSNR 从 23.07 到 23.20 dB；先做人体运动 warmup 再适配 AgiBot，在 5k 个 AgiBot 训练步处比直接适配高 0.94 dB。这个结果支持人体运动提供可迁移的视觉－动作先验；曲线横轴是后续适配步数，评估仍是视觉 FD，不能据此宣称同等总训练成本下获得了零样本机器人控制。[报告 Section 6.2.5，Figures 28–29](https://arxiv.org/pdf/2606.02800v4#page=71)

## 15. 训练工程：长序列怎样真正跑起来 {#infrastructure}

74k token 的多模态序列需要在样本长度、参数显存、激活显存和 attention 计算之间协调。报告的主要工程措施可以按瓶颈理解：

| 瓶颈 | 处理方式 | 作用 |
| --- | --- | --- |
| 多数据源长度差异 | JointDataLoader 与 look-ahead packing | 减少 padding 和流失衡 |
| 两种 attention mask 并存 | Two-way flat attention | AR 因果与 DM 全量注意分别用变长 kernel |
| 参数／梯度／优化器状态 | HSDP | 同组内分片，组与组之间复制 |
| 长序列激活 | Ulysses context parallelism | 在 token 与 head 分片之间 all-to-all |
| 反向重算代价 | Selective activation checkpointing | 优先保存昂贵的 attention 输出 |
| 主机与小算子开销 | 编译 Transformer block、CUDA Graph | 减少重复 launch 与调度成本 |
{.table-readable}

其中 two-way flat attention 对每层发起两种变长注意力调用，并通过样本边界让每个 DM query 只看自己的 $[R_i,G_i]$。报告在 Nano 上给出相对 FlexAttention 基线约 22% 的端到端训练吞吐提升；保存 attention 输出的选择性重算策略另报告约 13%。它们来自特定配置的消融，不能直接相乘作为总加速保证。

规模也不能忽略：报告的主要生成预训练使用 Nano 1024 张、Super 2048 张 GB200。Nano 的“较小”是相对 Super 而言；约 16B 双塔权重、长视觉上下文与视频采样仍有明显成本。

这与 [SuperMimic](../supermimic/)的高效动力学学习目标很不一样：SuperMimic 在低维身体状态上为策略提供可微替代动力学；Cosmos 3 则承担高维多模态生成、理解与动作表示，二者的训练预算和部署职责没有直接可比性。

## 16. 怎样判断能力边界与复现条件 {#limits}

Cosmos 3 为理解、可控生成和动作学习提供共同的架构与训练起点，并展示了后训练后的真实机器人部署。评估贡献时，可以把已有对照与还需补充的证据并列：

| 要判断的贡献 | 报告提供的证据 | 仍需单独验证什么 |
| --- | --- | --- |
| 更好的理解上下文帮助生成 | Appendix E.1 更换理解塔，两组生成塔均从头训练 | 复制生成塔初始权重的独立收益 |
| 动作中期训练改善适配起点 | PT-init／MT-init 使用相同下游适配流程比较 | 更长的前期训练、数据混合与单个动作来源各自贡献多少 |
| 多种动作模式可以共享知识 | PushT 联合 FD／ID／policy 改善 ID 与 coverage，FD 略降 | DROID 中辅助视频目标的独立收益；需固定其他设置再移除该目标 |
| 联合预测的动作与视频描述同一未来 | RoboLab 中执行同一动作块，再与预测视频计算 PSNR | 力、接触、长期状态和任务完成的独立评估 |
| 生成结果符合部分物理现象 | Physics-IQ、HUE 与定性片段 | 碰撞、守恒定律及分布外场景能否精确、稳定地满足约束 |
{.table-readable}

前两行尤其容易混淆：**更好的后训练起点，描述的是后续适配效率；完整训练成本还包含获得这个起点的全部前期工作。** 同样，统一架构与数据混合可以产生正迁移，也会出现领域或指标退步，配比应按目标分项选择。

官方[模型卡的限制说明](https://huggingface.co/nvidia/Cosmos3-Nano#limitations)也列出了时间不一致、接触不精确、动作漂移与分布外退化。阅读模型卡时还要区分通用接口说明与具体策略后训练配方。

### 报告中的数字差异怎样处理 {#报告中需要按上下文核对的三处数字}

这份版本有少量正文、图注或表格不一致，本文采用以下口径：

- **生成模式比例：** Figure 10 图注出现“均匀采样”的说法，但正文明确给出 T2I／T2V／I2V／V2V=20／56／16／8%，图内视频比例也为 70／20／10；本文采用明确比例。
- **RoboLab DreamZero：** Table 19 的 Specific 总分为 23.9%，紧随表后的文字写为 25.2%；本文比较表采用 Table 19，并不将两个值混合。
- **吞吐测试 GPU 数：** Section 5.2.8 正文写 Nano／Super 为 1024／2048 张 GB200，Table 8 图注则写 2048／4096；因此不从该吞吐表反推完整训练墙钟或统一成本。
- **HUE 每段题数：** 正文及可靠性段落写最多 20 题，附录标注协议段写最多 16 题；[评分核对](#evaluation-aggregation)按实际题数解释分母。
- **音频质量记号：** Section 6.2.3 的最终 AVQ 公式使用 AQ，邻近指标说明与 Table 15 列名则使用 PQ。将表中 SAV 与 PQ 各取一半，可以在显示精度下复算 Nano 约 7.34、Super 7.31；本文保留表值，复现时仍需确认公式的 AQ 是否对应 PQ。

发现这种差异时，适合保留版本、表号和评估条件，而不是把不一致数字做平均。复现时应进一步对照固定版本的配置与运行记录。

## 阅读自测与验收

可先回答问题，再展开下方答案核对。

1. **Cosmos 3 的 MoT 与稀疏 MoE 是同一种结构吗？**
2. **AR token 能读取正在去噪的 DM token 吗？**
3. **为什么同一张图还要有 ViT 和 VAE 两种编码？**
4. **人形动作的 29 维一定是 29 个关节吗？**
5. **$a_t$ 表示哪个时间区间？**
6. **音频 25 TPS、动作 15 Hz 怎样共享一秒时间？**
7. **15000 的位置间隔意味着新增 15000 个 token 吗？**
8. **干净条件 token 也加入目标重建损失吗？**
9. **生成预训练时 Reasoner 一直和 Generator 同时更新吗？**
10. **通用单臂伪动作能直接当 DROID 策略的关节输出吗？**
11. **四次去噪与 15 Hz 是同一种频率吗？**
12. **跳过视频解码意味着取消了联合视频预测吗？**
13. **RoboLab 的 39.7% 与 LIBERO 的 97.4% 能直接比较吗？**
14. **BoN 后的 Physics-IQ 分数与直接采样有相同成本吗？**
15. **联合 FD／ID／policy 训练在每个指标上都更好吗？**
16. **视频动作一致性足以证明物理控制可靠性吗？**
17. **不计条件重建损失，就能保证条件保持不变吗？**
18. **Table 11 图像分数和带评审循环的图像榜单是同一协议吗？**
19. **HUE 的 Unclear 应从分母删掉吗？**
20. **迁移矩阵中 A 帮助 B，就说明 B 同样帮助 A 吗？**

<details id="cosmos3-self-check-answers">
<summary>展开 20 题答案</summary>

1. 不是；这里按 AR／DM 路径使用不同参数，通过联合注意力交互。
2. 不能；AR 保持自身因果结构，DM 可以读取完整 AR 条件。
3. 理解特征与可重建的生成 latent 承担不同任务。
4. 不是；报告的中期几何表示是 ego、末端和抓握组件的组合。
5. 从视觉状态 $v_{t-1}$ 到 $v_t$ 的转移。
6. 分别使用 6／25 和 6／15 的位置增量，一秒都映射为 6 个位置单位。
7. 不意味着，它是 AR 与 DM 之间的坐标偏移。
8. 不加入；损失只覆盖需要生成的目标。
9. 该阶段 Reasoner 冻结，只更新生成相关参数。
10. 不能；策略后训练重新接入绝对关节位置接口和本体状态。
11. 不是；前者是每次采样的步数，后者是动作执行采样频率。
12. 不意味着；它省去 latent 到 RGB 的解码开销。
13. 不能；任务、指令、适配过程和评估协议不同。
14. 没有；它增加候选生成和奖励模型重排。
15. 不是；PushT 中 ID 与 coverage 提高，FD PSNR 下降。
16. 不足以；还需要独立验证闭环任务、长期状态和控制接口。
17. 不能；采样还应固定条件值，只更新目标 latent。
18. 不是；提示词改写、候选生成与选择方式要分别记录。
19. 不应；它按未满足标准计数，总分按全部视频的问答汇总。
20. 不能；行是评估域，列是加入域，两个方向可能有不同幅度甚至相反的效果。

</details>

## 参考资料与继续阅读

- [Cosmos 3 v4 报告](https://arxiv.org/abs/2606.02800v4)：架构见 Section 2，数据与训练见 3、4，工程见 5，结果见 6，关键机制消融见 Appendix E。
- [NVIDIA 项目页](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)与[Cosmos 代码入口](https://github.com/nvidia/cosmos)：查看演示、实现和当前发布材料。
- [Cosmos3-Nano 模型卡](https://huggingface.co/nvidia/Cosmos3-Nano)、[Super 模型卡](https://huggingface.co/nvidia/Cosmos3-Super)、[Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)：核对检查点、输入输出和运行方式。
- [Transformer 与 Attention](../transformer-attention/)：补充因果 mask、QKV、位置编码与缓存。
- [VLA、世界模型与数据链](../vla-world-model-data/)：理解示教、视频预测与动作监督怎样连接。
- [SuperMimic 详解](../supermimic/)：对照低维可微动力学模型与高维多模态世界模型的不同用途。
- [Video-to-Data](../video-to-data/)：进一步核对从视觉轨迹到可执行控制之间的时间与接口转换。
