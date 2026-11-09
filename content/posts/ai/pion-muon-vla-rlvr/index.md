---
title: "Pion 详解：Muon 在 VLA 与 RLVR 中的谱失效与高通修正"
date: 2026-10-30
lastmod: 2026-11-09
draft: false
tags: ["Muon", "Pion", "VLA", "RLVR", "GRPO", "Paper Notes"]
categories: ["人工智能"]
authors: ["chase"]
summary: "推导 Muon 与 Pion 的谱变换，解释 VLA 的低有效秩、RLVR 的低信噪比与分头更新；用交互实验和逐步代码检查动量、衰减、ε 与 BF16，并核对官方配置和论文实验。"
description: "推导 Muon 与 Pion 的谱变换，解释 VLA 的低有效秩、RLVR 的低信噪比与分头更新；用交互实验和逐步代码检查动量、衰减、ε 与 BF16，并核对官方配置和论文实验。"
contentLanguage: "zh-CN"
math: true
toc: true
images: ["assets/update-flow-v2.webp"]
imageZoom: true
reading_prerequisites: "梯度下降、矩阵奇异值分解与注意力；了解 VLA 和 GRPO 会更容易连接实验"
reading_focus: "先看谱变换改变了什么，再推导两个多项式；区分整矩阵与分头模式，最后按模块分配、训练预算和评估分母理解实验。"
related_posts:
  - "/posts/ai/transformer-attention"
  - "/posts/ai/ppo-dpo-grpo"
  - "/posts/ai/rlhf-rlvr-rlcd"
  - "/posts/ai/vla-evolution"
---

把一个在语言模型预训练中表现良好的优化器，直接用于机器人动作学习或推理强化学习，为什么可能变差？《[Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR][paper]》把问题落在了**梯度矩阵的奇异值如何被改写**上。

Muon 以将动量矩阵的非零奇异方向拉到相近尺度为目标。这种更新方式在某些预训练设置中有利，但如果主要信息集中在少数方向，其余方向大多是噪声或微弱扰动，统一放大就可能破坏更新。论文提出的 **Pion** 保留矩阵多项式计算方式，将谱变换改成“先提升，再抑制”：保留较强方向，压低较弱方向。提升次数也可以设为零，此时直接执行抑制。

这项工作包含两个互相关联的设计：**高通谱变换控制保留哪些奇异方向；分头模式决定在哪个矩阵块上独立进行归一化与变换。** 前者是主要方法，后者用于适应 RLVR 中已有的注意力头结构。

本文依据 **43 页、arXiv:2605.19282v1、2026 年 5 月 19 日预印本**。作者为 Chongyu Fan、Gaowen Liu、Mingyi Hong、Ramana Rao Kompella 和 Sijia Liu。公式与论文实验以该版本为准；工程入口另核对[官方仓库的固定提交 `8796501`][code]。项目的[交互介绍][project]可辅助观察谱变换。

阅读路线：

- 想先理解方法：阅读[矩阵更新](#muon-matrix-update)、[两个多项式的推导](#pion-polynomials)和[具体数值例子](#spectral-example)。
- 关心机器人学习：重点看[VLA 的有效秩](#vla-effective-rank)、[梯度与动量的秩](#vla-gradient-rank)、[模块分工](#optimizer-assignment)与[仿真、真机结果](#vla-results)。
- 关心推理强化学习：重点看[梯度信噪比](#rlvr-snr)、[有限样本的 SNR 估计](#snr-estimation)、[相关观测的反例](#snr-correlated-groups)、[组内优势与梯度](#grpo-group-gradient)、[分头更新](#per-head)与[RLVR 消融](#rlvr-results)。
- 想动手：先运行[标准库谱计算](#runnable-example)，再跟踪[动量、零梯度与衰减的逐步更新](#stateful-step)；[平坦谱幅度](#flat-rank-study)和[多进程补齐](#vla-padding-recovery)的对照在该节折叠展开。
- 想比较调度与输入尺度：使用[交互谱实验](#interactive-spectrum)，再看[两个头的反例](#per-head-counterexample)和[数值精度说明](#numeric-precision)。

<figure class="article-figure" id="fig-pion-update-flow">
  {{< post-image src="assets/update-flow-v2.webp" alt="Pion 参数更新流程：梯度先累计为动量，再选择整矩阵或分头路径；对每个选定块做 Frobenius 归一化、kp 次提升和五减 kp 次抑制，分头结果拼回后更新权重" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">两条路径是替代选择，每个选定块各执行五次多项式。下方三个块只表示独立分头处理，不代表具体模型的头数。VLA 动作矩阵与 RLVR 注意力是论文中的用途示例；RLVR 的普通 MLP 仍可使用整矩阵模式。图中省略可选 Nesterov 混合、权重衰减、尺寸缩放、参数排除规则与分布式通信。</span></figcaption>
</figure>

## 1. Muon 在矩阵上做了什么 {#muon-matrix-update}

### 1.1 从梯度与动量开始

对权重矩阵 $W\in\mathbb{R}^{m\times n}$，普通梯度下降沿负梯度更新。Muon 先累计矩阵动量，论文采用的简化记法是：

$$
M_t=\mu M_{t-1}+G_t,
\qquad G_t=\nabla_W L_t.
$$

随后不是直接使用 $M_t$，而是对它做矩阵谱变换：

$$
W_t=W_{t-1}-\eta\,\operatorname{msign}(M_t).
$$

这里的 `msign` 指矩形矩阵的极分解方向。把 $M$ 写成**只包含严格正奇异值**的紧致 SVD：

$$
\begin{aligned}
M&=U\Sigma V^\top,\\
\Sigma&=\operatorname{diag}(\sigma_1,\ldots,\sigma_r),
\quad \sigma_i>0.
\end{aligned}
$$

则：

$$
\operatorname{msign}(M)=UV^\top.
$$

也就是说，原来大小不同的非零奇异值都被替换为 1，奇异方向不变。这里不是对每个矩阵元素取正负号，也不是要求权重 $W$ 自身正交。若矩阵不满秩，零奇异方向不应被凭空补成非零；逆平方根表达式也需在非零子空间理解，或使用相应的广义逆。

从优化几何看，$-UV^\top$ 是谱范数约束 $\|D\|_2\le 1$ 下，使线性项 $\langle M,D\rangle$ 最小的一个方向。

### 1.2 为什么称为统一谱白化

考虑一个有四个正奇异值的矩阵：

$$
\Sigma=\operatorname{diag}(0.8,0.4,0.2,0.02).
$$

原来最弱方向只有最强方向的 $1/40$。理想极分解将两者都变成 1，消除了这个幅度差。是否有利，取决于弱方向到底是什么：它可能是值得探索的有效梯度，也可能主要来自采样误差、数值扰动或不适合放大的更新。

因此，**小奇异值不自动等于噪声，大奇异值也不自动等于正确方向**。论文的机制解释依赖其训练场景中“信息较集中、尾部更容易受噪声支配”的结构；它不是对任意梯度矩阵的信号识别定理。

### 1.3 实际 Muon 用有限次多项式近似

每步求完整 SVD 代价较高。Muon 使用 Newton–Schulz 风格的奇次矩阵多项式。先做 Frobenius 范数归一化：

$$
X_0=\frac{M}{\|M\|_F+\epsilon}.
$$

全零矩阵约定直接返回零；非零矩阵按上式归一化，使初始奇异值位于 $[0,1]$，并满足平方和不超过 1。之后重复：

$$
X\leftarrow aX+bXX^\top X+cX(X^\top X)^2.
$$

论文所用的常见 Muon 系数为：

$$
(a,b,c)=(3.4445,-4.7750,2.0315).
$$

**有限五步并不会把每个正奇异值精确变成 1。** 例如该多项式在输入 1 时，一步输出为 0.701；多步输出在目标尺度附近变化，还可能超过 1。后文数值图画的是实际五步计算，不是理想阶跃替代。必须区分“Muon 的极分解目标”和“特定系数、特定步数的实际谱响应”。

### 1.4 逐坐标自适应与矩阵谱变换 {#optimizer-comparison}

AdamW 也是自适应优化器，但其重标度依据与 Muon／Pion 不同：

| 方法 | 重标度依据 | 更新的主要特点 |
| --- | --- | --- |
| [AdamW](https://arxiv.org/pdf/1711.05101v3#page=3) | 每个坐标的梯度一阶、二阶矩 | 按坐标调整步幅，权重衰减与梯度更新解耦 |
| Muon | 当前动量矩阵的奇异方向 | 用有限步多项式近似极分解，让非零方向趋向相近尺度 |
| Pion | 当前动量矩阵的归一化奇异值 | 通过提升／抑制调度，偏向保留较强方向 |
{.table-readable}

AdamW 的二阶矩是梯度平方的历史统计，不是 Hessian。它的逐坐标缩放一般也会改变矩阵的奇异方向，因此不能由后文同一条标量谱响应曲线概括。Muon 与 Pion 在精确算术下保留输入矩阵的奇异方向，改变各方向的系数；两者的差别主要在于怎样处理谱尾。

## 2. 为什么 VLA 的动作模块可能不适合统一白化 {#vla-effective-rank}

### 2.1 VLA 的不同模块面对不同结构

VLA 将视觉观测、语言指令与机器人动作连接起来。论文考察了两类动作学习方式：

- VLA-Adapter 通过 $\ell_1$ 回归预测动作；
- VLANeXt 使用 flow matching 学习连续动作生成。

两种目标在[附录 A.1](https://arxiv.org/pdf/2605.19282v1#page=17)中分别写为动作回归与向量场回归。前者最小化预测动作与示范动作的 $\ell_1$ 距离；后者先取示范动作 $a_1$ 与高斯噪声 $a_0$，再从 0 到 1 均匀采样插值变量 $t$，构造：

$$
a_t=(1-t)a_0+t a_1,
\qquad u=a_1-a_0.
$$

$$
L_{\mathrm{FM}}=\mathbb{E}\!\left[
\|v_\theta(a_t,t\mid x,c)-u\|_2^2
\right].
$$

这里 $x,c$ 是视觉观测与语言条件，$u$ 是插值轨迹对 $t$ 的目标变化率；$v_\theta$ 是生成过程中的向量场，不是机器人执行时测得的物理速度。两种目标的梯度形成方式不同，因此同时测试它们，比只验证一种动作损失更有说服力。

视觉与语言模块处理较复杂的表征，动作模块最终输出则受到机器人动作空间约束。不过，“动作只有 7 维”并不直接证明任意中间权重梯度的秩都不超过 7：批量、时间、网络结构与动作分块都会影响梯度。论文给出的关键证据是训练过程中**实际观测到的模块梯度有效秩差异**。

### 2.2 有效秩如何定义

论文采用奇异值分布的熵有效秩：

$$
\begin{aligned}
p_i&=\frac{\sigma_i}{\sum_j\sigma_j},\\
H(p)&=-\sum_i p_i\log p_i,\\
\operatorname{erank}(G)&=\exp(H(p)).
\end{aligned}
$$

这里用的是**奇异值归一化**，不是奇异值平方归一化，零概率项按 $0\log 0=0$ 处理。若四个奇异值相等，erank 为 4；若只有一个非零方向，erank 为 1；中间值描述谱的集中程度，不是某个固定阈值下数出的整数秩。全零矩阵没有可归一化分布，本文计算脚本约定返回 0。

论文 Figure 1 在 VLA-Adapter / LIBERO Object 上每隔 900 步记录一次模块平均 erank：视觉约为 292–309，语言约为 34–59，动作约为 4–6。它反映该实验中的模块差异，不是所有 VLA 的固定常数。有效秩也受到矩阵尺寸限制，换架构或比较不同宽度的层时，应同时记录形状。

图中统计的是梯度 $G$，实际谱变换处理的是动量 $M$。动量累计也会改变谱，训练日志中应分别记录二者，不能把一次梯度的 erank 直接当成动量的精确秩。

<details id="vla-gradient-rank">
<summary>动作维数、梯度秩和动量秩有什么区别？</summary>

把同一线性权重的使用位置合并为 $N$ 行，输入为 $H\in\mathbb R^{N\times d_{\mathrm{in}}}$，输出侧损失导数为 $\Delta\in\mathbb R^{N\times d_{\mathrm{out}}}$。按 PyTorch 权重布局：

$$
\begin{aligned}
Y&=H W^\top,\\
\nabla_W L&=\Delta^\top H.
\end{aligned}
$$

因此该层梯度满足：

$$
\operatorname{rank}(\nabla_W L)
\le\min(N,d_{\mathrm{in}},d_{\mathrm{out}}).
$$

这里 $N$ 可包含 batch、时间或 token 位置。最终输出矩阵的行数确实限制它自己的梯度秩；如果动作 chunk 被展平到输出轴，其行数还会随布局改变。这个结论不能直接移用为所有隐藏层的秩界，因为隐藏层的反向导数会随样本及非线性激活改变。

一个标量输出的构造就能说明这一点。取 $d$ 个输入，隐藏参数 $W=I_d$、输出头参数 $v=\mathbf1$。令第 $i$ 个输入 $x_i$ 的第 $i$ 个坐标为 1，其余为 $-a$，其中 $a=1/(2d)$。输出为 $v^\top\operatorname{ReLU}(Wx_i)$，以零为目标计算平均绝对误差。每个输出为 1；所有预激活非零，残差也为正，因此该点可微。隐藏层梯度为：

$$
\nabla_W L=\frac{(1+a)I_d-a\mathbf1\mathbf1^\top}{d}.
$$

沿全一向量及其正交方向的特征值分别为 $\frac{d+1}{2d^2}$ 与 $\frac{2d+1}{2d^2}$，都为正，故梯度满秩。取 $d=8$：

| 参数 | 权重形状 | 梯度秩 |
| --- | --- | ---: |
| 隐藏矩阵 $W$ | $8\times8$ | 8 |
| 标量输出头 | $1\times8$ | 1 |

再看时间上的累积。采用 [11.5 节](#stateful-step)的 EMA 约定，令 $\beta=0.5$、初始缓冲为零，依次输入三个不同对角方向的秩一梯度：

| 更新步 | EMA 动量对角线 | 动量秩 |
| ---: | --- | ---: |
| 1 | `(0.5, 0, 0)` | 1 |
| 2 | `(0.25, 0.5, 0)` | 2 |
| 3 | `(0.125, 0.25, 0.5)` | 3 |

表中为普通矩阵秩，熵有效秩还取决于系数的比例。每步梯度都是秩一，历史方向却同时留在动量中。因此，低维动作、当前梯度秩与动量谱需要各自判断；使用 Nesterov 时，还应记录真正进入谱变换的混合矩阵。

解压实验包后运行[标准库脚本](vla_gradient_rank.py)：

```bash
python3 -B vla_gradient_rank.py > vla-gradient-rank.json
```

[完整记录](assets/vla-gradient-rank-reference.json)保存输入、梯度和精确特征值证书。检查分别用网络损失有限差分、有理数消元和 NumPy 矩阵反传／SVD 复核；真实模块仍需按自身计算图与测得的谱判断。

</details>

<figure class="article-figure" id="fig-pion-module-rank">
  {{< post-image src="assets/paper-module-rank.png" alt="论文 Figure 1：视觉、语言、动作模块的梯度有效秩依次降低；固定视觉语言使用 AdamW 时，动作模块 Muon 的成功率为 82.2%，低于 AdamW 的 93.6%，低秩 Muon 达到 97.0% 但耗时更长" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text">来源：<a href="https://arxiv.org/pdf/2605.19282v1#page=4">论文 Figure 1，第 4 页</a>。左右实验均为 VLA-Adapter / LIBERO Object；成功率与耗时比较采用 4,500 步预算，视觉和语言优化器固定为 AdamW，只替换动作模块。耗时面板使用断轴；20.81 小时不能按柱高直接与 1.31、1.36 小时比较。</span></figcaption>
</figure>

### 2.3 为什么图中 Muon 变差，主结果却变好

图 2 中，只把动作模块从 AdamW 换成 Muon，成功率从 93.6% 降为 82.2%；SVD 低秩版本 LRMuon 达到 97.0%，但总耗时从约 1.36 小时增加到 20.81 小时。

后文主结果里的“Muon”则同时改变了视觉、语言、动作模块的二维矩阵更新方式，预算也不同。因此，“动作模块单独换 Muon 变差”和“全模型的 Muon 配置优于全 AdamW”并不矛盾。前者定位模块问题，后者比较整个配置。

LRMuon 的结果说明保留较少方向可能有益，但论文的这个基线使用**每步精确 SVD 的 top-k 极分解**。约 15 倍耗时属于该实现和设置，不能推广为所有低秩优化器、随机投影方法的固定开销。

## 3. 为什么 RLVR 的梯度噪声值得单独考虑 {#rlvr-snr}

### 3.1 可验证奖励不意味着低噪声梯度

RLVR 可以用数学答案、程序测试等规则判断一条回答是否成功。但奖励正确，并不意味着每个 token 都获得准确、稠密的学习信号。以 GRPO 为例，同一问题采样多条回答，利用组内相对奖励构造优势，再更新策略。

论文定义梯度信噪比为：

$$
\operatorname{SNR}(G)=
\frac{\|\mathbb{E}[G]\|_F^2}
{\mathbb{E}[\|G-\mathbb{E}[G]\|_F^2]}.
$$

分子描述平均梯度的强度，分母描述采样波动。它不是梯度范数本身，也不是单批梯度谱中某一个奇异值的大小。估计它时，应在同一模型检查点和明确的采样条件下重复取得梯度；若直接混合不同训练步，参数变化引起的方向漂移也会进入方差。记录 SNR 时，还应说明 batch、rollout 与梯度平均方式，以及统计的是原始梯度、动量还是 Nesterov 混合输入；这些随机量的 SNR 不能直接互换。

<details id="snr-estimation">
<summary>怎样估计梯度 SNR：有限样本偏差、负校正值与零样本方差</summary>

先定义一次观测是什么：例如固定检查点下，一个独立采样 batch 的完整梯度。将同形矩阵按固定顺序展平，记 $K\ge2$ 次独立同分布观测为 $G_1,\ldots,G_K$，总体均值为 $\mu$、总方差为 $V$：

$$
\begin{aligned}
\mu&=\mathbb E[G],\\
V&=\mathbb E\|G-\mu\|_F^2.
\end{aligned}
$$

有限样本平均 $\bar G$ 本身也有噪声。因此，直接把它的平方范数当成信号功率会向上偏：

$$
\begin{aligned}
\bar G&=\frac1K\sum_{k=1}^K G_k,\\
\mathbb E\|\bar G\|_F^2&=\|\mu\|_F^2+\frac VK.
\end{aligned}
$$

在上述独立同分布、有限二阶矩条件下，可以先估计总方差，再校正信号功率：

$$
\begin{aligned}
\widehat V&=\frac1{K-1}\sum_{k=1}^K\|G_k-\bar G\|_F^2,\\
\widehat S&=\|\bar G\|_F^2-\frac{\widehat V}{K}.
\end{aligned}
$$

两者的期望分别为 $V$ 与 $\|\mu\|_F^2$。但 **$\widehat S/\widehat V$ 一般不是无偏的 SNR 估计**，因为分母也是随机量。

有限样本下 $\widehat S$ 可能为负，表示这次校正超过了样本均值功率。例如，只有两条观测时，校正功率恰好等于它们的 Frobenius 内积；两个相反梯度就会给出负值。真实信号功率与 SNR 仍非负，直接把负校正值截成零会改变估计的期望。若 $\widehat V=0$，这批观测没有显示波动，但不足以证明总体没有噪声；下面的脚本将两个比值都记为 `null`。

[标准库脚本](pion_snr.py)构造四个坐标的梯度：均值为 $(s,0,0,0)$，每个坐标独立加入等概率的 $+0.5$ 或 $-0.5$。固定随机种子 2026、每例取 $K=256$ 次观测，得到：

| 一次观测的构造 | 已知总体 SNR | 直接比值 $\|\bar G\|_F^2/\widehat V$ | 有符号校正比值 $\widehat S/\widehat V$ |
| --- | ---: | ---: | ---: |
| $s=1$，单次梯度 | 1 | 1.018945 | 1.015039 |
| $s=1$，先平均 8 个独立梯度 | 8 | 7.495807 | 7.491901 |
| $s=0$，均值为零 | 0 | 0.003615 | −0.000291 |
{.table-readable}

这些是构造分布的一次有限样本结果，不是训练测量或准确率。第二行改变了**一次观测的定义**：独立平均使总体总方差从 1 降到 $1/8$，均值不变，所以总体 SNR 为 8。每例重复 256 次是为了估计这个量，不意味着优化器实际使用了 256 次观测的平均。解压[实验包](pion-spectral-lab.zip)后可直接复算：

```bash
python3 -B pion_snr.py > snr-result.json
```

[默认完整记录](assets/snr-reference.json)还包含两条相反观测和重复相同观测的边界例子。检查脚本枚举独立正负噪声的全部组合，用有理数核对方差与校正功率的期望；这不验证真实训练数据满足独立假设。

迁移到 GRPO 日志时，不能把共享奖励均值与标准差的一组回答直接视作独立观测；同步后各进程上的同一梯度副本，也不构成独立重复。应明确以完整组还是 batch 为单位，并记录梯度位于 loss unscale、梯度累积、跨进程平均和 clipping 的哪一侧。这个小脚本会将全部观测保存在内存中；大模型需要按块或流式汇总，同时保持同一梯度定义。若为避免浮点溢出而缩放，应对全部观测使用同一个非零常数；逐条归一化会改变所估计的 SNR。

</details>

<details id="snr-correlated-groups">
<summary>同一梯度复制多次，为何会使独立样本校正失效？</summary>

先固定校正公式：对 $K\ge2$ 条同形观测，样本总方差 $\widehat V$ 使用分母 $K-1$。用样本均值功率减去 $\widehat V/K$，得到 $\widehat S$。即使观测相关，也有精确代数恒等式：

$$
\widehat S=\frac{\sum_{i\ne j}\langle G_i,G_j\rangle_F}{K(K-1)}.
$$

它把校正功率写成不同观测间内积的平均；计算脚本仍使用均值和中心化平方差，不必实际遍历所有样本对。若每条观测的均值同为 $\mu$，且二阶矩有限，记总交叉协方差为：

$$
C_{ij}=\mathbb E\langle G_i-\mu,G_j-\mu\rangle_F.
$$

对上述恒等式取期望可得：

$$
\mathbb E\widehat S
=\|\mu\|_F^2+
\frac{\sum_{i\ne j}C_{ij}}{K(K-1)}.
$$

因此，直接套用独立样本校正时，交叉协方差之和为正会向上偏，为负会向下偏。仅打乱记录顺序不会改变这个量。

用一个能完整枚举的标量例子检查：独立采样 $B=4$ 组，每组梯度为 $1+Z_b$，其中 $Z_b$ 独立等概率取 $\pm1$。每个原始观测的均值和方差都为 1。将每组原样复制 $n$ 次后，日志有 $K=4n$ 行，但仍只有四次独立采样。复制不改变样本均值；这个样本均值的真实方差始终为 $1/4$，平方的期望始终为 $1.25$。

| 每组复制次数 $n$ | 记录行数 $K$ | 逐行校正的功率期望 | 按四组校正的功率期望 |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 1.000000 | 1 |
| 2 | 8 | 1.142857 | 1 |
| 8 | 32 | 1.225806 | 1 |
| 32 | 128 | 1.244094 | 1 |
{.table-readable}

左侧校正的期望为 $1+(n-1)/(4n-1)$；复制越多，并不让结果更接近真实功率 1。这里列的是**校正功率的期望**，不是 SNR 比值的期望，也不是每次实验都会得到的数值。

[标准库脚本](snr_correlated_groups.py)给出精确分数、十进制值，以及一组有限样本的两种统计结果。检查脚本枚举全部独立正负组合，分别计算每条重复记录与每个独立组的样本矩，用有理数复核上述期望。解压实验包后运行：

```bash
python3 -B snr_correlated_groups.py > snr-correlated-groups.json
```

[完整记录](assets/snr-correlated-groups-reference.json)中的有限样本也说明，单次校正值可以低于真值；期望向上偏并不意味着每次都偏大。实际日志应保留采样组标识，以完整的独立采样单位统计；分组本身不能证明组间独立。真实 GRPO 组内的相关性不必等于本例的完全复制，不能直接套用表中的偏差大小。

</details>

论文在 Qwen3-1.7B / MATH 的实验中发现 GRPO 梯度 SNR 明显低于 SFT。其解释包括：整条轨迹共享较粗的奖励、组内标准化、重要性比率，以及 clipping 对梯度的重加权。

### 3.2 全对或全错的组为何没有优势信号

对二值奖励，若一组 $g$ 条回答全部正确或全部错误，奖励标准差为零，组内标准化优势也为零。假设每条回答独立、成功概率同为 $p$，混合成功与失败的概率为：

$$
q_{\mathrm{nd}}=1-p^g-(1-p)^g.
$$

这是附录 C 用于分析的简化模型。比如 $p=0.01$、$g=8$ 时，约只有 7.73% 的组同时包含成功和失败；大量 rollout 不会产生这种组内对比信号。真实回答并不一定独立，但这个例子说明了奖励退化组为何会降低有效学习信号。

这里为零的是组内奖励优势；如果训练器还加入 KL 正则等其他损失，整体梯度不一定为零。后文核对的官方 RLVR 启动脚本就启用了 KL loss。

混合组中的优势之和也为零，但策略梯度是优势与各条回答的 score 相乘后再求和；不同回答的 score 不同，梯度不一定抵消。

<details id="grpo-group-gradient">
<summary>优势之和为零，为什么仍能更新？只保留混合组又改变了什么？</summary>

取单步 Bernoulli 策略 $p=\operatorname{sigmoid}(\theta)$，回答 $y\in\{0,1\}$，奖励也等于 $y$，则 score 为 $\partial_\theta\log\pi_\theta(y)=y-p$。固定旧策略、只看未裁剪的任务项，组内标准差采用除以 $g$ 的公式，忽略 ε；优势算出后作为常数参与求导。

<div style="break-inside: avoid">

例如 $p=0.5$、两条回答为 $(1,0)$ 时，优势为 $(1,-1)$，score 为 $(0.5,-0.5)$。优势之和为零，但平均上升方向为：

$$
\frac{1\cdot0.5+(-1)\cdot(-0.5)}{2}=0.5.
$$

</div>

一般地，一组 $g$ 条回答中有 $k$ 次成功时：

$$
\begin{aligned}
d&=\frac1g\sum_i A_i(y_i-p),\\
d&=h(k)=\frac{\sqrt{k(g-k)}}g.
\end{aligned}
$$

这里 $d$ 表示上升方向；若用梯度下降最小化负目标，损失梯度为 $-d$，SNR 不因这个整体变号而变。

全对或全错时，优势和 $h(k)$ 均为零。这个构造与[附录 C 式 A16](https://arxiv.org/pdf/2605.19282v1#page=20)的奖励对比系数相同；它没有额外的轨迹 score 残差。

若各条回答独立，成功次数服从二项分布。按所有可能的成功次数求矩，还可核对：

$$
\mathbb E[h(k)^2]=\frac{g-1}{g}p(1-p).
$$

下表固定 $g=8$，比较**每个完整提议组**的总体 SNR，以及**只保留混合组**后的条件 SNR。每个保留组的平均提议次数为 $1/q_{\mathrm{nd}}$，假设各次提议独立。

| 成功概率 $p$ | 非退化概率 | 全部提议组的 SNR | 仅混合组的条件 SNR | 提议组数／保留组 |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.077255 | 0.083432 | 309.959421 | 12.944095 |
| 0.10 | 0.569533 | 1.258836 | 45.535177 | 1.755825 |
| 0.50 | 0.992188 | 60.901986 | 117.937799 | 1.007874 |
{.table-readable}

<div style="break-inside: avoid">

本例中，退化组的 $d=0$。当 $q=q_{\mathrm{nd}}>0$ 时，记保留组内 $d$ 的条件均值与方差为 $\mu_c,V_c$，则：

$$
\begin{aligned}
\mathbb E[d]&=q\mu_c,\\
\operatorname{Var}(d)&=qV_c+q(1-q)\mu_c^2.
\end{aligned}
$$

</div>

方差中的第二项来自混合组与退化组两类组之间的波动，条件 SNR 没有包含它。

在 $p=0.01$ 时，丢掉大量零更新组后，只看保留组会得到很高的条件 SNR，但每保留一组平均要提议约 12.94 组。两个数描述不同的采样分布，不能把条件统计直接当成全部 rollout 的统计，也不能忽略被丢弃组的采样成本。

解压实验包后运行[标准库脚本](grpo_group_signal.py)：

```bash
python3 -B grpo_group_signal.py > grpo-group-signal.json
```

[完整记录](assets/grpo-group-signal-reference.json)用 80 位 Decimal 保存两种分布的矩及边界情形；零方差的比值记为 `null`。独立检查既枚举二进制回答序列，又对固定优势的对数似然与重要性比率目标做有限差分。这个单参数构造省略多 token 残差、clipping 和 KL，不能代替完整训练的梯度 SNR 测量。

</details>

[附录 C 式 A13–A20](https://arxiv.org/pdf/2605.19282v1#page=20)先把每条回答的 token 对数概率梯度取平均，得到轨迹 score；再按奖励取条件均值，把偏离该条件均值的部分记为残差。成功与失败的条件均值之差给出奖励对比方向，组内加权的残差再贡献采样方差。论文的低 SNR 近似以这部分残差方差为主，忽略奖励对比系数自身波动带来的较小项。

[上面的 Bernoulli 例子](#grpo-group-gradient)只有标量 score $y-p$，奖励也等于 $y$。固定 $p$ 时，给定奖励后 score 已确定，条件残差恒为零。因此，它适合检查组内求和与保留条件；论文的低 SNR 近似关注残差占主导的情形。

这部分 SFT / GRPO 推导采用代表性回答长度、token score 方差结构及条件独立等假设；重要性采样与 clipping 又使用近似分解或随机掩码模型。复核训练日志时，应逐项检查这些条件。

### 3.3 统一放大弱方向可能怎样破坏更新

在低 SNR 下，一次小批量得到的尾部奇异方向更容易随采样改变。理想白化仍给它们与主方向相同的系数，可能把不稳定变化变成显著参数更新。论文在后面的八组 RLVR 实验中观察到所用 Muon 配置的准确率接近零。

这个观察支持作者的谱失配解释。要区分谱过滤与其他因素的影响，还需同时记录学习率、更新缩放、优化器分组和数值精度。

## 4. 把矩阵问题变成一维多项式 {#spectral-polynomial-view}

令 $X=U\Sigma V^\top$。矩阵乘积具有：

$$
\begin{aligned}
XX^\top X&=U\Sigma^3V^\top,\\
X(X^\top X)^2&=U\Sigma^5V^\top.
\end{aligned}
$$

因此一次更新等价于：

$$
aX+bXX^\top X+cX(X^\top X)^2
=U f(\Sigma)V^\top,
$$

其中每个奇异值独立经过：

$$
f(\sigma)=a\sigma+b\sigma^3+c\sigma^5.
$$

**在精确算术下，多项式改变奇异值对应的系数，保留原奇异方向，不需要显式求出 $U$、$V$。** 这使 Pion 可以用矩阵乘法实现谱滤波，SVD 只在分析和本文的小型校验中使用。浮点实现还会产生舍入误差，见[数值精度对照](#numeric-precision)。

这里的“高通”是：归一化奇异值较大的方向通过，较小的方向受抑制。它不涉及时间 FFT，也不是给机器人动作做高频增强，更不能与 [One Euro 等时间滤波器](../../robotics/control/robot-filters/)混为一谈。

## 5. Promotion 与 Suppression 怎样推导 {#pion-polynomials}

### 5.1 先固定强方向的终点

[论文附录 E](https://arxiv.org/pdf/2605.19282v1#page=23) 推导了两个阶段的系数。Pion 希望已经到达 1 的方向保持不变，并在 1 附近减弱一阶扰动，因此共同要求：

$$
f(1)=1,
\qquad f'(1)=0.
$$

代入五次奇多项式，得到：

$$
a+b+c=1,
\qquad a+3b+5c=0.
$$

两个方程不能唯一确定三个系数，需要根据提升或抑制的目的再加入条件。

### 5.2 Promotion 尽量提升，同时保持顺序

先用 $c$ 表示另两个系数：

$$
a=\frac{3+2c}{2},
\qquad b=-\frac{1+4c}{2}.
$$

要求在 $[0,1]$ 上单调不减，并在 1 附近不向上越出谱范围。将导数因式分解：

$$
f'(\sigma)=(1-\sigma^2)\left(1.5+c-5c\sigma^2\right).
$$

第一个因子非负，第二个因子对 $\sigma^2$ 是线性的，只需核对两个端点都非负，即 $1.5+c\ge0$、$1.5-4c\ge0$。因此 $-1.5\le c\le0.375$，对应 $0\le a\le1.875$。上界也恰好使边界曲率 $f''(1)=-3+8c\le0$。由于原点斜率 $f'(0)=a$ 控制弱方向的提升速度，选择允许的最大值：

$$
\boxed{f_p(\sigma)=1.875\sigma-1.25\sigma^3+0.375\sigma^5.}
$$

它的导数可写成：

$$
f_p'(\sigma)=1.875(1-\sigma^2)^2\ge0.
$$

所以它不会颠倒奇异值的大小顺序，也将 $[0,1]$ 映射回 $[0,1]$。较小输入近似变为 $1.875\sigma$，较大输入向 1 靠拢。Promotion 本身也提升尾部；它需要与后续 Suppression 配合。

### 5.3 Suppression 去掉原点的线性放大

抑制阶段再要求：

$$
f_s'(0)=0.
$$

这直接给出 $a_s=0$。联合前两个约束，唯一得到：

$$
\boxed{f_s(\sigma)=2.5\sigma^3-1.5\sigma^5.}
$$

两种映射正好取上述可行系数族的两端：Promotion 取 $c=0.375$，使原点斜率最大；Suppression 取 $c=-1.5$，使原点斜率为零。

它满足：

$$
f_s'(\sigma)=7.5\sigma^2(1-\sigma^2)\ge0.
$$

在原点附近，它按三次量级衰减，而不是线性放大。但 **Suppression 并非让所有输入都变小**。将它与恒等映射相减：

$$
f_s(\sigma)-\sigma
=\frac12\sigma(1-\sigma^2)(3\sigma^2-2).
$$

所以除了 0 和 1，还存在一个不稳定固定点：

$$
\sigma_* = \sqrt{2/3}\approx0.816497.
$$

低于这个值的方向向 0 收缩，高于它的方向向 1 增长。这个固定点解释了反复抑制为什么会形成高通响应。

### 5.4 为什么先提升，再抑制

总步数固定为 5，分配为 $k_p$ 次提升和 $k_s=5-k_p$ 次抑制：

$$
F_{k_p}(\sigma)=f_s^{\circ k_s}\!\left(f_p^{\circ k_p}(\sigma)\right).
$$

<details id="spectral-step-trace">
<summary>查看归一化输入 0.35 的逐步变化</summary>

保持总次数为五，只将第二轮从抑制改成提升：

| 轮次 | 一提升四抑制 | 两提升三抑制 |
| ---: | ---: | ---: |
| 输入 | 0.350000 | 0.350000 |
| 1 | 提升 0.604626 | 提升 0.604626 |
| 2 | 抑制 0.431380 | 提升 0.887682 |
| 3 | 抑制 0.178280 | 抑制 0.921930 |
| 4 | 抑制 0.013896 | 抑制 0.959961 |
| 5 | 抑制 0.000006707 | 抑制 0.988760 |

一提升得到的 0.604626 仍低于单步抑制的不稳定固定点 0.816497，后续不断收缩；两提升得到的 0.887682 越过该点，后续向 1 增长。这里比较的是同一个已归一化输入，没有改变原始谱或识别噪声。

下载包中运行以下标准库命令，可查看各轮 Float64 计算值：

```bash
python3 -B pion_spectral.py --sigma 0.35 --kp 1 --trace
python3 -B pion_spectral.py --sigma 0.35 --kp 2 --trace
```

[逐步记录](assets/trace-reference.json)也保留每轮 Muon 系数。输入已经归一化，`--trace` 不会把单个数再归一化成 1。

</details>

先提升会让更多输入越过后续抑制的分界。在这里固定五步的多项式族中，还可以给出更明确的关系：

$$
f_p(\sigma)-f_s(\sigma)
=1.875\sigma(1-\sigma^2)^2\ge0.
$$

由于后续抑制映射也是单调的，将一次抑制改为提升，不会降低任一相同归一化输入的最终系数。因此，**提升次数越多，超过同一幅度阈值的方向不会减少；抑制越强，保留范围越集中。** $k_p=5$ 时没有抑制阶段，不能再将其描述为强尾部过滤。

固定五步时，输出不是硬阶跃。为比较过渡位置，本文计算满足 $F_{k_p}(\sigma)=0.5$ 的输入：

| 提升次数 | 抑制次数 | 五步后输出 0.5 的输入 |
| ---: | ---: | ---: |
| 0 | 5 | 0.797811 |
| 1 | 4 | 0.492230 |
| 2 | 3 | 0.263330 |
| 3 | 2 | 0.130517 |
| 4 | 1 | 0.059526 |
| 5 | 0 | 0.023241 |

这些数值由[标准库脚本](pion_spectral.py)二分求得，是有限步响应的比较点；它们既不是硬 top-k 的秩，也不是抑制多项式单步固定点 $0.816497$。

归一化还带来一个直观约束。设某个调度的 0.5 响应点为 $\tau$，有 $q$ 个方向的输出系数至少达到 0.5。由于响应单调，这些输入都至少为 $\tau$，于是：

$$
q\tau^2\le\sum_i\sigma_i(X_0)^2\le1,
\qquad q\le\left\lfloor\frac{1}{\tau^2}\right\rfloor.
$$

因此，纯抑制五步最多只有 **1 个方向的系数达到 0.5**；一提升四抑制时，这个上界是 4。它限制的是超过指定幅度的方向数，**不是说输出的数学秩最多为 1 或 4**，其余小系数仍可参与更新。分头独立归一化时，这个约束分别作用在每个块上，也有助于理解为何分块会改变结果。

<figure class="article-figure" id="fig-pion-spectral-maps">
  {{< post-image src="assets/spectral-polynomials-v2.png" alt="四个确定性谱计算面板：五步 Muon 的非均匀响应与 Pion 一提升四抑制的高通曲线；两种单步多项式；提升次数增加时过渡左移；构造四方向谱经过不同变换后的系数" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text">由论文公式 (3)、(7)、(8) 直接计算。A 中 Muon 曲线为实际五步多项式，不是理想矩阵 sign；B 的虚线标记单步抑制固定点；C 固定总步数为 5，列出全部六种调度；D 使用下文构造谱，对低于 10⁻¹² 的系数仅在显示时截到图底并标倒三角，完整值仍保存在<a href="assets/spectral-reference.json">数值记录</a>。这四个面板不表示训练成功率。</span></figcaption>
</figure>

## 6. 一个能直接复算的谱例子 {#spectral-example}

对前面的四个奇异值先除以 Frobenius 范数，得到约：

$$
(0.872664,\ 0.436332,\ 0.218166,\ 0.021817).
$$

保持相同奇异方向，分别应用 Muon 五步和 Pion 一次提升加四次抑制：

| 原方向 | 归一化输入 | Muon 五步 | Pion 一提升四抑制 |
| ---: | ---: | ---: | ---: |
| 1 | 0.872664 | 0.822939 | 1.000000 |
| 2 | 0.436332 | 1.132615 | 0.045052 |
| 3 | 0.218166 | 0.699203 | 1.404 × 10⁻¹⁸ |
| 4 | 0.021817 | 0.730070 | 2.815 × 10⁻⁹⁷ |
{.table-readable}

Pion 将更新集中到前面的方向，Muon 则把原来很弱的第四方向提升到了显著尺度。有限 Muon 的第二个系数甚至高于第一个：它的五步响应并不单调。表中的编号始终指原输入方向，没有按输出系数重新排序。这个例子只说明谱响应，并未给各方向标注任务信号或噪声。

还要看到另一个方向的风险：若四个奇异值相等，归一化后都是 0.5。纯抑制五步会把每个系数都压到约 $5.908\times10^{-32}$。因此，**强抑制可以把一个平坦谱中的所有方向一起压小**，而不只是“自动找出正确低秩”。实际训练需要监控更新范数，避免把很小的更新误认为优化已经收敛。

Pion 的两个映射在 $(0,1]$ 上都严格为正。因此，有限五步在精确算术下会把某些系数压到极小，并不把原本非零的奇异方向精确置零。浮点下溢得到零，也不构成新的精确秩证明。

### 6.1 交互比较谱调度与归一化 {#interactive-spectrum}

下面从四个原始奇异方向的幅度开始，先按 Frobenius 范数归一化，再计算五步响应。改变提升次数，可以观察保留范围怎样移动；选择平坦谱，则能同时看到“输出有效秩仍高”和“更新幅度已经很小”。加入 ε 后，再比较微小幅值与普通幅值，能看出归一化本身也会影响结果。

可以依次做三个对照，每次只改一个因素：

1. 选择“集中谱 · 1+4”，比较同一个弱方向在 Muon 与 Pion 下的系数。
2. 选择“平坦谱 · 0+5”，确认两个有效秩都为 4，但 Pion 输出范数很小；随后只把提升次数改为 1，观察幅度变化。
3. 选择“微小幅值 + ε”，随后只取消 ε。输入幅度和谱形状都未改变，响应差异来自分母中的固定 ε。

{{< pion-spectrum id="pion-spectrum-lab" >}}

实验保留输入方向的顺序，不把幅度较小的方向自动标为噪声。原始输入的 erank 描述谱形状，输出范数描述变换后的幅度，两者不能互相替代。

### 6.2 已知信号的对照：强方向也可能是扰动 {#signal-alignment}

为了检验“谱强弱不等于任务语义”，固定干净矩阵 $S$，再构造 $G=S+N$。本例令动量 $M=G$，不累计历史：

$$
S=\operatorname{diag}(1,0,0,0).
$$

第一种情形让弱扰动位于尾部：

$$
N=\operatorname{diag}(0,0.2,0.1,0.05).
$$

第二种情形让强扰动压过信号：

$$
N=\operatorname{diag}(0,5,0,0).
$$

对更新矩阵 $U$，用 Frobenius 余弦比较它与已知干净方向的一致性：

$$
\operatorname{align}(S,U)=
\frac{\langle S,U\rangle_F}{\|S\|_F\|U\|_F}.
$$

保持 Pion 为一提升四抑制、ε 为零，得到约：

| 构造情形 | 归一化梯度 | Muon 五步 | Pion 一提升四抑制 |
| --- | ---: | ---: | ---: |
| 弱扰动位于尾部 | 0.974740 | 0.487458 | 1.000000 |
| 强扰动压过信号 | 0.196116 | 0.740895 | 7.148 × 10⁻²² |
{.table-readable}

第一种情形符合高通的有利条件；第二种情形中，Pion 保留了更强的扰动方向，显著压低了预先定义的真信号。**高通根据谱幅度选择方向，并不知道哪一项具有任务意义。** 表中数值是与已知矩阵的方向余弦，不是训练成功率；两组固定扰动也不是对随机梯度 SNR 的估计。

下载包中运行 `python3 -B pion_alignment.py` 可复算，[完整记录](assets/alignment-reference.json)保存干净矩阵、扰动与各更新的对角元素。

## 7. RLVR 为什么还需要分头更新 {#per-head}

### 7.1 分头改变了归一化与谱分析的范围

一个注意力投影可以由多个头对应的子矩阵组成。整矩阵模式只计算一次归一化与谱变换；分头模式先拆成多个块，每个块分别归一化、分别应用相同的五步变换，再拼回原形状。

两者一般不等价：

$$
\mathcal{P}\!\left(\operatorname{Concat}_h M_h\right)
\ne
\operatorname{Concat}_h\mathcal{P}(M_h).
$$

原因不仅是实现上的 `reshape`。各块的归一化分母和奇异子空间都发生了变化；整矩阵中的较弱块，可能在本头中包含重要方向。

### 7.2 为什么预训练的头差异可能有功能意义

在附录 G 的标准注意力分析中，把 Q/K 投影的乘积拆成“范数乘积 × 归一化矩阵”，再固定后者与输入，Q、K 权重范数的乘积就控制注意力 logits 的尺度。乘积越大，softmax 的逆温度越高，非均匀注意力通常越尖锐；V、O 范数则影响该头输出和梯度的尺度上界。

这个尺度关系需要结合实际前向计算理解。[Qwen3 技术报告第 2 节](https://arxiv.org/html/2505.09388v1#S2)说明，Qwen3 引入了 QK-Norm；[Transformers 4.51.3 的实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py#L195-L218)在投影后的 query、key 上分别做 RMS 归一化。以 ε 为零为例，正比例缩放一个非零向量会被归一化消去；实际还需考虑 ε 与可学习缩放。因此，不能只由 Q/K 权重范数推断 Qwen3 某个头的 softmax 温度。附录 G 提供标准注意力下的分头动机，具体架构仍需沿完整前向图分析。

这说明不同头的范数差异可能与已有功能有关。不过，**分头 Pion 不保证精确保留原来的权重范数，也不自动按权重范数比例更新**。它避免将所有头放进同一个谱变换中；具体更新大小仍取决于本头动量谱、学习率和尺寸缩放。论文 Figure 4 给出了分头模式更有效的实验证据。

<figure class="article-figure" id="fig-pion-per-head">
  {{< post-image src="assets/paper-per-head.png" alt="论文 Figure 4：Qwen3 1.7B 在 MATH 上使用 GRPO 时，分头 Pion 的准确率优于整矩阵 Pion；单独分头 Muon 仍失效；右侧显示预训练 Q 投影的跨头范数方差和两种 Pion 更新方差" >}}
  <figcaption><span class="article-figure__number">图 4</span><span class="article-figure__text">来源：<a href="https://arxiv.org/pdf/2605.19282v1#page=7">论文 Figure 4，第 7 页</a>。左侧比较 AdamW、Muon 与 Pion 的整矩阵／分头模式；右侧是 Q 投影在不同层的跨头范数方差。它说明该设置中分头与高通需要配合；不是只要 reshape 就能修复 Muon。</span></figcaption>
</figure>

### 7.3 Q、K、V 与输出投影不能沿同一轴乱切

以 PyTorch `Linear` 权重的“输出维 × 输入维”布局为例：

| 投影 | 应识别的头维度 | 分块注意事项 |
| --- | --- | --- |
| Q | 输出侧 | 按 query 头数拆输出行 |
| K、V | 输出侧 | GQA 应使用 KV 头数，不能直接照搬 Q 头数 |
| O | 输入侧 | 各头输出拼接后进入 O，因此按输入列拆 |
| 普通 MLP | 没有同样的注意力头语义 | 通常仍做整矩阵变换 |
{.table-readable}

Qwen3 技术报告 Table 1 列出的 Q／KV 头数，1.7B 模型为 16／8，4B 为 32／8。两者的 K/V 都按各自的 8 个 KV 头识别，不能直接使用 Q 头数。

以[固定版本的 Qwen3-4B 配置](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json)为例，`hidden_size=2560`、`head_dim=128`，未分片权重按上述布局为：

| 投影 | 头数 | 完整权重形状 | 每头块形状 |
| --- | ---: | --- | --- |
| Q | 32 | 4096 × 2560 | 128 × 2560 |
| K、V | 8 | 1024 × 2560 | 128 × 2560 |
| O | 32 | 2560 × 4096 | 2560 × 128 |

这里 32 × 128 = 4096，并不等于隐藏维度 2560；切分时需读取实际投影形状，不能把每头维度一律假定成 `hidden_size / num_attention_heads`。

融合 QKV、张量并行和参数分片还会改变本地布局。官方实现提供分头轴与头数配置，并在不兼容形状下回退整矩阵；接入工程时应记录每个参数实际走了哪个分支。只看到类名带 `PerHead`，不足以证明模型所有注意力投影都已经正确分头。

### 7.4 两个同形谱的头能说明什么 {#per-head-counterexample}

取一个未分片的对角动量矩阵，并把前两行、后两行视作两个块：

$$
M=\operatorname{diag}(8,1,0.8,0.1).
$$

两个块的非零奇异值分别为 $(8,1)$ 与 $(0.8,0.1)$，形状相同，原始幅度相差 10 倍。都采用一次提升加四次抑制，ε 设为零，且不加入学习率或尺寸缩放：

| 块 | 原动量范数 | 整矩阵变换后的块范数 | 分头变换后的块范数 |
| --- | ---: | ---: | ---: |
| 前两行 | 8.062258 | 1.000000 | 1.000000 |
| 后两行 | 0.806226 | 1.298 × 10⁻⁴⁴ | 1.000000 |
{.table-readable}

整矩阵归一化时，第二块的最大奇异值相对较小，经过高通几乎被压掉；分头归一化后，第二块在自己的范围内也有一个强方向，因此可以保留。但两块原有的 10:1 幅度比例也被独立归一化消除了，输出范数相同。

拼回后的总 Frobenius 范数也从整矩阵模式的约 1 变成分头模式的约 $\sqrt{2}$。所以，即使外层学习率相同，两种模式的总更新幅度也未必相同；比较实验时应同时记录更新范数与权重范数之比。

这说明分头模式改变了**比较谱强弱的范围**，并不保证保留输入动量的范数比例，更不能保证保留预训练权重的比例。若各头的谱形状不同，输出范数也可以不同。这个构造例子没有加载任何预训练权重，不模拟 Qwen 的真实头分布。

下载包中运行 `python -B pion_matrix.py --head-report` 可以复算；[完整记录](assets/per-head-reference.json)保存原对角元素、两种变换结果与逐块范数。

## 8. VLA 中究竟把哪部分换成 Pion {#optimizer-assignment}

论文主 VLA 实验对二维矩阵的主要分工如下。Token embedding、语言模型输出层、一维参数等排除项使用 AdamW；实际脚本还按名称和形状细分路由，见 [11.3 节](#113-官方训练代码从哪里进入)。

| 配置名称 | 视觉二维矩阵 | 语言二维矩阵 | 动作二维矩阵 | 辅助参数 |
| --- | --- | --- | --- | --- |
| AdamW | AdamW | AdamW | AdamW | AdamW |
| Muon | Muon | Muon | Muon | AdamW |
| Pion | Muon | Muon | Pion 整矩阵模式 | AdamW |
{.table-readable}

**论文主配置中的 Pion，是混合优化器分工。** 它并不是把全模型所有二维矩阵都做高通。附录 K 的模块消融显示，在其他模块保持 AdamW 时，将视觉模块换为 Pion 的设置只有 17.8%，而视觉换为 Muon 的设置达到 96.8%。这支持“高通也可能丢掉有效尾部信息”的解释。

附录还比较了动作模块的整矩阵与分头 Pion：四套任务平均分别为 97.25% 和 96.85%，整矩阵模式略高；Goal 单项则分头略高。因此，不能把 RLVR 的分头经验直接升级成 VLA 动作模块的通用选择。附录的解释针对新训练动作头，不能据此认定所有 VLA 动作专家都没有预训练结构。

需留意一处报告差异：[附录 Table A6](https://arxiv.org/pdf/2605.19282v1#page=39) 的全 AdamW 基线为 43.6%，主 Figure 5 / Table A5 为 32.2%，均标注 Object 的 1,500 步。本文保留各表的原数，不把两组基线混合计算；现有报告信息不足以把它们当作同一次严格配对试验。

## 9. VLA 实验结果应该怎样读 {#vla-results}

### 9.1 VLA-Adapter 的固定预算比较

主结果与 [Table A5](https://arxiv.org/pdf/2605.19282v1#page=38) 给出：Object 训练 1,500 步，其余套件各训练 15,000 步。

| 配置 | Object | Spatial | Goal | Long | 四套平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | 32.2 | 97.0 | 89.2 | 69.6 | 72.00 |
| Muon | 97.0 | 99.0 | 95.8 | 88.0 | 94.95 |
| Pion 整矩阵动作头 | 100.0 | 99.4 | 97.2 | 92.4 | 97.25 |
{.table-readable}

表中单位均为成功率百分比。Pion 比 Muon 的四套平均高 **2.30 个百分点**；比 AdamW 高 25.25 个百分点。Object 的巨大差距出现在特定的早期训练预算，各套件的差值应分别阅读。

论文还报告 Pion 在 Object 的 500 步达到 95.4%、1,500 步达到 100%。这支持该设置下的步数效率，但训练步数不直接等于端到端墙钟时间，更不等于推理加速。

### 9.2 VLANeXt 与场景扰动

对 flow-matching 模型 VLANeXt，[Table 1](https://arxiv.org/pdf/2605.19282v1#page=8) 的平均成功率为：

| 配置 | LIBERO | LIBERO-Plus |
| --- | ---: | ---: |
| AdamW | 79.45 | 64.57 |
| Muon | 93.65 | 72.34 |
| Pion | 96.35 | 75.93 |

Pion 相对 Muon 分别高 2.70、3.59 个百分点。LIBERO-Plus 的 Language、Noise、Robot 三类扰动，分别从 77.53、69.98、57.36 提升至 86.93、76.09、63.18，即 **9.40、6.11、5.82 个百分点**。这是成功率之差，不应直接写成相对提升 9.40%、6.11%、5.82%。

这些结果把证据从 VLA-Adapter 的回归动作头扩展到 VLANeXt 的 flow-matching 动作头。

### 9.3 真机结果及其分母

真机使用 Franka Research 3、$\pi_{0.5}$、DROID 设置，200 条遥操作示范，训练 20,000 步。每种优化器、每个任务评估 30 个随机初始配置，每次最多 300 个控制步。

| 配置 | 黄瓜到盘子 | 方块到盘子 | 方块到碗 | 平均 |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 40.0 | 33.3 | 20.0 | 31.1 |
| Muon | 56.7 | 33.3 | 26.7 | 38.9 |
| Pion | 93.3 | 83.3 | 80.0 | 85.6 |
{.table-readable}

来源为论文 [Table 3](https://arxiv.org/pdf/2605.19282v1#page=9) 与[真机配置 Table A3](https://arxiv.org/pdf/2605.19282v1#page=28)。按 30 次试验与舍入百分比回推，Pion 三项分别对应 28、25、24 次成功，合计 77/90，约为 85.56%，舍入后为 85.6%。平均时应使用舍入前的比例。85.6% 是这三个任务的平均结果，**不是 85.6% 的任意机器人任务成功概率**。

它支持有限预算下这三个真机任务的有效性。单次 rollout 图片说明成功或失败方式，不能独立证明“谱尾放大必然造成某次碰撞”的因果链；30 次任务试验也不能替代不同训练种子的稳定性评估。

## 10. RLVR 的结果与关键消融 {#rlvr-results}

论文交叉比较两种算法、两种模型规模和两个数据集，共八种设置：

- GRPO 与 GMPO；
- Qwen3-1.7B 与 Qwen3-4B；
- MATH levels 3–5 训练、MATH500 评价；或 GSM8K 的训练／测试划分。

GRPO 使用 token 级重要性比率的聚合，GMPO 使用序列级几何聚合；[附录 A.2](https://arxiv.org/pdf/2605.19282v1#page=17) 给出各自目标。Pion 改变的是目标梯度之后的参数更新方式，没有重新定义这两种算法的奖励或优势。

[Table A2](https://arxiv.org/pdf/2605.19282v1#page=27) 给出的采样与训练预算为：

| 配置项 | MATH | GSM8K |
| --- | ---: | ---: |
| 最大 prompt / response 长度（token） | 1,024 / 3,000 | 512 / 1,024 |
| 每批 prompt 数 | 128 | 1,024 |
| 每个 prompt 的 rollout 数 | 8 | 5 |
| 训练步数 | 80 | 40 |

按每批 prompt 数乘 rollout 数，完整采样一批分别产生 1,024、5,120 条候选回答。实际 token 数还取决于回答长度，训练计算也包含梯度更新等工作。因此，两张曲线中的“一个训练步”不代表相同的采样量或计算量。

<figure class="article-figure" id="fig-pion-rlvr">
  {{< post-image src="assets/paper-rlvr.png" alt="论文 Figure 6 的八组 RLVR 曲线：两种算法、两种 Qwen3 规模、MATH 与 GSM8K 设置中，所用 Muon 配置准确率接近零，Pion 与 AdamW 能学习，Pion 通常收敛更快" >}}
  <figcaption><span class="article-figure__number">图 5</span><span class="article-figure__text">来源：<a href="https://arxiv.org/pdf/2605.19282v1#page=9">论文 Figure 6，第 9 页</a>。指标为验证准确率；MATH 与 GSM8K 分别使用 80、40 步预算。此处保留原曲线，不从图中估读出未经报告的精确终值或训练种子误差条。</span></figcaption>
</figure>

论文观察到 Muon 配置持续接近零，而 Pion 可以稳定学习，并在多种设置下更快或更高。需要结合两项消融理解：

1. **只分头的 Muon 仍然失效。** 它没有处理统一白化对弱方向的放大。
2. **整矩阵 Pion 也未达到分头 Pion 的效果。** 高通改善噪声适应性，分头进一步改变注意力块的更新几何。

作者还构造了低通版本 LPMuon：偏向保留较小的非零奇异方向，抑制较大奇异值。它在所示 RLVR 设置中基本停留在初始性能，形成方向相反的消融。不过 LPMuon 的系数通过多步多项式拟合获得，**不是简单交换 Promotion 和 Suppression 的顺序**。

这些对照支持高通方向在所测 Qwen3、GRPO／GMPO 与数学数据集中的作用；比较不同方法时，应连同超参数与训练预算一起报告。

## 11. 先运行谱计算，再接入训练 {#runnable-example}

### 11.1 不装深度学习框架也能检查响应

下载[谱计算小实验](pion-spectral-lab.zip)，解压后进入目录。Python 3.10 以上即可运行标准库部分：

```bash
python3 -B pion_spectral.py --sigma 0.35 --kp 1
python3 -B pion_spectral.py > spectral-result.json
python3 -B check_examples.py
```

`--kp` 是提升次数，总次数固定为 5。完整输出包含四方向构造谱、实际五步 Muon / Pion 系数、有限步的 0.5 响应点，以及文章表格的算术复核。文件也可分别下载：[标量脚本](pion_spectral.py)、[NumPy 矩阵核心](pion_matrix.py)、[信号方向对照](pion_alignment.py)、[绘图脚本](make_figures.py)、[默认结果](assets/spectral-reference.json)。

### 11.2 用 NumPy 验证矩阵多项式与 SVD 一致

下面以 Python 3.12 创建独立的 NumPy 环境。解压后进入目录，在 Linux／macOS 中运行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.2.6
.venv/bin/python -B pion_matrix.py
.venv/bin/python -B check_examples.py --numpy
```

下面的完整小例子将矩阵结果与“先 SVD，再逐奇异值变换”的独立计算比较：

```python
import numpy as np
from pion_matrix import transform
from pion_spectral import pion

momentum = np.array([[0.8, 0.0, 0.0], [0.0, 0.2, 0.0]])
u, sigma, vt = np.linalg.svd(momentum, full_matrices=False)
normalized = sigma / np.linalg.norm(sigma)
filtered = np.array([pion(float(s), 1) for s in normalized])
expected = (u * filtered) @ vt
actual = transform(momentum, promotion_steps=1)
np.testing.assert_allclose(actual, expected, atol=1e-12)
print(actual)
```

矩阵核心在较小的一侧构造 Gram 矩阵，避免不必要的大型中间矩阵。它只实现 Float64 多项式与分头分块，省略动量状态、Nesterov、weight decay、BF16 和尺寸缩放，便于核对数学。**它不是可直接替换真实训练器的完整优化器。**

标准库检查用 90 位 Decimal 独立复算组合多项式；可选 NumPy 检查还覆盖长宽矩阵、零矩阵、秩一矩阵、转置一致性和 Q/O 两种分头轴。小实验验证公式和实现约定，不复现论文训练成功率。

### 11.3 官方训练代码从哪里进入

官方仓库按任务提供优化器与启动脚本，并不是一个统一的顶层 pip 包。可先查看固定版本：

```bash
git clone https://github.com/OPTML-Group/Pion.git
cd Pion
git checkout 87965014b360bde604632cba0aa12e423c6c63c6
```

主要入口是：

| 任务 | 仓库目录 | 需要核对的实现 |
| --- | --- | --- |
| VLA-Adapter | `VLA/VLAAdapter` | `pion_optim/muon.py` 与 `scripts/run_pion.sh` |
| VLANeXt | `VLA/VLANeXt` | 模态优化器分配与动作训练入口 |
| 真机 $\pi_{0.5}$ | `VLA/openpi` | 训练器中的混合优化器与数据配置 |
| RLVR | `RL/verl` | `verl/utils/muon.py` 与分头配置、训练启动脚本 |
{.table-readable}

核对的 VLA-Adapter 与 RLVR Pion 启动脚本将提升次数默认设为 0，即纯抑制五步；本文用于解释的“一提升四抑制”只是另一种合法谱响应，不能冒充全部论文实验的固定设置。论文附录 H 统一说明使用 $k_s\ge3$、即 $k_p\le2$ 的偏抑制调度，具体运行仍应记录实际参数。

工程实现还包含动量混合、可选 Nesterov、BF16 计算、尺寸缩放与 AdamW 分支；简化公式中的相同 $\eta$ 不代表不同实现拥有相同有效步幅。应沿实际参数分组核对 embedding、输出层和一维参数的归属。

对二维权重，以尺寸因子 $\gamma\sqrt{\max(m,n)}$ 为例，固定提交中的类默认值也并不一致：

| 实现 | 默认尺寸常数 $\gamma$ | 尺寸取自哪里 |
| --- | ---: | --- |
| [VLA-Adapter `DefaultPion`](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/VLA/VLAAdapter/pion_optim/muon.py#L227-L278) | 0.2 | 整矩阵 |
| RL/verl [整矩阵类](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/RL/verl/verl/utils/muon.py#L427-L449)／[分头类](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/RL/verl/verl/utils/muon.py#L611-L641) | 5.0 | 整矩阵；分头命中时取本头块 |
| [openpi `DefaultPionAdamW`](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/VLA/openpi/src/openpi/training/muon_optim.py#L212-L239) | 2.0 | 整矩阵 |
{.table-readable}

这些是实现的类默认值，实际配置仍可覆盖。跨 pipeline 比较时，需要同时核对学习率、矩阵形状与尺寸因子；本文标量表保留的是尚未乘这些因子的谱系数。[逐步更新例子](#stateful-step)会显式乘回该因子。

固定提交中，[VLA-Adapter 的参数路由](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/VLA/VLAAdapter/vla-scripts/finetune_pion.py#L948-L1024)还包含额外排除规则：除维数小于 2 的参数、名称含 `embed` 但不含 `patch_embed` 的参数、LM head 外，`action_queries`、部分特殊 token，以及名称同时包含 `action_head` 与 `fc2` 的参数也进入 AdamW 辅助组。可选的 `proprio_projector` 名称归入 O 组，默认启动脚本令 O 使用 Muon。因此，论文的模块分配表应与实际参数名逐项对应，不能当成所有动作相关二维张量都进入 Pion 的保证。

该路由按 `ndim >= 2` 和排除名称分组，四维卷积权重也可能进入 Muon／Pion 组。[VLA-Adapter 的 DefaultPion 四维分支](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/VLA/VLAAdapter/pion_optim/muon.py#L271-L299)把卷积梯度按输出通道展平，再做谱变换；更新缩放读取原权重最后两维。例如，权重形状为 `(2,3,5,7)` 时，变换输入为 `(2,105)`，源码尺寸因子取 $\gamma\sqrt7$；按展平矩阵推算则会得到 $\gamma\sqrt{105}$。下文的缩放推导与 NumPy 演算保持二维参数的约定。处理卷积时，应记录原始形状、变换形状与实际缩放；该分支用 `view` 展平，还需核对梯度和动量的内存步幅是否兼容。

[RLVR 参数分组](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/RL/verl/verl/workers/config/optimizer.py#L43-L141)则按投影名称识别 Q/K/V/O，其他符合条件的二维矩阵走整矩阵分支；排除名称与低维参数走 AdamW。公开的 [Qwen3-1.7B / GSM8K 三份启动脚本](https://github.com/OPTML-Group/Pion/tree/87965014b360bde604632cba0aa12e423c6c63c6/RL/verl/examples/grpo_trainer)均启用 KL loss，并设置 `trainer.total_epochs=15`；这个停止条件不能直接等同于论文 Table A2 的 40 步。复现时还要对齐训练停止条件、评估检查点和优化器之外的设置。

VLANeXt 的[公开配置](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/VLA/VLANeXt/config/libero_train_pion_config.yaml#L27-L35)将 `data.max_steps` 默认设为 6,000，而论文 Table A1 分别使用 Object 4,000 步、其他套件 10,000 步。准备论文对照时，应按具体套件显式设置预算。

训练环境、数据与检查点按各子目录 README 准备后，VLA-Adapter 目录可使用 `bash scripts/run_pion.sh`；RL/verl 目录则按任务选择相应脚本。论文 Table A1 的 LIBERO VLA 训练使用 8 张 RTX A6000，Table A2 的 RLVR 使用 2 张 H100；运行数学小实验与复现完整训练，是两个不同的计算规模。

### 11.4 理想归一化与 BF16 实现怎样区分 {#numeric-precision}

本节与下一节的官方 CPU 探针另需 PyTorch。若只运行小实验，Linux 可以按 [PyTorch 2.8.0 的 CPU 安装入口](https://pytorch.org/get-started/previous-versions/)建立独立环境；不需要准备 VLA 训练数据或 GPU：

```bash
python3.12 -m venv .torch-env
.torch-env/bin/python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

下文 `python -B probe_...` 中的 `python` 可换为 `.torch-env/bin/python`；需要 NumPy 的精度探针还应在该环境安装 `numpy==2.2.6`。官方源码路径来自 11.3 的固定提交，探针会先核对文件哈希。这个环境只用于 CPU 小实验，完整训练仍按官方各 pipeline 配置。

本文的标量表与默认 NumPy 核心使用 Float64、ε 为零。对于非零矩阵，正比例缩放原始动量不会改变归一化后的谱。但在分母加入固定的正 ε 后，归一化奇异值还要乘上一个幅度相关因子：

$$
\frac{\sigma_i(M)}{\|M\|_F+\epsilon}
=\frac{\sigma_i(M)}{\|M\|_F}
\cdot\frac{\|M\|_F}{\|M\|_F+\epsilon}.
$$

当动量范数与 ε 相当或更小时，后一个因子不能忽略。此时把同一个矩阵缩小，不再只是保持原谱形状：所有输入都会进一步靠近抑制区。可以直接比较：

```bash
python3 -B pion_spectral.py --spectrum 0.8 0.4 0.2 0.02 --kp 1
python3 -B pion_spectral.py --spectrum 8e-9 4e-9 2e-9 2e-10 --kp 1 --epsilon 1e-7
```

第二条命令的 Float64 输出最大系数约为 $6.867\times10^{-55}$，而同样微小的输入在 ε 为零时，最大系数仍接近 1。这是归一化定义的差异，不是增加了噪声或改变了输入谱的相对形状。

官方 VLA-Adapter 多项式还先转为 BF16，再进行归一化与矩阵运算。固定该源码，用 PyTorch 2.8.0 的 CPU 调用观察普通幅值对角例子，得到：

| 原方向 | 本文 Float64、ε 为零 | 官方 CPU BF16、ε 为 10⁻⁷ |
| ---: | ---: | ---: |
| 1 | 1.000000 | 1.000000 |
| 2 | 0.045052 | 0.042969 |
| 3 | 1.404 × 10⁻¹⁸ | 9.622 × 10⁻¹⁹ |
| 4 | 2.815 × 10⁻⁹⁷ | 0 |
{.table-readable}

两列同时存在 ε 与运算精度的差别，不能把全部偏差只归于其中一项。第四方向在 BF16 中为零，也不能倒推其精确多项式系数本来就是零。矩阵乘法的舍入还可能受设备和内核影响，不应期待本文 Float64 数值与 GPU 训练逐位相同。

非对角矩阵还会受到逐项舍入的影响。取外积 $M=uv^\top$，其中 $u=(1,2,3)^\top$、$v=(1,-2,3,-4)^\top$，它在精确算术下秩为 1。对同一个 CPU 探针，一提升四抑制的输出最大奇异值约为 0.999109，第二个约为 $4.890\times10^{-4}$；改成两提升三抑制，最大奇异值约为 1.004936。因此，“保留原奇异方向”“谱系数位于 $[0,1]$”是精确多项式的性质，不能直接作为 BF16 输出的硬性断言。这里的小次要奇异值来自构造矩阵的数值计算，不代表发现了新的任务信号。

可选的 [CPU 探针](probe_official_pion.py)先核对固定官方文件的哈希，再调用其多项式；[观察记录](assets/official-precision.json)同时保存 ε 为零、仅加入 ε、官方 BF16 三种结果。它需要另外安装 PyTorch 和 NumPy，只检查小矩阵计算，不运行 VLA 模型或训练器。

### 11.5 从谱系数走到一次参数更新 {#stateful-step}

谱计算回答“给定矩阵会变成什么”；训练还要决定**送入谱变换的是哪个矩阵、变换后乘多少，以及没有梯度时是否更新参数**。固定版本的 VLA-Adapter 与 RL/verl 使用下面的 EMA 约定，$\beta$ 对应前文的动量系数 $\mu$，令 $B_0=0$：

$$
B_t=\beta B_{t-1}+(1-\beta)G_t.
$$

不使用 Nesterov 时，变换输入 $D_t=B_t$；启用时使用：

$$
D_t=(1-\beta)G_t+\beta B_t.
$$

这对应源码中的 `buffer.lerp_(gradient, 1-beta)` 与 `gradient.lerp(buffer, beta)`。开头论文记法 $M_t=\beta M_{t-1}+G_t$ 省略了 EMA 的 $1-\beta$ 因子；零初始化且不使用 Nesterov 时，两种缓冲满足 $B_t=(1-\beta)M_t$。在 ε 为零的理想归一化下，正比例因子会被消去；固定正 ε 和 Nesterov 的额外梯度混合都需要按实际公式计算。

记 $\mathcal P$ 为包含归一化的 Pion 变换，二维整矩阵分支的参数更新为：

$$
\begin{aligned}
s&=\gamma\sqrt{\max(m,n)},\\
W_t&=(1-\eta\lambda)W_{t-1}
-\eta s\,\mathcal P(D_t).
\end{aligned}
$$

这里 $\gamma$ 是源码的 `scale_factor`，$\lambda$ 是解耦权重衰减。衰减先作用于原权重，再减去谱更新；它不等同于先把 $\lambda W$ 加入梯度，因为后者也会进入动量和非线性谱变换。

还可以把这个尺寸因子换算成每个参数的均方根更新幅度。只考虑梯度项的增量 $\Delta W_{\mathrm{grad}}=-\eta s\mathcal P(D_t)$，不含权重衰减，则：

$$
\begin{aligned}
\operatorname{RMS}(\Delta W_{\mathrm{grad}})
&=\frac{\|\Delta W_{\mathrm{grad}}\|_F}{\sqrt{mn}},\\
&=\frac{\eta\gamma\,\|\mathcal P(D_t)\|_F}
{\sqrt{\min(m,n)}}.
\end{aligned}
$$

以满秩矩阵的理想极分解为参照，其输出范数为 $\sqrt{\min(m,n)}$，此时 RMS 就是 $\eta\gamma$。Pion 的谱输出范数随过滤而变，所以相同学习率和尺寸因子也不代表相同的参数更新幅度。脚本用 `gradient_update_rms` 记录这一梯度项的 RMS。

<details id="flat-rank-study">
<summary>平坦谱：有效秩不变，更新幅度为何仍会随维数急降？</summary>

取完整方阵 $D=I_r$，令 ε 为零。它有 $r$ 个相等奇异值，经 Frobenius 归一化后都变成 $1/\sqrt r$。若五步调度的标量响应记为 $F$，那么所有输出系数仍相等，且精确算术下：

$$
\begin{aligned}
\operatorname{erank}(\mathcal P(I_r))&=r,\\
\frac{\operatorname{RMS}(\Delta W_{\mathrm{grad}})}{\eta\gamma}
&=\left|F\!\left(\frac1{\sqrt r}\right)\right|.
\end{aligned}
$$

因此，平坦谱没有较弱的尾部可单独去掉；维数增大时，每个归一化奇异值都一起靠近零。只要系数仍为正，有效秩就保持 $r$，即使每个系数已经小得几乎无法存储。

下降为何如此快？在原点附近，单次抑制以 $2.5\sigma^3$ 为主，五次复合的最低次项变成：

$$
f_s^{\circ5}(\sigma)\sim
\left(\frac52\right)^{121}\sigma^{243},
\qquad \sigma\to0^+.
$$

代入 $\sigma=1/\sqrt r$，就得到大维数下约按 $r^{-121.5}$ 下降的幅度。这是原点附近的渐近式；下面的图表仍使用完整多项式，并排除权重衰减：

<figure class="article-figure" id="fig-pion-flat-rms">
  {{< post-image src="assets/flat-spectrum-rms.png" alt="完整单位矩阵的平坦谱比较：横轴为维数，纵轴为梯度项更新 RMS 与学习率尺寸常数乘积之比的十进对数；纯抑制随维数急降，增加提升次数会减缓下降，理想极分解保持比值一" >}}
  <figcaption><span class="article-figure__number">图 6</span><span class="article-figure__text">完整单位矩阵的五步响应。Pion 曲线用 100 位 Decimal，Muon 用 Float64；两条水平线标出 BF16 与 Float64 格式最小正数，仅作表示范围参考。</span></figcaption>
</figure>

图 6 展示构造输入的更新幅度，不是模型梯度测量或训练成绩。格式最小正数不能预测某个设备、矩阵乘法内核或中间步骤的实际下溢位置。同一个构造还可以比较高精度数学参考与实际低精度计算。下表使用**零提升、五抑制**；右列是固定官方辅助函数在 CPU 上、ε 为零时的观察值；PyTorch 2.8.0+cu128 与独立的 2.8.0+cpu 环境在这 48 组平坦谱观察中给出相同系数：

| 维数 $r$ | 高精度参考系数 | 官方 CPU BF16 系数 |
| ---: | ---: | ---: |
| 2 | 2.589 × 10⁻⁵ | 2.110 × 10⁻⁵ |
| 4 | 5.908 × 10⁻³² | 5.393 × 10⁻³² |
| 8 | 4.065 × 10⁻⁶⁵ | 0 |
| 16 | 3.131 × 10⁻¹⁰⁰ | 0 |
| 64 | 2.335 × 10⁻¹⁷² | 0 |
| 128 | 9.108 × 10⁻²⁰⁹ | 0 |

这里从 $r=8$ 起存成零，是这组输入、调度与 CPU 实现的结果，不能当成所有 BF16 设备的统一阈值。标准库 Float64 演算在 $r=4096$ 时也得到零，而高精度参考仍为正。**数学上的非零秩、存储后的零谱、更新的实际幅度需要分别记录。** 已经舍入为零的输出，事后再乘学习率或尺寸因子也无法恢复；尚未变为零的系数还可能在后续乘法中继续舍入或下溢。

下载包内的 [平坦谱脚本](flat_spectrum.py)默认只用标准库，不分配大型矩阵：

<div style="break-inside: avoid">

```bash
python3 -B flat_spectrum.py > flat-spectrum.json
# 可选：另装 PyTorch，核对固定官方源码后，加入小型 CPU/BF16 矩阵观察
python -B flat_spectrum.py \
  --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py \
  > flat-bf16.json
```

</div>

[标准库参考](assets/flat-spectrum-reference.json)保留未乘学习率与尺寸因子的系数及其对数；[CPU BF16 记录](assets/flat-spectrum-bf16-reference.json)另外保存 $r\le128$、三种调度与两种 ε 的 48 组观察。图中增加提升次数能保留更大的系数，但这没有证明某个调度对实际任务更优，也不能据有效秩自动选择 $k_p$。

</details>

[逐步计算脚本](pion_step.py)采用 Float64，返回新权重、新缓冲和中间记录，不改写输入数组。准备好 11.2 的 NumPy 环境后，在下载包目录运行：

```bash
.venv/bin/python -B pion_step.py
.venv/bin/python -B pion_step.py --missing-gradient decay
```

默认构造 $W_0=\operatorname{diag}(1,2)$，依次提供 $G_1=\operatorname{diag}(0.8,0.2)$、零矩阵、缺失梯度。为了方便手算，取 $\beta=0.5$、Nesterov 开启、$\eta=0.01$、$\lambda=0.1$、$\gamma=0.2$、ε 为零，谱调度为一提升四抑制。这些是本例的演算参数。第一项对角元素的过程为：

| 输入事件 | 新缓冲 $B_{11}$ | 变换输入 $D_{11}$ | 更新后 $W_{11}$ |
| --- | ---: | ---: | ---: |
| 非零梯度 | 0.4 | 0.6 | 0.996172 |
| 零梯度 | 0.2 | 0.1 | 0.992347 |
| 缺失梯度，跳过参数 | 0.2 | — | 0.992347 |
{.table-readable}

第二步虽然当前梯度为零，历史动量仍然非零；其谱形状与第一步相同，ε 为零时仍产生近似相同的谱更新。**零梯度不等于零更新，动量范数下降也不直接意味着归一化后的更新缩小。** 第三步则没有新的梯度观测：脚本默认的 `skip` 保持缓冲与权重，`decay` 保持缓冲、仅将权重乘以 0.999，此时 $W_{11}$ 约为 0.991355。

这两个缺失梯度选项分别表示固定 [RL/verl 整矩阵优化器](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/RL/verl/verl/utils/muon.py#L449-L512)的跳过策略与 [VLA-Adapter](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/VLA/VLAAdapter/pion_optim/muon.py#L259-L294)的仅衰减分支。[完整三步记录](assets/step-reference.json)保留两种结果；它解释单参数策略，多参数通信循环还需另查执行顺序。

该脚本只处理一个未分片二维矩阵。参数路由、AdamW 辅助组、BF16 与分布式执行仍由真实训练器负责。可选的 [官方多步 CPU 探针](probe_official_steps.py)以非零动量、Nesterov 开关、零／缺失梯度及权重衰减调用固定的 RL 优化器；它使用秩一输入来单独核对更新顺序，[结果记录](assets/official-steps.json)不代表一般矩阵的混合精度逐位一致性。

<details id="vla-buffer-order">
<summary>多参数异步通信：为何还要检查相邻的缺失梯度</summary>

固定版本的 [VLA-Adapter 循环](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/VLA/VLAAdapter/pion_optim/muon.py#L271-L310)在 `grad is None` 时，把更新输入指向 `update_buffer_views[rank]` 并清零；等待上一批通信、读取上一批结果的操作却在其后。同一元素数分组内，若前一参数有梯度、后一参数缺失，清零与尚未消费的接收缓冲区会发生存储冲突。

[缓冲时序探针](probe_official_vla_buffers.py)保留该类源码，只把 CUDA 缓冲分配改到 CPU，再确定性重放两种完成顺序。两个权重都从全 1 的 $2\times2$ 矩阵开始，前一参数梯度为 $\operatorname{diag}(1,0)$、后一参数为 `None`；取 $\eta=0.01$、$\lambda=0.1$、$\gamma=0.2$，分别独立更新时，其首元素应为约 **0.996172、0.999000**。观察到：

| 结果写入时机 | 前一参数首元素 | 缺失梯度参数首元素 |
| --- | ---: | ---: |
| 清零前已完成 | 0.999000 | 0.999000 |
| 等待时才完成 | 0.996172 | 0.996172 |
{.table-readable}

前一种顺序丢失了前一参数的梯度更新；后一种把旧更新带给缺失梯度参数。另用保持同一存储的扁平输出视图适配单进程 CPU/Gloo，本次也观察到了后一种结果。以上 24 个 CPU 组合中，21 个零梯度、反向排列及不同元素数分组对照均符合独立更新；Gloo 的具体完成顺序可能变化。

进一步直接调用未改动的官方类、原始 CUDA 缓冲分配与未包装的单进程 NCCL：在 RTX 5090 D v2、PyTorch 2.8.0+cu128、NCCL 2.27.3 上重复 5 轮、每轮 8 种组合，35 个对照均符合独立更新，同组“非零梯度 → None”的 5 次调用都使前一参数只衰减到 **0.999000**，丢失了应有的梯度项。[完整记录](assets/vla-buffer-reference.json)保存这 40 个 CUDA 组合与前述 24 个 CPU 组合。它覆盖单卡、单进程和微小矩阵，没有运行模型或多卡训练。

接入该固定版本时，应在实际后端增加“有梯度参数后紧跟缺失梯度参数”的顺序回归，同时检查两者的权重和缓冲。单矩阵的 `missing_gradient="decay"` 不能作为这段多参数通信流程正确的证明。可选探针另需 PyTorch，下载包内运行：

<div style="break-inside: avoid">

```bash
python -B probe_official_vla_buffers.py \
  --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py \
  --gloo
# 有可用 CUDA GPU 时，另查原始单卡 NCCL 路径
python -B probe_official_vla_buffers.py \
  --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py \
  --gloo --nccl
```

</div>

</details>

<details id="vla-padding-recovery">
<summary>两个或三个进程：全部参数有梯度时，尾部补齐也可能覆盖上一批更新</summary>

缺失梯度并非唯一触发点。相同元素数的参数按进程数分批，最后一批不足时，固定源码的[补齐分支](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/VLA/VLAAdapter/pion_optim/muon.py#L279-L310)同样把 `g` 指向接收缓冲并清零，随后才消费上一批通信结果。于是，**即使每个真实参数都有梯度，补齐用的零也可能改写仍要读取的更新。**

[多进程 CPU/Gloo 探针](probe_official_vla_padding.py)构造两个进程、三个同形参数。每个参数都是 $2\times2$ 矩阵，初值依次为全 1、全 2、全 3；第二个参数的梯度为 $-e_1e_2^\top$。取 $\eta=0.01$、$\lambda=0.1$、$\gamma=0.2$、一提升四抑制，不使用 Nesterov。第一步独立计算时，第二个参数的 $(1,2)$ 元素应为：

$$
\begin{aligned}
&2(1-0.01\cdot0.1)+0.01\cdot0.2\sqrt2\\
&\quad=2.000828427\ldots
\end{aligned}
$$

探针保留真实 Gloo 通信，并明确区分两种完成顺序：自然异步返回，以及在包装函数返回句柄前先等待完成。后者确定性重放“上一批已完成、但尚未消费”的顺序；[PyTorch 2.8 的 CPU `Work.wait()`](https://docs.pytorch.org/docs/2.8/distributed.html#torch.distributed.Work)会阻塞至工作完成。这是受控时序，不是对未包装后端的普遍观察。在此顺序下：

| 同一个参数元素 | 进程 0 | 进程 1 |
| --- | ---: | ---: |
| 独立正确结果 | 2.000828427 | 2.000828427 |
| 原类的受控时序结果 | 2.000828427 | 1.998000000 |

进程 1 丢失了梯度项，两个副本也出现分歧；三个进程、四个同组参数时，进程 1、2 均出现相应问题。自然异步运行中，补齐例子本次没有出错，但另一组缺失梯度例子出现了**各进程结果相同、却共同偏离独立正确结果**的情况。因此，回归需要同时检查副本一致性与更新正确性，不能只检查进程之间相等。

探针另构造一个**显式修改内存中类定义的候选版本**：把缺失梯度和补齐的两处接收缓冲引用改为私有零张量，避免清零尚未消费的接收存储：

<div style="break-inside: avoid">

```python
# 两个分支各替换一次；其余通信循环保持原样。
g = torch.zeros_like(update_buffer_views[self.rank])
```

</div>

候选版本只用来检验这一存储冲突的解释。它与原类都把 CUDA 缓冲分配适配到 CPU，并将 Gloo 输出改成共享原存储的扁平视图；官方源文件不被重写。每组依次运行“配置的梯度、零梯度但保留历史动量、全部缺失梯度”三步，逐步核对动量状态的归属、初始化与 EMA 数值，另设整批参数和不同元素数分组作为对照。

<div style="break-inside: avoid">

```bash
python -B probe_official_vla_padding.py \
  --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py \
  --world-size 2 --output padding-two.json
python -B probe_official_vla_padding.py \
  --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py \
  --world-size 3 --output padding-three.json
```

</div>

[保存的两／三进程记录](assets/vla-padding-reference.json)共 48 个配置组合、360 次进程内 `step()` 调用：候选版本的 24 个组合全部符合独立三步更新；原类的 24 个组合中，本次有 10 个组合至少出现一步偏差。自然完成时序的观察可随运行改变。这个结果没有验证多 GPU/NCCL、模型训练、吞吐量或所有参数形状；候选改动进入真实训练器前，仍须在目标后端检查状态、权重与多进程结果。

</details>

## 12. 接入自己的任务时应记录什么 {#practical-checks}

先保留一个明确的 AdamW 或现有 Muon 基线，再只改目标模块的更新方式。对 VLA，按视觉、语言、动作分别观察梯度谱，而不是凭模块名称决定全部配置；对 RLVR，额外核对 Q/K/V/O 的头数、切分轴和分片布局。

建议随训练记录以下量，因为它们能帮助解释“性能变化来自哪里”：

| 记录 | 回答的问题 |
| --- | --- |
| 参数组与实际优化器分支 | 是否意外把 embedding、输出层或不适合的视觉层做了高通？ |
| $k_p$、总步数、归一化与尺寸缩放 | 所用谱响应和有效更新幅度是否与预期一致？ |
| 参数、动量与多项式的 dtype | 舍入发生在哪一步，是否混用了不同精度的状态？ |
| 模块谱、erank、梯度项更新 RMS／权重 RMS | 尾部被压低时，是否也把全部有用更新压小？记录梯度项与衰减项各自的幅度。 |
| 固定检查点的组／batch 梯度 SNR、采样组标识与保留率 | 独立采样单位是什么？是否只统计了保留组，或把同步副本重复计数？ |
| 分头命中／回退、各头更新范数 | 是否真的按正确头结构更新？ |
| RLVR 奖励退化组、clip fraction、KL | 优化器之外，是否已经缺乏可用优势信号或出现过大策略变化？ |
| 固定预算的独立评估与重复训练 | 收益能否跨检查点与种子复现？ |
{.table-readable}

记录原始梯度谱时，还要明确采样位置。固定 VLA-Adapter 实现的 Nesterov 分支会用 `g.lerp_` 原地混合梯度，Muon 辅助函数还可能原地归一化 BF16 输入；若在 `step()` 之后读取 `.grad`，看到的可能已经是混合或归一化后的值。应在更新前、完成所需梯度同步与缩放还原后保存原始梯度，再分别记录动量与最终更新；分片训练还需注明统计的是本地分片还是完整逻辑矩阵。

仅看训练 loss 不足以确认谱过滤正确；仅看 erank 降低也不足以确认去除了噪声。对于 flat spectrum，强抑制甚至可以几乎清空更新；对于有用信息位于尾部的模块，高通可能直接丢失信息。

比较调度时，可以先复核论文与固定源码使用的参数，再在同一模块、同一预算下比较 $k_p=0,1,2$，同时记录实际更新 RMS。若另做幅度匹配对照，用来区分“改变谱形状”与“整体步幅变小”的影响，应把它明确标为机制诊断：重新缩放后的优化器已经改变，不能与原超参数复现混为一组。单看有效秩或本节的单位矩阵曲线，都不能决定新任务的最优调度。

Pion 提供的是一种**不显式求 SVD 的软谱选择机制**。它改变了矩阵更新的归纳偏置，保留多少方向取决于归一化后的谱与调度。接入新任务时，应把谱与更新幅度作为诊断量，以固定预算的独立评估判断是否有效。

## 13. 常见疑问 {#faq}

### Pion 是否只保留前几个奇异值？ {#faq-top-k}

没有预先指定整数 top-k。各方向经过同一个平滑函数，超过过渡区域的方向更容易保留。不同层、不同训练步中显著参与更新的方向数可以变化；在精确算术下，有限五步不会把原本非零的方向直接截成零。

### 有效秩低，就一定要用 Pion 吗？ {#faq-erank}

不一定。有效秩只描述谱集中程度，不能标记噪声，也不包含梯度方向是否正确的信息。还需比较更新幅度和独立任务指标。

还要区分两种归一化：erank 用奇异值之和，谱多项式的输入则先除以 Frobenius 范数。一个 erank 数值不能唯一确定各个归一化奇异值，也不能直接给出应选择的提升次数。

### 分头模式是不是让各头获得不同学习率？ {#faq-head-learning-rate}

它使用独立归一化与谱变换，不等同于直接为每个头指定标量学习率。改变子空间分块后，即使外层学习率相同，更新也一般不同。

### 五步相同，是否可以保证训练时间相同？ {#faq-step-time}

这里的五步是五次多项式迭代；按所核对的典型源码，每步包含三次矩阵乘法算子，五步合计约十五次，不需要每步 SVD。实际耗时还受矩阵形状、分头 batch、内核、分布式通信与框架影响。论文报告的近似同耗时应留在其实现与硬件范围内。

### 保留较少奇异方向，会自动减少显存吗？ {#faq-memory}

低秩矩阵仍可以每个元素都非零。[固定实现的二维 Pion 分支](https://github.com/OPTML-Group/Pion/blob/87965014b360bde604632cba0aa12e423c6c63c6/RL/verl/verl/utils/muon.py#L476-L492)仍保存同形动量，并用稠密矩阵计算；分头 reshape 本身也不压缩参数或状态。这里的二维分支只维护动量，与 AdamW 的一、二阶矩状态不同；峰值内存还取决于梯度与 Gram 中间量的存储、状态精度，以及通信与分片方式。

### Pion 会让机器人轨迹更平滑吗？ {#faq-motion-smoothing}

论文展示了更成功的操作与部分定性 rollout，但谱高通不是动作时间滤波器，不能仅由多项式形式推出轨迹平滑或无碰撞保证。

### 论文最值得保留的方法结论是什么？ {#faq-takeaway}

选择矩阵优化器时，需要关注谱结构与训练阶段。统一白化、软高通、整矩阵和分头更新分别引入不同偏置；在论文所测设置中，VLA 的模块分工和 RLVR 的分头高通比直接照搬 Muon 更有效。 作者在[附录 M](https://arxiv.org/pdf/2605.19282v1#page=43)预期 Pion 用于 LLM 预训练会落后于 Muon，并把调整高通以恢复预训练探索能力列为后续工作。

## 14. 延伸阅读与复算材料 {#references}

- [论文 v1，正文、附录与实验配置][paper]；[项目介绍与交互谱响应][project]；[官方代码固定版本][code]。
- 图 2、4、5 裁取自 Fan 等上述论文的对应面板，保留原数值与曲线；原文按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 公开。
- 本站：[Transformer 与注意力](../transformer-attention/)、[PPO、DPO 与 GRPO](../ppo-dpo-grpo/)、[RLHF、RLVR 与 RLCD](../rlhf-rlvr-rlcd/)、[VLA 演进](../vla-evolution/)。
- [谱计算小实验](pion-spectral-lab.zip)与[完整数值记录](assets/spectral-reference.json)。图 3 可用包内 `make_figures.py` 重画，需要 NumPy 与 Matplotlib。

## 阅读自测与验收 {#self-check}

先独立作答或操作，再核对本节答案；实践结果以实际记录为准。

1. 矩阵msign与逐元素sign有什么不同，零奇异方向如何处理？
2. 五步Muon是否会把每个奇异值精确变为1？
3. erank使用奇异值还是奇异值平方，是否就是整数秩？
4. 动作只有7维是否足以证明所有动作模块梯度的秩不超过7？
5. Pion的高通是否指动作的时间频率？
6. Promotion与Suppression的系数分别由哪些约束确定？
7. Suppression为何也会提升一部分输入，0.816497与五步0.5响应点有什么区别？
8. 纯抑制面对平坦谱，是否会只去掉噪声？
9. 分头模式是否精确保留预训练权重范数或直接指定不同学习率，标准注意力的范数解释能否直接用于含QK-Norm的Qwen3？
10. PyTorch布局中的Q/K/V与O为何沿不同轴拆分，GQA使用哪个头数？
11. VLA主结果中的Pion配置是否将所有二维矩阵都高通？
12. 图2的4,500步动作模块对照能否与主结果1,500步全模型对照直接合并？
13. VLANeXt提升应写成百分比还是百分点？
14. 85.6%的真机平均对应哪些任务、多少次试验？
15. 只做分头Muon和反向低通消融各说明了什么？
16. 本文数学小实验与完整训练优化器还差哪些部分？

17. 加入固定 ε 后，缩放同一非零矩阵是否仍保持相同响应？
18. 分头反例中的 10:1 动量比例是否由分头变换保留？
19. 官方启动脚本的默认停止条件是否自动等于论文预算？
20. 平坦谱的输出有效秩较高，能否说明更新幅度也较大？
21. 强扰动占据最大奇异方向时，Pion能否根据任务语义保留较弱真信号？
22. 零梯度与缺失梯度是否等价，两套官方实现的缺失梯度策略与多参数通信应分别怎样检查？
23. 平坦谱维数增大时，为什么输出有效秩不变而 RMS 可以趋近零，数学参考与存储后的零又有何区别？
24. 全部真实参数都有梯度时，多进程尾部补齐为何仍需回归；副本相等能否证明更新正确，私有零候选验证到什么范围？
25. 有限样本的均值梯度功率为何向上偏；校正功率无偏是否意味着比值无偏，负校正值与零样本方差分别说明什么？
26. 把四次独立梯度各复制八次后，能否按 32 次独立采样校正信号功率；按组汇总又需要什么条件？
27. 组内优势之和为零，为何混合组的策略梯度仍可非零；只统计非退化组的 SNR 能否直接与全部提议组比较？
28. 单步梯度每次都是秩一时，EMA 动量是否也保持秩一？

<details id="pion-self-check-answers">
<summary>展开 28 题答案</summary>

1. 前者沿非零奇异子空间取极分解方向；后者逐元素取符号。零奇异方向不应凭空补成非零。
2. 不会；有限系数与步数只有近似谱响应，部分输出可高于1。
3. 使用奇异值归一化；它是熵指数，通常不是整数，也不是阈值秩。
4. 不足；单个线性层满足rank(∇W L)≤min(N,d_in,d_out)，其中N可合并批量与时间位置。最终输出维数限制其自身输出矩阵；样本间变化的非线性反传可使隐藏梯度满秩，标量输出的构造也能给出秩8的隐藏梯度。
5. 不是；这里保留较大的归一化奇异值，与动作时间FFT无关。
6. 二者共享f(1)=1与f′(1)=0；提升再满足单调与边界约束并最大化原点斜率，抑制增加f′(0)=0。
7. 高于该不稳定固定点的输入向1增长；五步0.5响应点则由完整调度求得，二者不是同一阈值。
8. 不会；例如四个相等奇异值归一化后为0.5，纯抑制五步把全部更新系数一起压到极小。
9. 都不是；它独立归一化与过滤各头，实际幅度仍由动量谱、缩放和学习率决定。Qwen3还在投影后归一化Q/K，不能只由投影权重范数推断注意力温度。
10. Q/K/V的头在输出侧，O的头在输入侧；GQA分别使用query与KV头数。
11. 不是；视觉、语言二维矩阵仍用Muon，动作二维矩阵用Pion，排除项等辅助参数用AdamW。实际路由还需核对参数名与形状。
12. 不能；预算与改变的模块不同，必须保留各自对照。
13. 这里给出成功率之差，应写百分点；相对百分比需要再除以基线。
14. 三个抓放任务各30次，Pion的舍入值对应28、25、24次成功，合计77/90。
15. 前者说明仅分块不足以修复统一白化；后者支持保留较强方向的过滤方向在所测设置中的作用。
16. 标量与矩阵核心只计算谱变换；逐步例子补上单矩阵的动量、Nesterov、衰减与尺寸因子，仍缺AdamW辅助组、自动微分、混合精度、参数路由与分布式训练。

17. 不一定；固定正 ε 会引入与动量范数有关的因子，微小输入受到额外压低。
18. 不保留；独立归一化可以消除幅度比例，该例的两个输出块范数相同。
19. 不能；核对的三份 GSM8K 脚本按 15 个 epoch 设置，仍需核对实际步数和评估检查点。
20. 不能；有效秩描述相对谱形状，输出可以在保持较高有效秩时几乎没有幅度。
21. 不能；构造对照表明，高通也会保留强扰动并压低弱信号，它没有任务语义标签。
22. 不等价；零矩阵仍更新缓冲，历史动量可以继续产生谱更新。缺失梯度时，固定RL/verl分支跳过权重，VLA-Adapter分支意图只衰减权重。单矩阵演算不验证异步通信；VLA多参数循环还要检查相邻缺失梯度对接收缓冲的复用顺序。
23. 完整单位矩阵归一化后每个奇异值都是1/√r；五步过滤给出相等的正系数，故数学有效秩仍为r，RMS/(ηγ)却随该系数一起下降。低精度可能将其存成全零，需另记零谱约定，不能倒推精确秩为零。
24. 补齐分支也可能在消费上一批通信结果前清零接收存储。需要同时核对跨进程一致性和独立正确结果；副本可能共同出错。私有零候选只通过小型CPU/Gloo两／三进程对照，尚未验证多GPU、模型训练或性能。
25. 独立重复K次时，样本均值功率的期望多出V/K。减去无偏样本总方差的1/K可校正功率，但随机分母使比值一般仍有偏。负校正值是有限样本结果，不能解释为负的真实SNR；样本方差为零也不能证明总体无噪声，示例将比值记为null。
26. 不能；组内副本相关，平均梯度的真实方差没有因复制而降低。本例逐行校正的功率期望为1+7/31，而以四个原始独立组校正的期望为1。应保留采样组标识，并另行满足组间独立、同一检查点和一致梯度定义等条件；按组并不使SNR比值自动无偏。
27. 梯度还包含每条回答的score；例如优势(1,-1)乘score(0.5,-0.5)后平均为0.5。只保留混合组会改变采样分布，条件SNR不包含大量被丢弃的零更新组，还应记录保留概率和提议成本。本例没有多token残差、clipping或KL，不能直接预测真实训练SNR。
28. 不一定；不同方向的秩一梯度可共同留在历史中。零初始缓冲、beta=.5的三方向例子给出动量对角线(.125,.25,.5)，秩为3。应分别记录当前梯度、动量和实际送入谱变换的Nesterov混合矩阵。
</details>

[paper]: https://arxiv.org/abs/2605.19282v1
[project]: https://chongyu-fan.netlify.app/posts/pion/
[code]: https://github.com/OPTML-Group/Pion/tree/87965014b360bde604632cba0aa12e423c6c63c6

**通过条件：** 复算谱形、更新幅度与状态步进，并明确零梯度和缺失梯度的区别；高精度/CPU 结果与 BF16、GPU、多 Rank 训练结论分别记录。
