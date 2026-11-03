---
title: "Muon 的学习率与 Batch Size：缩放规律、噪声与训练预算"
date: 2026-11-03
draft: false
tags: ["Muon", "Learning Rate", "Batch Size", "Machine Learning", "Regularization"]
categories: ["人工智能"]
authors: ["chase"]
summary: "推导 Muon 学习率随 Batch Size 从平方根增长到饱和的条件，分析平均场近似、谱各向异性与反例，并把动量记忆、权重衰减、梯度累积和训练预算放到同一数据时钟下。"
description: "推导 Muon 学习率随 Batch Size 从平方根增长到饱和的条件，分析平均场近似、谱各向异性与反例，并把动量记忆、权重衰减、梯度累积和训练预算放到同一数据时钟下。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
images: ["assets/batch-three-clocks.png"]
reading_prerequisites: "梯度、方差、二阶 Taylor 展开与矩阵奇异值分解；建议先了解 Muon 的极分解方向"
reading_focus: "先固定 Batch 与学习率约定，再推导局部最优步长及其近似条件，最后区分优化器步数与数据量，并联动检查动量、衰减和调度。"
related_posts:
  - "/posts/ai/muon-optimizer"
  - "/posts/ai/foundations-modern-ai"
  - "/posts/ai/transformer-attention"
---

**Muon 没有一条脱离条件的“Batch Size 增大多少，学习率就乘多少”的规律。** 在苏剑林《[重新思考学习率与 Batch Size（三）：Muon][su-batch]》采用的平均场与曲率近似下，局部最优步长呈现“较小 Batch 时近似平方根增长、较大 Batch 时趋于饱和”的形式。真正迁移训练配置时，动量、权重衰减、调度与更新次数也同时发生变化。

这篇文章先把这条公式推导清楚，再解释它省略了什么。Muon 的几何与四种尺寸约定见[第一篇](../muon-optimizer/)；本文的 $B$ 始终表示**一次优化器更新对应的全局有效 Batch**，不是单卡显存里一次前向的大小。

<figure class="article-figure" id="fig-muon-batch-clocks">
  {{< post-image src="assets/batch-three-clocks.png" alt="全局有效 Batch 同时改变梯度统计、固定数据预算下的优化器步数，以及动量和累计权重衰减；比较时需要使用相同样本或 token 预算，并共同选择学习率、动量与衰减" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text"><strong>改 Batch 同时改了三个时钟。</strong>统计噪声、更新次数与历史记忆共同影响比较；图中 N 表示总数据预算。</span></figcaption>
</figure>

## 1. 先固定 Batch、损失与学习率的定义 {#conventions}

### 1.1 全局 Batch 与 microbatch

若每个数据并行副本处理 $b$ 个等权样本，共有 $D$ 个副本，累积 $A$ 次后更新，那么：

$$
B=bDA.
$$

改变 microbatch、同时反向调整累积次数使 $B$ 不变，通常不构成本文统计意义上的 Batch 变化。浮点求和、随机算子和跨样本运算仍可能让实际结果不完全相同。

对于 token 数、样本权重或有效 mask 数量不一致的情况，应该累计损失梯度的总和，再除以全局有效权重。不能把每个 microbatch 的均值再等权平均。例如两组分别有 1、3 个有效样本，均值梯度为 3、1，正确全局均值为 1.5，等权平均却得到 2。

### 1.2 同一个学习率数字未必代表同一步长

Muon 的参数更新一般写成：

$$
\Delta W=-\eta\,s(W)\,O,
\qquad O\approx\operatorname{polar}(M).
$$

$\eta$ 是外层学习率，$s(W)$ 是尺寸因子。第一篇讨论的朴素、Jordan、μP 形式与 Moonlight 约定具有不同的 $s(W)$。比较学习率曲线前，应先统一它们，或直接记录实际的更新 RMS 与相对更新量 $\|\Delta W\|_F/\|W\|_F$。

下文推导先取 $s(W)=1$，不加入动量、衰减和有限步 NS。若单个矩阵使用固定正系数 $s$，相同参数位移对应的外层学习率应再除以 $s$。

## 2. 一个局部最优步长的通用表达式 {#local-step}

设平均损失为 $L(W)$，真实梯度为 $G$，随机更新方向为 $\Phi_B$。Hessian 用作用于矩阵的线性算子 $\mathcal H$ 表示：

$$
\begin{aligned}
\mathbb E[L(W-\eta\Phi_B)]
&\approx L(W)-\eta a_B+\frac{\eta^2}{2}b_B,\\
a_B&=\mathbb E\langle G,\Phi_B\rangle,\\
b_B&=\mathbb E\langle\Phi_B,\mathcal H[\Phi_B]\rangle.
\end{aligned}
$$

当 $a_B>0$、$b_B>0$ 时，这个二次近似的最优步长是：

$$
\eta_B^*=\frac{a_B}{b_B}.
$$

这条表达式把问题分成两部分：随机方向与真实梯度对齐多少，以及这些方向上的曲率有多大。它不是完整训练的最优学习率定理；非凸区间、较大步长和更新后变化的曲率，都可能使局部二阶近似失效。

### 2.1 SGD 为什么常见线性增长到饱和

用向量化的协方差 $\Sigma$ 描述单样本梯度噪声，$H_{\rm vec}$ 表示向量化后的 Hessian。样本独立，且 SGD 取 $\Phi_B=\widehat G_B$ 时：

$$
\begin{aligned}
a_B&=\|G\|_F^2,\\
b_B&=\langle G,\mathcal H[G]\rangle
+\frac1B\operatorname{tr}(H_{\rm vec}\Sigma).
\end{aligned}
$$

若真实梯度方向的曲率为正，噪声的曲率加权项也非负，于是可写成 $\eta_B^*=\eta_\infty/(1+C/B)$。当噪声项占主导时近似随 $B$ 线性增加，进入信号与曲率主导区后饱和。这里的 $C$ 带有曲率权重，不能直接用普通的梯度方差比替代。

## 3. Muon 的平均场推导 {#mean-field}

### 3.1 先写出矩阵噪声模型

令 $G\in\mathbb R^{m\times n}$、$m\ge n$ 且满列秩。随机平均梯度满足：

$$
\widehat G_B=G+E_B,
\qquad \mathbb E[E_B]=0.
$$

暂时假定不同矩阵元素的噪声不相关、每个元素的方差均为 $\sigma^2/B$。于是：

$$
\mathbb E[\widehat G_B^\top\widehat G_B]
=G^\top G+\frac{m\sigma^2}{B}I_n.
$$

这里 $\sigma^2$ 是**单个坐标**的单样本方差；总噪声功率则是 $v=mn\sigma^2$。把两者混用，会在临界 Batch 中多出或少掉矩阵元素数量。

### 3.2 第一次近似：把非线性期望拆开

理想 Muon 取 $\Phi_B=\operatorname{polar}(\widehat G_B)$。平均场近似写成：

$$
\begin{aligned}
\mathbb E[\Phi_B]
&=\mathbb E\!\left[\widehat G_B
(\widehat G_B^\top\widehat G_B)^{-1/2}\right]\\
&\approx G\left(G^\top G+
\frac{m\sigma^2}{B}I_n\right)^{-1/2}.
\end{aligned}
$$

这不是严格的期望恒等式：随机矩阵与它的逆平方根相关，不能仅凭均值与方差就完全确定这个期望。[平均场推导][su-batch]

若 $G=U\operatorname{diag}(\sigma_i)V^\top$，上式在第 $i$ 个奇异方向上的幅度为：

$$
c_i(B)=\frac{\sigma_i}{\sqrt{\sigma_i^2+m\sigma^2/B}}.
$$

即便接受第一次近似，不同奇异方向也有不同的噪声与信号比例。

### 3.3 第二次近似：用平均能量替代整个谱

再用：

$$
G^\top G\approx\frac{\|G\|_F^2}{n}I_n,
$$

得到：

$$
\begin{aligned}
\mathbb E[\Phi_B]&\approx
\frac{\operatorname{polar}(G)}{\sqrt{1+B_c/B}},\\
B_c&=\frac{mn\sigma^2}{\|G\|_F^2}
=\frac v{\|G\|_F^2}.
\end{aligned}
$$

分母必须是梯度 Frobenius 范数的**平方**：分子与分母都是梯度平方量，比例才具有 Batch 的含义。

### 3.4 曲率条件同样重要

苏剑林的推导进一步选择如下 Hessian 近似：

$$
\mathcal H[X]=\eta_\infty^{-1}X(G^\top G)^{1/2}.
$$

若随机梯度几乎必然满列秩，则 $\Phi_B^\top\Phi_B=I_n$。因此：

$$
\begin{aligned}
b_B
&=\eta_\infty^{-1}
\operatorname{tr}\!\left[(G^\top G)^{1/2}\right],\\
a_B
&\approx\frac{\operatorname{tr}[(G^\top G)^{1/2}]}
{\sqrt{1+B_c/B}}.
\end{aligned}
$$

代入局部二次最优步长，得到核心结果：

$$
\boxed{\eta_B^*\approx
\frac{\eta_\infty}{\sqrt{1+B_c/B}}}.
$$

它依赖独立噪声、坐标方差简化、非线性期望近似、谱各向同性近似，以及特定曲率结构。严格秩亏时使用广义逆，也不能直接继续套用满列秩的等距恒等式。

## 4. 平方根区间与饱和区间 {#scaling-rule}

由上式：

$$
\begin{aligned}
B\ll B_c &: \quad
\eta_B^*\approx\eta_\infty\sqrt{B/B_c},\\
B\gg B_c &: \quad
\eta_B^*\approx\eta_\infty.
\end{aligned}
$$

从 $B_0$ 改到 $B_1$，局部学习率比例是：

$$
\frac{\eta_{B_1}^*}{\eta_{B_0}^*}
\approx\sqrt{\frac{1+B_c/B_0}{1+B_c/B_1}}.
$$

例如 $B_c=4096$，从 256 增加到 1024，比例为 $\sqrt{17/5}\approx1.844$，不是恰好 2。接近饱和区时继续机械乘平方根，会越过这个模型给出的合理步长。

<figure class="article-figure" id="fig-muon-batch-rule">
  {{< post-image src="assets/batch-scaling-law.png" alt="Muon 平均场学习率曲线与平方根外推、SGD 线性到饱和曲线比较，横轴为 Batch 与临界 Batch 的比值，Muon 曲线从平方根区间进入饱和区间" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text"><strong>先判断所在区间。</strong>曲线来自局部模型，纵轴已除以各自的大 Batch 极限；它们不是不同优化器的训练效果排名。</span></figcaption>
</figure>

## 5. 为什么同一个噪声尺度仍不够 {#spectrum}

### 5.1 相同 Frobenius 能量不代表相同谱

比较两个构造梯度：

$$
\begin{aligned}
G_A&=\operatorname{diag}(1,1),\\
G_B&=\operatorname{diag}(\sqrt{1.99},0.1).
\end{aligned}
$$

它们都有 $\|G\|_F^2=2$，所以在相同总噪声功率下，标量 $B_c$ 相同。但在第一次 Gram 期望近似中，令噪声项为 $0.1I$，第二个矩阵的弱方向幅度只有约 0.302，而强方向约为 0.976；第一个矩阵的两个方向都约为 0.953。

<figure class="article-figure" id="fig-muon-anisotropy">
  {{< post-image src="assets/gram-anisotropy.png" alt="两个 Frobenius 能量相同的梯度矩阵，在保留各向异性谱的 Gram 近似下具有不同方向衰减；平坦谱两个方向接近零点九五三，尖锐谱的弱方向约零点三零二" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text"><strong>一个临界 Batch 会压缩掉谱信息。</strong>这是第一次平均场近似内的代数比较，不是随机极分解期望的精确测量。</span></figcaption>
</figure>

因此，平均场曲线适合组织直觉，却不能代替对弱奇异方向、噪声方向与真实更新分布的检查。小奇异值还会让极分解方向对扰动更敏感。

### 5.2 一般曲率下，最优学习率未必单调增加

考虑一个 $2\times1$ 矩阵，真实梯度 $G=(1,0)^\top$，Hessian 为 $\operatorname{diag}(100,1)$。每个样本只在第二个坐标加入均值为零、标准差为 $s$ 的独立高斯梯度噪声，图中取 $s=100$。

平均 $B$ 个样本后，随机梯度可写成：

$$
\widehat G_B=(1,sZ/\sqrt B)^\top,
\quad Z\sim\mathcal N(0,1).
$$

极分解就是单位化该列向量。记它的第一个分量为 $D_1$，则局部二次损失下的**精确**最优步长为：

$$
\eta_B^*=
\frac{\mathbb E[D_1]}{1+99\mathbb E[D_1^2]}.
$$

分子与分母会同时随 $B$ 改变：噪声减少让方向更对齐真实梯度，也让更新更集中到高曲率坐标上。取 $B=1,10,100$，积分得到的最优步长分别约为 $0.01916$、$0.02199$、$0.01958$；大 Batch 极限为 $1/100$，中间区域出现了更大的最优步长。

<figure class="article-figure" id="fig-muon-curvature-counterexample">
  {{< post-image src="assets/curvature-counterexample.png" alt="独立高斯噪声和各向异性二次损失下，精确期望积分得到的 Muon 列向量最优学习率随 Batch 变化的非单调曲线，与大 Batch 极限零点零一比较" >}}
  <figcaption><span class="article-figure__number">图 4</span><span class="article-figure__text"><strong>曲率可以改变缩放形状。</strong>这是构造的二次目标与一维高斯积分，不是模型训练结果；它不满足前文特定的 Hessian 近似。</span></figcaption>
</figure>

这个例子保留了独立采样与 $1/B$ 噪声方差，变化的是曲率结构。它说明“Muon 一定服从平方根增长到饱和”不是算法定义所保证的普遍结论。

## 6. 动量给 Batch 增加了历史 {#momentum}

### 6.1 稳态 EMA 的等效降噪

对固定参数位置、恒定均值梯度和跨步独立噪声，EMA：

$$
M_t=(1-\beta)\widehat G_t+\beta M_{t-1}
$$

的稳态噪声协方差为：

$$
\operatorname{Cov}(M)
=\frac{1-\beta}{1+\beta}\frac\Sigma B.
$$

因此可定义一个用于比较噪声方差的等效 Batch：

$$
B_{\rm eff}=B\frac{1+\beta}{1-\beta}.
$$

例如 $\beta=0.95$ 时，因子为 39。这个量描述稳态线性平均的降噪，不能直接断言“带动量的 Muon 等价于无动量、Batch 放大 39 倍”：方向非线性、均值漂移、跨步相关和初始瞬态都没有被保留。

Nesterov 还会混入当前梯度，其权重序列与普通 EMA 不同。若使用 $H_t=(1-\beta)G_t+\beta M_t$，当前梯度权重为 $1-\beta^2$，往前第 $j$ 步的权重是 $(1-\beta)\beta^{j+1}$，其中 $j\ge1$。它的噪声方差因子应由这些权重的平方和计算，而不能沿用普通 EMA 的结果。

### 6.2 保持更新步记忆与保持数据量记忆

经过 $S$ 次更新，旧信息保留比例是 $\beta^S$。如果 Batch 变为原来的 $k$ 倍，在相同样本数下更新次数减少为 $1/k$。保持相同的数据量记忆，应选：

$$
\beta_{\rm new}=\beta_{\rm old}^{\,k}.
$$

例如从 $B$ 改为 $4B$，$0.95$ 对应约 $0.814506$。保持 $\beta=0.95$ 则保持的是更新步记忆，换算成样本数的记忆长度增加四倍。这是两种不同的选择，不存在同时自动保持的规则。

## 7. 权重衰减也有自己的数据时钟 {#weight-decay}

忽略梯度贡献，常数学习率与衰减下，每步权重乘：

$$
q=1-\eta\lambda.
$$

处理 $N$ 个样本、Batch 为 $B$ 时，总收缩约为 $q^{N/B}$。Batch 增大但不改 $\eta\lambda$，同一数据预算内的衰减次数变少。

若 Batch 乘 $k$，学习率乘 $s$，希望保留同样的按数据量累计收缩，应满足：

$$
\begin{aligned}
q_{\rm new}&=q_{\rm old}^{\,k},\\
\lambda_{\rm new}
&=\frac{1-(1-\eta_{\rm old}\lambda_{\rm old})^k}
{\eta_{\rm new}}\\
&\approx\frac{k}{s}\lambda_{\rm old}.
\end{aligned}
$$

最后一行仅用于 $\eta\lambda$ 很小时。它只保持衰减这一分量的累计作用，不保证完整参数轨迹或最优正则化强度相同。若学习率随数据量变化，应比较整个乘积 $\prod_t(1-\eta_t\lambda_t)$，或其小步长近似 $\exp(-\sum_t\eta_t\lambda_t)$。

这也不同于[第一篇](../muon-optimizer/#scaling)的“换 Muon 尺寸约定”：后者通常不改变更新次数，保持的是同一步中的更新量与收缩量；这里改变了同一数据预算中的步数。

## 8. 梯度累积、分布式与 sum/mean {#accumulation}

### 8.1 先平均原始梯度

取两个构造梯度：

$$
G_1=\operatorname{diag}(4,1),
\qquad G_2=\operatorname{diag}(-1,4).
$$

先平均再极分解，得到 $I$；先分别极分解再平均，得到 $\operatorname{diag}(0,1)$。这不只是步长差异，第一个方向已经消失。因而不能把每个 microbatch 都做一次 Muon，然后平均这些方向，来替代一次全局 Batch 的 Muon。

### 8.2 理想尺度不变性不能替代正确的损失归约

如果整条梯度历史都乘同一个正常数，线性动量也同比缩放，理想极分解可以消去该比例。这解释了某些固定 Batch 情况下，sum 与 mean 不会像 SGD 那样简单改变 Muon 的参数步长。

但实际还存在变化的有效 token 数、固定 $\varepsilon$、有限精度、混合优化器和历史动量。尤其 Embedding 或向量参数仍使用 AdamW 时，所有组必须遵循明确一致的损失与梯度缩放约定。

AMP 的 loss scaling 也不是 Batch Size；应该按训练框架规定先正确反缩放与聚合，再进入优化器。

## 9. 如何估计噪声尺度并组织实验 {#measurement}

### 9.1 用独立梯度记录估计信号和噪声

在同一个参数位置收集 $K>1$ 份独立、每份含 $b$ 个等权样本的平均梯度 $g_j$。令：

$$
\begin{aligned}
\bar g&=\frac1K\sum_jg_j,\\
V&=\frac1{K-1}\sum_j\|g_j-\bar g\|_F^2.
\end{aligned}
$$

可估计单样本总噪声功率为 $bV$，并用：

$$
\widehat S=\|\bar g\|_F^2-\frac VK
$$

校正有限次平均造成的信号功率上偏。当 $\widehat S>0$ 时，$bV/\widehat S$ 可作为噪声尺度的探索性估计。它是两个估计量的比值，不会因为分子、分母分别有无偏性质就自动无偏。

若校正后信号非正，应该报告估计不确定或样本不足；不能把负值直接解释为“真实信号为负”。相同样本的复制、相关序列与参数不断变化的梯度，也不能冒充独立同分布记录。

### 9.2 固定比较的分母

| 比较条件 | 保持了什么 | 仍可能改变什么 |
| --- | --- | --- |
| 固定 optimizer steps | 更新次数 | 总样本数与计算量 |
| 固定 samples / tokens | 数据预算 | 更新次数、动量记忆、累计衰减 |
| 固定训练 FLOPs | 主要算术预算 | 数据量、模型大小、通信占比 |
| 固定 wall-clock | 实际耗时 | 硬件利用率、通信与样本数 |
{.table-readable}

局部公式回答的是当前位置的一次更新，不会直接决定这些完整实验中哪个配置最好。 warmup、峰值学习率与衰减结束位置应同时记录其计数单位；若目标是固定数据预算，通常需要把调度位置也换算到 samples 或 tokens。

## 10. 一个有条件的迁移流程 {#procedure}

1. 固定参数分组、Muon 尺寸缩放、损失归约和 NS 数值设置，先确认实际全局 Batch。
2. 区分只改变 microbatch 的工程调整，还是确实改变了有效独立样本数量。
3. 使用旧配置和相近参数位置的统计，判断是否有理由把平方根规则当作局部初值，而不是直接倍增峰值学习率。
4. 明确动量保持更新步记忆还是数据量记忆；明确衰减保持每步强度还是累计收缩。
5. 按同一数据预算与评估协议做小范围对照，记录实际更新 RMS、相对更新量、损失、异常值与尾延迟。
6. 只有在上述配置稳定后，再评估吞吐和分布式实现；额外的 QK-Clip 等反馈机制也应单独记录触发统计。

[随文脚本](batch_math.py)复算平均场比例、动量与衰减换算、噪声估计和二次目标反例：

```bash
python3 -B batch_math.py
```

它使用构造数据和数值积分，不加载模型。图表可用[绘图脚本](make_figures.py)重画；绘图另需 NumPy 与 Matplotlib。将它与 `batch_math.py` 保存在同一目录，运行 `python3 -B make_figures.py`，三幅数值图会写入旁边的 `assets/`。正交化的有限步、矩阵布局和更新幅度，可回到[Muon 原理与实现](../muon-optimizer/)一起核对。

## 阅读自测与验收

1. microbatch 改变但全局 Batch 不变，是否应机械缩放学习率？
2. 不等长 microbatch 的梯度均值为什么要加权？
3. 局部最优学习率中的分子和分母分别度量什么？
4. Muon 的平均场推导用了哪两次不同的近似？
5. 单坐标方差和总噪声功率怎样影响临界 Batch 的定义？
6. 为什么临界 Batch 的分母是梯度范数的平方？
7. 平方根规律在哪个区间成立，什么时候趋于饱和？
8. 相同 Frobenius 能量是否保证相同的方向衰减？
9. 一般 Hessian 下，学习率能否出现非单调变化？
10. EMA 的等效 Batch 是否等于完整 Muon 的等效训练？
11. Batch 放大四倍，保持数据量记忆时动量应怎样变？
12. 同一数据预算内，为什么权重衰减次数也会变化？
13. 两个 microbatch 分别正交化后再平均，能否替代全局更新？
14. 信号功率估计非正时，是否说明真实信号为负？
15. 固定更新步数与固定数据量为何可能给出不同结论？
16. 改变 Muon 缩放版本与改变 Batch 时钟，是同一种换算吗？

## 资料与继续阅读

- 苏剑林：[重新思考学习率与 Batch Size（三）：Muon][su-batch]，本文平均场主线的出发点。
- 苏剑林：[Muon 优化器指南][guide]与 [Muon 续集][su-moonlight]，先对齐尺寸与 RMS 约定。
- Keller Jordan：[Muon 原始博客][jordan]，核对更新机制、数值近似与参数分配。
- [Moonlight 技术报告][moonlight]，区分局部分析与固定计算预算的实证比较。
- [Muon 原理与实现](../muon-optimizer/)，核对衰减、QK-Clip 与分布式聚合位置。

[su-batch]: https://kexue.fm/archives/11285
[guide]: https://kexue.fm/archives/11416
[su-moonlight]: https://kexue.fm/archives/10739
[jordan]: https://kellerjordan.github.io/posts/muon/
[moonlight]: https://arxiv.org/html/2502.16982v1
