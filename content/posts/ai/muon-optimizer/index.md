---
title: "Muon 优化器详解：原理、实现与关键细节"
date: 2026-11-03
draft: false
tags: ["Muon", "Machine Learning", "Deep Learning", "Regularization", "Paper Notes"]
categories: ["人工智能"]
authors: ["chase"]
summary: "从谱范数约束推导 Muon，讲清极分解、Newton–Schulz、尺寸缩放、Moonlight 与 QK-Clip；通过矩阵单步复算和可恢复的 PyTorch 训练例子，把参数分组、梯度累积和状态管理连起来。"
description: "从谱范数约束推导 Muon，讲清极分解、Newton–Schulz、尺寸缩放、Moonlight 与 QK-Clip；通过矩阵单步复算和可恢复的 PyTorch 训练例子，把参数分组、梯度累积和状态管理连起来。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
images: ["assets/muon-update-flow.png"]
reading_prerequisites: "梯度下降、矩阵乘法与奇异值分解；标量复算用 Python 标准库，矩阵例子用 NumPy，完整训练例子另需 PyTorch"
reading_focus: "先分清方向、幅度与权重，再理解有限步谱变换和尺寸缩放；用单步矩阵与完整训练两个例子核对更新、分组和恢复，最后理解 QK-Clip。"
related_posts:
  - "/posts/ai/muon-learning-rate-batch-size"
  - "/posts/ai/transformer-attention"
  - "/posts/ai/foundations-modern-ai"
---

**Muon 把矩阵梯度的更新问题变成了矩阵几何问题：先平滑梯度，再改变更新矩阵的奇异值，最后决定步长与权重衰减。** 理解它时，应分开三个对象：梯度与动量、最终更新量、被优化的权重。正交化作用在更新方向上，权重本身仍是自由矩阵。

本文围绕 Keller Jordan 的[原始博客][jordan]、苏剑林的[上手指南][guide]与[Muon 续集][su-moonlight]、[Moonlight 技术报告][moonlight]及 [QK-Clip 文章][qk-blog]展开。重点是推导、数值实现和超参数含义；学习率与 Batch Size 的关系另见[第二篇](../muon-learning-rate-batch-size/)。

| 想弄清的问题 | 阅读入口 |
| --- | --- |
| 为什么不是直接沿负梯度更新 | [谱范数下的最速下降](#geometry) |
| SVD 与实际代码有什么区别 | [Newton–Schulz 迭代](#newton-schulz) |
| 同名 Muon 为什么需要不同学习率 | [四种缩放约定](#scaling) |
| 0.2 与 Weight Decay 从哪里来 | [Moonlight 的改动](#moonlight) |
| 怎么分配参数和检查一次更新 | [参数分组与运行示例](#implementation) |
| 想直接运行一个完整小模型 | [Muon 与 AdamW 联合训练](#toy-training)、[完整下载包](#examples) |
| 想记录梯度并估计噪声尺度 | [固定参数位置的采集例子](../muon-learning-rate-batch-size/#noise-capture) |
| QK-Clip 究竟裁剪什么 | [注意力分数与权重缩放](#qk-clip) |
{.table-readable}

<figure class="article-figure" id="fig-muon-update-flow">
  {{< post-image src="assets/muon-update-flow.png" alt="Muon 更新流程：正确聚合梯度后形成动量，归一化并做 Newton–Schulz 变换，再按尺寸约定缩放和施加学习率；旧权重经过独立衰减，与更新量合成新权重，其他参数使用独立 AdamW 分组" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text"><strong>方向、幅度与衰减分别处理。</strong>图示省略具体动量约定和尺寸系数；聚合发生在非线性矩阵变换之前，权重衰减不经过正交化。</span></figcaption>
</figure>

## 1. 从普通梯度下降走向矩阵更新 {#update}

设线性映射为 $y=Wx$，权重按 PyTorch Linear 的方式存储：

$$
W\in\mathbb R^{d_{\rm out}\times d_{\rm in}},
\qquad G_t=\nabla_W L_t.
$$

用一种常见的非归一化动量记法，Muon 可以写成：

$$
\begin{aligned}
M_t&=\beta M_{t-1}+G_t,\\
H_t&=M_t\quad\text{或}\quad G_t+\beta M_t,\\
O_t&\approx\operatorname{polar}(H_t),\\
W_t&=(1-\eta_t\lambda)W_{t-1}-\eta_t s(W)O_t.
\end{aligned}
$$

第二行分别对应普通动量与一种 Nesterov 形式；$s(W)$ 是后文讨论的尺寸缩放。这里的 Weight Decay 是独立的乘法收缩，不能先把 $\lambda W$ 加入梯度，再一起送入非线性的极分解。

有些实现改用指数移动平均（EMA）：$M_t=\beta M_{t-1}+(1-\beta)G_t$。从零状态开始、固定 $\beta$ 时，两种动量只差固定的正比例；若 Nesterov 也一致换算，理想极分解会消去这个比例。固定 $\varepsilon$、有限精度和动态变更动量时，仍应逐行核对实现，而不是仅对照变量名。

## 2. 为什么是谱范数下的最速下降 {#geometry}

### 2.1 先说清楚用什么约束变化

对一次小更新 $D$，损失的一阶变化是 Frobenius 内积：

$$
L(W+D)-L(W)\approx\langle G,D\rangle
=\operatorname{tr}(G^\top D).
$$

“下降最快”需要附带约束。若规定 $\|D\|_F\le\eta$，最优方向是归一化的负梯度；若规定 $\|D\|_2\le\eta$，控制的就是线性映射的最大放大率：

$$
\|Dx\|_2\le\|D\|_2\|x\|_2.
$$

因此考虑：

$$
\min_{\|D\|_2\le\eta}\langle G,D\rangle.
$$

把非零奇异值对应的紧致奇异值分解（SVD）写为 $G=U_r\Sigma_rV_r^\top$。谱范数与核范数的对偶关系给出：

$$
\langle G,D\rangle\ge-\|G\|_*\|D\|_2
\ge-\eta\sum_{i=1}^r\sigma_i.
$$

选择 $D=-\eta U_rV_r^\top$ 即可取等号。这就是理想 Muon 使用的矩形极分解方向，也常写成 $\operatorname{msign}(G)$。[几何解释][su-moonlight]

这个结论是对当前线性近似的解。把真实梯度换成动量，是引入平滑估计；损失有曲率时，还要选择合适的 $\eta$。它并不单独证明完整训练一定比其他优化器收敛更快。

可以用一个对角矩阵直观看出区别。若两个奇异方向的梯度强度是 10 和 1，则：

$$
\begin{aligned}
G&=\operatorname{diag}(10,1),\\
D_F&=-\frac{\eta}{\sqrt{101}}\operatorname{diag}(10,1),\\
D_2&=-\eta\operatorname{diag}(1,1).
\end{aligned}
$$

$D_F$ 满足 Frobenius 预算，保留了 10:1 的强弱比例；$D_2$ 满足谱范数预算，让两个非零奇异方向使用同样的幅度。两者使用的约束不同，不能只看同一个 $\eta$ 就比较谁的总位移更小。Muon 对弱方向的相对放大，也正是它可能同时放大噪声方向的原因。

### 2.2 零奇异值和矩形矩阵

极分解把每个**非零**奇异值改成 1，保留对应的左右奇异方向；它不等于逐元素取符号。

例如，下面的矩阵所有元素都为正，但它的两个奇异方向分别是坐标的和与差：

$$
\begin{aligned}
G&=\begin{pmatrix}5.5&4.5\\4.5&5.5\end{pmatrix},\\
\operatorname{sign}_{\rm element}(G)
&=\begin{pmatrix}1&1\\1&1\end{pmatrix},\\
\operatorname{polar}(G)&=I_2.
\end{aligned}
$$

这里 $G$ 的奇异值仍是 10 和 1。逐元素取符号会抹掉两列之间的差异，把矩阵变成秩一；极分解则保留两个奇异方向，只改变各自的幅度。理解 Muon 时，应该看矩阵把哪些输入方向映射到哪些输出方向。

若 $G$ 严格秩亏，零方向保持为零。直接对完整 SVD 的全部列相乘，可能把任意补出的零空间也变成更新，因而要使用紧致的非零部分，或正确解释广义逆。

当 $m\ge n$ 且矩阵满列秩时，极因子 $O$ 满足 $O^\top O=I_n$；满行秩时对应 $OO^\top=I_m$。一般矩形矩阵不能同时满足两式，也不能被称为方形正交矩阵。秩亏时得到的是部分等距映射，最优解在零空间上还可能不唯一。

### 2.3 与 SGD、AdamW、Shampoo 的关系

| 方法 | 更新如何处理梯度 | 理解时要保留的区别 |
| --- | --- | --- |
| SGD / Momentum | 直接使用梯度或动量 | 大小与坐标方向共同影响更新 |
| AdamW | 使用逐坐标一、二阶矩 | 自适应统计沿参数坐标建立 |
| Muon | 对矩阵动量做谱变换 | 同时利用行、列之间的结构 |
| Shampoo | 累积矩阵预条件统计 | 与瞬时矩阵归一化有联系，但状态、计算与历史依赖不同 |
{.table-readable}

在精确极分解下，旋转参数的行、列正交基会相应旋转更新方向；逐坐标缩放通常不具有这种不变性。这说明两类方法使用了不同几何结构，不构成“矩阵方法总是更好”的排序。[原始博客中的相关方法讨论][jordan]

<details>
<summary>Muon 与 Shampoo 的瞬时矩阵关系是什么？</summary>

先取满秩方阵 $G=U\Sigma V^\top$。若只用当前梯度构造左右矩阵，并暂时移除阻尼项，双侧四次方根预条件化满足：

$$
\begin{aligned}
P(G)&=(GG^\top)^{-1/4}\,G\,(G^\top G)^{-1/4}\\
&=U\Sigma^{-1/2}\Sigma\Sigma^{-1/2}V^\top\\
&=UV^\top.
\end{aligned}
$$

两个半次逆幂刚好抵消中间的奇异值，于是得到极因子。矩形或秩亏时，可以只在非零谱上取逆幂，零谱分量设为零；对应的是前文的规范部分等距映射。[Jordan 的推导][jordan]

这个恒等式解释了两类矩阵方法的联系。通常的 [Shampoo][shampoo] 保留跨步累积的左右预条件矩阵；加入历史和阻尼后，当前梯度的奇异值不会再按上式逐项抵消。Muon 则先形成动量，再计算有限步谱变换。比较两者时，要同时保留历史统计、动量位置与数值近似的定义。

例如，当前梯度为 $\operatorname{diag}(2,1)$，过去还出现过一次 $\operatorname{diag}(0,3)$。忽略阻尼，左右累积矩阵都为 $\operatorname{diag}(4,10)$，预条件化后的当前方向是 $\operatorname{diag}(1,1/\sqrt{10})$。过去第二方向的强梯度仍在影响本次幅度；当前梯度自己的极因子则为 $I_2$。

</details>

## 3. 实际计算的是有限步 Newton–Schulz {#newton-schulz}

### 3.1 从矩阵多项式变成标量谱映射

反复做 SVD 的代价较高。Muon 使用以矩阵乘法为主的 Newton–Schulz（NS）多项式迭代。先按 Frobenius 范数归一化，再做：

$$
\begin{aligned}
X_0&=H/(\|H\|_F+\varepsilon),\\
A_k&=X_kX_k^\top,\\
X_{k+1}&=aX_k+(bA_k+cA_k^2)X_k.
\end{aligned}
$$

因为 $\|H\|_2\le\|H\|_F$，初始奇异值被压进适合迭代的范围。数学分析常先取 $\varepsilon=0$；实际代码需要单独处理全零输入，并说明是否在范数中加固定 $\varepsilon$。

写 $X_k=U\operatorname{diag}(x_{k,i})V^\top$，则每个非负奇异值遵循同一标量映射：

$$
x_{k+1}=f(x_k),
\qquad f(x)=ax+bx^3+cx^5.
$$

原始博客采用的快速系数是：

$$
(a,b,c)=(3.4445,-4.7750,2.0315).
$$

它着重快速提升较小的奇异值，并容忍输出在 1 附近摆动。**五步结果是近似谱整形，不是一次精确 SVD 的代用品。** 例如 $f(1)=0.701$，已经说明不能把“每一步都把 1 保持在 1”当作这组系数的性质。[系数设计与五步选择][jordan]

<figure class="article-figure" id="fig-muon-spectral-response">
  {{< post-image src="assets/spectral-response.png" alt="同一快速 Newton–Schulz 标量多项式的一步与五步响应，和精确极分解的单位奇异值响应比较；零输入保持零，有限步输出围绕一变化" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text"><strong>有限步迭代保留了幅度误差。</strong>横轴是归一化后的奇异值，曲线由给定系数直接计算；它们不是训练损失或硬件精度测量。</span></figcaption>
</figure>

### 3.2 为什么会先转置再转回来

$X$ 为 $m\times n$ 时，$XX^\top$ 是 $m\times m$。若 $m>n$，可临时转置，使迭代在较小的 Gram 矩阵上进行，最后转回原形状。

这个转置只改变计算方式，不应改变参数的输入、输出维度含义。尤其不能在临时转置后，直接拿工作数组的形状去计算非对称的学习率缩放。

### 3.3 正比例不变性有条件

理想情况下，正数 $c$ 满足 $\operatorname{polar}(cH)=\operatorname{polar}(H)$。因此，更新既可能抑制大奇异值，也可能相对放大很弱的非零方向；它不会根据任务语义识别哪个方向是真信号。

若采用精确范数、$\varepsilon=0$ 且单独处理零矩阵，**归一化之后的有限步 NS 同样具有正比例不变性**；有限步本身带来的是与精确极分解的幅度差异。固定 $\varepsilon$、BF16 的舍入和下溢才会进一步破坏这项不变性。

实际实现还存在两种常见的分母约定：

| 归一化分母 | 非零输入的初始 Frobenius 范数 | 接近零时的行为 |
| --- | --- | --- |
| $\|H\|_F+\varepsilon$ | 范数与“范数加 ε”的比值 | 始终小于 1 |
| $\max(\|H\|_F,\varepsilon)$ | 范数与截断分母的比值 | 范数达到 ε 后等于 1 |
{.table-readable}

本文前面的公式使用第一种写法；后文实际运行的 [PyTorch 2.14.1 源码][torch-muon-source]使用第二种。当输入范数恰好等于 $\varepsilon$ 时，两种初始范数分别为 0.5 和 1，五步输出也可能不同。比较实现前，应把归一化方式与系数、精度一起记录。

此外，裁剪当前梯度并不等于同时缩放全部历史动量，不能据此断言“梯度裁剪对 Muon 永远没有作用”。

### 3.4 快在哪里，额外成本又在哪里？ {#cost}

为估计算术量，把矩阵两边尺寸按大小记为 $p\ge q$。使用较小的 Gram 矩阵时，一步 NS 的主要矩阵乘法约需要：

$$
4pq^2+2q^3
$$

次浮点运算，$T$ 步就乘 $T$。固定五步时，它与常规稠密 SVD 同为 $O(pq^2)$；两边尺寸同比扩大时才表现为三次量级。实际优势来自固定数量的矩阵乘法、低精度执行和硬件上的高效内核，不能只用“复杂度更低”概括。[原博客的运行成本分析][jordan]

若该线性层在一次全局更新中处理了 $N_{\rm vec}$ 个输入向量，前向与反向的主要矩阵乘法约为 $6pqN_{\rm vec}$。因此 NS 相对于这部分计算的额外比例约为：

$$
\frac{T(4pq^2+2q^3)}{6pqN_{\rm vec}}
\le\frac{Tq}{N_{\rm vec}}.
$$

大 Batch 能摊薄这项与输入数量无关的更新成本，小 Batch 时它可能很明显。这里的输入向量数是算术计数；长序列中的 token 可以增加算术工作，却未必增加同样多的独立梯度信息。分布式通信、内核启动、矩阵分块与实际精度也会改变耗时，FLOPs 比例不能直接当作 wall-clock 比例。

长期优化器状态方面，普通 Muon 为所选矩阵保留一份动量；AdamW 通常保留一、二阶矩两份状态。但 NS 还要临时保存更新矩阵和 Gram 矩阵，峰值显存不由长期状态数量单独决定。判断吞吐与内存收益时，应记录整个训练步骤；[第二篇的预算例子](../muon-learning-rate-batch-size/#local-budget)进一步解释了固定开销与 Batch 的关系。

## 4. 四种缩放约定决定学习率的数值 {#scaling}

### 4.1 相同名字可以对应不同更新幅度

沿用 $W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$，把更新写成 $-\eta s(W)O$。苏剑林的指南整理了四种常见约定：[四种版本][guide]

| 约定 | 尺寸因子 $s(W)$ | 尺寸含义 |
| --- | --- | --- |
| 朴素版 | $1$ | 不做额外尺寸缩放 |
| Jordan 约定 | $\sqrt{\max(1,d_{\rm out}/d_{\rm in})}$ | 输出维度小于输入维度时不继续缩小 |
| 指南中的 μP 形式 | $\sqrt{d_{\rm out}/d_{\rm in}}$ | 保留输入、输出维度的非对称性 |
| Moonlight 约定 | $0.2\sqrt{\max(d_{\rm out},d_{\rm in})}$ | 以更新均方根（RMS）对齐为出发点 |
{.table-readable}

这里的“μP 形式”指表中的尺寸因子，不代表仅改变这一个系数，就自动完成了整套 maximal update parametrization；初始化、其他参数组及模型参数化仍需一起定义。

<details>
<summary>为什么会出现输出维度除以输入维度的平方根？</summary>

向量 RMS 是 Euclidean 范数除以元素数量的平方根。因此对固定输入和一次更新 $D$：

$$
\begin{aligned}
\|Dx\|_{\rm RMS}
&\le\sqrt{\frac{d_{\rm in}}{d_{\rm out}}}
\|D\|_2\|x\|_{\rm RMS}.
\end{aligned}
$$

若希望输入 RMS 不超过 1 时，线性层输出的变化不超过 $\eta$，可以约束右侧的算子范数。前面的最速下降方向于是变成：

$$
D=-\eta\sqrt{\frac{d_{\rm out}}{d_{\rm in}}}
\operatorname{polar}(G).
$$

这解释了表中非对称因子的来源：它与输入、输出的 RMS 单位有关；Moonlight 讨论的则是**更新矩阵元素**的 RMS。二者度量的对象不同。完整推导见 Jeremy Bernstein 的 [Deriving Muon][deriving-muon]。

</details>

### 4.2 形状不能脱离线性层的存储方式

PyTorch Linear 执行 $xW^\top$，所以 `W.shape == (d_out, d_in)`；若手写 `x @ W`，则通常存储为 `(d_in, d_out)`。同一个 `shape[0] / shape[1]` 在两种写法中代表相反比例。

例如，输入 1024、输出 4096 时，Jordan 与表中 μP 因子都是 2，Moonlight 因子是 12.8。输入、输出反过来时，前两个分别为 1 和 0.5，Moonlight 仍为 12.8。方阵测试无法暴露把维度写反的问题。

<figure class="article-figure" id="fig-muon-shape-scaling">
  {{< post-image src="assets/shape-scaling.png" alt="四种尺寸约定在扩张矩阵、方阵和压缩矩阵上的缩放因子，以及精确满秩极因子的更新 RMS；Moonlight 的 RMS 为零点二，非对称约定随输入输出维度变化" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text"><strong>先统一约定，再比较学习率。</strong>数值由矩阵尺寸与精确满秩极因子计算，尚未乘外层学习率；有限步 NS 的实际 RMS 可以偏离这些值。</span></figcaption>
</figure>

### 4.3 学习率和衰减要一起换算

如果两种实现使用同一个 $O$，希望它们有相同的更新与收缩，应满足：

$$
\eta_A s_A=\eta_B s_B,
\qquad \eta_A\lambda_A=\eta_B\lambda_B.
$$

所以：

$$
\eta_B=\eta_A\frac{s_A}{s_B},
\qquad
\lambda_B=\lambda_A\frac{s_B}{s_A}.
$$

从 Moonlight 约定换成方阵上的朴素约定，若 $d=4096$，学习率需乘 $0.2\sqrt{4096}=12.8$，衰减系数除以 12.8，才保持这两个乘积不变。指南中“约放大十倍”的经验应放在这个尺寸与版本背景下理解，不能成为所有 Muon 配置的固定规则。

当各层形状不同，$s_A/s_B$ 也可能不同。单个全局学习率只能近似换算；在数学上精确对应，需要按矩阵分组，或在更新内部应用尺寸因子。

## 5. Moonlight 的两个关键补充 {#moonlight}

### 5.1 为什么要有解耦权重衰减

对 $W_t=(1-\eta_t\lambda)W_{t-1}-\eta_t U_t$，若 $0\le\eta_t\lambda\le1$、$\lambda>0$ 且 $\|U_t\|_2\le C$，三角不等式给出：

$$
\|W_t\|_2\le
\max\left(\|W_{t-1}\|_2,\frac C\lambda\right).
$$

这说明有界更新配合衰减，可以控制权重范数。理想无尺寸缩放极因子可取 $C=1$；加上尺寸因子或使用有限步 NS 后，必须相应调整 $C$。这个推导也不能单独给出每个注意力头的精确 logit 上界。[权重衰减分析][su-moonlight]

### 5.2 为什么出现 0.2

若极因子秩为 $r$，则：

$$
\|O\|_F^2=r,
\qquad \operatorname{RMS}(O)
=\sqrt{\frac r{d_{\rm out}d_{\rm in}}}.
$$

满秩时，$r=\min(d_{\rm out},d_{\rm in})$，乘表中的 Moonlight 尺寸因子后，更新 RMS 恰好为 0.2。Moonlight 据其 AdamW 更新统计选择这个量级作为迁移起点。[技术报告第 2.2 节与附录 A][moonlight]

这不是所有 AdamW 训练的恒等式。严格秩亏时，形状缩放后的 RMS 是：

$$
0.2\sqrt{\frac r{\min(d_{\rm out},d_{\rm in})}}.
$$

有限步 NS 也不具有单位奇异值谱。直接把实际更新除以其 RMS，再乘 0.2，才是另一种显式 RMS 归一化，两种操作不能逐输入等同。

<details>
<summary>0.2 与 Adam 的动量降噪有什么联系？</summary>

把一维梯度写成均值为 $\mu$、方差为 $\sigma^2$ 的 iid 序列。Adam 的归一化更新为 $u=m/\sqrt v$。在稳态、忽略偏差修正与 $\varepsilon$，再用 $v$ 的均值近似随机分母时：

$$
\begin{aligned}
\mathbb E[u^2]
&\approx\frac{\mu^2+c\sigma^2}{\mu^2+\sigma^2},\\
c&=\frac{1-\beta_1}{1+\beta_1}.
\end{aligned}
$$

噪声占主导时，取 $\beta_1=0.9$ 得到约 0.2294 的 RMS 量级；信号占主导时，近似值可以趋于 1。这为常见的 0.2 量级提供了一种解释，而没有把它变成普遍常数。

式子没有出现 $\beta_2$，是因为把随机分母替换成了其均值，不能据此证明 $\beta_2$ 对真实 Adam 没有影响。跨步相关、均值漂移与不可忽略的 $\varepsilon$ 也会改变结果。推导与模拟见苏剑林的[Adam Update RMS 分析][adam-rms]。

</details>

### 5.3 报告证明了什么

Moonlight 技术报告把权重衰减与尺寸缩放结合，并实现分布式矩阵更新。在其 compute-optimal 对照中，报告用约 52% 的训练 FLOPs 达到对应性能；这描述的是该实验的计算效率，不是每个优化器步骤都快一倍。[Moonlight 技术报告][moonlight]

同一报告还比较了不同预训练和微调优化器的组合，显示切换优化器并不总带来优势。更新 RMS 对齐提供调参起点，不能消除初始化、谱结构、数据和任务目标的影响。

## 6. 从参数分组到完整训练 {#implementation}

### 6.1 不能只写 `p.ndim == 2`

原始博客主要把 Muon 分配给内部稠密线性层。Embedding、最终输出头、bias 与归一化参数保留在 AdamW 组中。卷积核可以在明确等效线性映射后展平，但展平方式也定义了被优化的矩阵结构。[参数分配建议][jordan]、[指南][guide]

因此，应按模块语义生成参数组，并核对每个参数恰好出现一次。共享或 tied 参数不能因为有两个名称就被更新两次；同一个矩阵怎样拆块，也属于优化器定义的一部分。

### 6.2 PyTorch 接口示例

下面使用 PyTorch 2.14 的接口，只检查合成二次损失的一次更新。`match_rms_adamw` 指对应的尺寸调整约定；不要把名称理解为对每次有限步输出强制测量并归一化 RMS。[版本化接口文档][torch-muon]

```python
import torch

matrix = torch.nn.Parameter(torch.full((4, 8), 0.1))
bias = torch.nn.Parameter(torch.zeros(4))

matrix_optimizer = torch.optim.Muon(
    [matrix], lr=2e-4, weight_decay=0.1,
    adjust_lr_fn="match_rms_adamw",
)
other_optimizer = torch.optim.AdamW(
    [bias], lr=2e-4, weight_decay=0.0,
)

matrix_optimizer.zero_grad(set_to_none=True)
other_optimizer.zero_grad(set_to_none=True)
loss = matrix.square().sum() + (bias - 1).square().sum()
loss.backward()
matrix_optimizer.step()
other_optimizer.step()
```

正式训练还需保存两个优化器的状态，并协调学习率调度、AMP、梯度累积和恢复位置。对丢失梯度与显式零梯度，也应确认实现的策略：`grad is None` 常表示跳过；零梯度仍可让已有动量与权重衰减继续作用。

### 6.3 聚合完成后才做非线性变换

梯度累积与数据并行首先形成同一次全局更新的平均梯度。随后再更新动量并执行 NS。一般来说：

$$
\begin{aligned}
\bar G&=(G_1+G_2)/2,\\
\operatorname{polar}(\bar G)
&\ne\frac{\operatorname{polar}(G_1)+\operatorname{polar}(G_2)}2.
\end{aligned}
$$

张量并行把一个矩阵分成几块后，各块独立正交化通常也不等于原矩阵的 Muon。可以分布式计算整个矩阵所需的运算，或明确定义新的块优化器，但不能仅因最终都调用了 all-reduce 就认为算法相同。

### 6.4 例一：沿状态跟踪矩阵更新 {#matrix-update-example}

[矩阵参考脚本](matrix_update.py)把 EMA、Nesterov、谱变换、尺寸因子和解耦衰减分别返回，便于逐项核对。它使用 NumPy 的 Float64 运算；`polar` 模式用 SVD，`ns` 模式用五步多项式，两者是两个不同的选项。

以对角矩阵为例，设旧权重为 2、3，旧 EMA 动量为 0.4、−0.2，当前梯度为 1、2。取动量系数 0.5、学习率 0.1、衰减系数 0.2：

1. 新 EMA 动量为 0.7、0.9。
2. Nesterov 输入为 0.85、1.45。
3. 精确极分解把两个正的奇异值都变成 1。
4. Moonlight 尺寸因子约为 0.28284；每个对角方向的梯度更新约为 −0.02828。
5. 先将旧权重乘 0.98，再叠加梯度更新，得到约 1.93172、2.91172。

这些值可以直接运行：

```python
import numpy as np
from matrix_update import step

trace = step(
    np.diag([2.0, 3.0]),
    np.diag([0.4, -0.2]),
    np.diag([1.0, 2.0]),
    lr=0.1, decay=0.2, beta=0.5, transform="polar",
)
print(np.diag(trace["buffer"]))
print(np.diag(trace["transform_input"]))
print(np.diag(trace["weight"]))
```

改成 `transform="ns"` 后，第三步不再保证每个奇异值等于 1，最终更新也会变化。脚本的默认演示还会依次输入非零梯度、零梯度与缺失梯度：零梯度可以保留历史动量并继续衰减；缺失梯度则按这里的策略跳过该矩阵，连衰减也跳过。

这里 SVD 的数值秩通过明确的相对阈值选择。严格的数学零方向，与浮点数中一个非常小的非零奇异值，是两个不同问题；脚本不会根据数值大小判断哪个方向是真信号。

### 6.5 例二：两套优化器的训练与恢复 {#toy-training}

[完整 CPU 例子](toy_training.py)使用自动生成的回归数据，不下载模型或数据集。模型包含输入投影、一个隐藏线性层与输出头；本例只把 `hidden.weight` 交给 Muon，其余权重与 bias 交给 AdamW，并检查每个参数恰好属于一个组。

<figure class="article-figure" id="fig-muon-toy-groups">
  {{< post-image src="assets/toy-optimizer-groups.png" alt="随文回归模型的前向依次经过输入线性层、Tanh、隐藏线性层、Tanh 与输出头；只有隐藏层权重交给 Muon，所有 bias 和输入输出权重交给 AdamW，两套优化器状态与模型、采样器及位置一起保存" >}}
  <figcaption><span class="article-figure__number">图 4</span><span class="article-figure__text"><strong>按参数角色分组，再按训练状态恢复。</strong>颜色和文字共同标明本例的分配：隐藏层也含使用 AdamW 的 bias。图中的保存项对应这个没有 AMP、调度器与随机前向算子的模型。</span></figcaption>
</figure>

关键分组代码是：

```python
muon_parameters = [model.hidden.weight]
muon_ids = {id(p) for p in muon_parameters}
adamw_parameters = [
    p for p in model.parameters() if id(p) not in muon_ids
]
```

这段分组适用于随文模型的模块命名。迁移到自己的模型时，需要重新定义哪些模块是内部稠密映射、哪些是输入或输出接口，不能把这里的名称判断直接当作通用分类器。

一次全局更新使用 32 个样本，但 microbatch 为 13，实际分成 13、13、6。每一块的平均损失乘“该块样本数除以 32”后再反向，**三次反向结束才调用一次 Muon 与一次 AdamW**。固定输出维度下，这等于正确的全局平均损失梯度。

程序默认训练 160 次更新，并在中间位置保存模型、两套优化器、采样器状态与更新位置。恢复后继续使用同一采样顺序，检查最终参数是否与连续训练一致；另一个分支只丢掉 Muon 动量，展示“模型权重恢复了”与“训练状态恢复了”的差别。

也可以真的退出一次 Python，再从保存的位置继续。`--steps` 表示累计完成的更新次数，因此第二条命令是从第 80 次继续到第 160 次：

```bash
python -B toy_training.py --save-checkpoint run.pt --steps 80
python -B toy_training.py --resume run.pt --steps 160
```

保存命令拒绝覆盖已有文件；恢复命令保留原来的 Batch 与 microbatch 配置。恢复到第 160 步时输出的参数 SHA-256，应与默认命令连续训练到第 160 步的结果比较；第 80 步检查点的指纹当然不同。这个核对限于同一环境，不承诺不同硬件或 PyTorch 版本之间逐位一致。

这个模型没有 dropout、随机数据增强、AMP 或学习率调度器。若加入这些组件，还需要恢复相应随机数、scaler 与 scheduler 状态；仅保存采样器不能覆盖它们。

```bash
python -B toy_training.py
python -B toy_training.py --batch 32 --microbatch 7
```

输出会列出参数分组、训练与留出集 MSE、最后一步的更新 RMS，以及恢复前后的参数差异。这些数值只验证小例子的接线和状态管理，不用于宣称 Muon 比 AdamW 更好。本文使用的 CPU 环境与运行方法见[数值复算](#examples)。

同一个小模型还可以固定在初始化位置，用于[采集原始梯度并估计噪声](../muon-learning-rate-batch-size/#noise-capture)。这个采集例子不执行训练更新，便于把梯度统计与优化器状态分开核对。

## 7. QK-Clip 为什么直接作用于权重 {#qk-clip}

### 7.1 监控的是 softmax 前的注意力分数

对一个无 bias 的注意力头：

$$
\begin{aligned}
Q&=XW_q^\top,\\
K&=XW_k^\top,\\
S&=QK^\top/\sqrt{d_h}.
\end{aligned}
$$

QK-Clip 监控有效 token 对、样本及必要的并行分片上的最大分数 $S_{\max}$。若 $S_{\max}>\tau>0$，定义 $\gamma=\tau/S_{\max}$，对两侧投影各乘 $\sqrt\gamma$，则在**相同输入、相同权重快照**上，整个双线性分数乘 $\gamma$。

这属于优化器更新后的参数缩放，不是对梯度做 norm clipping，也不是在每次前向中加入 QK-Norm 或 softcap。[QK-Clip 原文][qk-blog]、[Kimi K2 报告第 2.1 节][kimi]

### 7.2 为什么要逐头处理

若两个头的最大分数是 200、50，目标阈值为 100，只应处理第一个头。对全层统一缩放会把第二个头也压到 25，产生额外干预。

多头潜在注意力（Multi-head Latent Attention，MLA）还需区分头独立与共享的部分。若分数由 content 项与 rotary 项相加组成，报告采用：

$$
\begin{aligned}
W_{qc}^{h}&\leftarrow\sqrt{\gamma_h}W_{qc}^{h},\\
W_{kc}^{h}&\leftarrow\sqrt{\gamma_h}W_{kc}^{h},\\
W_{qr}^{h}&\leftarrow\gamma_h W_{qr}^{h},
\end{aligned}
$$

共享的 $W_{kr}$ 保持不变。于是 content 项的两侧缩放乘积是 $\gamma_h$，rotary 项也因 query 一侧的完整缩放而乘 $\gamma_h$。若把 rotary query 也只乘平方根，却不缩放共享 key，两部分就不会同时得到目标比例。

<figure class="article-figure" id="fig-muon-qk-clip">
  {{< post-image src="assets/qk-clip-example.png" alt="构造例比较逐层与逐头 QK 裁剪，并分解 MLA 中内容项一百二十与旋转项八十；正确缩放后合计一百，只缩放旋转 query 平方根时仍超过一百" >}}
  <figcaption><span class="article-figure__number">图 5</span><span class="article-figure__text"><strong>共享项决定缩放该放在哪里。</strong>图中采用构造的无 bias、固定输入双线性分数；数值用于复算缩放，不是报告中的训练轨迹。</span></figcaption>
</figure>

### 7.3 数学恒等式与真实反馈时序

若先记录分数，再更新权重，旧 $S_{\max}$ 与更新后权重产生的分数未必相同。报告算法使用前向期间收集的统计来控制后续权重；这是反馈机制，不能被解释成对任意新输入、每个中间状态都存在一个无条件的硬上界。

例如，旧统计为 200，优化器更新后同一输入的未裁剪最大分数变成 400，仍按旧统计取 $\gamma=1/2$，缩放后是 200 而非 100。精确的同比例结论必须保留“测量与缩放基于同一快照”的条件。

实现时还要确认是否包含 $1/\sqrt{d_h}$、是否排除 mask 位置、如何跨分片归约，以及共享 key 的分组关系。分组查询注意力（GQA）的共享 key 也不能被多个 query 头按不同比例重复缩放。对带 bias 或额外加性分数的结构，缩放权重是否仍让整个 logit 同比变化，需要重新推导。

报告中的 $\tau=100$ 是其训练配置，不是任意架构都应使用的常数；关于更易发生 logit 增长的谱解释，也应保留为作者提出的机制假设，而不是仅由 Muon 定义就能推出的定理。

## 8. 下载、运行与实现核对 {#examples}

[完整下载包](muon-examples.zip)同时包含两篇文章的 Python 例子、依赖文件、参考输出与独立检查；解压后先看 `README.txt`。代码不下载模型或数据集。

只复算标量谱多项式、尺寸因子和 QK-Clip 数字，可以直接下载[标准库脚本](muon_math.py)：

```bash
python3 -B muon_math.py
```

若使用完整下载包，在解压目录建立独立环境。下面以 Python 3.12 为例：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-numpy.txt
.venv/bin/python -B optimizer/matrix_update.py
.venv/bin/python -B checks/test_muon_articles.py --numpy
```

运行完整训练例子时，再安装官方 CPU 版本的 PyTorch：

```bash
.venv/bin/python -m pip install -r requirements-torch.txt
.venv/bin/python -B optimizer/toy_training.py
.venv/bin/python -B checks/test_muon_training.py
```

本文用 CPython 3.12、NumPy 2.2.6 与 PyTorch 2.14.1+cpu 检查了这些命令。在 Windows 下，将 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`。数学脚本本身不要求这套 PyTorch 环境；只用标准库时，可以先运行 `checks/test_muon_articles.py`，需要 NumPy 的组会明确跳过。

可选的[绘图脚本](make_figures.py)需要 NumPy 与 Matplotlib；完整包提供 `requirements-plots.txt`。原始图表以可复算的公式与构造数据生成，不能替代模型训练对照。

核对实现时，最有用的记录包括：参数分组及共享关系、原始矩阵布局、动量递推、NS 系数和步数、归一化与 $\varepsilon$、尺寸缩放约定、衰减应用位置、全局梯度聚合方式，以及 QK-Clip 的测量时刻。

## 阅读自测与验收

1. Muon 正交化的是权重还是更新方向？
2. 谱范数约束与 Frobenius 范数约束为什么产生不同方向？
3. 严格零奇异值是否应被补成单位更新？
4. 矩形满列秩极因子满足哪一个正交关系？
5. 为什么五步快速 NS 不等于精确极分解？
6. 临时转置能否改变输入、输出维度的定义？
7. 四种缩放约定为何对应不同学习率数值？
8. 从 Moonlight 约定换成朴素约定时，为什么衰减也要换算？
9. 0.2 是 AdamW 更新 RMS 的普遍恒等式吗？
10. 严格秩亏与有限步 NS 怎样影响形状 RMS 对齐？
11. 为什么不能仅按二维形状选择全部 Muon 参数？
12. 梯度先聚合再正交化，与先正交化再平均是否相同？
13. QK-Clip 与梯度裁剪、QK-Norm 分别作用在哪里？
14. 为什么 MLA 的 rotary query 使用完整裁剪比例？
15. 用旧分数裁剪新权重，能否保证任何新输入都满足阈值？
16. Moonlight 报告的约两倍计算效率能否写成每步快两倍？
17. 范数恰好等于 ε 时，加 ε 与 clamp 两种归一化是否相同？
18. 固定五步 NS 是否改变了稠密矩阵相对于 SVD 的渐近计算阶数？
19. 动量非零时，全零梯度与缺失梯度是否会产生相同的参数变化？
20. 为什么训练恢复还需要另一套优化器状态与采样位置？
21. 双侧四次方根预条件化在什么条件下等于当前梯度的极因子？

<details>
<summary>展开核对答案</summary>

1. 作用在梯度或动量形成的更新方向上，权重不被约束成正交矩阵。
2. 两个约束使用不同的单位球；谱预算让每个非零奇异方向都能达到同一幅度。
3. 不应。本文采用零空间为零的规范极因子；完整 SVD 任意补出的方向可能造成额外更新。
4. 满列秩时是 $O^\top O=I$；满行秩时相应换成 $OO^\top=I$。
5. 快速多项式改变奇异值，但五步后一般没有把它们精确变成 1。
6. 不能。临时转置只改变计算路径，尺寸因子仍对应原始权重布局。
7. 实际位移取决于 $\eta s(W)$；相同数字不保证相同更新。
8. 既要保持 $\eta s(W)$，也要保持每步乘法收缩中的 $\eta\lambda$。
9. 不是。它是特定统计与配置下的对齐尺度；有均值信号时，Adam 的更新 RMS 也会改变。
10. 秩决定理想极因子的 Frobenius 范数，有限步响应还会改变非零奇异值的幅度。
11. 参数角色与共享关系也重要；输入、输出和 hidden weight 不能仅靠二维形状区分。
12. 一般不同，极分解是非线性变换。
13. QK-Clip 在更新后缩放权重；梯度裁剪改变优化器输入，QK-Norm 改变前向中的 Q/K。
14. 因为共享 rotary key 保持不变，完整比例必须由 query 一侧承担。
15. 不能；同比例恒等式需要相同输入与相同权重快照。
16. 不能。计算预算下达到目标的效率与单步耗时是两个量。
17. 不同。初始归一化范数分别为 $1/2$ 和 1，后续有限步谱响应也可不同。
18. 不改变固定步数下的稠密渐近阶数；实际优势来自矩阵乘法、精度与硬件执行等因素。
19. 本文例子里不同：零梯度仍可通过历史动量移动参数并触发衰减；缺失梯度跳过该矩阵。
20. 下一步还依赖 AdamW 的历史和将读到的数据；只恢复模型与 Muon 动量无法重现这条训练路径。
21. 只取当前梯度构造左右矩阵、移除阻尼，并在非零谱上取逆幂时，奇异值恰好抵消；跨步累积的 Shampoo 仍保留历史。

</details>

## 资料与继续阅读

- Keller Jordan：[Muon 原始博客][jordan]，理解快速 NS、参数分配和竞争性实验依据。
- 苏剑林：[Muon 优化器指南][guide]、[Muon 续集][su-moonlight]，核对尺寸约定、RMS 对齐与衰减。
- [Moonlight 技术报告][moonlight]，保留实验设置、计算预算和优化器组合条件。
- 苏剑林：[QK-Clip][qk-blog]；[Kimi K2 技术报告][kimi]，核对逐头与共享投影的处理。
- [Muon 的学习率与 Batch Size](../muon-learning-rate-batch-size/)，继续分析噪声、曲率、动量与预算。

[jordan]: https://kellerjordan.github.io/posts/muon/
[guide]: https://kexue.fm/archives/11416
[su-moonlight]: https://kexue.fm/archives/10739
[moonlight]: https://arxiv.org/html/2502.16982v1
[qk-blog]: https://kexue.fm/archives/11126
[kimi]: https://arxiv.org/html/2507.20534v1#S2.SS1
[torch-muon]: https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html
[torch-muon-source]: https://github.com/pytorch/pytorch/blob/v2.14.1/torch/optim/_muon.py
[deriving-muon]: https://jeremybernste.in/writing/deriving-muon
[adam-rms]: https://kexue.fm/archives/11267
[shampoo]: https://proceedings.mlr.press/v80/gupta18a.html
