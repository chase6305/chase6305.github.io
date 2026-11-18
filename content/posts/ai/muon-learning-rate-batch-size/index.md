---
title: "Muon 的学习率与 Batch Size：缩放规律、噪声与训练预算"
date: 2026-11-03
lastmod: 2026-11-13
draft: false
tags: ["Muon", "Learning Rate", "Batch Size", "Machine Learning", "Regularization"]
categories: ["人工智能"]
authors: ["chase"]
summary: "逐步推导 Muon 学习率随 Batch Size 增长到饱和的条件，用精确标量积分和随机矩阵复核解释近似误差，再分析有效样本、动量、衰减与训练预算，并提供交互换算和可运行代码。"
description: "逐步推导 Muon 学习率随 Batch Size 增长到饱和的条件，用精确标量积分和随机矩阵复核解释近似误差，再分析有效样本、动量、衰减与训练预算，并提供交互换算和可运行代码。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
images: ["assets/batch-three-clocks.png"]
reading_prerequisites: "梯度、方差、二阶 Taylor 展开与矩阵奇异值分解；建议先了解 Muon 的极分解方向"
reading_focus: "先固定样本计数与学习率约定，再推导局部步长并用标量、矩阵反例核对；最后用数据时钟与交互计算器联动检查动量、衰减和预算。"
related_posts:
  - "/posts/ai/muon-optimizer"
  - "/posts/ai/foundations-modern-ai"
  - "/posts/ai/transformer-attention"
---

**Muon 没有一条脱离条件的“Batch Size 增大多少，学习率就乘多少”的规律。** 在苏剑林《[重新思考学习率与 Batch Size（三）：Muon][su-batch]》采用的平均场与曲率近似下，局部最优步长呈现“较小 Batch 时近似平方根增长、较大 Batch 时趋于饱和”的形式。真正迁移训练配置时，动量、权重衰减、调度与更新次数也同时发生变化。

这篇文章先把这条公式推导清楚，再解释它省略了什么。Muon 的几何与四种尺寸约定见[第一篇](../muon-optimizer/)；本文的 $B$ 表示**一次优化器更新对应的全局 Batch 计数**，不是单卡一次前向的大小。计数何时可以代表独立信息，见[有效样本的条件](#effective-batch)。

第一次阅读先用第 1 节确认 Batch 的计数，再顺着第 2～4 节掌握“对齐收益除以曲率代价”。第 5 节检查这条规律为什么可能失准，第 6～7 节再把它放回有动量、衰减的训练。想迁移具体配置，可以先试[交互换算](#batch-calculator)，但应先决定比较时固定数据量还是更新次数；这会改变后面的换算目标。

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

这些学习率公式针对 Muon 矩阵组。混用 AdamW 时，应分别检查各组的学习率与调度；初始更新 RMS 对齐并不会自动让两种优化器具有相同的 Batch 响应。

### 1.3 计数增加，不一定等于独立信息增加 {#effective-batch}

公式 $B=bDA$ 统计的是参与一次更新的样本数量。后面的 $1/B$ 噪声规律还要求这些梯度来自独立采样，并使用明确的权重。复制同一个样本或同一份梯度，不会产生一份新的独立观测。

这里的独立性取决于抽样机制：有放回的独立抽样允许结果偶然重复；预先决定把一份梯度复制多次，会让这些记录共享同一份噪声。

两个简单模型可以看出差别：

- 对独立、等噪声协方差的单样本梯度，若使用预先给定的非负权重 $w_i$，加权均值对应的等权样本数是“权重和的平方，除以权重平方和”。四个观测的权重为 1、1、1、3 时，这个数是 3，而不是 4。
- 对等权、等方差且任意两份噪声具有同一非负相关系数 $\rho$ 的构造模型，方差对应的等效数量是 $B/[1+(B-1)\rho]$。取 $B=16$、$\rho=0.2$，得到 4；完全复制时 $\rho=1$，得到 1。

<details>
<summary>查看这两个有效数量的方差推导</summary>

若独立噪声都具有协方差 $\Sigma$，固定权重均值的协方差为：

$$
\begin{aligned}
\operatorname{Cov}(\bar g_w)
&=\frac{\sum_iw_i^2}{(\sum_iw_i)^2}\Sigma,\\
B_{{\rm eff},w}&=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
\end{aligned}
$$

第二个模型共有 $B$ 个方差项与 $B(B-1)$ 个协方差项，因此：

$$
\operatorname{Cov}(\bar g)
=\frac{1+(B-1)\rho}{B}\Sigma.
$$

这里的权重公式针对单个等方差观测；含 1 个样本和含 3 个样本的两个 microbatch 均值，噪声方差本来不同，不能直接把它们当成上述两个等方差观测。

</details>

这些“有效 Batch”都只匹配一个噪声方差模型。它们不自动匹配均值、矩阵谱、曲率、数据覆盖或完整训练轨迹；也不应不加检查地替代所有公式中的 $B$。

还有一种常见边界：在固定有限数据集上，只考虑样本选择的随机性，不放回抽样会产生有限总体修正；一次更新覆盖全部数据时，相对于该经验损失的抽样噪声为零。后文的 $1/B$ 模型对应独立抽样，或抽样比例足够小时的近似。

## 2. 一个局部最优步长的通用表达式 {#local-step}

固定同一参数位置 $W$。$L(W)$ 是对目标数据分布取平均的损失，$G=\nabla L(W)$ 是这个平均损失的梯度；它与某次抽样得到的梯度不同。用 $\Phi_B$ 表示 Batch 为 $B$ 时产生的随机更新，实际位移为 $-\eta\Phi_B$。本节先把 $\Phi_B$ 当作已经算好的方向，只沿学习率 $\eta$ 做一维选择。

Hessian 算子 $\mathcal H[X]$ 表示“沿矩阵方向 $X$ 变化时，梯度怎样变化”。先对固定方向展开，再对抽样取期望：

$$
\begin{aligned}
\mathbb E[L(W-\eta\Phi_B)]
&\approx L(W)-\eta a_B+\frac{\eta^2}{2}b_B,\\
a_B&=\mathbb E\langle G,\Phi_B\rangle,\\
b_B&=\mathbb E\langle\Phi_B,\mathcal H[\Phi_B]\rangle.
\end{aligned}
$$

这里 $a_B$ 是沿随机方向更新的平均对齐收益，$b_B$ 是这些方向承担的平均曲率代价。两者比较的是**同一个参数位置反复抽样**会发生什么：更对齐真实梯度可以增加下降收益，但若同时朝向更陡的方向，曲率代价也可能增大。学习率需要在两者之间取平衡。

一次项 $-\eta a_B$ 奖励下降，二次项 $\eta^2b_B/2$ 描述弯曲带来的代价。对 $\eta$ 求导得到 $-a_B+\eta b_B$；当 $a_B>0$、$b_B>0$ 时，令它为零便得到：

$$
\eta_B^*=\frac{a_B}{b_B}.
$$

把矩阵缩成一个数：取 $L(w)=5w^2$、当前位置 $w=0.2$，方向固定为 $\Phi=1$。先求梯度 $L'(w)=10w=2$，再求二阶导数 $L''(w)=10$；因此 $a=2\times1=2$、$b=1\times10\times1=10$，相除得到最优步长 0.2。更新后的位置是 $0.2-\eta$：

| 学习率 | 更新后的 w | 更新后的损失 |
| --- | --- | --- |
| 0 | 0.2 | 0.2 |
| 0.1 | 0.1 | 0.05 |
| 0.2 | 0 | 0 |
| 0.4 | −0.2 | 0.2 |

步长达到最优值的两倍时，下降收益已经消失；继续加大会让损失上升。这个二次函数的二阶展开恰好精确，也说明为什么方向更对齐之后，仍要检查它承担的曲率代价。

这个结果来自当前位置的一步分析。非凸区间、较大步长和更新后变化的曲率，都可能使局部二阶近似失效；完整训练还需要观察沿途统计与评估表现。

### 2.1 SGD 为什么常见线性增长到饱和

把矩阵按固定顺序展平成向量后，用 $\Sigma$ 表示单样本梯度噪声的协方差，$H_{\rm vec}$ 表示同一排列下的 Hessian。独立平均使噪声协方差变成 $\Sigma/B$。

对 SGD，将 $\Phi_B=\widehat G_B=G+E_B$ 代回前式。因为 $\mathbb E[E_B]=0$，对齐项中的噪声消失；二次型展开中的交叉项也消失，但噪声平方项仍保留：

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

暂时假定不同矩阵元素的噪声不相关、每个元素的方差均为 $\sigma^2/B$。这里的 $\sigma^2$ 是单样本、单坐标方差，$\sigma_i$ 则留给第 $i$ 个奇异值，二者含义不同。

展开 Gram 矩阵时，$G^\top E_B$ 与 $E_B^\top G$ 的期望都为零。剩下的 $E_B^\top E_B$ 是 $n\times n$ 矩阵：每个对角元素累加 $m$ 个噪声平方，期望为 $m\sigma^2/B$；非对角元素累加不相关坐标的乘积，期望为零。因此：

$$
\mathbb E[\widehat G_B^\top\widehat G_B]
=G^\top G+\frac{m\sigma^2}{B}I_n.
$$

总噪声功率还要再对全部 $mn$ 个坐标求和，得到 $v=mn\sigma^2$。前式的 $m$ 来自一次 Gram 对角求和，$v$ 中的 $mn$ 来自整个梯度的平方范数；这两种计数不能互换。

### 3.2 第一次近似：把非线性期望拆开

理想 Muon 取 $\Phi_B=\operatorname{polar}(\widehat G_B)$。先对每个随机梯度求极因子，再问这些方向的平均是什么；不能先把梯度平均成 $G$，再假定更新平均也等于 $\operatorname{polar}(G)$。平均场近似写成：

$$
\begin{aligned}
\mathbb E[\Phi_B]
&=\mathbb E\!\left[\widehat G_B
(\widehat G_B^\top\widehat G_B)^{-1/2}\right]\\
&\approx G\left(G^\top G+
\frac{m\sigma^2}{B}I_n\right)^{-1/2}.
\end{aligned}
$$

近似号同时做了两件事：把随机逆平方根中的 Gram 矩阵换成其期望，再把左侧随机梯度换成 $G$。原来二者相关，这些替换并非期望恒等式；下一节的谱平均是在此之后才做的另一层近似。[平均场推导][su-batch]

若 $G=U\operatorname{diag}(\sigma_i)V^\top$，上式在第 $i$ 个奇异方向上的幅度为下面的 $c_i(B)$。其中带下标的 $\sigma_i$ 表示真实梯度的奇异值，噪声的单坐标方差仍用 $\sigma^2$ 表示：

$$
c_i(B)=\frac{\sigma_i}{\sqrt{\sigma_i^2+m\sigma^2/B}}.
$$

即便接受第一次近似，不同奇异方向也有不同的噪声与信号比例。$c_i(B)$ 是**平均更新在该方向上的系数**，不是对每次随机梯度直接再乘的实现参数。弱真实方向更容易被随机扰动偏转，许多次单位方向相互抵消后，平均系数仍可很小。

### 3.3 第二次近似：用平均能量替代整个谱

先把上一步的矩阵表达式改写成：

$$
\begin{aligned}
G(G^\top G+\delta I)^{-1/2}
&=\operatorname{polar}(G)\,R_\delta,\\
R_\delta&=\left[I+\delta(G^\top G)^{-1}\right]^{-1/2},\\
\delta&=m\sigma^2/B.
\end{aligned}
$$

这样分成了两个对象：$\operatorname{polar}(G)$ 保留真实梯度的奇异方向，$R_\delta$ 描述噪声对这些方向的衰减。接下来**只在衰减因子中**使用：

$$
G^\top G\approx\frac{\|G\|_F^2}{n}I_n,
$$

把每个奇异值平方用平均能量替代，衰减因子才变成同一个标量，得到：

$$
\begin{aligned}
\mathbb E[\Phi_B]&\approx
\frac{\operatorname{polar}(G)}{\sqrt{1+B_c/B}},\\
B_c&=\frac{mn\sigma^2}{\|G\|_F^2}
=\frac v{\|G\|_F^2}.
\end{aligned}
$$

分母必须是梯度 Frobenius 范数的**平方**：分子与分母都是梯度平方量，比例才具有 Batch 的含义。

第二次近似把“每个奇异方向各有一个衰减系数”压成“整个极因子共用一个系数”，因此后面只剩单个 $B_c$。它保留了大 Batch 极限：当 $B$ 趋于无穷，$R_\delta$ 趋于单位矩阵；代价是丢失谱内部的差异。它既没有证明原奇异值相等，也没有消除第一次期望近似的误差。

### 3.4 曲率条件同样重要

前两次近似给出了平均方向，尚未算出步长分母。苏剑林的推导进一步令曲率作用为：

$$
\mathcal H[X]=\eta_\infty^{-1}X(G^\top G)^{1/2}.
$$

记 $C=(G^\top G)^{1/2}$，它是当前固定梯度对应的 $n\times n$ 矩阵。若随机梯度几乎必然满列秩，每次极因子都满足 $\Phi_B^\top\Phi_B=I_n$，于是曲率代价可以逐次化简：

$$
\begin{aligned}
\langle\Phi_B,\mathcal H[\Phi_B]\rangle
&=\eta_\infty^{-1}\operatorname{tr}
(\Phi_B^\top\Phi_B C)\\
&=\eta_\infty^{-1}\operatorname{tr}(C).
\end{aligned}
$$

右侧不再含随机方向，取期望后也不随 $B$ 变化。另一方面，$G^\top\operatorname{polar}(G)=C$；将第 3.3 节的平均方向代入 $a_B$，分子中出现同一个迹：

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

分子、分母共同的 $\operatorname{tr}(C)$ 抵消，代入 $a_B/b_B$ 就得到：

$$
\boxed{\eta_B^*\approx
\frac{\eta_\infty}{\sqrt{1+B_c/B}}}.
$$

这条公式将独立同方差噪声、两次平均场近似与上述曲率模型合在了一起。仅有 $1/B$ 噪声方差，不足以确定 $a_B/b_B$ 的形状；严格秩亏时也不能继续用满列秩等距关系化简分母。第 4.1 节单独检查期望近似，第 5 节再分别放开谱与曲率条件。

这里 $\eta_\infty$ 是**当前参数位置、当前矩阵与当前曲率模型**对应的大 Batch 步长极限。它不是任意优化器的已有峰值学习率，也不必在整个训练过程中保持不变。

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
  {{< post-image src="assets/batch-scaling-law.png" alt="左图比较条件成立时的 Muon、SGD 学习率曲线与平方根外推；右图在一维高斯梯度模型中比较可精确积分的符号期望与平均场近似，两者渐近形状相同但数值不同" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text"><strong>缩放规律有区间，平均场也有误差。</strong>左图的横、纵轴分别按各方法自己的临界尺度与大 Batch 学习率归一化，用来比较局部公式的形状。右图的标量高斯例子在下一小节展开；这些曲线不表示训练效果排名。</span></figcaption>
</figure>

### 4.1 一个能精确计算的近似误差 {#gaussian-sign-check}

取 $1\times1$ 的矩阵，即正的标量真实梯度 $g$，单样本噪声为标准差 $s$ 的独立高斯变量。此时极分解就是取符号，Batch 平均后的更新方向是：

$$
\Phi_B=\operatorname{sign}(g+sZ/\sqrt B).
$$

它沿正确方向的期望可以直接积分，得到：

$$
\mathbb E[\Phi_B]
=\operatorname{erf}\!\left(\sqrt{\frac{B}{2B_c}}\right),
\qquad B_c=s^2/g^2.
$$

因为 $g>0$，采样梯度为正时的极因子为 $+1$，为负时为 $-1$。二者分别乘发生概率再相加，就是“正梯度概率减负梯度概率”；连续高斯变量恰好为零的概率为零。对高斯密度积分，才得到上面的 erf 式。它描述多次随机方向的平均，不是单次极因子的幅度。

沿用前面的特殊曲率时，最优学习率除以 $\eta_\infty$，恰好就是这个期望。取 $B=B_c$，精确值约为 0.68269，平均场公式给出约 0.70711。两条曲线都有小 Batch 平方根与大 Batch 饱和的渐近形状，但中间数值不同。

这个一维边界例只用来检验“期望与非线性变换不能随意交换”；不能把它的误差大小推广成大矩阵 Muon 的误差估计。

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

还可以直接检验这一步近似：给两个真实梯度的每个元素分别加入独立高斯噪声，Batch 梯度的每坐标噪声方差取 0.05。这样噪声对 Gram 期望的贡献正好是 $0.1I$。每个矩阵独立抽样 20 万次，逐次用 Float64 SVD 求极因子再平均，得到平坦谱两个方向约 0.987；不均匀谱的强、弱方向则约为 0.978、0.336。

<figure class="article-figure" id="fig-muon-anisotropy">
  {{< post-image src="assets/gram-anisotropy.png" alt="平坦谱和不均匀谱拥有相同梯度能量，但 Gram 近似与高斯矩阵 SVD 采样均值得到不同的方向衰减；不均匀谱的弱方向均值约零点三三六，图中误差条是蒙特卡洛近似区间" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text"><strong>既要保留谱，也要检查期望近似。</strong>实心柱来自 Gram 公式，斜线柱来自每组 20 万次独立高斯矩阵的 SVD 均值；误差条为均值正负 1.96 个标准误的近似区间。这里固定参数位置、不做优化器更新。</span></figcaption>
</figure>

小奇异值会让极分解方向对扰动更敏感，单个 $B_c$ 无法保留这种差异。

在前文特殊 Hessian 条件下，把这些采样更新投影到真实梯度上，还可估计精确局部最优步长与大 Batch 极限的比例：平坦谱约为 0.9873，不均匀谱约为 0.9354；标量平均场却对两者都给出约 0.9535。差别来自近似过程，不能由一个总噪声功率比消除。[采样脚本](mean_field_probe.py)与[完整结果](assets/mean-field-probe-reference.json)都可复算。

为什么弱方向均值约为 0.336，局部步长比例仍能达到 0.9354？第 2 节的对齐收益按真实梯度加权：强方向贡献更多，不能将两个方向的系数等权平均来求步长。既要看方向是否稳定，也要看它对下降与曲率分别贡献多少。

在已安装 NumPy 的环境运行 `python3 -B mean_field_probe.py`，就会输出这两组结果、各方向的标准误和相应步长比例。脚本默认使用固定种子与每组 20 万次采样，计算的是同一参数位置上的期望。

### 5.2 一般曲率下，最优学习率未必单调增加

考虑一个 $2\times1$ 矩阵，真实梯度 $G=(1,0)^\top$，Hessian 为 $\operatorname{diag}(100,1)$。每个样本只在第二个坐标加入均值为零、标准差为 $s$ 的独立高斯梯度噪声，图中取 $s=100$。

平均 $B$ 个样本后，随机梯度可写成：

$$
\widehat G_B=(1,sZ/\sqrt B)^\top,
\quad Z\sim\mathcal N(0,1).
$$

极分解就是单位化该列向量。记两个分量为 $D_1,D_2$，它们满足 $D_1^2+D_2^2=1$。真实梯度只有第一个坐标非零，所以对齐收益为 $a_B=\mathbb E[D_1]$；曲率代价为 $b_B=\mathbb E[100D_1^2+D_2^2]=1+99\mathbb E[D_1^2]$。代回第 2 节的比值，局部二次损失下的**精确**最优步长为：

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

继续固定参数位置，使均值梯度恒定、各步噪声独立，并取 $0\le\beta<1$。指数移动平均（EMA）为：

$$
M_t=(1-\beta)\widehat G_t+\beta M_{t-1}
$$

把递推反复展开，当前梯度的权重为 $1-\beta$，前一步为 $(1-\beta)\beta$，再前一步为 $(1-\beta)\beta^2$。稳态下这些权重之和为 1，所以均值不变；独立噪声的方差则按**权重平方**累加：

$$
\sum_{j=0}^{\infty}(1-\beta)^2\beta^{2j}
=\frac{(1-\beta)^2}{1-\beta^2}
=\frac{1-\beta}{1+\beta}.
$$

乘原来的 Batch 均值噪声协方差，就得到：

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

这个换算只匹配旧状态的累计保留。取 $\beta=0.5$、初始动量为零：先收到梯度 1 得到 0.5，再收到 3，得到 $0.5(0.5)+0.5(3)=1.75$。合并 Batch 后只收到均值 2，改用 $\beta_{\rm new}=0.25$，得到 $0.25(0)+0.75(2)=1.5$。两种做法都将旧状态乘 0.25，但前者更重视后来到达的梯度；合并为均值后，这个顺序已丢失。

<details>
<summary>如果目标是保持 EMA 的稳态噪声方差，公式会怎样？</summary>

沿用固定参数位置、独立平稳噪声的条件，记 $c(\beta)=(1-\beta)/(1+\beta)$。Batch 乘 $k$ 后，保持 EMA 协方差要求：

$$
\begin{aligned}
c(\beta_{\rm new})&=k\,c(\beta_{\rm old}),\\
\beta_{\rm new}
&=\frac{1-kc(\beta_{\rm old})}{1+kc(\beta_{\rm old})}.
\end{aligned}
$$

例如 $\beta_{\rm old}=0.95$、$k=4$，得到 $35/43\approx0.813953$；保持旧状态数据量记忆的结果则约为 0.814506。两者很接近，但保持的量仍然不同。

当 $kc(\beta_{\rm old})>1$ 时，上式需要负的动量系数，因此无法在通常的 $0\le\beta<1$ 范围内匹配原方差。无动量的大 Batch 均值的噪声方差已小于目标值，继续加正 EMA 只会进一步降噪。这里匹配的是普通 EMA；Nesterov 需要按上一节的权重平方和重算。这个结论仍限于稳态线性滤波，不能直接当作完整 Muon 的等效训练规则。

</details>

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

这里 $V$ 描述一份 $b$ 样本平均梯度的波动；在独立等权采样假设下，乘回 $b$ 才得到单样本总噪声功率估计 $bV$。再用：

$$
\widehat S=\|\bar g\|_F^2-\frac VK
$$

校正信号功率。原因是有限次平均 $\bar g$ 仍含噪声：$K$ 份独立记录均值的总噪声功率等于单份记录的 $1/K$，直接取 $\|\bar g\|_F^2$ 会把这部分也当成信号。减去 $V/K$ 后，才形成 $\widehat S$。

这里用同一个 $V$ 做两件不同的换算：乘 $b$ 还原单样本噪声，除 $K$ 校正记录均值。只有 $\widehat S>0$ 时才报告比值 $bV/\widehat S$；分子、分母的无偏性不保证这个随机比值也无偏。

若校正后信号非正，应该报告估计不确定或样本不足；不能把负值直接解释为“真实信号为负”。相同样本的复制、相关序列与参数不断变化的梯度，也不能冒充独立同分布记录。

可以先用四份构造记录复算这些量。每份记录代表 8 个等权样本的平均梯度；下面的代码在 [batch_math.py](batch_math.py) 所在目录运行，只需标准库：

```python
from batch_math import noise_estimate

gradients = [[3, 1], [1, 3], [3, 3], [1, 1]]
print(noise_estimate(gradients, microbatch=8))
```

按“记录均值 → 记录方差 → 两次尺度换算”的顺序手算：均值为 $(2,2)$；四份偏差依次为 $(1,-1)$、$(-1,1)$、$(1,1)$、$(-1,-1)$，每份平方范数都是 2，总计 8。除以 $K-1=3$，得 $V=8/3$。均值平方范数为 8，减去 $V/K=2/3$，得校正信号 $22/3$；噪声则乘 $b=8$，得 $64/3$，相除为 $32/11\approx2.9091$。这是四份构造记录的结果，不能由样本方差偶然为零反推总体无噪声。

#### 9.1.1 固定参数采集原始梯度 {#noise-capture}

<details>
<summary>从随文模型采集梯度、保存记录并离线复算</summary>

[梯度采集脚本](../muon-optimizer/gradient_noise.py)复用第一篇的小型回归模型，固定在初始化位置，只记录 `hidden.weight` 的原始梯度。这里的“固定”表示不更新参数；参数仍需开启自动求导。

在完整下载包的解压目录、安装过 CPU PyTorch 的环境中运行：

```bash
.venv/bin/python -B optimizer/gradient_noise.py --save-records gradients.json
python3 -I -S -B optimizer/gradient_noise.py --replay gradients.json
```

第一条采集 $K=64$ 份记录，每份包含 $b=16$ 次独立、有放回的样本抽取，并保存参数形状、抽样索引和原始梯度。第二条只用标准库复算，不重新采样，也不需要 PyTorch 或 NumPy。保存拒绝覆盖已有文件，回放保留原记录。

脚本使用 [torch.autograd.grad][autograd-grad] 返回所选矩阵的梯度，不向已有 `.grad` 累积，也不执行优化器更新。它在采集前后核对全部模型参数的指纹。随文模型没有 dropout 或运行统计；扩展到其他模型时，还需固定会影响前向的状态，并明确损失与采样目标。

默认 CPU 运行中，校正信号功率约为 0.08111，单样本总噪声功率约为 0.47031，比值约为 5.7982。[参考输出](../muon-optimizer/assets/gradient-noise-reference.json)记录了环境与完整字段。这只是**当前矩阵、当前参数位置**的统计估计；谱各向异性、曲率与动量仍需分别分析。

这里的 $K$ 是用于估计统计量的重复观测次数，$b$ 是每份记录的样本数；两者与训练时一次更新的全局 $B$ 分别定义。采集的是动量和 NS 之前的梯度，Float32 模型求导后再做统计，也不会把前向和反向变成 Float64 计算。

固定种子用于复算同一组记录；重放或复制同一文件不会增加独立观测数量。需要更多记录时，应实际增加采样次数，并继续保持同一个统计目标。

</details>

### 9.2 固定比较的分母

| 比较条件 | 保持了什么 | 仍可能改变什么 |
| --- | --- | --- |
| 固定 optimizer steps | 更新次数 | 总样本数与计算量 |
| 固定 samples / tokens | 数据预算 | 更新次数、动量记忆、累计衰减 |
| 固定训练 FLOPs | 主要算术预算 | 数据量、模型大小、通信占比 |
| 固定 wall-clock | 实际耗时 | 硬件利用率、通信与样本数 |
{.table-readable}

局部公式回答的是当前位置的一次更新，不会直接决定这些完整实验中哪个配置最好。 warmup、峰值学习率与衰减结束位置应同时记录其计数单位；若目标是固定数据预算，通常需要把调度位置也换算到 samples 或 tokens。

### 9.3 学习率更大，为什么不一定更省预算？ {#local-budget}

把局部二次目标对 $\eta$ 配方，在 $a_B,b_B>0$ 时，最优一次更新带来的近似损失减少量是：

$$
\Delta L_B^*=\frac{a_B^2}{2b_B}.
$$

在前面的 Muon 平均场与特殊曲率条件下，$b_B$ 不随 Batch 变化，于是归一化后得到：

$$
\frac{\Delta L_B^*}{\Delta L_\infty^*}
\approx\frac{B}{B+B_c}.
$$

这解释了为什么更大的 Batch 可以让**一次更新**走得更有效，但不能直接说明它更省样本。若把改善量除以所用样本数，局部收益率正比于 $1/(B+B_c)$，反而随 Batch 增大而下降。

再考虑一个构造的成本模型：每步成本为 $C(B)=c_0+c_1B$，其中 $c_0$ 表示固定开销，$c_1$ 表示每个样本的线性成本。每单位成本的局部收益率便正比于：

$$
\frac{B}{(B+B_c)(c_0+c_1B)}.
$$

当 $c_0,c_1,B_c>0$ 时，收益率为正。对其对数求导，比直接对分式求导更容易看清三个来源：

$$
\frac1B-\frac1{B+B_c}
-\frac{c_1}{c_0+c_1B}=0.
$$

通分后，分子是 $B_c c_0-c_1B^2$，因此峰值在 $B^*=\sqrt{B_c c_0/c_1}$。例如 $B_c=4096$，固定开销等价于 256 个样本成本，即 $c_0/c_1=256$，开方得到 1024。继续加到 4096，单步改善仍提高，单位成本收益却下降。

<figure class="article-figure" id="fig-muon-batch-budget">
  {{< post-image src="assets/batch-budget-efficiency.png" alt="固定局部统计与每步成本 B 加二百五十六的构造模型：单步改善随 Batch 增加而饱和，单位样本改善下降，单位成本改善在 Batch 一千零二十四附近达到最大" >}}
  <figcaption><span class="article-figure__number">图 5</span><span class="article-figure__text"><strong>最优步长与最优 Batch 回答不同问题。</strong>曲线冻结了梯度和曲率统计，成本单位也由构造模型定义；图中最优点不是实际训练或硬件测量结果。</span></figcaption>
</figure>

这是固定参数位置的局部收益率分析，与[数据效率和时间效率的权衡][noise-scale]相呼应；沿训练路径改变的统计与真实硬件成本仍需另测。

### 9.4 动手比较三种换算 {#batch-calculator}

下面同时展示条件平均场学习率、按样本量保留的动量，以及两种不同目标下的权重衰减。先看默认的四倍 Batch，再切到接近饱和的区间，观察学习率倍率为什么趋近 1。

两个衰减输出是**二选一的实验约定**：保持每步收缩，适合核对单次更新；保持同数据预算下的累计收缩，则要补偿更新次数变少。不要把它们依次应用。换算完成后，回到第 9.2 节选择同一个预算分母，再比较训练结果。

{{< muon-batch >}}

## 10. 一个有条件的迁移流程 {#procedure}

1. 固定参数分组、Muon 尺寸缩放、损失归约和 NS 数值设置，先确认实际全局 Batch。
2. 区分只改变 microbatch 的工程调整，还是确实改变了有效独立样本数量。
3. 使用旧配置和相近参数位置的统计，判断是否有理由把平方根规则当作局部初值，而不是直接倍增峰值学习率。
4. 明确动量保持更新步记忆还是数据量记忆；明确衰减保持每步强度还是累计收缩。
5. 按同一数据预算与评估协议做小范围对照，记录实际更新 RMS、相对更新量、损失、异常值与尾延迟。
6. 只有在上述配置稳定后，再评估吞吐和分布式实现；额外的 QK-Clip 等反馈机制也应单独记录触发统计。

[随文脚本](batch_math.py)复算平均场比例、动量与衰减换算、噪声估计和二次目标反例。[两篇文章的完整下载包](../muon-optimizer/muon-examples.zip)还包含依赖与检查；本篇文件位于解压目录的 `batch/`。

```bash
python3 -B batch_math.py
```

它使用构造数据和数值积分，不加载模型。图表可用[绘图脚本](make_figures.py)重画；绘图另需 NumPy 与 Matplotlib。将它与 `batch_math.py`、[随机矩阵复核脚本](mean_field_probe.py)保存在同一目录，运行 `python3 -B make_figures.py`，四幅数值图会写入旁边的 `assets/`。完整下载包已经包含这些文件。正交化的有限步、矩阵布局和更新幅度，可回到[Muon 原理与实现](../muon-optimizer/)一起核对。

## 阅读自测与验收

先独立作答或操作，再核对本节答案；实践结果以实际记录为准。

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
17. 复制样本或使用不等权重时，计数是否仍等于方差对应的有效样本数？
18. 一维高斯例子在 B=Bc 处的精确比例与平均场比例分别是多少？
19. 保持旧动量的衰减比例，是否同时保持新梯度的时间权重？
20. 随机矩阵图中的误差条能否约束真实训练或平均场公式的误差？
21. 局部成本模型中的最优 Batch 为什么还依赖固定开销？
22. 计算器的两个衰减输出，是否应该同时应用？
23. 保持 EMA 稳态方差与保持旧状态的数据量记忆，是否得到同一个动量公式？
24. 采集 K 份梯度记录时，为什么要固定参数位置，并区分 K、每份样本数 b 与训练的全局 Batch？

<details>
<summary>展开核对答案</summary>

1. 通常不应。先检查全局归约、随机算子及跨样本运算是否也保持一致。
2. 各组包含的有效样本或权重不同，均值需要按对应总权重汇总。
3. 分子是方向与真实梯度的平均对齐，分母是方向上的平均曲率代价。
4. 先把极分解的非线性期望拆开，再把谱相关的噪声因子压缩成标量。
5. 总噪声功率包含矩阵元素数量；本篇 $B_c=v/\|G\|_F^2$ 使用总功率。
6. 分子、分母都必须是梯度平方量，才能形成无量纲的样本数量尺度。
7. 在所述近似下，$B\ll B_c$ 时近似平方根增长，$B\gg B_c$ 时趋于饱和。
8. 不保证。各奇异方向的信号强度与噪声敏感性仍然可以不同。
9. 可以。Batch 同时改变对齐与曲率，两者之比未必单调。
10. 不等于。EMA 公式只在固定参数位置与独立平稳噪声等条件下匹配方差。
11. 用 $\beta_{\rm new}=\beta_{\rm old}^4$ 匹配旧状态经过相同数据量后的保留比例。
12. 相同数据量对应的更新次数减少，因此同一个每步衰减会带来更弱的累计收缩。
13. 不能，非线性矩阵变换应在正确聚合之后进行。
14. 不是。有限样本校正可能得到非正估计，这提示统计分辨率不足。
15. 固定步数会让大 Batch 读取更多数据；固定数据量则让它执行更少更新。
16. 不同。前者匹配同一步参数位移，后者改变统计与时间离散化。
17. 不一定。独立性、权重和相关性共同决定均值的方差。
18. 精确值约为 0.68269，平均场值约为 0.70711；极限形状相似不代表有限处严格相等。
19. 不保持。合并后只看到一份均值，已经丢失内部梯度到达的次序。
20. 不能。它们仅描述固定构造分布中有限次 SVD 采样均值的不确定性。
21. 更大 Batch 降低每步固定开销的摊销，但每次更新的收益又会饱和；在本文模型中两者给出平方根形式的成本最优点。
22. 不应。它们分别保持每步收缩与同数据预算累计收缩，对应两个实验目标。
23. 一般不同。前者匹配“滤波方差因子除以 Batch”，后者匹配旧状态经过相同数据量后的保留比例；方差匹配还有非负动量的可行条件。
24. 这样均值与噪声指向同一统计目标；K 控制重复观测数量，b 决定每份均值的噪声，训练的全局 Batch 则定义优化器更新。参数不断变化时，记录间差异还混入了训练漂移。

</details>

**通过条件：** 在同一梯度、动量、步长与 decay 约定下核对 Batch 计算器及更新；能解释学习率规则的假设，不用单步不变量宣称完整训练轨迹等价。


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
[noise-scale]: https://arxiv.org/html/1812.06162
[autograd-grad]: https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html
