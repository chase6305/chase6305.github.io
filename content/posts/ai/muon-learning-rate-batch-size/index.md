---
title: "Muon 的学习率与 Batch Size：缩放规律、噪声与训练预算"
date: 2026-11-03
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

第一次阅读可先顺着第 2～4 节掌握“对齐收益除以曲率代价”这条主线，再读第 6～7 节的动量与衰减。想比较具体配置，可以先试[交互换算](#batch-calculator)，随后对照每个输出所保持的量。

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

一个能精确算完的例子是 $L(w)=5w^2$。在 $w=0.2$ 处，取固定更新方向 $\Phi=1$，便有 $a=2$、$b=10$，最优步长为 0.2：

| 学习率 | 更新后的 w | 更新后的损失 |
| --- | --- | --- |
| 0 | 0.2 | 0.2 |
| 0.1 | 0.1 | 0.05 |
| 0.2 | 0 | 0 |
| 0.4 | −0.2 | 0.2 |

步长达到最优值的两倍时，下降收益已经消失；继续加大会让损失上升。这个二次函数的二阶展开恰好精确，也说明为什么方向更对齐之后，仍要检查它承担的曲率代价。

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

若 $G=U\operatorname{diag}(\sigma_i)V^\top$，上式在第 $i$ 个奇异方向上的幅度为下面的 $c_i(B)$。其中带下标的 $\sigma_i$ 表示真实梯度的奇异值，噪声的单坐标方差仍用 $\sigma^2$ 表示：

$$
c_i(B)=\frac{\sigma_i}{\sqrt{\sigma_i^2+m\sigma^2/B}}.
$$

即便接受第一次近似，不同奇异方向也有不同的噪声与信号比例。

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

这一步保留了大 Batch 极限：当 $B$ 趋于无穷，$R_\delta$ 趋于单位矩阵。它并没有证明原矩阵的奇异值本来就相等，也没有让第一次非线性期望近似变成精确结果。

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

因此，平均场曲线适合组织直觉，却不能代替对弱奇异方向、噪声方向与真实更新分布的检查。小奇异值还会让极分解方向对扰动更敏感。

在前文特殊 Hessian 条件下，把这些采样更新投影到真实梯度上，还可估计精确局部最优步长与大 Batch 极限的比例：平坦谱约为 0.9873，不均匀谱约为 0.9354；标量平均场却对两者都给出约 0.9535。差别来自近似过程，不能由一个总噪声功率比消除。[采样脚本](mean_field_probe.py)与[完整结果](assets/mean-field-probe-reference.json)都可复算。

在已安装 NumPy 的环境运行 `python3 -B mean_field_probe.py`，就会输出这两组结果、各方向的标准误和相应步长比例。脚本默认使用固定种子与每组 20 万次采样，计算的是同一参数位置上的期望。

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

这个换算保持的是**旧动量状态的累计衰减**，不保证新梯度进入动量时的权重也一样。取 $\beta=0.5$、初始动量为零，先后输入梯度 1 和 3，两次 EMA 后得到 1.75；把两份梯度平均为 2、使用 $\beta_{\rm new}=0.25$，一次更新得到 1.5。旧状态保留比例都为 0.25，新信息的时间权重却不同。

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

当两个成本系数与 $B_c$ 都为正时，对这个表达式求极值得到 $B^*=\sqrt{B_c c_0/c_1}$。例如取 $B_c=4096$，固定开销等价于 256 个样本的成本，最优点在 1024；继续把 Batch 加到 4096，单步改善仍提高，单位成本收益却下降。

<figure class="article-figure" id="fig-muon-batch-budget">
  {{< post-image src="assets/batch-budget-efficiency.png" alt="固定局部统计与每步成本 B 加二百五十六的构造模型：单步改善随 Batch 增加而饱和，单位样本改善下降，单位成本改善在 Batch 一千零二十四附近达到最大" >}}
  <figcaption><span class="article-figure__number">图 5</span><span class="article-figure__text"><strong>最优步长与最优 Batch 回答不同问题。</strong>曲线冻结了梯度和曲率统计，成本单位也由构造模型定义；图中最优点不是实际训练或硬件测量结果。</span></figcaption>
</figure>

这个计算只比较同一参数位置的局部收益率，没有把参数沿训练路径的变化积分进去。它与早期大 Batch 文献讨论的[数据效率和时间效率权衡][noise-scale]相呼应，但具体峰值不能迁移到真实硬件或完整训练。

### 9.4 动手比较三种换算 {#batch-calculator}

下面同时展示条件平均场学习率、按样本量保留的动量，以及两种不同目标下的权重衰减。可以先看默认的四倍 Batch，再切到接近饱和的区间，观察学习率倍率为什么趋近 1。

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

</details>

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
