---
title: "Muon 优化器详解：原理、实现与关键细节"
date: 2026-11-03
draft: false
tags: ["Muon", "Machine Learning", "Deep Learning", "Regularization", "Paper Notes"]
categories: ["人工智能"]
authors: ["chase"]
summary: "从谱范数约束推导 Muon，讲清极分解、Newton–Schulz 近似、四种尺寸缩放、学习率与衰减换算，再分析 Moonlight、QK-Clip、参数分组和分布式实现。"
description: "从谱范数约束推导 Muon，讲清极分解、Newton–Schulz 近似、四种尺寸缩放、学习率与衰减换算，再分析 Moonlight、QK-Clip、参数分组和分布式实现。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
images: ["assets/muon-update-flow.png"]
reading_prerequisites: "梯度下降、矩阵乘法与奇异值分解；运行数值复算只需 Python 标准库"
reading_focus: "先区分更新方向、更新幅度与权重，再理解有限步正交化和四种缩放，最后核对参数分组、QK-Clip 与实现顺序。"
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

有些实现改用 EMA：$M_t=\beta M_{t-1}+(1-\beta)G_t$。从零状态开始、固定 $\beta$ 时，两种动量只差固定的正比例；若 Nesterov 也一致换算，理想极分解会消去这个比例。固定 $\varepsilon$、有限精度和动态变更动量时，仍应逐行核对实现，而不是仅对照变量名。

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

把非零奇异值对应的紧致 SVD 写为 $G=U_r\Sigma_rV_r^\top$。谱范数与核范数的对偶关系给出：

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

## 3. 实际计算的是有限步 Newton–Schulz {#newton-schulz}

### 3.1 从矩阵多项式变成标量谱映射

反复做 SVD 的代价较高。Muon 使用以矩阵乘法为主的多项式迭代。先按 Frobenius 范数归一化，再做：

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

固定 $\varepsilon$ 会让 $H$ 与很小的 $cH$ 得到不同初始谱；BF16 的舍入、下溢与有限步迭代也会带来差别。此外，裁剪当前梯度并不等于同时缩放全部历史动量，不能据此断言“梯度裁剪对 Muon 永远没有作用”。

## 4. 四种缩放约定决定学习率的数值 {#scaling}

### 4.1 相同名字可以对应不同更新幅度

沿用 $W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$，把更新写成 $-\eta s(W)O$。苏剑林的指南整理了四种常见约定：[四种版本][guide]

| 约定 | 尺寸因子 $s(W)$ | 尺寸含义 |
| --- | --- | --- |
| 朴素版 | $1$ | 不做额外尺寸缩放 |
| Jordan 约定 | $\sqrt{\max(1,d_{\rm out}/d_{\rm in})}$ | 输出维度小于输入维度时不继续缩小 |
| 指南中的 μP 形式 | $\sqrt{d_{\rm out}/d_{\rm in}}$ | 保留输入、输出维度的非对称性 |
| Moonlight 约定 | $0.2\sqrt{\max(d_{\rm out},d_{\rm in})}$ | 以更新 RMS 对齐为出发点 |
{.table-readable}

这里的“μP 形式”指表中的尺寸因子，不代表仅改变这一个系数，就自动完成了整套 maximal update parametrization；初始化、其他参数组及模型参数化仍需一起定义。

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

### 5.3 报告证明了什么

Moonlight 技术报告把权重衰减与尺寸缩放结合，并实现分布式矩阵更新。在其 compute-optimal 对照中，报告用约 52% 的训练 FLOPs 达到对应性能；这描述的是该实验的计算效率，不是每个优化器步骤都快一倍。[Moonlight 技术报告][moonlight]

同一报告还比较了不同预训练和微调优化器的组合，显示切换优化器并不总带来优势。更新 RMS 对齐提供调参起点，不能消除初始化、谱结构、数据和任务目标的影响。

## 6. 参数分组与一个可执行的单步例子 {#implementation}

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

MLA 还需区分头独立与共享的部分。若分数由 content 项与 rotary 项相加组成，报告采用：

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
  <figcaption><span class="article-figure__number">图 4</span><span class="article-figure__text"><strong>共享项决定缩放该放在哪里。</strong>图中采用构造的无 bias、固定输入双线性分数；数值用于复算缩放，不是报告中的训练轨迹。</span></figcaption>
</figure>

### 7.3 数学恒等式与真实反馈时序

若先记录分数，再更新权重，旧 $S_{\max}$ 与更新后权重产生的分数未必相同。报告算法使用前向期间收集的统计来控制后续权重；这是反馈机制，不能被解释成对任意新输入、每个中间状态都存在一个无条件的硬上界。

例如，旧统计为 200，优化器更新后同一输入的未裁剪最大分数变成 400，仍按旧统计取 $\gamma=1/2$，缩放后是 200 而非 100。精确的同比例结论必须保留“测量与缩放基于同一快照”的条件。

实现时还要确认是否包含 $1/\sqrt{d_h}$、是否排除 mask 位置、如何跨分片归约，以及共享 key 的分组关系。GQA 的共享 key 也不能被多个 query 头按不同比例重复缩放。对带 bias 或额外加性分数的结构，缩放权重是否仍让整个 logit 同比变化，需要重新推导。

报告中的 $\tau=100$ 是其训练配置，不是任意架构都应使用的常数；关于更易发生 logit 增长的谱解释，也应保留为作者提出的机制假设，而不是仅由 Muon 定义就能推出的定理。

## 8. 数值复算与实现核对 {#examples}

[随文脚本](muon_math.py)使用标准库复算标量谱多项式、尺寸因子和 QK-Clip 数字：

```bash
python3 -B muon_math.py
```

代码给出数学构造值，不会加载模型。可选的[绘图脚本](make_figures.py)需要 NumPy 与 Matplotlib；将它与 `muon_math.py` 保存在同一目录，再运行 `python3 -B make_figures.py` 即可重画三幅数值图。用独立 SVD 计算检查矩形、秩亏、零矩阵和有限步多项式时，也应将数值精度与设备条件另行记录。

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
