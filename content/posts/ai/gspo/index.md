---
title: "GSPO 论文详解：从 token 概率比到序列优化，为什么能改善 MoE 强化学习？"
date: 2026-09-09
lastmod: 2026-09-09
draft: false
tags: ["Reinforcement Learning", "GSPO", "GRPO", "RLHF", "MoE", "Qwen", "PyTorch"]
categories: ["人工智能"]
authors: ["chase"]
summary: "分析 GSPO 的序列概率比、长度归一化、裁剪与梯度，核对 Qwen3 MoE 实验的证据边界，并提供可运行 PyTorch 对照。"
description: "分析 GSPO 的序列概率比、长度归一化、裁剪与梯度，核对 Qwen3 MoE 实验的证据边界，并提供可运行 PyTorch 对照。"
contentLanguage: "zh-CN"
math: true
toc: true
reading_prerequisites: "概率、自动微分、PPO/GRPO 概念与 PyTorch 基础"
reading_focus: "分清原始重要性权重与长度归一化替代目标，沿公式、梯度和实验逐项核对 GSPO。"
related_posts:
  - "/posts/ai/ppo-dpo-grpo"
  - "/posts/ai/distributed-training-memory"
---

**GSPO 保留 GRPO 的组内相对优势，把每个 token 各自的概率比，换成整条回答共享的、经过长度归一化的序列概率比，再以回答为单位裁剪。** 变化看起来只有几行公式，却会改变同一条回答中各 token 的梯度权重，以及一条样本何时停止贡献策略梯度。

本文分析 Qwen 团队的 *Group Sequence Policy Optimization*。依据是所提供的 7 页 PDF，文件页边标注 **arXiv:2507.18071v1，2025-07-24**，首页内部日期为 2025-07-25。[arXiv 版本页](https://arxiv.org/abs/2507.18071v1)还列出了后续 v2；下文的公式编号、图表和结论范围均对应这份 v1，不将后续框架行为混入论文原始定义。

阅读重点有三项：理解概率比为什么改，弄清长度归一化改变了什么，以及实验究竟验证到了哪一步。配套代码仅在 CPU 上核对目标与一阶梯度，不下载模型，也不声称复现 Qwen3 的训练收益。

| 阅读目的 | 入口 |
| --- | --- |
| 先理解算法变化 | [核心公式](#objective)、[梯度对照](#gradients) |
| 判断理论解释是否充分 | [重要性采样与归一化](#importance-sampling) |
| 正确实现裁剪与 detach | [裁剪](#clipping)、[GSPO-token](#gspo-token) |
| 评估论文证据 | [实验解读](#evidence)、[MoE 与工程边界](#moe) |
| 自己运行验证 | [Python 实验](#run)、[迁移清单](#implementation) |

## 1. 论文想解决什么问题 {#problem}

大模型 RL 往往先批量生成回答，再把 rollout 切成多个 mini-batch 更新参数。同一批回答由旧策略生成，后续更新时当前策略已经变化，因此会出现采样策略与训练策略的差异。

GRPO 不训练 critic，而是对同一个问题生成一组回答，根据组内奖励计算相对优势。一个回答通常只有一个最终奖励，其所有 token 共享优势；但 GRPO 仍为每个 token 使用不同的 current/old 概率比，并逐 token 裁剪。论文认为，这种组合在长回答和稀疏 MoE 上会放大训练噪声，并报告了严重的训练稳定性问题。

GSPO 的回应是让奖励、概率变化度量和裁剪决策都以完整回答为单位。它不是新的奖励模型，也没有取消 old policy；与 GRPO 一样，仍要采样、评分、计算优势和更新策略。

本文比较的是论文式 (2) 的 outcome GRPO：每回答先平均 token，再平均回答，并省略 KL 项。不同实现对长度归约、KL、采样和截断的选择可能不同，不能只看算法名称就认定目标完全一致。PPO、GRPO、DPO 的基础关系可先看[已有教程](../ppo-dpo-grpo/)。

## 2. 先把三个概率比写清楚 {#objective}

设 x 为问题，$y_i=(y_{i,1},\ldots,y_{i,L_i})$ 为组内第 i 条回答，$L_i$ 为有效回答 token 数，G 为组大小。用以下简写表示对**同一条已采样回答、同一个前缀**计算的概率：

$$
p_{i,t}=\pi_\theta(y_{i,t}\mid x,y_{i,<t}),\qquad
p^{\mathrm{old}}_{i,t}=\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t}).
$$

### 2.1 GRPO 的 token 概率比

$$
w_{i,t}=\frac{p_{i,t}}{p^{\mathrm{old}}_{i,t}}.
$$

每个 token 拥有自己的比例，某一处概率增加、另一处概率下降，会分别影响它们的梯度和裁剪状态。

### 2.2 原始序列概率比

自回归分解给出：

$$
R_i=\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)}
=\prod_{t=1}^{L_i}w_{i,t}.
$$

这才是完整回答分布之间的原始似然比。长序列上直接连乘既不便计算，也可能产生很大的数值变化。

### 2.3 GSPO 实际采用的比例

论文式 (8) 定义：

$$
s_i=R_i^{1/L_i}
=\exp\left(\frac1{L_i}\sum_{t=1}^{L_i}
\big[\log p_{i,t}-\log p^{\mathrm{old}}_{i,t}\big]\right).
$$

它是 token 概率比的**几何平均**，不是算术平均，也不是完整序列概率比本身。实现时先对 log-probability 差求有效 token 平均，再取 exp。

例如每个 token 的概率比都为 1.01，长度 100 时原始序列比约为 2.7048，长度 1000 时约为 20959.16，但两条回答的 GSPO 比例都为 1.01。归一化使比例更容易在统一数值尺度上比较，同时也弱化了长度累积的变化。

![每个 token 的概率比都为 1.01 时，原始序列比随长度增长，GSPO 的归一化比例保持 1.01](assets/length-normalization.png "根据配套算例绘制；纵轴为对数刻度，不是论文训练曲线。")

### 2.4 优势与最终目标

GSPO 保留组内标准化奖励：

$$
\widehat A_i=\frac{r(x,y_i)-\operatorname{mean}_j r(x,y_j)}
{\operatorname{std}_j r(x,y_j)}.
$$

其最大化目标为论文式 (6)：

$$
J_{\mathrm{GSPO}}=
\mathbb E\left[\frac1G\sum_{i=1}^{G}
\min\left(s_i\widehat A_i,
\operatorname{clip}(s_i,1-\epsilon,1+\epsilon)\widehat A_i\right)\right].
$$

期望覆盖问题与旧策略采样的回答组。实际实现需处理零方差；本文代码采用总体标准差 `correction=0`，分母加 $10^{-8}$，同分组的优势为零。这是明确的数值实现约定，不能从论文未写出的 epsilon 或标准差自由度直接猜出官方实现。

论文为突出策略目标而省略 KL，不意味着 GSPO 必须禁用 KL。old policy 是本批数据的行为策略，reference policy 是可能用于 KL 约束的参考模型，两者不能混为一谈。

## 3. 重要性采样：支持算法动机，但不能跳过归一化这一步 {#importance-sampling}

论文式 (5) 使用完整序列比得到重要性采样恒等式：

$$
\mathbb E_{y\sim\pi_\theta}[r(x,y)]
=\mathbb E_{y\sim\pi_{\mathrm{old}}}
\left[\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{old}}(y\mid x)}r(x,y)\right].
$$

这要求行为分布覆盖目标分布所需的支持集，并且奖励函数按该期望定义固定。若用 $R_i^{1/L_i}$ 替换 $R_i$，上述恒等式一般不再成立；再加裁剪和组内标准化后，更应把实际目标理解为用于稳定更新的替代目标。

可以用两个长度均为 2 的完整回答验证。旧分布为 `[0.5, 0.5]`，新分布为 `[0.8, 0.2]`，奖励为 `[1, 0]`。目标期望是 0.8；使用原始权重 `[1.6, 0.4]` 时，旧分布加权结果仍是 0.8。若对权重开平方，则得到：

$$
0.5\sqrt{1.6}\times1+0.5\sqrt{0.4}\times0
\approx0.63246.
$$

这不是代码 bug，而是长度归一化改变目标的直接结果。因此，不能把“序列比有清晰的重要性采样解释”扩展为“GSPO 的归一化裁剪目标是原奖励期望的无偏估计”。

论文第 3 节还强调：每个前缀的 next-token 分布只观察到一次采样，token 权重难以发挥修正作用。阅读时需要保留一个区别：**单样本会带来高方差，但不会仅因样本数为 1 就使重要性采样恒等式失效。** 对完整回答奖励的期望，单个条件概率比又确实不能直接替代完整序列比，因为前缀分布和后续生成也参与了联合分布变化。

更稳妥的理解是：作者提出了一个针对 GRPO surrogate 的失稳解释，并用训练结果支持序列层设计；这不构成“所有 token 级策略梯度都无效”的一般性证明。

## 4. 梯度究竟改变在哪里 {#gradients}

先忽略 clipping，把采样数据、old logps 和组内优势视为常量。GSPO 对单条回答的梯度为：

$$
\nabla_\theta J_i^{\mathrm{GSPO}}
=s_i\widehat A_i\frac1{L_i}\sum_t\nabla_\theta\log p_{i,t}.
$$

与之对照，论文所写 GRPO 的梯度为：

$$
\nabla_\theta J_i^{\mathrm{GRPO}}
=\frac{\widehat A_i}{L_i}\sum_t w_{i,t}\nabla_\theta\log p_{i,t}.
$$

GSPO 让整条回答共享一个标量权重 $s_i\widehat A_i$；GRPO 让每个 token 再乘自己的 $w_{i,t}$。所谓“token 权重相同”指这一个外部系数，不表示各 token 对模型参数的梯度向量相同，也不表示长短回答的梯度范数相同。

当 current 与 old 完全相等、两边使用相同优势与归约时，所有比值为 1，二者的策略目标一阶梯度相同。差异主要在同批数据经过更新、current 偏离 old 后显现；不能用更新前的一次梯度相等证明两个算法始终等价。

### 一个能手算的两 token 例子

令两个 token 的概率比分别为 0.5 和 2，优势为 +1，教学裁剪范围为 [0.8, 1.2]。GRPO 的第二个 token 被上界裁剪，GSPO 的比例则为 $\sqrt{0.5\times2}=1$，两 token 都贡献梯度。

| 方法 | 目标值 | 对第一个 logp 的导数 | 对第二个 logp 的导数 |
| --- | --- | --- | --- |
| GRPO | 0.85 | 0.25 | 0 |
| GSPO | 1 | 0.5 | 0.5 |
| GSPO-token，同一优势 | 1 | 0.5 | 0.5 |

这里展示的是**最大化目标对 token logp 的导数**。如果代码最小化 `loss=-objective`，符号反转；对模型参数的梯度还要乘各 logp 的参数雅可比。

![同一回答的两个 token 比例为 0.5 和 2 时，GRPO 与 GSPO 对 log-probability 的导数不同](assets/ratio-gradient.png "由 PyTorch float64 自动微分结果绘制，优势 +1、epsilon=0.2，仅用于解释机制。")

这一例子也揭示代价：一个 token 的概率下降可被另一个 token 的上升抵消。序列比接近 1，不等于每个 token 都接近旧策略，也不能单独当作 token KL 的上界。

## 5. 序列裁剪不是“比例越界就全部丢掉” {#clipping}

必须同时看比例和优势符号。对论文的 `min` 型 clipped surrogate，在边界之外：

| 优势 | 比例情况 | 当前策略目标的行为 |
| --- | --- | --- |
| 正 | $s>1+\epsilon$ | 取常数裁剪分支，该回答的此项梯度为零 |
| 正 | $s<1-\epsilon$ | 仍使用 $sA$，有梯度 |
| 负 | $s<1-\epsilon$ | 取常数裁剪分支，该回答的此项梯度为零 |
| 负 | $s>1+\epsilon$ | 仍使用 $sA$，有梯度 |

它限制的是已经朝奖励偏好方向变化过多的更新。若概率变化方向不利，仍保留纠正信号。边界处是不可微点，具体自动微分约定不应通过离散表格推断；本文测试刻意避开边界。

在原始 GSPO 中，一条回答的所有 token 共享裁剪决定。这里“梯度为零”仅指该 clipped 策略项；如果训练还包含 KL、熵或其他损失，并不意味着这些项也停止贡献梯度。

本文为方便手算使用 $\epsilon=0.2$，**不是论文公布的 GSPO 最优超参数**。v1 指出 GSPO 与 GRPO 的裁剪范围往往相差数量级，但没有在正文给出完整可复现的训练配置。不能直接复制 GRPO 的范围后，将结果视为公平的 GSPO 对照。

## 6. GSPO-token：数值相同，梯度从另一路进入 {#gspo-token}

论文式 (15) 定义：

$$
s_{i,t}=\operatorname{sg}(s_i)
\frac{p_{i,t}}{\operatorname{sg}(p_{i,t})},
$$

其中 sg 表示停止梯度。右侧分母是**当前概率的停止梯度版本**，不是 old probability。前向计算中后一个比例为 1，所以每个 token 看到的数值仍是 $s_i$；反向传播时，$s_i$ 被固定，只从当前 token 的概率进入梯度。

在 log-probability 空间可以写成：

```python
ratio = sequence_ratio.detach()[:, None] * (
    current_logps - current_logps.detach()
).exp()
```

再对每个 token 计算 clipped surrogate，并按有效长度平均。当一条回答内所有 token 使用相同优势时，GSPO-token 与 GSPO 的目标值、裁剪条件和一阶梯度一致。它们的计算图不同，不应进一步推断二阶导数也一致。

若给不同 token 设置不同优势，GSPO-token 允许更细的分配；尤其在多轮 RL 中可以表达回合或片段级差异，但优势如何设计仍需另外论证。此时正负优势可能并存，不能再认为整条回答的裁剪梯度状态必然一致。

## 7. 论文实验支持了什么 {#evidence}

以下事实来自所提供 PDF 第 4–6 页，主要对应图 1–3。正文没有完整超参数表、逐点原始数据或误差条，因此适合做机制与趋势分析，不适合据图计算精确加速倍数。

| 项目 | v1 报告的设置或观察 | 阅读边界 |
| --- | --- | --- |
| 起点 | 从 Qwen3-30B-A3B-Base 微调得到的 cold-start 模型 | 不等于直接从 Base 权重开始 RL |
| 对照 | GSPO 与调参后的 GRPO；GRPO 使用 Routing Replay | 不是一个完全没有稳定化措施的 GRPO 基线 |
| AIME’24 | 32 次采样评估 Pass@1 | 不能写成 Pass@32 或 Best-of-32 |
| LiveCodeBench | 202410–202502，8 次采样评估 Pass@1 | 不能写成 Pass@8 |
| CodeForces | Elo Rating | 是不同量纲的评估指标 |
| 图 1 | GSPO 在训练奖励与三个基准曲线上显示更好的效率趋势 | 横轴 Training Compute 没有给出可换算的绝对刻度 |
| 图 2 | clipped token fraction：GSPO 0.15，GRPO 0.0013 | 约 115 倍差异，不代表同等倍数的速度或样本效率增益 |
| 图 3 | GRPO 有无 Routing Replay 的稳定性对照 | 支持该 MoE 配置下路由处理的重要性 |

“Pass@1 over 32 samplings”表示用多次采样估计单次回答成功率，不是允许从 32 个候选中挑一个答对的通过率。对推理模型，解码预算和采样方式会明显影响指标解释。

图 1 的叙述还涉及训练中调整 query set、延长生成长度、增加计算投入。它体现整套训练流程的扩展表现，但缺少更多受控消融时，不能把所有提升都精确归因于某一项数学变化。

图 2 的裁剪比例很有启发性：保留更多 token 梯度不一定产生更有效的更新。但它也不能单独证明“裁剪越多越好”。统计 token 比例时，长回答权重更高；还应区分纯粹越过裁剪区间的比例，与结合优势符号后真正停止该项梯度的比例。

## 8. MoE 与训练基础设施：有价值的证据，也有条件 {#moe}

### 8.1 Routing Replay 为什么出现在论文中

MoE 只激活部分专家。论文在 48 层 Qwen3-30B-A3B-Base 上观察到：一次 RL 梯度更新后，对同一个 rollout 样本，新旧策略激活的专家约有 10% 不同。这个百分比描述专家激活变化，不是“10% 的参数被更新”或“10% 的 token 错误”。

作者先前用 Routing Replay 缓存旧策略的专家选择，在训练计算中重放，使 current 与 old 的相关 token 计算使用一致的激活网络。它有额外存储和通信开销，也会约束更新时的路由行为。图 3 展示了这种处理对该 GRPO 配置的作用。

GSPO 在作者报告的实验中不依赖 Routing Replay，仍保持较好的训练稳定性。序列层汇总可能缓和局部 token 概率剧烈变化带来的影响，这是算法具有工程吸引力的地方。

不过，序列平均是否降噪还取决于扰动相关性。若各 token 的 log-ratio 都朝同一方向偏移，平均并不会消除这个共同偏差。第 4 节的抵消例子也说明，聚合可能掩盖局部大变化。因此应写成“论文在这些设置中观察到稳定化”，而不是“GSPO 从理论上保证所有 MoE 永不崩溃”。

### 8.2 能否直接使用推理引擎返回的 old logps

论文第 5.4 节提出，序列层比例可能更容忍训练引擎与推理引擎的数值差异，因此有望省去训练端重算 old logps。其措辞与证据性质更接近工程潜力，不能当作任意训练系统都可删除该步骤的证明。

实际迁移时应核对行为策略的定义：采样温度、top-k/top-p、约束解码、模型版本、精度与路由设置都会影响“采样分布”和“所保存概率”是否一致。尤其是截断采样，不能拿任意未经处理的模型概率，就声称得到了真实行为分布的重要性权重。

可以分别记录训练端与推理端的平均 log-ratio 差、差异分布尾部、GSPO 裁剪状态变化以及实际奖励曲线，再判断能否省略重算。本文的 CPU 目标实验不包含 MoE 或双引擎，不能回答这项性能问题。

## 9. 可运行 Python：核对公式与梯度 {#run}

下载 [gspo_lab.py](gspo_lab.py)、[test_gspo_lab.py](test_gspo_lab.py) 和 [requirements.txt](requirements.txt)，放在同一目录。使用 Python 3.10+：

```bash
python3 -m venv .venv-gspo
source .venv-gspo/bin/activate
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest -v test_gspo_lab.py
python gspo_lab.py --output results.json
```

本文实际运行环境为 PyTorch 2.8.0+cu128，全部实验张量仍在 CPU 上；上面的安装命令选择 CPU wheel，不需要 GPU。六项测试检查正负优势的单侧裁剪、变长回答下 GSPO-token 的一阶梯度等价、old 与优势停止梯度、NaN padding 的屏蔽、零方差组与空回答，以及 current=old 时的梯度一致性。

### 9.1 关键实现只有几步，但归约顺序不能错

脚本中 `current`、`old`、`mask` 的形状均为 `[回答数, padded token 数]`。优势是一条回答一个值；例子按等大小组展开，因此直接平均回答对应每问题等权。

```python
lengths = mask.sum(-1)
c = torch.where(mask, current, torch.zeros_like(current))
o = torch.where(mask, old.detach(), torch.zeros_like(old))
sequence_ratio = ((c - o).sum(-1) / lengths).exp()
a = advantages.detach()
objective = torch.minimum(
    sequence_ratio * a,
    sequence_ratio.clamp(1 - epsilon, 1 + epsilon) * a,
).mean()
loss = -objective
```

先按回答求平均 log-ratio，再 exp、再裁剪。若先裁剪各 token 比例然后平均，已经回到了另一种目标。若把全批 token 一起平均，长回答会获得额外权重，也改变了本文采用的回答等权约定。

实际训练应先用 `log_softmax(logits)`，再 gather 当前回答 token 的 logp，并保证 logits 与 next-token label 错位对齐。prompt 和 PAD 不计入回答长度；EOS 是否计入，应与真实生成、奖励和旧概率记录保持一致。本文使用已对齐的 logps 隔离目标差异，相关 token 构造可参考[已有 token 实验](../ppo-dpo-grpo/#token-lab)。

掩码在相减前生效：直接算出 NaN 后再乘零，并不能把 NaN 清掉。空回答则直接报错，不能靠长度 clamp 悄悄制造一个虚构训练样本。脚本对比值溢出也会报错，而不额外截断 log-ratio 来改变论文目标。

### 9.2 应看到哪些结果

运行后可对照[本文实际结果 JSON](assets/results.json)。关键字段为：

- `cancellation`：正优势下 GRPO 梯度约 `[0.25, 0]`，GSPO 与 GSPO-token 为 `[0.5, 0.5]`；负优势下分别约 `[0, -1]` 与 `[−0.5, −0.5]`。
- `clipping`：验证第 5 节四种情况，而不是只统计概率比是否越界。
- `is_identity`：原始权重给出 0.8，开方权重给出约 0.63246。
- `advantages`：同分组为零；非同分组接近 ±1，微小偏差来自分母 epsilon。

这是一套固定 log-probability 的目标与梯度检查，不含在线 rollout、奖励模型、优化器训练或 benchmark 评测。通过测试说明这些原子计算与推导一致，不能说明已经复现论文的训练效率。

可选下载 [plot_results.py](plot_results.py) 重画本文两张机制图：

```bash
python -m pip install matplotlib==3.10.6
python plot_results.py --input results.json --output figures
```

## 10. 从原子实验迁移到训练系统 {#implementation}

一个可核对的训练流程应依次完成：

1. 固定行为策略与采样配置，对每个问题生成 G 条回答，保存回答、有效 mask、旧策略概率和版本信息。
2. 评分并在同题组内构造优势，明确标准差、零方差组和截断回答的处理方式。
3. 当前策略对相同 token、相同前缀计算 logps，以每条回答的有效长度计算 $s_i$。
4. 对回答应用 clipped surrogate，必要时另加明确定义的 KL 或其他损失，再反向更新。
5. 复用本批 rollout 时保持 old logps 不变；新一批采样才更新相应行为策略信息。

除了奖励，还应观察回答长度、有效组比例、平均与尾部 log-ratio、按优势符号区分的裁剪比例、梯度范数，以及长度分桶后的指标。零方差组占比很高时，应先检查任务难度和奖励区分度；更换比例定义不会凭空产生优势信号。

做 GRPO/GSPO 对照时，尽量固定起始 checkpoint、问题分布、奖励、rollout 数、生成预算、优化器和评估协议。裁剪范围允许按各目标的数值尺度调参，但应报告调参预算。MoE 路由策略与 old logps 的计算来源也属于实验条件，不能隐藏在同一个算法标签下。

## 阅读自测与验收

- 能否区分 token 比例、原始序列比例和长度归一化比例，并解释为什么后者不保留原始重要性采样恒等式？
- 运行六项测试，解释正负优势的裁剪方向，以及 GSPO-token 与 GSPO 一阶梯度等价的条件。
- 能否区分论文的训练趋势、裁剪比例与工程潜力，不把 CPU 原子实验当作 MoE 训练复现？

<details>
<summary>展开核对：关键结论</summary>

- 原始序列比是 token 比的乘积，GSPO 使用其长度次方根；归一化改变了加权目标。
- 只有特定优势符号与越界方向的组合进入平坦裁剪分支；GSPO-token 在同回答优势一致时与 GSPO 的值和一阶梯度等价。
- v1 提供单一 cold-start 模型设置下的主要曲线证据；原子脚本不涉及 MoE、在线 RL 或跨引擎吞吐。

</details>

## 参考资料与版本

- [Group Sequence Policy Optimization，arXiv v1](https://arxiv.org/abs/2507.18071v1)：本文分析对象；式 (6)–(8) 为主目标，式 (14)–(18) 为 GSPO-token，图 1–3 为实验与路由对照。
- [Qwen 官方 GSPO 介绍](https://qwenlm.github.io/blog/gspo/)：作者团队对方法与工程动机的补充说明。
- [PPO 原始论文](https://arxiv.org/abs/1707.06347)、[DeepSeekMath / GRPO 原始论文](https://arxiv.org/abs/2402.03300)：用于继续追溯基线；本文比较公式以所提供 GSPO v1 中的定义为准。
