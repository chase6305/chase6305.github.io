---
title: "RRSI 论文详解：让 Agent 的自我改进跨任务泛化"
date: 2026-10-15
lastmod: 2026-10-16
draft: false
tags: ["RRSI", "LLM Agents", "Agent Harness", "Self-Improvement", "Regularization", "LLM", "Evaluation", "Ablation", "Paper Notes"]
categories: ["人工智能"]
authors: ["chase"]
summary: "详解 RRSI 如何在冻结模型权重时，通过稀疏编辑、演化历史、探索、泄漏审查、噪声下限、成本约束与结构剪枝改进 Agent harness，并核对跨任务结果、消融、置信区间和源码实现细节。"
description: "详解 RRSI 如何在冻结模型权重时，通过稀疏编辑、演化历史、探索、泄漏审查、噪声下限、成本约束与结构剪枝改进 Agent harness，并核对跨任务结果、消融、置信区间和源码实现细节。"
contentLanguage: "zh-CN"
math: true
toc: true
images: ["assets/frozen-model-evolving-harness.webp"]
imageZoom: true
reading_prerequisites: "LLM 工具调用、训练集与测试集、均值和误差的基本概念；公式会逐项解释"
reading_focus: "把模型权重与外围程序分开，再按候选提议、筛选、独立评估和成本口径阅读，重点检查附录中的完整接受规则。"
related_posts:
  - "/posts/ai/llm-training-metrics"
  - "/posts/ai/rlhf-rlvr-rlcd"
  - "/posts/ai/foundations-modern-ai"
  - "/posts/ai/jev-decision-layer"
---

一个 Agent 在反复测试后，自动给自己增加提示、记忆、检查步骤和工具，榜单分数持续上涨，这是否说明它学会了更通用的工作方法？还要看：换一批任务后是否仍然有效，以及它用了多少额外推理。

[《RRSI: Regularized Recursive Self-Improvement of Agent Harnesses》](https://arxiv.org/abs/2609.24972v3)研究的正是这个问题。作者来自 Google Cloud AI Research 等机构，提出 **RRSI，Regularized Recursive Self-Improvement**：让大模型自动修改 Agent 的外围程序，同时约束提议与筛选过程，减少对演化任务的过拟合。

本文以所提供的 **27 页 PDF、标注 arXiv:2609.24972v3 的版本**为主，结合[官方仓库](https://github.com/google-research/rrsi)的提交 `be50316e1db0` 解释实现。[项目页](https://regularized-rsi.com/)提供逐轮轨迹，其运行配置差异见[第 11.3 节](#source-versions)。实验数字采用论文报告，离线复算与完整基准的复现入口见[第 12 节](#reproduction)。

首先区分两件事：**RRSI 演化的是模型周围的程序，底层模型权重保持冻结；程序变好需要由未参与搜索的任务验证。** 这也是全文理解公式、实验和限制的主线。

<figure class="article-figure" id="fig-frozen-model-harness">
  {{< post-image src="assets/frozen-model-evolving-harness.webp" alt="RRSI 演化前后使用相同权重的冻结大模型，修改的是外部提示、控制流、工具与记忆和上下文管理；最终固定 harness 后在未见任务上做迁移评估" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">同一个模型可以被不同的外围程序组织起来，表现出不同的任务完成能力。模块箭头表示上下文与协调关系，省略具体调用时序；图中的编辑标记不表示每轮都会修改所有组件。迁移测试发生在搜索之后，不把未见任务的反馈送回演化循环。</span></figcaption>
</figure>

## 阅读路线与核心术语 {#reading-map}

- **理解方法：** 从[模型与外围程序](#what-evolves)、[自适应过拟合](#adaptive-overfitting)进入[搜索循环](#overview)，再读[提议](#proposal)与[筛选](#selection)。重点回答“改什么、为什么约束、如何保留”。
- **核算一次决策：** 跟随[候选例子](#worked-example)和[多轮选择](#multi-round-state)，观察当前版本、历史最高分与组件历史分别怎样变化；实现时查[源码](#source-map)与[复现入口](#reproduction)。
- **判断实验结论：** 先读[评测协议](#evaluation-protocol)，再看[主结果](#main-results)、[成本消融](#ablation-cost)与[跨模型实验](#transfer-cases)，区分演化集提升、未见任务迁移与额外推理开销。

| 术语 | 本文含义 |
| --- | --- |
| Backbone / policy | 执行任务的大模型，记为 $\pi$，演化过程中权重不更新 |
| Harness | 围绕模型的提示、控制流、工具、上下文、记忆和输出管理程序，记为 $H$ |
| Evolve set | 反复用于产生反馈、评估候选和选择改动的任务集 |
| Incumbent | 当前已接受、下一轮在其基础上修改的 harness，记为 $H_t$ |
| Critic | 读取候选 diff，审查任务特化、泄漏和无效机制的外层角色 |
| Verifier / judge | 对 Agent 执行任务后的实际产物计分的测试、模拟器或评审模型 |
| ID held-out | 与演化任务同分布、但不参与搜索的留出任务 |
| OOD | 不同基准或任务分布上的评估，检验跨环境迁移 |
{.table-readable}

## 1. “自我改进”具体改了什么 {#what-evolves}

### 1.1 Agent 的能力来自模型与外围程序的组合

论文把 Agent 表示为 $A=(\pi,H)$。模型负责根据上下文生成下一步内容，harness 则决定模型能看到什么、何时调用工具、如何保存记忆、如何压缩长轨迹、遇到失败怎样恢复，以及什么时候结束。

同一个模型可能因为外围程序不同而出现这些差别：

- 一个程序把长命令输出全部塞回上下文，另一个保留头尾并支持继续读取。
- 一个程序看到命令失败就结束，另一个识别可恢复错误，进行有界重试。
- 一个程序提交文件后直接停止，另一个重新打开实际产物，核对结构与用户要求。
- 一个程序每轮都重新放入长规则，另一个压缩历史，同时保留必要任务约束。

这些改动可以改变成功率和 token 消耗，即使神经网络权重一个都没变。因此，RRSI 属于**Agent 系统层的递归改进**，与通过反向传播更新模型参数的 SFT、RL 后训练分别位于不同层次。

### 1.2 为什么称为 recursive

当前 harness 执行一组任务，产生轨迹和失败反馈；外层 proposer 根据这些反馈修改 harness；新的 harness 又生成下一轮轨迹。前一轮的系统状态和经验影响后续候选，形成递归的改进循环。

“自动改自己的程序”并不意味着无需任务、验证器或预算，也不意味着同一套改动能在任意模型上持续提升。论文研究的是给定模型、环境和有限轮数下，这种自动搜索能否产生可迁移的程序机制。

## 2. 权重不变，为什么仍然会过拟合 {#adaptive-overfitting}

### 2.1 反复使用同一组反馈，相当于在搜索程序

设任务 $x$ 的执行轨迹为 $\tau$，验证器给出 $r(x,\tau)\in[0,1]$。论文关心两项统计：

$$
S(H;D)=\mathbb E_{x\sim D,\,\tau\sim(\pi,H)}[r(x,\tau)],
\qquad
C(H;D)=\mathbb E_{x\sim D,\,\tau\sim(\pi,H)}[c(\tau)].
$$

$S$ 表示任务得分，$C$ 表示策略模型在轨迹中消耗的 token。实际只能在有限任务、有限次采样上得到 $\hat S$ 和 $\hat C$。

普通演化不断尝试候选，并留下演化集上得分最高的版本。问题在于，第 $t$ 轮的候选已经参考了前几轮在同一批任务上的结果。即使不直接复制答案，几十轮改动也可能逐渐适配这批任务的题型、文档风格、工具行为或评分偏好。

所以，**evolve set 应当被视为优化数据**。把它叫“测试时演化”，不会使其重新成为独立测试集。

### 2.2 三种风险会一起积累

| 风险 | 一种可能的表现 | 为什么演化集分数不能识别它 |
| --- | --- | --- |
| 基准特化 | 加入仅对当前题型、文件结构或评分套路有效的规则 | 当前任务正好具有这些特征 |
| 追逐噪声 | 偶然多解出一道题的候选被永久保留 | 一次随机评估无法区分稳定增益与好运 |
| 复杂度堆积 | 不断增加长提示、检查调用、重试和常驻记忆 | 更多计算有时能换得小幅提分，却未增加可迁移机制 |
{.table-readable}

这三者不要求显式出现任务 ID。一个不包含任何题目名称、却反复强调某基准特有评分方式的提示，仍可能只在那个分布上有效。论文附录 F 的案例正是为了说明：简单关键词检查不足以证明泛化。

## 3. RRSI 的整体循环 {#overview}

RRSI 保留开放的编辑空间：提示、控制流、配置、输出处理、上下文、工具、技能、记忆和子 Agent 都可以改。约束作用在**每轮如何提出候选、哪些候选可以成为长期状态**。任务划分、验证器和搜索规则仍属于外部实验协议；开放编辑空间不意味着候选可以修改评分器给自己加分。

论文的七项机制分为两组：提议侧包括编辑预算、历史归因和结构探索；选择侧包括泄漏审查、历史分数下限、成本约束和结构剪枝。剪枝虽然归在选择侧，却通过把删除目标交给下一轮 proposer 执行，因此同时依赖历史记录与重新评估。

<figure class="article-figure" id="fig-regularized-loop">
  {{< post-image src="assets/regularized-evolution-loop-v2.webp" alt="RRSI 外层循环：候选经审查后测量得分和 token，选择器应用历史下限、成本与领域规则；测量和选择结论都进入历史，下一轮继续提议，最终留出评估不反馈搜索" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text">提议侧控制一次改多少、参考哪些历史以及何时探索；筛选侧检查特化内容、噪声、成本和领域约束。有效测量与选择结论共同进入历史，包含未获胜候选；剪枝目标返回 proposer，删除改动也要重新评估。图中省略 smoke check、有限次数的修复和基础设施故障处理。</span></figcaption>
</figure>

可以用以下顺序理解一轮：先分析当前失败和演化历史，计算编辑预算，给 proposer 提供探索与剪枝目标；检查实际源码 diff；让通过审查和基本运行检查的候选在完整演化集上评估；应用分数下限、成本规则和领域 guard；最后选择合格候选中分数最高的一个。没有合格候选时，保留当前版本。

反馈也经过选择和压缩。公开实现先从低分任务选择较差 trial，并从少量成功任务选择较好 trial；只读 digester 分析具体轨迹，batch analyst 再汇总失败机制、能力缺口与成功习惯。proposer 依据这些报告和历史改程序，而不是每轮自动看完所有原始 trial。反馈覆盖不足时仍可能漏掉某类失败，完整测量与回归检查因此仍然必要。[分析器](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/analyst.py)、[轨迹选择](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/loop.py)

**保留当前版本是正常结果。** RRSI 不要求每轮都制造一个“改进”，也不把提议理由写得好当成接受依据。

## 4. 提议侧：怎样限制搜索对反馈的过度反应 {#proposal}

### 4.1 退火编辑预算：限制独立改动数

一个候选同时改提示、记忆、压缩策略和停止条件，即使总分上涨，也很难判断什么起了作用。RRSI 给每个候选设置最多可以捆绑的独立编辑数：

$$
b_t=\left\lceil b_{\min}+(b_{\max}-b_{\min})
\frac{1+\cos(\pi t/T)}{2}\right\rceil.
$$

$T$ 是总演化轮数，$t$ 是当前轮索引；$b_{\max}$ 允许早期组合少量协调改动，后期预算逐渐缩小，便于归因。这里的“一个编辑”指可独立描述的机制，不是一行代码、一个文件，也不是一次模型调用。

论文称它为 **$L_0$-style**：若每项独立编辑有一个是否启用的二进制指示变量，预算限制的是启用项数量。但它没有优化固定参数向量的 $L_0$ 范数损失；候选编辑池本身可以随轮次改变。

**取整与轮索引有一个复现细节。** 官方 `schedule.py` 使用上式向上取整，`driver.py` 执行 $t=0,\ldots,T-1$。在 $T=20,b_{\min}=1,b_{\max}=4$ 时，实际最后一轮 $b_{19}=2$，只有调度端点 $t=T$ 才得到 1。因此，论文“后期单编辑”的动机不能直接当成该提交中每个末轮候选最多一项的保证；复现时应打印实际预算序列。[调度源码](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/schedule.py)、[轮次执行](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/driver.py)

### 4.2 用完整编辑历史分配信用

外层历史记录每项已测量编辑的轮数、组件、假设、源码 diff、得分变化、成本变化以及是否获胜。它既保留成功，也保留回归和未被选择的证据，避免 proposer 反复提出已被实验否定的机制。

持久化记录与模型上下文要分开看：JSONL ledger 可以保存整个演化过程，`History.render()` 默认只把最近 40 条编辑记录整理给 proposer，并限制没有有效测量的失败记录数量。探索集合、剪枝和已接受组件计数则从完整记录计算；analyst 也保留跨轮失败模式摘要。因此，“参考演化历史”不意味着每轮把所有原始 diff 和轨迹完整重放进模型上下文。[历史摘要](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/history.py)、[一轮的上下文准备](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/loop.py)

这里有两个容易混淆的“历史”：**外层演化记录**帮助搜索器改程序；**harness 的 memory 组件**则是待演化程序的一部分。后者不能因为被称为记忆，就在不同评估 trial 之间保存任务答案、源文件内容或任务身份。官方 critic 会检查这种运行时状态泄漏。

另外，候选里捆绑的多项编辑共享同一个候选级 $\Delta S$ 和 $\Delta C$。记录“这个组件出现在提分候选中”，不等于已经通过因果消融证明它独立有效。稀疏编辑减轻归因困难，但不会自动消除混杂。

### 4.3 停滞后探索未测量过的组件

实现比较窗口两端的**净变化**：当已有至少 $w$ 轮记录，且 $\hat S_t-\hat S_{t-w}\le\delta$ 时，系统认为搜索停滞，并预留一部分候选位置给未被有效测量过的组件。例如分数依次为 0.80、0.83、0.82、0.815，$w=3,\delta=0.017$，窗口净增仅 0.015，仍触发停滞；中途曾上涨 0.03 并不会取消这一判定。

官方组件标签包括 `prompt`、`control_flow`、`config`、`output_plumbing`、`context_mgmt`、`client_tool`、`skill`、`memory` 和 `subagent`。这是一套跟踪搜索覆盖面的标签，不要求所有组件最终都必须存在。

提议侧的“未探索”与后文选择侧的“新颖度”还有不同定义：前者看**是否被测量过**，后者看某些结构类型**是否曾进入获胜路径**。一个 skill 候选已测量但落选，就不再属于从未探索过的类型，但仍可能具有选择侧新颖度。

组件标签也不能只凭 proposer 自报：实现会在候选 diff 中寻找对应模式，缺乏信号的声明会重新分类。这是一套启发式检查，不能证明声明的工具、技能或记忆已经被实际调用；合法声明的模式匹配还先于纯文本回退分类，具体例子见[源码对应](#source-map)。实际泄漏和机制有效性仍需 critic 与评估验证。[组件归类](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/components.py)

### 4.4 结构剪枝：把长期无用机制交回 proposer 删除

结构剪枝是论文选择侧的约束，使用的证据同样来自编辑历史：RRSI 跟踪每个组件最近窗口内的最佳已测量得分变化。若一个已经尝试过的组件，在窗口内没有严格正增益，便成为剪枝目标；窗口中没有测量时，其近期收益按负无穷处理。

剪枝通过下一轮的删除提议实现，**并不会跳过验证、直接删掉当前代码**。删除后的候选仍要通过审查、性能下限、成本和领域检查。

论文把这种删除离散结构的作用类比为 **Lasso / $L_1$-style**。这也不是对程序文本施加 $L_1$ 范数，更不是证明某个组件最近没有提分就永远无用；它是一种基于近期观测的保留策略。[历史、探索与剪枝实现](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/history.py)

## 5. 筛选侧：一项改动怎样获得保留资格 {#selection}

### 5.1 Critic 审查实际 diff，并先于完整评估

critic 检查的重点包括任务或实体特化、评分器访问与投机、无效机制、未声明的独立改动，以及跨 trial 保存任务内容等。理由里可以说明想修复什么失败，不能把对应任务知识写进最终运行代码。

这个步骤在完整评估之前发生，避免泄漏候选先得到一个诱人的演化集高分，再影响后续搜索。实现同时使用确定性模式检查和 LLM 审查，并允许有限次数修复；不能修复的候选会被丢弃。

但 critic 不是泛化证明，也不能代替程序运行测试：语法、构造与 smoke check 由其他步骤负责。一个通过 diff 检查的候选仍可能学到难以显式识别的分布偏好。[critic 实现](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/critic.py)

critic 与任务评分器的职责也不同：前者读程序改动，决定候选是否进入完整评估；后者读运行结果，给出任务成绩。公开设置中，Harvey LAB 的 rubric 评审使用 Gemini 3.5 Flash，coding 与 engineering 则使用各自测试或冻结的模拟器。执行任务的 policy、审查 diff 的 critic 和评审产物的 judge 应在复现记录中分别注明。[模型与评审设置](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/README.md)

### 5.2 历史最佳分数下限防止逐轮下滑

先分清三个参照，后面的公式才不会混用：

| 参照 | 回答什么问题 | 用在哪里 |
| --- | --- | --- |
| 当前版本 $H_t$ | 这次改动比当前多得几分、多耗多少 token？ | $\Delta S$、$\Delta C$ 和领域 guard |
| 已接受路径的历史最高分 $S^\star$ | 累积回退是否已经超出允许范围？ | 候选分数下限 |
| 搜索前的基础版本 $H_0$ | 最终演化结果比起点改善了多少？ | 主结果与成本对照 |
{.table-readable}

用 $S^\star$ 表示已接受路径中的历史最高测量分数，用重复评估基础 harness 得到的 $\delta$ 表示经验噪声容忍度。候选必须满足：

$$
\hat S(H')\ge S^\star-\delta.
$$

假设历史最高为 0.82，$\delta=0.017$，下限就是 0.803。当前分数可能略低于历史最高，但下一候选不能只以“相对当前下降很少”为理由继续一路降低，因为参照的历史最高不会随之下移。

两个细节很重要。第一，$S^\star$ 来自保留路径，不是任意一个因成本或领域规则而被拒绝的高分候选。第二，$\delta$ 是经验噪声带，不是形式化的泛化界，也不等于后文报告最终得分增益的 95% 置信区间。

这条下限检查的是**聚合得分**，没有要求每个任务都不退步。某些任务的改善可以抵消另一些任务的回归；需要单独保护的行为，应由对应的回归检查或领域 guard 约束。

### 5.3 明确提分的候选，新增 token 要有理由

相对当前 incumbent 定义：

$$
\Delta S=\hat S(H')-\hat S(H_t),
\qquad
\Delta C=\frac{\hat C(H')-\hat C(H_t)}{\hat C(H_t)}.
$$

当 $\Delta S>\delta$，使用：

$$
\Delta C\le\beta_0+\beta_1\Delta S.
$$

$\Delta C=0.25$ 表示 token 增加 25%；$\Delta S=0.01$ 表示分数增加一个百分点。公式中的分数始终按 $[0,1]$ 输入，不能把“加 1 个百分点”误填成 1。

例如 coding 配置的 $\beta_0=0.10,\beta_1=44.5$。如果得分增加 0.025，允许的相对 token 增长上限是 $0.10+44.5\times0.025=1.2125$，即 **121.25%**。这说明该规则会允许足够大增益购买显著额外计算，绝不是“任何候选最多增加 10%”。

论文把聚合资源增长控制类比为 **Ridge / $L_2$-style**：抑制无理由膨胀，却不要求某个具体组件必须被删除。它没有计算程序权重的平方和；源码部分注释沿用的其他范数命名，不改变上述实际分支条件。

### 5.4 噪声带内使用另一条规则

主文把完整细节留到附录 D.3。对 $\Delta S\le\delta$ 的候选，代码不再使用上一条成本上限，而是要求：

$$
w_s\Delta S-w_c\Delta C+w_n\nu_t(H')>0.
$$

$\nu_t$ 是此次触及、但此前没有进入获胜编辑路径的**不同结构组件类型数**，只在 `client_tool`、`skill`、`memory`、`subagent` 四类中计数。重写提示、改控制流或增加配置项没有这项奖励；同一类型改多次也不是多个新类型。

coding 配置令 $w_s=0$，所以噪声带内的小幅提分不能单独支持接受；节省成本或第一次引入某种结构机制才可能获得正值。workspace 与 engineering 使用正的 $w_s$，因此不能把 coding 的特殊规则推广到全部实例。

新颖度信用也有明确大小：在 coding 噪声带内，只有一个新结构类型、其他条件均满足时，成本要求是 $\Delta C<0.5/15=1/30$（约 3.33%，等号不通过）。该类型第一次获胜之后，同一类型不再提供这份信用。

这条分支也可能允许相对当前略微降分的候选，前提是没有跌破历史最佳下限，并有足够的成本或新颖度补偿。**RRSI 约束的是允许的转移，不保证每一轮分数都单调上升。**

两条成本分支在边界处还可能不连续。代入公开 engineering 参数，构造当前 $S=0.810$、$S^\star=0.820$、成本 1M、$\delta=0.020$、无新颖度且领域约束已满足的情况：候选得分 0.830、成本 2M 时，噪声带内的值为 $244\times0.020-2\times1=2.88>0$，可以通过；把分数提高到 0.831、成本保持 2M，就切到明确提分分支，允许增长仅为 $0.15+24.4\times0.021=66.24\%$，低于实际 100%，因而拒绝。这是公式与公开参数的构造推演，说明跨越分支时，分数更高也不保证更容易通过成本规则。可在[工程边界算例](/labs/rrsi/#engineering-boundary)中分别载入这两个候选，核对分支与阈值；按钮会重置整个实验。[两条分支实现](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/selection.py)

### 5.5 最后还要通过领域 guard，再按得分选赢家

有些条件不能用更高总分补偿。例如工程实例还限制有效提交率下降和无提交比例增加。最终必须同时满足审查、有效评估、历史分数下限、对应成本分支和领域规则。

公开 engineering 配置会在 `valid_rate` 相对当前版本下降超过 0.03，或 `no_payload_rate` 上升超过 0.02 时拒绝候选。这是两个独立比例的变化，不是 headline 得分变化；即使总通过率提高，有效率下降四个百分点仍会触发拒绝。coding 与 workspace 的默认 adapter 没有这两项工程 guard。[工程守卫](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/eng/adapter.py)

在所有合格候选中，选择测量得分最高的一个；不是选择成本最低者，也不是在所有分数、成本、新颖度上计算一个统一加权总排名。若只剩一个略低分但合格的候选，它也可能成为下一 incumbent。[完整选择实现](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/selection.py)

<figure class="article-figure" id="fig-selection-gates">
  {{< post-image src="assets/selection-gates.webp" alt="通过审查并完成测量的候选先检查历史分数下限，再按增益是否超过噪声带走两条互斥成本分支；通过领域约束后进入合格池，最终选取得分最高者，失败候选被拒绝，池为空则保留当前版本" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text">从上到下追踪一个候选：增益超过噪声带时检查允许的 token 增长，否则计算分数、成本与结构新颖度组合。两条分支只走一条；合格池内才按得分选赢家。图从已通过审查并得到有效测量的候选开始，省略更早的提议、修复和评估失败处理；具体不等式见本节正文。</span></figcaption>
</figure>

## 6. 一个可复算的选择例子 {#worked-example}

下面使用 coding 的成本参数演示数学分支：假设候选已通过 critic 和运行检查，得分都是 $[0,1]$ 小数，成本都是每条轨迹的平均 policy tokens。`novelty` 来自前述组件历史。F 额外设置一个失败的 guard，用来展示非补偿约束的优先级；它不是 coding 默认 adapter 的真实拒绝案例。代码省略真实轨迹采集与状态持久化。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    name: str
    score: float
    cost: float
    novelty: int = 0
    guard_ok: bool = True


def admissible(c, score=0.81, cost=1_000_000, best=0.82, delta=0.017):
    if c.score < best - delta:
        return False, "floor"
    ds = c.score - score
    dc = (c.cost - cost) / cost
    if ds > delta:
        paid = dc <= 0.10 + 44.5 * ds
    else:
        paid = -15.0 * dc + 0.5 * c.novelty > 0  # coding: w_s = 0
    if not paid:
        return False, "cost"
    if not c.guard_ok:
        return False, "guard"
    return True, "admissible"


candidates = [
    Candidate("A", 0.835, 1_800_000),
    Candidate("B", 0.819, 1_260_000),
    Candidate("C", 0.810, 950_000),
    Candidate("D", 0.800, 700_000),
    Candidate("E", 0.809, 1_000_000, novelty=1),
    Candidate("F", 0.850, 2_000_000, guard_ok=False),
]
qualified = []
for c in candidates:
    ok, reason = admissible(c)
    print(c.name, ok, reason)
    if ok:
        qualified.append(c)
winner = max(qualified, key=lambda c: c.score) if qualified else None
print("winner:", winner.name if winner else "keep current")
```

结果是 A、C、E 合格，最终选择 A：

| 候选 | 变化 | 判断 |
| --- | --- | --- |
| A | 得分 +2.5 个百分点，成本 +80% | 增益超过噪声，80% 小于 121.25% 的允许增长 |
| B | 得分 +0.9 个百分点，成本 +26% | 仍在噪声带内，又没有节省或结构新颖度，拒绝 |
| C | 分数不变，成本 −5% | 满足下限，节省成本使噪声带内规则为正 |
| D | 得分 −1 个百分点，成本 −30% | 虽省成本，但 0.800 低于历史下限 0.803，拒绝 |
| E | 得分 −0.1 个百分点，触及尚未被接受的结构类型 | 仍在下限以上，具有新颖度信用；合格但输给 A |
| F | 得分 +4 个百分点，成本 +100% | 数值上通过前两关，但违反领域 guard，拒绝 |
{.table-readable}

这个例子展示了两个不同概念：**admissible 表示有资格竞争，accepted 表示最终进入保留路径。** C、E 的测量仍会成为后续提议的证据，但它们不是本轮被接受的编辑。

也可以打开 [候选筛选交互实验](/labs/rrsi/)，载入 A～F 后调整噪声带、token 成本与组件历史，或把多个候选加入同一轮。A～F 只替换候选本身，不会重置当前路径；“重置整个实验”才返回本文的 coding 起点。

### 6.1 一轮选择怎样改变下一轮 {#multi-round-state}

先重置，再依次载入 A、C、E，每次点击“加入当前候选”。三者都合格，但应用本轮只接受 A：当前得分与历史最高分都变为 0.835，成本变为 1.8M。**E 没有获胜，所以 `skill` 不会被记为已接受。** 对下一轮而言，只有进入保留路径的类型才失去新颖度信用，合格却落选的候选仍只是测量证据。

再看另一个起点相同的实验：重置后，只加入 E 并应用。当前分数会从 0.810 变为 0.809，历史最高仍为 0.820，`skill` 被记入获胜路径。再次载入完全相同的 E，这次增益、成本变化和新颖度都是零，噪声带内表达式也为零，因要求严格大于零而拒绝；当前版本保持 0.809。

历史下限不会随着这次回退下移。继续输入分数 0.804、成本 900,000 的候选，它可以靠节省成本获得接受，但 $S^\star$ 仍是 0.820，下限仍为 0.803。接着输入分数 0.802，即使成本进一步降低，也不能通过下限。这些数字都是构造例子，不是论文中重新运行的演化轨迹。

这里的类型历史还带有“曾经接受过”的含义：即使后来删除了 skill 机制，再引入同一类型也不会自动恢复新颖度信用。计数来自已接受编辑的历史，并不通过扫描当前代码来重新判断哪些类型仍存在。

交互实验会列出每轮的 `ACCEPTED`、`LOST`、`REJECTED`，并更新当前得分、成本、历史最高和获胜路径类型。轮次记录可下载为 JSON，保留各轮起点、参数、候选和下一状态，便于离线核算。它按候选展示决策，没有重现官方逐编辑 JSONL 的完整字段，也没有运行 proposer、critic 或任务评估。手动修改当前路径会开始新的实验记录；刷新页面则清空本页状态。

## 7. 实验怎样区分演化与迁移 {#evaluation-protocol}

论文在三个领域分别演化各自的 harness，并在各自的未见任务上保持最终 harness 不变。不是先在所有领域共同搜索，再报告混合训练集分数。

| 领域 | 演化任务 | 独立评估 |
| --- | --- | --- |
| Coding | Terminal-Bench 2.1 的 89 个任务 | SWE-bench Verified，跨到真实仓库 issue 修复 |
| Agentic workspace | Harvey LAB 的固定 120 个任务 | 40 个 Harvey LAB ID 留出任务；JobBench、GDPval、APEX-Agents 三个 OOD 基准 |
| Engineering | 61 个无需专有模拟器的 EngDesign 任务 | Frontier-Eng；排除与演化集复用的 EngDesign 任务 |
{.table-readable}

总共是 **8 个基准、9 个结果分项**，因为 Harvey LAB 同时有演化和 ID 留出两项；独立于演化的数据共有 **1 个 ID 留出分项和 5 个 OOD 基准**。EngDesign 没有单独划出 ID 留出集。

Harvey LAB 的 120/40 划分还有数据版本条件：公开生成器固定上游提交 `1da4750`，只选择含 rubric 与文档、条目数不超过 100 且文档包不超过 20 MB 的任务，再按业务领域和条目复杂度确定性取样。换一个上游版本，即使仍叫“120/40”，任务、文档或条目权重也可能变化。复现应保存生成的 split 文件与上游 commit，不能只保存任务数量。[划分生成器](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/workspace/split_workspace.py)

主实验的执行 policy 为冻结的 Claude Opus 4.8，proposer、analyst 和 critic 也使用这一模型，但职责不同。相同基准上的对比方法使用相同基础 harness、policy、演化集和候选预算，基础版本与演化版本在同一评估窗口比较。

**$H_0$ 是各领域已有的完整 Agent。** Coding 从 Harbor 的 Terminus-2 开始；workspace 与 engineering 从 Archipelago 的 `react_toolbelt` 开始，后者已有 ReAct 工具循环、动态工具目录和 ReSum 式上下文摘要。因此，三领域的 $H_0$ 不是同一份程序；“从相同起点比较”指同一领域内各方法共享起点。分析最终改动时，应检查相对该起点新增、修改或删除了什么机制。[Coding 起点](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/coding/README.md)、[Workspace 起点](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/workspace/README.md)、[Agentic harness](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/third_party/archipelago/harness_eng/README.md)

### 7.1 不同“分数”的分母不同

| 基准 | 应怎样读分数 |
| --- | --- |
| Terminal-Bench | 任务隐藏测试通过的比例，不能靠写一段看似正确的说明替代 |
| SWE-bench Verified | 修复 fail-to-pass 测试，同时保持 pass-to-pass 测试通过的 resolve rate |
| Harvey LAB | rubric 条目通过数占全部条目数；条目多的任务权重更大，不是逐任务得分的简单平均 |
| JobBench | 基准加权 rubric 得分，而非所有任务严格二值通过率 |
| GDPval | 产物与人类专家产物比较的偏好胜率，使用多评审和呈现顺序控制 |
| APEX-Agents | 480 个任务的 pass@1，缺失轨迹仍在分母中 |
| EngDesign | 冻结模拟器或 testbench 检验是否满足设计要求 |
| Frontier-Eng | 相对固定金银铜门槛给出的 Medal Score，不是“完成任务百分比” |
{.table-readable}

Frontier-Eng 用固定 v1 门槛，达到金、银、铜分别获得 1、0.67、0.33 信用，再按其任务集合计算分数。论文说明环境无法构建的任务不给分，两组使用相同口径，47 项中有 38 项可贡献信用。仿真评分具有确定性，不代表模型采样或完整演化搜索没有随机性。

因此，不应把这些基准的绝对分数直接平均成“通用任务成功率”。本文后面的 OOD Avg. 专指论文表 1～2 中 **JobBench、GDPval、APEX-Agents 三项分数的算术平均**。

Harvey LAB 的加权可以用一个小例子理解：假设任务 A 的 2 条 rubric 全部通过，任务 B 的 8 条通过 4 条，则条目通过率为 $(2+4)/(2+8)=60\%$；若先平均任务得分，会得到 $(100\%+50\%)/2=75\%$。二者分母不同。这里是构造算例，解释为什么聚合器必须保留每条 trial 的权重，而不只是保存一个任务平均分。

## 8. 主结果：哪些改进迁移到了未见任务 {#main-results}

下面转录 PDF 图 3 的首轮主实验结果，并列出表 6 的增益区间，重采样方法见第 8.2 节。数值按论文保留一位小数；“增益”及其区间使用各基准的分数点，不是相对百分比。

| 基准与分项 | $H_0$ | RRSI | 增益 | 95% 区间 |
| --- | ---: | ---: | ---: | ---: |
| Terminal-Bench 2.1，Evolve | 74.2 | 80.2 | +6.0 | [4.3,9.3] |
| SWE-bench Verified，OOD | 82.0 | 83.8 | +1.8 | [1.1,4.0] |
| Harvey LAB，Evolve | 89.4 | 90.5 | +1.1 | [0.8,2.1] |
| Harvey LAB，ID held-out | 86.9 | 89.2 | +2.3 | [0.9,3.8] |
| JobBench，OOD | 36.0 | 40.7 | +4.7 | [2.9,7.4] |
| GDPval，OOD | 48.8 | 52.3 | +3.5 | [1.8,6.9] |
| APEX-Agents，OOD | 34.2 | 37.9 | +3.7 | [1.4,7.5] |
| EngDesign，Evolve | 50.0 | 54.9 | +4.9 | [1.8,9.8] |
| Frontier-Eng，OOD | 17.7 | 22.0 | +4.3 | [1.1,12.1] |

读这些结果，最有用的问题是“演化集上涨后，未见任务保留了多少”。例如 coding 演化集加 6.0 分，迁移到 SWE-bench 后加 1.8 分；workspace 演化集只加 1.1 分，三个 OOD 基准却分别加 4.7、3.5、3.7 分。提升大小没有必要按演化集的顺序排列。

**版本中的一处数字差异也应保留：** PDF 摘要写 OOD 最高 +4.3，但正文图 3、表 6、表 7 和公开 README 的 JobBench 都是 +4.7；+4.3 是 Frontier-Eng 的结果。本文按具体表格逐项报告，不把摘要概述覆盖到分项数据上。

### 8.1 为什么高演化集分数不足以选择方法

论文表 1 中，Meta-Harness 在 Harvey LAB 演化集达到 93.0，RRSI 是 90.5；二者 ID 留出均为 89.2。但三个 workspace OOD 的平均值约为 40.6 与 43.6。

TTHE 的演化集为 91.1，高于 RRSI，却在 OOD 平均约 38.0，低于 $H_0$ 的 39.7。其余对照的 OOD 平均约为 AHE 39.2、HarnessX 39.7。这个对照支持论文关注的取舍：高演化分数可能伴随较差迁移。

这些结论对应论文的相同预算比较，不能据此断言某个基线在任意预算、模型和实现中都不泛化。RRSI 的实验也没有把它的最终机制解释成所有 Agent 的最优结构。

### 8.2 任务不确定性和搜索不确定性分开检查

论文附录 E.2 用 **10,000 次任务级配对 bootstrap** 给增益估计 95% 区间：每次对同一批任务重采样，两种 harness 使用相同采样索引，再计算分数差。不是把同一任务的多次 trial 当作相互独立样本。

“配对”为什么有用？下面构造 100 个二值任务，比较固定的 $H_0$ 与最终 harness：

| 任务数 | 最终通过 | 最终失败 |
| --- | ---: | ---: |
| $H_0$ 通过 | 75 | 5 |
| $H_0$ 失败 | 7 | 13 |

基础版本通过 80 个，最终版本通过 82 个，净增益为 $7-5=2$ 个百分点，但其中有 5 个原本通过的任务退步。只保存 80% 和 82% 两个总分，会丢失这种对应关系。配对重采样保留同一任务在两个版本下的结果，不能把两组任务分别打乱后相减。对于 Harvey LAB 等带权指标，每次还要按原来的条目权重重新聚合，不能套用这个等权二值算例。该表只解释统计方法，不是论文原始任务记录。

<figure class="article-figure" id="fig-gain-intervals">
  {{< post-image src="assets/heldout-gains.webp" alt="RRSI 相对基础 harness 的九项分数增益及论文报告的任务级配对 bootstrap 95% 区间；包含三个演化分项、一个 ID 留出和五个 OOD 基准" >}}
  <figcaption><span class="article-figure__number">图 4</span><span class="article-figure__text">点值来自 PDF 图 3，误差线来自表 6；方形表示演化分项，圆形表示留出或 OOD 分项，颜色表示领域。区间由论文对任务重采样得到，本文未用原始试验重新估计。横轴是各基准自身的分数点，不同评分含义见第 7 节；区间不表示一条新演化搜索的全部不确定性。</span></figcaption>
</figure>

例如 SWE-bench 的增益区间为 $[1.1,4.0]$，Frontier-Eng 为 $[1.1,12.1]$。两者都高于零，但后者更宽，不能仅依据 +4.3 的点估计认定其真实增益非常精确。

另一项实验则重新运行演化搜索，只改变 proposer 随机性。workspace OOD 平均在两次运行中为 43.6 和 43.7；Terminal-Bench 的最终分数为 80.2 和 82.0，其他主结果分项的运行间差异不超过 0.6 分。它提供了重复搜索的证据，但仅有两次运行，仍不足以刻画长尾失败或所有超参数配置。

## 9. 消融与成本：为什么“少用 token”必须写清参照 {#ablation-cost}

论文表 2 把提议侧和接受侧正则化分组移除，其他设置保持可比。

| 方案 | Evolve | ID held-out | OOD Avg. | 每 trial policy tokens，百万 |
| --- | ---: | ---: | ---: | ---: |
| $H_0$ | 89.4 | 86.9 | 39.7 | 1.56 |
| 无正则化演化 | 92.8 | 88.9 | 40.3 | 3.80 |
| 去掉提议侧约束 | 90.7 | 88.8 | 41.9 | 2.69 |
| 去掉接受侧约束 | 91.5 | 88.7 | 41.0 | 3.59 |
| RRSI | 90.5 | 89.2 | 43.6 | 2.42 |
{.table-readable}

去掉任意一组，演化集分数都更高，OOD 却下降；两组都去掉，演化集最高、OOD 接近基础版本，同时 token 消耗最大。这支持两侧约束的组合价值，但这是**分组消融**，没有分别证明七个机制各自贡献了多少。

<figure class="article-figure" id="fig-cost-transfer">
  {{< post-image src="assets/cost-transfer-ablation.webp" alt="表 2 成本与迁移对照：基础 harness 为 1.56 百万 token 和 39.7 OOD 平均分；RRSI 为 2.42 和 43.6；无正则化演化为 3.80 和 40.3" >}}
  <figcaption><span class="article-figure__number">图 5</span><span class="article-figure__text">数据来自论文表 2。横轴在 workspace 演化集上测最终 harness 的平均推理成本，纵轴在三个 OOD 基准上测迁移分数；它们不是同一批任务上的联合测量。表中没有原始试验误差线，因此图中不补造区间。搜索器自身的提议、审查和反复评估成本另算。</span></figcaption>
</figure>

### 9.1 相对无正则化更省，相对基础版本仍更贵

按表 2 的舍入值计算，RRSI 相对无正则化的 token 降幅是：

$$
1-\frac{2.42}{3.80}\approx36.3\%.
$$

而相对基础版本，增长为：

$$
\frac{2.42}{1.56}-1\approx55.1\%.
$$

因此，摘要“约少 30% policy tokens”的概述，不能读成相对 $H_0$ 降低成本；表 2 的这个特定对比约为少 36.3%。RRSI 用更多推理换取了相对基础版本的提升，只是比无约束演化更克制。

policy tokens 也不是美元、墙钟时间、工具执行成本或评审成本。若使用输入缓存、不同模型计价或并行环境，token 与实际费用、延迟之间还需要额外换算。

### 9.2 最终推理成本与搜索总成本分开记录

workspace 设置为 20 轮、每轮两个候选、120 个演化任务、每任务两次 trial。若每个候选都进入完整评估，仅候选评估部分的名义预算就是 $20\times2\times120\times2=9{,}600$ 次 rollout；尚不含基线、分析、提议、critic、修复和重测。

实际候选会在审查或有效性检查中被过滤，附录 F 的 workspace 案例记载评估了 32 个候选、接受四项编辑。不能把 2.42M 的最终平均成本直接乘名义预算当成论文实测搜索消耗，因为中间 harness 的成本并不恒定。

## 10. 跨模型实验和具体改动说明了什么 {#transfer-cases}

### 10.1 独立搜索与直接迁移是两种实验

表 3 为两种冻结的执行 policy 分别独立搜索 coding harness：Claude Opus 4.8 和 Gemini 3.5 Flash。后者从 64.6 到 78.7，SWE-bench 从 76.8 到 79.0；它说明方法不限于一种 policy 家族，但每种执行模型都进行了自己的搜索。

表 4 才是把 Gemini 3.5 Flash 搜索得到的最终 harness **不修改地交给未参与搜索的 Gemini 3.1 Flash Lite**。Terminal-Bench 从 11.2 到 14.6，绝对增益 3.4 个百分点，相对增益约 30.4%。较大的相对百分比来自较低起点，不能写成“多解决了 30.4 个百分点的任务”。

这一实验支持某些机制可跨能力等级复用，仍不等于任意模型替换都兼容相同工具协议、上下文和调用接口。

### 10.2 被保留的机制比较像通用工作步骤

附录 F 给出几类具体决策：

- Coding R0-A 加入有界的完成前检查，以及长任务的非阻塞轮询指导，演化集增加 3.93 分，被接受。
- R0-B 的表面目标类似，但只有 +1.69 分且多耗 26.1% token，落在 coding 噪声带内，遭成本规则拒绝。
- R8-B 即使节省 13.6% token，也因 −2.81 分跌破性能下限而被拒绝。
- Engineering R2 处理“工作目录必须已存在”这一可复用工具错误，通过数从 122/244 到 128/244，token 增加 1.6%，被保留。

workspace 最终的四项改动集中在基础 harness 已有的完成前复查阶段：重新检查覆盖与深度，并用标准库重新读取产物，报告可观察的结构性缺失。与之对照，部分基线积累了长篇评分导向规则或特定文档类型协议。

这些案例说明“检查实际交付物”“有界恢复”可能成为可迁移机制；它们不是把某条固定提示复制进所有 Agent 就能重现论文增益的承诺。critic 只能检查显式问题，真正的迁移仍依靠独立评测。

## 11. 论文与源码怎样对应 {#source-map}

官方代码把通用搜索核心与领域适配分开。这里的链接固定到本文核对的提交。

| 代码位置 | 对应职责 |
| --- | --- |
| `rrsi/schedule.py` | 编辑预算的余弦调度与取整 |
| `rrsi/history.py`、`components.py` | 编辑历史、已探索类型、新颖度和剪枝目标 |
| `rrsi/propose.py`、`critic.py` | 候选实施、编辑标签、diff 审查与有限修复 |
| `rrsi/evaluate.py` | 按领域权重聚合得分，另汇总可用 token 成本 |
| `rrsi/selection.py` | 历史分数下限、两条成本分支和赢家选择 |
| `rrsi/loop.py`、`driver.py` | worktree、演化分支、日志、恢复与轮次执行 |
| `domains/<name>/adapter.py` | 各基准任务运行、计分、工具和领域 guard |
| `domains/<name>/rrsi.json` | 每个领域的固定实验配置 |
{.table-readable}

[核心代码目录](https://github.com/google-research/rrsi/tree/be50316e1db05914068a973f322770ef08ed7ba1/rrsi)中，score 聚合支持权重：coding 和 engineering 的 trial 权重为 1，Harvey LAB 的权重来自 rubric 条目数。缺失 trial 影响成绩分母，不能只平均成功完成的运行；token 则对已有的正数计数取简单平均，不使用 rubric 权重，也不会把缺失成本按零补齐。

**有分数，不一定有完整成本。** 在固定版本仓库中运行下面的构造例子：同一任务的两个 trial 得分都已记录，但候选漏记了第二个 token 数。两个版本的得分同为 0.5，程序却报告成本从 500,000 降到 80,000，即下降 84%：

```python
from rrsi.evaluate import TaskResult, aggregate, relative_cost_change

base = aggregate("base", 2, {
    "task": TaskResult([1, 0], tokens=[100_000, 900_000]),
})
partial = aggregate("partial", 2, {
    "task": TaskResult([1, 0], tokens=[80_000, None]),
})
print(base.S, partial.S)  # 0.5 0.5
print(base.C, partial.C)  # 500000.0 80000.0
print(relative_cost_change(partial.C, base.C))  # -0.84
```

这缺的是 token 记录，不是任务结果，例中的 `missing` 仍为 0。若漏记的成本其实是 920,000，候选平均成本仍为 500,000，并未降本。按 coding 参数、没有结构新颖度时，这两种成本记录甚至会使同一分数的候选分别通过和不通过成本项。因此，复现记录还应保存**有效 token 计数条数与预期 trial 数**，确认比较覆盖相同任务与重复次数，再解释降本；这个例子不表示论文实际运行存在漏记。

另一个实现边界是 `relative_cost_change`：若任一侧完全没有 token 计数，它返回 0。这与部分缺失时只平均剩余记录不同；两种情况都不能把“未记录的成本”解释为“没有增长”。[评估聚合源码](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/evaluate.py)

组件归类有一处调用顺序：`normalize()` 先保留合法、且有模式证据的声明；只有这一步未满足，才调用 `classify_diff()`。后者会把纯字符串或注释修改归为 `prompt`，但不能反过来覆盖已经保留的声明。在固定版本仓库中运行下面的构造例子，会得到 `prompt skill`：

```python
from rrsi.components import classify_diff, normalize

diff = '-"Read the guide"\n+"Read skills/example.md"'
print(classify_diff(diff), normalize("skill", diff))  # prompt skill
```

原因是字符串里的 `skills/` 已匹配 skill 模式；这个返回值并不证明新增了可执行技能，也不代表完整 critic 会放行该候选。复核标签时应连同 diff 与实际行为一起看。[分类与声明校验](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/components.py)

**计分完整性与测量有效性是两层检查。** Coding adapter 按 verifier reward 计分，没有 reward 时填 0；`AgentTimeoutError`、`VerifierTimeoutError` 本身不被归类为基础设施缺失，已有 reward 仍会读取。识别为容器、镜像等基础设施错误的 trial 则计入 `missing`，缺失槽位仍以零分保留在分母中。[Coding 结果分类](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/coding/adapter.py)

候选评估的公开 coding 配置允许最多 20% 缺失：178 个预期 trial 中，35 个缺失未超过这条门槛，36 个缺失会重试一次；重试后仍超限，才标为 `eval_invalid`，不进入分数与成本竞争。这里的 35／36 是配置边界算例，不是论文实际缺失数；通过完整性门槛也不代表候选已通过选择规则。[候选测量与重试](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/loop.py)、[缺失比例配置](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/coding/rrsi.json)

### 11.1 关键超参数

下表以论文表 5 为基础；最后三行的噪声带内权重补自各领域公开 JSON。

| 参数 | Coding | Workspace | Engineering |
| --- | ---: | ---: | ---: |
| 演化轮数 $T$ | 20 | 20 | 40 |
| 每任务 trial 数 $k$ | 2 | 2 | 4 |
| 每轮候选数 $m$ | 2 | 2 | 2 |
| 噪声带 $\delta$ | 0.017 | 0.004 | 0.020 |
| $b_{\min},b_{\max}$ | 1, 4 | 1, 3 | 1, 4 |
| 停滞窗口 $w$ | 3 | 3 | 3 |
| 预留探索候选数 | 1 | 1 | 1 |
| 剪枝窗口 | 4 | 4 | 5 |
| $\beta_0$ | 0.10 | 0.10 | 0.15 |
| $\beta_1$ | 44.5 | 35.4 | 24.4 |
| $w_s$ | 0 | 1414 | 244 |
| $w_c$ | 15 | 15 | 2 |
| $w_n$ | 0.5 | 0.5 | 0.5 |

大权重不意味着该领域更重视质量：Harvey LAB 把大量条目汇总成 $[0,1]$ 分数，权重是在对应计分单位上标定的。噪声带也应从未改变的基础系统重复运行得到，不能为了让更多候选通过而调低。论文把超参数固定在演化设置中，不使用 ID 留出或 OOD 结果来调参。

[Coding 配置](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/coding/rrsi.json)、[Workspace 配置](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/workspace/rrsi.json)、[Engineering 配置](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/eng/rrsi.json)。

### 11.2 噪声标定程序与最终结果区间使用不同重采样

论文的噪声容忍度来自不改变 harness 时的评估波动，代码提供两条估计路径：有多次独立基础评估时，使用这些分数的标准差估计两次评估之差；只有一次含多 trial 的基础评估时，在每个任务内部重采样 trial，估计固定任务集上的策略采样噪声。默认标定形式是 $\delta=z\,\mathrm{sd}_{\mathrm{null}}$，配置的 $z=2$；它不是对所有未来搜索轮次的显著性保证。

这与第 8.2 节的**最终增益区间**不同：最终区间按任务做配对重采样，关注换一批任务时的评估不确定性；`calibrate.py` 的单评估分支在任务内部重采样 trial，关注同一批任务重跑时的波动。二者回答不同问题，不能互换。

论文附录 E.1 对 engineering 的固定噪声带另描述了基础 trial 半分构造 null、读取分位数的过程；公开工具的默认两倍标准差估计不是这一过程的逐步复刻。因此，在新环境运行工具不保证恰好重得 0.020，应该记录估计方法，而不是为了匹配论文常数调整输入。

还有一个配置优先级：`Run.delta()` 先使用非空的 `cfg.delta`，只有它为 `null` 时才读取 `calibration.json`。发布的三个领域配置都带固定值，所以运行 `baseline` 或 `calibrate` 写出了新文件，**并不代表后续搜索自动采用这个新值**。复用论文固定设置时保留原值；在新模型或新任务集上重新标定时，先将该实验配置中的 `delta` 设为 `null`，再记录实际使用值与标定来源。[噪声估计实现](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/calibrate.py)、[配置读取优先级](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/loop.py)

### 11.3 论文结果与 explorer 轨迹按各自运行配置读取 {#source-versions}

项目页的逐轮 explorer 很适合查看 diff 和拒绝理由，但它展示的几条轨迹并非全部使用论文表 5 与当前公开 JSON 的同一配置。2026-10-15 查阅其[轨迹数据](https://regularized-rsi.com/data/evolution.js)时，可直接核对到：

| 核对项 | 论文或公开固定配置 | Explorer 对应记录 |
| --- | --- | --- |
| Opus coding 噪声带 | 0.017 | 0.03406；注释说明重新标定 |
| Opus coding 最终演化分数 | PDF 图 3 为 80.2 | 最终 incumbent 为 144/178，约 80.90 |
| Coding 搜索预算 | 公开默认 20 轮、每轮 2 候选 | Gemini 轨迹注明两段各 15 次迭代、每次 1 候选 |
| Engineering 搜索预算 | 公开默认每轮 2 候选 | 轨迹记录为 1 候选，并带 24 任务预筛 |
{.table-readable}

因此，阅读一条真实决策时，应使用该轨迹自己的 incumbent、历史最高分、噪声带和成本记录；不能把第 11.1 节的常数套回所有 explorer 记录。本文主结果与数值图继续采用所提供 PDF 的表格；第 6 节是按公开 coding 参数构造的复算例子。现有材料能说明运行配置不同，不能据此把两组最终分数合并、平均，或自行认定哪一组替代了另一组。

## 12. 复现入口：先验证核心，再准备真实任务环境 {#reproduction}

### 12.1 离线核心测试

官方核心测试不需要调用模型，也不需要基准容器，可以先核对调度、选择分支、历史和计分聚合。

```bash
git clone https://github.com/google-research/rrsi.git
cd rrsi
git checkout be50316e1db05914068a973f322770ef08ed7ba1
python3 -B tests/test_core.py
```

本文实际运行了这条测试命令，八组方法核心检查通过。这不包含候选生成质量、critic 有效性或外部基准可复现性。

### 12.2 完整 coding 搜索需要另一层环境

主搜索环境需要 Python 3.10 以上；coding 的 benchmark runner 还需要 Docker 与单独的 Harbor 环境。**Harbor 0.18.0 的发行元数据要求 Python ≥3.12**，所以不能把主搜索包的最低版本直接套给 runner。官方要求 `harbor>=0.18`；下面固定为已验证导入和构造的 0.18.0，并用 `python3.12` 创建独立环境。按官方 README 设置 Vertex AI 的模型访问与凭据；后续 `smoke`、`baseline`、`run` 和 OOD 评估会运行任务并调用配置模型。[Harbor 0.18.0 的 Python 要求](https://pypi.org/project/harbor/0.18.0/)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python3.12 -m venv domains/coding/.venv
domains/coding/.venv/bin/pip install "harbor==0.18.0"

# 在完成模型凭据、Docker 和领域设置后执行。
python3 rrsi.py --domain coding smoke
python3 rrsi.py --domain coding baseline
python3 rrsi.py --domain coding run
python3 rrsi.py --domain coding status

# 搜索结束，固定最终版本后做独立 OOD 评估。
bash domains/coding/scripts/swe_eval.sh
```

`smoke` 验证基本执行链路，不能代替完整任务评估。workspace 和 engineering 需要各自数据、MCP 工具与评分环境，并通过 `RRSI_AGENT_PYTHON` 等设置区分解释器。不能只安装搜索包，就认为所有基准依赖齐全。[官方快速开始](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/README.md)、[Coding 设置](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/coding/README.md)

本文的环境验证覆盖搜索包依赖、CLI 帮助、Harbor 0.18.0 下的 Terminus-2 类导入与构造，以及前述八组离线核心测试。构造检查禁用了网络、没有调用推理；它不等于运行过需要 Docker、任务数据和模型访问的 `smoke` 或完整基准。

还要区分 CLI 中 `round --t 0 --dry-run` 的含义：它在 proposer 之前停止；若该轮没有缓存的分析报告，仍会运行轨迹分析并可能调用模型，而且会写入分析文件。需要纯离线检查时，使用第 12.1 节的核心测试。[dry-run 的停止位置](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/loop.py)

### 12.3 保留可审计记录，而不是只保存最终提示

复现至少保存最终 commit、候选 diff、声明编辑、critic 判定、原始 trial、计分分母、token 统计、历史最高分、噪声标定、领域 guard 和留出任务清单。`runs/<domain>/frontier.json` 记录当前保留版本，轮目录与历史文件说明为什么发生一次转移。

恢复运行时，先区分三种操作：

| 操作 | 是否重新测量 | 对状态的影响 |
| --- | --- | --- |
| 继续未完成的轮次 | 已存在的候选 `eval.json` 会复用 | 从已保存产物继续；再次执行命令不等于获得独立重复测量 |
| `reevaluate` | 重跑已提交候选的测量，可用 `--variants A` 限定候选 | 随后调用重新裁决，可能改变赢家 |
| `readjudicate` | 使用已有测量，只重算选择规则 | 更新决策、编辑历史、frontier 与演化分支 |
{.table-readable}

两条命令均用 `--t <t>` 指定轮次。重新裁决要求 frontier 中不存在第 $t$ 轮之后的轮次，因为后续候选是从原先的赢家继续修改的；改变旧赢家后，后续程序不能直接接在新路径上。`reevaluate` 会先重跑测量，再调用这项检查，因此也应先核对轮次状态。比较不同阈值时，应在独立 checkout 和运行目录中保存各自演化路径。[恢复、重测与裁决实现](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/loop.py)

重新裁决不会为新阈值产生新证据；根据留出集表现反复调整阈值，也会把留出集变成新的优化数据。

文中数值可从 [results.json](results.json) 读取，图 4、图 5 可用仓库脚本 `python3 -B scripts/plot_rrsi_results.py` 重建。该脚本只重画论文数据，不会重跑模型实验。

## 13. 实验证据与适用边界 {#limitations}

论文的证据支持：在给定三类环境与预算下，对搜索过程施加约束，可以比对照演化方法得到更好的留出迁移；分组消融、两种独立 policy 搜索、一次跨模型直接迁移和两次演化运行为这个判断提供了不同角度的证据。

边界也应一并保留：

1. 模型权重冻结，结果不能直接覆盖权重与程序联合更新的自我改进。
2. 七种机制主要按两组消融，单个机制的独立贡献仍不清楚。
3. LLM critic 会漏判或误判，确定性任务评分也不能消除生成与搜索随机性。
4. 噪声带、预算和成本标定依赖环境；有限 evolve set 的自适应使用风险并未被理论上消除。
5. token 是资源代理指标，最终推理更省不意味着离线搜索总成本更低。
6. 跨模型直接迁移只覆盖给定模型组合，跨架构、工具生态和更长演化过程仍需验证。
7. 任务级 bootstrap 与两次搜索重跑提供有限的可靠性证据，不能推出任意新任务必然获益。

文档本身也有需要复现者核对的地方：除前述 OOD 摘要数字、调度端点外，附录 B.5 同时给出 GDPval 的 185 个任务与“每位评审 204 次比较”，仅靠这些描述无法还原一致分母。若要精确复现，应检查实际评测 manifest、双顺序比较与投票记录，而不要自行补出缺失协议。

与[大模型训练和指标](/posts/ai/llm-training-metrics)对照，可以把 RRSI 放在“优化什么、拿什么验证”的框架中；与[RLHF、RLVR、RLCD](/posts/ai/rlhf-rlvr-rlcd)对照，可以看到两者都依赖反馈质量，但优化变量不同；[现代 AI 的归纳偏置与泛化](/posts/ai/foundations-modern-ai)提供了理解正则化类比的背景。[Jev 决策层](/posts/ai/jev-decision-layer)则帮助区分任务运行时的协调，与这里跨评估轮次修改协调程序的外层搜索。

## 阅读自测与验收

**方法与搜索**

- **RRSI 更新模型权重吗？** 不更新；演化提示、控制流、工具、上下文等外围程序。
- **为什么 evolve set 不能继续当作独立测试集？** 后续候选反复适配它的反馈，它已经参与优化。
- **$L_0$、$L_1$、$L_2$ 是实际的范数惩罚损失吗？** 不是；分别类比编辑数量、离散结构删除与聚合资源增长控制。
- **官方代码末轮一定只允许一个编辑吗？** 不一定；该提交使用向上取整和 0 到 T−1 的轮次，20 轮示例末轮预算仍是 2。
- **提议侧未探索类型和选择侧新颖度有什么区别？** 前者未有有效测量，后者是四类结构中未进入过获胜路径的类型。
- **剪枝目标会立即从当前程序中删除吗？** 不会；删除作为后续候选提出，并继续通过评估与筛选。
- **三个领域的基础 harness H0 是同一份空白程序吗？** 不是；coding 从 Terminus-2 开始，workspace 与 engineering 从已有 ReAct、动态工具与上下文摘要的 react_toolbelt 开始；同起点比较发生在各领域内部。
- **审查候选 diff 的 critic 和给任务产物计分的 judge 是同一职责吗？** 不是；critic 在完整评估前审查改动，任务 verifier/judge 在执行后计分。公开 Harvey LAB 评审使用 Gemini 3.5 Flash，policy 与外层搜索角色另有设置。
- **纯字符串修改一定会被归成 prompt 吗？** 不一定；`normalize()` 会先保留有模式证据的合法声明，之后才回退分类，不能仅凭返回标签证明真实机制已实现。

**筛选与跨轮状态**

- **历史下限为什么使用最高保留分数而非当前分数？** 防止多轮小回归逐步降低允许的底线。
- **噪声带内的小幅提分会自动被接受吗？** 不会；还要满足 shaped rule、历史下限和领域 guard，coding 中 w_s 为零。
- **合格候选是否都会进入最终 harness？** 不会；只选合格候选中得分最高者，其他有效测量仍进入历史。
- **合格却落选的 skill 编辑会消耗下一轮新颖度吗？** 不会；有效落选测量进入历史，但只有获胜并接受的结构类型进入新颖度已用集合。
- **只有一个略低分但合格的候选时，会保留原版本吗？** 不会；原版本只是没有合格候选时的回退；采用候选后历史最高分仍取此前与新分数的最大值。
- **候选分数稍微提高，就一定更容易通过成本规则吗？** 不一定；跨过噪声带会切换分支，公开 engineering 参数下两条规则并不连续，应按实际分支计算。
- **候选的成本变化与历史最高分对应的版本比较吗？** 不比较；$\Delta S$、$\Delta C$ 以当前 $H_t$ 为参照，历史最高分只确定分数下限。
- **聚合分数通过历史下限，是否保证每个任务都不退步？** 不保证；不同任务的增减可以抵消，需要保留逐任务结果并单独检查关键回归。

**结果与统计**

- **八个基准为什么有九个主结果分项？** Harvey LAB 同时报告演化与 ID 留出；另有五个 OOD 基准。
- **workspace OOD Avg. 平均了哪些指标？** JobBench、GDPval、APEX-Agents 三个分数，不是全部五个 OOD 基准。
- **RRSI 的 2.42M tokens 比基础版本更省吗？** 没有；比 1.56M 约高 55.1%，比无正则化 3.80M 约低 36.3%。
- **任务级 bootstrap 能覆盖演化搜索随机性吗？** 不能；固定最终 harness 的任务重采样，与重新进行搜索是不同实验。
- **Gemini 两组实验有什么区别？** 一组为每个 policy 独立搜索，另一组把已演化 harness 直接交给未参与搜索的较小模型。
- **只有两个版本的总分，能重建任务级配对 bootstrap 吗？** 不能；必须保留相同任务在两个版本下的结果与原计分权重。
- **trial 得分完整，是否就能确认 token 降本可信？** 不能；token 可能部分漏记，成本均值只使用已有的正数记录，需另查成本覆盖条数和对应 trial。

**源码与复现**

- **生成新的 calibration 文件后，搜索一定采用新的噪声带吗？** 不一定；非空的 `cfg.delta` 优先，只有设为 `null` 才读取标定文件。
- **可以用论文固定参数直接重算 explorer 的所有决策吗？** 不可以；部分轨迹有不同的噪声带、候选预算或预筛阶段，应先读取那条运行的元数据。
- **只通过离线核心测试说明什么？** 数学和状态规则的相应实现通过测试，不能证明模型搜索、critic 和完整基准结果已复现。
- **RRSI 的 round --dry-run 保证不调用模型吗？** 不保证；它在 proposer 之前停止，未缓存的轨迹分析仍可能调用模型并写入报告。
- **继续轮次、重测与重新裁决有何区别？** 继续轮次会复用测量缓存；重测重新运行候选并触发裁决；单独裁决复用测量，但会更新赢家、历史与分支，且不能保留依赖旧赢家的后续轮次。
- **35／178 与 36／178 个缺失 trial 在默认 coding 配置中怎样处理？** 前者未超过 20% 门槛，仍补零计分；后者重试一次，若仍超限则标为无效评估。

## 参考资料

1. Peng Xia et al., [*RRSI: Regularized Recursive Self-Improvement of Agent Harnesses*](https://arxiv.org/abs/2609.24972v3)，arXiv:2609.24972v3。本文依据所提供的 27 页 PDF，公式见 §2～3 与附录 D，主结果见图 3、表 1～4，可靠性见附录 E.2，案例见附录 F。
2. [RRSI 官方项目页](https://regularized-rsi.com/)。
3. [RRSI 官方代码：本文核对的固定提交](https://github.com/google-research/rrsi/tree/be50316e1db05914068a973f322770ef08ed7ba1)。
4. [调度](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/schedule.py)、[演化历史](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/history.py)、[候选筛选](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/selection.py) 与 [计分聚合](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/evaluate.py)。
5. [泄漏 critic](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/critic.py)、[组件标签与新颖度](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/components.py) 与 [官方离线核心测试](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/tests/test_core.py)。
