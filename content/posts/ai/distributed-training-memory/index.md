---
title: "大模型分布式训练与显存优化指南：从 DDP、ZeRO 到 FSDP"
date: 2026-08-27
lastmod: 2026-09-05
draft: false
tags: ["Distributed Training", "GPU Memory", "PyTorch"]
categories: ["人工智能"]
authors: ["chase"]
summary: "建立训练显存账本，解释 DDP、NCCL、ZeRO 与 FSDP 的数据所有权，并梳理混合精度、重计算和 OOM 排查。"
math: true
toc: true
description: "建立训练显存账本，解释 DDP、NCCL、ZeRO 与 FSDP 的数据所有权，并梳理混合精度、重计算和 OOM 排查。"
contentLanguage: "zh-CN"
reading_prerequisites: "PyTorch 训练循环与 GPU 基础"
reading_focus: "沿参数、梯度、优化器状态和激活的生命周期追踪显存，不只计算权重大小。"
related_posts:
  - "/posts/cuda/warp"
  - "/posts/ai/transformer-attention"
---

本文从显存账本出发，解释 DP、DDP、NCCL 集合通信、ZeRO-1/2/3 与 PyTorch FSDP，并讨论混合精度、Activation Checkpointing、通信隐藏和显存碎片。重点不是背术语，而是回答两个工程问题：**每张 GPU 此刻持有什么，以及下一次通信为什么发生。**

> **学习目标**：能够估算 1B 模型的训练显存，解释 All-Reduce、Reduce-Scatter、All-Gather 的张量所有权，并根据显存容量、互联带宽和计算量选择 DDP、ZeRO 或 FSDP。

## 快速目录

- [训练显存由什么组成](#1-先建立训练显存账本)
- [分布式基础术语](#2-分布式训练的基本坐标系)
- [1B 模型计算示例](#3-1b-模型到底占多少显存)
- [DP 与 DDP](#4-dp-与-ddp)
- [NCCL 与集合通信](#5-nccl-与集合通信)
- [ZeRO-1/2/3](#6-zero-1zero-2zero-3)
- [FSDP 工作流程](#7-fsdp如何在计算时临时还原参数)
- [混合精度](#8-bf16fp32-与混合精度)
- [激活与 Checkpointing](#9-activation-checkpointing)
- [计算通信重叠](#10-让计算覆盖通信而不是让-gpu-空等)
- [碎片、监控与排错](#11-显存分配器碎片与排错)
- [选型与配置清单](#12-如何选择-ddpzero-和-fsdp)

## 1. 先建立训练显存账本

一次训练的峰值显存不只是模型文件大小：

$$
M_{\text{peak}}\approx
M_P+M_G+M_O+M_A+M_T+M_C+M_F
$$

| 符号 | 项目 | 生命周期 | 主要影响因素 |
|---|---|---|---|
| `P` | Parameters，参数 | 长期存在 | 参数量、参数 dtype、是否分片/量化 |
| `G` | Gradients，梯度 | 反向到 optimizer step | 梯度 dtype、是否分片、是否用 bucket view |
| `O` | Optimizer States，优化器状态 | 长期存在 | 优化器类型、master weight、状态精度、是否分片 |
| `A` | Activations，激活值 | 前向保存到反向消费 | Batch、序列长度、层数、隐藏维、Checkpointing |
| `T` | Temporaries，临时张量 | 某些算子期间达到峰值 | Attention、logits、融合算子、cuBLAS workspace |
| `C` | Communication Buffers，通信缓冲 | Collective 前后 | bucket 大小、All-Gather 预取、并发 Collective |
| `F` | Fragmentation/Allocator，碎片与保留空间 | 动态变化 | 动态 shape、分配顺序、缓存分配器 |

ZeRO/FSDP 主要减少 `P/G/O` 的冗余，不会自动消除激活、临时张量或碎片。模型状态已经能放下但仍 OOM，通常要继续检查 `A/T/C/F`。

### 1.1 通用计算公式

若参数量为 `N`，每个元素 `b` bit：

$$
M_{\text{GB}}=\frac{N\times b}{8\times10^9}
$$

这里使用十进制 GB；`1 GiB = 2³⁰ Byte`，因此操作系统或监控工具显示的数字可能略有差异。

## 2. 分布式训练的基本坐标系

| 术语 | 含义 | 例子 |
|---|---|---|
| Process | 独立训练进程 | DDP 通常每张 GPU 一个进程 |
| Rank | 进程在通信组中的编号 | `0…world_size-1` |
| Local Rank | 当前节点内的设备编号 | 双节点各 8 卡时，每节点为 `0…7` |
| World Size | 默认通信组的总进程数 | 2 节点 × 8 GPU = 16 |
| Process Group | 一组参与同一 Collective 的 Rank | DP、TP、PP 可使用不同通信组 |
| Node | 一台服务器 | 节点内常有 NVLink，节点间常用 InfiniBand/RoCE |

`rank` 不是 GPU 编号，`local_rank` 才通常用于选择本机 CUDA device。多维并行时，同一个 Rank 可能同时属于一个数据并行组、一个张量并行组和一个流水线组。

## 3. 1B 模型到底占多少显存

`1B = 10⁹` 个参数。只加载一份权重时：

| dtype | 每参数 Byte | 1B 参数大小 |
|---|---:|---:|
| FP32 | 4 | 4 GB |
| BF16 / FP16 | 2 | 2 GB |
| FP8 / INT8 | 1 | 1 GB（仅理论原始数据，不含缩放元数据） |

按上述十进制单位计算：

$$
1\text{B}\times\frac{16}{8}=2\text{ GB},\qquad
1\text{B}\times\frac{32}{8}=4\text{ GB}
$$

### 3.1 Adam 的 `m`、`v` 为什么是 8 GB

Adam 为每个参数维护一阶矩 `m` 和二阶矩 `v`。二者若均为 FP32：

$$
2\text{ states}\times1\text{B}\times\frac{32}{8}=8\text{ GB}
$$

对当前梯度 `gₜ`，Adam 的核心状态更新为：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

| 状态 | 数学意义 | 直觉作用 |
|---|---|---|
| `m` / `exp_avg` | 梯度的一阶矩指数滑动平均 | 平滑梯度方向，形成类似 Momentum 的惯性 |
| `v` / `exp_avg_sq` | 梯度平方的二阶原始矩指数滑动平均 | 估计每个参数的梯度尺度，自适应调节步长 |

`v` 经常被简称为“方差”，但严格说它是 **未中心化二阶矩**，因为没有减去梯度均值。两个状态从 0 初始化，早期会偏小，所以 Adam 使用 Bias Correction：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

忽略 Weight Decay 时，参数更新近似为：

$$
\theta_t=\theta_{t-1}-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

因此 `m` 提供主要更新方向，`sqrt(v)` 按历史梯度幅度归一化每个参数的步长，`ε` 防止除零。AdamW 则把 Weight Decay 从梯度更新中解耦。`m`、`v` 都与参数形状相同；若均用 FP32，每个参数额外需要 `4+4=8 Byte`。

但 **8 GB 只是 `m+v`**。一种常见的 BF16 混合精度 Adam 账本是：

| 状态 | dtype | 1B 大小 |
|---|---|---:|
| 低精度模型参数 | BF16 | 2 GB |
| 梯度 | BF16 | 2 GB |
| FP32 master parameters | FP32 | 4 GB |
| Adam `m` | FP32 | 4 GB |
| Adam `v` | FP32 | 4 GB |
| 合计 | — | **16 GB** |

如果梯度保留 FP32，合计变为 18 GB；如果优化器不维护独立 FP32 master copy，则可能少 4 GB。不同框架、优化器和 FSDP 策略并不保证恰好 16 Byte/parameter，因此估算前应确认真实 state dtype。

### 3.2 可运行的显存估算器

```python
def model_state_gb(
    params_billion: float,
    world_size: int,
    param_bytes=2,
    grad_bytes=2,
    master_bytes=4,
    adam_m_bytes=4,
    adam_v_bytes=4,
):
    """只估算持久模型状态；不含激活、临时张量、通信 buffer 和碎片。"""
    import math
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    values = (params_billion, param_bytes, grad_bytes, master_bytes, adam_m_bytes, adam_v_bytes)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("parameter count and byte sizes must be finite and nonnegative")
    n = params_billion * 1e9
    P = n * param_bytes / 1e9
    G = n * grad_bytes / 1e9
    O = n * (master_bytes + adam_m_bytes + adam_v_bytes) / 1e9
    K = world_size
    return {
        "DDP": P + G + O,
        "ZeRO-1": P + G + O / K,
        "ZeRO-2": P + (G + O) / K,
        "ZeRO-3/FULL_SHARD lower bound": (P + G + O) / K,
    }


for mode, gb in model_state_gb(1, world_size=8).items():
    print(f"{mode:30s} {gb:5.2f} GB/GPU")
```

在上述假设和 8 张 GPU 下，理论持久状态约为 DDP 16 GB、ZeRO-1 5.5 GB、ZeRO-2 3.75 GB、ZeRO-3 2 GB/GPU。ZeRO-3/FSDP 计算时仍会临时 All-Gather 当前模块参数，所以 2 GB 不是实际峰值。

下面的完整代码把上述结果画成堆叠柱状图。它与正文使用同一组字节假设，运行后生成 `compute_docs/assets/zero-memory-comparison.png`：

```bash
python -m pip install matplotlib numpy
```

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

world_size = 8
P, G, O = 2.0, 2.0, 12.0  # 1B: BF16 参数、BF16 梯度、FP32 master+m+v
labels = ["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
parameters = np.array([P, P, P, P / world_size])
gradients = np.array([G, G, G / world_size, G / world_size])
optimizer = np.array([O, O / world_size, O / world_size, O / world_size])

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(labels, parameters, label="Parameters", color="#3b82f6")
ax.bar(labels, gradients, bottom=parameters, label="Gradients", color="#f97360")
ax.bar(
    labels,
    optimizer,
    bottom=parameters + gradients,
    label="Optimizer states",
    color="#8b5cf6",
)
totals = parameters + gradients + optimizer
for index, total in enumerate(totals):
    ax.text(index, total + 0.25, f"{total:.2f} GB", ha="center")
ax.set_ylabel("Persistent model-state memory per GPU (GB)")
ax.set_title("1B parameters, BF16 params/grads, FP32 Adam, 8 GPUs")
ax.legend()
ax.set_ylim(0, totals.max() * 1.12)
fig.tight_layout()

output = Path("compute_docs/assets/zero-memory-comparison.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180, bbox_inches="tight")
print(f"saved: {output.resolve()}")
```

<figure class="article-figure">
  {{< post-image src="assets/zero-memory-comparison.png" alt="1B 模型在 DDP 与 ZeRO 各阶段的每卡持久状态显存" >}}
  <figcaption>
    <span class="article-figure__number">图 1</span>
    <span class="article-figure__text">8 卡时 P/G/O 的理论常驻显存；不包含激活、临时张量、通信缓冲和参数 All-Gather 峰值。</span>
  </figcaption>
</figure>

### 3.3 理论下限、稳态与峰值不是同一个数

- **理论下限**：只把 `P/G/O` 按公式分片后的总和。
- **稳态显存**：加上长期激活缓存、通信 Bucket、CUDA Context 和分配器保留块。
- **峰值显存**：某一瞬间同时存在旧张量、新张量、预取参数和算子 workspace。

容量规划要看峰值，优化效果要同时报告 `max_memory_allocated` 与 `max_memory_reserved`。只用 `nvidia-smi` 的单个时刻或只报 ZeRO 理论公式，都不足以说明任务能否稳定运行。

## 4. DP 与 DDP

### 4.1 Data Parallel（DP）

经典数据并行的逻辑是：每个设备拿到不同 mini-batch 分片，运行相同模型，然后同步梯度。PyTorch `nn.DataParallel` 是单进程多线程实现，通常由主设备负责 scatter/gather，容易形成单进程和主 GPU 瓶颈；多 GPU 训练一般优先使用 DDP。

### 4.2 DistributedDataParallel（DDP）

DDP 通常是一张 GPU 对应一个进程和一个 Rank：

1. 每个 Rank 保存完整参数、梯度和优化器状态。
2. DistributedSampler 让不同 Rank 读取不同数据。
3. 各 Rank 独立前向、反向。
4. 梯度 Bucket 就绪后执行 All-Reduce。
5. 每个 Rank 用相同的同步梯度执行相同 optimizer step。

DDP 不会在每一步广播新参数；参数副本之所以保持一致，是因为初始参数一致、归约后的梯度一致、优化器更新也一致。

```python
# torchrun --standalone --nproc_per_node=8 train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().to(local_rank)  # 替换为实际模型
model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# DataLoader 需要 DistributedSampler；每个 Rank 的 batch 不同
for inputs, targets in loader:
    inputs, targets = inputs.to(local_rank), targets.to(local_rank)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(inputs), targets)
    loss.backward()   # DDP hook 按 bucket 触发梯度 All-Reduce
    optimizer.step()

dist.destroy_process_group()
```

全局 Batch Size 通常为：

$$
B_{global}=B_{micro}\times K\times N_{accum}
$$

其中 `K` 是数据并行 Rank 数，`N_accum` 是梯度累积步数。改变 GPU 数量却不调整学习率、Scheduler 或训练 token 数，训练行为可能随之改变。

梯度累积时，若每个 micro-batch 都同步梯度，会产生不必要的 All-Reduce。DDP 可在非最后一个 micro-step 使用 `no_sync()`，只在累积边界同步；但此时完整未同步梯度会继续占据本地显存，并且 Loss 缩放必须与累积步数保持一致。

## 5. NCCL 与集合通信

NCCL（NVIDIA Collective Communications Library）是面向 NVIDIA GPU 的集合通信库。它负责高效执行 All-Reduce、All-Gather、Reduce-Scatter、Broadcast 等原语；DDP/FSDP 是训练策略，NCCL 是它们常用的通信后端，不要把两者视作同一层组件。

<figure class="article-figure">
  {{< post-image src="assets/collective-communications.webp" alt="All-Reduce、Reduce-Scatter 与 All-Gather" >}}
  <figcaption>
    <span class="article-figure__number">图 2</span>
    <span class="article-figure__text">All-Reduce 让所有 Rank 得到相同归约结果；Reduce-Scatter 只保留各自结果分片；All-Gather 将各分片拼回完整张量。</span>
  </figcaption>
</figure>

### 5.1 All-Reduce：求和还是平均

All-Reduce 本身执行 `SUM/MAX/MIN` 等归约。DDP 的最终梯度语义通常是跨 Rank 平均，但“求和后除以 world size”的具体位置由框架实现处理。假设两张 GPU 的局部梯度分别为 `g₀`、`g₁`：

$$
g=\frac{g_0+g_1}{2}
$$

不要在 DDP 已经平均后再次手动除以 `world_size`。还要区分 loss 在本地 batch 上是 `mean` 还是 `sum`，否则很容易得到多除或少除的梯度尺度。

### 5.2 三个原语的所有权

| 原语 | 每个 Rank 输入 | 每个 Rank 输出 | 典型用途 |
|---|---|---|---|
| All-Reduce | 完整张量 | 相同的完整归约张量 | DDP 梯度同步 |
| Reduce-Scatter | 完整张量 | 不同的归约分片 | ZeRO/FSDP 梯度归约并分片 |
| All-Gather | 不同分片 | 相同的完整拼接张量 | FSDP 计算前还原参数 |
| All-to-All | 发往各 Rank 的不同分块 | 来自各 Rank 的不同分块 | MoE Token 路由 |

#### Broadcast：一份数据复制给所有 Rank

<figure class="article-figure">
  {{< post-image src="assets/collective-broadcast.png" alt="四 Rank Broadcast 前后数据所有权" >}}
  <figcaption>
    <span class="article-figure__number">图 3</span>
    <span class="article-figure__text">Broadcast 只有 source Rank 提供有效输入，操作后每个 Rank 都得到相同的 X；它不做求和。</span>
  </figcaption>
</figure>

常用于同步初始化参数、配置或控制信息。`src=0` 指通信组中的 Rank 0，不一定等于物理 GPU 0；使用子 Process Group 时要特别核对 Rank 空间。

#### All-Gather：分片拼成完整张量

<figure class="article-figure">
  {{< post-image src="assets/collective-all-gather.png" alt="四 Rank All-Gather 前后数据所有权" >}}
  <figcaption>
    <span class="article-figure__number">图 4</span>
    <span class="article-figure__text">Rank 0～3 分别提供 A/B/C/D，所有 Rank 最终按 Rank 顺序得到完整的 `[A|B|C|D]`。</span>
  </figcaption>
</figure>

All-Gather 不做数值归约，只做收集和拼接。若每 Rank 输入 `M/K`，操作后每 Rank 输出约 `M`，因此会产生显著的瞬时完整参数峰值。

#### Reduce-Scatter：先归约，再分发结果分片

<figure class="article-figure">
  {{< post-image src="assets/collective-reduce-scatter.png" alt="四 Rank Reduce-Scatter SUM 数值示例" >}}
  <figcaption>
    <span class="article-figure__number">图 5</span>
    <span class="article-figure__text">四个向量先逐元素求和为 `[1111,2222,3333,4444]`，随后 Rank 0～3 各保留一个不同分片。</span>
  </figcaption>
</figure>

Reduce-Scatter 同时完成 Reduction 和 Sharding。它非常适合 FSDP/ZeRO 梯度：每个 Rank 不需要保留完整归约梯度，只保留与本地参数 shard 对应的部分。

#### All-Reduce：每个 Rank 都得到完整归约结果

<figure class="article-figure">
  {{< post-image src="assets/collective-all-reduce.png" alt="四 Rank All-Reduce SUM 数值示例" >}}
  <figcaption>
    <span class="article-figure__number">图 6</span>
    <span class="article-figure__text">四个输入逐元素求和后，每个 Rank 都获得相同的完整结果 `[1111,2222,3333,4444]`。</span>
  </figcaption>
</figure>

与 Reduce-Scatter 的区别不是“是否求和”，而是结果所有权：All-Reduce 在每个 Rank 保留完整结果，Reduce-Scatter 只在每个 Rank 保留不同结果分片。

### 5.3 PyTorch Collective 调用形状

```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
world = dist.get_world_size()
device = torch.device("cuda", torch.cuda.current_device())

# Broadcast：仅 src 的初始值会被保留
x = torch.tensor([42.0 if rank == 0 else 0.0], device=device)
dist.broadcast(x, src=0)

# All-Reduce SUM：原地把每个 Rank 的 local 相加
local = torch.tensor([float(rank + 1)], device=device)
dist.all_reduce(local, op=dist.ReduceOp.SUM)
average = local / world  # 需要平均语义时再显式除；DDP 内部会处理自身语义

# All-Gather：每 Rank 输入 [N]，每 Rank 输出 [world*N]
shard = torch.full((2,), rank, dtype=torch.float32, device=device)
gathered = torch.empty(world * shard.numel(), device=device)
dist.all_gather_into_tensor(gathered, shard)

# Reduce-Scatter：每 Rank 输入 [world*N]，每 Rank 输出 [N]
full = torch.arange(world * 2, dtype=torch.float32, device=device) + rank
reduced_shard = torch.empty(2, device=device)
dist.reduce_scatter_tensor(reduced_shard, full, op=dist.ReduceOp.SUM)
```

所有 Rank 必须以相同顺序进入 Collective，并满足 API 要求的 shape、dtype 和设备约束。`async_op=True` 只表示调用可异步返回；真正安全复用输出前仍要正确等待 Work，并确认计算流依赖。

在相同分块语义下，可以把 All-Reduce 理解为 `Reduce-Scatter + All-Gather`。Ring 算法下，每 Rank 的近似传输量分别为：

$$
V_{RS}\approx\frac{K-1}{K}M,\quad
V_{AG}\approx\frac{K-1}{K}M,\quad
V_{AR}\approx2\frac{K-1}{K}M
$$

这是用于直觉估算的 Ring 模型；真实 NCCL 会根据拓扑、消息大小和版本选择 Ring、Tree 等算法。

### 5.4 延迟—带宽模型

一次 Collective 的时间可以粗略理解为：

$$
T_{comm}\approx n_{round}\times\alpha+\frac{V}{B_{effective}}
$$

`α` 是每轮通信的启动延迟，`V` 是传输量，`B_effective` 是考虑拓扑和竞争后的有效带宽。小 Tensor 往往受延迟支配，因此需要 Bucket 合并；大 Tensor 更受带宽支配，继续合并不一定有益。跨节点通信还可能经过 PCIe、NIC、交换机，多层拓扑中最慢链路决定暴露时间。

判断瓶颈时应分别测量节点内与跨节点：若单机 8 卡扩展良好、多机骤降，优先检查 NIC 带宽、GPU Direct RDMA、Rank 到 NIC/GPU 的亲和性和跨机 Process Group，而不是先改模型算子。

### 5.5 用 1B 梯度估算通信下限

假设 1B 模型的梯度以 BF16 通信，完整梯度 `M=2 GB`，8 Rank Ring All-Reduce 每 Rank 近似传输：

$$
V_{AR}\approx2\times\frac{7}{8}\times2=3.5\text{ GB}
$$

若实测有效带宽为 `25 GB/s`，完全不重叠时仅大消息数据传输下限约为 `3.5/25=0.14 s`，还没有计算启动延迟和竞争。若网卡标称 `200 Gb/s`，换算理论上限是 `25 GB/s`，因为 8 bit = 1 Byte；协议、拓扑与并发会让实际带宽更低。不要混用 `Gb/s` 与 `GB/s`。

这只是单次梯度同步估算。ZeRO-3/FSDP 还要考虑逐模块参数 All-Gather；其总量、次数和暴露比例取决于 wrap、reshard 和 prefetch，而不是只由模型参数量决定。

## 6. ZeRO-1、ZeRO-2、ZeRO-3

ZeRO 的核心是消除数据并行 Rank 之间重复保存的模型状态：

<figure class="article-figure">
  {{< post-image src="assets/ddp-zero-sharding.webp" alt="DDP 与 ZeRO 三阶段模型状态分片" >}}
  <figcaption>
    <span class="article-figure__number">图 7</span>
    <span class="article-figure__text">ZeRO-1 分优化器，ZeRO-2 再分梯度，ZeRO-3 进一步分参数；图中不包含激活和临时峰值。</span>
  </figcaption>
</figure>

| 策略 | 参数 P | 梯度 G | 优化器 O | 主要新增通信/管理 |
|---|---|---|---|---|
| DDP / ZeRO-0 | 完整 | 完整 | 完整 | 梯度 All-Reduce |
| ZeRO-1 | 完整 | 完整 | `1/K` | 更新后同步参数分片 |
| ZeRO-2 | 完整 | `1/K` | `1/K` | 梯度 Reduce-Scatter |
| ZeRO-3 | `1/K` 常驻 | `1/K` | `1/K` | 计算前 All-Gather 参数，之后重新分片 |

### 6.1 Stage 3 的常见叫法

规范写法是 **ZeRO Stage 3** 或 **ZeRO-3**，而不是 “ZeRO state-3”。在资料和代码中还会看到这些相关名称：

| 名称 | 所属生态 | 与 ZeRO-3 的关系 |
|---|---|---|
| ZeRO Stage 3 / ZeRO-3 | DeepSpeed | 官方 Stage 名称；分片 P/G/O |
| Full Parameter Sharding | 通用描述 | 强调参数也被分片，不特指某个库 |
| Fully Sharded Data Parallel | 通用概念 / PyTorch | 数据并行 Rank 间完整分片模型状态 |
| `FULL_SHARD` | PyTorch FSDP1 | 与 ZeRO-3 核心思想对应的策略名 |
| `fully_shard` | PyTorch FSDP2 | Composable API；实现和状态表示与 FSDP1 不同 |
| ZeRO-Offload | DeepSpeed | 将部分状态或计算卸载到 CPU，不是 ZeRO-4 |
| ZeRO-Infinity | DeepSpeed | 将 CPU/NVMe 纳入异构内存层次，不是新 Stage |

可以说“FSDP Full Shard 与 ZeRO-3 属于同类全分片策略”，但不应说两者是同一个实现。它们的预取、reshard、状态字典、初始化方式和配置项不同。

### 6.2 如何选择 Stage

- 模型状态能放下且追求吞吐：先用 DDP。
- Adam 状态造成 OOM：ZeRO-1 往往是最低通信代价的切入点。
- 参数能放下，但参数加完整梯度放不下：ZeRO-2。
- 单张 GPU 连完整参数或整体模型状态都放不下：ZeRO-3/FSDP Full Shard。
- GPU 显存仍不足：再考虑 CPU/NVMe Offload，但会引入 PCIe、内存或存储带宽瓶颈。

Stage 越高不代表训练一定越快。更强分片降低显存冗余，却增加 Collective、预取、同步和状态管理压力。

### 6.3 DeepSpeed ZeRO-3 配置骨架

下面是结构完整的起点配置，不是适用于所有集群的最佳参数：

```json
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "bf16": {"enabled": true},
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": 0.0003, "betas": [0.9, 0.95]}
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 100000
  }
}
```

Bucket 数值按元素计还是按 Byte 计要以所用 DeepSpeed 版本和字段文档为准。更大的 Bucket 可能提高带宽利用率，却增加峰值显存并推迟通信启动；`overlap_comm=true` 也会为了并发保留额外 Buffer。先使用保守配置跑通，再根据 profiler 调整。

## 7. FSDP：如何在计算时临时还原参数

PyTorch FSDP 的 Full Shard 与 ZeRO-3 在核心思想上相近：常驻时参数、梯度、优化器状态都分片；某个模块计算前 All-Gather 其完整参数，计算后释放完整副本；反向得到完整局部梯度后用 Reduce-Scatter 归约并只保留本 Rank 分片。

```text
常驻参数 shard
  → All-Gather 当前模块完整参数
  → Forward 当前模块
  → Reshard / 释放完整参数
  → Backward 前再次 All-Gather（取决于策略）
  → Backward 当前模块
  → Reduce-Scatter 梯度
  → 本 Rank optimizer 更新自己的 shard
```

### 7.1 FSDP2 最小结构

当前 PyTorch 文档推荐 FSDP2 的 composable `fully_shard` API；旧 `FullyShardedDataParallel` 类通常称为 FSDP1。下面展示结构而非完整训练脚本：

```python
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = Transformer(config).to(local_rank)
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

# 先从叶子层向根模块应用，形成可预取、可重叠的通信组
for block in model.layers:
    fully_shard(block, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

实际项目还需要处理 meta-device 初始化、Checkpoint、共享参数、CPU Offload、梯度裁剪和 Hybrid Sharding。Wrap 粒度太大会形成巨型阻塞 All-Gather，太小则产生大量延迟敏感的小通信。

### 7.2 FSDP 与 ZeRO 的关系

| 概念 | DeepSpeed ZeRO | PyTorch FSDP |
|---|---|---|
| 只分优化器 | ZeRO-1 | 没有完全一一对应的常用 Full Shard 策略 |
| 分梯度和优化器 | ZeRO-2 | `SHARD_GRAD_OP` / 对应 FSDP2 reshard 策略 |
| 分参数、梯度、优化器 | ZeRO-3 | `FULL_SHARD` / FSDP2 `fully_shard` |
| Offload | ZeRO-Offload / Infinity | CPU Offload 策略 |

它们不能只按 Stage 名字比较速度；还要对齐 wrap 粒度、prefetch、reshard、混合精度、offload、checkpoint 和网络拓扑。

### 7.3 分布式 Checkpoint

Full Shard 下每个 Rank 只持有状态分片。保存时若先把完整模型和 Adam 状态聚合到 Rank 0，可能在训练已经能运行后反而于 Checkpoint 阶段 OOM，并形成单机内存和存储带宽瓶颈。

优先采用 Sharded State Dict 或 Distributed Checkpoint：各 Rank 并行写自己的分片，并保存模型配置、优化器、Scheduler、global step、随机数状态和数据游标。恢复时还要确认是否支持改变 `world_size` 重分片。完整权重通常只在导出或推理部署阶段离线合并。

Checkpoint 是 Collective 工作流的一部分：所有 Rank 必须以一致顺序参与。只让 Rank 0 进入一个内部包含 All-Gather 的保存函数，其他 Rank 跳过，可能导致永久等待。

### 7.4 初始化阶段也可能 OOM

即使稳态分片后能放下，若先在每张 GPU 构造完整模型、再调用 `fully_shard`，仍可能在包装前 OOM。大模型通常采用 meta device 或 deferred initialization：先创建不分配真实存储的参数结构，再按 Rank materialize 本地分片并加载 Sharded Checkpoint。

同样要警惕 optimizer state 的惰性初始化：Adam 的 `m/v` 常在第一次 `optimizer.step()` 才创建，所以仅成功完成 Forward/Backward 不能证明训练显存足够。容量测试至少要跑完若干个完整 step，并包含验证和 Checkpoint。

## 8. BF16、FP32 与混合精度

“LLM 一般 BF16、Vision 一般 FP32”过于绝对。现代 LLM 常用 BF16 计算是因为动态范围接近 FP32 且 Tensor Core 吞吐高；视觉模型也大量使用 FP16/BF16/TF32 混合精度，只是某些归一化、损失、归约或数值敏感算子可能保留 FP32。

### 8.1 前向 BF16，不代表梯度一定 FP32

需要分别指定：

| 对象 | 可能 dtype | 谁决定 |
|---|---|---|
| 参数存储/master 参数 | BF16 或 FP32 | 模型加载、优化器、分片框架 |
| Forward/Backward 计算参数 | BF16/FP16/FP32 | autocast 或 FSDP `param_dtype` |
| 激活 | 混合 | autocast 的算子规则 |
| 局部梯度 | 常跟计算/参数 dtype 相关 | Autograd 与参数配置 |
| 梯度通信 | BF16 或 FP32 | DDP/FSDP `reduce_dtype` 等配置 |
| Optimizer `m/v` | 常见 FP32，也可低精度 | 优化器实现 |

例如 FSDP 可让 `param_dtype=torch.bfloat16`，同时令 `reduce_dtype=torch.float32`；也可以在 BF16 中归约以节省带宽。部分实现会在 optimizer step 前把低精度梯度转回 FP32，但这不是所有训练栈的固定规律。

### 8.2 模型配置里的 dtype 不等于完整训练策略

以 Qwen 等模型为例，配置中的 `torch_dtype` 常表示权重保存或默认加载 dtype。`from_pretrained(..., torch_dtype=...)`、autocast、FSDP MixedPrecision、优化器状态和硬件支持共同决定运行时 dtype。BF16 权重“能加载”不等于所有算子都会用 BF16，也不保证该训练配置数值稳定。

FP16 的指数范围较小，训练常需要 GradScaler；BF16 动态范围接近 FP32，通常不需要 Loss Scaling，但精度尾数更少。归一化、Softmax、Loss、梯度范数和某些 Reduction 使用 FP32 往往更稳。

### 8.3 不猜 dtype，直接检查

```python
import torch

def print_training_dtypes(model, optimizer):
    parameter = next(p for p in model.parameters() if p.requires_grad)
    print("parameter dtype:", parameter.dtype)
    print("gradient dtype :", None if parameter.grad is None else parameter.grad.dtype)
    state = optimizer.state.get(parameter, {})
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            print(f"optimizer {name}: shape={tuple(value.shape)}, dtype={value.dtype}")


# 应在 loss.backward() 后查看 gradient；在 optimizer.step() 后查看惰性创建的 m/v
print_training_dtypes(model, optimizer)
```

还可在 Forward Hook 中打印关键激活 dtype，并用 profiler 查看算子实际 kernel。配置文件、Checkpoint dtype 和 Tensor Core 实际计算精度是三个不同层面。

## 9. Activation Checkpointing

Activation Checkpointing（也叫 Gradient Checkpointing）不保存选定区域的全部中间激活，反向时重新执行一次前向来恢复它们：

$$
\text{更少激活显存}\quad\Longleftrightarrow\quad\text{更多重计算}
$$

它主要降低 `M_A`，不会分片参数、梯度或优化器状态。对长序列 Transformer，激活常随 `B × S × L × H` 增长，Checkpointing 可能比 ZeRO-1 更直接；若模型状态本身放不下，则仍需要 ZeRO/FSDP。

朴素 Attention 若显式保留概率矩阵，还可能包含近似 `B × heads × S²` 的中间量；FlashAttention 类内核通过分块避免物化完整矩阵，降低这部分峰值。Checkpointing 和 FlashAttention 优化的是不同来源，可以同时使用。

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for block in self.layers:
        # use_reentrant=False 是现代 PyTorch 常用模式
        x = checkpoint(block, x, use_reentrant=False)
    return x
```

注意随机数状态、Dropout、自定义 Autograd、原地修改与包含非 Tensor 参数的函数。应以实际吞吐和峰值显存评估粒度，而不是盲目 checkpoint 每个小算子。

## 10. 让计算覆盖通信，而不是让 GPU 空等

训练优化目标不是简单要求“总计算时间一定大于总通信时间”，而是让每个通信 Bucket 尽早启动，并被之后仍在进行的计算覆盖：

$$
T_{step}\approx T_{compute}+T_{exposed\_comm}
$$

若某个 Bucket 的通信时间小于其后可并行的计算窗口，它大部分可以被隐藏；最后一个 Bucket、过大的 Collective 或慢网络通常形成 exposed communication tail。

<figure class="article-figure">
  {{< post-image src="assets/compute-communication-overlap.webp" alt="串行通信与计算通信重叠时间线" >}}
  <figcaption>
    <span class="article-figure__number">图 8</span>
    <span class="article-figure__text">Bucket 就绪后立即通信可与后续反向计算重叠；真正增加 Step Time 的主要是未被覆盖的通信尾部。</span>
  </figcaption>
</figure>

### 10.1 优先优化顺序

1. **先保证计算够大**：过小 micro-batch、短序列或大量小 kernel 会降低 GPU 利用率。
2. **合理 Bucket/Wrap 粒度**：太大则启动晚，太小则延迟和 launch 开销高。
3. **避免隐式同步**：训练热路径中的 `.item()`、频繁日志、CPU/GPU 往返会造成空等。
4. **利用拓扑**：节点内 NVLink/NVSwitch 与跨节点 InfiniBand 带宽差异巨大。
5. **预取与重叠**：FSDP All-Gather、Reduce-Scatter 与相邻模块计算重叠。
6. **再考虑压缩/低精度通信**：降低带宽但可能影响收敛或增加转换开销。

推荐用 PyTorch Profiler 或 Nsight Systems 查看 Compute、NCCL kernel 和空隙时间。只看 GPU Utilization 百分比无法判断是通信、数据加载还是 CPU launch 瓶颈。

### 10.2 吞吐与利用率指标

- **Tokens/s 或 Samples/s**：最直接的端到端吞吐，必须同时记录 Global Batch 和序列长度。
- **Step Time**：拆分数据等待、前向、反向、Collective、optimizer 和 checkpoint。
- **MFU（Model FLOPs Utilization）**：模型理论 FLOPs 相对硬件峰值的比例；口径必须一致。
- **Scaling Efficiency**：`K` 卡吞吐除以单卡吞吐的 `K` 倍，衡量扩卡收益。

显存利用率高不等于算力利用率高，GPU Utilization 高也不等于有效模型计算高。长时间运行的 NCCL kernel、数据搬运或低效小 kernel 都可能让监控显示“GPU 很忙”。

## 11. 显存分配器、碎片与排错

PyTorch CUDA caching allocator 会保留已经申请的显存以便复用：

- `memory_allocated()`：活跃 Tensor 真正占用的显存。
- `memory_reserved()`：分配器向 CUDA 保留的显存，包含可复用空闲块。
- `nvidia-smi`：进程占用视角，通常更接近 reserved 而非 allocated。

```python
import torch

torch.cuda.reset_peak_memory_stats()
# 执行一个完整训练 step
allocated = torch.cuda.max_memory_allocated() / 2**30
reserved = torch.cuda.max_memory_reserved() / 2**30
print(f"peak allocated={allocated:.2f} GiB")
print(f"peak reserved ={reserved:.2f} GiB")
print(torch.cuda.memory_summary(abbreviated=True))
```

碎片的典型现象是 reserved 很高、allocated 较低，但无法找到满足新大张量的连续可用块。处理顺序：

1. 找到是否有 Tensor 被列表、闭包、日志或未 detach 的 Loss 意外持有。
2. 稳定输入 shape，减少频繁变化的 Batch/Sequence Length。
3. 使用 `zero_grad(set_to_none=True)`，避免不必要的梯度清零写入和存储。
4. 调整 FSDP 预取、All-Gather 并发和 wrap 粒度，降低瞬时峰值。
5. 通过 `memory_snapshot()` 分析真实分配历史。
6. 对动态 shape 工作负载评估 `PYTORCH_ALLOC_CONF=expandable_segments:True`。

`torch.cuda.empty_cache()` 只能释放缓存中的空闲块供其他进程使用，不能释放仍被 Tensor 引用的活跃显存，也通常不会让当前程序获得额外可用 Tensor 显存。

### 11.1 OOM 定位表

| 现象 | 优先检查 |
|---|---|
| Forward 立即 OOM | 参数副本、输入分辨率/序列长度、All-Gather 峰值 |
| Backward 中 OOM | 激活、梯度、Checkpointing、通信 Bucket |
| optimizer.step OOM | Adam 状态首次惰性初始化、FP32 master 参数 |
| 第二步开始增长 | 图被意外保留、Loss 未 `.detach()`、缓存列表 |
| 只有某个 Rank OOM | 数据长度不均、Rank 0 额外日志/验证/保存完整权重 |
| 多机挂起而非 OOM | Collective 次序/shape 不一致、某 Rank 提前异常、网络问题 |

调试挂起可启用 `TORCH_DISTRIBUTED_DEBUG=DETAIL`，并检查所有 Rank 是否以相同顺序、相同 shape 和 dtype 进入 Collective。NCCL Collective 不是某个 Rank 可以随意跳过的普通函数。

## 12. 如何选择 DDP、ZeRO 和 FSDP

| 情况 | 首选起点 | 原因 |
|---|---|---|
| 完整训练状态轻松放入单卡 | DDP | 路径简单、吞吐通常最好 |
| Adam 状态是主要压力 | ZeRO-1 | 以较小改动分片最大状态项 |
| 参数能放下，梯度+优化器放不下 | ZeRO-2 / SHARD_GRAD_OP | 保留完整计算参数，减少状态冗余 |
| 完整参数本身接近或超过单卡容量 | ZeRO-3 / FSDP Full Shard | 参数按模块临时 All-Gather |
| 节点内快、节点间慢 | Hybrid Shard | 节点内分片、节点间复制以控制跨机通信 |
| 激活远大于模型状态 | Checkpointing、FlashAttention、序列并行 | 仅分片 P/G/O 不解决主要矛盾 |
| 单层参数本身过大 | Tensor Parallel + DP/FSDP | FSDP 临时完整层也可能 OOM |

### 12.1 上线前配置清单

- [ ] 按真实 dtype 分别统计参数、梯度、master 参数、`m/v`。
- [ ] 估算理论持久状态后，再预留激活、临时张量、Collective 和碎片空间。
- [ ] 确认 Global Batch、梯度累积和 Scheduler 的语义。
- [ ] 确认各 Rank 数据不同但 step 数一致，Sampler 每个 epoch 正确设种子。
- [ ] 核对 `param_dtype`、`reduce_dtype`、optimizer state dtype，而不是只看模型配置文件。
- [ ] 用单 GPU 小模型验证 Loss，再扩到 DDP，最后引入分片。
- [ ] 用固定 batch 比较吞吐、峰值 allocated/reserved 和 exposed communication。
- [ ] 测试分布式 Checkpoint 保存与恢复；不要只在 Rank 0 聚合超大完整状态后才发现 OOM。
- [ ] 为一个 batch 做过拟合测试，并验证扩卡前后梯度和 Loss 尺度。

### 12.2 当数据并行分片仍然不够

ZeRO/FSDP 属于数据并行维度的状态分片。更大的模型通常把多种并行方式组成二维或三维 Device Mesh：

| 并行维度 | 切分对象 | 主要通信 | 解决的问题 |
|---|---|---|---|
| Tensor Parallel（TP） | 单层矩阵/Head | All-Reduce、Reduce-Scatter、All-Gather | 单层参数或计算无法放入单卡 |
| Pipeline Parallel（PP） | 连续层/Stage | 点对点发送激活与梯度 | 模型深度与参数容量 |
| Context/Sequence Parallel（CP/SP） | 序列维或激活 | Ring/P2P/Collective | 超长序列激活和 Attention |
| Expert Parallel（EP） | MoE Experts | All-to-All | 专家参数与稀疏路由 |
| Data Parallel / FSDP | 数据与模型状态 | AR 或 RS+AG | 提升吞吐并减少副本状态 |

例如可以在节点内做 TP、节点间做 FSDP/DP。并行维度越多，通信组、Checkpoint 和性能调优越复杂；只有确认单一策略的容量或吞吐瓶颈后再叠加下一维。

### 12.3 从单卡扩到多机的推荐顺序

1. **单 GPU 正确性**：固定 Seed，小数据过拟合，保存基准 Loss、梯度范数和吞吐。
2. **单机 DDP**：保持 Global Batch 等价，验证参数更新与单卡数值接近。
3. **单机分片**：只改变 ZeRO/FSDP 策略，确认峰值显存下降且 Checkpoint 可恢复。
4. **多机小规模**：先跑短任务，检查所有 Rank step 数、NCCL 错误和扩展效率。
5. **加入混合精度与 Checkpointing**：每次只改变一个变量，记录数值与性能。
6. **最后调 Bucket/Prefetch/Offload**：用时间线证据优化 exposed communication。

每个阶段都记录同一组指标：Global Batch、Tokens/s、Step Time、Loss、Gradient Norm、峰值 allocated/reserved、网络吞吐和 Checkpoint 时间。否则扩展后即使“跑得起来”，也无法判断性能或收敛变化来自哪里。

## 13. 关键结论速查

```text
DDP:     P + G + O 全复制；梯度 All-Reduce
ZeRO-1:  P、G 全复制；O 分片
ZeRO-2:  P 全复制；G、O 分片
ZeRO-3:  P、G、O 分片；计算前按模块 All-Gather P
FSDP:    Full Shard 与 ZeRO-3 核心思想相近；梯度 Reduce-Scatter
AC/GC:   主要减少激活，代价是反向时重算前向
NCCL:    通信后端，不是训练并行策略
```

最稳妥的优化原则是：**先测量显存由谁占用，再选择分片对象；先定位暴露通信，再谈重叠；先保证正确性，再追求利用率。**

## 14. 官方参考资料

- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch FSDP API](https://docs.pytorch.org/docs/stable/fsdp.html)
- [NVIDIA NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [DeepSpeed ZeRO Documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html)
- [PyTorch CUDA Memory Management](https://docs.pytorch.org/docs/stable/cuda.html#memory-management)


## 阅读自测与验收

- 按实际 dtype 分别列出参数、梯度、优化器状态和激活；把估算与同一训练阶段的峰值测量比较，不混用 GB 与 GiB。
- 保存并恢复一次小规模训练，检查步数、优化器状态及各 Rank 的一致性；只恢复权重不能证明训练可续跑。
