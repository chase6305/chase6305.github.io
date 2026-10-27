---
title: "视觉 Transformer 全景：从 ViT、Swin、CMT 到 DiT 与 SiT"
date: 2026-10-12
lastmod: 2026-10-14
draft: false
tags: ["Transformer", "Attention", "Computer Vision", "Diffusion Models", "Generative Models", "Pretraining", "VLM"]
categories: ["人工智能"]
authors: ["chase"]
summary: "从图像 patch 与注意力复杂度出发，比较 ViT、Swin、PVT、CMT 等视觉骨干，梳理 MAE、DINO、I-JEPA/V-JEPA、CLIP/SigLIP 等表征路线，介绍 SAM 系统，并区分 DiT、SiT 与视频 Transformer 的职责。"
math: true
toc: true
images: ["assets/vision-transformer-map.webp"]
imageZoom: true
description: "从图像 patch 与注意力复杂度出发，比较 ViT、Swin、PVT、CMT 等视觉骨干，梳理 MAE、DINO、I-JEPA/V-JEPA、CLIP/SigLIP 等表征路线，介绍 SAM 系统，并区分 DiT、SiT 与视频 Transformer 的职责。"
contentLanguage: "zh-CN"
reading_prerequisites: "神经网络、矩阵乘法和基本注意力机制；无需预先阅读相关论文"
reading_focus: "先区分模型架构、预训练目标和下游任务，再比较 token 数、多尺度特征、注意力范围、数据条件与实际计算代价。"
related_posts:
  - "/posts/ai/transformer-attention"
  - "/posts/ai/diffusion-models"
  - "/posts/ai/internvl-3-5"
---

“Vision Transformer”有时指一类视觉模型，有时特指 2020 年提出的 ViT；DiT 则是把 Transformer 用作扩散去噪器的模型家族。Swin、CMT、SiT 也常被放在同一张模型清单里，但它们解决的问题并不相同。

本文沿着三个问题梳理这条路线：**图像如何变成 Transformer 可处理的序列？如何让视觉模型兼顾局部细节、多尺度和计算效率？生成模型中的 DiT、SiT 又与分类骨干有什么关系？** 重点介绍代表性架构，不追求把每个变体都列成名词表。

> **读完后应能做到**：给定一个模型名，判断它主要处理分类、密集预测、视觉表征、多模态对齐还是图像生成；并根据 token 数、多尺度结构、注意力范围和训练目标解释它的主要取舍。

<figure class="article-figure" id="fig-vision-transformer-map">
  {{< post-image src="assets/vision-transformer-map.webp" alt="视觉 Transformer 的三个设计维度：ViT、Swin、CMT 骨干，MAE、DINO、CLIP 训练目标，分类、检测分割与图文系统；下方为带时间条件的 DiT/SiT 潜空间采样与 VAE 解码流程" >}}
  <figcaption>
    <span class="article-figure__number">图 1</span>
    <span class="article-figure__text">上半部分列出骨干、训练目标与任务系统三个可组合的维度，各列中的行不一一对应。下半部分展示常见潜空间生成：采样器反复调用 DiT/SiT 骨干并更新潜变量，完成采样后由 VAE 解码；图中把采样器更新合并到 Updated latent 箭头，索引只表示相邻采样步，不限定连续或离散时间实现。分类栏的小数只是输出形式示例，不是模型实测结果。</span>
  </figcaption>
</figure>

## 快速阅读路线

1. 第 1～3 节建立共同语言：模型角色、patch/token、位置编码、注意力复杂度和标准 ViT。
2. 第 4～6 节梳理预训练信号、层级视觉骨干，以及分类之外的检测与分割任务。
3. 第 7 节理解 CLIP、SigLIP 如何把视觉编码器接入图文对齐和视觉语言模型。
4. 第 8 节转向生成模型，区分 DiT 与 SiT，并了解文本到图像和视频模型中的 Transformer。
5. 第 9～13 节统一比较与选型，厘清常见误解，再按阅读路线和[自测](#13-自测)巩固。

### 快速目录

- [从图像到 token](#2-为什么图像能用-transformer-处理)
- [注意力模式与复杂度](#25-几种注意力模式的复杂度)
- [ViT 与训练数据效率](#3-vit简单通用但依赖训练条件)
- [预训练与表征学习](#4-数据效率与预训练改变训练信号不一定改变骨干)
- [层级骨干与卷积融合](#5-从全局注意力到层级视觉骨干)
- [下游视觉任务](#6-识别任务之外视觉骨干如何接入下游系统)
- [CLIP 与视觉语言模型](#7-clipsiglip-与视觉语言模型中的-vit)
- [DiT、SiT 与生成模型](#8-生成模型dit-与-sit-是另一条问题线)
- [比较、选型与误解](#9-一个统一的比较框架)
- [论文阅读顺序与自测](#12-建议的阅读顺序)
- [自测问题与参考答案](#13-自测)

## 1. 先给这些缩写分组

| 名称 | 主要角色 | 典型任务 | 先记住的设计点 |
|---|---|---|---|
| ViT | 视觉编码器/分类骨干 | 分类、迁移学习 | 固定大小 Patch Token，标准全局自注意力 |
| DeiT、BEiT、MAE、DINOv2/v3、I-/V-JEPA | ViT 或视觉编码器的训练/预训练路线 | 分类、通用表征、视频预测 | 架构可以相近，监督信号、数据目标和输出特征不同 |
| Swin、PVT、CMT、MaxViT | 层级或高效视觉骨干 | 检测、分割、分类 | 局部注意力、多阶段特征、卷积融合或多轴注意力 |
| EfficientViT-M / EfficientViT-L | 名称相近的两条独立效率路线 | 轻量识别、高分辨率密集预测 | 分别核对级联分组注意力、多尺度线性注意力及论文副标题 |
| Vim / VMamba / MambaVision | 状态空间视觉骨干或 Mamba-Transformer 混合骨干 | 高分辨率视觉表征 | 线性序列建模与注意力混合属于不同设计 |
| CLIP / SigLIP | 图文对齐预训练系统 | 检索、零样本分类、多模态编码 | 图像塔常用 ViT，但模型不等于 ViT 架构 |
| DiT、U-ViT、MMDiT | 生成模型中的去噪网络/骨干 | 图像、视频生成 | 输入随噪声时间步变化，输出用于去噪或预测速度 |
| SiT | DiT 骨干上的生成建模研究 | 扩散与流匹配式生成 | 系统比较插值路径、训练目标和采样选择 |
{.table-readable}

表中“模型架构”和“训练方法”是不同层次。ViT 可以用监督学习、掩码重建、蒸馏或图文对比目标训练；同样，扩散模型的去噪器可以采用 U-Net 或 Transformer。

## 2. 为什么图像能用 Transformer 处理

### 2.1 从二维图像到一串 token

输入图像通常写成 `x ∈ R^{H×W×C}`。ViT 把它按 `P×P` 的不重叠小块切分，将每块展平后用线性层投影到 `D` 维：

$$
N=\frac{H W}{P^2}, \qquad z_i = \operatorname{Flatten}(x_i) E + e_i
$$

`N` 是图像 patch 数，`E` 是可学习投影，`e_i` 是位置表示。比如 224×224 图像、Patch 16，会得到 14×14=196 个图像 token；如果再加一个分类 token，Transformer 序列长 197。这里的 token 是连续向量，不是 NLP 中的离散词表 ID。

这个计数假设图像高宽都能被 `P` 整除，且 patch 不重叠。实际预处理若遇到非整除尺寸，可能先缩放、裁剪或填充；重叠 patch 则会改变 token 数。

### 2.2 用张量形状核对 Patch Embedding

下面用 `Conv2d` 实现不重叠的 patch 投影。卷积核大小和步长都设为 `P`，因此每个卷积位置恰好读取一个 patch；这与“展平 patch 后乘同一个线性投影矩阵”在计算上等价。

```python
import torch
from torch import nn

batch, channels, height, width = 2, 3, 224, 224
patch_size, embed_dim = 16, 768
image = torch.randn(batch, channels, height, width)
patch_embed = nn.Conv2d(
    channels, embed_dim, kernel_size=patch_size, stride=patch_size
)

feature_map = patch_embed(image)                 # [2, 768, 14, 14]
tokens = feature_map.flatten(2).transpose(1, 2)  # [2, 196, 768]
print(tokens.shape)
```

`Conv2d` 输出仍是二维特征图 `[B,D,H/P,W/P]`；展平空间维并转置后，才成为 Transformer 常用的 `[B,N,D]` 序列。此处尚未加入位置编码、CLS token 或 Transformer Block，因此它只是 Patch Embedding 演示，不是可分类的完整 ViT。

Transformer Encoder 由多头自注意力、前馈网络、残差连接和归一化组成。分类时，早期 ViT 把可学习的 `[CLS]` token 与图像 token 一起输入，最后用其表示预测类别。也可以池化 patch 表示，不同实现并不都依赖 CLS。

单头自注意力的核心计算仍是：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

`Q`、`K`、`V` 分别是查询、键和值；图像 Transformer 的主要变化在于 token 如何构造、位置如何编码、允许哪些 token 交互，以及如何形成多尺度特征。计算细节可参阅站内 [Transformer Attention 指南](/posts/ai/transformer-attention)。

若批量大小为 `B`、注意力头数为 `h`、每头维度为 `d`，则常见张量形状是 `Q,K,V: [B,h,N,d]`，注意力分数为 `[B,h,N,N]`，输出回到 `[B,h,N,d]`。注意力乘法的计算量随 `N²D` 增长；Q/K/V 投影与前馈网络也有各自的成本，整层不只有注意力矩阵。

### 2.3 Patch 大小为何重要

在固定图像尺寸下，patch 边长减半，token 数约变为四倍。标准全局自注意力的主要注意力矩阵规模随 `N²` 增长，因此理论上对应的注意力计算可能增加约 16 倍（忽略通道维、实现和其他模块）。实际吞吐还取决于硬件、内核、批量大小、精度和内存布局，不能只从 FLOPs 推断。

以 Patch 16 为例，224×224 输入有 `14×14=196` 个 patch，448×448 输入有 `28×28=784` 个 patch。分辨率每边翻倍，token 数变为 4 倍；每头注意力分数从 `196²=38,416` 个位置对，增至 `784²=614,656` 个，约 16 倍。若有 CLS token，序列长度分别是 197 和 785，比例几乎不变。

FlashAttention 一类内核通过分块和 IO 感知减少中间注意力矩阵的物化及高带宽显存读写；它计算的仍是精确注意力，标准全局注意力的理论计算仍随 token 数呈二次增长。[FlashAttention 论文](https://arxiv.org/abs/2205.14135) **降低显存峰值**和**降低渐近计算量**是相关但不同的优化目标。

较小 patch 保留更细的空间细节，却提高显存与计算成本；较大 patch 更省，但小目标和精确边界可能难以表示。图像分辨率、patch 大小、窗口大小和后续下采样共同决定代价。

这个例子也能拆开两项常被混为一谈的成本：保持通道宽度不变时，token 增加四倍会让 Q/K/V 投影与逐 token MLP 的主项约增加四倍，而全局注意力的 token 交互项约增加十六倍。因此整个模型的 FLOPs 倍率要按各项占比计算，不能直接写成十六倍；内核是否物化完整分数矩阵又决定不同的显存峰值。

### 2.4 位置编码在视觉里的作用

自注意力本身不带图像网格的位置信息。ViT 通常给 token 加绝对位置嵌入；其他架构也会用相对位置偏置、旋转/坐标编码、卷积或局部窗口结构注入空间关系。分辨率变化时，位置嵌入如何插值、窗口如何划分，都会影响迁移效果。

例如，把预训练输入从 224×224 提高到 448×448、仍用 Patch 16 时，图像位置嵌入需要从 14×14 网格适配到 28×28。常见做法是先取出图像部分，恢复二维网格，在空间维插值，再展平回序列；CLS 的位置项单独保留。这里插值的是位置表示，不是把 196 个已提取的图像 token 插值成 784 个来替代高分辨率图像编码。[ViT 第 3.2 节](https://arxiv.org/html/2010.11929v2#S3.SS2)

### 2.5 几种注意力模式的复杂度

可以把注意力粗略看作 Query 与 Key 的匹配。令 `N` 为 token 数、`D` 为总通道维度，表中只比较主要注意力交互项，省略批量、头数和常数因子：

| 模式 | 每个 Query 可匹配的 Key 数 | 主要交互计算 | 注意力分数规模 |
|---|---:|---:|---:|
| 全局注意力（ViT） | `N` | `O(N²D)` | `O(N²)` |
| 固定窗口（Swin） | 窗口内 `M` 个 | `O(NMD)` | `O(NM)` |
| 空间缩减（PVT） | 约 `N/r²` 个 | `O(N²D/r²)` | `O(N²/r²)` |
| 通道交叉协方差（XCiT） | 在通道维交互 | `O(ND²)` 量级 | 通道矩阵约 `O(D²)` |
{.table-readable}

Swin 的 `M` 是每个固定窗口中的 token 数；窗口大小不变时，注意力交互随图像 token 数 `N` 近似线性增长。PVT 的 `r` 表示 Key/Value 特征图边长缩小倍率，因此空间位置数约缩至 `N/r²`。XCiT 把注意力矩阵从 token-token 关系转到通道-通道关系，使注意力交互项对 `N` 线性，但矩阵计算、投影和前馈网络仍需计算。

这些是帮助理解设计取舍的主项，不是完整模型 FLOPs。Q/K/V 投影与 MLP 通常仍包含 `O(ND²)` 项；不同分辨率阶段、头数、内核与硬件也会改变真实延迟。因此“理论注意力复杂度更低”不自动等于端到端推理更快。[XCiT 原论文](https://arxiv.org/abs/2106.09681)

## 3. ViT：简单、通用，但依赖训练条件

ViT 的重要想法是：把图像切成 patch 后，可以直接用标准 Transformer Encoder 做分类。它不在每层强制局部卷积结构，因此模型能较早建立远距离 patch 之间的关系；相对地，局部平移等视觉先验弱于经典 CNN，训练数据和正则化方案会显著影响结果。

原始工作在大规模图像数据预训练后迁移到较小任务，展示了纯 Transformer 视觉骨干的潜力。它不意味着“Transformer 在任何数据量、训练预算或任务上都胜过 CNN”。比较时应对齐预训练数据、输入分辨率、训练策略、参数量、计算量和推理硬件。

ViT 也不等同于完整视觉系统。用于检测或分割时，通常还需要特征金字塔、解码器、检测头或分割头；用作多模态大模型视觉塔时，还要有把视觉特征接到语言模型上的连接器。

### 3.1 一个最小 ViT 分类器的数据流

下面把 patch 投影、CLS token、位置嵌入、Transformer Encoder 和分类头连起来。为便于展示，它固定输入分辨率，不包含训练循环、数据增强、预训练权重或位置嵌入插值；这不是某个论文配置的复现。

```python
import torch
from torch import nn


class TinyViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=10):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        embed_dim, num_heads, depth = 192, 3, 4
        num_patches = (image_size // patch_size) ** 2
        self.image_size = image_size
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=depth)
        # TransformerEncoder clones the provided layer; give each clone fresh linear weights.
        for layer in self.encoder.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, image):
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("input resolution does not match configured image_size")
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))


model = TinyViT().eval()
with torch.inference_mode():
    logits = model(torch.randn(2, 3, 224, 224))
print(logits.shape)  # [2, 10]
```

前向路径的形状依次是 `[2,3,224,224] → [2,196,192] → [2,197,192] → [2,10]`。第一步把图像映射成 patch token，第二步在序列前拼上 CLS 并加入位置嵌入，Encoder 混合 token 信息，最后只取 CLS 表示做分类。实际模型的初始化、归一化位置、激活函数、正则化与分类池化可能不同；阅读实现时应核对配置，而不是只看名字。

示例显式使用 GELU 和 Pre-Norm，并关闭 dropout 便于核对前向计算；训练配方需要另行设置。`eval()` 切换模块行为，`inference_mode()` 关闭推理时的梯度记录，两者职责不同。最后输出的是尚未归一化的类别 logits，不是概率；做单标签分类训练时可直接交给 `CrossEntropyLoss`，不要先做 softmax 再传入该损失。

<figure class="article-figure" id="fig-vit-token-shapes">
  {{< post-image src="assets/vit-token-shapes-prenorm.webp" alt="ViT 分类数据流：224 方形 RGB 图像以 Patch 16 投影成 196 个 D 维 token，加 CLS 和位置嵌入后序列长 197，Encoder 输出 CLS 分类 logits；下方 Pre-Norm block 的两个残差分支分别绕过注意力和 MLP" >}}
  <figcaption>
    <span class="article-figure__number">图 2</span>
    <span class="article-figure__text">图中 B 为 batch 大小、D 为嵌入宽度、L 为 Encoder 层数、K 为类别数；对应代码示例为 B=2、D=192、L=4、K=10。下方每个残差分支从归一化前出发，在相应子层后相加；MLP 逐 token 处理，注意力负责混合 token。图中省略 dropout 和注意力内部投影。</span>
  </figcaption>
</figure>

## 4. 数据效率与预训练：改变训练信号，不一定改变骨干

### 4.1 DeiT：让数据效率成为架构之外的设计问题

DeiT（Data-efficient Image Transformers）研究如何在 ImageNet 规模数据上有效训练 ViT，并提出使用卷积教师网络进行知识蒸馏的方案。其蒸馏 token 让学生 Transformer 接收教师监督。它的启示是，ViT 结果不仅由网络结构决定，也取决于增强、正则化、优化器、训练时长和蒸馏目标。

### 4.2 BEiT 与 MAE：遮住图像再学习恢复

掩码图像建模（Masked Image Modeling, MIM）随机遮住部分图像块，让模型利用上下文学习表示，但“预测什么”有多种做法：

- **BEiT**：预测被遮挡 patch 对应的离散视觉 token，目标来自预训练的图像 tokenizer。
- **MAE**：编码器只处理未遮挡 patch，轻量解码器补入 mask token 后重建像素；原始预训练损失只计算被遮挡的 patch。
- 后续路线也会预测特征、潜表示或其他目标，不能把所有 MIM 都等同于像素重建。

预训练完成后，常把编码器迁移到分类或密集预测任务。预训练目标提供的是学习信号，不自动保证每个下游任务都更好。

以 224×224、Patch 16 为例，共有 196 个图像 patch；遮挡 75% 后，重型 Encoder 只处理 49 个可见 patch 的表示，mask token 到轻量 Decoder 才补回去。迁移到常规分类任务时使用完整图像，也通常不保留重建 Decoder。因此，高遮挡率带来的是预训练计算收益，不表示分类推理只需读取四分之一的图片。[MAE 原论文](https://arxiv.org/html/2111.06377#S3)

### 4.3 DINO / DINOv2：不依赖人工类别标签的视觉表征

DINO 系列用自蒸馏等方式训练视觉编码器，使不同视图产生一致而有结构的表示。DINOv2 扩大数据与训练体系，发布可用于多种视觉任务的通用特征模型。它是**训练方法和预训练模型家族**，不是一种与 Swin 并列的注意力算子；具体模型骨干仍可采用 ViT 等架构。

DINOv2 的学习信号同时覆盖图像与 patch：DINO 目标对齐不同视图的全局表示，iBOT 目标让被遮挡的学生 patch 对齐教师在相应位置的表示；教师由学生参数的指数移动平均更新。这种局部目标有助于解释它为什么能提供密集特征，但这里预测的是教师表示，不是 MAE 的像素值，也不是 CLIP 的文本表示。[DINOv2 第 4 节](https://arxiv.org/html/2304.07193#S4)

使用 DINOv2 特征做分类、分割或几何任务时，必须区分冻结特征、线性探测、微调和接解码头等设置。仅凭“特征通用”不能推断下游性能、运行成本或对特定领域的适用性。

### 4.4 还会遇到的视觉预训练方法

- **I-JEPA**：从同一图像的上下文区域预测目标区域的表示，而不是逐像素重建目标区域。它代表“在表示空间预测”的自监督路线。[论文](https://arxiv.org/abs/2301.08243)
- **V-JEPA 2 / V-JEPA 2.1**：把 JEPA 的特征预测路线扩展到视频。V-JEPA 2 在大规模图像和视频上训练无动作条件的表征模型；V-JEPA 2.1 的稠密预测损失同时监督被遮挡与可见的上下文 token，并在多个中间层施加自监督信号，以改善图像/视频的稠密特征。它们改变的是训练目标和表征，不是新的注意力家族。[V-JEPA 2](https://arxiv.org/abs/2506.09985)、[V-JEPA 2.1](https://arxiv.org/abs/2603.14482)
- **EVA / EVA-02**：探索大规模视觉预训练和掩码建模，也包括从强视觉语言编码器中学习或重建表示的做法。名称相近的不同版本，其目标与数据配方并不完全相同。[EVA-02 论文](https://arxiv.org/abs/2303.11331)
- **DINO、DINOv2 与 iBOT**：都与自监督视觉 Transformer 有关，但教师学生机制、token 级目标和训练数据各有差别；不应只凭“自监督”就认为表征等价。
- **DINOv3**：DINO 系列的后续视觉基础模型路线，继续扩大数据和模型规模，并用 Gram anchoring 缓解长时间训练中密集特征退化的问题。它更新的是预训练方法、规模和特征质量，不代表出现了新的注意力类别；使用时仍要查看具体骨干、权重版本和特征接口。[论文](https://arxiv.org/abs/2508.10104)

选预训练模型时，检查权重具体版本、输入预处理、训练数据许可、输出层/特征层和目标任务。模型卡中的线性探测、微调和零样本成绩回答的是不同问题。

## 5. 从全局注意力到层级视觉骨干

纯 ViT 的序列通常保持同一空间尺度。计算机视觉里的检测、分割和关键点任务却需要不同分辨率的特征。层级 Transformer 将 CNN 的多阶段思路带入注意力模型：逐步缩小空间尺寸、增加通道宽度，并向任务头提供多尺度特征。

### 5.1 Swin Transformer：窗口注意力与移位窗口

Swin 在每层把 token 划分到局部窗口，只在窗口内做注意力；下一层把窗口划分平移，使相邻窗口之间逐步交换信息。Patch Merging 逐阶段下采样，形成类似金字塔的特征层级。

以输入 224×224、初始 Patch 4、基础通道宽度 $C$ 的四阶段配置为例，特征形状沿层级变化如下；这里省略 batch 维，通道值用符号表示，不指定某个型号：

| 阶段 | 空间网格 | 通道宽度 | 空间 token 数 |
| --- | ---: | ---: | ---: |
| Stage 1 | 56×56 | $C$ | 3,136 |
| Stage 2 | 28×28 | $2C$ | 784 |
| Stage 3 | 14×14 | $4C$ | 196 |
| Stage 4 | 7×7 | $8C$ | 49 |

一次 Patch Merging 将相邻 2×2 token 的特征拼成 $4C$ 维，再投影到 $2C$ 维，因此空间位置减为四分之一，通道宽度翻倍。它与改变窗口划分是不同操作：**窗口移位改变交互对象，Patch Merging 改变特征尺度。** [Swin 第 3.1 节](https://arxiv.org/html/2103.14030#S3.SS1)

实现中常用循环移位来高效组织窗口，但要用 attention mask 阻止图像两侧被循环移动后错误地视为相邻区域；计算结束再移回原坐标。这个掩码处理空间边界，不是语言模型的因果掩码，图像 token 仍可在允许的窗口区域内双向交互。[Swin 第 3.2 节](https://arxiv.org/html/2103.14030#S3.SS2)

优点是高分辨率下计算更可控，并能自然输出多尺度特征；代价是局部与跨窗口信息传播受窗口结构影响，窗口大小、移位和分辨率匹配也增加实现细节。窗口注意力不是“整个网络永远看不到全图”，而是信息通过后续层跨窗口传播。

Swin V2 进一步调整训练稳定性、相对位置表示和高分辨率扩展方法。使用 Swin 的论文或代码时要区分版本、窗口配置与预训练分辨率。

### 5.2 PVT 与 SegFormer：为密集预测设计金字塔

Pyramid Vision Transformer（PVT）试图让 Transformer 像 CNN 骨干一样产生金字塔特征。其空间缩减注意力保留所有位置的 Query，同时对 Key/Value 的空间网格降采样，减少参与匹配的位置数；后续 PVT 变体继续改进注意力和训练效率。

SegFormer 采用分层 Transformer 编码器配合轻量 MLP 解码器做语义分割。[论文](https://arxiv.org/abs/2105.15203) 它说明“骨干”和“完整任务模型”要分开看：编码器抽取多尺度特征，解码器融合特征并输出逐像素类别。

### 5.3 CMT、CvT、CoAtNet：卷积与 Transformer 的融合路线

“卷积还是 Transformer”并非只能二选一。多种混合设计把卷积的局部性、平移归纳偏置与注意力的内容自适应关系建模结合起来：

- **CMT（Convolutional Neural Networks Meet Vision Transformers）**：层级骨干中用深度卷积构成 Local Perception Unit（LPU），并以 Inverted Residual Feed-Forward Network（IRFFN）增强局部交互，再结合注意力建模更广范围的关系。
- **CvT（Convolutional vision Transformer）**：在 token 投影或注意力结构中加入卷积操作，并采用层级阶段设计。
- **CoAtNet**：按网络阶段组合卷积与注意力模块，探索数据规模、归纳偏置和模型扩展的折中。

这些名称代表不同具体设计，不能概括成一个统一的“卷积 Transformer 公式”。比较时应看卷积出现在哪一层、注意力作用于什么分辨率、是否共享权重，以及下游输出特征如何组织。

### 5.4 MaxViT、Twins 与其他注意力效率设计

Swin 不是唯一控制注意力复杂度的办法。MaxViT 交替使用块内局部注意力和跨块网格注意力，让局部与更广范围的交互在各阶段进行；以 [Twins-SVT](https://arxiv.org/abs/2104.13840) 为例，它组合局部分组自注意力与对降采样特征做注意力的模块。共同目标是减少高分辨率全局注意力的开销，同时保留足够的信息交互。

[EfficientFormer](https://arxiv.org/abs/2206.01191)、[MobileViT](https://arxiv.org/abs/2110.02178)、[LeViT](https://arxiv.org/abs/2104.01136) 等则重点考虑移动端延迟、内存访问和算子实现。轻量模型选型不能只看参数量或 FLOPs：实际设备上的延迟、批量、量化支持、算子融合和运行时常常更关键。

检索论文时注意 **EfficientViT 是重名家族**：**EfficientViT-M** 用 sandwich block 和 Cascaded Group Attention，重点压低注意力与内存操作的开销；**EfficientViT-L** 用多尺度线性注意力面向高分辨率密集预测。两者的架构、任务重点和型号后缀不同。EfficientViT-M 与 EfficientViT-L 是两条不同论文路线，引用或比较时应同时核对副标题、版本后缀和任务定位。[EfficientViT-M 论文](https://arxiv.org/abs/2305.07027)、[EfficientViT-L 论文](https://arxiv.org/abs/2205.14756)

### 5.5 其他常见骨干：按设计问题认识它们

| 家族 | 代表想法 | 放在路线图中的位置 |
|---|---|---|
| CaiT | 研究深层 ViT 的优化与分类 token 更新方式 | 仍是全局注意力的 ViT 改进；[论文](https://arxiv.org/abs/2103.17239) |
| XCiT | 先在特征通道维度建模交互，降低空间 token 注意力的负担 | 另一种注意力轴选择；[论文](https://arxiv.org/abs/2106.09681) |
| FocalNet | 用多层级的局部上下文聚合构造逐渐扩大的感受范围 | 以调制机制取代自注意力的路线；[论文](https://arxiv.org/abs/2203.11926) |
| Hiera | 借助强预训练目标简化层级视觉 Transformer 的专用组件 | 说明预训练可以改变“需要多复杂的架构”；[论文](https://arxiv.org/abs/2306.00989) |
| PoolFormer / MetaFormer | 用池化等简单 token mixer 替换注意力，保留 MetaFormer 总体骨架 | 用于判断收益来自整体骨架还是 Attention 本身；[论文](https://arxiv.org/abs/2111.11418) |
| Vision Mamba（Vim）/ VMamba | Vim 用双向状态空间块处理图像序列；VMamba 用四向二维选择扫描聚合视觉上下文 | 状态空间计算随 token 序列长度近似线性增长，可减少对全局自注意力的依赖；它们不是 Swin 一类窗口注意力 Transformer。[Vim](https://arxiv.org/abs/2401.09417)、[VMamba](https://arxiv.org/abs/2401.10166) |
| MambaVision | 层级式 Mamba-Transformer 混合骨干，在后段加入自注意力建模远程空间关系 | 展示状态空间模块与 Transformer 注意力可以组合；它不是纯 Mamba，也不是纯注意力骨干。[论文](https://arxiv.org/abs/2407.08083) |
| ConvNeXt | 现代化 CNN，而非 Transformer | 重要对照组：现代 CNN 仍能与视觉 Transformer 竞争 |
{.table-readable}

名称列表只是地图，不代表这些模型在所有任务上都比前述骨干更优。Vision Mamba、VMamba 与 MambaVision 属于状态空间或混合骨干，不能统称为窗口注意力 Transformer。理论复杂度也不能直接推出设备吞吐量；阅读时先问作者在解决什么瓶颈：局部性、全局交互、层级特征、深层优化、训练数据还是设备延迟。

## 6. 识别任务之外：视觉骨干如何接入下游系统

视觉 Transformer 通常作为骨干，不是任务本身。常见组合包括：

- **分类**：最后的类别 token 或池化特征接分类头。
- **目标检测**：骨干输出多尺度特征，检测器再预测类别和边界框；如 DETR 展示了 Transformer 检测器的集合预测路线，但不等于所有 Transformer 检测都采用 DETR 结构。
- **语义/实例分割**：高分辨率特征送入像素解码器或掩码头。Mask2Former 等工作将掩码分类和 Transformer 解码结合。
- **可提示分割**：Segment Anything（SAM）把图像编码器与提示编码、掩码解码组合起来，支持点、框等提示驱动的分割；SAM 2 将任务扩展到视频并使用记忆机制。它们是面向分割的完整系统，不是单一视觉骨干的别名。
- **概念级分割与跟踪**：SAM 3 在 SAM 2 的提示式分割基础上，加入短文本概念和图像示例提示，可在图像/视频中定位、分割并跟踪符合概念的多个实例。它把“按提示分出一个对象”和“找出所有符合概念的对象”连接起来；短语范围、歧义和领域迁移仍需单独评估。[SAM](https://arxiv.org/abs/2304.02643)、[SAM 2](https://arxiv.org/abs/2408.00714)、[SAM 3](https://arxiv.org/abs/2511.16719)
- **深度、姿态与跟踪**：骨干特征需经任务头转换为连续深度、坐标、轨迹或关联结果，输出约束与评测协议各异。
- **视频理解**：加入时间维 token、时空注意力或逐帧特征交互。TimeSformer 比较多种时空注意力拆分方式，ViViT 则探索时空 token 化及空间/时间维度分解；两者都说明图像骨干需要额外的时间建模才能理解动作变化。直接把所有帧 patch 拼成一个序列会快速增加计算，因此常使用分解注意力、稀疏采样或局部时空窗口。[TimeSformer](https://arxiv.org/abs/2102.05095)、[ViViT](https://arxiv.org/abs/2103.15691)
- **视频表征与世界模型**：V-JEPA 2-AC 是在无动作条件的 V-JEPA 2 上，利用少量机器人轨迹视频后训练得到的动作条件潜空间世界模型；论文展示了它按图像目标规划机械臂操作。V-JEPA 2 的基础表征模型与 2-AC 的规划模型不能混为一个能力接口。两者预测的是潜表示或状态，不以合成未来视频像素为目标，因此与 DiT 类视频生成模型的训练目标和输出不同。[V-JEPA 2 论文](https://arxiv.org/abs/2506.09985)

不要把 ImageNet 分类精度直接当作检测、分割、鲁棒性或部署速度的代理指标。

## 7. CLIP、SigLIP 与视觉语言模型中的 ViT

CLIP 用大量图文对进行对比学习，使图像编码器和文本编码器的表示可以相互匹配。其视觉塔可以是 ViT，也可以采用其他视觉骨干。SigLIP 改变图文对齐损失的形式。它们属于**视觉语言预训练系统**，并非一种新的图像 patch 架构。

以一个含 `B` 组配对样本的 batch 为例，图像编码器输出 `v_i`，文本编码器输出 `t_j`，归一化后计算两两相似度：

$$
S_{ij}=s\,\frac{v_i^\top t_j}{\lVert v_i\rVert\lVert t_j\rVert}
$$

`S` 是 `B×B` 相似度矩阵，对角线是匹配图文，非对角线是 batch 内的错配组合。CLIP 对每张图像找正确文本、对每段文本找正确图像，使用对称的交叉熵对比目标；可学习尺度 `s` 控制相似度 logits 的幅度。模型因此学到一个共享嵌入空间，而不是逐 token 翻译图像和文本。

这种 batch 内负例构造也有局限：非配对不一定代表语义无关，两张不同图片可能描述同一概念，因而会产生“假负例”；数据组成、batch 采样和训练规模会影响对齐质量。

SigLIP 改用成对的 sigmoid 二分类目标：每个图文组合判断匹配或不匹配，不对整行或整列做 softmax 归一化。论文实现还包含偏置与批次缩放细节；这里的重点是区分“归一化 softmax 对比”和“成对 sigmoid”两种目标，不把损失函数差异误说成新的视觉骨干。

零样本分类时，把类别写成文本提示（如“a photo of a {class}”），由文本编码器得到候选向量；图像编码器输出图像向量后，计算它与各候选的相似度并取最高者。提示模板、类别描述和图像分辨率都会影响结果。图文对齐也不自动带来精确定位、可靠 OCR 或空间推理能力；这些能力仍受训练数据和任务设置影响。

多模态大语言模型常把 ViT 产生的图像特征经投影器、重采样器或查询模块映射到语言模型表示空间。高分辨率切图（tiling）、视觉 token 压缩与图像编码器的 patch 数是不同层级的问题，可参阅站内 [InternVL3.5 架构解析](/posts/ai/internvl-3-5)。

## 8. 生成模型：DiT 与 SiT 是另一条问题线

分类骨干处理相对固定的输入图像并输出表征或类别。扩散模型则在多个噪声时间步反复运行去噪器：输入包含带噪图像/潜变量、时间条件和可选条件，输出噪声、速度或其他训练目标。二者可以共用 Transformer 构件，但输入分布、条件注入和训练目标不同。文本等条件常通过交叉注意力注入：图像 token 作 Query，条件 token 提供 Key 和 Value；AdaLN（Adaptive Layer Normalization）则用时间步或类别条件生成归一化的缩放、偏移或门控参数。

### 8.1 DiT：Transformer 作为扩散去噪器

DiT（Diffusion Transformer）将扩散模型潜空间中的表示切成 patch token，用 Transformer block 取代传统 U-Net 去噪骨干。时间步和类别条件可通过 AdaLN 等调制方式注入；模型规模、深度、宽度与 token 数可系统扩展。原论文在 ImageNet 条件生成中研究模型计算量和生成质量的关系。[DiT 原论文](https://arxiv.org/abs/2212.09748) 这里的潜空间通常由变分自编码器（VAE）编码器把图像压缩而来，去噪后的潜变量再由 VAE 解码成像素图像；因此潜变量 patch 数受图像尺寸、VAE 压缩率和 DiT patch 大小共同影响。

要点是：**DiT 改的是扩散模型的去噪网络，不是把扩散过程本身换成 Transformer。** 噪声调度、训练目标、采样器与去噪骨干是可以分别设计的组件。DiT 中常见的 patch token 也不应与输入 RGB 图像的 ViT patch 无条件混为一谈：DiT 经常在 VAE 潜空间而非像素空间操作。

原论文的 256×256 图像经 VAE 压成 32×32×4 潜变量；使用潜空间 Patch 2 时，得到 16×16=256 个 token。512×512 图像对应 64×64×4 潜变量，仍用 Patch 2 则有 32×32=1,024 个 token。**DiT-XL/2 的 /2 指潜变量 patch 边长，不是 RGB 图像里每 2 个像素切一块。** token 数还要乘上多步采样的重复调用成本，才接近一次生成的总计算口径。[DiT 第 3 节](https://arxiv.org/html/2212.09748#S3)

### 8.2 SiT：基于 DiT 骨干研究插值与生成动力学

本文所说的 SiT 是 *Scalable Interpolant Transformers*。它沿用 DiT 的网络骨干，模块化考察连接数据分布与噪声分布的插值、离散或连续时间学习、训练目标，以及确定性或随机采样等选择。[SiT 原论文](https://arxiv.org/abs/2401.08740) 它不是 Swin 的某个视觉分类版本，也不是“另一种注意力层”。

| 比较点 | DiT | SiT |
|---|---|---|
| 角色 | 潜空间扩散模型使用的 Transformer 去噪骨干 | 建立在 DiT 骨干上的插值生成模型家族 |
| 主要研究对象 | 骨干规模、token 数与去噪网络设计 | 概率路径/插值、时间参数化、预测目标和采样随机性 |
| 过程视角 | 以扩散加噪/去噪过程为基线 | 插值框架可涵盖扩散路径，也可研究其他连接噪声与数据的路径 |
| 读论文时的重点 | Transformer 如何替代 U-Net 承担去噪 | 固定骨干时，改变生成路径与学习/采样选择会带来什么影响 |
{.table-readable}

原论文在条件 ImageNet 256/512 的比较中控制网络结构、参数量和 GFLOPs，重点隔离建模选择的影响；结果应按该实验设置理解，不能推成 SiT 在所有任务都优于 DiT。Flow Matching（FM）是训练连续正规化流的一类方法：给定噪声到数据的概率路径，模型回归路径对应的速度场，生成时再沿学到的场积分。[Flow Matching 原论文](https://arxiv.org/abs/2210.02747) 该框架可使用扩散型或非扩散型路径；它描述训练动力学，不是固定网络架构。站内 [扩散模型指南](/posts/ai/diffusion-models) 已更细致推导 DDPM、DDIM、CFG 与 DiT 的位置。

### 8.3 U-ViT、PixArt、MMDiT：生成 Transformer 的不同方向

- **U-ViT**：把时间、条件和带噪图像 patch 都表示成 token，并在浅层与深层 Transformer block 之间加入长跳跃连接。这里的 U 形主要描述跨层连接；原论文不依赖 U-Net 式逐阶段空间下采样与上采样，不能把它直接当作层级多尺度骨干。[U-ViT 原论文](https://arxiv.org/abs/2209.12152)
- **PixArt-α**：文本到图像 DiT 路线，使用文本交叉注意力，并将训练拆分为不同阶段来改善文本对齐与图像质量；数据标注和训练配方也是贡献的重要部分。
- **MMDiT**：Stable Diffusion 3 使用的多模态扩散 Transformer 为文本与图像表示保留不同的投影/参数，再把两种模态的 token 放入联合注意力交互。它与 PixArt 常见的文本交叉注意力方案不同；具体实现可对照[技术报告](https://arxiv.org/abs/2403.03206)和模型代码。
- **视频 DiT**：把时空 patch、帧或潜变量序列交给 Transformer 建模。以 [CogVideoX](https://arxiv.org/abs/2408.06072) 为例，它用 3D VAE 同时压缩空间和时间维度，并以 expert Transformer 和自适应归一化加强文本—视频融合。时空 token 数和注意力成本随分辨率、帧数迅速增长，因而分块、压缩、稀疏注意力和分层生成都很重要。

这些模型说明生成质量不由骨干名称单独决定。潜空间压缩、文本编码器、条件注入、数据质量、训练预算、去噪目标与采样策略共同影响最终结果。

## 9. 一个统一的比较框架

| 维度 | ViT / DeiT | Swin / PVT / CMT | CLIP / DINOv2 | DiT / SiT |
|---|---|---|---|---|
| 主要用途 | 分类与通用骨干 | 多尺度视觉任务与高分辨率骨干 | 可迁移视觉或图文表征 | 条件/无条件生成的去噪网络 |
| 输入 | 图像 patch | 图像 patch/特征图，多阶段 | 图像；CLIP 还配对文本 | 每步带噪图像或潜变量及时间条件 |
| 空间尺度 | 常见为单尺度 | 通常逐阶段下采样 | 取决于所用骨干和目标 | 由潜空间、patch 和去噪设计决定 |
| 注意力范围 | 常见全局 | 窗口、缩减、局部/全局混合 | 继承骨干 | 取决于生成骨干与 token 长度 |
| 学习信号 | 类别监督或迁移目标 | 监督、蒸馏或预训练 | 自监督/图文对比等 | 噪声、速度或相关生成目标 |
| 主要代价 | 高分辨率 token 的二次复杂度 | 结构与实现更复杂 | 数据构造、预训练成本与分布偏差 | 多步采样和高 token 数的计算成本 |
{.table-readable}

这张表描述典型路线，不是硬性定义。具体模型可能改变池化方式、注意力算子、损失或多尺度路径；阅读论文时应以模型配置和方法章节为准。

## 10. 如何按任务选型

| 需求 | 起步思路 | 需要核对的因素 |
|---|---|---|
| 小数据分类 | 先比较预训练 ViT 与 CNN/混合骨干 | 预训练来源、增强、微调策略和分辨率 |
| 目标检测/分割 | 选能提供多尺度特征的骨干 | 特征层接口、检测/解码头、目标尺度和训练预算 |
| 大分辨率或细节密集任务 | 比较层级/窗口注意力、更小 patch 的 ViT 与 EfficientViT-L | 小目标召回、边界质量、窗口效应、显存和实际硬件延迟 |
| 通用冻结特征 | 评估 DINOv2 等自监督模型，也要对照监督特征 | 线性探测、微调、领域差异、特征层和吞吐 |
| 图文检索/零样本识别 | 采用 CLIP/SigLIP 类图文对齐模型 | 文本模板、领域覆盖、类别名称和图像分辨率 |
| 按文本找出图像/视频中的多个对象 | 评估 SAM 3 等概念级分割系统 | 概念短语是否清楚、实例覆盖、视频跟踪和领域迁移 |
| 多模态大模型图像输入 | 评估视觉塔、切图和连接器的组合 | 每图 token 数、OCR/定位需求、上下文长度和推理预算 |
| 文本到图像生成 | 根据训练资源与任务选 U-Net、DiT 类方案 | 数据、潜空间、文本条件、训练/采样配置和许可 |
| 移动端/实时视觉 | 以真实设备基准筛选轻量架构 | 端到端延迟、内存带宽、量化和目标运行时 |
{.table-readable}

合理的对比实验应固定数据划分和预处理，报告参数量、FLOPs 或 MACs、峰值显存、实际延迟、训练预算和任务指标。分类精度只回答分类任务上的一部分问题。

## 11. 高频误解

1. **“ViT、Swin、DiT、SiT 都是视觉分类网络。”** 不是。DiT/SiT 主要讨论生成；CLIP/DINOv2 主要描述预训练目标和表征系统。
2. **“Transformer 天生比 CNN 更强。”** 架构效果依赖数据、计算、目标、训练配方与部署条件。
3. **“Swin 的窗口让远处永远互不可见。”** 窗口移位与多层传播建立跨窗口联系，但传播路径和全局交互方式不同于单层全局注意力。
4. **“所有 ViT 都把 CLS token 作为唯一输出。”** 有些模型池化 patch token，密集任务会保留空间特征图。
5. **“DINOv2 是一种网络层。”** 它更准确地说是自监督训练路线及其预训练模型家族；模型骨干另有具体架构。
6. **“DiT 就是把普通 ViT 拿来生成图片。”** 它使用 Transformer 作为带时间条件的去噪网络，训练目标和推理调用方式来自生成过程。
7. **“参数少或 FLOPs 低就一定更快。”** 真实速度还受内存访问、内核、批量、分辨率和硬件影响。
8. **“EfficientViT 只指一种架构。”** EfficientViT-M 与 EfficientViT-L 是不同论文路线；要按副标题和模型后缀核对注意力模块及目标任务。
9. **“同样叫 Transformer，结果就可以直接横向比。”** 任务、数据、预训练、输入尺寸和评价协议必须对齐。

## 12. 建议的阅读顺序

### 先懂架构

1. ViT：理解 patch、位置编码和全局注意力。
2. DeiT：理解蒸馏和数据效率。
3. Swin：理解窗口注意力与层级特征。
4. PVT/CvT/CMT/MaxViT：按任务补充高效注意力和卷积融合路线。

### 再懂表征学习

1. MAE 或 BEiT：理解掩码图像建模的不同预测目标。
2. DINOv2：理解视觉自监督特征。
3. CLIP 或 SigLIP：理解图文对齐与零样本迁移。

### 最后进入生成

1. 先读站内[扩散模型指南](/posts/ai/diffusion-models)，弄清训练目标、噪声调度与采样器。
2. 再读 DiT，理解 Transformer 去噪骨干。
3. 接着读 SiT，关注插值路径、生成目标和采样方式的实验拆分。
4. 最后比较 PixArt、MMDiT 或视频生成架构，关注文本和时空条件如何交互。

## 阅读自测与验收

- **Swin 的窗口移位和 Patch Merging 各改变什么？** 移位让跨窗口 token 交换信息，配合掩码排除循环移位引入的错误邻接；Patch Merging 则降低空间分辨率并增加通道。

- **U-ViT 的 U 形长跳跃连接是否意味着空间多尺度下采样？** 原论文用浅层与深层 block 的长跳跃连接融合特征，不依赖 U-Net 式空间下采样与上采样；跨层融合与层级多尺度是不同结构。

### 关键结论速查

- **输入表示决定成本**：224×224、Patch 16 得到 196 个图像 token；标准全局注意力的主要交互项约随 token 数平方增长。
- **骨干结构决定信息交互方式**：ViT 常用全局注意力；Swin 用移位窗口和层级特征；PVT、XCiT 等用不同方式减少高分辨率交互开销。
- **训练目标不是骨干**：DeiT 借助蒸馏提升训练效率，MAE 以遮挡重建预训练，DINOv2/v3 学习自监督视觉特征，CLIP/SigLIP 对齐图文表示；这些训练路线可以与具体骨干组合。
- **下游系统不止编码器**：检测、分割和视频任务还需要任务头、解码器、提示接口或时序/记忆模块；SAM 系列是系统设计的例子。
- **生成 Transformer 属于另一条路线**：DiT 是扩散去噪骨干；SiT 基于可扩展插值 Transformer 研究生成过程。噪声过程、目标、条件注入和采样应分开看。
- **理论成本不能替代实测**：参数量、FLOPs 和注意力复杂度不能单独预测端到端或移动端延迟。

## 13. 自测

- 224×224 图像使用 Patch 16 时有多少图像 token？加入 CLS 后长度是多少？
- TinyViT 示例为何从 196 个 patch token 变成 197 个 Encoder 输入？
- Patch 边长减半，标准全局注意力矩阵的 token 维规模大约如何变化？
- 窗口大小固定时，Swin 的注意力交互项如何随图像 token 数增长？为什么这不等于整个模型都只有线性成本？
- ViT 与 Swin 的关键区别是“有没有 Transformer”还是注意力范围和尺度组织？
- DeiT、MAE、DINOv2 和 CLIP 分别主要改变架构、监督信号还是训练目标？
- CLIP 的相似度矩阵中哪些位置是正配对？SigLIP 与 CLIP 的损失形式差在哪？
- 为什么检测和分割常更喜欢层级特征？
- CMT 中“卷积与 Transformer 结合”具体解决了哪类取舍？
- DiT 相比 ViT 多了哪些生成过程中的输入条件与输出职责？
- SiT 与 DiT 的关系是什么？它是否意味着一种新的图像分类骨干？
- 评估移动端模型时，为什么 FLOPs 不能代替设备延迟？
- EfficientViT-M 与 EfficientViT-L 为什么不能仅凭名称直接比较？

**参考答案要点**：196 个 patch token，带 CLS 共 197；token 数翻四倍时注意力矩阵元素数约增至 16 倍；Swin 以局部窗口和移位窗口组织注意力，并形成多尺度特征；DeiT/MAE/DINOv2/CLIP 分别偏向蒸馏训练、掩码重建、自监督表征和图文对齐；密集任务需要空间细节与尺度信息；CMT 用卷积补充局部归纳偏置；EfficientViT-M 使用 sandwich block 与级联分组注意力，EfficientViT-L 使用多尺度线性注意力做高分辨率密集预测；DiT 每个噪声时间步处理带噪输入及条件并预测生成目标；SiT 是基于 DiT 骨干研究插值和生成动力学的路线；真实延迟还取决于内存、算子和硬件。

## 参考论文与官方资料

下面按主题列出入门和追溯设计来源时值得优先阅读的代表作。它是路线图，不是穷尽所有变体的文献目录；模型版本、训练配方和代码实现仍应回到具体论文与项目页核对。

### 图像骨干与注意力设计

- [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale（ViT）](https://arxiv.org/abs/2010.11929)
- [Training data-efficient image transformers & distillation through attention（DeiT）](https://arxiv.org/abs/2012.12877)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)；[Swin Transformer V2](https://arxiv.org/abs/2111.09883)
- [Pyramid Vision Transformer（PVT）](https://arxiv.org/abs/2102.12122)；[Twins-SVT](https://arxiv.org/abs/2104.13840)
- [CMT](https://arxiv.org/abs/2107.06263)；[CvT](https://arxiv.org/abs/2103.15808)；[CoAtNet](https://arxiv.org/abs/2106.04803)
- [MaxViT](https://arxiv.org/abs/2204.01697)；[XCiT](https://arxiv.org/abs/2106.09681)
- [EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention（EfficientViT-M）](https://arxiv.org/abs/2305.07027)
- [EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction（EfficientViT-L）](https://arxiv.org/abs/2205.14756)
- [CaiT](https://arxiv.org/abs/2103.17239)；[FocalNet](https://arxiv.org/abs/2203.11926)；[Hiera](https://arxiv.org/abs/2306.00989)
- [MobileViT](https://arxiv.org/abs/2110.02178)；[LeViT](https://arxiv.org/abs/2104.01136)；[EfficientFormer](https://arxiv.org/abs/2206.01191)
- [MetaFormer](https://arxiv.org/abs/2111.11418)；作为非 Transformer 对照的 [ConvNeXt](https://arxiv.org/abs/2201.03545)

### 视觉状态空间骨干

- [Vision Mamba（Vim）](https://arxiv.org/abs/2401.09417)；[VMamba](https://arxiv.org/abs/2401.10166)
- [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)

### 预训练与视觉语言表征

- [BEiT](https://arxiv.org/abs/2106.08254)；[Masked Autoencoders（MAE）](https://arxiv.org/abs/2111.06377)
- [DINOv2](https://arxiv.org/abs/2304.07193)；[DINOv3](https://arxiv.org/abs/2508.10104)
- [I-JEPA](https://arxiv.org/abs/2301.08243)；[V-JEPA 2](https://arxiv.org/abs/2506.09985)；[V-JEPA 2.1](https://arxiv.org/abs/2603.14482)
- [EVA](https://arxiv.org/abs/2211.07636)；[EVA-02](https://arxiv.org/abs/2303.11331)
- [CLIP](https://arxiv.org/abs/2103.00020)；[SigLIP](https://arxiv.org/abs/2303.15343)

### 下游任务与视频理解

- [DETR](https://arxiv.org/abs/2005.12872)；[Mask2Former](https://arxiv.org/abs/2112.01527)；[SegFormer](https://arxiv.org/abs/2105.15203)
- [Segment Anything（SAM）](https://arxiv.org/abs/2304.02643)；[SAM 2](https://arxiv.org/abs/2408.00714)；[SAM 3](https://arxiv.org/abs/2511.16719)
- [TimeSformer](https://arxiv.org/abs/2102.05095)；[ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)（含 V-JEPA 2-AC）

### 图像与视频生成

- [Scalable Diffusion Models with Transformers（DiT）](https://arxiv.org/abs/2212.09748)
- [SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://arxiv.org/abs/2401.08740)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [U-ViT: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152)
- [PixArt-α](https://arxiv.org/abs/2310.00426)；[Scaling Rectified Flow Transformers（Stable Diffusion 3 / MMDiT）](https://arxiv.org/abs/2403.03206)
- [CogVideoX](https://arxiv.org/abs/2408.06072)

