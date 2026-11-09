---
title: "Ultralytics 源码详解：从 YOLO 检测到训练、跟踪与部署"
date: 2026-11-05
draft: false
tags: ["Ultralytics", "YOLO", "Computer Vision", "Deep Learning", "PyTorch", "Object Detection"]
categories: ["人工智能"]
authors: ["chase"]
summary: "从 Ultralytics 8.4.202 源码追踪 YOLO 检测、训练、跟踪与部署，用交互实验和可运行算例复算坐标与 NMS，再核对双检测头、恢复状态和 ONNX 输出。"
description: "从 Ultralytics 8.4.202 源码追踪 YOLO 检测、训练、跟踪与部署，用交互实验和可运行算例复算坐标与 NMS，再核对双检测头、恢复状态和 ONNX 输出。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
images: ["assets/ultralytics-overview.webp"]
reading_prerequisites: "Python、PyTorch 张量和卷积网络基础；训练自有模型还需要图像标注与独立验证集"
reading_focus: "先建立 API、任务和执行器的对应关系，再用交互实验复算坐标与 NMS；沿双检测头与训练状态理解源码，最后核对指标和部署接口。"
related_posts:
  - "/posts/ai/vision-transformers"
  - "/posts/calibration/opencv"
  - "/posts/ai/distributed-training-memory"
  - "/posts/ai/muon-optimizer"
---

拿到一帧相机图像，希望输出“哪些地方有零件、每个零件是什么、下一帧还是不是同一个零件”，需要的不只是一个神经网络。图像读取、尺寸变换、标注、训练、结果坐标、跨帧关联和部署运行时，都必须遵守一致的接口。

**Ultralytics 把这些环节组织成一套视觉工程框架。** `YOLO("yolo26n.pt")` 是常见入口，但仓库还包括多种视觉任务、其他模型系列、数据处理、训练器、验证器、导出器、跟踪器和应用方案。理解这个项目，应同时看“模型怎样预测”和“框架怎样把模型变成可用工具”。

本文依据本地仓库的 **`8.4.202`** 版本，固定提交为 [`1a767da5e58467ab8fbc2407b600b7632033dbc5`](https://github.com/ultralytics/ultralytics/tree/1a767da5e58467ab8fbc2407b600b7632033dbc5)。下文的默认值和调用关系以该提交为准；在线文档会继续演进。这里已经包含 YOLO26，不能直接套用早期 YOLOv8 教程里的全部参数和输出假设。

## 阅读路线

初次接触可以先读[项目定位](#scope)、[快速运行](#quickstart)和[结果接口](#results)，再按需求进入[训练](#training)、[跟踪](#tracking)或[部署](#export)。希望修改源码时，沿[目录与分发](#architecture) → [网络结构](#network) → [推理调用链](#inference) → [损失与分配](#loss)阅读。

## 1. 项目边界：YOLO、任务与运行模式 {#scope}

YOLO 是目标检测模型的一类名称；Ultralytics 是维护模型与工具链的项目。包版本 `8.4.202` 也不表示“只能使用 YOLOv8”：模型系列由实际加载的权重或结构文件决定。

### 1.1 先选择任务，再选择模型

本提交的 `YOLO.task_map` 注册了七种任务。它们的输出语义、训练标签和验证指标不同：

| 任务 | 回答的问题 | 典型结果字段 |
| --- | --- | --- |
| `detect` | 哪些轴对齐矩形里有什么物体 | `boxes` |
| `segment` | 每个物体实例占据哪些像素 | `boxes`、`masks` |
| `classify` | 整张图像属于什么类别 | `probs` |
| `pose` | 物体或人体的关键点在哪里 | `boxes`、`keypoints` |
| `obb` | 带方向的物体区域在哪里 | `obb` |
| `semantic` | 每个像素属于什么类别 | `semantic_mask` |
| `depth` | 每个像素对应怎样的深度预测 | `depth` |
{.table-readable}

实例分割区分两个同类物体，语义分割主要给像素分类；分类不提供局部定位。深度字段的尺度与单位要根据具体权重、任务配置及训练目标核对，不能看到一张深度图就当作米制距离。

任务注册意味着框架存在相应实现，**不意味着任意 YOLO 权重都支持全部任务**。例如检测权重不会因为调用时写了 `task="pose"` 就生成关键点；需要对应的结构、标签和权重。

仓库也暴露 `SAM`、`FastSAM`、`RTDETR`、`YOLOWorld`、`YOLOE` 等入口。提示式分割、开放词汇检测与固定类别检测有各自的数据和交互方式，不能把全部模型当作同一个检测头。

### 1.2 mode 决定当前执行什么

| 模式 | 核心工作 | 常用 Python 入口 |
| --- | --- | --- |
| `train` | 用训练数据更新参数 | `model.train(...)` |
| `val` | 在指定数据划分上统计指标 | `model.val(...)` |
| `predict` | 对新输入产生结果 | `model.predict(...)` 或 `model(...)` |
| `track` | 检测后维护跨帧关联 | `model.track(...)` |
| `export` | 转换计算图与部署格式 | `model.export(...)` |
| `benchmark` | 比较支持的导出后端 | `model.benchmark(...)` |
{.table-readable}

跟踪是检测／实例结果上的时序处理流程，不是第八种逐像素任务；导出也不会自动提高模型准确率。项目还包含 `solutions/` 下的计数、区域分析等应用组合，适合在理解基础结果之后阅读。

## 2. 目录与架构：一套外观，多种执行器 {#architecture}

最有价值的源码路线如下。链接固定在本文使用的提交，可对照本地文件阅读。

| 路径 | 重点问题 |
| --- | --- |
| [`ultralytics/__init__.py`](https://github.com/ultralytics/ultralytics/blob/1a767da5e58467ab8fbc2407b600b7632033dbc5/ultralytics/__init__.py) | 公共入口、版本与延迟导入 |
| `cfg/__init__.py`、`cfg/default.yaml` | CLI 解析、参数合法性与公共默认配置 |
| `engine/model.py` | 加载模型、合并参数、选择当前执行器 |
| `models/yolo/model.py` | 不同 task 对应哪些模型、训练器、验证器、预测器 |
| `engine/predictor.py`、`models/yolo/detect/predict.py` | 通用推理循环与检测后处理 |
| `engine/trainer.py`、`engine/validator.py` | 训练状态、验证循环与回调 |
| `nn/tasks.py`、`nn/modules/` | YAML 解析、网络构建、基础模块与任务头 |
| `data/` | 数据检查、加载器、增强与标签变换 |
| `utils/loss.py`、`utils/tal.py`、`utils/nms.py` | 监督目标、正样本分配与候选框筛选 |
| `nn/backends/`、`engine/exporter.py` | 多运行时推理与导出 |
| `engine/results.py`、`trackers/` | 结果对象与跨帧关联 |
{.table-readable}

表中除第一行外的路径都相对于 `ultralytics/`。阅读时先抓住两个层次：`YOLO`／`Model` 是用户操作的外观对象；真正执行训练、预测等工作的，是按任务选择的执行器。

例如检测任务的分发表对应 `DetectionModel`、`DetectionTrainer`、`DetectionValidator` 和 `DetectionPredictor`。分割任务换成相应的分割实现，复用同一组外部方法。**统一的是调用方式，任务特有的数据处理和损失仍然存在。**

<figure class="article-figure" id="figure-1">
  {{< post-image src="assets/ultralytics-overview.webp" alt="Ultralytics 架构：Python 或 CLI 进入 YOLO 和 Model，经任务映射选择训练、验证、预测或导出执行器；检测预测从图像读取，经 LetterBox、RGB 转换和网络前向，到后处理及原图坐标结果" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">任务映射选择执行器，底部展开检测推理的数据流。后处理根据模型分支采用常规 NMS 或端到端结果筛选；训练、验证与导出是不同执行路径。</span></figcaption>
</figure>

## 3. 安装与第一次推理 {#quickstart}

### 3.1 在独立环境中安装本地源码

为了让解释与代码一致，可以从固定提交安装。以下为 Linux／macOS 的 Bash 示例，使用 Python 3.11；PyTorch 与 CUDA 的组合应先按运行机器确定。已经有本地 checkout 时，跳过克隆步骤，把安装路径换成自己的源码目录。

```bash
git clone --no-checkout https://github.com/ultralytics/ultralytics.git ultralytics-src
git -C ultralytics-src checkout --detach 1a767da5e58467ab8fbc2407b600b7632033dbc5
python3.11 -m venv .venv-ultralytics
source .venv-ultralytics/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./ultralytics-src
python -m pip check
python -c 'import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)'
yolo checks
```

可编辑安装让修改源码后无需重新打包，但不会冻结依赖。需要复现实验时，还应记录提交、Python/PyTorch 版本、驱动、CUDA、依赖锁和训练配置。`ultralytics.__file__` 能帮助发现“运行时导入了另一个环境的包”。

本提交的项目元数据声明 Python `>=3.8`，但特定扩展、导出后端和平台 SDK 有额外要求，不能据此假定任意 Python／平台组合都可运行全部功能。

### 3.2 Python 与命令行

下面使用仓库自带的 `bus.jpg`。第一次加载标准权重名时，若本地没有权重，框架可能联网下载；离线环境应提前准备可信来源的权重并传入实际路径。

```python
from ultralytics import YOLO
from ultralytics.utils import ASSETS

model = YOLO("yolo26n.pt")
results = model.predict(
    source=ASSETS / "bus.jpg",
    imgsz=640,
    conf=0.25,
    device="cpu",
    save=True,
    project="runs/demo",
    name="bus",
)
for result in results:
    print(result.orig_shape, result.speed)
    print(result.boxes.xyxy if result.boxes is not None else "no boxes")
```

等价的 CLI 形式使用 `参数=值`，不是给每个选项加 `--`：

```bash
yolo detect predict model=yolo26n.pt source=ultralytics-src/ultralytics/assets/bus.jpg imgsz=640 conf=0.25 device=cpu save=True project=runs/demo name=bus-cli
```

`n/s/m/l/x` 通常表示同一系列的不同模型规模，不能把字母当作固定硬件速度保证。先用小模型贯通输入输出，再在同一数据、分辨率和运行时下比较规模。

Python 推理与 CLI 的保存默认值也有差别。本提交的 `Model.predict()` 让 CLI 默认保存，而 Python 默认不保存；显式传入 `save`、`project`、`name` 更容易追踪结果，已有目录还可能自动递增编号。

## 4. 网络：从多尺度特征到检测头 {#network}

### 4.1 Backbone、Neck 与 Head 怎样分工

以 YOLO11 的检测 YAML 为入口，可以把网络理解为三段：

1. **Backbone 提取特征。** 多次下采样让网络从局部边缘逐渐获得更大的感受野，通道表达更丰富的语义。
2. **Neck 融合尺度。** 上采样、拼接和再次下采样，让细粒度位置与深层语义交换信息。
3. **Head 预测位置和类别。** 在多个尺度上产生候选，随后解码成边界框。

YAML 文件把融合层也放在 `head:` 段中，因此概念上的 Neck 不一定单独命名为一个配置区。YOLO11 使用 `C3k2`、`SPPF`、`C2PSA` 等模块；修改模块前，要沿 `nn/modules/` 找到实现，而不是只凭名字猜测计算内容。

标准 P3/P4/P5 检测层的步幅为 8、16、32。若网络实际输入恰好为 `640×640`，对应特征图空间大小为 `80×80`、`40×40`、`20×20`，合计 8,400 个空间位置。这是**特征位置数**，不是“检测到 8,400 个物体”。矩形推理、不同输入尺寸和不同结构都会改变这个数字。

在多个尺度上预测，是为了同时处理不同大小的目标：更细的空间网格保留位置细节，更粗的网格具有更强的语义与更大的感受野。不过某个目标最终由哪些位置负责，还要由训练时的分配器决定，不能把 P3/P4/P5 当成互不重叠的三个目标尺寸区间。

### 4.2 结构 YAML 不是普通超参数表

每层通常写成 `[from, repeats, module, args]`：

```yaml
# 选自网络结构的写法示例
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
```

`-1` 表示前一层，`from` 也可以是多个层索引；`repeats` 和通道数还会受深度、宽度缩放影响。`nn/tasks.py` 中的 `parse_model()` 负责解析模块、推导通道和建立连接。

`YOLO("yolo26n.yaml")` 按结构创建网络，不能等同于加载已训练模型；`YOLO("yolo26n.pt")` 则从检查点载入模型。只改 YAML 中的模块名通常还不够，新模块必须被解析器识别，并正确处理输入输出通道及保存连接。

### 4.3 YOLO11 与 YOLO26 必须分开理解

在本文提交中：

| 比较点 | YOLO11 检测配置 | YOLO26 检测配置 |
| --- | --- | --- |
| 结构主线 | 多尺度卷积特征与融合 | 同样有多尺度结构，但若干模块参数不同 |
| 框回归 | 默认检测头使用分布回归，`reg_max=16` | YAML 显式设置 `reg_max: 1` |
| 端到端开关 | 不默认启用端到端头 | YAML 设置 `end2end: True` |
| 训练分支 | 常规检测监督 | 包含 one-to-many 与 one-to-one 分支 |
| 推理筛选 | 常规分支需要 NMS 去重 | 本提交默认仍选择常规 NMS；显式 `nms=False` 才选择可用的端到端头 |
{.table-readable}

在本提交的 `Detect.forward()` 中，训练阶段 one-to-one 分支使用 `detach()` 后的特征，因此该分支的损失不会沿这条路径反传到骨干；它仍然更新自身头部参数。one-to-many 分支继续提供共享特征学习的监督。不能仅凭“双分支”就假定两个损失都以相同路径更新骨干。

YOLO26 去掉 DFL，不表示整个框回归损失都消失了。`BboxLoss` 仍计算 CIoU；`reg_max<=1` 时，距离回归改走按图像尺度归一化的 L1 损失。当前 `v8DetectionLoss.loss_names` 也会相应选择 `l1_loss`，而 YOLO11 的分布回归记录为 `dfl_loss`。

官方 [YOLO26 文档](https://docs.ultralytics.com/models/yolo26/)也介绍了端到端推理与检测头变化；这里的具体实现应继续对照固定提交，尤其不要把“省去 NMS”误解为“不需要任何后处理”。

### 4.4 原始张量、解码候选与最终框是什么关系

固定 `640×640` 输入、COCO 的 80 类和 batch 大小 $B$，可以用下表追踪普通检测头。这里列的是张量轴含义，不代表任意导出文件都采用相同布局。

| 阶段 | YOLO11 常规头 | YOLO26 常规头 |
| --- | --- | --- |
| 距离回归原始值 | `B×64×8400`，四条边各 16 个区间 | `B×4×8400`，四条边各一个值 |
| 类别原始 logits | `B×80×8400` | `B×80×8400` |
| 解码后密集候选 | `B×84×8400` | `B×84×8400` |
| 后处理后单图结果 | `N×6`，N 随图像和阈值变化 | `N×6`，N 随图像和阈值变化 |
{.table-readable}

84 来自四个框坐标加 80 个类别分数。这里没有额外的独立 objectness 通道，因此不能直接沿用旧式 `5+nc` 张量解析或固定执行“objectness × class probability”。分数也不应直接解释成已经校准过的业务事件概率。

YOLO26 选择端到端头后，推理阶段进一步选择高分候选，常见输出是 `B×300×6`，六列依次是 `x1,y1,x2,y2,score,class_index`。300 对应常用的最大结果数设置；最终 `Results` 仍会按置信分数和类别筛选，实际 N 可以更小或为零。PyTorch 的非导出前向还可能把解码结果与原始预测一起返回，导出模式则有不同包装，读取底层返回值时应先确认当前模式。

## 5. 一次推理在源码中怎样流动 {#inference}

### 5.1 从调用到预测循环

对普通检测模型，主要路线是：

```text
YOLO(...)
  → Model.predict()
  → 按 task 取得 DetectionPredictor
  → BasePredictor.__call__()
  → stream_inference()
      → setup_source()：选择图像、视频或流加载器
      → preprocess()：尺寸、布局、颜色、数值范围
      → inference()：后端前向
      → DetectionPredictor.postprocess()
      → construct_results() / construct_result()
      → Results
```

本提交的 `Model.predict()` 合并模型覆盖项、预测专用默认值与当前调用参数，当前调用具有更高优先级。预测器通常会复用；影响初始化的参数变化又可能触发重建。因此，把模型对象当作没有状态的纯函数共享给并发业务，并不稳妥。多路服务应明确对象所有权、队列和工作线程／进程策略。

### 5.2 输入类型决定预处理责任

常见图像数组和张量的约定不同：

| 输入 | 常见约定 | 需要注意 |
| --- | --- | --- |
| 路径、PIL 图像 | 加载器负责转入统一图像流程 | 不要预先重复归一化 |
| OpenCV／NumPy 数组 | `H×W×3`、BGR、`uint8` | 图像流程会转换 RGB 与布局 |
| PyTorch 张量 | `B×3×H×W`、RGB、浮点 `[0,1]` | 按张量入口要求自行准备布局与尺寸 |
{.table-readable}

`BasePredictor.preprocess()` 的非张量分支先做尺寸变换，再转换为 BCHW、BGR→RGB、浮点化并除以 255。张量分支不会再做同样的 LetterBox 和颜色翻转。加载器可能对某些输入做检查或修正，但可靠的调用应主动满足约定，避免依赖警告后的自动处理。

### 5.3 LetterBox 怎样改变坐标

直接把宽图拉伸成正方形会改变物体比例。LetterBox 通常先等比例缩放，再填充边缘。设原图宽高为 $W,H$，目标为 $W_t,H_t$，在允许该缩放的条件下：

$$
\begin{aligned}
r&=\min\left(\frac{W_t}{W},\frac{H_t}{H}\right),\\
x'&=rx+p_x,\\
y'&=ry+p_y.
\end{aligned}
$$

其中 $p_x,p_y$ 是左侧与上侧的填充。还原框时必须减去填充，再除以比例。实际实现还涉及取整、是否居中、是否限制放大和步幅对齐。

举一个**固定方形填充**的手算例子：原图 `1280×720`，目标 `640×640`，缩放后 `640×360`，上下各填充 140。原图框 `(320,180,960,540)` 变成网络输入框 `(160,230,480,410)`。逆变换为：

$$
x=\frac{x'}{0.5},\qquad y=\frac{y'-140}{0.5}.
$$

这个例子明确假设固定方形输出。默认预测可能启用 `rect=True`，在支持的后端及批次形状下只填充到步幅要求，实际张量不一定是 `640×640`。因此不能在业务代码里永远减去 140，应该使用本次预处理的真实比例和填充。

同一张图如果使用步幅 32 的最小矩形填充，原本需要补的 280 行会先取 `280 mod 32 = 24`，上下各补 12，最终形状变成 `640×384`。此时同一个框是 `(160,102,480,282)`；两种输入的缩放比例都是 0.5，但平移量不同。

<figure class="article-figure" id="figure-2">
  {{< post-image src="assets/letterbox-coordinates.svg" alt="原图 1280×720 的同一目标框，缩放到 640×360 内容后，固定方形输入上下各补 140 像素，最小矩形输入上下各补 12 像素；两种填充产生不同网络坐标" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text">同一框经过两种 LetterBox 配置的精确坐标对照。各面板按相同显示宽度排版，像素单位以标注为准；橙色表示填充区，蓝色虚线表示目标框。</span></figcaption>
</figure>

像素缩放需要对新宽高取整，奇数填充还会分到上下或左右两侧。一般程序应记录并复用实际变换参数，不能只在结果出来后依据“理想方形”猜测。框坐标通常表示连续边界，面积用 `(x2-x1)*(y2-y1)`；与某些老代码的整数像素闭区间 `+1` 规则混用，会改变小框的面积和 IoU。

{{< ultralytics-geometry id="letterbox-experiment" >}}

先保持原图和目标边长不变，只切换最小矩形选项：缩放比例不变，但输入高度、填充面积与网络框的纵坐标都会变化。再尝试竖图、奇数宽高或只有一个像素宽的图像，观察取整后内容至少保留一个像素。逆变换结果来自保存的变换参数；它不评价网络有没有正确检测目标。

### 5.4 NMS 与端到端筛选

常规检测分支会为同一物体产生多个候选。NMS 按分数保留候选，并删除与已选框重叠过大的同类框；`agnostic_nms` 可以改变类别处理方式。两个框的重叠程度用交并比描述：

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

`conf` 控制候选置信分数门槛，`iou` 在这条路径中控制 NMS 重叠门槛。提高 `conf` 一般会减少候选，但也可能增加漏检；提高 NMS 的 `iou` 通常保留更多相邻框，不能把两者当成同一种“严格程度”。

本提交尤其需要区分**结构支持**与**运行选择**：`BasePredictor.setup_model()` 传入 `end2end=self.args.nms is False`，而 `cfg/default.yaml` 的 `nms` 默认是 `None`。因此默认预测选择常规分支及外部 NMS；对包含相应头的 YOLO26，显式调用 `model.predict(..., nms=False)` 才选择端到端分支。导出器也有类似选择，另有格式兼容例外；不能把 YAML 的 `end2end: True` 当作所有 API 调用的有效默认值。

检测预测器仍会调用名为 `non_max_suppression()` 的统一函数，但 `utils/nms.py` 对端到端输出有提前返回分支：按置信分数、类别和 `max_det` 筛选，不执行常规 IoU 去重。因此看到函数名还不能判定实际算法，也不能期待调节 `iou` 必然改变端到端结果。

`nms` 的三个值还会因运行模式而产生不同交付接口：

| 参数 | PyTorch 预测／验证 | 支持该设置的检测导出 |
| --- | --- | --- |
| `nms=None`，默认 | 常规头，框架在网络后做 NMS | 常规头，图外另做 NMS |
| `nms=False` | 选择可用的 one-to-one 头 | 导出可用的端到端头 |
| `nms=True` | 常规头，框架执行 NMS | 尝试把 NMS 包进导出图 |
{.table-readable}

这里的 `False` 对有 one-to-one 头的模型才有相应能力；不支持的模型或格式可能警告、回退或拒绝。特别是 `nms=True` 在预测中不表示“再额外做两次去重”，它与导出图内包含 NMS 的语义位置不同。

用三个构造候选可以分清这些开关。A 是类别 0、分数 0.95、框 `(0,0,100,100)`；B 是类别 0、分数 0.90、框 `(10,10,110,110)`；C 与 B 同位置，但类别为 1、分数为 0.85。假定三者都已通过 `conf` 筛选，A 与 B 的交集面积为 8,100，并集为 11,900，IoU 约为 0.680672。

| 常规 NMS 配置 | 保留候选 | 原因 |
| --- | --- | --- |
| 按类别，`iou=0.5` | A、C | A 抑制同类 B；不同类 C 保留 |
| 不分类别，`iou=0.5` | A | B、C 都被 A 抑制 |
| 按类别，`iou=0.7` | A、B、C | A 与 B 的重叠没有超过门槛；B、C 不同类 |
{.table-readable}

<figure class="article-figure" id="figure-3">
  {{< post-image src="assets/nms-decisions.svg" alt="同一组三个候选在按类别 NMS、跨类别 NMS 和较高 IoU 门槛下分别保留 A 与 C、只保留 A、保留全部；B 与 C 坐标相同但类别不同" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text">NMS 的构造反例：类别处理方式与 IoU 门槛分别影响去重。所有候选已经通过置信分数筛选；端到端分支的筛选不使用这里的 IoU 去重规则。</span></figcaption>
</figure>

### 5.5 不下载模型也能复算的最小例子 {#geometry-example}

下载 [detection_geometry.py](detection_geometry.py)，在保存它的目录运行：

```bash
python3 -B detection_geometry.py --check
```

这个文件只依赖 Python 标准库，会打印两种填充的输出尺寸、目标框坐标、检测标签及上面的 NMS 保留结果。`--check` 还检查极窄图像、奇数填充、坐标往返、空候选和非法数值。

也可以逐步使用其中的函数：

```python
from detection_geometry import Box, letterbox

original = Box(320, 180, 960, 540)
placement = letterbox(1280, 720, minimal=True, stride=32)
network_box = placement.forward(original)
print(placement.output_size)      # (640, 384)，顺序为宽、高
print(network_box)                # Box(160, 102, 480, 282)
print(placement.inverse(network_box))  # 回到原图框
```

它只计算居中、保持比例的几何关系和常规贪心 NMS，不读取图像、不运行网络，也没有覆盖拉伸、掩码、关键点、后端并列分数规则和全部导出行为。实际项目继续消费框架的 `Results`；这个小文件用于检查自己是否理解了变换与去重，不能替代完整预测器。

## 6. Results：业务代码应该消费什么 {#results}

`Results` 是框架的结构化边界。直接解析控制台字符串或绘制后的图片，会丢掉精确坐标与类别信息。

检测结果常用属性如下：

| 属性 | 含义 |
| --- | --- |
| `orig_shape` | 原图 `(height, width)` |
| `names` | 类别 ID 到名称的映射 |
| `boxes.xyxy` | 原图像素坐标的左上、右下角 |
| `boxes.xywh` | 原图像素坐标的中心、宽、高 |
| `boxes.xyxyn`、`boxes.xywhn` | 相对于原图宽高的归一化坐标 |
| `boxes.conf`、`boxes.cls` | 分数与类别 ID |
| `boxes.id` | 有跟踪结果时的 ID，可能为 `None` |
| `speed` | 预处理、前向、后处理的计时信息 |
{.table-readable}

检测预测器在构造 `Results` 时已经缩放框到原图，**不要再对 `boxes.xyxy` 做一次 LetterBox 逆变换**。分类任务的 `boxes` 则可能是 `None`；同样不能假设每种任务都有 mask、关键点或跟踪 ID。

```python
from ultralytics import YOLO
from ultralytics.utils import ASSETS

model = YOLO("yolo26n.pt")
result = model.predict(ASSETS / "bus.jpg", device="cpu", verbose=False)[0]
records = []
if result.boxes is not None:
    boxes = result.boxes.cpu()
    for xyxy, score, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        class_id = int(class_id)
        records.append({
            "class_id": class_id,
            "label": result.names[class_id],
            "confidence": float(score),
            "xyxy_pixels": xyxy.tolist(),
        })
print(records)
```

先把框统一移到 CPU，再转 Python 对象，可避免循环中多次零散 GPU 同步。空检测会得到空记录列表，不应当被业务误判为程序失败。结果还支持绘图、保存文本和结构化导出，具体可查看 `Results` 与其导出 mixin。

`result.plot()` 默认返回适合 OpenCV 使用的 BGR 图像；交给以 RGB 为约定的显示工具前，应处理颜色顺序。旋转框要读取 `obb` 对应表示，不能只取外接轴对齐框后还声称保留了方向。

## 7. 自定义数据训练：先保证监督正确 {#training}

### 7.1 最小检测数据集

对轴对齐目标检测，一个常用布局是：

```text
/data/parts/
├── images/
│   ├── train/frame001.jpg
│   └── val/frame101.jpg
├── labels/
│   ├── train/frame001.txt
│   └── val/frame101.txt
└── parts.yaml
```

`parts.yaml` 使用明确的数据根路径：

```yaml
path: /data/parts
train: images/train
val: images/val
names:
  0: bolt
  1: nut
```

一行检测标签的格式是 `class_id x_center y_center width height`，坐标按**原图宽高**归一化，不带置信分数。例如原图 `1000×500`，框为 `(100,50,300,150)`，标签是：

```text
0 0.2 0.2 0.2 0.2
```

类别编号从 0 开始，必须与 `names` 对齐。框的范围、非零尺寸、图像方向、标注遗漏，都应在训练前检查。分割多边形、姿态关键点、OBB 和分类目录有不同规范，不能直接复用这五列标签。

视频数据尤其要避免把相邻帧随机拆进训练集和验证集。应该根据采集批次、场景、相机或时间段做划分，让验证集真正检验部署时的变化。负样本也要检查是否确实没有目标，否则“背景”监督会教模型忽略物体。

### 7.2 微调与小规模冒烟运行

下面先跑一个 epoch，检查数据与执行链；它不是推荐训练预算，也不构成效果结论。

```bash
yolo detect train model=yolo26n.pt data=/data/parts/parts.yaml epochs=1 imgsz=640 batch=4 device=0 workers=2 project=runs/parts name=smoke
```

没有 CUDA GPU 时可改 `device=cpu`。确认图像／标签可视化、类别统计、损失和输出目录正常后，再建立正式配置，例如：

```python
from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")
    model.train(
        data="/data/parts/parts.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        seed=0,
        project="runs/parts",
        name="baseline",
    )

if __name__ == "__main__":
    main()
```

主入口保护适合需要启动数据加载子进程的平台。`epochs=100` 和 `batch=8` 在这里仅用于说明参数；实际预算取决于数据量、收敛情况与显存。默认 `optimizer=auto` 还会根据实现选择优化策略，不能只设置一个 `lr0` 就认定最终优化器和学习率一定按预想运行，应查看日志与保存的配置。

具体到本提交，`BaseTrainer.build_optimizer()` 在 `auto` 模式下按内部估算的迭代次数选择：超过 10,000 使用 MuSGD，否则使用 AdamW，并重新确定学习率。这个估算与数据量、配置 batch／名义 batch、epoch 数有关，不是直接读取模型名称；所以“加载 YOLO26”不等于“本次微调自动使用了论文中的 MuSGD 配方”。

MuSGD 在矩阵与卷积参数上结合 Muon 风格的方向变换与 SGD 更新，其他参数使用相应的普通更新路径。它与独立的纯 Muon 优化器有不同参数分组和步长约定，不能把另一篇教程的学习率直接搬过来。研究优化器时可接着读 [Muon 原理](../muon-optimizer/)，实现仍以本仓库的 `optim/muon.py` 为准。

### 7.3 训练循环有哪些状态

训练器组织数据加载、增强、前向、损失、反向、梯度累积、优化器更新、学习率、EMA、验证、早停和检查点。AMP 与多设备执行有独立设置；显存不足应先区分图像分辨率、物理 batch、模型规模和中间激活，而不是只看权重文件大小。可接着阅读[训练显存与分布式通信](../distributed-training-memory/)。

Mosaic、随机缩放和颜色增强能扩大训练变化，但也可能偏离真实场景。对于固定工业相机、精细边界和小目标，应检查增强后的图像，而不是把所有增强都开到最大。`close_mosaic` 控制末段是否关闭 Mosaic，效果仍需由独立验证确认。

训练目录通常保存 `args.yaml`、结果曲线、`results.csv` 和 `weights/`。`last.pt` 对应最近保存状态，`best.pt` 对应训练器采用的最佳验证 fitness。本提交普通检测的 `Metric.fitness()` 对 Precision、Recall、mAP50、mAP50-95 使用 `[0,0,0,1]` 权重，也就是按 mAP50-95 选择；这与训练损失或业务误报成本是不同的判断标准。

### 7.4 恢复训练与重新微调

```python
from ultralytics import YOLO

model = YOLO("runs/parts/baseline/weights/last.pt")
model.train(resume=True)
```

这适用于恢复**尚未完成、且保留训练状态**的运行。恢复需要优化器、调度、epoch、EMA 等状态，不能用只含推理权重的导出文件替代。正常训练结束后检查点可能被清理优化器状态，已完成训练的 `last.pt` 也不应默认当作可继续的中断现场。

这个版本还有一个需要主动查看日志的行为：如果 `resume=True` 对应的检查点缺少有效 epoch 或优化器状态，`Model.train()` 会警告并改成新训练，而不是恢复旧历史。确认启动日志中的恢复 epoch，并检查新保存的 `args.yaml`；不能因为调用没有抛异常就认为优化器历史已经接上。

如果只是以旧权重初始化新的数据或预算，应另开运行调用 `train(data=..., epochs=...)`，这属于微调，不等同于完整恢复。类别集合变化时还要核对检测头的类别语义；本提交有按类别名称重映射的配置，也不能省略名称检查。

## 8. 正样本分配与损失：框是怎样学出来的 {#loss}

检测训练没有为每个特征点预先指定唯一物体。分配器要先决定哪些位置负责哪些标注，然后损失才有比较对象。

`utils/tal.py` 的任务对齐分配综合分类预测与框重叠来挑选正样本。这里常见的“anchor point”是特征图上的参考点，并不等同于旧式预设多种宽高的 anchor box。分配过程还有候选区域、Top-K 与冲突消解等步骤，不能简化成“物体中心所在格子就是唯一正样本”。

小目标可能小到没有网格参考点落在原始框内。本提交在选择候选点时，会把短边小于 `stride_val` 的有效标注框围绕中心扩展到该大小；默认步幅 `[8,16,32]` 下，`stride_val` 为 16。这个扩展只用于构造候选区域，回归目标仍然是原始标注。比如中心 `(8,8)`、边长 2 的框，可以把 `(4,4)` 与 `(12,12)` 纳入候选范围，再参与后续筛选；它不会把真实物体的框永久改成 16 像素。

对常规 YOLO11 检测，可以概括为分类项、框重叠项与分布回归项的加权组合：

$$
\mathcal L=\lambda_{\mathrm{box}}\mathcal L_{\mathrm{box}}
+\lambda_{\mathrm{cls}}\mathcal L_{\mathrm{cls}}
+\lambda_{\mathrm{dfl}}\mathcal L_{\mathrm{dfl}}.
$$

DFL 把边到参考点的距离表示为离散区间上的分布，网络预测分布后再恢复距离；它不是类别预测上的普通 Focal Loss。例如一条边的监督距离为 2.3，落在区间 2 与 3 之间，对应的目标权重分别是 0.7 与 0.3：

$$
\mathcal L_{\mathrm{DFL}}(2.3)=-0.7\log p_2-0.3\log p_3.
$$

网络的距离解码则使用预测分布的期望。监督目标怎样分到相邻区间，与预测分布怎样还原距离，是相互关联的两个步骤。具体检测损失还包含正样本权重、归一化、批次缩放等细节，上面的总损失公式只用于辨认组成。

本提交 YOLO26 使用的 `E2ELoss` 同时计算 one-to-many 与 one-to-one 损失，再随训练进度改变二者权重；初始化约为 `0.8/0.2`，最终朝 `0.1/0.9` 变化。它与文件中另一个名为 `E2EDetectLoss` 的实现不能混为一谈，应该从 `DetectionModel.init_criterion()` 确认实际选择。

此外，one-to-one 分支的分配器在本版本使用 `tal_topk=7, tal_topk2=1`，不能只凭“one-to-one”名称就把完整实现写成简单的一行 Top-1。想研究训练细节时，必须把网络分支、分配器、损失权重和梯度路径一起看。

## 9. 验证：mAP、阈值与业务成本 {#validation}

```python
from ultralytics import YOLO

model = YOLO("runs/parts/baseline/weights/best.pt")
metrics = model.val(data="/data/parts/parts.yaml", split="val", device="cpu")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
```

Precision 关心输出中有多少是真的，Recall 关心真实目标找到了多少：

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN}.
$$

AP 汇总一个类别在不同分数阈值下的精确率—召回率表现；mAP 再对类别取平均。`mAP50` 在 IoU 0.5 下评价；`mAP50-95` 汇总 0.50 到 0.95、步长 0.05 的多个匹配标准，更严格地评价定位质量。

评估匹配还要求同类、同一张图，并且一个真实目标不能被重复认领成多个 TP。假设有两个真实目标，按分数排序后的结果依次是“正确框、同一目标的重复框、另一个目标的正确框、背景误报”，对应的 TP/FP 序列是 `TP, FP, TP, FP`：

| 保留前几个预测 | 累积 TP | 累积 FP | Recall | Precision |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1/2 | 1 |
| 2 | 1 | 1 | 1/2 | 1/2 |
| 3 | 2 | 1 | 1 | 2/3 |
| 4 | 2 | 2 | 1 | 1/2 |

这是评估匹配的构造序列，不是该版本 AP 积分结果的手算替代。它说明了为什么重复框不能提高召回，以及为什么一个固定阈值下的 Precision 不能代表完整 PR 曲线。

**验证匹配的 IoU 阈值与预测 NMS 的 `iou` 是不同概念。** 改变后处理去重门槛，不是在把报告中的 mAP50 自动改成另一个指标。

评估通常保留更低分数的候选以形成曲线。本提交普通检测验证的默认 `conf` 与预测默认值不同；不要为了让可视化干净而把很高的部署阈值直接用于标准 AP 比较。部署阈值应在验证集上结合漏检成本、误报成本和每类需求确定，然后在独立测试集报告结果。

日志中汇总的 P、R 也不应直接理解为“部署 `conf=0.25` 时的表现”：当前 `ap_per_class()` 会在 IoU 0.5 对应的曲线上，根据平滑后的平均 F1 选择一个共同置信阈值，再提取 P、R。业务阈值若不同，就要在相同验证集上重新统计该工作点的误报和漏检。

只看一个总 mAP 还不够。要按类别、目标尺寸、遮挡、光照、相机、场景检查错误；少数关键类别即使平均指标很好，也可能完全不满足业务。固定随机种子有助于排查，但不能保证跨设备、算子版本和多进程执行逐位一致。

还要确认验证集实际包含哪些类别。当前 AP 计算以存在真实标注的类别建立统计项；`names` 中列出了某个类别，不代表验证集已经评价过它。训练验证与部署若选择不同检测头，应显式使用对应的 `nms` 参数重新验证，例如部署端到端头时运行 `model.val(..., nms=False)`，不能直接沿用另一分支的分数。

## 10. 视频与跟踪：识别类别之外还要维护身份 {#tracking}

### 10.1 `stream=True` 解决什么问题

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
for result in model.predict(source="input.mp4", stream=True, device="cpu"):
    count = len(result.boxes) if result.boxes is not None else 0
    print(result.path, count)
```

默认非流式调用会收集结果列表；`stream=True` 返回逐步产生结果的生成器，适合长视频和实时源。这个开关决定结果怎样交给调用者，是否重叠解码与推理还要看加载器、设备与预取条件：本提交会在满足条件的 CUDA 图像／视频输入上使用 `_prefetch()`，实时流加载器也有各自的读取线程。生成器不会消除业务自己把所有 `result` 放进列表造成的增长。要真正开始处理，还需要迭代它。[官方推理文档](https://docs.ultralytics.com/modes/predict/)提供了输入类型和流式接口说明。

直播源还应区分“逐帧不丢”与“尽快处理最新帧”。`stream_buffer` 涉及实时源的缓冲策略，`vid_stride` 控制采样间隔；吞吐跟不上摄像头时，持续积压会让显示 FPS 看似稳定、控制信息却越来越旧。应同时记录采集时间戳和消费时间。

### 10.2 跟踪器不是分类器记忆

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
for result in model.track(
    source="input.mp4",
    tracker="bytetrack.yaml",
    stream=True,
    device="cpu",
):
    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        print(boxes.id.cpu().tolist(), boxes.xyxy.cpu().tolist())
```

跟踪在相邻帧检测之间建立关联，用运动、重叠或外观线索维护轨迹。ByteTrack、BoT-SORT 是常见选择；本提交还注册了其他跟踪器，具体列表见 `trackers/track.py`，不能沿用旧教程“只支持两种”的结论。

ByteTrack 会分别利用高分候选与低分候选完成关联。本提交 `Model.track()` 未显式指定 `conf` 时使用 0.1，让跟踪器能收到低分检测；`bytetrack.yaml` 中还分别设置 `track_high_thresh`、`track_low_thresh` 与 `new_track_thresh`。若把检测入口的 `conf` 提得过高，部分用于恢复已有轨迹的低分框会在关联前被删掉，因此“可视化更干净”不一定意味着 ID 更稳定。

若自己用 OpenCV 逐帧调用 `model.track(frame, persist=True)`，`persist=True` 表示保留前一次调用的轨迹状态，调用者必须保证帧来自同一连续序列。切换视频或相机时应重置跟踪状态或使用独立对象；也不能把多路摄像头的数字 ID 直接视作全局人员／零件身份。

遮挡、交叉运动、低帧率与漏检都可能导致 ID 切换。计数应用需要定义穿线方向、重复计数规则与轨迹终止策略；单帧检测 mAP 不能代替时序效果验证。

## 11. 导出与部署：计算图只是交付的一部分 {#export}

### 11.1 先建立 ONNX 基线

```python
from ultralytics import YOLO

model = YOLO("runs/parts/baseline/weights/best.pt")
artifact = model.export(format="onnx", imgsz=640, dynamic=False)
print(artifact)
```

如果希望导出 YOLO26 的端到端头，可在支持该分支的格式中显式设置 `nms=False`；默认导出不能仅凭模型系列名推断输出分支。

导出可能需要额外依赖，并受网络、设备与运行时版本影响；应在单独的构建环境准备这些工具。`dynamic=False` 便于建立固定输入的对照，动态形状则需要验证实际支持的尺寸和 batch 范围。

为了先检查框架内的后端一致性，可以再加载导出的文件：

```python
from ultralytics import YOLO
from ultralytics.utils import ASSETS

model = YOLO("runs/parts/baseline/weights/best.onnx", task="detect")
results = model.predict(ASSETS / "bus.jpg", imgsz=640, device="cpu")
print(results[0].boxes)
```

这条路径仍使用 Ultralytics 的预处理与结果封装。若改用裸 ONNX Runtime、C++ 或其他 SDK，就需要自己实现并验证相同的颜色顺序、张量布局、数值范围、LetterBox、输出解码及原图坐标还原。

### 11.2 不要硬编码一个“通用 YOLO 输出形状”

常规检测导出可能暴露密集预测张量，端到端导出可能暴露已筛选的框行；模型、任务、导出参数、是否包含 NMS 都会改变接口。不能把网上的 `(1,84,8400)` 解析代码直接用于所有模型。

部署前至少确认：

1. 输入名称、dtype、形状和动态范围。
2. 输出名称、轴含义、坐标表示、分数与类别编码。
3. 图内是否已经包含解码、筛选或 NMS。
4. 标签映射、训练分辨率与预处理参数是否随模型一起交付。
5. 同一批验证图像在 PyTorch 与目标运行时上的框、分数和指标差异。

端到端检测仍可能需要分数过滤和坐标变换；额外套用一份旧 NMS 实现则可能改变模型原有行为。最可靠的方法是检查具体导出图与实际输出，再与框架推理结果逐项对照。

### 11.3 TensorRT、量化与测速

在支持的 NVIDIA 环境，可尝试 TensorRT 导出。本提交把精度选项统一为 `quantize`，旧的 `half`、`int8` 参数走弃用兼容处理；新示例可直接写：

```python
from ultralytics import YOLO

model = YOLO("runs/parts/baseline/weights/best.pt")
artifact = model.export(format="engine", device=0, imgsz=640, quantize=16)
print(artifact)
```

FP16 与 INT8 的速度、精度及支持程度取决于目标设备与后端。INT8 还需要代表性校准数据及相应导出配置；校准集不能只挑容易图像，也不能只观察总体 mAP，应该检查小目标和关键类别的变化。

TensorRT engine 需要按构建时的硬件与软件兼容条件管理，不能像纯数据文件一样假设到处可用。发布物应记录模型哈希、类别表、预处理、运行时、精度、形状约束与构建环境。

测速应同时报告 batch、分辨率、硬件、精度、预热和统计方法。`result.speed` 反映预测流程内的若干阶段，不能代表包括相机采集、视频解码、排队、CPU/GPU 传输、绘制与网络发送的全部端到端延迟。机器人应用还应关注输入时间戳、延迟尾部和丢帧策略，而不只是平均 FPS。

### 11.4 一个已经运行过的 CPU 对照

本文使用官方 `v8.4.0` release 中的 `yolo26n.pt`，在仓库自带的 `bus.jpg` 上复核了两种检测头。设置为 `imgsz=640`、`rect=False`、`conf=0.25`、batch 1、CPU FP32；导出使用 ONNX opset 17、固定形状与图简化。输入固定为 `1×3×640×640`，所以这里的输出形状也有明确前提。

| 检测头 | ONNX 原始输出 | 这张图的阈值后结果 | 对照范围 |
| --- | --- | --- | --- |
| 默认常规头 | `1×84×8400` | 5 个框 | 对照全部密集输出，并对照框架后处理结果 |
| `nms=False` 端到端头 | `1×300×6` | 5 个框 | 对照分数高于 0.25 的选中框，并对照框架结果 |
{.table-readable}

运行环境为 Python 3.10.20、PyTorch 2.9.0、ONNX 1.19.1、ONNX Runtime 1.23.2、onnxslim 0.1.82。普通头的原始输出存在约 `0.00450` 的最大绝对差，端到端已选中结果约为 `0.0000763`；这些值混合了坐标和分数维度，不能当作 mAP 或统一的误差单位。两条路径还分别检查了原图坐标下的最终框与分数，在设定的相对／绝对数值容差内相符。

这个检查说明这组权重、输入与 CPU 运行时的接口能够对上。它没有评估完整验证集，没有比较任意低分 Top-K 并列次序，也没有验证 TensorRT、GPU 或其他量化精度。不同头即使都返回五个框，其位置和分数也可能不同；不能据此认定两条分支在所有图像上等价。

## 12. 常见问题与修改源码的切入点 {#troubleshooting}

| 现象或需求 | 优先检查的位置与原因 |
| --- | --- |
| 训练没有有效框 | 标签格式、类别编号、原图归一化、图片与标签路径 |
| 推理框整体偏移 | LetterBox 比例／填充、重复坐标还原、宽高顺序 |
| 颜色或预测异常 | BGR/RGB、BCHW/BHWC、重复除以 255 |
| 视频占用持续增长 | 是否返回整段结果列表、业务是否累计保留结果、源缓冲 |
| 同一物体重复框 | 模型是否端到端、当前分支是否执行 NMS、阈值与类别策略 |
| 修改 `iou` 没效果 | 是否走端到端提前返回分支，是否混淆验证匹配阈值 |
| GPU 显存不足 | 输入分辨率、实际 batch、激活、增强和设备上的其他任务 |
| 导出成功但效果变化 | 预处理、输出轴、精度、动态形状、图内／图外后处理 |
| 希望新增网络模块 | `nn/modules/` 的模块实现与 `nn/tasks.py` 的解析规则 |
| 希望自定义损失或数据 | 对应任务 Trainer／Model criterion／Dataset 与变换链 |
{.table-readable}

尽量先用回调、继承执行器或单独的集成层完成扩展，减少直接改动公共训练循环。项目已有 tests，但新模块仍应检查训练与推理两种状态、梯度、batch 维、导出和目标后端；能在 PyTorch 中前向成功不代表能部署。

仓库标注 **AGPL-3.0**，官方也提供 Enterprise 授权。准备交付代码、权重或嵌入产品前，应结合实际分发与服务方式阅读[仓库 LICENSE](https://github.com/ultralytics/ultralytics/blob/1a767da5e58467ab8fbc2407b600b7632033dbc5/LICENSE)及[官方授权说明](https://www.ultralytics.com/license)，明确适用条款；不要把“可以 pip 安装”当作授权范围的判断依据。

## 阅读自测与验收

先独立作答或操作，再核对下方参考答案与解析；实践结果以实际记录为准。

1. Ultralytics 包版本、YOLO 模型系列、task 与 mode 分别决定什么？
2. 为什么统一 API 不意味着所有任务使用相同标签与损失？
3. NumPy 图像和 PyTorch 张量输入分别由谁负责颜色、布局与归一化？
4. 1280×720 图像固定填充到 640×640 时，框怎样变换和还原？
5. 为什么不能对 Results 的原图坐标重复做 LetterBox 逆变换？
6. YOLO26 的结构开关与预测参数 nms 怎样共同决定是否执行 IoU 去重？
7. YOLO26 去掉 DFL 后，框回归监督与 one-to-one 梯度路径发生了什么变化？
8. 新数据微调与恢复中断训练需要的状态有什么区别？
9. mAP50 的匹配标准与预测参数 iou 为什么不能混用？
10. stream=True、源缓冲与 persist=True 分别解决什么问题？
11. 导出后为什么还要验证预处理、输出轴、坐标和关键类别指标？
12. 模型内部计时为什么不等于摄像头到业务输出的端到端延迟？
13. 固定方形与步幅最小矩形为什么有相同比例、不同平移量？
14. 84 个候选通道是否包含独立 objectness，为什么旧式 5+nc 解析会出错？
15. optimizer=auto 为什么不保证 YOLO26 微调使用 MuSGD？
16. 单张图像的 ONNX 数值对照能证明什么，还不能证明什么？

实际验收可从一组人工核对过的图像开始：检查类别和原图框，完成独立数据划分，跑通小规模训练，再用相同图像对照 PyTorch 与导出后端。本文中的坐标与特征图数字是构造算例，不是训练效果或硬件性能测量；完整训练、视频关联和目标设备部署仍需在对应环境中执行。

<details>
<summary>参考答案与解析（16 项，顺序对应上文）</summary>

1. 包版本决定框架实现，模型系列决定结构与配方，task 决定检测/分割/姿态等任务，mode 决定 train/val/predict/export 等路径；必须记录完整组合，不能只写 YOLO。

2. 统一入口会按任务选择不同数据格式、头部、损失及执行器。五列轴对齐检测标签不能直接用于分割、关键点、OBB 或分类。

3. 常规 NumPy 图像路径由预测器做 BGR→RGB、HWC→CHW 与数值归一化；直接张量输入需调用者提供符合接口的布局、颜色和范围，不能假设同样自动转换。

4. 1280×720 固定到 640×640 时 r=0.5，上下各补 140；框 (320,180,960,540) 变成 (160,230,480,410)，逆变换先减填充再除以 r。

5. Results 已把框还原到原图；再次做逆 LetterBox 会重复去填充与放大。只有读取网络坐标的原始输出时才按真实预处理参数还原。

6. 固定提交中结构要支持 one-to-one 头，且显式 nms=False 才选相应端到端分支；默认 nms=None 走常规头与 IoU 去重。统一后处理函数对端到端输出提前筛选返回，函数名不能证明执行了 NMS。

7. 去 DFL 后仍有 CIoU 与按图像尺度归一化的 L1 回归；one-to-one 使用 detach 特征，更新自身头而不沿该路更新骨干，one-to-many 仍监督共享特征。

8. 新数据微调从权重开始新训练配置；恢复中断还需优化器、调度/训练步与相关状态，并按恢复入口处理。能加载权重不等于连续恢复同一训练轨迹。

9. mAP50 的 0.5 是预测与真值匹配阈值，预测参数 iou 是常规 NMS 候选间去重阈值；两者对象不同，不应互换。

10. stream=True 逐项产出结果避免一次积攒全部帧，源缓冲策略决定积压或丢帧，persist=True 用于跨调用保留跟踪状态。生成器本身不保证源低延迟或多视频 ID 正确隔离。

11. 导出后端可能改变输入、输出包装与后处理位置；应核对颜色/布局、归一化、轴顺序、框尺度、空结果及类别质量，再在目标设备测性能。

12. 模型计时只覆盖所定义的预处理/推理/后处理等内部阶段，摄像头采集、排队、通信和业务动作另有成本；端到端延迟从实际采样到业务结果计算。

13. 步幅最小矩形将总竖直填充 280 mod 32=24，上下各 12，输出 640×384；固定方形上下各 140。r 相同但平移不同，框还原必须使用实际参数。

14. 84=4 个框坐标+80 类分数，没有额外 objectness；旧式 5+nc 会错列或错算分数。端到端常见 B×300×6 又是另一种格式。

15. 固定版本 auto 按估算迭代数选择，超过 10000 用 MuSGD，否则 AdamW，并重设学习率；模型叫 YOLO26 不保证选择同一优化器，须看日志。

16. 单图 ONNX 对照验证该输入下的数值/接口一致性，不能证明全数据质量、动态尺寸、所有算子边界或设备性能；需更多代表性输入和独立质量评估。

</details>

**通过条件：** 坐标往返、NMS 分支及训练状态练习通过；用独立划分图像核对原图框与类别，再按目标后端对照，不能将单图一致性当作完整部署验收。


## 参考资料

- [本文对应的 Ultralytics 源码提交](https://github.com/ultralytics/ultralytics/tree/1a767da5e58467ab8fbc2407b600b7632033dbc5)：任务映射、网络、数据、损失和执行器的依据。
- [固定版本默认配置](https://github.com/ultralytics/ultralytics/blob/1a767da5e58467ab8fbc2407b600b7632033dbc5/ultralytics/cfg/default.yaml)：核对本版本参数名称、类型和默认值。
- [固定版本 YOLO11 配置](https://github.com/ultralytics/ultralytics/blob/1a767da5e58467ab8fbc2407b600b7632033dbc5/ultralytics/cfg/models/11/yolo11.yaml)与 [YOLO26 配置](https://github.com/ultralytics/ultralytics/blob/1a767da5e58467ab8fbc2407b600b7632033dbc5/ultralytics/cfg/models/26/yolo26.yaml)：多尺度结构与端到端开关。
- [官方文档](https://docs.ultralytics.com/)：查询具体任务、后端和部署平台的使用说明，阅读时注意与固定版本的差异。
