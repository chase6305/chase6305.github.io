---
title: "NVIDIA Video to Data 详解：从人类视频到机器人仿真、强化学习与 GR00T 数据"
date: 2026-10-15
lastmod: 2026-10-16
draft: false
tags: ["Video to Data", "CHORD", "Robot Learning", "Egocentric", "3D Vision", "Isaac Lab", "Reinforcement Learning", "VLA"]
categories: ["人工智能"]
authors: ["chase"]
summary: "沿源码拆解 NVIDIA Video to Data 的视频检索、三维重建、动作重定向、接触引导强化学习和 GR00T N1.7 数据链路，给出纸巾盒示例的运行步骤、数据接口、验收标准与适用边界。"
description: "沿源码拆解 NVIDIA Video to Data 的视频检索、三维重建、动作重定向、接触引导强化学习和 GR00T N1.7 数据链路，给出纸巾盒示例的运行步骤、数据接口、验收标准与适用边界。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
reading_prerequisites: "三维坐标变换、逆运动学、模仿学习与强化学习的基本概念；不要求熟悉全部视觉模型"
reading_focus: "按每阶段的输入、输出和验收条件阅读，重点区分人类运动、机器人参考轨迹、物理策略和策略训练数据。"
related_posts:
  - "/posts/ai/stable-hand"
  - "/posts/ai/vla-world-model-data"
  - "/posts/ai/lerobot-act"
  - "/posts/ai/molmo-motion"
---

人类拍下一段“拿起纸巾盒，再放回桌面”的视频，机器人能从中学到什么？视频记录了外观和运动，却没有直接给出机器人关节指令、物体质量、接触力，也没有保证这个动作适合另一种手掌、臂长和关节布局。

[NVIDIA Isaac Video to Data](https://github.com/nvidia-isaac/video_to_data)，简称 **V2D**，围绕这些缺口构建了一条数据生产链：先从视频恢复场景与人体运动，再把运动映射到机器人，在物理仿真中训练可执行的专家策略，最后采集机器人示范，供视觉语言动作模型训练使用。它的价值在于把感知、几何、机器人学与策略学习之间的接口做成可检查的工程流程。

本文依据 **2026-10-15 核对的提交 `193382a18430`**，正文源码链接固定到这一版本。仓库包含三类可独立使用的组件：`video_ingestion_agent/`、`reconstruction/`、`robotic_grounding/`；根目录的 `run_e2e.sh` 则串起一条从第一视角视频到 GR00T N1.7 的完整示例。**这条 E2E 示例可以直接使用指定视频，不要求先运行视频检索 Agent。** [版本与入口](https://github.com/nvidia-isaac/video_to_data/tree/193382a184308f6800331f3db829f76dc4294cfe)

<figure class="article-figure" id="fig-v2d-pipeline">
  {{< post-image src="assets/video-to-data-pipeline.webp" alt="V2D 数据流：人类视频依次经过场景与人手重建、机器人动作重定向、物理仿真强化学习、专家示范采集、GR00T 微调和闭环评估；视频检索是可选的上游片段选择步骤" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">从上排左侧顺箭头读到下排左侧：重建交付几何与人体运动，重定向交付机器人参考运动，仿真训练交付专家策略，专家 rollout 才产生机器人观测与动作。图中省略环境安装、数据审计与部分中间文件；机器人外形用于说明模块职责，具体关节与相机由 embodiment contract 定义。</span></figcaption>
</figure>

## 阅读路线 {#reading-map}

- **理解数据链：** 从[各阶段交付物](#outputs)进入[重建](#reconstruction)、[重定向](#retargeting)、[物理训练](#grounding)与[GR00T 数据](#groot-data)，追踪视频怎样变成机器人观测与动作。
- **核对学习接口：** 对照[专家观测](#expert-observations)、[VOC 辅助课程](#voc-curriculum)、[状态与动作布局](#fig-state-action-layout)、[控制时钟](#fig-control-clocks)和[高度判定](#fig-height-criteria)，区分策略输入、训练标签与成功条件。
- **动手运行与评估：** 结合[环境要求](#requirements)和[纸巾盒 E2E](#e2e)，逐阶段验收；再用[官方结果](#results)与[故障定位](#failure-modes)判断证据覆盖到哪一层。

几个词贯穿全文：**重建**估计拍摄时发生了什么；**重定向**把人体运动目标映射到机器人；**grounding**让运动目标接受物理接触与动力学的检验；**rollout**是策略与环境交互产生的一段轨迹；**闭环**指执行后继续观察，并据此计算下一步动作。

## 1. V2D 交付的究竟是什么 {#outputs}

同样叫“数据”，不同阶段的文件具有完全不同的语义。

| 阶段 | 输入 | 主要交付物 | 可以回答的问题 |
| --- | --- | --- | --- |
| 视频入库与检索 | 长视频、查询文字 | 动作片段、实体关系、帧向量、时间边界 | 哪段视频包含想要的交互？ |
| 三维重建 | RGB 或经过标定的多视角数据 | 相机参数、深度、掩码、物体网格与位姿、人体或人手运动 | 场景中什么在什么位置运动？ |
| 动作重定向 | 人体运动、机器人模型 | 机器人关节与末端参考、物体运动、支撑面 | 这种身体如何近似完成该运动？ |
| 物理强化学习 | 参考运动、仿真场景、奖励与控制配置 | 能与物体交互的专家策略 | 机器人在动力学约束下怎样执行？ |
| 示范与策略训练 | 专家 rollout、相机与动作约定 | HDF5、LeRobot 数据、GR00T checkpoint | 视觉策略怎样从观测生成机器人动作？ |
{.table-readable}

这里最容易误解的是第三、四行。**机器人参考轨迹描述“希望到哪里”，策略描述“看到当前情况后怎样行动”。** 把参考关节角逐帧写进可视化器，能得到流畅动画；物理执行还要处理重力、摩擦、碰撞、执行器和控制误差。V2D 的 Robotic Grounding 正是在这个间隙里工作。

另一个区别是训练目标：V2D 的检索模块没有训练机器人控制器；重建模块没有直接输出 GR00T 动作标签；专家 RL 和后续 GR00T 微调也使用不同的观测与学习方式。阅读目录时，应先确认自己停在哪一个接口。[Robotic Grounding 架构](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/docs/ARCHITECTURE.md)

## 2. Video Ingestion Agent：先把长视频变成可查询片段 {#ingestion}

### 2.1 为什么需要图数据库和向量数据库

假设积累了几百小时操作视频，现在要找“双手拿起一个盒子”的片段。逐个视频做完整三维重建代价很高；只搜索文件名，也无法表达手、对象与动作的关系。

入库模块先用视觉语言模型切分动作，再抽取实体和关系，同时计算视觉特征。它保存两份 SQLite 数据库：

- `graph.db` 保存人物、物体、位置、关系以及动作片段的时间范围，适合查询“谁对什么做了什么”。
- `vector.db` 保存 SigLIP-2 帧嵌入，适合根据视觉语义找到相似画面；帧上的 `segment_id` 再把结果映射回动作边界。

两者解决的是不同的信息需求：一张手靠近杯子的图片，未必能说明后面发生了“拿起”；一个动作文字描述，也未必包含所有外观细节。检索 Agent 会结合图搜索与帧搜索，必要时逐级放宽动作或对象条件，再综合结果。[入库模块说明](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/video_ingestion_agent/README.md)

### 2.2 verify/refine 是可用能力，不是默认已开启

LangGraph 编排的入库流程可以概括为：动作切分，检查片段是否符合动作描述，必要时重新细化，提取实体关系，生成帧嵌入，再写库和输出报告。

但本版本的 `configs/ingestion.yaml` 把 `enable_verification` 与 `enable_refinement` 都设成了 `false`。默认 VLM 是 `Qwen/Qwen3-VL-8B-Instruct`，模型取帧参数为 4 fps，处理块长度为 15 秒、重叠为 1.5 秒。这些是**视频理解模块的采样与切块设置**，不能当成重建帧率或机器人控制频率。

如果要启用完整检查循环，应复制配置、将两个开关设为 `true`，再按接口设置最大迭代次数。检查更严格时，也可能没有任何片段被接受；空数据库需要结合日志分析，不能立即归因于向量模型失效。[默认配置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/video_ingestion_agent/configs/ingestion.yaml)

### 2.3 最小运行方式

下面是独立的检索路径；从仓库根目录进入子目录，使用模块自己的环境。

```bash
cd video_ingestion_agent
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras
source .venv/bin/activate

# 以后台进程启动模型服务，并等待健康检查。
python scripts/serve.py -c configs/ingestion.yaml
python scripts/serve.py --status
```

确认服务就绪后，在同一目录与环境中继续执行：

```bash
python scripts/run_ingestion.py /absolute/path/to/demo.mp4 \
  -c configs/ingestion.yaml
python scripts/run_retrieval.py "Find clips where someone picks up a box" \
  -d outputs/ -c configs/retrieval.yaml
```

这里按默认配置使用 `outputs/` 数据库目录。`-o` 指定的运行报告目录与数据库目录不是同一个概念，改路径时要同时检查配置。`GIT_LFS_SKIP_SMUDGE=1` 用于避开该版本服务依赖中的 LFS checkout 问题；项目本身需要的视频和机器人资产仍须另外拉取。模型服务不再使用时，可用 `python scripts/serve.py --stop` 释放对应 GPU 资源。[服务管理脚本](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/video_ingestion_agent/scripts/serve.py)

检索返回的片段只是重建候选，语义匹配并不保证遮挡、视角和接触过程适合后续重建。

## 3. Reconstruction：从像素恢复物体、相机与人手 {#reconstruction}

### 3.1 重建模块是工具箱，需要选择一条路径

`reconstruction/` 按模块封装检测、分割、深度、网格、位姿和人体估计，主机端 Python 负责编排，各模型在自己的 Docker 镜像里推理。中间结果写到磁盘；`v2d_common` 提供深度、相机内参、三维变换、框和掩码等类型。

对 RGB 人类示教，以下模块各自负责一类信息；具体编排由所选路径决定：

| 需要恢复的信息 | 对应组件举例 | 下游为什么需要它 |
| --- | --- | --- |
| 目标区域 | Grounding DINO、SAM2 | 确定要跟踪的物体及其可见轮廓 |
| 深度与几何尺度 | MoGe、UniDepth；双目场景可用 FoundationStereo | 为表面、尺度和三维对齐提供约束 |
| 物体模型 | SAM3D 或已有网格；部分多视角路径使用 BundleSDF | 提供物体表面、可视化与碰撞资产的来源 |
| 逐帧物体位姿 | FoundationPose | 描述物体在视频中的平移和旋转 |
| 人手运动 | HaMeR、DynHaMR、HaWoR 等路径 | 提供腕部、手指和交互参考 |
| 相机运动与重力 | DROID-SLAM、GeoCalib 等可选处理 | 把运动统一到时间一致的世界坐标 |
{.table-readable}

第一视角视频、多相机 rosbag 和双目物体扫描有不同的输入要求与编排。比如多视角 HOI 路径会涉及相机标定、人体和物体重建；E2E 的纸巾盒示例走的是单目第一视角路径。不能把多视角命令中的外参文件、已知物体网格，默认为任意 MP4 都自带。[重建模块目录](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/README.md)

### 3.2 相机在动，坐标不能跟着混用

用 $W$ 表示固定世界系，$C_t$ 表示第 $t$ 帧相机系，$O$ 表示物体局部系。若位姿跟踪器给出物体到当前相机的变换，则世界系物体位姿满足：

$$
T_{W\leftarrow O}(t)
=T_{W\leftarrow C_t}(t)\,T_{C_t\leftarrow O}(t).
$$

这是坐标组合关系，不是某个视觉模型的学习公式。左右手也必须进入相同的固定坐标系，才能比较它们与物体的距离。若把每帧相机系中的手姿直接当成世界轨迹，相机抬头或走动就会被误当成手和物体的运动。

此外，单目尺度、网格尺度和手模型尺度必须一致。**重力对齐只确定方向，不能单独解决绝对尺度，也不会自动得到机器人基座标定。** 源视频看起来对齐，并不排除世界坐标中的长程漂移。

### 3.3 本版本第一视角入口做了哪些整合

统一入口是 `reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py`，当前支持三种手部路径：

- `--hand_tracking dynhamr`：原有 ViPE/DynHaMR 链路。
- `--hand_tracking hamer`：WiLoR/SAM2/HaMeR 链路，可增加 DROID-SLAM、GeoCalib 重力对齐与 gsplat refinement。
- `--hand_tracking hawor`：复用对象/SAM2 处理链，以 HaWoR 作为主要手部轨迹来源。

根目录 `run_e2e.sh reconstruct --hand-tracking` 当前只暴露 `hamer` 与 `dynhamr` 两项；本文纸巾盒示例使用 `hamer`。底层模块支持的选项，需要与根启动器已经接入的选项分别核对。

这些名字对应不同职责：手检测与估计、掩码跟踪、相机轨迹、世界坐标对齐、渲染驱动细化。开启 gsplat refinement 是增加一个优化阶段，不能据此认定输出已经具备正确质量、摩擦或接触力。

WiLoR 路径还处理了“这一帧的手属于哪条轨迹”：它渲染检测得到的 MANO 轮廓，再与 SAM2 传播掩码做 IoU 匹配，关联逐帧检测与持续跟踪 ID。缺失帧可以插值填补，相关中间 JSON 用 `interpolated` 标记区分；平滑、无缺帧的轨迹因此不一定每一帧都来自有效视觉检测。[检测、关联与填补实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/modules/v2d_pipelines/run_ego_wilor.py)

若提供已知物体网格，可以替换生成几何的一部分工作；对象文字提示仍会用于检测、分割和位姿初始化。导入真实尺寸网格时，是否继续估计尺度也应明确选择。[统一入口源码](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py)、[第一视角设置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/docs/ego_e2e_setup.md)

### 3.4 重建结果怎样验收

根目录 E2E 启动器把最终 bundle 放在运行目录的 `reconstruction/result_slam_gravity_aligned/`，其中需要有非空的 `result.npz`、`mesh.obj`、`manifest.json` 和 `threejs_scene/index.html`。检查文件存在只是第一步，还应看：

1. 开始、抓取、搬运和放回时，物体模型能否持续覆盖目标。
2. 世界坐标中桌面与静止背景是否稳定，手和物体是否一起漂移。
3. 物体尺寸、手掌尺寸、支撑面高度是否处于合理比例。
4. 遮挡时的姿态是否跳变，左右手身份是否交换，接触附近是否明显穿透。
5. 关键接触片段是否主要由插值补齐，检测覆盖与填补区间是否有记录。

这些是下游任务所需的几何条件。发现错误后，应回到具体模块、帧段和缓存输出定位，而不是直接增加策略训练步数。

## 4. Retargeting：把人体运动转换为机器人参考 {#retargeting}

### 4.1 不能直接复制人手关节角

MANO 等人体模型与机器人具有不同的骨架、手掌尺度、自由度和限位。一个人体手指弯曲参数不等于某个机器人电机角度。重定向通常比较关键点、末端位姿和相对关系，再通过机器人正运动学与逆运动学寻找合适的关节配置。

为了说明这一类问题，可以写出一个通用 IK 目标：

$$
\min_q\;\sum_i w_i\left\|f_i(q)-\hat x_i\right\|^2
+\lambda\left\|q-q_{\mathrm{prev}}\right\|^2,
\qquad q_{\min}\le q\le q_{\max}.
$$

其中 $f_i(q)$ 是机器人关键点或任务空间量，$\hat x_i$ 是经过坐标与尺度处理的目标；后一项抑制逐帧跳变。这是帮助理解重定向的概括，**不是仓库所有机器人共用的一条完整损失**。具体实现还取决于手型、手腕或全身约束、关节映射与碰撞处理。[重定向配置](https://github.com/nvidia-isaac/video_to_data/tree/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/retarget)

Robotic Grounding 也能从 ARCTIC、HOT3D、TACO、GRAB 等已有运动数据进入，不必每次先从原始视频做视觉重建。数据加载阶段可把 MANO 参数变成关节和物体运动，再生成物体 URDF、机器人运动与仿真支撑面。这里只能复用相同的数据语义；不同数据集的格式、许可与预处理仍要分别处理。

### 4.2 运动文件的约定比扩展名更重要

项目定义了 `motion_v1` Parquet 接口，包括时间、机器人身份、末端、物体和可选接触信息。读文件时至少应核对：

- 时间轴在第一维，`fps` 明确且为正，机器人与物体的帧数一致。
- `ee_pose_w` 是世界系末端位姿，位置与四元数合成七维；物体也有对应的世界系位姿。
- 四元数约定是 **`wxyz`**，位置加旋转的顺序为 `[x, y, z, qw, qx, qy, qz]`。
- `robot_joint_positions` 的每列必须对应 `robot_joint_names`，不能只检查总维数。
- `motion_kind` 区分 `single_robot` 与 `dual_hand`，两类要求的字段不同。

归一化四元数不能证明其排列正确：交换分量后，它仍可能是单位四元数。关节总数一致也不能证明次序正确：左右手或不同手指错位时，数组照样能加载。[运动格式说明](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/motion_schema/README.md)

### 4.3 Vega 示例的 QA 检查什么

本文使用 `vega_sharpa_joint` 约定。重定向验收应同时检查分区运动文件、支撑场景，以及 `vega_retarget_qa/` 中的三类产物：

| 产物 | 主要内容 | 阅读重点 |
| --- | --- | --- |
| `<id>_vega_sharpa_report.json` | 收敛帧数、迭代次数、位置与姿态误差、尺度、警告 | 查看 mean、p95、max 与 `worst_frame`，不只看平均误差 |
| `<id>_vega_sharpa_ik.npz` | 原生帧率下的 IK 结果、目标、关节名与变换 | 回到最坏源帧核对目标和机器人 FK |
| `<id>_vega_sharpa.mp4` | 整段机器人、物体与接触的验证画面 | 检查完整交互，特别是抓取、遮挡和放回 |
{.table-readable}

`position_error_m` 的单位是 m，`active_orientation_error_rad` 的单位是 rad；`converged_pct` 是 IK 求解器的收敛比例。它们各自描述几何优化，不是操作成功率。报告的源帧率、输出参考帧率与 GR00T 的 20 Hz 控制频率也要分别记录。

还应核对有限值、帧数、发布的 **58 关节顺序**和 `warnings`。`*_is_valid=False` 的帧可能保存有限但陈旧的姿态；部分物体资产或视频导出失败也会写警告后继续生成其他文件。验收需要确认物体资产和完整视频实际存在。[Vega 重定向与 QA 实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/scripts/retarget/ego_recon_to_dexmate_sharpa.py)

该脚本自动新建物体 URDF 时默认质量为 0.3 kg，并按网格包围盒近似惯量；复用已有资产时参数来自对应 URDF。这些是仿真设定，应与几何重建结果分别核对。

这 58 维属于当前 Vega 双臂与双手约定，不能推广成所有 Sharpa、Dex3 或 G1 的通用维数。新增机器人需要新的模型、映射、控制与观测配置，以及相应 contract；更换 URDF 文件名只覆盖其中很小一部分。

## 5. Robotic Grounding 与 CHORD：让动作接受接触物理的检验 {#grounding}

### 5.1 为什么低 IK 误差仍可能抓不起来

同一组手指位置，可能因为物体尺度、表面法向、摩擦和执行器响应不同而产生完全不同的结果。手在物体旁边，未必形成有效支撑；接触位置接近示教，也未必能抵抗让物体翻转的力矩。

因此，物理训练要让策略在当前状态和运动参考下输出动作，由仿真器推进机器人与物体，再根据运动跟踪、接触等信号学习。Isaac Lab 路径使用 RSL-RL/PPO；仓库另有独立的 FlashCHORD 路径，见第 9 节。

<figure class="article-figure" id="fig-reference-policy">
  {{< post-image src="assets/reference-to-policy-v2.webp" alt="机器人运动参考提供策略条件与跟踪目标，人手接触参考单独进入接触奖励；当前状态经过策略产生动作，物理仿真返回下一状态，奖励经训练过程更新策略参数" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text">机器人运动参考与人手接触参考承担不同职责；物体如何运动，由动作、控制器和物理环境共同决定。实线表示 rollout 信息，绿色虚线表示训练参数更新，省略 critic、优化器、课程和具体观测项。图示为接触参考匹配分支，纸巾盒 Vega 配方采用的 force-closure 分支见第 5.4 节；后续 GR00T 接口另行定义。</span></figcaption>
</figure>

### 5.2 接触 wrench 比“手指离目标点多远”多表达了什么

[CHORD](https://nvidia-isaac.github.io/video_to_data/chord/) 是仓库相关的接触引导灵巧操作工作，其核心是以物体为中心比较接触能产生的力与力矩。这里的 **wrench** 是六维力旋量，可用一个接触的基本关系理解：

$$
w=\begin{bmatrix}f\\\tau\end{bmatrix},
\qquad \tau=r\times f.
$$

若 $f$ 是以 N 计的实际力，$r$ 是相对参考点、以 m 计的位置，则 $\tau$ 的单位是 N·m。比如 $r=(0.05,0,0)$ m，$f=(0,2,0)$ N，得到 $\tau=(0,0,0.1)$ N·m。这是力学计算示例，不是视频测得的接触力，也不是论文实验值。

它说明了一个直观差别：相同方向的力，施加位置不同，扭转物体的效果就可能不同；只追踪接触位置，遗漏了法向、摩擦可行方向与力臂。

### 5.3 代码比较的是接触能力表示，不是视频里的真实六轴力传感器读数

在 FlashCHORD 的 `objectives/wrench.py` 中，每个激活接触由物体系下的位置、法向与摩擦锥射线表示。射线生成六维 wrench 特征，力矩部分使用物体半径做尺度处理；参考与仿真共享采样方向，比较这些方向上的 support 特征。

可以把特征计算读成四步：接触位置 $p_i$ 与可行力射线 $f_{ij}$ 生成 $w_{ij}$；把全部接触的射线列放入六行矩阵 $W$；给定共同方向 $b_k$；再计算该方向上最大的投影。FlashCHORD 当前实现把负值截为零：

$$
w_{ij}=\begin{bmatrix}f_{ij}\\(p_i\times f_{ij})/r_c\end{bmatrix},
\qquad
s_k=\max\left(0,\max_{i,j}b_k^\top w_{ij}\right).
$$

$r_c$ 是当前物体刚体的包围半径。接触点或射线数量变化会改变 $W$ 的列数，但选定的方向数量固定，因此人手与机器人最终能得到同长度的 $s$。这里的最大值不是把所有接触力相加，不能据此宣称恢复了完整的、多接触同时作用的净 wrench 可行集合；CHORD 论文 §3.1 也明确区分了这种方向能力表示与精确可行多面体。

<figure class="article-figure" id="fig-contact-wrench-support">
  {{< post-image src="assets/contact-wrench-support-v2.webp" alt="三个参考接触和两个机器人接触分别生成六行 wrench 特征矩阵，图中每接触仅画两条射线；在共享方向上得到同长度的 support 向量，再逐方向比较接触能力" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text">接触数不同，特征矩阵的列数可以不同；共享方向把它们转换成可逐项比较的向量。矩阵为每个接触只画两条射线，条形只画六个方向，实际数量由配置决定。条形没有实验数值或测量单位；图中还省略逐帧、逐手和逐刚体索引。接触点表示几何线索，不能当作传感器测得的真实力。</span></figcaption>
</figure>

这种表示回答的是“接触能沿哪些方向对物体施加作用”。归一化射线采用奖励特征的尺度；真实接触力、物体质量与摩擦参数则需要另行测量或设定。物理量与归一化特征的单位应分别记录。[wrench 特征实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/flash_chord/src/flash_chord/objectives/wrench.py)

训练仍需运动模仿、物体跟踪以及任务相关配置共同约束。接触引导提供更有用的优化信号，不能保证每个重建片段都可解，也不能修正所有上游几何错误。

### 5.4 接触参考有噪声时，不能一味追随它

CHORD 论文 §3.2 对高噪声重建给出另一种选择：降低对人手接触 support 精确匹配的依赖，改为鼓励机器人在各采样方向上形成正 support，也就是离散方向上的 **force-closure（力闭合）引导**。两者用途不同：参考匹配更忠实于示教的接触效果，后者偏向形成能够抵抗多方向扰动的抓取。有限方向检查仍受采样、尺度和接触模型影响，不能单独替代实际任务完成率。

**纸巾盒 E2E 的 Vega 配方实际采用后者。** 在 `VegaManipRewardsCfg` 中，force-closure 权重为 5.0，直接接触 wrench 匹配、漏接触和非预期接触项的权重均为 0。当前实现逐手计算并对激活的手取平均，也不是把双手接触合并后做完整的力闭合检验。[Vega 奖励配置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_manip_env_cfg.py)、[奖励实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/mdp/rewards/contact_rewards.py)

奖励的激活时机由接触标签控制。重定向器根据机器人 FK 探针与物体网格的近邻关系生成标签，默认距离门槛 0.018 m、至少连续两帧。连续性检查在目标参考帧率上执行，默认是 50 Hz 参考中的两帧，与 20 Hz 控制更新数分开计。还可用物体运动窗口抑制静止阶段的近邻误报，默认开启。未检测到明显物体运动时，窗口函数返回空值，保留未加窗口限制的标签。因此这些标签应结合几何与运动片段检查，不能当作真实触觉测量。[接触标签实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/scripts/retarget/contact_labels.py)、[重采样后的标签生成](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/scripts/retarget/ego_recon_to_dexmate_sharpa.py)

因此，复现时应记录实际运行的是哪条接触目标、摩擦系数、物体半径、采样方向与随机种子，而不只记录“启用了 CHORD”。若重建的手物接触本身明显错位，应先审查几何和参考，不宜靠增大奖励权重掩盖问题。[CHORD 论文](https://arxiv.org/abs/2607.00033v2)

### 5.5 专家训练看到了什么 {#expert-observations}

纸巾盒的 PPO 专家是**基于状态的参考跟踪策略**。`VegaManipPolicyCfg` 把关节位置与速度、参考差量、手腕和物体位姿、轨迹进度、上一动作等拼成观测；它没有把三路 RGB 图像输入该专家。配置中的 actor 与 critic 都是隐藏层为 512、256、128 的 MLP；这里没有额外配置一个“只有 critic 才看得到物体”的观测组。[观测与策略配置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_manip_env_cfg.py)、[PPO 网络配置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/vega_sharpa/agents/rsl_rl_ppo_cfg.py)

| 策略 | 运行时的主要输入 | 输出与学习方式 |
| --- | --- | --- |
| Vega PPO 专家 | 仿真本体与物体状态、运动参考和轨迹进度 | 学习参考附近的关节残差，由控制器形成绝对目标 |
| 后续 GR00T 策略 | 三路相机图像、58 维关节状态与语言指令 | 从专家示范学习绝对关节目标 chunk，再做视觉闭环 |
{.table-readable}

前者已知物体状态和示教接下来的参考，后者需要从视觉中判断当前交互阶段。GR00T 微调是在用专家轨迹监督另一个模型，不是把 PPO 网络权重转换成 GR00T 权重。闭环评估器仍可读取物体真值来计分，这与策略能看到哪些输入是两个接口。

参考窗口也有自己的时间约定：配置取三个参考时刻，底层偏移从 0 开始，所以包含当前参考；默认 `dt_future_frames=0.1`、20 Hz 播放时，对应偏移 0、2、4 个播放帧。接近轨迹或选定窗口的末端时，索引会截到该末帧，因此也可能重复同一参考。它们不是 GR00T 的 16 步预测 chunk，也不能把参数名中的 `future` 读成三个都严格晚于当前时刻。[参考索引与播放时钟](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/mdp/commands/tracking_command.py)

### 5.6 VOC 的全局课程与 reset 辅助是两个时钟 {#voc-curriculum}

**Virtual Object Controller（VOC）** 根据参考与当前物体的位姿误差，直接向物体施加辅助力和力矩，帮助训练早期建立接触。刚性物体实现还包含重力补偿、阻尼与力/力矩限幅；它的 `action_dim` 为 0，这份辅助不是 PPO 额外预测的机器人动作。[VOC 实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/actions/virtual_rigid_object_control.py)

在当前 Vega 配方中，全局目标从 1.0 开始，环境步阈值每增加 30,000 步降低 0.1，300,000 步对应零档阈值。每次 PPO 更新采样 `num_steps_per_env=24` 步，因而分别折合 1,250 和 12,500 次更新的采样量。图的横轴使用环境步计数除以 24，避免与 checkpoint 的零起始迭代编号混淆；并行环境数量不再乘进阈值。[全局课程实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/mdp/curriculum/curriculum.py)

这条阶梯表示配置值。配方容器采用 Isaac Lab 2.3.2，其默认环境在 reset 处理中调用课程检查；环境步刚越过阈值时，实际全局目标可能要等下一次 reset 才切档。[容器版本](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/workflow/Dockerfile)、[环境调用顺序](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)

**全局目标为零，不等于每次 reset 后立即没有辅助。** 训练配置仍有 `reset_freeze_steps=50`、`voc_decay_steps=10`、`voc_reset_scale=1.0`。令 $k$ 为本次命令重采样后的更新计数：$k\le40$ 时尺度保持 1，之后 10 次更新下降到全局目标；参考尚未结束时，$k=51$ 才首次前进。50 次更新按 20 Hz 折算为 2.5 秒的名义预算。冻结的是参考时间索引，物理环境与策略仍会执行。

下图记录命令更新后的尺度。默认环境先推进本步物理，再更新 command，这个新尺度供后续物理步使用；自动 reset 后，同一步末尾还会进行一次命令更新。脚本的 $k=0$ 表示刚重采样的状态，因此不把 $k$ 或 $k/20$ 直接当作实际施力时刻或自动 reset 后的精确经过时间。

<figure class="article-figure" id="fig-voc-clocks">
  {{< post-image src="assets/voc-two-clocks.webp" alt="上图是 reset 时检查的全局 VOC 配置阶梯，环境步 300000 对应零档阈值；下图取实际全局目标已为零且参考尚未结束的情况，命令计数到 40 时辅助尺度仍为 1，到 50 时为零，51 时参考才前进" >}}
  <figcaption><span class="article-figure__number">图 4</span><span class="article-figure__text">两条横轴属于不同时间尺度。上图是配置阶梯，下图是全局目标已为零、参考尚未结束时，各次命令更新后的尺度；它们不表示实测力、奖励或成功率。实际力还依赖误差、阻尼、重力补偿与限幅。蓝线连接离散更新样本，橙色虚线对应关闭辅助和参考冻结的采集／评估配置。</span></figcaption>
</figure>

因此，无辅助采集与评估要同时把全局初值、reset 辅助尺度、衰减步数和冻结步数置零，并关闭课程。当前源导出器与 GR00T 闭环入口已经这样设置；只改全局目标会漏掉 reset 阶段。[源采集设置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/scripts/rsl_rl/export_parallel_rollouts.py)、[闭环设置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/scripts/rsl_rl/gr00t_infer.py)

下载配套 [voc_schedule_lab.py](voc_schedule_lab.py) 后，可在文件所在目录离线打印全局目标、reset 尺度与参考推进次数；推进次数按尚未触及轨迹末端计算，真实环境还会截到该轨迹的最后参考帧。图 4 可用 `python3 -B scripts/plot_v2d_voc.py` 重建：

```bash
python3 -B voc_schedule_lab.py --iteration 12500 --steps 0 40 45 50 51
python3 -B voc_schedule_lab.py --self-test
```

另外，该 Vega 配方关闭机器人链接重力，以匹配重力补偿机械臂，物体仍受场景重力影响。比较新机器人或真机结果时，应保存这一动力学设定和 VOC 状态，而不只记录“在物理仿真中运行”。

## 6. 从专家策略到 GR00T：训练标签在这里产生 {#groot-data}

### 6.1 先采集无相机 rollout，再回放录制

GR00T post-training 路径先运行专家，导出机器人与物体轨迹，再选出固定数量的合格 episode，用标定相机回放并写成语义 HDF5，随后转换为 LeRobot 数据。

把物理交互采集与相机渲染拆开，可以分别控制两者的并行度。数据回放必须保留同一条轨迹的机器人、物体、相机时序与身份；仅看视频清晰，无法证明图像与动作对齐。

本版本的回放录制会把已导出的关节与物体位姿逐帧写入场景，再执行 `forward` 与渲染来更新相机，**正式录制阶段不再推进物理动力学**。它为同一条专家交互轨迹补录观测，操作成败仍由源 rollout 和后续闭环评估判断。预热阶段与正式取帧使用不同的执行方式。[回放取帧实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/scripts/rsl_rl/replay_record.py)

这里有两份独立输入：**embodiment contract** 定义关节字段与顺序、相机、控制频率和推理环境等；**task profile** 定义对象文字提示、语言指令、目标物体与任务评估器。物体从纸巾盒改成其他物品，既可能影响重建，也可能影响任务成功条件。[GR00T 数据路径](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/README.md)

这条 Vega joint-rollout 导出路径目前只接受刚性场景物体，遇到不支持的对象类型会报错。CHORD 研究覆盖关节物体，并不表示该 GR00T 录制配方已经包含关节物体的全部状态、回放和评估接口；扩展任务时需要逐项检查具体入口。[源导出器的对象检查](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/scripts/rsl_rl/export_parallel_rollouts.py)

### 6.2 58 维状态、58 维动作不代表同一个东西

在 Vega 示例中，每帧状态包含 58 个关节位置，动作包含 58 个**绝对关节位置目标**。这 58 个都是转动关节，位置与目标以 rad 表示；前方相机与腕部相机流按 20 FPS 录制。动作是希望控制器达到的目标，状态是实际记录的位置，两者可能因跟踪误差而不同。

专家内部的原始策略输出可以是相对参考运动的残差，导出的训练标签则取控制器**实际收到的绝对目标**。采集器先快照动作前的关节与物体状态，再执行一步，最后读取该步保存的 `last_applied_actions`。因此配对关系是“当前观测 → 本步发出的目标”，不是“下一帧位置 → 已清零的 reset 动作”；回合结束时也需要保留终止前实际施加的目标。[源轨迹导出](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/scripts/rsl_rl/export_parallel_rollouts.py)

这个转换还包含状态与顺序。以当前 Vega PPO 配方的一只关节为例，$u_t$ 是送入残差动作项的策略输出，$r_t$ 是滤波后的角度残差，$q_t^{\mathrm{ref}}$ 是本步参考角度，导出的绝对目标为 $a_t$：

$$
\begin{aligned}
r_t&=\operatorname{clip}\!\left(0.3r_{t-1}+0.7(0.15u_t),\,-1,\,1\right),\\
a_t&=\operatorname{clip}\!\left(q_t^{\mathrm{ref}}+r_t,\,q_{\min},\,q_{\max}\right).
\end{aligned}
$$

先缩放原始输出，再做 EMA、裁剪残差、加当前参考，最后裁剪到关节限位；残差与目标以 rad 计。0.3 是旧残差的权重，不能当成当前输出的权重。假设 reset 后残差为 0，两步参考为 0.50、0.55 rad，原始输出为 2、0，且未触及限位：

| 控制步 | 残差 $r_t$ / rad | 目标 $a_t$ / rad |
| --- | ---: | ---: |
| 0 | 0.210 | 0.710 |
| 1 | 0.063 | 0.613 |

第二步原始输出已为零，残差仍由历史衰减而来，所以目标并未立即回到参考 0.55 rad。这是按配置构造的数值例子，不是实测关节响应。[残差动作实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/mdp/actions/reference_residual_action.py)、[Vega 配方](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_manip_env_cfg.py)

下载配套 [data_contract_lab.py](data_contract_lab.py) 后，在文件所在目录运行，即可复算两步残差与目标：

```bash
python3 -B data_contract_lab.py --residual-example
```

GR00T 推理入口改用 `scale=1.0`、`use_default_offset=False` 的绝对关节位置动作项，直接接收预测目标；不能再把预测值当成 PPO 原始残差，重新乘 0.15 并加一次参考。`last_applied_actions` 保存的是送入执行器的目标，尚未经过臂的延迟缓冲与 PD 计算，因此它既不是下一帧实测位置，也不是关节力矩。[GR00T 动作配置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_gr00t_env_cfg.py)

**这两个 58 维向量的字段排列也不同。** 转换器按 `state_fields` 和 `action_fields` 各自的顺序拼接，而不是共享一个扁平索引：

| 字段 | 状态向量中的切片 | 动作向量中的切片 | 宽度 |
| --- | --- | --- | ---: |
| 右臂 `right_arm` | `[0:7]` | `[0:7]` | 7 |
| 左臂 `left_arm` | `[7:14]` | `[29:36]` | 7 |
| 右手指 `right_finger` | `[14:36]` | `[7:29]` | 22 |
| 左手指 `left_finger` | `[36:58]` | `[36:58]` | 22 |

例如状态第 7 项是左臂首个关节，而动作第 7 项是右手指首个关节。只有先按字段名和关节名对应，再比较同一物理量，误差才有意义。状态里的 `arm_joint_pos` 与 `finger_joint_pos` 来自不同观测项；动作则从 `action_target` 取切片。[具身约定](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/contracts.py)、[字段拼接实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/convert_to_gr00t.py)

<figure class="article-figure" id="fig-state-action-layout">
  {{< post-image src="assets/state-action-layout.webp" alt="状态按右臂、左臂、右手指、左手指拼接，动作按右臂、右手指、左臂、左手指拼接；两者都是 58 维，但索引 7 到 35 的字段边界不同" >}}
  <figcaption><span class="article-figure__number">图 5</span><span class="article-figure__text">Vega 示例的两套字段布局。数字是从零开始、右端不包含的切片边界，块宽与字段维数成比例。重排只能让已有位置按相同关节对齐，无法把实测状态变成专家动作目标。</span></figcaption>
</figure>

因此，不能把下一帧状态直接当成本帧动作，也不能把绝对目标解释为角度增量或力矩。三条相机流的键分别为 `front`、`right_wrist_view`、`left_wrist_view`；转换后的第 $i$ 帧时间戳为 $i/20$ 秒。审计需要同时核对字段顺序、关节单位、相机身份和逐帧对应关系，不能只看数组维数或视频帧数。

当前模态配置的 `video` 和 `state` 都取 `delta_indices=[0]`，即查询时刻的三路图像与一帧机器人状态；`action` 取偏移 0～15。三路相机代表三个视角，专家观测中的三个参考时刻则代表时间偏移，两者承担不同职责。[Vega 模态配置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/vega_sharpa_joint_config.py)

GR00T 的动作 chunk 长度、每次实际执行多少步以及控制频率也是三个不同参数：当前约定的 chunk 为 16 步，对应 20 Hz 下的 0.8 秒控制覆盖时长；默认先执行其中 4 步，覆盖 0.2 秒控制时间，再重新观察并查询。这里用步数除以频率描述控制覆盖时长，16 个采样点的首末时间戳之差则是 $15/20=0.75$ 秒。实际墙钟耗时还要计入推理、通信和渲染延迟。

当前配方还有两个不能混用的“4 步”：物理步长是 0.01 秒，`decimation=5` 形成 0.05 秒的控制间隔；臂的 DelayedPD 配置延迟 4 个物理子步，对应 40 ms。GR00T 每次执行 4 个预测动作，则是 4 个控制间隔、0.2 秒的名义控制覆盖时长。前者是执行器模型里的配置延迟，后者是策略重新观察的节奏，都不能直接当成实测端到端延迟，时长对照见[图 6](#fig-control-clocks)。[物理时钟与动作配方](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_manip_env_cfg.py)、[DelayedPD 配置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/vega_sharpa.py)、[Isaac Lab 延迟实现](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/actuators/actuator_pd.py)

以一个具备完整动作窗口的训练时刻为例：第 $t$ 帧的三路图像、58 维实测状态和任务文字，对应标签 $[a_t,\ldots,a_{t+15}]\in\mathbb R^{16\times58}$；每个 $a$ 都按动作字段顺序排列。运行时，模型预测的是 $\hat a_{t+j\mid t}$，下标表示“根据第 $t$ 步观测，为第 $t+j$ 步预测的目标”。例如连续运行且未发生 reset 时：

| 控制步 | 使用哪次观测 | 发出哪些目标 |
| --- | --- | --- |
| 100～103 | 第 100 步查询一次 | 依次执行该次 chunk 的第 0～3 项 |
| 104～107 | 用第 104 步的新观测重新查询 | 执行新 chunk 的第 0～3 项 |
{.table-readable}

因此，第 104 步使用的是 $\hat a_{104\mid104}$，上一 chunk 尚未执行的 $\hat a_{104\mid100}$ 等后续预测被替换。这是用新观测纠正后续动作的时机；每次查询后的四步使用同一份预测，底层关节控制仍在运行。任一并行环境 reset 时，适配器会提前重新查询整批，避免继续使用旧回合的缓存。[动作缓存与重新查询实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/closed_loop/action_adapter.py)

<figure class="article-figure" id="fig-control-clocks">
  {{< post-image src="assets/control-clock-durations.webp" alt="同一秒单位坐标轴上的四种配置时长：臂延迟四个物理子步为0.04秒，一个控制间隔为0.05秒，执行四步动作为0.20秒，预测十六步动作覆盖0.80秒" >}}
  <figcaption><span class="article-figure__number">图 6</span><span class="article-figure__text">条形共用线性时间轴，比较当前配方的配置时长；执行长度取本文的 4 步设置。它们不表示实测推理延迟、网络延迟或关节响应。16 步动作按控制间隔覆盖 0.80 秒，首末动作的时间戳之差则为 0.75 秒；图中没有加入推理等待或参考运动的时间重采样。</span></figcaption>
</figure>

本版本的启动器使用外部 `Isaac-GR00T` Python 3.10 环境执行统计、微调和开环评估，默认基座为 `nvidia/GR00T-N1.7-3B`。这与 Isaac Lab 专家训练环境、视频入库的 vLLM 环境分开管理。[端到端指南](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/docs/e2e_workflow.md)

### 6.3 三种“通过”必须分别报告

| 检查 | 本版本具体含义 | 单靠它还不能证明什么 |
| --- | --- | --- |
| Source eligibility | 专家 episode 到达配置的 timeout，符合采集选择条件 | 物体任务一定完成 |
| 数据与模型接口通过 | HDF5、Parquet、图像与动作一致，统计有限，模型可加载，开环误差有限 | 策略执行后能恢复自身误差 |
| 闭环任务通过 | 策略实际控制环境，按 task profile 统计任务完成 | 换物体、环境或真机后仍有同样表现 |
{.table-readable}

尤其注意 `--target-successes` 的命名：采集阶段指选出对应数量的 **timeout-eligible 源轨迹**。它不能直接作为机器人操作成功率的分子。开环 MSE/MAE 有限也主要证明数据和模型链路能工作；小的预测误差仍可能在接触处造成大幅执行偏差。

当前纸巾盒 profile 的语言指令是 `lift the box and hold it`，但 `lift_hold` 评估还包含放回阶段的高度稳定条件：有抬升与保持门槛、末段高度与连续变化约束。只执行文字里的“举起并保持”，可能通过抬升与保持检查，却未通过完整任务检查。**它基于高度检查，没有检查 XY 放置位置、真实接触或手是否松开。** 定义自己的任务时，应同时核对语言指令、源示范与评估器是否描述了相同的完成标准。[任务 profile](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json)

### 6.4 同样抬起 11 cm，为什么最终结论可能不同

当前纸巾盒 profile 的阈值如下。高度均相对本 episode 的初始物体高度；在线跟踪器在 reset 观测中建立基准，再对每次动作前的观测调用 `update()`。终止动作之后的状态不进入旧回合指标，自动 reset 返回的观测只用于新回合，避免把“物体被重置回桌面”误当作平稳放回。

下面的离线算例按官方 `evaluate_lift_hold()` 接口，把首样本用于基准，之后的样本才计入更新数，所以长度为 $N$ 的输入对应 $N-1$ 次更新。在线日志的 `metric_sample_counts` 则是实际动作前更新次数；比较两者时，先核对是否单独保存了基准样本。[闭环采样时序](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/scripts/rsl_rl/gr00t_infer.py)

| 条件 | 当前阈值 | 判定范围 |
| --- | --- | --- |
| 最大抬升 | 至少 0.10 m | 整段最高值 |
| 连续保持 | 至少 20 个样本不低于 0.05 m | 连续段，不能累计零散样本 |
| 平稳下降 | 开始达到保持高度后，单步下降不超过 0.01 m | 整段，不只检查最后几帧 |
| 末段放回 | 最后 20 个样本均不高于 0.03 m | 必须已有完整窗口 |
| 末段稳定 | 相邻高度差不超过 0.005 m，窗口极差不超过 0.01 m | 最后 20 个样本 |
| 最终下限 | 最后高度不低于 −0.01 m | 防止明显落到初始支撑面下方 |
{.table-readable}

抬升与保持是两个独立条件：**最高达到 0.10 m，以及连续 20 次更新不低于 0.05 m**；不要求连续 20 次都在 0.10 m 以上。保持段也不必包含最高点，完整通过仍要满足表中其余条件。[评估器实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/closed_loop/evaluation.py)

20 Hz 下每个控制间隔是 0.05 秒。20 个连续样本对应评估器要求的 20 次更新；首末样本时间戳之差为 0.95 秒，不能把这个计数与连续时间保持证明混为一谈。

<figure class="article-figure" id="fig-height-criteria">
  {{< post-image src="assets/height-criteria-examples.webp" alt="三条构造的 20 Hz 高度曲线均抬升到 0.11 米：平稳放回通过；持续举在空中没有通过末段放回；突然下降后虽末段稳定仍因单步下降 0.11 米失败" >}}
  <figcaption><span class="article-figure__number">图 7</span><span class="article-figure__text">三条构造轨迹使用同一组已发布阈值，不是机器人实测结果。上图满足完整条件；中图虽达到抬升与保持要求，却没有满足末段放回高度；下图末段已经静止，仍因早先突然下降而失败。蓝虚线和紫点线分别是抬升、保持阈值，绿色带表示最终高度允许范围，右侧淡紫色区标出最后 20 个样本；这些视觉标记没有替代其他条件。</span></figcaption>
</figure>

打开 [高度判定交互实验](/labs/video-to-data/)，可以载入这三条曲线，再调整下降步数、末段高度、波动与样本数。上升和高处停留保持不变，因此可单独观察“下降太快”或“末段仍在抖动”为何导致失败；最后 20 次更新也可能跨进过渡段，不能只看自己命名的末段。

配套 [data_contract_lab.py](data_contract_lab.py) 只需 Python 标准库。运行后可查看各曲线的峰值、最长连续保持样本数、最大单步下降及最终结论，也可用 `assemble_state()` 和 `positions_in_action_order()` 检查字段排列：

```bash
python3 -B data_contract_lab.py
python3 -B data_contract_lab.py --self-test
```

若要逐项复算，可以先导出构造序列，再读取同一文件；自有输入使用米为单位的高度数组，并先核对 20 Hz 采样及首样本基准约定：

```bash
python3 -B data_contract_lab.py --show-trace stable_place > height.json
python3 -B data_contract_lab.py --trace height.json
```

还可在配套脚本所在目录复算两种阈值的差别。下面只有一个样本达到 11 cm，随后以每步 0.5 cm 降到 6 cm、保持 20 次，再缓慢放回。先用厘米构造样本，调用评估器时统一换成米：

```python
from data_contract_lab import height_result

trace_cm = [0, 11]
trace_cm += [11 - 0.5 * i for i in range(1, 11)]
trace_cm += [6] * 20
trace_cm += [6 - 0.5 * i for i in range(1, 13)] + [0] * 25
result = height_result([z / 100 for z in trace_cm])
print(result["success"], result["max_hold_samples"])  # True 33
```

33 次连续保持包括峰值那一次、下降到 6 cm 的 10 次、停留的 20 次，以及随后 5.5、5 cm 的两次；并非只计算代码中显式写出的停留。这个构造序列用于核对评估逻辑，初次抬升没有模拟动力学，不能作为可执行机器人轨迹。

其中 `positions_in_action_order()` 只重排同一帧测得的位置，便于核对字段，**不会生成训练用的专家目标**。图 5、图 7 可在本站仓库运行 `python3 -B scripts/plot_v2d_contracts.py` 重建；图 6 对应 `python3 -B scripts/plot_v2d_control_clocks.py`。曲线结论已与固定版本的官方评估器交叉核对。

### 6.5 转换后的文件与终止字段怎样读 {#dataset-files}

当前转换器写出标记为 `codebase_version: v2.1` 的 LeRobot 布局，供对应 GR00T 数据路径读取。三条相机视频分别编码，不嵌进每一行 Parquet：

```text
gr00t_dataset/
├── meta/
│   ├── info.json           # 总帧数、相机与数据布局
│   ├── modality.json       # state/action 各字段的切片
│   ├── tasks.jsonl         # task_index 与语言指令
│   └── episodes.jsonl      # 每条轨迹的长度
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/observation.images.<camera>/episode_000000.mp4
```

Parquet 的 `episode_index` 标识轨迹，`index` 是跨轨迹递增的全局行号；`timestamp` 在每个 episode 内从零开始，每帧增加 $1/20$ 秒。转换器没有额外写一个 `frame_index` 列；想定位视频帧时，应结合该 episode 内的行位置、时间戳与实际编码帧数。

还有两个名称容易被误读：`next.done` 只在序列最后一行置为 `true`，`next.reward` 则统一写为 0。它们是这条模仿数据转换路径中的边界与占位字段，**不表示最后一帧操作成功，也不是专家 PPO 奖励**。任务完成仍由前面的评估器和独立闭环结果回答。

默认转换器还把全部 episode 写入 `info.json` 的 `splits.train`，没有自动划出验证集。目录结构完整、字段审计通过，说明数据满足接口约定；数据是否独立留出，需要另外检查 episode 与源视频的来源关系。[转换器与 split 元数据](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/convert_to_gr00t.py)

## 7. 运行前的环境与资产准备 {#requirements}

### 7.1 不同模块有不同的资源需求

| 路径 | 官方本版本给出的环境或验证范围 | 如何理解 |
| --- | --- | --- |
| 视频入库默认 VLM | Linux、Python 3.10/3.11、ffmpeg；8B 级 VLM 约需 24 GB 显存 | 这是视频理解路径的需求 |
| 完整重建到 GR00T 示例 | 单张 48 GB RTX A6000；也验证过 L40、L40S | 是指定示例与执行配置的可运行范围 |
| Isaac Lab 专家训练 | Docker、NVIDIA Container Toolkit；项目推荐 580.126.09 驱动/CUDA 13.0 | 模型依赖与仿真环境按各自镜像管理 |
| FlashCHORD 完整训练 | 单独的 Newton/Warp/JAX 容器；48 GiB GPU，空闲显存至少 46,000 MiB | 查看或回放的需求低于完整训练 |
{.table-readable}

这些条件不意味着所有模型都可在一张卡上同时常驻。入库 VLM、重建模型、渲染器和策略训练会竞争显存；阶段切换时，应按各自的运行方式释放服务。物理环境数量与带相机渲染的环境数量也不能照搬。

另一个明确限制来自 **v0.2 多视角 HOI 的 `v2d_cusfm` 路径**：其 TensorRT/cuVSLAM 版本不支持 compute capability 12.0 的 Blackwell GPU，文档列出的验证设备包括 A6000 和 L40S。这个限制有具体模块范围，不能扩大成“V2D 的所有组件都不支持 Blackwell”。[HOI 兼容性说明](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/modules/v2d_hoi_object_reconstruction/README.md)

### 7.2 代码、权重和数据是三类准备工作

Git checkout 完成后，还需要 Git LFS 资产、模型权重、MANO 文件和所选数据集。MANO 需要自行取得并按接口放置；SAM3D 等模型可能需要 Hugging Face 访问权限；FoundationPose 下载步骤涉及单独的模型条款。

仓库根许可证区分文档与源码，外部模型和数据仍有各自条款，使用时应逐项查看对应来源。安装脚本成功不能证明所有资产已齐全；几百字节的 `.mp4` 可能仍然是 LFS 指针。[仓库许可证](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/LICENSE)、[数据与 MANO 设置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/docs/SETUP.md)

## 8. 纸巾盒 E2E：逐阶段运行与验收 {#e2e}

以下命令按官方本版本整理，**用于说明复现路径，本文没有在本机完成 GPU 重建、RL 训练或 GR00T 实验**。运行前完成第 7 节，以及官方的 Ego Reconstruction、Robotic Grounding 与外部 Isaac-GR00T 设置。路径中的占位值需要替换；训练完成后要选择并验证实际 checkpoint。

### 8.1 固定代码版本并检查视频

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-isaac/video_to_data.git
cd video_to_data
git checkout 193382a184308f6800331f3db829f76dc4294cfe
git lfs install
git lfs pull
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration \
  -of json reconstruction/assets/tissue_box.mp4
```

首次复现可以从完整 LFS 资产开始；若需要减少下载量，应按所选任务确认依赖，再使用定向拉取。不要只拉视频却遗漏机器人模型或所选专家策略。

自有视频还要核对时间轴：当前 Vega 重定向脚本默认 `source_fps=30`、`target_fps=50`，根启动器没有从上述 `ffprobe` 输出自动传入源帧率。准备非 30 fps 视频时，应先明确重采样方式，或在底层入口显式设置匹配的源帧率，再检查报告。单独改文件里的 `fps` 数值会改变帧对应的时间，不会完成真实重采样。

### 8.2 初始化一次运行

在仓库根目录使用已准备好的主机编排环境，指定独立运行目录：

```bash
./run_e2e.sh init \
  --run-root "$PWD/.e2e/tissue_box" \
  --sequence-id tissue_box \
  --embodiment-contract vega_sharpa_joint \
  --task-profile "$PWD/robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json"

E2E_CONFIG="$PWD/.e2e/tissue_box/e2e_config.json"
./run_e2e.sh --config "$E2E_CONFIG" status --json
```

`init` 保存任务与机器人约定，并把仓库内的 `.e2e/current` 指向这次运行。再次初始化或切换其他运行，会改变不带 `--config` 的命令所操作的对象。后文显式传入 `E2E_CONFIG`，并先用 `status` 核对 `config`、`run_root` 与输出位置；`--config` 只覆盖本次命令，不修改活动指针。新开终端时需要重新设置这个 shell 变量。[配置选择与活动指针](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/tools/e2e/config.py)

改变物体或指令后，要同步检查 task profile，尤其是目标对象选择与评估标准。

### 8.3 设置环境、检查依赖、重建

下面的模型条款参数只应在阅读并接受相应条款后使用。

启动重建前，可以先在同一条 `reconstruct` 命令中加入 `--dry-run`，检查展开后的输入视频、对象提示、输出目录和阶段选项。它会解析配置并打印待执行命令，不启动重建子进程，也不保存这次新增的阶段参数；正式运行时仍要传入相同选项。预览成功只说明命令能够构造，模型、镜像与 GPU 是否可用仍由 `doctor` 和实际阶段验证。[预览与参数持久化](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/tools/e2e/cli.py)、[阶段执行器](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/tools/e2e/runner.py)

```bash
./run_e2e.sh --config "$E2E_CONFIG" setup \
  --mano-dir /absolute/path/to/mano \
  --isaac-groot-dir /absolute/path/to/Isaac-GR00T \
  --accept-nvidia-model-eula
./run_e2e.sh --config "$E2E_CONFIG" doctor
./run_e2e.sh --config "$E2E_CONFIG" reconstruct \
  --video "$PWD/reconstruction/assets/tissue_box.mp4" \
  --hand-tracking hamer \
  --run-gsplat-refinement
./run_e2e.sh --config "$E2E_CONFIG" inspect
```

根启动器会启用参考帧 0、去畸变、DROID-SLAM、重力对齐与 Three.js 导出。这里使用已构建镜像，省略官方开发示例中的 `--dev`；该开关用于把本地模块源码挂入容器。`doctor` 应先通过，`inspect` 后应实际查看三维场景。

若已有合格重建 bundle，可以在设置阶段使用 `--skip-reconstruction`，然后运行 `import-reconstruction --bundle /absolute/path/to/result_slam_gravity_aligned`。这只跳过重型重建，后面的加载、重定向和训练环境仍需准备。

### 8.4 重定向并训练源专家

```bash
./run_e2e.sh --config "$E2E_CONFIG" retarget --embodiment vega
./run_e2e.sh --config "$E2E_CONFIG" simulate --embodiment vega
./run_e2e.sh --config "$E2E_CONFIG" train-expert \
  --embodiment vega \
  --max-iterations 20000 \
  --num-envs 128
```

先检查 `vega_retarget_qa/` 的报告和视频，再用 `simulate` 检查 Isaac Lab 任务的加载与运行，最后启动长训练。20,000 iterations 与 128 个环境是此版本指南给出的 Vega 配方参数；一次训练迭代包含多环境采样与优化，**不能换算成 20,000 条示范或 20,000 个仿真时间步**。[启动器选项与阶段实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/tools/e2e/stages.py)

根启动器的 `simulate` 当前调用 headless `dummy_agent.py`：单环境、仅运行 2 步，用于确认任务注册并能推进。完整运动与操作效果应结合 QA 视频和专家 rollout 检查。

训练目录在运行根目录的 `train_expert/vega/logs/rsl_rl/vega_sharpa_whole_body/` 下。启动器不会替你挑选“最佳 checkpoint”；应根据训练指标和实际验证选择权重，并保存其哈希。

纸巾盒还有随仓库提供的 Vega ONNX 专家，可作为省去源专家训练的快捷路径。它只替代这一项产物，仍需采集、审计、微调和闭环评估；换物体或重定向运动后，也不能假定该专家继续适用。

### 8.5 先测 episode 长度，再采集

当前 Vega 专家的参考运动速度为 `motion_speed=0.5`，采集控制频率为 20 Hz。采集步数上限需要覆盖专家 episode timeout，不能不看源运动就复制一个常数。

这还涉及一个内部重采样率：加载器按 $1/(\Delta t_{\mathrm{env}}s)$ 重采样参考，其中 $s$ 是 `motion_speed`。本例 $\Delta t_{\mathrm{env}}=0.05$ 秒、$s=0.5$，得到沿**原参考时间轴**的 40 Hz 采样，再以每秒 20 帧的控制节奏消费，形成半速播放。因此，重定向文件的 50 Hz、内部重采样的 40 Hz、控制与数据的 20 Hz 可以同时成立。`motion.dt` 仍须等于环境步长 0.05 秒，不能因为文件是 50 Hz 就改成 0.02 秒。[播放时钟约束](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/motion_schema/timing.py)

估计控制步数的方法是：源帧数除以源帧率，得到原始时长；再除以运动速度，得到播放时长；最后乘控制频率并向上取整。随后增加约 10%～20% 余量。实际统一 horizon 由 pilot 测量，**这个安全上限不等于最终记录帧数**。

下面仅作预算计算：假设 300 源帧、30 fps，得到 400 个控制步，增加 15% 后上限为 460。后半段假设实际每条记录有 400 帧，用于解释训练样本窗口数。

```python
from math import ceil

source_frames, source_fps = 300, 30
motion_speed, control_fps = 0.5, 20
estimated_steps = ceil(source_frames / source_fps / motion_speed * control_fps)
collection_max_steps = ceil(estimated_steps * 1.15)
print(estimated_steps, collection_max_steps)  # 400 460

# 假设测得 F=400，每条轨迹以 H=16 构造完整动作窗口。
epochs, episodes, frames, horizon, batch = 5, 10, 400, 16, 32
optimizer_steps = ceil(epochs * episodes * (frames - horizon + 1) / batch)
print(optimizer_steps)  # 602；不是把所有原始视频帧直接当成优化步。
```

将以下两个变量改为已经验证的权重路径，以及按实际运动计算出的上限；这里的 460 只对应上面的假设。

```bash
RL_CHECKPOINT=/absolute/path/to/validated/model.pt
COLLECTION_MAX_STEPS=460
test -f "$RL_CHECKPOINT"
sha256sum "$RL_CHECKPOINT"

./run_e2e.sh --config "$E2E_CONFIG" collect \
  --rl-checkpoint "$RL_CHECKPOINT" \
  --target-successes 10 \
  --pilot-attempts 32 \
  --collection-max-steps "$COLLECTION_MAX_STEPS" \
  --render-envs 1
```

32 次 pilot 是单独的试运行；完整采集的尝试次数由测得的合格率与安全系数估计。目标为 10 条不意味着总共只运行 10 次。此版本从参考首帧 reset，关闭 reset freeze、VOC、课程和非 timeout 终止；臂、手指、物体 XY 与 yaw 的 reset 噪声默认均为零。可选噪声通过 reset 事件扰动仿真状态，不直接修改参考或专家动作。

多条文件也不自动等于多样示范。在固定参考、近似相同 reset 和确定性策略条件下，episode 可能非常相近。pilot manifest 的 `diversity` 统计在相同任务阶段跨 episode 计算状态、动作和物体位置的变化；应结合这些数值与视频判断覆盖范围，再选择采集分布，不能只增加 `--target-successes`。[轨迹多样性统计](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/source_rollout.py)

### 8.6 微调并独立做闭环评估

```bash
./run_e2e.sh --config "$E2E_CONFIG" finetune --epochs 5 --global-batch-size 32
```

启动器按 episode 数、实测帧数、16 步完整窗口和 batch size，把 `--epochs` 换算成传给外部训练入口的 `--max-steps`。比较训练预算时，应保存这个实际步数及数据规模；同样写“5 epochs”，数据长度不同，优化步数也不同。[训练预算与启动命令](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/tools/e2e/stages.py)

这一阶段依次转换数据、计算统计、审计帧与字段对应关系、训练并保存完整 checkpoint，再运行开环评估。**默认开环命令读取的仍是训练所用的 `dataset_host` 目录，没有另设留出数据目录。** 因而这里的有限 MSE/MAE 是训练数据上的链路检查，不能据此声称已经验证未见示范的预测误差。[开环命令构造](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/tools/e2e/stages.py)

若要进一步评估泛化，先按完整 episode 及其源视频分组留出，再从各组生成窗口。同一条轨迹的相邻 16 步窗口高度重叠，随机拆窗口会把几乎相同的动作放到训练与验证两侧；同一源参考的重复 rollout 也应保留来源分组。归一化统计只从训练组估计，并分别报告“未见 rollout”“未见源视频”或“未见任务”，因为三者检验的变化不同。这是扩展评估协议，当前启动器没有自动完成这一步。

闭环运行则用命令打印的真实 checkpoint 路径：

```bash
FINETUNED_CHECKPOINT=/absolute/path/to/checkpoint-N
./run_e2e.sh --config "$E2E_CONFIG" evaluate \
  --checkpoint "$FINETUNED_CHECKPOINT" \
  --episodes 1 \
  --num-envs 1 \
  --execution-length 4
```

单 episode 先确认相机预热后非黑帧、reset 与动作语义正确、回合达到预期长度，再用同一评估器增加 episode 数。例如保持单个渲染环境做 100 次评估，或测出渲染安全并行度后调整 `--num-envs`。本版本接口预测 16 步动作 chunk，`--execution-length 4` 表示先执行其中 4 步，再重新查询策略；它不是控制频率或完整回合长度。[命令行语义](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/tools/e2e/cli.py)

### 8.7 应保存哪些证据

| 文件或目录 | 主要用途 |
| --- | --- |
| `e2e_config.json`、`run_manifest.json`、`logs/` | 记录配置、代码与运行来源、各阶段状态和失败尝试 |
| `reconstruction/result_slam_gravity_aligned/`、`vega_retarget_qa/` | 追查几何、坐标、重定向和帧数问题 |
| `contracts/`、`source/pilot/manifest.json` | 固定机器人与任务约定，记录 pilot 分母和实测时长 |
| `recording/data.h5`、`semantic_audit.json` | 核对录制数据与选中轨迹的语义一致性 |
| `gr00t_dataset/`、`dataset_audit.json` | 核对 LeRobot、相机、统计和动作转换 |
| `finetune/checkpoint-*/`、`open_loop/` | 保存可服务的模型与开环检查证据 |
| 外部 Isaac-GR00T 的 commit 与实际训练配置 | 另行记录训练／冻结的模块和依赖版本，明确微调设置 |
| `closed_loop/evaluation_<N>.json` | 查看完成数、成功数、成功率、回合长度与终止原因 |
{.table-readable}

最后运行 `./run_e2e.sh --config "$E2E_CONFIG" status --json`。根启动器根据命令、声明输入、必需产物和源码哈希是否匹配决定恢复方式。重建内部还有自己的阶段缓存，一些步骤以中间文件存在为完成条件。`--force` 强制外层阶段执行，内部缓存仍需分别检查；更换视频、对象或网格时，优先使用新的运行目录，再逐阶段验收。[阶段恢复逻辑](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/tools/e2e/runner.py)

终端出现 `[DONE]` 只表示评估执行结束，结果仍可能是零成功。

## 9. 两种更小的入口：深度与 FlashCHORD {#smaller-entrypoints}

### 9.1 只验证重建基础设施，可以先跑 MoGe

如果尚未准备完整人体资产，可以先运行 MoGe 深度示例。主机端需要同时安装仓库内的共享 `v2d-docker` 包与 MoGe 容器包装包；模型推理依赖在容器内准备。从仓库根目录执行：

```bash
cd reconstruction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e modules/v2d_docker -e modules/v2d_moge/docker
python -m v2d.moge.docker.build
python -m v2d.moge.docker.run_download_weights --output_dir data/weights/moge
python -m v2d.moge.docker.run_video_to_depth \
  --video_path modules/v2d_moge/assets/test_video.mp4 \
  --depth_folder data/outputs/moge/depth \
  --intrinsics_folder data/outputs/moge/intrinsics \
  --weights_path data/weights/moge
```

这条路径检查 GPU 容器、模型下载、挂载与结果落盘，输出逐帧深度和内参。物体跟踪、人手恢复、重定向和 RL 属于后续独立阶段。[MoGe 主机端包的依赖声明](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/modules/v2d_moge/docker/pyproject.toml)

### 9.2 FlashCHORD 是单独的训练运行时

FlashCHORD 使用 Newton、Warp、JAX，并提供 PPO 与 FlashSAC 路径；它有自己的镜像、锁文件、Hydra 配置和 checkpoint 格式。使用前先查看机器人、场景和参考回放，再做物理回放、训练与策略评估。

其中两个回放按钮有不同用途：`Show kinematic reference` 直接展示记录姿态；`Run configured replay` 通过控制器与物理执行，便于检查实际跟踪与接触行为。

官方示例覆盖 floating Sharpa 与 G1/Dex3。其 MPPE 使用物体局部轴上的固定关键点计算位姿偏差，以 cm 报告；这类跟踪误差和 E2E 的任务成功率不是同一个指标。PPO、FlashSAC 与 Isaac Lab/RSL-RL 权重也不能仅凭文件名互换。[FlashCHORD 设置与工作流](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/flash_chord/README.md)

CHORD 论文的全身扩展还区分参考来源：只有手部或末端轨迹时，先用运动补全模型生成全身参考，再学习残差控制；第三视角已有全身重建时，则先重定向该参考，并针对噪声较大的手指接触使用 force-closure 目标。这是 G1 全身任务的另一条参考构造路径，本文的 Vega 纸巾盒命令没有调用该补全模块。[CHORD v2，§3.2 与附录 E](https://arxiv.org/abs/2607.00033v2)

## 10. 官方结果说明了什么 {#results}

阅读数字时，先分清对象是单任务数据链路，还是 CHORD 的大规模仿真评估。

| 来源与设置 | 官方报告 | 适用范围 |
| --- | --- | --- |
| CHORD 项目页 | 4,739 个 simulation-ready 任务；在 1,831 个基准任务上的平均成功率为 82.12% | 不等于 4,739 个任务全部评估，也不是网页视频转真机成功率 |
| E2E 纸巾盒仿真案例 | 用 100 条 rollout 训练后，100 次完整闭环中完成 87 次 | 对应该示例的策略、数据和评估设置，不是本文上面 10 条采集示例的结果 |
| 纸巾盒真机适配 | Sim200 预训练后用 Real10 继续训练 5 epochs，完成 4/5 | 单一任务、5 次实机试验，需要按小样本案例理解 |
{.table-readable}

第一行来自 [CHORD 官方项目页](https://nvidia-isaac.github.io/video_to_data/chord/)，后两行来自固定版本的 [E2E 结果说明](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/docs/e2e_workflow.md#sim-to-real-validation-with-limited-real-data)。

### 10.1 CHORD 的任务成功如何定义

CHORD 论文 §4.1 对每个任务重复执行 rollout：执行到任务末尾，且没有触发物体位置误差超过 15 cm 或旋转误差超过 40° 的提前终止，才算该次 rollout 成功。**Completion ratio 是该任务成功 rollout 数除以总 rollout 数**，不是单条轨迹完成了多少比例的时间步。这个比例大于 0.7，才把该任务计为成功。

因此，82.12% 是 1,831 个任务按上述规则判定后的任务成功比例，不能直接读成“随机上传视频后有 82.12% 的真机执行成功”。这个口径也没有覆盖任务语义的全部细节，例如所有物体都精确回到指定放置区域。

此外，大规模实验训练的是**逐参考序列的专家策略**。论文附录 B 明确把它称为训练配方在异质任务上的可扩展性，而非同一个策略的跨任务泛化。后续从专家采集视觉数据、微调 GR00T 是另一层学习过程；不能把 1,831 个任务上的专家结果直接写成一个通用 VLA 的表现。

接触引导的价值还有更直接的消融证据。论文从基准中选出 60 个任务，每任务使用三个训练种子；仅改变接触奖励，其余物体跟踪和运动模仿项保持相同。表 2 的两项连续误差如下，数值是报告的均值 ± 标准差，均越低越好：

| 接触奖励 | 关节物体角度误差，° | 物体 MPPE，cm |
| --- | ---: | ---: |
| CHORD wrench | 4.34 ± 0.25 | 5.11 ± 0.39 |
| 仅接触位置 | 6.41 ± 0.44 | 7.63 ± 0.06 |
| 无接触奖励 | 9.81 ± 0.73 | 14.38 ± 2.23 |

这里第一项是 articulated object 的关节位置误差，不是机器人 58 维关节误差；MPPE 使用物体关键点表示位姿偏差。该对照支持“接触作用方向提供了位置距离以外的有效信息”，实验范围仍是这组任务和训练配方。[CHORD v2，§4.1～4.2、表 2](https://arxiv.org/abs/2607.00033v2)

### 10.2 少量真实数据适配提供了什么证据

真机适配中的 Real10 是**由专家生成的一条轨迹在真机上开环回放得到的十条记录**，并非十种任务，也不是人类遥操作数据。文档中的对照，原始 GR00T N1.7 只用 Real10 或 Real30 各训练 5 epochs，均为 0/5。该案例支持“这项任务中，仿真预训练提供了有用初始化”，还不能推出跨任务或跨物体的普遍成功率。

还应保留失败证据：官方另有直接混合仿真与真实数据的 co-training 检查，Sim100+Real10 和 Sim100+Real20 都是 0/5。它提醒我们，仿真数据数量、真实适配顺序、外观差异与采样配比都会影响结果，不能把“加入仿真数据”视为充分条件。

## 11. 常见失败应该回到哪一层 {#failure-modes}

| 现象 | 优先核对 | 原因 |
| --- | --- | --- |
| 检索得到画面，但没有完整抓取过程 | 切分边界、动作描述、verify/refine 与查询放宽级别 | 帧相似不能替代动作时间边界 |
| 视频叠加看着正常，世界轨迹却漂移 | 相机轨迹、变换方向、尺度与重力对齐 | 相机系中的一致性不等于世界系稳定 |
| IK 误差很小，手却抓不住 | 物体尺度、碰撞几何、摩擦、执行器与接触目标 | 几何接近没有保证足够的接触支撑 |
| 动画很好，物理回放失败 | 是否只看了 kinematic reference，是否有辅助控制 | 直接设置姿态绕过了动力学 |
| 相机无输出或大量黑帧 | LFS 资产、渲染预热、相机路径、渲染并行度 | 相机路径与纯物理采集的资源要求不同 |
| GR00T 能训练，执行时却动作错乱 | 关节顺序、绝对/增量动作、统计、相机与时间对齐 | 相同维数仍可能具有不同语义 |
| 日志采集成功很多，任务成功率却低 | source eligibility 与 task evaluator 是否混用 | 到达 timeout 不代表完成操作 |
| 仿真成功，真机失败 | 标定、外观、动力学、延迟和真实适配协议 | 仿真闭环只验证了对应仿真分布 |
{.table-readable}

对于新任务，先保存一个完整、可信的小样本：能解释每个坐标系、每条动作和每个成功条件，再扩展数据与并行度。否则错误会在多阶段缓存、数据转换和训练中被放大，而且很难再定位到原始帧。

## 12. 与本站相关文章的联系 {#related-reading}

- [机器人学习数据来源综述](/posts/ai/vla-world-model-data)解释人类视频、遥操作和 UMI 数据的监督差异；V2D 展示了把人类视频继续转换为仿真机器人示范的一条路径。
- [StableHand](/posts/ai/stable-hand)重点讨论世界坐标双手轨迹恢复与观测质量；V2D 进一步处理物体、机器人映射、物理执行与策略数据。本文版本的 V2D 手部路径不应被误写成默认集成 StableHand。
- [MolmoMotion](/posts/ai/molmo-motion)预测物体表面点的未来三维运动；V2D 则恢复示教中已发生的交互，并用它构造仿真与机器人学习任务。
- [LeRobot 与 ACT](/posts/ai/lerobot-act)有助于理解图像、本体状态、动作及 action chunk；V2D 的 LeRobot 转换也必须先定义这些字段的语义。

## 阅读自测与验收

**感知、几何与物理**

- **V2D 的视频检索是不是 E2E 的必选第一步？** 不是。它用于筛选候选片段；纸巾盒 E2E 可直接从指定视频开始。
- **为什么参考轨迹不能直接等同于机器人策略？** 参考给出目标，策略根据当前状态产生动作，物理结果还受接触和动力学影响。
- **verify/refine 默认开启吗？** 本文版本默认关闭；启用后也可能因为严格检查而没有合格片段。
- **世界系物体位姿怎样获得？** 将当前相机系物体位姿与相机到世界的变换按方向正确组合，并统一尺度与时间。
- **`motion_v1` 的四元数和关节顺序有什么要求？** 四元数为 wxyz，关节位置列与显式关节名严格对应；检查单位范数和维数还不够。
- **CHORD 的 wrench 表示是否意味着从视频测出了接触力？** 不意味着。接触几何和摩擦约束形成作用能力特征，归一化奖励特征不能直接读作真实力传感器数值。
- **纸巾盒 Vega 配方直接匹配人手接触 wrench 吗？** 当前配方启用逐手门控的 force-closure 奖励；直接 wrench 匹配及漏接触、非预期接触项权重为零。

**专家、观测与动作**

- **Vega 的 58 维动作是什么？** 当前 contract 中双臂与双手的绝对关节位置目标，单位为 rad；维数与字段语义由该机器人约定。
- **状态和动作都是 58 维，能按相同下标直接比较吗？** 不能；当前状态的左臂在 `[7:14]`，动作的左臂在 `[29:36]`，必须先按字段和关节名对应。
- **纸巾盒 PPO 专家与 GR00T 使用相同的视觉输入吗？** 不相同；专家使用仿真状态和参考运动，GR00T 使用三路 RGB、机器人状态和文字任务。
- **三个参考时刻是否都严格晚于当前时刻？** 不是；默认参考偏移为 0、2、4 个播放帧，包含当前参考。
- **VOC 全局目标到零后，训练 reset 还会有辅助吗？** 会；当前训练配方重采样时仍从尺度 1 开始，参考到第 51 次命令更新才允许前进；采集和评估另外关闭辅助、冻结和课程。
- **50 Hz 的参考文件意味着应把 motion.dt 设成 0.02 秒吗？** 不意味着；当前环境为 20 Hz，motion.dt 应为 0.05 秒，加载器另外根据运动速度重采样参考。
- **预测 16 步、执行 4 步时，第 104 步沿用第 100 步的预测吗？** 无 reset 的常规节奏下，第 104 步会用新观测重新查询，替换旧 chunk 中未执行的预测。
- **PPO 原始输出为零，绝对目标是否立即回到参考？** 不一定；上一控制步的滤波残差仍会衰减。GR00T 已预测绝对目标，推理时不再套用这条残差合成。

**采集与数据接口**

- **`--target-successes 10` 等于十次操作成功吗？** 它指采集选择十条 timeout-eligible 源轨迹；总尝试数和闭环任务成功率另算。
- **采集步数上限和实测记录帧数为何不同？** 前者是覆盖运动时长并留余量的安全上限，后者是 pilot 测出的实际 episode horizon。
- **输入视频、重定向参考与控制频率能共用一个 fps 吗？** 不能；应分别核对真实源帧率、参考重采样频率和 20 Hz 控制约定。
- **数据里的 `next.done=true` 是任务成功标签吗？** 不是；当前转换器用它标记序列最后一行，`next.reward=0` 也只是占位。
- **默认开环评估使用独立留出数据吗？** 没有；转换器将全部 episode 标为 train，启动器的开环命令仍读取训练数据目录。
- **为什么不把同一轨迹的 16 步窗口随机拆成训练与验证？** 相邻窗口高度重叠；应先按完整 episode 及源视频分组，再生成窗口并仅用训练组估计统计。

**任务评估与运行**

- **开环 MSE 有限和 `[DONE]` 各证明什么？** 分别证明开环接口产生了有限误差、评估执行已结束；任务成功仍需查看闭环指标。
- **纸巾盒 `lift_hold` 是否验证任意位置放置与松手？** 没有；当前评估基于抬升、保持和末段高度稳定等条件，不检查 XY、接触或手释放。
- **抬升门槛为 10 cm，是否必须在 10 cm 以上连续保持 20 次？** 不需要；峰值至少 10 cm 与连续 20 次不低于 5 cm 分别检查，完整通过还须满足下降与末段条件。
- **掉落后最后一秒静止，是否一定通过？** 不一定；整段最大单步下降也要通过，末段稳定不能覆盖此前的下降突变。
- **87/100、4/5 与 82.12% 能相互比较吗？** 不能直接比较：它们分别来自纸巾盒仿真、少量真机适配试验和 CHORD 多任务仿真评估。
- **末段样本数少于 20，就一定没有完整稳定窗口吗？** 不一定；评估器取整段最后 20 次更新，窗口可能包含缓慢的过渡段，仍须检查高度、相邻变化和极差。
- **切换运行后，如何确认命令仍作用于纸巾盒目录？** 显式传入它的 `--config` 路径，并用 `status --json` 核对 `config` 和 `run_root`；新终端须重新设置 `E2E_CONFIG`。
- **V2D 重建预览会保存新阶段参数吗？** 不会；`--dry-run` 打印命令，正式运行仍需传入相同选项。

## 参考资料

1. [NVIDIA Isaac Video to Data 仓库与固定版本](https://github.com/nvidia-isaac/video_to_data/tree/193382a184308f6800331f3db829f76dc4294cfe)。
2. [Video Ingestion Agent](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/video_ingestion_agent/README.md) 与 [默认入库配置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/video_ingestion_agent/configs/ingestion.yaml)。
3. [Reconstruction 模块](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/README.md)、[第一视角重建设置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/docs/ego_e2e_setup.md) 与 [统一入口](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py)。
4. [Robotic Grounding 架构](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/docs/ARCHITECTURE.md)、[运动格式](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/source/robotic_grounding/robotic_grounding/motion_schema/README.md) 与 [环境和数据设置](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/docs/SETUP.md)。
5. [CHORD: Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration](https://nvidia-isaac.github.io/video_to_data/chord/)，官方项目页与技术报告入口。
6. [FlashCHORD](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/flash_chord/README.md) 与 [接触 wrench 特征实现](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/flash_chord/src/flash_chord/objectives/wrench.py)。
7. [重建到策略的 E2E 指南](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/docs/e2e_workflow.md) 与 [GR00T post-training 接口和成功语义](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/robotic_grounding/groot_finetune/README.md)。
