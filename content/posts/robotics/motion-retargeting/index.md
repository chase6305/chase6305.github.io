---
title: "人形机器人动作重定向详解：GMR、UMR 与 OTRetarget"
date: 2026-11-24
lastmod: 2026-11-24
draft: false
tags: ["Motion Retargeting", "Humanoid Robots", "Cross-Embodiment", "Inverse Kinematics", "Optimal Transport", "Paper Notes"]
categories: ["机器人", "人工智能"]
authors: ["chase"]
summary: "沿手工连杆映射、学习表面点对应与最优传输三条路线，解释 GMR、UMR 和 OTRetarget 怎样生成机器人参考、保留接触并决定物体轨迹，结合固定源码、可手算例子与论文实验说明如何选型。"
description: "沿手工连杆映射、学习表面点对应与最优传输三条路线，解释 GMR、UMR 和 OTRetarget 怎样生成机器人参考、保留接触并决定物体轨迹，结合固定源码、可手算例子与论文实验说明如何选型。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
images: ["assets/box-height.webp"]
reading_prerequisites: "刚体坐标变换、正运动学与最小二乘；正文从代价矩阵和边缘约束解释最优传输"
reading_focus: "先判断示教给出了哪些几何信息，再比较谁建立对应、每帧有哪些优化变量、哪些接触条件进入代价或约束；用搬箱算例检验选型。"
related_posts:
  - "/posts/robotics/contact-preserving-retargeting"
  - "/posts/ai/video-to-data"
  - "/posts/ai/card-retargeting"
  - "/posts/ai/morphmimic"
  - "/posts/ai/light-loco-parkour"
---

同一段人体示教里，舞步要求脚在合适的时刻落地，双手搬箱则要求手贴住箱体、箱底最终落到桌面。换成前臂较短的机器人后，即使腕部姿态看起来很像，手也可能够不到原来的箱子轨迹。要解决这个问题，先要决定重定向能改变哪些量，包括连杆姿态、表面接触位置和箱子本身的运动。

**GMR 用人工指定的人体—机器人连杆对应驱动逆运动学。UMR 在标准姿态学习稠密表面点对应，再逐帧用这些点拟合姿态与接触。OTRetarget 论文通过最优传输转移表面交互目标，并在求机器人姿态时一并调整可移动物体的位姿。** 三者都生成供后续跟踪器使用的运动学参考。本文把 OTRetarget 论文方法与 lucidrains 的 PyTorch 实现分开讲，后者可用于理解和复用其中的几何计算。版本记录见[论文与源码](#sources)。

## 1. 同一段运动可以怎样描述 {#representations}

重定向的数据流是“人体与场景记录 → 目标几何 → 机器人参考 → 策略或控制器跟踪”。例如，输入可能是 BVH 骨架，可能是由 SMPL-X 参数生成的逐帧人体网格，也可能同时带箱子网格及其世界位姿。视频重建负责恢复这些输入。求解器根据机器人模型得到浮动基座位姿和关节位置，跟踪器再结合状态反馈和动力学执行。给定机器人构型 $q_t$，其手腕、掌面与脚底各在哪里，由正运动学和对应的局部几何共同决定。

| 方法 | 如何建立几何对应 | 逐帧优化变量 | 适用输入 |
| --- | --- | --- | --- |
| [GMR](#gmr) | 人工配置关键连杆、缩放与姿态偏移 | 机器人构型 | 已有 [BVH 骨架](#gmr-bvh)或 SMPL-X 动作，主要复现全身姿态与步态 |
| [UMR](#umr) | 对齐标准姿态，学习共享索引的表面点 | 机器人构型。场景或物体轨迹是给定参考 | SMPL-X 网格或带动作的[源角色 MJCF](#umr-character)，需要保留表面姿态或接触 |
| [OTRetarget 论文](#otretarget) | 手工指定部件对，部件内通过最优传输匹配表面采样点 | 机器人构型及各可移动物体的位姿 | 已有交互场景几何和示教轨迹，允许物体路径适应机器人。可先试[求解回调算例](#ot-solver-callback) |
{.table-readable}

需要区分三个层次。**姿态拟合**量化连杆或表面是否跟上示教，**交互拟合**量化哪一侧、哪个位置、以多大间隙接近另一个表面，**动态可行性**还涉及支撑力、摩擦、扭矩和跟踪带宽。三者对应各自的验收指标，尤其要把几何接触与反馈控制下的动态接触分开测量。

## 2. GMR：先校准连杆，再做两阶段逆运动学 {#gmr}

### 2.1 配置表就是人体与机器人的接口

输入适配器把 SMPL-X、BVH 等数据整理成同一帧的命名部件及其世界位姿。目标机器人的 MJCF 提供骨架、关节限位和正运动学。在 `smplx_to_g1.json` 中，机器人 `left_hip_roll_link` 对应人体 `left_hip`，每个条目还带位置权重、旋转权重、局部平移偏移和旋转偏移。`pelvis` 是根部，配置假定的人体参考身高是 1.8 m，根部与腿部缩放系数为 0.9，肩、肘、腕为 0.8。新机器人的主要工作是确认这些部件对应、静止姿态下的坐标轴、权重和模型限位。 [SMPL-X → G1 配置](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/ik_configs/smplx_to_g1.json)

用 $p_0$ 表示源根部位置、$p_i$ 表示人体部件位置，$s_0,s_i$ 为配置表中的系数，$a=h/h_{\rm ref}$ 为输入与参考身高比。应用静止姿态偏移前，位置目标为

$$
\tilde p_0=a s_0p_0,\qquad
\tilde p_i=\tilde p_0+a s_i(p_i-p_0).
$$

这里的位移 $p_i-p_0$ 以根部为原点，分量仍沿世界坐标轴。假设 $a=1$，骨盆沿世界 $x$ 在 1.00 m，手腕在 1.50 m，目标骨盆在 0.90 m，目标手腕在 1.30 m。若把手腕的 0.8 直接乘整个世界坐标，会得到 1.20 m，连根部位移也跟着手腕系数缩小。固定偏移另外处理，源码先右乘旋转偏移，再用已校正的朝向旋转局部平移偏移，将其加到世界位置上。 [GMR 论文 §IV](https://arxiv.org/html/2510.02252v1#S4) · [缩放与偏移实现](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/motion_retarget.py#L272-L321)

### 2.2 两张任务表让粗对齐与细化使用不同权重

论文第一阶段优先对齐部件朝向与手脚位置，第二阶段用更广的部件位置目标细化。发布代码按两张可独立启用的表建立 Mink `FrameTask`，实际启用哪些位置项以配置为准。例如 G1 的第一表还给骨盆位置非零权重，而第二表除自身任务权重外，在该固定版本的 `update_targets()` 中仍读取用第一表偏移预处理的目标。改 JSON 时，分开核对“这一项的权重”与“这一项最终跟踪的目标位姿”。 [GMR 论文 §IV](https://arxiv.org/html/2510.02252v1#S4) · [两阶段任务与目标实现](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/motion_retarget.py#L111-L248)

每个差分 IK 步先在当前构型的切空间用 Jacobian 预测残差，再求一个带阻尼和限位的局部更新。可将该步理解为寻找增量，使 $e(q)+J(q)\Delta q$ 的加权范数变小，并用模型提供的积分更新根部四元数和关节。复用同一 `GeneralMotionRetargeting` 实例会保留上帧配置，第二阶段也直接从第一阶段的结果开始。实现默认把 `ConfigurationLimit` 送进求解器，构造器的 `use_velocity_limit` 默认 `False`。开启后才加 `VelocityLimit`。求解步使用 MJCF 的 `opt.timestep`，输出帧率来自数据适配器。检查导出轨迹的真实关节速度时，应按输出采样时间重新差分。 [构造器与限位](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/motion_retarget.py#L15-L109) · [单片段导出源码](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/smplx_to_robot.py#L65-L167)

用固定提交的求解器跑一个合成铰链，可以看到这两种时间尺度如何不同。取内部步长 5 ms，开限位后单次积分允许约 0.0471 rad。两阶段多次积分后，一次 `retarget()` 调用从零更新了约 1.0367 rad。如果把前后两帧作为 30 FPS 的相邻导出帧，帧间差分约为 31.1 rad/s，超过内层的 $3\pi$ rad/s。启用该限位时还要检查名称。这个版本从模型的 actuator 名称构造限位字典，Mink 按 joint 名称解析。固定 G1 XML 的 29 对名称一致，合成模型把两者改成不同名字时会在启用限位的构造阶段报错。 [GMR 两段积分](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/motion_retarget.py#L183-L248) · [固定 G1 actuator 字段](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/assets/unitree_g1/g1_mocap_29dof.xml#L255-L285)

### 2.3 从一段 SMPL-X 动作生成 G1 参考

在固定版本的 GMR 仓库根目录准备 README 指定的 SMPL-X 模型与动作 `.npz`，然后运行单片段查看器和导出脚本。这个输入适配器默认用 $1.66+0.1\beta_0$ m 估计源人体身高，关节任务配置里还各有比例。接自有动作时，记录估算身高和配置假定身高。 [GMR 数据准备](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/README.md#data-preparation) · [SMPL-X 输入适配](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/utils/smpl.py#L15-L49)

```bash
uv sync
uv run scripts/smplx_to_robot.py \
  --smplx_file /path/to/motion.npz \
  --robot unitree_g1 \
  --save_path output/g1-motion.pkl
```

脚本把 30 FPS 作为对齐参数。固定提交中，若源帧率低于或等于 30 FPS，函数保留原帧并返回 30 FPS。源帧率超过 30 时，先以 $\lfloor f_{\rm src}/30\rfloor$ 为除数计算输出帧数，再在首尾索引之间对位置做线性插值、对旋转做球面线性插值。返回的帧率是 $f_{\rm src}\times N_{\rm out}/N_{\rm in}$。下面是将同一段合成 60 帧输入送进该函数时的结果。 [GMR 帧率适配实现](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/utils/smpl.py#L172-L260)

| 源 FPS | 适配后的帧数 | 返回的 FPS |
| ---: | ---: | ---: |
| 24 | 60 | 30 |
| 50 | 60 | 50 |
| 60 | 30 | 30 |
| 100 | 20 | 33⅓ |

导出包括 `root_pos`、`root_rot`、`dof_pos` 和 `fps`。内部 MuJoCo 的根部四元数采用 `wxyz`，这个脚本把 `root_rot` 转为 `xyzw` 再保存。单片段循环在取帧前先递增索引，普通非循环调用从**适配后**的索引 1 开始。尤其是原片段低于 30 FPS 时，应同时核对源时间、适配后帧数和导出的首帧，再将结果与原接触标注对齐。论文还描述了整段最低高度校正，单片段脚本直接写入求解得到的根部位置。 [GMR 导出循环](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/smplx_to_robot.py#L101-L167) · [论文 §IV 序列处理](https://arxiv.org/html/2510.02252v1#S4)

### 2.4 从 LAFAN1 BVH 生成参考 {#gmr-bvh}

只有 LAFAN1 骨架动作时，可用 GMR 的 BVH 入口。适配器把原始位置从厘米换算成米，将 Y-up 坐标旋转为 Z-up，用脚趾朝向和脚踝位置合成 `LeftFootMod`、`RightFootMod`，并用 1.75 m 作为源身高。命令里的 `--motion_fps` 写进结果。对于头部 `Frame Time` 为 `0.033333` 秒的片段，可传 30。换帧率时，应从文件头读取原间隔或先明确重采样。 [BVH 适配](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/utils/lafan1.py) · [BVH 命令](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/bvh_to_robot.py)

```bash
uv run scripts/bvh_to_robot.py \
  --bvh_file /path/to/walk3_subject5.bvh \
  --format lafan1 --robot unitree_g1 --motion_fps 30 \
  --save_path output/g1-walk.pkl
```

固定版本的 BVH 非循环脚本先求解和显示一帧，再检查是否结束。最后一帧触发退出时尚未写入列表。取 FRDV 附带的 `walk3_subject5.bvh` 前 60 帧复放，查看器收到 60 帧，PKL 保存原索引 0–58 的 59 帧，`dof_pos` 形状为 `(59, 29)`，`root_rot` 顺序仍为 `xyzw`。该复放保留 G1 的运动学链和任务配置，用记录式查看器捕获帧。 [BVH 导出循环](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/bvh_to_robot.py#L115-L171) · [FRDV 样例](https://github.com/DARoSLab/FRDV/blob/72298bc095b2ed5fe0fe94933c795668e5bf88af/assets/mocap/sample/walk3_subject5.bvh)

同一 BVH 也可用 [FRDV 导出 NPZ](#frdv-bvh)，对照两条管线保留的源帧和根部排列。

## 3. UMR：从无序点云学习可随身体移动的对应 {#umr}

### 3.1 训练发生在标准姿态，输出是点对应

源端需要对齐的标准 T-pose 网格，以及动作每帧的变形表面。机器人端需要 MJCF、外表面网格和对齐的 T-pose。UMR 给源表面样本编号，目标点云起初无序。PointNet 风格编码器读取目标点云，MLP 为每个源点预测位移。第 $i$ 个输出点是源点 $x_i^h$ 加上对应位移，因而保留同一个索引 $i$。训练信号有三项。

- **双向 Chamfer 距离**让预测点靠近机器人点云，也让机器人表面获得覆盖。仅用这个损失时，几何相近的点可以互换索引。
- **点排斥**惩罚预测点聚集，降低大量源点挤向机器人同一区域的风险。
- **模板测地邻接的位移平滑**让人体表面本来相邻的点接受相近的位移。沿网格连接的距离可以区分空间里很近、沿体表却相隔较远的区域。

发布的训练脚本还可惩罚位移的整体幅度。默认流水线配置使用 `geodesic` 邻接，单独调用训练脚本时 `--edge-graph` 的参数默认值是 `euclidean`。复算论文的测地邻接时，需要显式选 `geodesic` 并提供模板三角网格和样本所属面。 [训练参数与邻接构造](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L301-L447) · [流水线默认配置](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/humanoid_retarget_defaults.json#L58-L80)

绑定后，人体点随源网格三角形做重心插值，机器人点随所属连杆做刚体运动。若人体点在三角形三个顶点上的权重为 $0.2,0.3,0.5$，最后一个顶点单独抬高 20 cm，人体点就抬高 10 cm。第 $i$ 个机器人点则保留其连杆局部坐标 $s_i$，用该连杆在构型 $q_t$ 中的旋转与平移得到世界位置。两个点的变换方式各自匹配源网格和机器人模型，并通过同一索引参加位置残差。人体部位标签也沿索引传递，可用于选点与加权。 [UMR 论文 §III-B、III-C](https://arxiv.org/html/2609.02134v2#S3)

同一源模板和目标机器人可摊销学习成本。换成同一模型的新动作片段时通常复用对应点。更换源模板、机器人模型或标准姿态时，需要检查绑定是否仍适用。 [对应训练脚本](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py)

### 3.2 运行时用表面位置、朝向和接触求解机器人

固定点对随时间给出位置目标与相对标准姿态的法向变化。接触由示教选出。对源点 $x^h_{t,i}$ 找环境点云中的最近点 $y_{t,\pi(i)}$，把足够近的点对加入活动集合。位置残差控制全身表面拟合，接触残差提高特定配对的优先级。论文式 (10) 的外部接触向量使用同一个环境锚点，因而对于该固定锚点有

$$
(x^r_i(q_t)-y_{t,\pi(i)})-(x^h_{t,i}-y_{t,\pi(i)})
=x^r_i(q_t)-x^h_{t,i}.
$$

在这种写法里，锚点通过**活动点集合与权重**决定谁被重点拟合。两边使用同一个固定世界点时，差向量会消去它。自接触使用另一个身体表面对应点，机器人侧的第二个点也随构型变化。发布版本的 HOI 路径还有物体局部目标、地面和穿透选项，需要结合配置检查。 [UMR 论文式 (6)–(11)](https://arxiv.org/html/2609.02134v2#S3.SS3) · [表面求解代码](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py)

固定版本的单人表面求解默认使用 `tpose_offset` 法向目标。它把机器人标准姿态的法向沿绑定的人体三角面随动，再与当前机器人法向比较。这样即使两边在 T-pose 的法向略有差异，也可以跟踪同一演示中的局部转动。该求解器的 `self_contact_map_cost` 分支也有具体取舍。它只给机器人点对距离小于源点对距离的情况加惩罚。例如源点相隔 5 cm，机器人相隔 3 cm 时产生 2 cm 的距离差，相隔 7 cm 时这项代价为零。查看握持或双手相碰时，仍应测量点对是否实际接近。 [法向目标](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L2609-L2673) · [点对残差](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L1811-L1837)

求解器将残差在当前构型线性化，通过带阻尼的 Gauss–Newton 子问题更新姿态。以地面为例，当前表面点高度为 $z_i$，地面高度为 $z_f$，点高 Jacobian 为 $J_i^z$，一步近似约束就是 $-J_i^z\Delta q\le z_i-z_f$。源码用 MuJoCo 计算运动学，用 Clarabel 解局部子问题。HOI 路径把当前帧的物体位置和 `wxyz` 四元数从输入轨迹写入接触与碰撞计算，每步更新仍是机器人模型的切空间增量。离线批处理可分别正向、反向热启动，再通过动态规划在两路候选中选择，降低偶发差解与关节跳变。反向分支依赖未来帧。 [UMR 论文式 (12)–(14)](https://arxiv.org/html/2609.02134v2#S3.SS3) · [逐帧求解与输出代码](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L2695-L2875) · [批处理说明](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/README.md#batch-retargeting)

读取速度时要看采样周期。UMR 单人入口的 `max_dq` 默认值为 0.15，在一次局部 QP 里限制关节增量；根部的步长另有开关。一个输出帧通常包含多次迭代，接控制器前可以像 GMR 一样，用保存的源帧索引及时间重新计算关节速度。G1 的 29 个关节连同浮动基座，在 `qpos` 中占 36 个位置分量。自由基座用四元数写入朝向，它的转动增量只有 3 个分量，所以整个局部更新向量是 35 维。 [局部步长的默认值](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L444-L451) · [子问题大小与更新](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L1979-L2117)

### 3.3 从官方单片段示例开始

按照 UMR README 准备独立环境，将用户取得的 SMPL-X 模型按预期放在 `smpl/`。官方样例内含一段 LAFAN1 转成的 SMPL-X 舞蹈。以下入口会建立或复用点对应，导出包含 `qpos`、`fps` 和 `frame_ids` 的 NPZ，`--skip-view` 可关闭查看器。尝试 OMOMO 时，还需要相应的序列、物体几何与逐帧位姿，并采用仓库文档要求的目录结构。 [UMR 单片段入口](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline.py) · [UMR 上手说明](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/README.md#quick-start) · [OMOMO 输入说明](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/sample_data/omomo/README.md)

```bash
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --skip-view
```

单人表面求解器按 `motion.stride` 选帧，并用源帧率除以这个步长来写出 `fps`。例如 60 FPS 输入每隔一帧取一帧，导出为 30 FPS，`frame_ids` 保存实际选中的源索引。接触标签可用这些索引对齐。 [UMR 帧选择](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L2133-L2138) · [输出字段](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L2858-L2868)

手头没有 SMPL-X 模型时，可用仓库自带的源角色和动作先跑一次。 [Character CPU 样例](#umr-character)

入口会用模板名、机器人名或采样名，以及 `training_mode` 判断点对应缓存能否复用。默认 `all_samples` 模式容许旧缓存省略 `training_mode`，`robot2smpl_only` 则要求该字段匹配。固定版本的训练入口在显式固定模板或批大小设为 1 时都切换到 `robot2smpl_only`。用两组各 12 点的合成网格在 CPU 上训练，批大小为 1 时导出的源模板位移全为零，批大小为 2 时两组都会参与重建。调小批大小时也要核对 `training_mode`、`train_sample_indices` 和 `fixed_template_indices`。 [训练模式](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L400-L417) · [对应输出](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L558-L627)

若在原路径上换了机器人网格、标准姿态或采样设置，这些元数据可能保持一致。此时可用新的缓存目录，或给全流程加 `--force-build --force-train --force-retarget`，分别重建采样、训练对应和导出轨迹。只想核对将运行哪些阶段时，可以先加 `--dry-run`。 [缓存判定](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline.py#L379-L456) · [重建开关](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline.py#L139-L146)

如需核对上面的实现细节，可用固定 checkout 运行下面的源码复放。`--gmr` 在 MuJoCo 中加载单铰链模型，用 Mink 1.3.0、MuJoCo 3.13.0 和 DAQP 0.9.1 检查两阶段目标与更新。帧率适配使用合成两关节序列，导出测试用记录式查看器检查首帧、四元数和帧率字段。`--umr` 只依赖 NumPy，检查构造 NPZ 的缓存判断。脚本对选用的源文件校验 SHA-256。 [下载 `probe_motion_sources.py`](probe_motion_sources.py)

```bash
python3 -B probe_motion_sources.py --gmr /path/to/GMR
python3 -B probe_motion_sources.py --umr /path/to/UMR
```

在装有 CPU PyTorch、NumPy 和 SciPy 的环境中，给第二条命令加 `--check-umr-training`，可用上面的两个合成点云复算训练模式。它会分别走两次实际训练入口，并检查导出模板顺序、点对数量与模式字段。

### 3.4 在 CPU 上跑通 Character 样例 {#umr-character}

MimicKit Humanoid Character 样例包含源角色 MJCF 和转身踢腿动作，可以沿用同一套 G1 目标配置。每帧依次是根部 XYZ、根部旋转向量和源角色 MJCF 顺序的关节分量，换动作模型时要同步更换其定义。用原版适配器读取 `humanoid_spinkick.pkl` 得到 60 FPS、78 帧，每帧有 34 个数。配置中设 `motion.start=12`、`motion.stride=2`、`motion.max_frames=8`，会取 `12, 14, …, 26` 八个源索引。此时 30 FPS 描述输出样本的相对间隔，第一帧仍对应源动作开始后的 0.20 s。 [UMR Character 输入与命令](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/sample_data/humanoid_character/README.md) · [Character 帧选择与帧率](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_character_to_humanoid_surface_vector.py#L412-L423)

同一源连杆上的三个共点转轴会在适配器里合并为一个球关节，三个输入分量一起作为旋转向量解码。若把它们逐列写进原始 MJCF 的三个铰链，样例的右脚在源索引 50 处会偏离适配器计算的位置约 55.8 cm。用 MuJoCo 独立复算时，将这些关节组改成球关节，用指数映射生成四元数，78 帧各连杆的位置差就低于 1 µm。 [源模型的关节分组](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/mimickit_compat/mjcf_char_model.py#L112-L178) · [解码与正运动学](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/mimickit_compat/kin_char_model.py#L48-L71) · [Character 适配器](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/character_surface_retarget_common.py#L145-L173)

把下方配置放到 UMR checkout 的 `robot_configs/`。它沿用官方 G1 模型和 T-pose，每侧采 512 点，将匹配训练缩为 16 个 epoch，并用独立的 `output/umr-character-cpu/` 保存三阶段结果。这个设置适合先检查采样、训练、求解和导出的接口。 [下载 `umr-character-cpu.json`](umr-character-cpu.json) · [Character 默认配置](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/humanoid_retarget_defaults_humanoid_character.json)

```bash
python3 -B scripts/humanoid_retarget_pipeline_character.py \
  --config robot_configs/umr-character-cpu.json \
  --skip-view
```

用固定版本运行，得到两组各 512 个对应点和 4,096 条模板邻接边。`character_cpu_smoke_g1.npz` 中的 `qpos` 有 8 行、36 列，`fps=30`，`frame_ids` 是 `[12, 14, …, 26]`。若把 `motion` 改为 `{"start": 0, "stride": 1, "max_frames": 0}` 并沿用同一输出路径，可加 `--stage retarget --force-retarget` 重求轨迹。实测只改采样配置时会复用已有的 8 帧结果。强制重求后导出 78 帧、60 FPS，点对应缓存的文件哈希保持一致。 [Character 缓存入口](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/humanoid_retarget_pipeline_character.py#L330-L400) · [NPZ 写出字段](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_character_to_humanoid_surface_vector.py#L755-L780)

先裁切再求解会改变初始姿态、热启动历史和地面估计。在这两次小配置运行中，配对到相同源索引后，根部位置仍最多相差约 9.3 cm。需要截取整段参考的一部分时，可先对全段求解，再按 `frame_ids` 取出相应行。 [Character 帧选择与初始化](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_character_to_humanoid_surface_vector.py#L412-L447) · [前帧热启动及导出](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_character_to_humanoid_surface_vector.py#L648-L776)

若已装 MuJoCo，也可给 `--umr` 加 `--check-umr-character`，用固定版本的模型和转身踢腿样例复算分组解码与帧选择。

## 4. OTRetarget 论文：一并求手脚和箱子的位置 {#otretarget}

### 4.1 每个探针记录距离、最近表面点和方向

把人体、机器人和场景物体的表面采样成探针。对探针的世界坐标 $x$，设其比较对象 $c$ 的位姿为 $(R_c,t_c)$，先算对象局部坐标 $x_c=R_c^{\mathsf T}(x-t_c)$。随后记录三元组 $(d,w,n)$。其中 $d$ 是相对对象表面的有符号距离（外部为正，内部为负），$w$ 是最近表面点，$n$ 指向从 $w$ 到探针的方向。论文在物体局部坐标预计算用于查询的体素网格，物体运动时只更新坐标变换。以箱子为例，手随箱子整体平移时，局部间隙和接触位置可以保持。只盯着一个间隙数值，则可能把薄板两侧的接近误认为同一交互。 [OTRetarget 论文 §III-A、III-B](https://arxiv.org/html/2609.36602v1#S3)

### 4.2 最优传输负责匹配表面采样点

论文先人工指定人体部件对应哪些机器人连杆，再在每对部件内部做匹配。将对齐 T-pose 的两个局部点云分别中心化并除以各自的均方根半径，用平方欧氏距离构造代价矩阵 $C$。源索引是行，目标索引是列。熵正则最优传输在非负矩阵 $P$ 上求解

$$
\min_{P\ge 0}\ \langle P,C\rangle
+\varepsilon\sum_{i,j}P_{ij}(\log P_{ij}-1),
$$

满足 $P\boldsymbol 1=a$ 与 $P^{\mathsf T}\boldsymbol 1=b$，其中 $a,b$ 是两侧的均匀边缘分布。Sinkhorn 对核 $K_{ij}=\exp(-C_{ij}/\varepsilon)$ 交替缩放行和列。软计划让每个点都有分配额度。论文随后对每个目标列取最大项 $\arg\max_i P_{ij}$，复制该源点在当帧的交互三元组。多个目标列仍可能选中同一个源点，因此硬化后的对应不保证一一匹配。替换物体时，可以对示教物体和替换物体另建对应，将最近表面点映到新表面。用来建立对应的归一化只发生在匹配空间，求解中的间隙仍以真实尺度计。 [OTRetarget 论文式 (3)、(4) 与 §III-E](https://arxiv.org/html/2609.36602v1#S3)

设两边各有两个点，代价矩阵的对角为 0、非对角同为 $\varepsilon\log 4$。这时核矩阵对角为 1、非对角为 0.25，对称性让等比例缩放就能得到行列和均为 0.5 的计划。计划的对角为 0.4、非对角为 0.1，硬化后第 1 列选第 1 行，第 2 列选第 2 行。这里的“传输”是离线分配点对应，一帧里的箱子位姿还要由下面的逆运动学决定。

### 4.3 联合优化同时处理手—箱、箱—桌和脚—地面

论文给每个活动探针—对象对设置三类交互残差。距离项以 $d-\max(\hat d,0)$ 衡量间隙，避免把示教的负间隙直接变成穿透目标。位置项比较两个最近表面点在对象表面上的测地距离。方向项是单侧惩罚，用来阻止探针从示教接近方向的错误一侧靠近。接近权重随距离衰减，论文用 13 cm 的核宽度。位置、方向两项根据示教间隙加权，距离项则还会考虑当前间隙，因而能处理示教原本较远、当前却发生靠近的点对。 [OTRetarget 论文式 (5)、(6)](https://arxiv.org/html/2609.36602v1#S3.SS3)

每帧的变量包括机器人构型 $q_t$ 与各可移动物体的刚体位姿 $T_t^o$。目标由交互残差、机器人姿态风格、物体示教轨迹偏差和机器人帧间平滑组成。机器人姿态目标用校正后的示教朝向和机器人自己的连杆长度构造。论文用信赖域 SQP，每次从 Pinocchio 的解析 Jacobian 组装局部 QP 并用 ProxQP 求解，同时纳入关节位置和速度、物体位移与参考偏差、采样表面的不穿透约束。桌子可以固定，箱子可在搬运中离开示教路径，再根据箱—桌交互落回桌面。输出是机器人与物体的运动学轨迹。抓持力、质量与摩擦属于后续物理验证。 [OTRetarget 论文式 (7)、(8)](https://arxiv.org/html/2609.36602v1#S3.SS4)

从对象局部坐标能直接看出机器人和物体为什么会进入同一条约束。沿论文的世界系旋转增量更新 $R_c$，将 $x_c=R_c^{\mathsf T}(x-t_c)$ 做一阶展开，得到

$$
\delta x_c=R_c^{\mathsf T}\bigl[\delta x-\delta t_c-\delta\theta_c\times(x-t_c)\bigr].
$$

例如箱子从单位朝向绕世界竖直轴转 0.1 rad，且手原来位于箱体原点右方 0.20 m。若手和箱原点都留在原位，手的局部 $y$ 坐标一阶变化为 −0.02 m。若手同时沿世界 $y$ 移动约 0.02 m，两项就互相抵消。约束使用这样的局部变化更新采样点间隙，箱子是被比较对象时，它的位姿增量和机器人的关节增量都会影响这一距离。 [对象坐标与 SQP 式 (2)、(8d)](https://arxiv.org/html/2609.36602v1#S3)

### 4.4 一个可以手算的搬箱片段

把空中搬运一帧简化成一维。机器人手高度 $h$ 最大只能到 1.20 m，示教箱体抓持点高度 $b_0=1.35$ m，目标手与抓持点同高。固定箱体时，最小间隙为 0.15 m。若该帧允许箱体沿高度调整，最小化 $(h-b)^2+0.5(b-b_0)^2$，加上 $h\le 1.20$ m，得到 $h=1.20$ m、$b=1.25$ m，间隙为 0.05 m。若硬性要求 $h=b$ 并允许箱体下降至少 0.15 m，可将两者都取为 1.20 m。相反，若箱体由于桌面几何和支撑条件只能满足 $b\ge1.30$ m，硬性同高就不可达。这个算例里，增加物体自由度改变了可行域，而软代价与等式约束决定求解器是否允许留下间隙。

<figure class="article-figure" id="fig-box-height">
{{< post-image src="assets/box-height.webp" alt="一维算例的箱体抓持点高度与代价。手最大可达 120 厘米。箱体可动时最低总代价在 125 厘米，固定时只能选 135 厘米。" >}}
<figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">对每个箱体高度先求最优手高度。固定在 135 cm 时，间隙平方为 225 cm²。允许箱体移动时，125 cm 处的总代价为 75 cm²，由 25 cm² 的间隙项和 50 cm² 的箱体先验组成。数值来自上面的一维算例。</span></figcaption>
</figure>

## 5. lucidrains 的 OTRetarget 可以直接用到哪一步 {#pytorch}

lucidrains 的仓库在固定提交中标记为 WIP，`pyproject.toml` 把它命名为 `OTRetarget-pytorch`，版本为 `0.0.1`。可以把它当作 PyTorch 组件库来阅读或接入。 [仓库 README](https://github.com/lucidrains/OTRetarget/blob/e6e87b85435439b2fa330f2fa0878e8c9dd81ca8/README.md) · [包定义](https://github.com/lucidrains/OTRetarget/blob/e6e87b85435439b2fa330f2fa0878e8c9dd81ca8/pyproject.toml)

| 函数 | 固定版本中的行为 |
| --- | --- |
| `surface_correspondence` | 归一化、建立平方距离矩阵、求传输计划并沿源点维取最大项。可传入已有计划或返回计划以便复用。 |
| `proximity_triple` | 由调用者提供 `dist_fn` 与 `closest_fn`，根据探针和最近点计算方向。坐标变换也由调用者安排。 |
| `interaction_residuals` | 返回加权的逐探针代价。传入 `geodesic_fn` 才采用调用者定义的表面距离，否则位置项使用欧氏距离平方。 |
| `transfer_object_targets` | 为每个示教最近点找最近的原物体样本，再从传输计划选出替换物体样本。可一并返回给定的替换表面方向。 |
| `joint_retarget` | 汇总风格、交互与可选平滑代价。传入 `solver_fn` 时把变量、目标、约束和迭代预算交给它，否则只返回初始状态的目标值。 |
{.table-readable}

张量形状以源点、目标点和坐标为基本轴，传输计划最后两轴为“源点数 × 目标点数”。`constraints` 只是转交给求解回调的对象。若需要复现论文里的关节和物体速度限位、胶囊自碰撞、距离网格、SE(3) 更新和 SQP，需自行提供这些模块。仓库测试使用合成球面与调用方传入的 Adam，只断言优化后合成目标下降。该测试能读出组件调用方式，论文的 ProxQP 求解和真机实验则有各自的设置。 [PyTorch 实现](https://github.com/lucidrains/OTRetarget/blob/e6e87b85435439b2fa330f2fa0878e8c9dd81ca8/ot_retarget/ot_retarget.py) · [仓库测试](https://github.com/lucidrains/OTRetarget/blob/e6e87b85435439b2fa330f2fa0878e8c9dd81ca8/tests/test_ot_retarget.py)

### 5.1 在 CPU 上检查点云形状与传输边缘 {#ot-pytorch-contract}

在 lucidrains 的固定 checkout 安装包后，可用三个源点和两个目标点检查计划的轴。下例在 CPU PyTorch 上运行。每个目标列返回一个源索引，共两个索引。软计划仍给三个源点各分配 $1/3$ 的质量；硬化后，中间的源点可以没有任何目标索引。 [对应函数](https://github.com/lucidrains/OTRetarget/blob/e6e87b85435439b2fa330f2fa0878e8c9dd81ca8/ot_retarget/ot_retarget.py#L128-L157)

```python
import torch
from ot_retarget import surface_correspondence

human = torch.tensor([[-1, 0, 0], [0, 0, 0.3], [1, 0, 0]], dtype=torch.float64)
robot = torch.tensor([[-0.8, 0, 0], [0.8, 0, 0]], dtype=torch.float64)
source, plan = surface_correspondence(human, robot, return_plan=True)

assert source.tolist() == [0, 2]
assert plan.shape == (3, 2)
torch.testing.assert_close(plan.sum(dim=-1), plan.new_full((3,), 1 / 3))
torch.testing.assert_close(plan.sum(dim=-2), plan.new_full((2,), 1 / 2))
```

对这个输入，两组点各自平移或做正的均匀缩放，软计划相同。只把机器人点绕竖直轴转 180°，返回索引会变为 `[2, 0]`。建立对应前要统一标准姿态的轴向。复用 `plan` 时则要保持两边的采样顺序，或者对计划的行和列做同样的重排。 [点云归一化与计划索引](https://github.com/lucidrains/OTRetarget/blob/e6e87b85435439b2fa330f2fa0878e8c9dd81ca8/ot_retarget/ot_retarget.py)

若直接向 `sinkhorn()` 传入自定义代价，还应检查边缘质量。它先直接计算指数核，再交替归一化，缺省迭代 50 次。在目标函数中给每项代价加同一个常数只会增加一个常量，因为计划总质量固定为 1。在实现的 CPU Float64 测试中，给一个非退化代价矩阵整体加 1000，指数核则下溢成零，返回全零计划。这个算例用于检验自定义代价下的数值边界。`proximity_triple()` 在探针和最近点重合时返回零方向。在分离时，该方向从最近点指向探针。需要外法线时，应从可信的表面几何另外计算。 [指数核与近接函数](https://github.com/lucidrains/OTRetarget/blob/e6e87b85435439b2fa330f2fa0878e8c9dd81ca8/ot_retarget/ot_retarget.py#L24-L60)

### 5.2 通过回调复算固定箱与可动箱 {#ot-solver-callback}

把第 4.4 节的手高和箱高接到实际的 `joint_retarget()`。手高作为 `robot_config` 的唯一分量，箱高是 `object_poses` 的平移分量。给定的 `style_cost_fn` 计算箱体先验，`interaction_cost_fn` 计算手—箱间隙平方。回调从初值提取两个高度，调用 PyTorch 求目标梯度，再交给 SciPy SLSQP，并在 SLSQP 的 `bounds` 中执行 1.20 m 的手部上限和箱高边界。 [回调接口](https://github.com/lucidrains/OTRetarget/blob/e6e87b85435439b2fa330f2fa0878e8c9dd81ca8/ot_retarget/ot_retarget.py#L205-L238) · [下载 `probe_motion_sources.py`](probe_motion_sources.py)

```bash
python3 -B probe_motion_sources.py --ot /path/to/OTRetarget
```

回调得到三个间隙，固定箱为 15 cm，箱可调整时为 5 cm，箱受 1.30 m 支撑下界限制时为 10 cm，与一维解析解一致。省略回调只会得到初始手高 1.00 m、箱高 1.35 m 的总代价 0.1225 m²。用这个例子接自己的求解器时，可以先确认约束对象确实进入回调并被应用。完整机器人还需定义关节与物体的更新方式，以及连续运动中的接触和碰撞检查。

## 6. 怎样读论文数字、选择实现并检查输出 {#evaluation}

### 6.1 三篇论文各自测量什么

- **GMR** 在 Unitree G1 上用 BeyondMimic 分别跟踪不同来源的 LAFAN1 参考。论文表 I 的 `Run (stop & go)` 在 IsaacSim 域随机化条件下，GMR、PHC、ProtoMotions 和 Unitree 参考的下游成功率分别为 91.24%、20.46%、40.26% 和 99.83%。这是该片段、该跟踪设置下完成 rollout 的比例。表 II 的位置误差衡量跟踪器相对本方法参考的偏差，论文另用盲测比较动作像不像原示教。 [GMR 论文 §III、§V](https://arxiv.org/html/2510.02252v1#S5)
- **UMR** 的表 I 区分一次性对应准备和逐片段吞吐。LAFAN1 设置下，采样、测地预计算和训练合计 25.79 s，单独重定向求解是 121.26 FPS，含数据预处理是 65.29 FPS，平台为 RTX 4070 Ti SUPER 与 Core Ultra 7 265KF。在下游任务里，UMR 的 Stair 成功率为 43.53%，对照 OmniRetarget 为 11.01%，Chair 则为 75.24% 对 79.67%。两类场景需要分别核对数据、动作阶段和判定条件。 [UMR 论文表 I、表 IV](https://arxiv.org/html/2609.02134v2#S4)
- **OTRetarget** 的表 III 报告 OMOMO 机器人—物体交互 Jaccard 中位数为 87%，OmniRetarget 为 28%，对应间隙误差 8.7 mm 与 29.3 mm。Jaccard 是按帧、部位和对象形成的两组近接样本之交集除以并集。论文用 20 mm 判近接，穿透也会进入交互集合，间隙误差另计。表 II 在同为原场景尺度的条件下，固定到可移动物体的 Jaccard 从 84% 到 87%，物体平均位置偏移的片段中位数为 2.9 cm。基线未返回的 424 条 OMOMO 序列被排除在其汇总外。论文的实机 G1 示例使用跟踪策略来执行参考，其评估与表 III 的几何指标分开。 [OTRetarget 论文 §IV](https://arxiv.org/html/2609.36602v1#S4)

### 6.2 从最小输入和最难接触开始验证

**只有骨架或实时动捕流**时，可以从 GMR 开始。先用静止姿态检查前向轴、根部高度和左右镜像，再用一段走停或转身检查支撑足滑移、限位和输出时间。换机器人时先改配置与模型，确认各连杆任务目标使用的局部坐标。

**有可变形人体表面，且给定物体轨迹或静态场景**时，可以从 UMR 开始。先看 T-pose 的点对，重点检查手掌、脚底、左右侧和相邻部件边界，再看点沿表面运动时是否保持绑定。对接触片段记录接触开始、持续和释放时刻，并在不同几何或体型上重新检查对应缓存。

**已有物体几何和轨迹，并允许物体随机器人可达范围调整**时，可以用 OTRetarget 论文的联合变量设计。先用同一段素材比较固定物体与可移动物体的结果，同时报告手—物间隙、接触位置、脚—地面和箱—桌面误差、物体偏移分布，以及求解失败的帧或片段。复用 lucidrains 组件时，先用已知平面或球面检查 SDF 符号、最近点和局部坐标，再替换成真实几何与受约束求解器。

三条路线最终都应检查导出坐标系、四元数列序、模型关节名称和顺序、实际采样间隔、场景尺度与接触相位。把工作流拆成运动学输出、仿真跟踪和真机试验之后，才能定位失败来自源动作估计、形态对应、可达性，还是下游跟踪。

### 6.3 对齐源时间再比较接触 {#align-exports}

先确定每个输出样本对应的源时刻，再用该时刻的物体位姿和接触标签解释它。例如把 24 FPS 示教送给上述 GMR SMPL-X 单片段入口，源索引 24 的接触发生在 1.000 s。文件中它在第 23 个零基索引处，将这个索引直接除以写入的 30 FPS 会得到 0.767 s。把源轨迹与输出画在同一时间轴上，接触会提前约 233 ms。用原始采样率和保留索引能恢复正确的源时刻。 [GMR 帧率适配](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/utils/smpl.py#L172-L260) · [保存循环](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/smplx_to_robot.py#L101-L167)

<figure class="article-figure" id="fig-source-time">
{{< post-image src="assets/source-time.webp" alt="60 帧合成输入的源时钟与导出时钟。横轴是 GMR 保留的源帧索引。实线用源索引除以 24，虚线用该索引减 1 后除以 30，源索引 24 处相差约 233 毫秒。" >}}
<figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text">索引错一帧改变时间起点，帧率差异则让误差继续累积。60 帧的合成输入在源索引 59 处已相差 525 ms。</span></figcaption>
</figure>

即使 60 FPS 降到 30 FPS，也要检查源时间。这个版本用包含首尾端点的等距插值生成适配样本，再丢掉第一个适配样本。对 60 个源样本，适配有 30 个位置，实际保存 29 个。从输出零基索引 $j$ 可以恢复源索引 $s_j=59(j+1)/29$，例如第一行对应 $s_0\approx2.034$。末行对应约 983.3 ms，直接用输出索引除以 30 得到 933.3 ms。即使把两条时间轴都从第一条导出记录起算，首末样本的跨度仍相差约 16.1 ms。 [适配网格](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/utils/smpl.py#L178-L238)

| 固定版本的输出 | 源时间如何恢复 | 根部朝向如何读取 |
| --- | --- | --- |
| GMR 单片段 PKL | SMPL-X 先按适配网格还原到源索引，导出从适配索引 1 开始。LAFAN1 BVH 保留从 0 开始的前缀。 | `root_rot` 为 `xyzw` |
| UMR 单片段 NPZ | 用 `frame_ids` 取实际源索引，再按源时间轴取值。 | `qpos` 按目标 MJCF 排列，自由基座为 XYZ 后接 `wxyz` |
| [FRDV BVH NPZ](#frdv-bvh) | 用伴随配置还原 `start`，从该源索引开始每帧记录，`fps` 取 BVH 头部帧间隔的倒数。 | `qpos` 的自由基座为 XYZ 后接 `wxyz` |
| HOI-Retarget 发布行 | 行内各数组共享 30 FPS 采样网格，只把 `n_frames / fps` 当数据列的时长。 | `root_rot` 与 `object_rot` 均为 `wxyz` |
{.table-readable}

下面的脚本输出首末源时刻和两套时钟的偏差。追加 `--csv` 可把每个输出索引、可能为小数的源索引及两套时间写到标准输出。GMR 分支用报告的 PKL 帧数和帧率，BVH 分支还读源文件的 `Frames` 和 `Frame Time`。UMR 分支用 NumPy 读取 `frame_ids`，例如 [Character 样例](#umr-character)的裁切结果从源动作 0.200 s 开始，首末样本跨度与导出帧率一致。FRDV 的用法见[同一 BVH 的复跑](#frdv-bvh)。 [下载 `audit_motion_time.py`](audit_motion_time.py)

```bash
python3 -B audit_motion_time.py gmr-smplx --source-frames 60 --source-fps 60 \
  --exported-frames 29 --export-fps 30 --csv
python3 -B audit_motion_time.py gmr-bvh example.bvh --exported-frames 59 --export-fps 30
python3 -B audit_motion_time.py umr character_cpu_smoke_g1.npz --source-fps 60
```

第二条命令对应[第 2.4 节](#gmr-bvh)裁成 60 帧的 BVH 输入。GMR 脚本按固定提交的非循环单片段入口还原采样约定。插值位置是小数时，先算出源时间，再查接触区间的起止时刻，避免把相邻整数帧的接触值直接取平均。

需要统一采样时，在两条序列共同覆盖的时间窗内做位置插值和四元数球面插值。四元数 $q$ 和 $-q$ 表示同一朝向。估计根部角速度，可将相邻的相对旋转 $R_t^{\mathsf T}R_{t+1}$ 转为旋转向量，再除以实际时间差。接触标签按开始、结束时刻恢复为离散状态，同时保留边界处的匹配容差。验证间隙、锚点和穿透时按同一个时间对齐后的定义计算。 [UMR 导出字段](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L2858-L2868) · [HOI-Retarget 数据复算]({{< relref "/posts/robotics/contact-preserving-retargeting#released-data" >}}) · [接触指标脚本]({{< relref "/posts/robotics/contact-preserving-retargeting#contact-metrics" >}})

## 7. 相关论文与公开实现 {#related-implementations}

| 用途 | 项目与可读源码 |
| --- | --- |
| 保留地形与物体交互 | **OmniRetarget** 用交互网格拟合人体、场景和物体的相对布局。 [论文](https://arxiv.org/abs/2509.26633v3) · [Holosoma 模块](https://github.com/amazon-far/holosoma/tree/main/src/holosoma_retargeting) |
| 保留带标签的接触 | **HOI-Retarget** 用 GMR 作初值，再在多个相邻帧里拟合物体坐标系下的接触点。 [论文](https://arxiv.org/abs/2609.34674v1) · [官方源码](https://github.com/leggedrobotics/hoi-retarget) |
| 在物理仿真中细化参考 | **SPIDER** 用采样优化控制，覆盖灵巧手与人形机器人。**GRASP** 用已有仿真 rollout 拟合局部梯度，引导下一轮采样。 [SPIDER 论文](https://arxiv.org/abs/2511.09484) · [SPIDER 源码](https://github.com/facebookresearch/spider) · [GRASP 论文](https://arxiv.org/abs/2611.05754) · [GRASP 源码](https://github.com/ComputationalRobotics/GRASP_code) |
| 在仿真中直接追人体关键点 | **DDR** 用几何重定向提供初值，在 MuJoCo 中用采样 MPC 跟踪人体关键点。论文分别控制了跟踪空间和参考来源，用同一个求解器做对照。 [论文 v2](https://arxiv.org/abs/2605.23762v2) · [作者项目页](https://gepetto.github.io/robot-motion/ddr/) |
| 直接推理机器人参考 | **NMR** 提供推理代码与权重，可将 AMASS 动作导出为 50 FPS 的 BeyondMimic NPZ。仓库把训练代码列为待发布。 [论文](https://arxiv.org/abs/2603.22201) · [官方仓库](https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy) |
| 批量运动学重定向 | **FRDV** 提供 LAFAN1 BVH 和 AMASS 到 G1 的运动学入口，导出带 `qpos` 和 `fps` 的 NPZ。**SOMA Retargeter** 用 Newton 和 NVIDIA Warp 把 SOMA 骨架的 BVH 转为机器人关节 CSV。 [FRDV 论文](https://arxiv.org/abs/2610.28470) · [FRDV 源码](https://github.com/DARoSLab/FRDV) · [SOMA 技术文章](https://developer.nvidia.com/blog/adapt-human-motion-for-robots-with-nvidia-soma-retargeter) · [SOMA 源码](https://github.com/NVIDIA/soma-retargeter) |
| 对照两种 SMPL 求解器 | **PBHC** 的 G1 脚本分别提供基于 Mink 的差分 IK 与来自 PHC 的梯度优化，仓库示例实验采用 Mink 路线。 [PBHC 的 SMPL 重定向入口](https://github.com/TeleHuman/PBHC/blob/ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0/smpl_retarget/README.md) |
| 调整新机器人或检查已有轨迹 | **Human-Humanoid Tools** 提供 URDF 校准、机器人到机器人转换、交互查看和批处理入口。**CoRe** 提供 SOMA 源动作的接触感知重定向和机器人 NPZ 导出。 [Human-Humanoid Tools 源码与中文说明](https://github.com/Roboparty/human-humanoid-tools) · [CoRe 源码](https://github.com/MayurHulke/core) |
| 接入跟踪与训练 | **ProtoMotions** 将 AMASS 预处理和机器人重定向接到 MotionLib 和运动跟踪训练。**GRAIL** 将重建后的 4D 人—物轨迹经 GMR 适配转成 G1 和物体轨迹，再交给跟踪器。 [ProtoMotions 数据入口](https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/amass_preparation.rst) · [GRAIL 论文](https://arxiv.org/abs/2606.05160) · [GRAIL 重定向源码说明](https://github.com/NVlabs/GRAIL/blob/main/docs/source/retargeting.md) |
| 灵巧手遥操作 | **AnyTeleop** 及相关 `dex-retargeting` 库适合读位置、向量和 DexPilot 风格的手部优化器。 [AnyTeleop 论文](https://www.roboticsproceedings.org/rss19/p015.html) · [优化器仓库](https://github.com/dexsuite/dex-retargeting) |
{.table-readable}

如果已有搬箱示教，可继续对照同一段动作中的手—箱、脚—地面和箱—桌关系，阅读交互网格、物体系锚点与跨帧约束。 [《人形机器人接触保持重定向》]({{< relref "/posts/robotics/contact-preserving-retargeting" >}}) · [四帧接触指标算例]({{< relref "/posts/robotics/contact-preserving-retargeting#contact-metrics" >}})

从视频获取人体和物体运动时，可以对照站内的 Video to Data 纸巾盒示例，查看重建结果怎样接入重定向和后续策略。需要细化到手内关节时，CARD 的接触目标一节展示了如何选择指腹接触位置，并把位置、方向和防滑写进目标。 [Video to Data 的重定向章节]({{< relref "/posts/ai/video-to-data#retargeting" >}}) · [CARD 的接触目标章节]({{< relref "/posts/ai/card-retargeting#objective" >}})

面向动画角色时，ReCHOIR 在按部位编码的动作表示上注入接触条件，并单独解码目标物体轨迹。对照 OTRetarget，可看到同样是联合生成角色与物体运动，前者学习带接触条件的解码器，后者逐帧解带限位的几何优化。 [ReCHOIR 论文](https://arxiv.org/abs/2609.10982v1) · [作者项目页](https://cherry-leki.github.io/projects/ReCHOIR/)

### 7.1 PriMoRetarget 怎样保住更高优先级的任务 {#task-priority}

GMR 的第二阶段仍可更新第一阶段用过的自由度。PriMoRetarget 把姿态、交互和接触目标放进优先级层次，在每一步把低优先级梯度投影到高优先级残差 Jacobian 的零空间。这里保留的是每一项高优先级残差的一阶值。方法也能联合优化同一段交接动作里的多台机器人，并把逐帧残差的投影拆成小系统。作者项目页目前提供论文、演示和方法说明。 [PriMoRetarget §III、§IV](https://arxiv.org/html/2610.12642v2#S3) · [作者项目页](https://nickioan.github.io/primoretarget-page/)

取一个两自由度例子。脚的水平位移 $x$ 应为零，另一个坐标 $y$ 可自由调整。姿态参考希望 $x=10$ cm、$y=6$ cm。若最小化 $9x^2+(x-10)^2+(y-6)^2$，最优解是 $(1,6)$ cm，脚滑了 1 cm。若先严格满足 $x=0$，再沿零空间调 $y$，解是 $(0,6)$ cm。此处高优先级 Jacobian 是 $J=[1\ \ 0]$，所以投影去除任何会改变 $x$ 的低优先级分量。

论文的实际更新用阻尼投影算子处理秩亏与病态系统。

$$
P_\lambda=I-J^{\mathsf T}(JJ^{\mathsf T}+\lambda I)^{-1}J.
$$

对这个单行 $J$，$JJ^{\mathsf T}=1$，所以更新保留低优先级梯度的 $\lambda/(1+\lambda)$ 倍水平分量。若 $\lambda=1/9$，原本会移动 10 cm 的一步仍会移动 1 cm。这个数值只描述一次更新。读求解结果时还要检查高优先级残差的实际变化，因为非线性线性化、加权预热和直接加到梯度上的平滑项都会影响最终轨迹。 [PriMoRetarget 式 (6)–(10) 与算法 1](https://arxiv.org/html/2610.12642v2#S3)

### 7.2 用 FRDV 复跑同一段 BVH {#frdv-bvh}

在 FRDV 固定提交 `72298bc` 安装命令行依赖后，可直接选仓库样例的前 60 帧。下面用一个 worker 顺序求解，模型由仓库按指定 MuJoCo Menagerie 提交读取。 [安装及命令](https://github.com/DARoSLab/FRDV/blob/72298bc095b2ed5fe0fe94933c795668e5bf88af/README.md#quick-start) · [模型版本选项](https://github.com/DARoSLab/FRDV/blob/72298bc095b2ed5fe0fe94933c795668e5bf88af/src/frdv/assets.py#L51-L75)

```bash
export FRDV_MENAGERIE_REF=eea4c7431b63da8278b0ff31638e0b0e941cec0e
uv run retarget robot=g1 dataset=lafan1 \
  data=assets/mocap/sample/walk3_subject5.bvh start=0 end=60 \
  workers=1 mode=sequential out=outputs/g1-walk60
```

这次 CPU 运行用 16 个追踪点，输出 `qpos` 为 `(60, 36)`。样本对应源索引 0–59，`fps` 是 BVH 样本间隔 0.033333 s 的倒数，约为 30.0003。前述 GMR BVH 保存索引 0–58，并写入命令传入的整数 30 FPS。FRDV 的 30 个 `joint_names` 包括一个占 7 个位置列的浮动基座关节和 29 个单轴关节。拼接或导入时要按模型关节名找列，并保留同目录的 `frdv_config.yaml` 以恢复裁切起点。 [切片与导出实现](https://github.com/DARoSLab/FRDV/blob/72298bc095b2ed5fe0fe94933c795668e5bf88af/src/frdv/cli/retarget.py#L55-L110) · [BVH 帧率](https://github.com/DARoSLab/FRDV/blob/72298bc095b2ed5fe0fe94933c795668e5bf88af/src/frdv/motion/bvh.py#L184-L203)

把时间检查脚本放到 FRDV 仓库根目录，传入源 BVH、导出的 NPZ 和配置中的帧范围。这个例子的末样本对应源动作 1.966647 s。 [下载 `audit_motion_time.py`](audit_motion_time.py)

```bash
python3 -B audit_motion_time.py frdv-bvh assets/mocap/sample/walk3_subject5.bvh \
  outputs/g1-walk60/walk3_subject5.npz --start 0 --end 60
```

论文另用接触隐式 iLQR 检查带执行器限位的 MuJoCo 跟踪，并给出浮脚和滑脚细化的例子。固定版仓库公开的是运动学管线。上面的复跑检验帧序和字段，评估跟踪需要另行记录控制器、接触模型和实际 rollout。 [FRDV 论文 §III-D、§IV-D](https://arxiv.org/html/2610.28470v1) · [固定版的发布范围](https://github.com/DARoSLab/FRDV/blob/72298bc095b2ed5fe0fe94933c795668e5bf88af/README.md#L173-L183)

## 阅读自测与验收 {#self-check}

- **GMR 的根部缩放和手腕缩放作用在哪里？** 根部系数作用于世界根轨迹，手腕系数作用于手腕相对根的位移，之后再叠加缩放后的根位置。
- **UMR 的网络逐帧预测机器人关节吗？** 网络学习标准姿态的有序表面点对应，机器人关节由运行时的约束优化求出。
- **UMR 外部接触式中的共同锚点有什么用？** 固定世界锚点在向量差里会抵消，但锚点和接触阈值决定活动点与权重。自接触还依赖会随构型移动的第二个身体点。
- **Sinkhorn 的均匀边缘是否保证硬对应一一匹配？** 均匀边缘约束软传输计划，每列单独取最大项后，多个目标点仍可选择同一个源点。
- **允许箱体移动后，软接触代价总能把间隙做到零吗？** 软接触与物体轨迹先验会折中。一维算例里，间隙从 0.15 m 减到 0.05 m，零间隙依赖约束和箱体可移动范围。
- **lucidrains 的 `joint_retarget` 什么时候执行优化？** 调用者传入 `solver_fn` 时才会委托求解，无回调的调用只评估目标函数。
- **87% 的 Jaccard 能怎样理解？** 它是该论文评估口径下，示教与输出的几何交互集合交并比的片段中位数。下游策略试验使用独立的成功判据。
- **GMR 开了速度限位后，还要检查写出帧率吗？** 要。一个输出帧内可以有多次受限积分，帧间差分要按实际输出时间计算。这个版本还会给 24 FPS 的输入保留适配帧数并返回 30 FPS，应单独核对原片段时长。
- **GMR 的两个单片段入口都少写一帧时，如何对齐接触标注？** 固定版本的 SMPL-X 入口从适配后索引 1 开始，BVH 入口从索引 0 开始并在保存前退出最后一帧。先记录入口、原时间和保留索引。
- **GMR 把 60 FPS 写成 30 FPS，源索引可以直接乘 2 吗？** 60 个输入样本的首尾索引是 0 和 59。适配后共有 30 个样本，索引步长是 59/29，SMPL-X 导出还会丢掉首个适配样本。
- **同名 UMR 缓存可以判断网格是否换过吗？** 名称与模式检查不包含几何一致性。换网格或标准姿态后，应重建采样、对应和轨迹。
- **UMR Character 改了采样范围，怎样只更新轨迹？** 使用 `--stage retarget --force-retarget` 重求并保留已训练的点对应。再检查导出的 `frame_ids` 和 `fps`。
- **把支撑脚的权重升高，就能让脚的位置保持不动吗？** 加权和允许姿态目标换取少量脚滑；应检查脚的位置残差。严格零空间投影可保持高优先级残差的一阶值，阻尼会允许少量变化。

## 论文与源码 {#sources}

- GMR。João Pedro Araújo、Yanjie Ze 等，*Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking*。 [论文 v1](https://arxiv.org/abs/2510.02252v1) · [代码 `25f56db`](https://github.com/YanjieZe/GMR/tree/25f56db79ac411258b377b6d4ccba0c8ff3372ac)
- UMR。Hanyang Cao 等，*Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence*。 [论文 v2](https://arxiv.org/abs/2609.02134v2) · [代码 `d1cdaed`](https://github.com/hanyang9/UMR/tree/d1cdaedaf67a827198da71c5fa831f48b2e51f91)
- OTRetarget。Guillaume Besset、Erwann Carn 等，*OTRetarget: Joint Robot and Object Motion Retargeting via Optimal Transport*。 [论文 v1](https://arxiv.org/abs/2609.36602v1) · [作者项目页](https://simple-robotics.github.io/publications/otretarget/) · [lucidrains 实现 `e6e87b8`](https://github.com/lucidrains/OTRetarget/tree/e6e87b85435439b2fa330f2fa0878e8c9dd81ca8)
