---
title: "人形机器人接触保持重定向：OmniRetarget 与 HOI-Retarget"
date: 2026-11-24
lastmod: 2026-11-24
draft: false
tags: ["Motion Retargeting", "Humanoid Robots", "Cross-Embodiment", "Trajectory Optimization", "Paper Notes"]
categories: ["机器人", "人工智能"]
authors: ["chase"]
summary: "沿搬箱动作对照 OmniRetarget 的交互网格和 HOI-Retarget 的物体系接触点，拆解窗口求解与接触指标，并用公开 G1、H2 轨迹复算掌点误差。"
description: "沿搬箱动作对照 OmniRetarget 的交互网格和 HOI-Retarget 的物体系接触点，拆解窗口求解与接触指标，并用公开 G1、H2 轨迹复算掌点误差。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
images: ["assets/contact-distance.webp"]
reading_prerequisites: "刚体坐标变换、机器人正运动学与最小二乘；正文从邻点平均和离散差分解释两种目标"
reading_focus: "用搬箱的接近、握持、放下三段观察方法分别保存哪些量，再按输入、约束、窗口状态和指标检查导出结果。"
related_posts:
  - "/posts/robotics/motion-retargeting"
  - "/posts/ai/video-to-data"
  - "/posts/ai/card-retargeting"
  - "/posts/ai/light-loco-parkour"
---

一段搬箱示教包含几个不同的要求。靠近箱子时，手臂和躯干应从合适的一侧绕过去。举起箱子时，双掌要跟住箱体上的抓持位置。放下箱子时，桌面限制箱子的落点，双脚还要支撑机器人。如果手只碰到了箱子另一侧，最近表面距离可能很小，原来的抓持却已经丢失。

**OmniRetarget 用交互网格保存身体、物体和地形共同形成的空间布局。HOI-Retarget 用物体坐标系下的接触点保存指定部位在指定时刻应接近哪里，再用窗口优化恢复这些接触。** 两者都先生成运动学参考。前一篇的 OTRetarget 还可以把可移动箱体的位姿加入同一帧的求解变量。 [GMR、UMR 与 OTRetarget]({{< relref "/posts/robotics/motion-retargeting" >}})

## 1. 用三个阶段读同一段搬箱示教 {#task}

| 阶段 | 需要保留的关系 | 应该检查的量 |
| --- | --- | --- |
| 接近 | 手臂与箱体的相对布局，掌面接近的方向 | 有无提前接触或穿透，双手能否按时到达 |
| 握持并搬运 | 对应的掌面接触位置，物体与身体的相对运动，支撑足 | 掌到指定锚点的距离，物体局部坐标下的滑移，脚的位移 |
| 放下与释放 | 箱—桌关系，接触结束时刻，手的撤离路径 | 箱底与桌面间隙，松手后的非法再接触与碰撞 |
{.table-readable}

这三个阶段的输入可以相同，即人体骨架或 SMPL-X 动作、箱体几何、逐帧物体位姿和场景几何。HOI-Retarget 额外需要逐部位的活动接触标注，数据适配器可以按几何阈值生成它。时间戳、地面高度和物体局部原点贯穿整个流程。其中任何一项出错，即使求解器收敛，也可能稳定地跟踪错误目标。 [OmniRetarget §III](https://arxiv.org/html/2509.26633v3#S3) · [HOI-Retarget §III](https://arxiv.org/html/2609.34674v1#S3)

手头已有发布轨迹，可以直接[复算一段掌点误差](#released-data)或[对齐两位参与者](#released-pair)。要解释为什么手在放下后仍贴着物体，可先逐帧查看[接触指标算例](#contact-metrics)。

## 2. OmniRetarget 把场景和身体放进同一个网格 {#omni}

### 2.1 拉普拉斯坐标描述点相对邻居的位置

每帧选出人体语义关键点，再加入物体和环境表面样本，做 Delaunay 四面体化。求解侧用语义对应的机器人点代替人体点，例如将人体手部点替换成机器人手部连杆上的点，并沿用源网格邻接来计算两侧的局部几何。对点 $p_i$，邻居集合记为 $\mathcal{N}(i)$，论文的等权拉普拉斯坐标为

$$
L_i=p_i-\frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} p_j.
$$

假设把一维手部点放在 0.30 m，它的两个邻居分别在 0.20 m 和 0.40 m，则 $L_i=0$，表示手处在两邻点的中心。三个点同时平移 0.10 m，$L_i$ 仍为零。若只移动手点而把两个邻点固定，$L_i$ 就相应改变。实际网格有许多相互关联的点，因此一只手偏离时，也会影响相邻网格点的关系。保留整体布局和精确命中某一块掌面接触区，仍需要分别验收。 [OmniRetarget 论文式 (1)、(2)](https://arxiv.org/html/2509.26633v3#S3.SS1)

平移会从差向量里消去，但旋转会旋转它的分量。针对机器人—物体交互，论文把网格建在物体坐标系中，让箱子转动时仍能保持同一局部目标。做场景增强时，方法先改变物体尺寸、初始位姿或地形参数，再重求机器人参考。论文的物体位姿增强把新的初始位姿扰动逐步衰减回原物体轨迹。为了避免整个人连同物体刚体平移就满足目标，求解还可把下肢或初始脚位锚定到原参考。 [OmniRetarget §III-B 与附录 C](https://arxiv.org/html/2509.26633v3)

### 2.2 每帧匹配网格，同时尊重支撑和碰撞约束

目标由源/目标拉普拉斯坐标的平方误差和相对上一帧的正则组成。机器人构型包含浮动基座与关节。论文把关节位置、速度、不穿透和支撑足固定写入约束，使用上一帧结果热启动当前帧，在局部迭代中线性化网格和距离并解二阶锥子问题。人脚是否支撑由示教脚的水平速度启发式判定。贴着地面主动滑动、缓慢抬脚或噪声导致的短暂停顿，需要额外检查该判定是否符合任务。 [论文 §III-A 与附录 D](https://arxiv.org/html/2509.26633v3)

公开 Holosoma 模块在 `retarget_motion()` 中逐帧构造源网格、取其邻接与拉普拉斯坐标，再用机器人模型迭代求解。该公开路径使用 MuJoCo 和 CVXPY/Clarabel。论文描述了 Drake 自动微分；阅读代码或跑实验时应记录实际后端、碰撞开关、足端锁定与容差。物体位姿传入求解内部作为已知值，导出的场景轨迹可随预处理后的物体运动回放。 [Holosoma 交互网格源码](https://github.com/amazon-far/holosoma/blob/c9882fcefe195d8ec0549bd44c2f2853f8084735/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py)

复算足端判定时要区分论文与代码的单位。论文给的是 1 cm/s；这个 Holosoma 提交的示例调用 `extract_foot_sticking_sequence_velocity()`，实际将相邻二维位置之差的模直接与 `0.01` 比较，首帧标为非支撑。若位置用米、序列为 30 FPS，`0.01` 表示每帧 1 cm，折算为 30 cm/s。换帧率时应报告所用阈值与时间差。例如每帧移动 5 mm，在 30 FPS 是 15 cm/s，仍会通过代码默认的位移阈值。 [示例调用](https://github.com/amazon-far/holosoma/blob/c9882fcefe195d8ec0549bd44c2f2853f8084735/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py#L694-L700) · [足端判定实现](https://github.com/amazon-far/holosoma/blob/c9882fcefe195d8ec0549bd44c2f2853f8084735/src/holosoma_retargeting/holosoma_retargeting/src/utils.py#L688-L718)

## 3. HOI-Retarget 用物体系接触点和时间窗口求解 {#hoi}

### 3.1 先计算“碰哪里”，再决定物体放到哪里

令箱位姿为 $(R_t,p_t)$。示教接触标签激活时，把人体对应部位的世界位置 $x_t^h$ 转到箱体局部，得到锚点 $a_t$。给定机器人阶段采用的物体位姿 $(R'_t,p'_t)$，以及局部网格的均匀缩放 $s$，机器人的世界目标可写成

$$
a_t=R_t^{\mathsf T}(x_t^h-p_t),\qquad
x_t^{\rm target}=p'_t+R'_t(s a_t).
$$

当箱子局部锚点为 $(0.20,0.10)$ m，箱的世界位置为 $(1.00,0.50)$ m，绕竖直轴转 90° 时，二维世界目标是 $(0.90,0.70)$ m。若物体原点与朝向保持，网格缩为 0.8 倍，目标移动到 $(0.92,0.66)$ m。这个算例只改变局部网格尺寸。实际 IK 阶段还会按配置处理根运动、物体路径和落地平滑。第二阶段把该物体轨迹当作固定参数，调整机器人去追指定的局部锚点。 [HOI-Retarget 论文式 (1) 与 §III-A](https://arxiv.org/html/2609.34674v1#S3.SS1) · [公开流程说明](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/README.md#method)

论文使用双掌与双足接触标签、足端支撑标签，并对活动接触加权。距离目标针对指定的接触位置，手部姿态另有相对参考的朝向代价。若标签只说明“手碰了箱子”，源端的标志点却总在掌心几何之外，后续就可能把机器人掌心拉向一个悬空点。对应该保持粘着的接触段，论文和工具提供以物体系的段均值替代逐帧锚点、再人工纠正的做法。滑动接触需要保留随时间变化的锚点。官方 `ObjectConfig` 的 `contact_points_per_frame=True` 保留逐帧目标，不能因为论文讨论了段平均就假设普通命令会自动应用它。 [论文 §III-B、III-C](https://arxiv.org/html/2609.34674v1#S3) · [公开配置](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/config.py)

### 3.2 窗口内把到达接触前的几帧一起优化

以 GMR 派生的 IK 轨迹为初值，HOI-Retarget 用 Pinocchio 描述机器人运动学，以 CasADi 建模并由 IPOPT 求每个相邻帧窗口。目标叠加关节与连杆跟踪、物体接触、支撑足高度与方向，以及三阶差分正则。公开实现对模型中有有限速度上限的关节约束相邻帧差，浮动基座的平移与转动速度约束另有开关，默认关闭。默认支撑足的高度与朝向属于加权项，论文附录还单列了可选的身体—物体碰撞代价实验。 [论文式 (2)、表 II、附录 B](https://arxiv.org/html/2609.34674v1) · [窗口约束源码](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/optimization/window.py#L141-L190)

设三阶后向差分为 $\delta^3 q_t=q_t-3q_{t-1}+3q_{t-2}-q_{t-3}$。等步长下，物理 jerk 的离散近似是 $\delta^3 q_t/\Delta t^3$。论文表 II 写的是加速度的相邻差 $\delta\ddot q_t$，固定提交的 `jerk_reg` 也使用 $\delta^3 q_t/\Delta t^2$，相当于近似 jerk 乘一个时间步。源码对整条 `q` 向量做差，基座旋转包含在四元数分量中。比较不同输出帧率时，报告这个正则、独立的物理 jerk 指标和采样率。 [正则实现](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/optimization/window.py#L181-L190) · [论文表 II、附录 A](https://arxiv.org/html/2609.34674v1)

以手沿水平方向靠近箱子为例。第 0 帧位置为 0 cm，第 3 帧首次要求到达 12 cm，每帧最多移动 4 cm。若逐帧方案在无接触的第 1、2 帧都沿常值姿态参考留在 0 cm，第 3 帧只能走到 4 cm，仍差 8 cm。让第 1 至 3 帧一起参与，一个恰好满足接触和位移上限的序列是 $0,4,8,12$ cm。实际窗口还要权衡姿态跟踪和接触软权重。这个算例说明提前几帧开始协调的作用，能否利用这些帧也取决于窗口边界固定了什么。

公开 `WindowConfig` 的默认值是 60 帧窗口、3 帧重叠、3 帧固定。首窗无需拼接，下一窗从索引 57 开始，固定已经写出的 57–59 帧再优化后面的轨迹。末段太短时，代码把尾窗向前拉长。例如总长 118 帧，普通窗口依次覆盖 $[0,60)$、$[57,117)$，余下窗口 $[114,118)$ 仅 4 帧，低于默认的 5 帧下限，于是改求 $[58,118)$。该尾窗固定 58–60 帧，可重算 61–117 帧。 [窗口默认配置](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/config.py#L125-L146) · [拼接与尾窗](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/optimization/window.py#L354-L411)

<figure class="article-figure" id="fig-window-overlap">
{{< post-image src="assets/window-overlap.webp" alt="118 帧轨迹的三个求解区间。首窗为 0 到 59，第二窗为 57 到 116，延长后的尾窗为 58 到 117。第二窗固定 57 到 59，尾窗固定 58 到 60。" >}}
<figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">斜线段沿用已有帧，实色段参与求解。尾部剩下 4 帧时，延长后的尾窗只固定其最前 3 帧，因此会重算此前已经求过的 61–116 帧，并补上最后一帧。</span></figcaption>
</figure>

累积缓冲区先装入 IK 初值。默认配置下，窗口失败时会保留写入前的值，可能是 IK 初值，也可能来自先前窗口。全部窗口都失败时，接触阶段抛出错误，公开 CLI 此时可能已经写出了第一阶段文件。用 CLI 跑时保存逐窗日志，直接调用库时还可读取求解对象的 `window_status`。如果在库配置里开启 `keep_last_iterate_on_failure`，代码可能用 IPOPT 失败前的最后一次迭代写回结果，状态带 `_last_iterate`，应按残差和约束重新验收。 [窗口失败处理](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/optimization/window.py#L320-L429) · [阶段产物](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/cli.py#L120-L140)

### 3.3 改变窗口能提前多少步 {#window-experiment}

继续用一维手位移模型。设目标距离为 $D$，每步位移上限为 $v$，从起始帧到接触共有 $n$ 个间隔。起始帧之后再固定 $p$ 个间隔在零位置，则最多可前移 $(n-p)v$。末帧可达的最远目标内位置为 $x_* = \min(D,(n-p)v)$。要把起步时间尽量推迟，在第 $i$ 步保持 $x_i=\max(0,x_*-(n-i)v)$，因为余下 $n-i$ 步最多前移 $(n-i)v$。这给出所有可行到达路径中逐帧位移最小的一条。

先把固定间隔从 0 改为 1，看看在同样的 4 cm 位移上限下还剩多少间隙。再把每步上限改为 6 cm，比较增加一个可调间隔与放宽每步约束的效果。算例只考虑一个位置自由度和末帧接触目标，真实方法还要同时拟合关节、足底和手—物接触。

{{< contact-window >}}

也可下载 Python 版本。`--check` 对小规模整数位置逐帧枚举，从每个终点的历史中找总位置平方和最小的路径，再和闭式算例对照。 [下载 `contact_window.py`](contact_window.py)

将同一个到达问题改成原版窗口的软接触代价，会留下一个小间隙。合成滑块仍以每步 4 cm 为上限，离接触还有三步，位置参考始终为零。取各帧跟踪权重 1、最后一帧接触权重 1,000。若末帧位置 $x$ 接近 0.12 m，前两步的最小偏移为 $x-0.08$ 和 $x-0.04$。代入跟踪和接触代价，解得末帧约为 11.976 cm，距目标 0.239 mm。用固定提交的窗口函数和 CasADi/IPOPT 对这段合成运动学求解，得到相同结果。 [窗口接触项](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/optimization/window.py#L274-L293)

## 4. 先跑单片段，再比较接触误差 {#reproduction}

### 4.1 OmniRetarget 的物体交互入口

在 Holosoma 的 `src/holosoma_retargeting/holosoma_retargeting` 目录，按 README 安装依赖并准备 InterMimic 处理后的 OMOMO `.pt` 文件与对应几何。先跑原始片段，增强模式可接着读取名义结果。 [官方数据和命令说明](https://github.com/amazon-far/holosoma/blob/c9882fcefe195d8ec0549bd44c2f2853f8084735/src/holosoma_retargeting/README.md)

```bash
python examples/robot_retarget.py \
  --data_path demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub3_largebox_003 \
  --data_format smplh
```

自定义输入应按 README 提供 `(T, J, 3)` 的世界关节位置，以及一致的 `demo_joints` 和 `joints_mapping`。物体数据从 `(qw,qx,qy,qz,x,y,z)` 重排为 MuJoCo 场景中的 `(x,y,z,qw,qx,qy,qz)`，动态物体分量随 `qpos` 一起导出。结果 NPZ 的 `fps` 由写入路径设为 30，接自有输入时要核对上游采样间隔，避免把写入的标签当成已经重采样的证明。 [位姿重排](https://github.com/amazon-far/holosoma/blob/c9882fcefe195d8ec0549bd44c2f2853f8084735/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py#L437-L444) · [导出逻辑](https://github.com/amazon-far/holosoma/blob/c9882fcefe195d8ec0549bd44c2f2853f8084735/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py#L534-L541)

### 4.2 HOI-Retarget 的两个产物

准备 README 要求的 SMPL-X 模型及 InterMimic/OMOMO 文件。以下使用相同的大箱片段名做工程对照。默认接触模式先写 `kinematic_window.pkl`，再写 `contact_window.pkl`。比较二者时应使用该次运行导出的实际物体轨迹及尺寸，并从日志读窗口状态。把模式改成 `kinematic` 可单独查看第一阶段。 [官方 Quickstart](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/README.md#quickstart) · [CLI 源码](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/cli.py)

```bash
hoi-retarget \
  --input_file data/InterMimic/OMOMO_new/sub3_largebox_003.pt \
  --out_dir outputs/sub3_largebox_003 \
  --no-record_video
```

命令关闭可选的视频输出，便于先验收运动数据。对原始尺寸物体，可设置 `--object_scale 1.0`。此时该版本还可以由 `--scale_object_trajectory` 单独控制位置轨迹缩放；其他几何缩放值会应用机器人适配的轨迹缩放。复现实验应同时记录两项，分别检查物体网格、位姿与接触锚点使用的尺度。 [CLI 缩放参数](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/hoi_retarget/cli.py#L282-L339)

若要比较两套默认管线之外的算法差异，固定机器人模型、帧率、物体尺寸、输入位姿和接触标签，分别记下各自的预处理改动。保留未求解片段的分母，再比较共同成功片段中的误差。查看器里可按接近、握持、放下分段，记录掌到指定锚点的距离、最近表面距离、穿透和支撑足位移。

可以用两份固定源码在本地核对本文的几个实现细节。下载脚本，传入论文与源码一节列出的对应 checkout 根目录。`--hoi` 用标准库复放窗口拼接、失败保留和三阶差分，`--holosoma` 还需 NumPy、SciPy、trimesh，用确定的点和位姿检查足端阈值、四面体邻接与坐标变换。脚本先校验所读文件的 SHA-256，再运行选取的片段。 [下载 `probe_contact_sources.py`](probe_contact_sources.py)

```bash
python3 -B probe_contact_sources.py --hoi /path/to/hoi-retarget
python3 -B probe_contact_sources.py --holosoma /path/to/holosoma
```

给 `--hoi` 命令加 `--check-hoi-solver`，可在 NumPy、CasADi 3.7.2 和其 IPOPT 求解器可用时复算合成滑块的软接触。这个分支用一维滑块位置替代机器人全身运动学。

### 4.3 用公开单行数据复算掌点误差 {#released-data}

只想检查发布结果时，可以先读取 Hugging Face 上的一行 JSON，省去 SMPL-X、视频解码和重新求解。数据卡列出 6,952 段源动作及 13,904 条机器人轨迹，每段分别映射到 G1 与 H2。由 `sequence_id` 关联同一采集事件，由 `clip_id` 标识具体的机器人结果；同一个 `object` 文本可能对应不同网格，几何调试时应按 `object_model_path` 查找资源。数据卡的 `qc_pass` 检查腕部持续逼近限位、相对源动作过度折身等情况。验收掌点仍需单独读接触标签并复算距离。 [官方数据卡](https://huggingface.co/datasets/leggedrobotics/hoi-retarget) · [发布字段说明 b5ac812](https://huggingface.co/datasets/leggedrobotics/hoi-retarget/blob/b5ac8125e497e6735d6314e63a0e1eedf6229336/docs/SCHEMA.md) · [作者的 3D 查看器](https://huggingface.co/spaces/leggedrobotics/hoi-retarget-viewer)

以下命令读取 OMOMO 子集的第 0 行。省略 `--urdf` 时，脚本用 Python 标准库核对帧数、数组、时间和接触掩码。带上 `--urdf` 需要 NumPy 与 Pinocchio，只读取机器人描述的运动学树。脚本按关节名字写入角度，把根四元数由数据的 `wxyz` 改排为自由基座的 `xyzw`，再从每个激活的连杆取正运动学位置。 [下载 `inspect_hoi_release.py`](inspect_hoi_release.py) · [G1 URDF](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/assets/robots/g1/g1_29dof_w_hands_capsule.urdf) · [H2 URDF](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/assets/robots/h2/h2_hoi.urdf)

```bash
viewer=https://datasets-server.huggingface.co/rows
data='dataset=leggedrobotics%2Fhoi-retarget&config=omomo&split=train'
curl -fsS "$viewer?$data&offset=0&length=1" -o g1-row.json
python3 -B inspect_hoi_release.py g1-row.json --urdf /path/to/g1.urdf
```

在 b5ac812 数据版本的 OMOMO `train` 切分中，索引 0 和 4,421 分别是 `sub10_clothesstand_000` 的 G1 和 H2。两行各有 131 帧，30 FPS，`duration_s` 约为 4.367 s，首末帧的时间差约为 4.333 s。两只机器人的右掌各有 100 帧接触标注，其余三个连杆无活动锚点。查看器把非活动锚点的 NaN 序列化为 `null`。统计各连杆误差时，用 `per_link_contact_flags` 取样，空集合保持空值。 [G1 原始行](https://datasets-server.huggingface.co/rows?dataset=leggedrobotics%2Fhoi-retarget&config=omomo&split=train&offset=0&length=1) · [H2 原始行](https://datasets-server.huggingface.co/rows?dataset=leggedrobotics%2Fhoi-retarget&config=omomo&split=train&offset=4421&length=1)

| 单片段复算项 | G1 | H2 |
| --- | ---: | ---: |
| 关节数 | 29 | 31 |
| 发布的物体几何缩放 | 0.83 | 1.00 |
| 右掌到指定锚点的均值（mm） | 0.97 | 1.69 |
| 右掌到指定锚点的最大值（mm） | 6.58 | 9.41 |
{.table-readable}

这批数据已经发布处理后的局部锚点。直接用 `object_pos`、`object_rot` 把锚点转到世界系后，Pinocchio 与独立的 MuJoCo 正运动学复算一致。给 G1 锚点再乘一次 0.83，均值会从 0.97 mm 变成 76.80 mm；沿用原尺寸参考列 `object_pos_realsize_ref` 作平移，误差会变成 188.78 mm。H2 保持原网格尺寸，仍使用已适配的物体位置轨迹。表中数值只描述这一段动作在发布输出上的运动学误差。 [缩放与列定义](https://huggingface.co/datasets/leggedrobotics/hoi-retarget/blob/b5ac8125e497e6735d6314e63a0e1eedf6229336/docs/SCHEMA.md)

同样的 100 帧还可检查掌点到物体网格的最近距离。给 G1 命令追加 `--object-mesh clothesstand.obj --mesh-scale 0.83`，H2 使用同一个原始网格和尺度 1。这个分支需要 trimesh 与 Rtree。衣帽架网格未闭合，这里度量无符号点面距离。G1 与 H2 的右掌到最近三角形均值分别是 7.20 和 8.80 mm，有 11 帧和 13 帧达到或超过 20 mm。对应的发布锚点本身也分别有 11 帧和 13 帧达到这一距离。脚本列出这些帧的索引，可在回放时同时核对掌点和标注锚点。 [衣帽架 URDF](https://github.com/leggedrobotics/hoi-retarget/blob/b341dadde3fc745ef5ea606eaacbde9a0cbb916f/assets/objects/clothesstand.urdf) · [发布网格 OBJ](https://huggingface.co/datasets/leggedrobotics/hoi-retarget/resolve/b5ac8125e497e6735d6314e63a0e1eedf6229336/assets/objects/objects/clothesstand/clothesstand.obj)

### 4.4 对齐 CoRoleHOI 的两条协作轨迹 {#released-pair}

同一 `sequence_id` 可同时包含同一事件的 G1 和 H2。先选同一种机器人，再用 `partner_clip_id` 核对两行相互指向。以下命令复算 `corolehoi/100_camera__0_191` 的 G1。将行号 0、1 分别改为 456、457，可复算同一事件的 H2。脚本读取两个完整的单行 JSON，核对模型、缩放与采样网格，全程只用 Python 标准库。 [协作字段说明](https://huggingface.co/datasets/leggedrobotics/hoi-retarget/blob/b5ac8125e497e6735d6314e63a0e1eedf6229336/docs/SCHEMA.md) · [下载 `inspect_hoi_release.py`](inspect_hoi_release.py)

```bash
viewer=https://datasets-server.huggingface.co/rows
data='dataset=leggedrobotics%2Fhoi-retarget&config=corolehoi&split=train'
curl -fsS "$viewer?$data&offset=0&length=1" -o actor-a.json
curl -fsS "$viewer?$data&offset=1&length=1" -o actor-b.json
python3 -B inspect_hoi_release.py actor-a.json --partner actor-b.json
```

将 B 移到 A 的坐标时，向 B 的世界 $z$ 加上 A 的 `partner_z_offset_cm / 100`。先比较只用发布高度差的残差，再从两行物体的首帧位置求一个三轴平移，把同一个平移用于其余 189 帧。两对数据的物体旋转都一致，比较时考虑了四元数的正负号。 [G1 原始行](https://datasets-server.huggingface.co/rows?dataset=leggedrobotics%2Fhoi-retarget&config=corolehoi&split=train&offset=0&length=2) · [H2 原始行](https://datasets-server.huggingface.co/rows?dataset=leggedrobotics%2Fhoi-retarget&config=corolehoi&split=train&offset=456&length=2)

| 190 帧的物体位置对齐 | G1 | H2 |
| --- | ---: | ---: |
| B 到 A 的首帧平移 $(x,y,z)$（mm） | (0.395, 2.265, 16.508) | (0.516, 2.959, 21.568) |
| 只加发布高度差的最大残差（mm） | 2.299 | 3.004 |
| 应用首帧三轴平移后的最大残差（µm） | 0.085 | 0.124 |
{.table-readable}

合并场景时，B 的机器人根位置和物体位置应一起平移。因为物体系掌点用的是二者的世界位置之差，平移同时加在两边会抵消，局部锚点保持原值。按活动标记用正运动学复算，两种机器人各有 254 个连杆样本符合这一关系。只移动物体会让物体系下的连杆点偏移约 16.7 mm（G1）或 21.8 mm（H2）。

发布的 `pair_composable` 按 3 cm 高度差标记。对固定版 CoRoleHOI 的两个 Parquet 分片，按 `sequence_id` 和机器人各计一次，G1 有 134/228 对标为真，H2 有 120/228 对标为真。有 14 个采集事件的 G1 配对标为真，H2 配对标为假。例如 `129_calibration_board__0_142` 的高度差分别约为 2.92 cm 和 3.81 cm。按机器人读取这个标记后，还需检查两位参与者实际共享的接触与支撑。 [字段定义](https://huggingface.co/datasets/leggedrobotics/hoi-retarget/blob/b5ac8125e497e6735d6314e63a0e1eedf6229336/docs/SCHEMA.md) · [固定版分片](https://huggingface.co/datasets/leggedrobotics/hoi-retarget/tree/b5ac8125e497e6735d6314e63a0e1eedf6229336/data/corolehoi)

## 5. OMOMO 数字测量的是哪一种接触 {#benchmark}

下表来自 **HOI-Retarget 论文表 III**。总输入是 4,421 条 OMOMO 片段。OmniRetarget 有 424 条未返回可行解，因此表中整段指标在三种方法共有的 3,997 条上取均值，接触相关指标使用其中有手—物接触的 3,604 条。该组禁用了 HOI-Retarget 的网格缩放，比较采用相同的物体几何。时间是单个 AMD EPYC 7H12 核的完整处理时间。 [表 III 与指标定义](https://arxiv.org/html/2609.34674v1#S4.SS1)

| 指标 | GMR | OmniRetarget | HOI-Retarget |
| --- | ---: | ---: | ---: |
| 掌到指定接触点距离（cm） | 36.8 | 18.3 | 0.5 |
| 身体姿态偏差（°） | 20.0 | 25.0 | 21.0 |
| 计算时间（s/片段） | 9.1 | 156.2 | 33.7 |

指定点距离能识别发生在同一物体表面上的接触位置偏差。若掌面沿平坦箱面的切向偏离目标 20 cm，最近表面距离可以为零，同时指定点距离就是 20 cm。 [HOI-Retarget 附录 A](https://arxiv.org/html/2609.34674v1)

<figure class="article-figure" id="fig-contact-distance">
{{< post-image src="assets/contact-distance.webp" alt="同一块平坦箱面上的两个点。掌点已贴到表面，最近表面距离为零，但离指定抓持锚点有 20 厘米。" >}}
<figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text">在构造的箱面截面里，目标锚点与掌点都在表面。最近表面距离为 0 cm，沿箱面的指定点偏移为 20 cm。</span></figcaption>
</figure>

### 5.1 同一段接触可以在不同指标上得出不同结论 {#contact-metrics}

OTRetarget 把每个时刻、身体部位和对象通道的组合视作一个样本。评估时用当帧可视网格的顶点查询精确有符号距离，每个部位取间隙最小的顶点，求解器内部另用采样探针。示教和输出各取有符号间隙小于 20 mm 的样本集合 $\mathcal H$、$\mathcal R$。穿透的间隙是负数，也会入选。Jaccard 是两集合的交集大小除以并集大小。深度误差则在示教近接集合 $\mathcal H$ 上计算 $|d-\max(\hat d,0)|$ 的平均，$d$ 为机器人间隙，$\hat d$ 为示教间隙。论文还报告两侧最近表面点的平均欧氏偏移，并对片段级指标取中位数。 [OTRetarget §IV-A](https://arxiv.org/html/2609.36602v1#S4.SS1)

例如只取一个掌点和一块平坦箱面。第 1 至 4 帧的示教间隙分别为 40、0、0、35 mm，握持在第 2、3 帧。下表的指定点误差也只在这两帧计算，锚点取箱面上的示教掌点。

| 重定向结果 | 输出间隙（mm） | Jaccard | 深度误差（mm） | 指定点误差（mm） |
| --- | --- | ---: | ---: | ---: |
| 两个握持帧沿箱面错位 200 mm | 40、0、0、35 | 100% | 0 | 200 |
| 握持点正确，松手帧仍贴着 | 40、0、0、0 | 2/3 | 0 | 0 |
| 握持帧从锚点垂直穿入 10 mm | 40、−10、−10、35 | 100% | 10 | 10 |
{.table-readable}

第二行输出多了一次近接，两个距离指标的取样范围不含它。第三行的 Jaccard 仍为 100%，需要看间隙和穿透量。对自己的动作，还可以把接触样本数、非活动帧的意外近接数和最大穿透单独保留，避免混淆接触时序与几何偏差。

可用标准库脚本复算上表。它把每帧的示教和输出探针各自转到物体的局部系，用局部 $x=0$ 平面计算有符号间隙与最近点。`anchors_local_m` 单独列出当帧有标注的探针及锚点；空集合的均值或交并比输出 `null`。输入应已按时刻配对，时间戳严格递增，统计按探针—帧计数，即使采样间隔不均也不按时长加权。这个简化几何适用于同一平坦箱面附近。若探针可能跑过边缘或到达另一面，应使用完整物体的表面距离与对应关系。 [下载 `contact_audit.py`](contact_audit.py)

```bash
python3 -B contact_audit.py
python3 -B contact_audit.py --example > contact-example.json
python3 -B contact_audit.py contact-example.json --threshold-mm 20
```

### 5.2 算力预算和下游评估各有分母 {#benchmark-subsets}

两篇论文的 OMOMO 输入都含 4,421 条片段，还要核对各自保留了哪些输出。HOI-Retarget 的表 III 在共同成功子集上用均值，手部误差取有标注的片段。OTRetarget 的表 III 是每种方法返回结果的片段中位数，OmniRetarget 的 424 条失败被排除在其汇总外，主表也保留各自的缩放规则。OTRetarget 另跑了固定物体、匹配 OmniRetarget 场景尺度的消融。做对照时可同时报告全部片段的输出覆盖率，以及统一几何、时间和样本条件后所得的距离与时序指标。 [HOI-Retarget §IV-A](https://arxiv.org/html/2609.34674v1#S4.SS1) · [OTRetarget §IV-A 与表 III](https://arxiv.org/html/2609.36602v1#S4)

时间窗口的取舍也有相应分母。HOI-Retarget 附录 B 给 130 条长片段各分配一个 CPU 核和 16 GB 内存。加碰撞代价时，窗口法完成 130 条，整段优化完成 87 条。其余 43 条提高到 64 GB 后可完成。表 VII 的耗时和内存等连续量在五种配置都完成的 87 条上取中位数，这与统计求解失败使用的 130 条不同。 [HOI-Retarget 附录 B](https://arxiv.org/html/2609.34674v1)

两篇论文都还讨论下游动态阶段。OmniRetarget 展示了用参考训练的 G1 实机策略。HOI-Retarget 表 IV 分别用 Isaac Lab 的 PPO 跟踪器和 DynaRetarget 的 MuJoCo 采样优化细化两种运动学参考。该表使用四个条件都完成的 68 条共同片段，接触指标使用其中 63 条。在 PPO 细化后，OmniRetarget 和 HOI-Retarget 参考的指定点误差分别为 9.6 和 8.6 cm。用 DynaRetarget 细化后的两项是 14.1 和 9.9 cm。若要比较细化前后的误差变化，要在同一组片段上分别测量运动学参考与仿真输出。 [OmniRetarget §IV、§V](https://arxiv.org/html/2509.26633v3) · [HOI-Retarget 表 IV](https://arxiv.org/html/2609.34674v1#S4.SS4) · [DynaRetarget 论文](https://arxiv.org/abs/2602.06827) · [SBTO 源码](https://github.com/Atarilab/sbto)

## 6. 接触位置正确之后还要检验什么 {#dynamics}

考虑一个静止夹持的理想例子。箱子质量为 1 kg，两只手分别在竖直侧面施加 15 N 的法向力，库仑摩擦系数取 0.25。两侧最多合计提供 $2\times0.25\times15=7.5$ N 的竖直摩擦力，小于约 9.81 N 的重力。即使两个掌点都准确落在目标锚点，单靠这两处接触仍托不住箱子。上举加速时还需要额外的竖直合力。这个上限只检验竖直力平衡，完整评估还要看力矩、接触朝向、支撑和执行器限位。

把两侧法向力都提高到 25 N，合计竖直摩擦上限变为 12.5 N。再看同高的两个掌点，横坐标分别为 −10 cm 和 10 cm，箱子重心向右偏 4 cm。假设只用掌面法向力和竖直摩擦力，静止时两侧的法向力相互抵消，竖直力还要让重力的力矩抵消。由总力 9.81 N 和力矩平衡可得左掌需 2.94 N、右掌需 6.87 N。右掌已超过单侧 6.25 N 的上限，即使合计余量仍然存在。要检查这种偏载，应在各接触点分别验算摩擦约束，再配平合力与合力矩。

需要进一步生成控制时，SPIDER 在物理仿真中对候选控制做采样优化，让运动学示教提供任务目标。GRASP 从保存的控制—代价样本拟合局部模型，用它的梯度提出新样本，再按仿真代价接受或拒绝。两者的仿真输出依赖设定的物体质量、摩擦和控制器。阅读对照实验时，先区分参考轨迹、仿真控制重放和真机反馈轨迹，再看成功率的判定。 [SPIDER 论文 v3](https://arxiv.org/abs/2511.09484v3) · [GRASP 论文 v1](https://arxiv.org/abs/2611.05754v1)

DynaRetarget 的官方 SBTO 实现可以接 OmniRetarget 发布的机器人与物体参考。它逐步延长优化时域，在仿真中用采样搜索控制序列。固定版本为仓库示例默认启用 `flip_quat_pos`，把浮动关节从“四元数在前”换成 MuJoCo 的“XYZ 在前”。导入已经按 MuJoCo 排列的参考时，应关闭这个交换。接入自己的轨迹还要核对时间。该版本读取 NPZ 时用 `int(fps)`，因此 300 个标为 29.97 FPS 的合成样本按 29 FPS 建时钟，末样本会比源时刻晚约 334 ms。复用参考前可检查读入时钟与源帧时间是否相符。 [DynaRetarget 论文](https://arxiv.org/abs/2602.06827) · [SBTO 仓库示例](https://github.com/Atarilab/sbto/blob/cbed4ede80d16657585f93b2158fffbb3de9c33e/README.md#loading-a-motion-reference) · [参考读取源码](https://github.com/Atarilab/sbto/blob/cbed4ede80d16657585f93b2158fffbb3de9c33e/sbto/utils/extract_ref.py#L22-L44)

### 6.1 换支撑时，同时验收接触位置和受力方向 {#support-transition}

把夹箱任务延长到「双手夹箱侧面 → 大腿承托 → 两掌从底面接住」，手掌可能始终离箱子很近。只按箱子路径或掌到表面的距离打分，很难看出支撑是在哪一刻交给大腿、又何时回到手掌。可以逐相位记录承担支撑的连杆、物体系锚点、向物体按压的方向，以及这对接触允许粘着还是沿表面滑动。侧面夹持主要靠法向夹力带来的切向摩擦，底部承托则可由朝上的法向分力承担重量。

ReContact 把这些量写成物体系接触计划。重建时用承托连杆相对物体的关系跨过遮挡，全身重定向时拟合计划中的接触点和方向，随后用残差策略奖励指定连杆在目标附近沿计划方向施力。论文的大腿辅助重抓实验让各控制器跟踪同一份参考，并要求两掌同时托住底面至少占末段的一半。$83.5\pm5.2\%$ 是该仿真任务中三次独立训练所得成功率的均值和样本标准差。 [ReContact §III-A 至 §III-D、§IV-A 至 §IV-B](https://arxiv.org/html/2611.02968v1)

第一视角视频中的手—物交互也可以沿接触传递。Ego2X 用物体表面锚点、相对力信息、抓持模式和交互状态表达接触。物体系约束既用来校正视频重建中的手部深度漂移，也为灵巧手或平行夹爪的 IK 指定接触目标。将它与全身方法一起读时，可以追踪同一批锚点是在哪一阶段估出、校正和使用的。 [Ego2X 论文 v2](https://arxiv.org/html/2611.24138v2)

PRISM 从少量真实视频生成物体形状和交互方式不同的示教，再重建、重定向并训练跟踪策略。公开仓库的 `main` 分支提供教师训练、rollout 收集和学生蒸馏，`sim2real` 分支提供部署入口。InterMimicGen 的增强从已有的人—物运动参考出发，改变物体布置并把仿真中成功执行的轨迹纳回训练数据。评估这类增强时，要分别统计生成了多少候选、多少通过几何检查，以及多少在相同动态条件下完成目标。 [PRISM 论文](https://arxiv.org/abs/2609.38172) · [PRISM 源码](https://github.com/amazon-far/PRISM) · [InterMimicGen 论文](https://arxiv.org/abs/2610.06850) · [InterMimicGen 项目页](https://sirui-xu.github.io/InterMimicGen/)

### 6.2 两台机器人怎样共享一个箱子 {#shared-object}

把两条单人重定向轨迹放在同一坐标系后，还要检查它们是如何握住同一个物体的。CoCarry 先分别用 GMR 初始化两台机器人，再在一条固定的物体位姿轨迹旁联合优化两台机器人的构型。上身匹配物体系内的手与身体布局，下身互相协调并保留相对地面的支撑。四个掌点各有局部锚点、法向和随时间变化的激活权重，因此可以表示交接过程中的单手松开或一方退出。验收时要同时检查各掌对同一物体的锚点误差、两机间相对位姿与支撑阶段。拼接公开的 CoRoleHOI 行可以复查时间和坐标一致性，CoCarry 的联合求解进一步在目标函数里处理两台机器人的相互关系。 [CoCarry §III–IV](https://arxiv.org/html/2610.32460v1) · [CoRoleHOI 行的对齐](#released-pair)

把两人示教的一方换成机器人时，接触的另一端可以是人体网格上的顶点。REACT 先用网格距离提取接触候选，再用视觉语言模型筛查接触的部位与时段，最后把机器人一侧的接触点转到 G1 表面。论文用人工标注的 80 段 Inter-X 片段检查标签。下游仿真改用筛查后的标签评估教师和学生策略，分别在 15 cm 和 30 cm 阈值下计算接触召回。要比较这两种策略的接触保持，应读附录 C.5 中固定动作与阈值的对照。所列仓库版本提供项目说明和演示。 [REACT §3–4、附录 C.5、D.1 与 F](https://arxiv.org/html/2610.28523v1) · [项目页](https://yutoshibata07.github.io/REACT/) · [仓库 b9ed176](https://github.com/YutoShibata07/REACT-HHI/tree/b9ed176b36e0bf1bc008470d0475727e54f5a408)
## 阅读自测与验收 {#self-check}

- **交互网格里的点一起平移，拉普拉斯坐标怎么变？** 坐标保持，因为点和邻点的均值增加相同平移。整体旋转会旋转世界系的分量，因此物体交互用物体系表达。
- **箱子缩成 0.8 倍，原来的抓持锚点怎么变？** 先相对物体原点缩放局部锚点，再用求解所用的箱位姿转回世界坐标。
- **HOI-Retarget 会在窗口求解时为够不到的手移动箱子吗？** 物体轨迹在预处理阶段得到，窗口只优化机器人构型。放松物体轨迹需要把物体位姿明确加入求解变量。
- **窗口重叠了三帧，就能提前调节这三帧吗？** 默认固定三帧并沿用上一窗口的输出，当前窗口只能调整未固定的帧。
- **一维实验把前三个间隔中的第一个固定在原点，末帧还能到 12 cm 吗？** 每步最多 4 cm 时，可调两步最多走 8 cm，因此末帧仍差 4 cm。
- **0.5 cm 这个数字用了哪些片段？** 它是 HOI-Retarget 论文表 III 的指定接触点距离，使用共同成功片段中含手—物接触的 3,604 条。
- **只看 `contact_window.pkl` 能确认整段每窗成功吗？** 部分窗失败时可能留用初值，判断每窗结果要看状态记录并对照导出轨迹。
- **足端代码直接与 0.01 比较，在 30 FPS 时是什么含义？** 相邻采样的平面位移用米计，0.01 m/帧对应 0.30 m/s。应把这一实现阈值和论文速度阈值分开记录。
- **每一帧的近接集合都对上了，可以确认没有穿透吗？** 不能。穿透样本也进入近接集合，需要查看有符号间隙。沿箱面错位时还需要指定锚点或表面点的偏移。
- **读公开 G1 数据时要把锚点再乘 0.83 吗？** 发布列已处于适配后的物体系，直接用该行的 `object_pos` 与 `object_rot` 将锚点转到世界系，并只统计掩码激活帧。
- **对齐两条 CoRoleHOI 轨迹时，只移动共享物体可以吗？** 在被移动的那行上给机器人根位置和物体位置加同一个世界平移，物体系内的掌点才保持原值。
- **大腿辅助重抓怎样验收两掌托底？** ReContact 的仿真试验要求两掌在末段至少一半时间同时接触箱底。仅完成整段动作是另一项指标。
- **两掌都在正确位置，就能抬起 1 kg 的箱子吗？** 两侧各 15 N、摩擦系数 0.25 时，理想竖直摩擦上限合计 7.5 N，小于重力。

## 论文、源码与数据 {#sources}

- OmniRetarget。Lujie Yang、Xiaoyu Huang、Zhen Wu 等。 [论文 v3](https://arxiv.org/abs/2509.26633v3) · [作者项目页](https://omniretarget.github.io/) · [Holosoma 官方实现 c9882fc](https://github.com/amazon-far/holosoma/tree/c9882fcefe195d8ec0549bd44c2f2853f8084735/src/holosoma_retargeting)
- HOI-Retarget。Jihwan Shin、Adrià López Escoriza 等。 [论文 v1](https://arxiv.org/abs/2609.34674v1) · [作者项目页](https://shinben0327.github.io/hoi-retarget/) · [官方实现 b341dad](https://github.com/leggedrobotics/hoi-retarget/tree/b341dadde3fc745ef5ea606eaacbde9a0cbb916f) · [重定向数据](https://huggingface.co/datasets/leggedrobotics/hoi-retarget)
- OMOMO 提供人体—物体动作，InterMimic 的处理版本供两个仓库的上述入口读取。 [OMOMO](https://github.com/lijiaman/omomo_release) · [InterMimic](https://github.com/Sirui-Xu/InterMimic)
