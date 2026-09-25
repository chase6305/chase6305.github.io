---
title: "机器人滤波方法详解：从 Butterworth、One Euro 到位姿平滑与状态估计"
date: 2026-09-25T03:14:26+08:00
lastmod: 2026-09-25T06:03:17+08:00
draft: false
tags: ["Robotics", "Signal Processing", "Butterworth", "One Euro Filter", "Mass Spring Damper", "Weighted Moving Average", "Pose Filter", "Kalman Filter", "Teleoperation"]
categories: ["机器人技术"]
authors: ["chase"]
summary: "系统比较机器人中的滑动平均、Butterworth、One Euro、弹簧阻尼、鲁棒滤波与状态估计，解释参数、延迟、位姿几何和工程接入，并提供可复现算例。"
description: "系统比较机器人中的滑动平均、Butterworth、One Euro、弹簧阻尼、鲁棒滤波与状态估计，解释参数、延迟、位姿几何和工程接入，并提供可复现算例。"
contentLanguage: "zh-CN"
math: true
toc: true
imageZoom: true
reading_prerequisites: "离散采样、向量和基本反馈控制；位姿部分需要了解旋转矩阵或四元数"
reading_focus: "按噪声类型、延迟预算和信号几何选择方法，区分测量滤波、状态估计、参考平滑与运动约束。"
related_posts:
  - "/posts/robotics/control/impedance-control"
  - "/posts/trajectory/ruckig"
  - "/posts/pid"
  - "/posts/ai/molmo-motion"
---

**机器人里没有一种滤波器能够同时消除所有噪声、保持零延迟、保留突变，并保证运动约束。** 选型的起点是弄清：这个信号来自哪里，什么变化是真实运动，什么变化是噪声，允许延迟多少，以及结果将进入哪条控制链。

手柄静止时末端抖动，可以考虑低通或 One Euro；编码器求导后速度噪声很大，需要合适的微分估计；关节存在窄带振动，可能应识别频率后设置陷波；视觉定位偶尔跳点，先要处理异常与重定位；轨迹不能超过速度、加速度和 jerk 限制，则需要轨迹生成器。它们都可能被写在 `filter` 工具目录中，却解决不同问题。

本文重点展开 **Butterworth、mass–spring–damper、One Euro／位姿扩展、weighted moving average**，再把常见的窗口方法、频域方法、鲁棒方法、状态估计和约束方法放进同一张地图。资料与公开实现核对截至 **2026-09-25**。随文曲线来自明确参数的合成信号，不代表某台机器人上的测量结果。

工程代码中的 `mass_spring_damper`、`one_euro_pose_filter`、`weighted_moving_filter` 往往只是实现名称；读代码时仍要确认它们分别采用哪种动态方程、旋转表示和权重顺序。

## 阅读路线

| 当前问题 | 建议先读 |
| --- | --- |
| 不清楚这些方法的区别 | [分类与选型](#map) |
| 想知道窗口和权重造成多少延迟 | [滑动平均与 EMA](#averages) |
| 想在线实现 Butterworth | [频域滤波与 SOS 状态](#frequency) |
| 遥操作慢时稳、快时跟手 | [One Euro](#one-euro) |
| 平滑的是位置＋四元数 | [位姿滤波](#pose) |
| 想让目标有弹簧阻尼手感 | [二阶目标滤波](#spring-damper) |
| IMU、里程计和视觉需要融合 | [状态估计](#estimation) |
| 需要检查参数、延迟与异常行为 | [接入清单](#integration)、[随文实验](#lab) |

## 1. 先区分四类工作 {#map}

**测量滤波**估计某个被噪声污染的信号，例如力传感器读数；**状态估计**结合模型和多源测量，推断位置、速度、姿态和偏置；**参考平滑**主动改变希望机器人跟随的目标；**轨迹约束**将运动限制在指定速度、加速度或 jerk 范围内。

同一个低通算法放在不同位置，含义不同。低通测得的关节角，是在改变控制器看到的反馈；低通遥操作目标，是在改变参考输入。两者都会影响闭环，但不能用相同的“看起来更顺”标准验收。

<figure class="article-figure" id="fig-filter-map">
  {{< post-image src="assets/filter-map.png" alt="测量经时间单位坐标检查、降噪和状态估计进入控制器；目标经参考平滑和运动约束进入同一个控制器" >}}
  <figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text"><strong>先确定滤波器位于哪条信号链。</strong>图中列出可选功能位置，并非要求每个机器人都串联所有模块；估计器也可能直接使用带噪测量。 <a href="assets/filter-map.png">查看原图</a></span></figcaption>
</figure>

### 1.1 方法总览

| 家族 | 典型方法 | 主要用途 | 主要代价或假设 |
| --- | --- | --- | --- |
| 窗口平均 | SMA、WMA、Gaussian FIR | 抑制高频随机抖动 | 窗口延迟、边界处理 |
| 递归低通 | EMA／一阶 RC | 常数内存的在线平滑 | 跟随滞后 |
| 局部拟合 | Savitzky–Golay、局部多项式微分 | 保留低阶趋势、估计导数 | 窗口位置与噪声放大 |
| 动态微分估计 | 线性／非线性跟踪微分器（TD） | 同时估计平滑信号和导数 | 带宽、初始峰值与离散稳定性 |
| IIR 频率整形 | Butterworth、Bessel、Chebyshev I／II、Elliptic | 按频带要求设计 | 相位、振铃、数值稳定性 |
| 窄带抑制 | Notch、谐波陷波、自适应陷波 | 已识别的周期干扰 | 频率漂移、邻近相位变化 |
| 其他频率选择 | 高通、带通、去趋势 | 去直流／漂移、提取振动 | 会移除真实低频信息 |
| 统计与自适应 | Wiener、LMS／NLMS、RLS | 有统计模型或参考信号时的噪声抑制 | 统计假设、收敛与参考污染 |
| 多尺度处理 | 小波阈值、子带降噪 | 离线振动、故障与瞬态分析 | 阈值、边界与块处理延迟 |
| 鲁棒窗口 | 中值、截尾均值、Hampel | 孤立尖峰、离群测量 | 真实突变可能被误删 |
| 速度自适应 | One Euro、增益调度低通 | 遥操作、视觉目标跟踪 | 参数依赖速度尺度和单位 |
| 二阶动态 | 质量–弹簧–阻尼 | 连续参考与可解释动态 | 超调、离散化、非硬约束 |
| 轻量预测校正 | α–β、α–β–γ | 位置与速度／加速度估计 | 运动模型偏差 |
| 多传感器融合 | 互补滤波、KF、EKF、ESKF、UKF | 定位、姿态与偏置估计 | 模型、噪声与可观测性 |
| 未知相关性融合 | 协方差交集（CI） | 融合相关性不明的状态估计 | 保守性、输入误差界是否可信 |
| 多假设估计 | 粒子滤波、IMM | 多峰定位、模式切换 | 计算与假设设计 |
| 姿态观测器 | Mahony、Madgwick | IMU 姿态融合 | 加速度、磁干扰假设 |
| 平滑与窗口优化 | RTS、固定延迟平滑、因子图、MHE | 轨迹重建与状态一致性 | 是否用后续观测、计算与输出时延 |
| 约束与整形 | 限速器、死区、滞回、jerk 限制、输入整形 | 指令整形和运动约束 | 不能都解释成降噪滤波 |

这里“尽可能多”不意味着把全部方法串起来。一个典型链路可能只需要时间检查、一个异常门限和一个低阶滤波器。每多加一层，都应回答它针对哪一种可观察的问题。

### 1.2 先记录信号契约

至少写清这些信息：单位、采样时刻、实际更新率、坐标系、有效性标记、重置事件和下游用途。例子是“相机每 33 ms 给一个基座系目标位置，单位米，有采集时间戳，丢失时有状态标记”，而不是只有一个三维数组。

如果控制器以 500 Hz 读取 30 Hz 相机的最新值，它大多数时候拿到的是旧测量。重复读取可以构造零阶保持输入，但不能增加独立观测，也不能让滤波器误认为得到了 500 Hz 的新视觉数据。

### 1.3 四种常见实现的差异

| 方法 | 每次更新依赖什么 | 主要参数 | 适合先问的问题 |
| --- | --- | --- | --- |
| Weighted moving filter | 最近 N 次输入及权重 | 点数、权重顺序、启动策略 | 愿意保留多长时间窗，允许多少平均样本年龄？ |
| Butterworth | 当前输入和递归状态 | 阶数、采样率、截止频率 | 有明确的通带、阻带和相位预算吗？ |
| One Euro／pose 扩展 | 当前值、历史滤波值、变化率与时间间隔 | 最低截止频率、速度增益、导数截止频率 | 是否需要静止强平滑、快速运动时提高跟随速度？ |
| Mass–spring–damper | 目标及位置／速度两个动态状态 | 自然频率、阻尼比，或虚拟 M／B／K | 希望参考呈现怎样的二阶响应，是否允许超调？ |

前三种通常被描述为信号处理，第四种常被描述为参考动态，但这些名称没有规定它们必须位于哪条控制通道。最终选择要回到信号用途；四者都不会自动提供碰撞约束或完整的运动可行性保证。

## 2. 为什么降噪经常伴随延迟 {#delay}

考虑离散测量 $x_k=s_k+n_k$：真实信号为 $s_k$，噪声为 $n_k$，滤波输出为 $y_k$。如果只知道当前与过去的样本，一个突然变化既可能是噪声，也可能是真实运动开始。抑制它与立即跟随它之间存在取舍。

对线性时不变滤波器，频率响应 $H(e^{j\omega})$ 同时包含幅值和相位。低通不仅降低高频幅值，还会改变不同频率成分的时间关系。群延迟定义为：

$$
\tau_g(\omega)=-\frac{d\angle H(e^{j\omega})}{d\omega}.
$$

若 $\omega$ 以 rad/sample 表示，结果单位是采样点，再乘 $\Delta t$ 才是秒；若对连续角频率 rad/s 求导，结果直接是秒。不能把 `group_delay` 输出的“5”直接写成 5 ms。

### 2.1 截止频率不是滤波器的“响应时间”

一阶连续低通的时间常数为 $\tau=1/(2\pi f_c)$。例如 $f_c=5$ Hz，$\tau\approx31.8$ ms；从零跟随阶跃达到 95% 大约需要 $3\tau\approx95.5$ ms。这个关系属于一阶模型，不能照搬到任意阶数、阻尼比或自适应滤波器。

闭环也关心相位。纯延迟 $T_d$ 在频率 $f$ 引入的相位为 $-360fT_d$ 度。若交越频率约 10 Hz，额外 20 ms 延迟对应约 $-72^\circ$ 的相位变化。这个算例解释为什么反馈通道中过强的平滑可能让系统更容易振荡；实际稳定性还要结合完整环路分析。

### 2.2 在线与离线是两种信息条件

在线滤波只能使用已经到达的数据。居中的滑动窗口、前后向 `filtfilt`、RTS 平滑等会使用后来的样本；离线轨迹可以因此更平滑，在线控制不能免费获得这些未来信息。

[SciPy 的 `sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html) 是前向再反向的滤波。理想化地看，组合幅频响应为 $|H|^2$，因此单次滤波的 −3 dB 频点会变成组合的约 −6 dB。它不仅是“原滤波器去掉延迟”，连幅值响应也变了；有限数据的端点还受延拓策略影响。

## 3. 滑动平均、加权平均与 EMA {#averages}

### 3.1 简单滑动平均：窗口越长，记忆越久

长度为 $N$ 的因果 SMA：

$$
y_k=\frac1N\sum_{i=0}^{N-1}x_{k-i}.
$$

对白噪声且样本互不相关的情形，输出噪声方差为输入的 $1/N$；若噪声高度相关，收益没有这么大。对低频线性变化，窗口的平均样本年龄是：

$$
T_{\mathrm{age}}=\frac{N-1}{2}\Delta t.
$$

200 Hz 下，十一点窗口的平均年龄为 25 ms；100 Hz 下，相同点数则是 50 ms。因此，“窗口取 11”脱离更新率没有完整含义。

SMA 的频率响应有旁瓣与零点，不能把它想成理想的砖墙低通。窗口还可能削弱刚好处在零点附近的真实周期运动。

### 3.2 加权滑动平均：先确定哪一个权重乘最新值

本文约定 $w_0$ 乘最新样本：

$$
y_k=\sum_{i=0}^{N-1}w_i x_{k-i},\qquad
\sum_iw_i=1.
$$

若权重非负，它是窗口内样本的凸组合。低频下的平均样本年龄与独立白噪声方差比分别是：

$$
\begin{aligned}
T_{\mathrm{age}}&=\Delta t\sum_i i w_i,\\
\frac{\sigma_y^2}{\sigma_x^2}&=\sum_iw_i^2.
\end{aligned}
$$

以 $[0.4,0.3,0.2,0.1]$ 为例，平均年龄为一个采样周期，方差比为 $0.30$。它比四点等权平均的 1.5 个周期更偏重当前值，但降噪能力也略弱于后者的 $0.25$ 方差比。这里的延迟是低频近似，非对称 FIR 的群延迟一般不是全频率常数。

公开 [Unitree weighted moving filter](https://github.com/unitreerobotics/xr_teleoperate/blob/7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6/teleop/utils/weighted_moving_filter.py) 使用 `np.convolve`。卷积会翻转核的配对顺序：当数据队列从旧到新排列时，这组权重的第一个值实际乘最新样本。把它改成普通 `dot(weights, queue)` 而不调整顺序，就会得到相反的偏重。

这个版本还具有两个明确的工程行为：队列未满时直接返回最新输入；完全相同的连续输入不进入队列。后者使窗口按“发生变化的样本”前进，可能改变等间隔 FIR 的时间语义。它可以是应用选择，但不能继续毫无条件地用固定 $\Delta t$ 的窗口公式解释所有运行段。

### 3.3 EMA：用一个状态保存过去

指数滑动平均只有一个递归状态，其中 $0<\alpha\le1$：

$$
y_k=\alpha x_k+(1-\alpha)y_{k-1}.
$$

$\alpha$ 越小，历史占比越大，输出通常越稳，跟随也越慢。白噪声条件下，稳态方差比为 $\alpha/(2-\alpha)$；低频延迟近似为 $(1-\alpha)\Delta t/\alpha$。

把连续时间常数离散化，常见的两种系数是：

$$
\begin{aligned}
\alpha_{\mathrm{BE}}&=\frac{\Delta t}{\tau+\Delta t},\\
\alpha_{\mathrm{exp}}&=1-e^{-\Delta t/\tau}.
\end{aligned}
$$

前者对应后向 Euler 形式，后者对应固定输入区间下的指数衰减。两者在小 $\Delta t/\tau$ 时接近，数值并不完全相同。论文、代码和调参记录应说明使用哪一种。

离散化之后，参数中的 $f_c$ 也未必恰好等于数字滤波器的 −3 dB 频率。本文按固定系数递归式计算频率响应：采样率 200 Hz、名义参数 5 Hz 时，后向 Euler 形式的实际 −3 dB 点约为 **4.652 Hz**，指数形式约为 **5.010 Hz**。当截止频率远低于采样率时差异较小；跨实现比较带宽时仍应计算实际响应。

变间隔输入应使用实际 $\Delta t$ 计算系数。固定 `alpha=0.1` 在 30 Hz 和 300 Hz 下，代表的物理平滑强度差别很大。

### 3.4 Gaussian FIR 与 Savitzky–Golay

Gaussian FIR 用高斯形状的窗口权重平滑。居中核通常涉及未来样本；要在线使用，可以延迟输出中心时刻的结果，或设计单边核。单边核不再具有对称 FIR 的线性相位性质。

Savitzky–Golay 在局部窗口拟合低阶多项式，再取某个位置的值或导数。它可以较好保留多项式趋势，但不意味着所有尖峰都应该保留，更不意味着天然抗异常值。

默认居中的 SG 不适合直接伪装成当前时刻的因果输出。在线版本可以在窗口末端求值，例如：

```python
import numpy as np
from scipy.signal import savgol_coeffs

dt = 0.01
weights = savgol_coeffs(7, 2, deriv=1, delta=dt,
                       pos=6, use="dot")
# samples 必须按旧 -> 新排列，共 7 个等间隔位置样本
samples = np.arange(7, dtype=float) * dt
velocity = weights @ samples
assert np.isclose(velocity, 1.0)
```

端点拟合用过去预测当前导数，通常比居中估计更容易放大噪声。`delta` 决定导数单位，位置为米且 `delta` 为秒时，一阶导数才是 m/s。[SciPy 的系数接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html)

### 3.5 跟踪微分器：同时输出平滑量和变化率

跟踪微分器（Tracking Differentiator，TD）维护一组动态状态：一个跟踪输入位置，另一个估计其变化率。它适合编码器位置求速度、参考轨迹生成速度前馈等问题；这些导数仍是估计值，不能仅因为来自内部状态就当作直接测量。

第 8 节的线性二阶弹簧阻尼模型也提供位置与速度状态，可以从这个角度理解。非线性 TD 则通过不同的误差反馈函数调节收敛和噪声响应。例如 Wang 与 Shirinzadeh 的[非线性微分器论文](https://arxiv.org/abs/1102.2599v2)组合了线性校正与连续幂函数项，并分析相应信号条件下的误差。这里不把某一变体的收敛结论推广到所有 TD。

选择时要比较导数噪声、启动峰值、反向滞后和离散步长。更快追踪通常要求更高带宽，也可能放大高频噪声；“带滤波的微分器”没有消除这一矛盾。若最终需要满足硬速度与加速度约束，仍应检查轨迹生成层的约束定义。

## 4. Butterworth 与其他频域滤波 {#frequency}

### 4.1 Butterworth：通带幅值平坦

$n$ 阶模拟低通 Butterworth 的幅值满足：

$$
|H(j\Omega)|^2=
\frac{1}{1+(\Omega/\Omega_c)^{2n}}.
$$

在 $\Omega_c$ 处，幅值是通带的 $1/\sqrt2$，即约 −3.01 dB。“平坦”指通带幅频特性，不表示相位为零，也不表示阶跃一定无超调。提高阶数可以让过渡带更陡，但同时改变相位、瞬态和数值敏感性。

[SciPy `butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) 在给出 `fs` 时，截止频率与 `fs` 使用同一单位。若采样频率是 200 Hz、截止是 5 Hz，写 `butter(2, 5, fs=200, output="sos")`，而不是把 5 当成归一化频率。

阶数也可以从明确的频带要求推导。假设希望 5 Hz 内衰减不超过 1 dB、20 Hz 起衰减至少 30 dB，可以先求满足幅频要求的最低阶数：

```python
from scipy.signal import buttord, butter

order, cutoff = buttord(wp=5, ws=20, gpass=1, gstop=30, fs=200)
sos = butter(order, cutoff, fs=200, output="sos")
```

这组设置得到三阶滤波器。求得的 −3 dB 截止频率不会等于通带边缘 5 Hz，因为此处要求的是 1 dB 通带损失。`buttord` 解决幅频规格，得到的相位与阶跃响应仍需单独检查；若延迟预算无法满足，应重新讨论频带需求或控制结构。[阶数选择文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html)

### 4.2 在线运行的重点是保留状态

下面的类按一维标量流运行，把第一条测量作为常值稳态初始化，避免从零状态产生无意义的启动拖尾。

```python
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

class OnlineButterworth:
    def __init__(self, sample_hz, cutoff_hz, order=2):
        if not 0 < cutoff_hz < sample_hz / 2:
            raise ValueError("cutoff must be between 0 and Nyquist")
        self.sos = butter(order, cutoff_hz, fs=sample_hz,
                          output="sos")
        self.state = None

    def update(self, value):
        if not np.isfinite(value):
            raise ValueError("nonfinite measurement")
        if self.state is None:
            self.state = sosfilt_zi(self.sos) * value
        y, self.state = sosfilt(self.sos, [value], zi=self.state)
        return float(y[0])
```

每次调用后都保留 `state`。若每个控制周期重新初始化，再处理一个样本，得到的就不是预期的流式滤波器。多轴处理也必须为每个通道保留独立状态，不能把六个关节当成同一时间序列连续喂入。

SOS 将高阶 IIR 分解为二阶节，通常比高阶分子／分母多项式形式更适合数值实现。它降低系数敏感性，但不会修复错误采样率、错误轴顺序或不断被重置的状态。

这个例子假定等间隔采样。面对明显变间隔数据，可以先按明确策略重采样，或采用连续时间状态空间模型的变步长离散化。每来一帧就随意重算高阶 IIR 系数并沿用旧状态，并不能自动保证动态一致。

#### 在 C++ 实现里怎样核对二阶系数

如果代码只给出五个系数 $b_0,b_1,b_2,a_1,a_2$，先确认它采用的符号约定。令分母首项 $a_0=1$，直接差分方程为：

$$
\begin{aligned}
y_k&=b_0x_k+b_1x_{k-1}+b_2x_{k-2}\\
&\quad-a_1y_{k-1}-a_2y_{k-2}.
\end{aligned}
$$

对单位静态增益的二阶数字低通 Butterworth，采样率为 $f_s$、期望数字截止频率为 $f_c$，双线性变换并预畸变后可写为：

$$
\begin{aligned}
c&=\tan(\pi f_c/f_s),\\
D&=1+\sqrt2c+c^2,\\
b_0&=c^2/D,\\
b_1&=2b_0,\qquad b_2=b_0,\\
a_1&=2(c^2-1)/D,\\
a_2&=(1-\sqrt2c+c^2)/D.
\end{aligned}
$$

这里要求 $0<f_c<f_s/2$。例如 $a_1$ 为负时，差分式中的 $-a_1y_{k-1}$ 对应一个正的乘数；若实现已经把负号吸收到系数中，就不能再减一次。随文脚本将该式与 SciPy 设计结果、截止处幅值和极点位置分别核对。

预畸变用于让指定数字频点落在预期位置，不是简单把模拟模型的 $2\pi f_c$ 原样代入任何离散化都能得到同一结果。SciPy 的底层 `bilinear` 明确不自动做预畸变；高层滤波器设计接口承担了相应设计过程。[双线性变换文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear.html)

### 4.3 Bessel、Chebyshev 和 Elliptic 怎样选择

| 方法 | 设计侧重点 | 需要注意 |
| --- | --- | --- |
| Butterworth | 通带幅值平坦 | 相位非线性，高阶会振铃 |
| Bessel／Thomson | 模拟原型低频群延迟较平坦 | 同阶幅值滚降较慢；数字化后性质近似保留 |
| Chebyshev I | 允许通带等波纹，换取更陡过渡 | 真实信号的通带幅值会起伏 |
| Chebyshev II | 允许阻带等波纹 | 与 I 型的频率参数含义不同 |
| Elliptic／Cauer | 通带与阻带都允许波纹 | 过渡陡，但相位与瞬态取舍更明显 |

Bessel 的 `norm="phase"`、`"delay"` 和 `"mag"` 不是同一种截止约定；跨滤波器比较时，不能只把相同的数字塞进 `Wn`，就认为得到相同 −3 dB 带宽。SciPy 还说明，其数字 Bessel 使用双线性变换，模拟原型的群延迟性质不会在所有数字频率下严格保持。[Bessel 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html)

<figure class="article-figure" id="fig-iir-frequency">
  {{< post-image src="assets/iir-frequency-delay.png" alt="四种四阶数字 IIR 的幅频响应与通带群延迟对照，相同设计频率下仍具有不同幅值和相位行为" >}}
  <figcaption><span class="article-figure__number">图 2</span><span class="article-figure__text"><strong>幅值取舍与延迟取舍要一起看。</strong>采样率 200 Hz、阶数 4，设计参数为 5 Hz；Bessel 使用 mag 归一化，Chebyshev I 与 Elliptic 通带波纹设 1 dB，Elliptic 阻带衰减设 40 dB。这不是相同有效带宽下的排名。 <a href="assets/iir-frequency-delay.png">查看原图</a></span></figcaption>
</figure>

### 4.4 Notch：只针对确定的窄带成分

陷波器适合抑制已经识别的电源干扰、机械共振或周期噪声。常用二阶陷波由中心频率 $f_0$ 与品质因数 $Q$ 描述，近似带宽关系为 $\mathrm{BW}=f_0/Q$。

```python
from scipy.signal import iirnotch, tf2sos

b, a = iirnotch(w0=50.0, Q=25.0, fs=500.0)
sos = tf2sos(b, a)
```

这里的示例带宽约 2 Hz。它不会普遍消除所有高频噪声，中心频率附近也会改变相位。若转速变化导致干扰频率漂移，需要跟踪频率或做增益调度，同时检查系数变化带来的瞬态。[`iirnotch` 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html)

高通可以去直流偏置，但也会去掉真实的静态力与缓慢运动；带通适合振动分析，却不能直接作为绝对位置的通用预处理。先确定哪些频带承载任务信息，再谈“过滤掉低频或高频”。

### 4.5 Wiener、LMS／RLS 与小波：还需要哪些额外信息

**Wiener 滤波**依据目标信号与噪声的统计关系，在给定线性估计器范围内最小化均方误差。实际使用时要说明统计量来自哪里、是否近似平稳，以及实现是否因果。“最小均方误差”属于这些假设下的性质，不表示对所有机器人信号都最优。

还要区分理论家族与具体库接口：[`scipy.signal.wiener`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html) 提供的是基于局部窗口统计的 N 维数组处理，类似图像局部 Wiener 滤波。函数能接收一维数组，不意味着它自动成为维护状态的实时传感器滤波器；居中窗口、边界与噪声功率估计方式仍需检查。

**LMS／NLMS** 根据输出误差在线调整 FIR 权重，NLMS 再利用输入能量归一化步长。与 One Euro 根据速度调节截止频率不同，它需要定义输入、期望信号与误差。步长决定收敛速度和稳态误差，过大可能不稳定。[LMS 接口与算法说明](https://www.mathworks.com/help/dsp/ref/dsp.lmsfilter-system-object.html)

例如，机器人底座上的参考加速度计与末端传感器中的结构振动相关，可以尝试用自适应滤波预测这部分干扰，再从测量中扣除。关键假设是参考足以描述干扰，同时没有把需要保留的真实操作信号也当成“噪声参考”。没有独立参考或可信目标时，不能让算法凭空知道该删掉哪部分。

**RLS** 递推求解带遗忘因子的加权最小二乘问题，通常比简单 LMS 更快适应相关输入，但矩阵更新、初始化和数值条件更复杂。遗忘因子接近 1 时记忆较长，减小时更关注近期样本；这不是 One Euro 的速度增益参数，也不是传感器噪声方差。[RLS 算法说明](https://www.mathworks.com/help/dsp/ref/dsp.rlsfilter-system-object.html)

**小波降噪** 把信号分解到不同时间尺度，再对系数进行阈值处理与重建。硬阈值直接删除小系数，软阈值还会收缩保留系数；它适合比较局部瞬态与不同尺度成分，但可能削弱真实冲击。整段分解重建通常属于离线或块处理流程；若放进在线系统，必须计算缓冲和边界带来的延迟。[PyWavelets 阈值定义](https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html)

## 5. 尖峰与坏数据：中值、Hampel 和门控 {#robust}

普通均值容易被一个极大值拉偏。中值滤波对窗口内少量孤立尖峰更稳健，但也会改变窄脉冲和快速台阶的形状。因果五点中值通常需要窗口中足够多的新水平样本到达后，输出才跳到新的水平。

截尾均值先移除排序两端的一部分值，再平均剩余样本；它介于均值与中值之间。Hampel 则以局部中值 $m$ 和中位绝对偏差估计尺度：

$$
\begin{aligned}
\mathrm{MAD}&=\operatorname{median}_i|x_i-m|,\\
\hat\sigma&\approx1.4826\,\mathrm{MAD}.
\end{aligned}
$$

若当前点偏离局部中值超过给定倍数的尺度，就将它标记为异常；随后可以选择替换、拒绝或转入降级逻辑。1.4826 是高斯尺度一致性系数，不表示数据已经变成高斯，也不保证统一的误报概率。

公开 [MATLAB Hampel 接口](https://www.mathworks.com/help/signal/ref/hampel.html) 默认使用两侧邻居。在线系统必须明确改成因果窗口，或承认输出带有等待未来样本的延迟。MAD 为零时也需要明确策略，例如结合传感器分辨率设定尺度下限，避免正常的小变化都被判为异常。

### 5.1 不能靠“平滑”修复一切

NaN、时间戳倒退、坐标系重置和视觉定位跳变，应作为事件处理。把 NaN 喂入递归 IIR，后续状态可能一直被污染；把全局重定位跳变当噪声慢慢平滑，可能长时间输出既不属于旧地图也不属于新地图的位置。

同样，力传感器上的尖峰可能是真实接触。删除它会改善曲线外观，却可能损失控制器最需要的信息。应保留原始流与异常标记，并针对应用确定平滑流、碰撞检测流是否需要不同处理。

## 6. One Euro：用运动速度调节平滑强度 {#one-euro}

One Euro（1€）主要希望同时改善两个体验：静止时少抖动，快速移动时少拖尾。它不是预测未来，而是根据运动变化调节一阶低通的截止频率。[作者页面与调参说明](https://gery.casiez.net/1euro/)

可以把它看成四步：估计变化率；低通变化率；根据其大小设置截止频率；用该频率滤波原信号。

<figure class="article-figure" id="fig-one-euro">
  {{< post-image src="assets/one-euro-adaptation.png" alt="One Euro 上支路估计并低通过滤变化率以选择截止频率，下支路用自适应增益平滑当前信号" >}}
  <figcaption><span class="article-figure__number">图 3</span><span class="article-figure__text"><strong>速度决定放开多少带宽，当前测量仍是被滤波的信号。</strong>两条支路共享时间间隔，并保留上一时刻的滤波状态；图中变化率采用前一滤波值作为参照。 <a href="assets/one-euro-adaptation.png">查看原图</a></span></figcaption>
</figure>

$$
\begin{aligned}
d_k&=\frac{x_k-\hat x_{k-1}}{\Delta t_k},\\
\hat d_k&=\alpha_d d_k+(1-\alpha_d)\hat d_{k-1},\\
f_{c,k}&=f_{\min}+\beta|\hat d_k|,\\
\hat x_k&=\alpha_kx_k+(1-\alpha_k)\hat x_{k-1}.
\end{aligned}
$$

这里使用[作者 Python 实现的固定版本](https://github.com/casiez/OneEuroFilter/blob/d78925584245597f2aa9c4c01a802eb0f0b77fb9/python/OneEuroFilter/OneEuroFilter.py)中的**前一滤波值**计算变化率；其他实现也有使用前一原始值的变体，数值行为会不同。$\alpha_d$ 由导数截止频率 $f_d$ 计算，$\alpha_k$ 由自适应的 $f_{c,k}$ 计算，采用 $\alpha=\Delta t/(\Delta t+1/(2\pi f_c))$。

### 6.1 三个参数分别管什么

| 参数 | 主要影响 | 调节方向 |
| --- | --- | --- |
| $f_{\min}$ | 慢速和静止时的平滑 | 降低通常更稳，也更迟缓 |
| $\beta$ | 速度上升时放开多少带宽 | 增大通常减少快速运动拖尾 |
| $f_d$ | 变化率估计的反应速度 | 太低可能反应慢，太高可能被噪声带动 |

先令 $\beta=0$，根据静止抖动和慢速跟随调整 $f_{\min}$；再做较快运动，逐步增加 $\beta$。最后检查停止、反向和异常跳点，因为“快时放开带宽”也可能让高速异常值更容易穿过。

这里的截止参数通过一阶离散系数起作用，具有第 3 节所述的名义频率与实际带宽差异。启用速度自适应后，系数又随输入改变，整个滤波器不再是固定的线性时不变系统；一张固定频响图或一个常数群延迟，不能完整描述它在整段运动中的行为。

### 6.2 单位换了，参数也要换

若 $x$ 使用米，则变化率单位为 m/s，$\beta$ 的单位可写为 Hz/(m/s)。把输入改成毫米，数值速度放大 1000 倍；若要保持相同截止频率变化，$\beta$ 应缩小 1000 倍。

对三维位置，可以逐轴设置独立增益，也可以用滤波速度向量的范数产生一个共享增益。后者在固定正交坐标系旋转下更容易保持一致；前者允许不同轴采用不同响应。两者是设计选择，不能只用一个 `OneEuro` 类名就假设行为相同。

### 6.3 时间戳和失联处理比多调一位小数更重要

应使用测量采集时间，而不是随意采用消息被处理时的时间。重复或倒退的时间戳不能进入除法；长时间失联后直接用巨大 $\Delta t$ 更新，也可能让输出突然贴近重新出现的目标。

合理策略取决于应用：短暂丢帧保持状态或用运动模型预测；长时间中断后重新初始化；重新接管遥操作时将参考与当前机器人状态对齐。One Euro 自身不会替你定义这些行为。

## 7. 位姿滤波：旋转不能当普通四维向量 {#pose}

`one_euro_pose_filter` 通常指把 One Euro 思路扩展到位姿的工程实现，并不是一个只有唯一公式的标准算法。阅读此类代码时，要检查平移、旋转、速度估计和插值规则，而不只是参数名称。

### 7.1 欧拉角越过边界时会出现伪跳变

绕同一轴从 $179^\circ$ 旋转到 $-179^\circ$，真实的最短变化只有 $2^\circ$；若直接对这两个数做平均，得到 $0^\circ$，反而偏离了接近半周。

单个连续关节角可以根据关节语义做 unwrap，但三维姿态还存在轴耦合与欧拉角奇异性。不能把对一个角度有效的处理，原样扩展成通用姿态滤波器。

### 7.2 四元数先统一符号，再沿球面插值

单位四元数 $q$ 与 $-q$ 表示相同旋转。若直接平均它们，可能得到零向量，归一化也无法修复。常见操作是在相邻四元数点积为负时翻转其中一个，使它们处于同一半球，再做最短弧插值。

用旋转矩阵表示，同一个更新可以写成：

$$
\begin{aligned}
r_k&=\operatorname{Log}(\hat R_{k-1}^{\top}R_k),\\
\hat R_k&=\hat R_{k-1}\operatorname{Exp}(\alpha_k r_k).
\end{aligned}
$$

这里的 Log／Exp 在旋转群与其切空间之间转换，$\alpha_k$ 控制沿最短旋转走多远。这个公式没有对矩阵九个元素分别平均，因此输出仍是合法旋转。

几何与误差状态的系统推导可参考 [Solà 的四元数与 ESKF 讲义](https://arxiv.org/abs/1711.02508)。实际代码还要统一四元数顺序：SciPy 默认是 `xyzw`，其他库可能采用 `wxyz`。

### 7.3 自适应姿态增益需要角速度尺度

一种明确的位姿扩展是：平移用米／秒的变化率，旋转用相邻姿态的测地角除以 $\Delta t$，得到 rad/s，再分别设定 $f_{\min}$、$\beta$ 和导数滤波参数。

如果计算的是相对于前一滤波姿态的旋转误差，它包含滤波滞后，严格说不是传感器直接测到的真实角速度；若使用前一原始姿态，则更接近差分角速度，但也更容易被测量噪声影响。应在文档中说明采用哪一种。

随文脚本的 `RotationOneEuro` 采用前一种：先计算相对滤波姿态的最短旋转角，用时间间隔换成角速度尺度，再低通这个**非负标量**以调节增益。它与“低通三维角速度向量后再取范数”并不等价，反向旋转时尤其如此。平移部分可以配三个标量 `OneEuro` 实例；这种逐轴选择与共享速度范数的各向同性方案也应分别记录。

这种旋转更新还有一个可检查的性质：对全部输入和初始姿态施加同一个固定参考系变换，得到的输出也应随之作相同变换，而不应改变实际平滑效果。脚本用耦合三轴旋转验证了这一点，分别检查世界参考系与物体坐标轴的固定旋转。该性质依赖此处使用角度范数决定标量增益；它不适用于任意逐轴增益，也不能直接推广到随时间运动的参考系。

不要把位置的三个数与旋转向量的三个数拼起来直接求六维范数：米与弧度没有天然可以相加的尺度。平移和旋转独立平滑是一种实用选择；若要联合处理 SE(3)，还需定义平移与旋转的度量权重、扰动方向和参考系。

<figure class="article-figure" id="fig-pose-pipeline">
  {{< post-image src="assets/pose-filter-pipeline.png" alt="同一时间戳下，位置经三个标量 One Euro 更新，姿态经相对旋转、角速度尺度与 SO3 插值更新；平移和旋转分别使用自己的参数" >}}
  <figcaption><span class="article-figure__number">图 4</span><span class="article-figure__text"><strong>同一个位姿，使用两种几何更新规则。</strong>图中对应本文的逐轴平移与旋转标量增益方案；旋转更新同时需要相对旋转向量和自适应增益。单位四元数自身无量纲，rad 表示旋转角与 Log 映射的尺度。 <a href="assets/pose-filter-pipeline.png">查看原图</a></span></figcaption>
</figure>

接近 $180^\circ$ 的旋转具有最短路径方向歧义。若采样间真实转角可能超过半周，单靠相邻姿态无法恢复完整转动，需要更高采样率、陀螺仪或连续运动模型。

<figure class="article-figure" id="fig-rotation-wrap">
  {{< post-image src="assets/rotation-wrap.png" alt="姿态经过正负 180 度边界时，直接平滑包裹角度产生大误差，旋转群上的低通沿短弧跟随" >}}
  <figcaption><span class="article-figure__number">图 5</span><span class="article-figure__text"><strong>几何错误可能远大于噪声误差。</strong>两条平滑路径使用相同增益；输入四元数还故意逐帧交替正负号，表示的旋转保持不变。此处固定 β=0，隔离姿态几何问题。 <a href="assets/rotation-wrap.png">查看原图</a></span></figcaption>
</figure>

## 8. Mass–spring–damper：把参考目标变成二阶动态 {#spring-damper}

质量–弹簧–阻尼滤波常用于让遥操作目标或规划目标具有连续的运动响应。这里分析一个明确的形式：输入为目标位置 $u$，输出为平滑位置 $y$，阻尼作用于输出速度，三个参数 $M,B,K$ 均为正数。

$$
M\ddot y+B\dot y+K(y-u)=0.
$$

$M$、$B$、$K$ 可以是为参考动态选择的虚拟参数，不必等于机器人真实质量、摩擦和刚度。将其归一化：

$$
\begin{aligned}
\omega_n&=\sqrt{K/M},\\
\zeta&=\frac{B}{2\sqrt{MK}},\\
\frac{Y(s)}{U(s)}
&=\frac{\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}.
\end{aligned}
$$

$\omega_n$ 是自然角频率，单位 rad/s；$f_n=\omega_n/(2\pi)$ 才是 Hz。$\zeta$ 是阻尼比：小于 1 时欠阻尼，等于 1 时临界阻尼，大于 1 时过阻尼。

调参考动态时，先选择 $f_n$ 与 $\zeta$ 往往更直观。例如取虚拟 $M=1$、$f_n=5$ Hz、$\zeta=1$，可得到 $K\approx986.96$、$B\approx62.83$。将三个参数同时乘以 2，归一化方程与输出完全不变；只改变 $M$，则自然频率和阻尼比都会变化。因此，不能脱离其他参数，仅凭“质量更大”判断滤波会变成怎样。

### 8.1 临界阻尼不等于 Butterworth

在上述**二阶、单位静态增益、无速度前馈**的模型下，$\zeta=1/\sqrt2$ 对应二阶 Butterworth；临界阻尼则是 $\zeta=1$。

临界阻尼的阶跃响应没有欠阻尼振荡，但它的 −3 dB 频率为：

$$
\omega_{3\mathrm{dB}}
=\omega_n\sqrt{\sqrt2-1}
\approx0.6436\,\omega_n.
$$

因此，把 `natural_frequency=5 Hz` 与 `Butterworth cutoff=5 Hz` 放在一起，不是在比较相同带宽的两个滤波器。若需要公平比较，应对齐 −3 dB 点、噪声方差或允许延迟中的某个明确标准。

### 8.2 阻尼比控制超调，频率控制快慢

给定相同 $\omega_n$，欠阻尼通常更容易出现超调与振铃；临界阻尼在无振荡的二阶响应中具有较快的衰减；进一步增大阻尼可能更迟缓。不同参数是否“更好”，取决于参考是否允许超调，以及控制器能否承受相应动态。

这里的标准阶跃比较假定系统从静止出发。**已有速度时，临界阻尼也可能越过新目标。** 例如遥操作中突然停止或反向，滤波器保存的速度状态仍需要时间衰减；不能把阻尼比设为 1 就当成任意初始状态下的无超调保证。

对于临界阻尼，从静止跟随幅度为 $\Delta u$ 的阶跃，连续模型的速度峰值是：

$$
v_{\max}=\frac{|\Delta u|\omega_n}{e}.
$$

这揭示一个关键限制：**同一组弹簧阻尼参数，输入跳变越大，输出峰值速度越大。** 它并没有固定的硬速度上限。目标阶跃还会让加速度发生跳变，所以也不能把“位置和速度连续”当成“jerk 已经受限”。

<figure class="article-figure" id="fig-spring-response">
  {{< post-image src="assets/spring-damper-response.png" alt="相同自然频率下，阻尼比改变二阶系统的阶跃超调、衰减速度和负 3 dB 截止位置" >}}
  <figcaption><span class="article-figure__number">图 6</span><span class="article-figure__text"><strong>自然频率与截止频率是不同参数。</strong>左图为自然频率 5 Hz、不同阻尼比下的零阶保持精确离散阶跃响应；右图为对应连续模型的归一化幅频响应。 <a href="assets/spring-damper-response.png">查看原图</a></span></figcaption>
</figure>

### 8.3 连续稳定不代表随便离散也稳定

一种直接写法是显式 Euler：先由当前状态算加速度，再用 $v_{k+1}=v_k+a_k\Delta t$、$y_{k+1}=y_k+v_k\Delta t$ 更新。但稳定的连续系统，使用过大的步长后也可能数值发散。

对欠阻尼二阶系统，显式 Euler 的稳定条件要求：

$$
\Delta t<\frac{2\zeta}{\omega_n}.
$$

这不是所有积分器通用的限制，而是上述显式 Euler 的结果；半隐式 Euler、双线性变换和精确离散化各有自己的动态性质。

若目标在一个采样区间内保持不变，可以使用零阶保持的精确离散化。定义 $z=[y,\dot y]^\top$：

$$
\begin{aligned}
\dot z&=Az+Gu,\\
A&=\begin{bmatrix}0&1\\-\omega_n^2&-2\zeta\omega_n\end{bmatrix},\\
G&=\begin{bmatrix}0\\\omega_n^2\end{bmatrix},\\
A_d&=e^{A\Delta t},\\
G_d&=\int_0^{\Delta t}e^{A\tau}G\,d\tau,\\
z_{k+1}&=A_dz_k+G_du_k.
\end{aligned}
$$

随文脚本通过增广矩阵指数计算这两个离散矩阵，并用临界阻尼的解析阶跃响应验证。时间语义固定为：$u_k$ 在 $[t_k,t_{k+1})$ 保持，状态前进到 $t_{k+1}$；不能无说明地把更新后的状态再标成 $t_k$。

### 8.4 它与阻抗、导纳的区别在哪里

这里输入是位置目标，输出是平滑参考。[阻抗控制](/posts/robotics/control/impedance-control/) 则规定机器人运动与交互力之间的动态关系；导纳控制常用外力作为输入，生成位移或速度参考。方程都可能出现 $M$、$B$、$K$，但输入、输出和闭环位置不同。

即使都叫弹簧阻尼平滑，若阻尼项改为 $B(\dot y-\dot u)$，就引入了目标速度前馈，传递函数的分子也会变成 $Bs+K$。本文前面的频率和超调分析不能原封不动地套到该变体。

## 9. 状态估计：用模型推断当前运动 {#estimation}

低通通常只处理观测值本身。状态估计则显式描述运动如何演化、传感器如何观察它，以及不确定性怎样变化。它能够利用模型预测补偿部分跟随滞后，但也引入模型错误的风险。

### 9.1 α–β 与 α–β–γ：轻量预测校正

以位置测量 $z_k$、位置估计 $\hat p$ 与速度估计 $\hat v$ 为例，常速度预测为：

$$
\begin{aligned}
\hat p_k^-&=\hat p_{k-1}+\hat v_{k-1}\Delta t,\\
r_k&=z_k-\hat p_k^-,\\
\hat p_k&=\hat p_k^-+\alpha r_k,\\
\hat v_k&=\hat v_{k-1}+\frac{\beta}{\Delta t}r_k.
\end{aligned}
$$

它比直接位置低通多了一条速度状态。α–β–γ 再引入加速度状态，适合希望以较低计算量追踪运动目标的场景。但固定增益依赖采样率与运动假设，转弯、加速和离群点仍可能让预测变差；这里的 $\beta$ 也不是 One Euro 的同名参数。

### 9.2 互补滤波：让不同传感器负责不同频段

简单单轴姿态互补滤波可以写成：

$$
\begin{aligned}
\hat\theta_k^-&=\hat\theta_{k-1}+\omega_k\Delta t,\\
\hat\theta_k&=\lambda\hat\theta_k^-
+(1-\lambda)\theta_{\mathrm{acc},k}.
\end{aligned}
$$

陀螺仪提供短时间变化，重力方向提供低频校正。这里 $\lambda$ 乘的是预测项，与前文 EMA 中乘新测量的 $\alpha$ 约定不同。

加速度计测到的是比力，机器人快速平移时不能把它完全当重力方向。只有陀螺仪和加速度计的六轴 IMU，也无法凭空观测绝对航向；要抑制相应航向漂移，需要磁场、视觉或其他外部参考。

Mahony 与 Madgwick 都属于常见的姿态融合方法：前者利用姿态误差反馈构造非线性互补观测器，后者利用方向观测误差的梯度修正姿态估计。它们需要正确的 IMU 坐标、时间与传感器模型，不是对四元数四个分量套同一个低通。[Mahony 等人的论文](https://researchportalplus.anu.edu.au/en/publications/nonlinear-complementary-filters-on-the-special-orthogonal-group/)、[Madgwick 原始报告](https://x-io.co.uk/downloads/madgwick_internal_report.pdf)

### 9.3 Kalman Filter：增益来自不确定性

线性 Kalman Filter 的模型为：

$$
\begin{aligned}
x_k&=F_kx_{k-1}+w_k,\\
z_k&=H_kx_k+v_k.
\end{aligned}
$$

其中 $Q$ 描述过程噪声协方差，$R$ 描述测量噪声协方差。预测传播状态与协方差，测量更新再根据两者的不确定性分配权重。

通常，增大 $R$ 会降低对该测量的信任，增大 $Q$ 会增加对运动模型的怀疑。但“把 Q 调大就一定更平滑”是错误理解；它常常使更新更愿意跟随测量。滤波效果还取决于模型是否包含速度、偏置等状态。

对一维常速度模型，$F=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}$。过程噪声建模还要区分两种常见假设：

| 噪声假设 | 离散过程噪声协方差 |
| --- | --- |
| 每个区间内随机常值加速度，方差 $\sigma_a^2$ | $Q=\sigma_a^2 GG^\top$，$G=[\Delta t^2/2,\Delta t]^\top$ |
| 连续白加速度，谱密度 $q_a$ | $Q=q_a\begin{bmatrix}\Delta t^3/3&\Delta t^2/2\\\Delta t^2/2&\Delta t\end{bmatrix}$ |

两者参数的单位和采样间隔幂次不同，不能复制一个 Q 矩阵后只换采样率而不检查假设。

单位改变也会改变协方差。位置观测从米换成毫米，位置测量方差应乘 $10^6$；更一般地，状态改写为 $x'=Dx$ 时，协方差变为 $P'=DPD^\top$，过程模型和观测模型也要同步变换。保持数字不变、只改变量后缀，会改变估计器的实际信任关系。

多传感器融合还要检查测量间的相关性。例如同一个视觉里程计同时输出位置与由相同图像推算的速度，把它们无说明地当成两份独立证据，可能让协方差过度乐观。问题出在信息模型，事后再平滑均值无法补回正确的不确定性。

**相关性未知时，还可以考虑协方差交集（Covariance Intersection，CI）。** 它对估计的信息矩阵，也就是正定协方差的逆，做带权凸组合，再构造融合均值和保守协方差，适合存在共享历史信息的分布式估计。它是一种融合规则，前提是各输入本身具有可信的误差界，并且表示同一状态。[Julier 与 Uhlmann：Using covariance intersection for SLAM](https://doi.org/10.1016/j.robot.2006.06.011)

一个标量例子即可看清目的：同一估计被复制两份，每份方差均为 1，实际信息并未增加；若错误地假设独立，融合方差会变成 0.5。CI 对这两个相同的单位信息量做凸组合，得到的方差仍为 1。代价是可能较保守；它也不能修复错误坐标、时间错位或本来就低估的输入不确定性。

### 9.4 EKF、ESKF 与 UKF

| 方法 | 处理非线性的方式 | 机器人中的典型用途 |
| --- | --- | --- |
| EKF | 在当前估计处线性化 | 轮式里程计与多源定位 |
| ESKF | 名义状态保留几何结构，误差用局部小量表示 | IMU／视觉／GNSS 融合 |
| UKF | 用 sigma points 传播均值与协方差近似 | 难以方便求雅可比的非线性估计 |

ESKF 是误差状态形式的扩展 Kalman 思路，不是给 EKF 再加一个普通低通。姿态误差通常用三维小旋转描述，名义姿态保持为单位四元数或旋转矩阵。

UKF 避免显式推导部分雅可比，但并不保证在每个问题上更准、更稳；sigma point 的参数、噪声模型和状态几何同样重要。可参考 [robot_localization 的实际节点与参数说明](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst)，将输出频率、传感器超时与真实测量更新区分开。

### 9.5 粒子滤波、IMM 与鲁棒门控

粒子滤波用带权样本表示状态分布，适合保留“机器人可能在两个相似走廊中”的多峰假设；直接采样高维状态会显著增加样本需求，重采样也可能导致粒子贫化。可分解的状态结构或 Rao–Blackwellization 能减轻部分负担，但需要利用问题结构。[Thrun：Particle Filters in Robotics](https://arxiv.org/abs/1301.0607)

IMM 同时维护多个运动模型，例如匀速、转弯或加速，再估计模式概率。它适合模型切换问题，但模型集合与转换概率仍需设计，不会自动覆盖所有未知行为。

卡尔曼类估计器可以用创新 $r$ 与创新协方差 $S$ 计算马氏距离 $r^\top S^{-1}r$ 做门控。前提是残差维度、协方差和统计假设合理。把 $R$ 调得极大只是降低测量权重，不能替代坏数据识别；把所有大残差拒绝，又可能拒绝真正的重定位或快速机动。

### 9.6 平滑器与滑动窗口优化

RTS 平滑在前向 Kalman 结果之后做后向处理，利用后续测量修正更早状态。固定延迟平滑则在有限窗口内，以可控输出滞后换取更一致的历史估计。

因子图或 MHE（移动时域估计）把一段状态和观测放在窗口中优化，可以加入约束与鲁棒损失。它们**不必都读取未来数据**：以过去窗口估计当前状态仍可因果运行；若输出窗口中更早的状态，就利用了相对于该状态的未来观测。要分别记录信息条件和计算耗时。

### 9.7 迟到的测量，不能直接改成当前时间

假设状态估计已经推进到 1.000 s，一条视觉结果此时才到达，但图像采集于 0.920 s。它是关于过去状态的观测，不能只把时间戳改成 1.000 s 再做普通更新。以 0.5 m/s 匀速运动为例，80 ms 对应 4 cm 位移；这类时间误差不是随机抖动，增加低通强度无法消除它。

一种处理方式是保存有限历史：回到这条观测之前的状态，按采集时间插入更新，再重放后续测量与状态传播。`robot_localization` 的 `smooth_lagged_data` 提供这一类回退处理，`history_length` 决定保存多久；历史长度至少要覆盖需要接纳的迟到范围。[固定版本的延迟观测说明](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst)

这与等待未来观测来改善过去状态的离线平滑不同：回退重算仍只利用当前已经到达的信息，代价是历史存储、重放计算与输出修正。超出历史范围的测量需要另定策略；对当前状态的预测也必须携带相应不确定性，不能把改时间戳当成运动预测。

## 10. 有些“过滤器”其实在整形命令 {#constraints}

### 10.1 限速器、死区与滞回

一阶限速器可写为：

$$
\begin{aligned}
e_k&=u_k-y_{k-1},\\
b_k&=v_{\max}\Delta t,\\
y_k&=y_{k-1}+\operatorname{clip}(e_k,-b_k,b_k).
\end{aligned}
$$

它限制离散位置增量，但不自动限制速度变化或 jerk。死区可以忽略很小的摇杆输入；滞回使用不同的进入与退出阈值，减少状态在边界反复切换。它们是非线性规则，不能用一个固定截止频率完整描述。

还要决定限速器接在什么位置。先限制增量再平滑，与先平滑再限制增量，遇到饱和和反向时可能得到不同轨迹。实现需要检查最终输出，而不只是中间某一层看起来符合限制。

### 10.2 Jerk 受限的轨迹生成

若需求明确规定速度、加速度和 jerk 上限，可使用 [Ruckig](/posts/trajectory/ruckig/) 这类在线轨迹生成器。它根据当前状态、目标状态与运动约束计算参考，不是用“更强的低通”代替约束。

状态到状态规划与追踪连续变化的目标也不同。每个周期把移动目标作为新的终点，可能产生持续滞后；具体 tracking 功能和版本应按官方接口核对。[Ruckig 文档](https://docs.ruckig.com/tutorial.html)

### 10.3 输入整形与扰动观测器

输入整形利用系统振动模式设计若干延迟脉冲及权重，使命令激发的残余振动相互抵消，常见有 ZV、ZVD。它更像有模型依据的前馈命令设计，代价是命令延迟与对模态频率的敏感性；反馈通道中的陷波器则处理另一种位置的问题。[MIT 输入整形研究](https://dspace.mit.edu/entities/publication/e50f129d-2085-48af-ab21-27fcf10951e0)

扰动观测器或动量观测器利用动力学与测量估计外部扰动、摩擦或接触残差。它们内部常有低通与带宽参数，但整体功能是估计未建模输入，不能与“把力信号平均一下”画等号。模型误差与摩擦失配也会进入残差，不能把每个非零输出都解释成外部碰撞。[机器人碰撞检测与观测器综述](https://www.diag.uniroma1.it/~labrob/pub/papers/TRO_Collision_Dec2017.pdf)

## 11. 空间滤波：点云与深度图的另一条轴 {#spatial}

前文主要沿时间处理信号。机器人感知里还有沿空间邻域处理的过滤器，它们的窗口单位可能是像素或米，而不是毫秒。

| 方法 | 做什么 | 常见误用 |
| --- | --- | --- |
| PassThrough／ROI | 保留指定空间或数值范围 | 裁掉真实目标后仍当完整场景 |
| [VoxelGrid](https://pointclouds.org/documentation/tutorials/voxel_grid.html) | 用体素内点的质心代表这些点，减少点数 | 体素过大抹掉细杆、棱边和接触几何 |
| [Radius Outlier Removal](https://pointclouds.org/documentation/tutorials/remove_outliers.html) | 根据半径内邻居数删除孤立点 | 远处稀疏表面也可能被误删 |
| Statistical Outlier Removal | 用邻域距离统计识别稀疏离群点 | 单一全局阈值不适应密度变化 |
| 双边滤波 | 同时考虑空间邻近与数值相似，保留部分边缘 | 参数过大仍会跨边缘混合 |
| [MLS／局部曲面拟合](https://pointclouds.org/documentation/tutorials/resampling.html) | 用局部几何拟合平滑表面或法向 | 细节、尖角可能被过度拟合 |

例如 [PCL 的统计离群点滤波](https://pointclouds.org/documentation/tutorials/statistical_outlier.html) 根据点与邻居的距离分布筛选，而不是追踪某个点随时间的速度。时序滤波与空间滤波可以组合，但不能将“每帧第 100 个点”视为同一物理点，除非已有可靠的数据关联。

激光雷达运动去畸变也不能简单归入平滑：它使用各点的采样时刻与运动估计，把一帧扫描变换到共同时间参考。时间对齐错误可能表现成模糊或重影，增加空间滤波强度并不能从根本上修复。

## 12. 接入机器人前应怎样检查 {#integration}

### 12.1 从一段原始日志开始

记录原始测量、采集时间、到达时间、有效性、参考命令、滤波输出和机器人实际反馈。只保存滤波后的曲线，会让很多错误无法追溯。

先观察静止噪声、慢速运动、快速运动、反向、接触与丢帧，再决定滤波器家族。频谱中出现窄峰，并不必然是噪声：它也可能来自真实的周期任务；仍需结合动作与硬件状态解释。

对 IMU，还要区分**单次测量标准差、连续噪声密度和偏置随机游走**。以 Kalibr 的模型约定为例，采样间隔为 $\Delta t$ 时，白测量噪声的离散标准差为 $\sigma/\sqrt{\Delta t}$，偏置随机游走每步增量的标准差则为 $\sigma_b\sqrt{\Delta t}$。两者随时间间隔变化的方向不同，噪声密度也不能直接填进要求方差的 R 矩阵。

上述换算包含带宽与抗混叠假设，直接抽掉部分原始样本不满足同样的条件。静止记录的 Allan deviation 可以帮助识别不同时间尺度的噪声；在双对数坐标中，白噪声与随机游走的典型斜率分别为 −1/2、+1/2。但静止、恒温下得到的参数仍不能完整描述运动误差与温漂。[Kalibr IMU 噪声模型](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model)

### 12.2 抗混叠必须发生在降采样之前

100 Hz 采样时，70 Hz 与 30 Hz 的余弦可以产生完全相同的样本序列。高频成分已经折叠进低频后，事后再套数字低通无法判断原来是哪一个。

因此，ADC 前需要合适的模拟抗混叠设计；高采样率数字流降采样前，需要合适的数字低通。上采样或插值可以改变表示频率，却不能创造已经丢失的真实观测带宽。

也要检查库函数的信息条件：SciPy `decimate` 默认启用零相位处理，`resample_poly` 默认提供居中的 FIR 重采样结果。它们适合整段数据处理，但不能逐点调用后就宣称获得了无延迟的实时降采样器。在线版本应维护滤波状态、抽取相位与相应时延；这些状态都属于采样链的一部分。[decimate 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html)、[resample_poly 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)

### 12.3 坐标变换、滤波与时间对齐

固定线性变换与某些线性滤波在满足相同系数和状态条件时可以交换顺序。但若相机外参随时间变化、每轴增益不同，或者操作是四元数球面插值，这种交换通常不成立。

例如相机在运动时，直接平滑相机系中的物体位置，混合了物体运动与相机运动。应先明确要估计哪个参考系中的运动，并使用与测量同一时刻的变换。把“当前位置”和“延迟几十毫秒的姿态”拼成一个位姿，也可能导致末端轨迹畸变。

### 12.4 不要把多个滤波器的延迟藏起来

传感器驱动可能已经低通，感知系统可能做了滑动窗口，遥操作层再做 One Euro，控制器内部还有滤波。每一层单独看都合理，串起来却可能明显迟缓。

对近似线性的小信号情形，各级相位相加；对于限幅、自适应增益和异常门控，还要通过组合回放观察非线性行为。记录整条链的端到端延迟，比孤立记录某个函数耗时更有意义。

串联还会改变噪声的统计结构。独立白噪声经过 EMA 后，稳态相邻输出的相关系数为 $1-\alpha$；当 $\alpha=0.1$ 时，这个相关系数就是 0.9。若再把这些输出交给假定测量噪声时间独立的 Kalman Filter，仅减小 R 不能完整表达变化；应考虑原始测量、已有滤波动态与创新序列的相关性。更平滑的多个样本，不一定提供更多独立信息。

### 12.5 用任务指标验收

| 检查项 | 可以记录的量 |
| --- | --- |
| 静止质量 | 标准差、漂移、极值、异常率 |
| 动态跟随 | 相位差、阶跃上升时间、超调、反向滞后 |
| 几何一致性 | 四元数单位范数、旋转测地误差、坐标系与时间戳 |
| 数据异常 | NaN、重复／倒退时间、掉帧、长间隔、重定位恢复 |
| 控制结果 | 跟踪误差、接触力、饱和次数、振动、任务成功率 |
| 运行代价 | 更新耗时分布、内存分配、最坏耗时与调度抖动 |

先在日志回放中选参数，再在目标控制链中确认。离线 RMSE 较低无法单独证明闭环稳定；而曲线存在小幅高频波动，也不必然表示任务表现更差。

### 12.6 六个机器人场景，怎样缩小候选范围

下面是按问题结构提出的起点，不是可以跨硬件复制的默认参数。

| 场景 | 先检查什么 | 可以比较的处理 | 验收重点 |
| --- | --- | --- | --- |
| 手柄或手部追踪遥操作 | 静止噪声、掉帧、位置与姿态单位 | One Euro；SO(3) 姿态平滑；必要时增加末端运动约束 | 快移拖尾、停下后的收敛、重接管跳变 |
| 编码器估计关节速度 | 分辨率、更新时间、驱动是否已估速 | 因果 SG 微分；跟踪微分器；α–β；含速度状态的 KF | 低速量化抖动、反向误差、反馈相位 |
| 力／力矩测量 | 零偏、温漂、采样硬件、真实接触带宽 | 低阶低通；已确认干扰频率上的 Notch；独立异常标记 | 接触峰值、检测延迟、静态力是否保留 |
| 移动底盘定位 | IMU 与轮速时间、外参、打滑与定位重置 | EKF／ESKF；多假设时考虑粒子滤波 | 一致性、偏置收敛、失联传播和恢复 |
| 视觉目标送给机械臂 | 相机与基座时间对齐、几何跳变 | 门控＋位姿平滑；末端约束或在线轨迹生成 | 动态抓取误差、目标更新频率、最终速度约束 |
| 大臂或柔性负载残余振动 | 随姿态／负载变化的模态 | 模态识别后的输入整形或陷波 | 停稳时间、频率失配、任务节拍 |

例如，30 Hz 视觉定位发来一个目标，500 Hz 控制周期在下一帧前会多次读取同一目标。可以在测量到达时更新位置估计，再在控制时刻预测或推进参考动态；也可以明确定义目标的零阶保持。两条路径都需要记录测量年龄。把同一帧重复塞进“按新测量更新”的统计队列，会悄悄改变窗口和置信度。

如果使用 ROS 的插件链，配置文件里的顺序就是信号处理顺序。ROS `filters` 提供均值、中值和传递函数等组件以及统一链式接口，但不会替应用决定时间戳、坐标系和状态重置语义。配置时同时记录插件版本、系数、初始化方式和链路顺序，才能从日志还原输出。[ROS filters 文档](https://docs.ros.org/en/rolling/p/filters/)

## 13. 随文实验：同时观察抖动与跟随 {#lab}

下载 [robot_filters_lab.py](robot_filters_lab.py)，在安装 NumPy、SciPy 和 Matplotlib 的环境中运行：

```bash
python -m pip install numpy scipy matplotlib
python -B robot_filters_lab.py --output-dir filter-lab-output
```

不加 `--output-dir` 时只运行公式与边界检查；加上它会绘制五组曲线，并写出信号、指标与参数扫描 JSON。本文使用 Python 3.10、NumPy 2.2.6、SciPy 1.15.0、Matplotlib 3.10.9 运行。脚本不连接机器人、不下载模型。

### 13.1 固定参数下，谁更稳、谁更慢

信号以 200 Hz 采样，先静止、再以 0.2 m/s 匀速移动、保持，然后平滑返回；测量叠加标准差 6 mm 的高斯噪声和幅值 3 mm 的 20 Hz 正弦干扰，随机种子固定为 20260925。

{{< robot-filter-lab >}}

可以先保持 One Euro 的其他参数不变，把 β 设为 0，再逐步增大：观察“静止片段”和“开始运动”两种视图，比较噪声与跟随的变化。切换到质量–弹簧–阻尼后，固定自然频率，分别试欠阻尼、临界阻尼和过阻尼。浏览器与下方 Python 曲线使用[同一份合成输入](assets/filter-signal.json)，默认参数可以逐项核对。

<figure class="article-figure" id="fig-tracking-tradeoff">
  {{< post-image src="assets/tracking-tradeoff.png" alt="合成位置测量经四点加权平均、Butterworth、One Euro 与临界阻尼处理后的全程、静止和运动开始局部对照" >}}
  <figcaption><span class="article-figure__number">图 7</span><span class="article-figure__text"><strong>静止抖动与运动滞后要同时验收。</strong>这些曲线使用下表中的示例参数；WMA 更偏重最新样本，其他三种设置的静止抖动更小，但运动开始时响应不同。 <a href="assets/tracking-tradeoff.png">查看原图</a></span></figcaption>
</figure>

下表的静止标准差统计 0.8–1.8 s 区间；等效滞后用 2.5–3.5 s 匀速段的平均位置差除以速度得到。它是该段的跟随统计，不能视为整个滤波器的全频率固定延迟。

| 示例设置 | 静止标准差／mm | 匀速等效滞后／ms |
| --- | ---: | ---: |
| 原始测量 | 6.220 | −1.528 |
| 四点 WMA | 3.569 | 3.379 |
| 二阶 Butterworth，3 Hz | 0.730 | 73.878 |
| One Euro，最低 1 Hz，β=4 | 0.699 | 24.608 |
| 临界阻尼，自然频率 5 Hz | 0.835 | 65.023 |

逐项数值可下载 [默认参数统计 JSON](assets/filter-lab-results.json)。原始测量出现微小负值，是有限样本噪声造成的平均位置偏差，**不是传感器预知未来**。不同示例没有强制对齐带宽或噪声输出，因此这张表用于解释取舍，不是算法排名。换任务速度、噪声分布或参数，排序就可能改变。

### 13.2 脚本还验证什么

- 权重顺序与匀速输入下的样本年龄。
- SOS 分块处理与一次处理得到相同结果，前提是连续保留状态。
- 二阶 Butterworth 手写系数与 SciPy 一致，指定截止处幅值为 $1/\sqrt2$，数字极点位于单位圆内。
- One Euro 的米／毫米参数换算，以及重复时间戳拒绝。
- $q$ 与 $-q$ 不引入姿态变化，$179^\circ$ 到 $-179^\circ$ 沿短弧更新；耦合三轴运动在固定世界／物体参考系变换下得到对应输出。
- 二阶精确离散化与临界阻尼解析阶跃响应一致；已有速度时，临界阻尼仍可能越过新目标。
- 因果 SG 在常速度输入上得到正确导数单位。
- 采样混叠导致的不同连续信号不可区分。

所有曲线与统计均可重新生成。它们验证的是所写公式和指定实现行为，没有覆盖传感器驱动、网络传输、操作系统调度或真实机器人动力学。

### 13.3 参数扫描：先定抖动预算，再观察跟随

前面的单组设置不能代表整个滤波器家族。进一步固定输入、初始化方式和统计区间，扫描 126 组参数：WMA 的递减权重窗口从 1 到 128 点取 14 个值；其他曲线的频率参数在 0.5–20 Hz 之间取 28 个对数等距值。One Euro 分别固定 β=0 和 β=4，导数截止频率保持 1 Hz；MSD 固定为临界阻尼。

<figure class="article-figure" id="fig-parameter-tradeoff">
  {{< post-image src="assets/parameter-tradeoff.png" alt="同一合成信号上五条参数扫描曲线对照静止标准差与匀速等效滞后，并放大一毫米抖动附近的区间" >}}
  <figcaption><span class="article-figure__number">图 8</span><span class="article-figure__text"><strong>一个算法对应一组取舍，而不是一个固定分数。</strong>每个点是一组实际计算的参数；虚线是 1 mm 静止标准差预算，不是位置误差上限。曲线连接用于观察趋势，未计算的参数仍需单独验证。 <a href="assets/parameter-tradeoff.png">查看原图</a></span></figcaption>
</figure>

假设只把“该静止段标准差不超过 1 mm”作为第一道筛选，再从每条曲线的已扫描点中选匀速滞后较小的一点，会得到下面的例子。完整记录可下载 [filter-parameter-sweep.json](assets/filter-parameter-sweep.json)。

| 扫描曲线 | 满足预算的选中参数 | 静止标准差／mm | 匀速等效滞后／ms |
| --- | --- | ---: | ---: |
| 递减权重 WMA | 32 点 | 0.885 | 50.430 |
| 二阶 Butterworth | 截止约 4.450 Hz | 0.919 | 49.280 |
| One Euro，β=0 | 最低截止约 2.576 Hz | 0.928 | 60.591 |
| One Euro，β=4 | 最低截止约 2.247 Hz | 0.976 | 21.633 |
| 临界阻尼 MSD | 自然频率约 5.848 Hz | 0.938 | 55.727 |

这个场景中，One Euro 的自适应增益在运动时放开带宽，能在相近静止抖动下缩短跟随滞后；它并没有消除取舍。增加尖峰、改变运动速度或改变变化率估计，就可能得到不同结果。该表只比较本次有限网格，既不是连续参数空间的最优解，也不是跨任务的排名。

**标准差小也不等于位置准确。** 这些选中输出在静止段仍有约 1.1 mm 的均值偏差，记录中另存了该量。一个完全不动、但位置错误的输出甚至可以有零标准差。因此，抖动预算应与静态偏差、运动误差、异常恢复和闭环任务结果一起验收；真实日志调参后还应在独立记录上复查。

## 阅读自测与验收

- **Butterworth 的通带平坦是否表示零延迟？** 不是，幅值与相位是不同性质。
- **相同四点权重在任何采样率下都一样吗？** 不是，物理时间窗与延迟随更新率变化。
- **One Euro 换成毫米输入还能直接沿用原来的 β 吗？** 不能，速度数值与参数单位必须同步换算。
- **四元数逐分量平均后归一化是否总是正确？** 不是，符号等价、最短弧与旋转几何必须处理。
- **质量–弹簧–阻尼目标平滑是否自动保证速度与 jerk 上限？** 不保证，硬约束需要单独设计。
- **离线零相位曲线是否可以证明在线控制同样无延迟？** 不可以，两者可用的信息不同。
- **曲线更平滑是否足以判断滤波器更好？** 不足，还要检查延迟、失真、异常行为和闭环任务表现。
- **静止标准差小于 1 mm 是否等于位置误差小于 1 mm？** 不是；标准差描述波动，均值偏差与动态误差需要另外统计。

## 参考资料

1. [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)：FIR、IIR、频率响应与流式状态接口。
2. [1€ Filter 作者页面](https://gery.casiez.net/1euro/)与[作者 Python 实现](https://github.com/casiez/OneEuroFilter/blob/d78925584245597f2aa9c4c01a802eb0f0b77fb9/python/OneEuroFilter/OneEuroFilter.py)：算法和调参。
3. [Unitree weighted moving filter](https://github.com/unitreerobotics/xr_teleoperate/blob/7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6/teleop/utils/weighted_moving_filter.py)：权重顺序与队列行为实例。
4. [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508)：姿态、扰动与误差状态。
5. [robot_localization 状态估计节点](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst)：EKF／UKF 与传感器输入配置。
6. [Ruckig 文档](https://docs.ruckig.com/)：速度、加速度与 jerk 约束的在线轨迹生成。
