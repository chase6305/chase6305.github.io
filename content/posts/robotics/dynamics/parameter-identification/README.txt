机器人动力学参数辨识：两个可运行的离线示例
================================================

解压后进入本目录。示例不需要 RobotServer，不连接机器人，不修改 URDF 或控制器。
参考复现环境：CPython 3.12.14、Linux x86_64、NumPy 2.2.6；
Pinocchio 4.1.0 / Drake 1.58.0。其他平台请检查对应项目的 wheel 支持。

1. Pinocchio：URDF -> 惯性参数回归矩阵 -> 缩放 SVD -> 独立轨迹验证

  python3.12 -m venv .venv-pin
  .venv-pin/bin/python -m pip install -r requirements-pinocchio.txt
  .venv-pin/bin/python -B pinocchio_identification.py --output pin-result.json
  .venv-pin/bin/python -B check_examples.py --only common
  .venv-pin/bin/python -B check_examples.py --only pinocchio

安装包名为 pin，导入名为 pinocchio。构造真值保持 URDF 几何，改变质量、质心和惯量，
用解析状态导数和真值模型 RNEA 生成力矩，再叠加标准差 0.02 N*m 的独立力矩噪声。
真值参数不进入求解器。默认输出应接近 assets/pinocchio-reference.json：
802x20 回归矩阵、保留秩 6、验证含噪力矩 RMSE 约 0.0202 N*m。
名义 URDF 在同一验证集的 RMSE 约 0.1848 N*m；还会输出逐关节残差指标，
残差方向统一为观测力矩减预测力矩，平均值的正负号沿用这一约定。
20 维预测向量不是唯一物理参数；最小范数取自缩放参数坐标，不可直接写回 URDF。
列二范数使用稳定的 hypot 累加，避免直接平方造成上溢或下溢；不可表示的范数或
参数会明确报错。数值缩放不能恢复未激励的方向，也不等同于噪声加权。
结果保留原始列二范数（零列为0），与求解时为避免除零使用的缩放值1区分。
归一化条件数不代表激励强度；还需结合观测噪声、带单位的列范数与参数不确定性。
该 Y 只含刚体惯性块（含重力）。Pinocchio 4.1 的 RNEA 另加 model.armature*ddq，
但不自动使用 model.friction/model.damping 生成摩擦项。默认模型的 armature 为零；
自己的数据需另行补偿已知附加项或扩展未知参数列，不能把它们隐含归入刚体十参数。
结果会记录未纳入项与名义 armature；预处理工具不会自动做这些动力学补偿。

CSV 通道验证（example-csv 中的两个目标文件必须尚不存在）：

  .venv-pin/bin/python -B pinocchio_identification.py --csv-dir example-csv
  .venv-pin/bin/python -B pinocchio_identification.py --train example-csv/train.csv --validation example-csv/validation.csv

自己的有界转动关节固定基模型与处理后数据：

  .venv-pin/bin/python -B pinocchio_identification.py --urdf robot.urdf --train train.csv --validation validation.csv --output candidate-prediction.json

每个 CSV 至少两行，字段如下（每组覆盖模型中全部关节，可按名称重排列）：
  time_s,q/joint1,q/joint2,dq/joint1,dq/joint2,ddq/joint1,ddq/joint2,tau/joint1,tau/joint2
位置 rad，速度 rad/s，加速度 rad/s^2，模型关节侧力矩 N*m。
时间严格递增；所有必要值有限。训练/验证必须独立采集，输入字节哈希相同会被拒绝；
文件使用 UTF-8，读取时也接受 UTF-8 BOM；导出拒绝复数信号并保留已有文件。
内容不同仍不保证统计独立。脚本不负责同步、滤波、求导或电机侧到关节侧换算。
--output 可以更新结果 JSON，但不能指向 URDF 或输入 CSV 的同一路径、符号链接或硬链接。
两套结果保留后端版本、模型、算法与公共工具哈希；真实 CSV 另记训练/验证输入哈希。
移动关节需处理力单位，连续/球关节/浮动基需扩展配置表示，当前适配器拒绝它们。
根坐标系重力默认 [0,0,-9.81] m/s^2。可用 --gravity GX GY GZ 指定，例如
--gravity -9.81 0 0 仅适用于已确认重力沿根坐标系 -X 的安装。报告会保留此向量；
若 URDF 已含同一个固定安装旋转，避免再次重复换算。三个分量必须有限。

只有位置和力矩时，使用 NumPy 预处理工具（两组原始日志必须独立采集）：

  .venv-pin/bin/python -B preprocess_identification.py --make-demo raw-demo
  .venv-pin/bin/python -B preprocess_identification.py --input raw-demo/train-raw.csv --output processed/train.csv --joints joint1 joint2 --window 21 --degree 5 --max-gap 0.03
  .venv-pin/bin/python -B preprocess_identification.py --input raw-demo/validation-raw.csv --output processed/validation.csv --joints joint1 joint2 --window 21 --degree 5 --max-gap 0.03
  .venv-pin/bin/python -B pinocchio_identification.py --train processed/train.csv --validation processed/validation.csv --output raw-pipeline.json

--make-demo 仅生成构造记录，需 Pinocchio；处理自己的 CSV 只依赖 NumPy。
原始入口需要 time_s,q/<joint>...,tau/<joint>...；已有的 dq/ddq 列不使用，导数按位置重算。
使用实际时间戳的局部多项式拟合，时间先居中、缩放，再用最小二乘求系数与导数。
21 点/5 阶和 30 ms 间隔上限只针对这组构造信号，需要按真实运动和噪声选择。
示例位置噪声为 1e-5 rad；裁去边界后保留 381/281 点，验证 RMSE 约 0.0203 N*m。
每个 CSV 的 .meta.json 保存原始/输出/算法哈希、窗口跨度、条件数与裁剪数量。
需要新的 CSV 和元数据路径；输入文件不会被覆盖。

若已知固定力矩延迟，可加 --torque-delay 0.020：在 t+0.020 插值记录力矩，匹配 t 的
状态；负值表示力矩记录超前。仅取共同覆盖区间，不外推。这个延迟应用到全部
关节列；不同通道延迟不同应先分别对齐。工具不估计延迟、不解缠、不自动剔除
接触或分割故障段。平滑与插值会改变误差相关性；这不是同时滤波 Y/tau
的等价实现，也不评估物理准确性或参数置信区间。

在相同的中央时间段比较五组固定窗口（需要 Pinocchio）：
  .venv-pin/bin/python -B preprocessing_study.py --summary --output window-study.json
参考：assets/preprocessing-study.json。所有设置使用同一组 321 个训练时刻与另一
轨迹的 221 个验证时刻。解析导数和无噪力矩只做诊断，不参与回归或自动选窗。
21点/3阶的加速度误差约为21点/5阶的37倍，含噪力矩RMSE却几乎相同；81点/5阶
平滑偏差更明显。误差聚合两个关节，单位分别为rad/s^2与N*m。这是固定构造对照，
不是通用窗口推荐；相同点数在50 Hz与约42.9 Hz数据上对应不同物理时间跨度。

2. Drake：URDF -> 符号逆动力学 -> 参数组合分解 -> 独立验证

  python3.12 -m venv .venv-drake
  .venv-drake/bin/python -m pip install -r requirements-drake.txt
  .venv-drake/bin/python -B drake_identification.py --output drake-result.json
  .venv-drake/bin/python -B check_examples.py --only drake

仅参数化本包 pendulum_id.urdf，基座焊到世界。joint1 绕 +Y 轴旋转，q=0 时质心
朝下，重力 [0,0,-9.81] m/s^2。URDF 阻尼为零，额外黏性阻尼只加入一次。
方程为 tau = A*ddq + b*dq + 9.81*h*sin(q)，A=J_com_yy+m*ell^2，h=m*ell。
默认输出接近 assets/drake-reference.json：A=0.67020，b=0.08237，h=1.00018，
验证含噪力矩 RMSE 约 0.0198 N*m。三个组合不唯一确定四个未知量 m,ell,J,b。
若 ell 已由独立测量固定，可增加 --known-com-distance 0.5，得到质量约 2.00036 kg、
质心 Y 轴惯量约 0.17011 kg*m^2。仅恢复该单轴模型，并保留正质量、正惯量和非负阻尼
的必要检查；负估计不会被截断。这不等于完整三维惯量或已考虑距离测量不确定性。
符号构造器以 k=J/m 表示质心单位惯量，移到刚体原点后传入 SpatialInertia。
Drake 1.58 可能保留精确 0/m 项；脚本仅在物理域 m>0 上精确移除该表达式，
不通过数值采样猜测零项，也不把小估计值当作零。

同一单摆约定下的真实处理后 CSV 可使用 --train 和 --validation，列为
time_s,q/joint1,dq/joint1,ddq/joint1,tau/joint1。脚本不接受通用 --urdf 替换；
多关节或不同几何需要重新声明符号参数、驱动、外力与参数组合。

3. 独立复算时间延迟反例（Python 标准库，无需 NumPy 或动力学库）

  python3 -B timing_counterexample.py --check
  python3 -B timing_counterexample.py --delay-ms -20 --train-hz 0.7 --validation-hz 1.1
  python3 -B timing_counterexample.py --correct-training-delay

单轴移动系统 F=m*a+b*v，真值 2 kg 与 0.4 N*s/m，位移幅值 0.03 m。
解析力记录正延迟表示滞后状态；训练 400 点、验证 300 点，均为 100 Hz。
默认 20 ms 滞后可被错误系数吸收，训练残差仍接近零。验证相对同样滞后记录
的 RMSE 约 0.16561 N，相对已对齐参考约 0.11428 N。两个目标要分别解释。
已知延迟校正直接计算解析函数 t+delay，不验证离散插值、边界外推或真实测量精度。
默认参考见 assets/timing-reference.json，网页版本另用两列 QR 求解并与本闭式结果核对。

4. 同时检查两个后端

在一个环境中安装两个 requirements 文件后：
  python -B check_examples.py

检查包括独立两连杆解析方程、单摆闭式方程、两库同一模型的随机状态（两连杆名义与
真值模型各 200 组，另有含空间关节轴、旋转惯性坐标系及固定工具的分支模型 400 组）、
CSV 错误与回放、欠激励、秩和留出预测。源码仓库还执行文章中的两个 Python 片段；
ZIP 没有完整文章，因此两项文章片段检查明确跳过。正常完整运行：28 tests，2 skipped。
文章图 7 可用 Drake 环境附带的 Matplotlib 从保存的结果重画：
  .venv-drake/bin/python -B make_identification_figure.py --output held-out.png
Linux x86_64 / CPython 3.12 的完整 wheel 哈希锁定文件见 requirements-linux-cp312.txt，
可在独立环境中使用 pip install --only-binary=:all: --require-hashes -r 该文件。

参考误差来自固定模型、解析轨迹和随机种子，不是机器人测量精度。
没有验证接触、柔性、摩擦模型、物理惯量约束、实时性或硬件安全。
真实项目继续完成文章中的数据审核、物理一致性、不确定性与回滚步骤。

文章：https://chase6305.github.io/posts/robotics/dynamics/parameter-identification/
