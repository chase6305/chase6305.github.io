Pion 谱计算小实验
================

来源：Rethinking Muon Beyond Pretraining，arXiv:2605.19282v1。
文章：https://chase6305.github.io/posts/ai/pion-muon-vla-rlvr/

解压后进入本目录。Python 3.10+ 的标准库即可运行：

  python3 -B pion_spectral.py --sigma 0.35 --kp 1
  python3 -B pion_spectral.py > spectral-result.json
  python3 -B check_examples.py

默认完整结果见 assets/spectral-reference.json。总迭代次数严格为5；kp为提升次数，
其余为抑制次数。--sigma的输入是归一化后的奇异值，范围[0,1]，不是时间频率。
增加--trace可列出初始输入与每一轮Muon/Pion系数：
  python3 -B pion_spectral.py --sigma 0.35 --kp 1 --trace
  python3 -B pion_spectral.py --sigma 0.35 --kp 2 --trace
第一种调度仅将0.35提升到约0.605，随后抑制到约6.707e-6；第二种再提升到约0.888，
越过单步抑制固定点sqrt(2/3)，随后增长到约0.989。参考assets/trace-reference.json。
--trace只接受--sigma，调用者提供的值已归一化，不会在单方向上重新做Frobenius归一化。
标准库检查用90位Decimal独立核对组合多项式、100位Decimal核对归一化，
并检查固定点、单调性、有限步0.5响应、错误输入与表格算术。
它们不验证模型训练或噪声识别能力。

也可从原始幅度开始，比较固定epsilon对微小输入的影响：
  python3 -B pion_spectral.py --spectrum 0.8 0.4 0.2 0.02 --kp 1
  python3 -B pion_spectral.py --spectrum 8e-9 4e-9 2e-9 2e-10 --kp 1 --epsilon 1e-7
--sigma接收已经归一化的单个值；--epsilon只与--spectrum的原始幅度配合使用。
effective_rank接受未归一化的非负谱，使用奇异幅度而非平方；全零输入约定为0。

已知干净信号的固定扰动对照也只需要标准库：
  python3 -B pion_alignment.py
参考assets/alignment-reference.json。干净矩阵固定为diag(1,0,0,0)，分别加入弱尾部扰动
和占据最大奇异方向的强扰动；比较更新与已知信号的Frobenius余弦。
高通可以压低弱扰动，也会误压较弱真信号。余弦不是成功率，固定扰动不是随机梯度SNR估计。

有限样本梯度SNR的标准库实验：
  python3 -B pion_snr.py > snr-result.json
参考assets/snr-reference.json。每次观测是固定检查点、同一采样定义下的梯度；
在独立同分布且二阶矩有限时，均值平方范数的期望多出总体总方差/K。
脚本以无偏样本总方差/K校正信号功率，保留可能为负的校正值；
校正功率无偏不意味着它与随机方差之比无偏，零样本方差也不证明总体无噪声。
默认数据是四坐标独立正负噪声，不运行模型；GRPO组内回答不能自动当作独立观测。
同步后各进程上的同一梯度副本，也不能充当独立重复。
样本全部载入内存，大模型需按块或流式汇总，并明确unscale、累积、同步与clipping位置。
只有两条观测时，校正功率等于它们的Frobenius内积，相反梯度就会给出负值。
浮点功率超出范围会报错；可用同一个非零常数缩放全部观测，逐条归一化会改变SNR定义。

相关观测与独立采样单位的标准库反例：
  python3 -B snr_correlated_groups.py > snr-correlated-groups.json
参考assets/snr-correlated-groups-reference.json。四组独立标量梯度1+Z，Z等概率取正负1；
将每组复制1/2/8/32次，日志行数增多，但平均梯度的真实方差始终为1/4。
逐行套用独立样本校正的功率期望变为1+(n-1)/(4n-1)，按四个独立组校正则为1。
这些是功率期望，不是平均SNR比值；单次校正仍可低于真值。脚本另保留一组有限样本。
有理数检查枚举全部独立正负组合，复核逐行与按组的期望、同一均值与有限样本波动。
另以不同边际方差和负交叉协方差复核一般偏差公式，以及校正功率的样本对内积恒等式。
分组需要已知的独立采样单位；这里的完全复制不是对真实GRPO相关性的估计。

单步Bernoulli的GRPO组梯度与保留条件：
  python3 -B grpo_group_signal.py > grpo-group-signal.json
参考assets/grpo-group-signal-reference.json。奖励等于回答y，p=sigmoid(theta)，score=y-p。
优势采用组内总体标准差，epsilon=0，恒定奖励组定义为零优势；求导时优势保持常数。
一组g条回答中k次成功时，上升方向为sqrt(k*(g-k))/g，优势和为零不表示梯度为零。
80位Decimal分别计算所有提议组与仅保留混合组的矩，提议成本假设各次采样独立。
例如p=.01、g=8时，条件SNR约309.96，但完整提议组的SNR约.08343；
每保留一组平均需约12.94次提议。条件统计不包括被丢弃组，不代表完整rollout统计。
检查用二进制序列枚举、独立二阶矩公式、固定优势的对数似然及比率目标有限差分核对结果。
内部采用独立的80位Decimal上下文，固定舍入与范围；输入错误也不改变调用者上下文。
该构造没有多token残差、clipping、KL或模型训练；p=0/1仅表示边界分布，
零方差比值记为null。one_group只接受2..128个二进制整数，至多读取129项即拒绝超长组。

动作维数、线性层梯度秩和EMA动量秩的标准库构造：
  python3 -B vla_gradient_rank.py > vla-gradient-rank.json
参考assets/vla-gradient-rank-reference.json。对共享线性层Y=H*W.T，梯度为Delta.T*H，
秩至多为min(N,d_in,d_out)，N合并使用该权重的样本、时间或token位置。
标量输出ReLU网络的隐藏梯度可以满秩；记录给出输入、权重、梯度与精确正特征值证书。
所有预激活远离零且残差为正，因此有限差分不经过ReLU或绝对值的不可微点。
另以beta=.5的EMA累积三个不同方向的秩一梯度，动量秩依次为1、2、3。
检查用损失有限差分、有理数消元、闭式历史和可选NumPy反传/SVD分别核对。
这些是小型构造，不是VLA训练测量；应分别记录原始梯度与实际变换输入的谱。

平坦谱维数与更新幅度的标准库实验：
  python3 -B flat_spectrum.py > flat-spectrum.json
构造完整单位矩阵I_r，epsilon=0，归一化后每个奇异值为1/sqrt(r)。
在精确算术下，梯度项RMS/(eta*gamma)等于五步谱系数的绝对值，不含权重衰减。
相等的正系数始终给出数学有效秩r，但幅度可随维数急降；存成全零是另一个结果。
脚本不分配大型矩阵，100位Decimal给出数学参考，Float64另记其存储结果。
Decimal计算使用独立上下文，固定舍入与指数范围，不继承或改变调用者的设置。
assets/flat-spectrum-reference.json保存默认输出；记录的系数尚未乘eta或尺寸因子，
后续乘法也可能舍入或下溢。本实验不据有效秩自动选择调度。

可选NumPy矩阵部分请安装到独立环境。已核对NumPy 1.26.4与2.2.6，
下面使用Python 3.12，命令适用于Linux/macOS：

  python3.12 -m venv .venv
  .venv/bin/python -m pip install numpy==2.2.6
  .venv/bin/python -B pion_matrix.py
  .venv/bin/python -B check_examples.py --numpy

新增NumPy检查用SVD逐奇异值计算作为矩阵多项式的独立对照，
覆盖长宽矩阵、零矩阵、秩一矩阵、转置与Q/O两种分头轴。
矩阵实现使用Float64，省略动量状态、Nesterov、weight decay、尺寸缩放、
混合精度与分布式训练；不能作为完整训练优化器直接替换。
transform默认epsilon=0，另可显式传入epsilon；非零epsilon会打破极小输入下的尺度不变性。

两个头的构造反例：
  .venv/bin/python -B pion_matrix.py --head-report
参考assets/per-head-reference.json。原动量两个块的范数相差10倍；分头归一化后，
同形谱的输出块范数相同。它说明分头不保证保留范数比例，不包含预训练权重。

per_head_transform遵循PyTorch Linear的(out_features,in_features)布局：
Q/K/V按输出行拆，axis=0；O按输入列拆，axis=1。
GQA的Q头数与KV头数不同；本实验要求显式传入正确头数，维度不整除时报错。
它不自动发现模型结构，也不处理融合QKV或分布式参数分片。

单个整矩阵的动量与权重更新跟踪（同样只需要NumPy）：
  .venv/bin/python -B pion_step.py
  .venv/bin/python -B pion_step.py --missing-gradient decay
使用EMA约定B=beta*Bprev+(1-beta)*G，可选Nesterov输入D=(1-beta)*G+beta*B，
先解耦衰减原权重，再减去学习率、尺寸因子与谱变换的乘积。
演算序列为非零梯度、零矩阵、缺失梯度；beta=.5等参数仅用于方便检查计算。
零梯度仍演化历史动量；None保持动量，skip保持权重，decay仅衰减权重。
前者表示固定RL/verl的跳过策略，后者表示固定VLA-Adapter的仅衰减分支意图。
单矩阵演算不验证多参数异步通信，后者的缓冲复用顺序需另查。
assets/step-reference.json保存两种三步序列。gradient_update_rms只统计梯度项，不含解耦衰减；
configured_decay_factor与applied_decay_factor分别记录配置值与实际施加值，skip时后者为1。
该函数不修改传入数组，
仍不包含AdamW、自动微分、BF16、参数路由与分布式执行。

可选官方CPU探针的独立环境（Linux，Python 3.12）：
  python3.12 -m venv .torch-env
  .torch-env/bin/python -m pip install torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu
  .torch-env/bin/python -m pip install numpy==2.2.6
官方安装入口：https://pytorch.org/get-started/previous-versions/
下面的python可换为.torch-env/bin/python；这些CPU探针不要求GPU或模型数据。
NumPy用于部分SVD精度检查，纯平坦谱和多进程补齐脚本本身不直接依赖NumPy。
该命令已在CPython 3.12.14 / Linux x86_64独立环境跑通；完整训练仍按官方各pipeline配置。

可选官方BF16 CPU探针（另需PyTorch；不作为默认测试的依赖）：
  python -B probe_official_pion.py \
    --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py
使用官方提交87965014b360bde604632cba0aa12e423c6c63c6中的文件，脚本先核对SHA-256。
它分别记录Float64 epsilon=0、Float64 epsilon=1e-7、官方CPU BF16结果。
另以整数外积构造3x4秩一矩阵，记录逐项舍入后的次要奇异值与原方向残差，
说明精确多项式的子空间和[0,1]区间性质不应作为BF16结果的硬性断言。
本次观察见assets/official-precision.json，环境为PyTorch 2.8.0+cu128但计算只用CPU。
不同设备或内核可能产生不同舍入，不能把该记录当作GPU训练逐位复现要求。

平坦单位矩阵的可选官方CPU/BF16观察：
  python -B flat_spectrum.py \
    --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py \
    > flat-bf16.json
同样先核对VLA官方文件SHA-256。仅分配维数1至128的小型单位矩阵，比较
kp=0/1/2、epsilon=0/1e-7，共48组。assets/flat-spectrum-bf16-reference.json
保存本次PyTorch 2.8.0+cu128的CPU观察。零提升五抑制时，从维数8起输出存成零；
这是该构造和计算环境的结果，不是跨设备的统一阈值，也不说明精确多项式秩为零。

可选官方RL整矩阵优化器多步CPU探针（另需PyTorch）：
  python -B probe_official_steps.py \
    --official-file /path/to/Pion/RL/verl/verl/utils/muon.py
同样先核对固定提交的源文件SHA-256。12组配置、每组4次调用，覆盖非零动量、
Nesterov开关、权重衰减、零与缺失梯度。秩一输入使BF16响应为带符号单位系数，
以便单独核对更新顺序；它不证明一般矩阵的Float64/BF16逐位相同。
记录见assets/official-steps.json；不加载模型，不运行GPU或VLA分布式通信。

可选VLA多参数缓冲时序探针（另需PyTorch）：
  python -B probe_official_vla_buffers.py \
    --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py
增加--gloo可运行单进程CPU/Gloo通信。脚本先核对官方VLA文件SHA-256，
将其CUDA接收缓冲分配重定向到CPU；重放“结果已写入”与“等待时写入”两种完成顺序。
可选Gloo仅将输出改为共享原存储的扁平视图，以满足后端形状约束。
这些CPU适配不修改官方源文件；它们与下面的原始CUDA执行路径分别记录。
同一元素数分组内，“非零梯度后紧跟None”可丢失前一更新或把旧更新带给后一参数；
零梯度、反向排列及不同元素数分组作为对照。Gloo完成时序可能改变。
有可用CUDA GPU时，增加--nccl可直接调用原始CUDA缓冲分配与未包装的单进程NCCL，
重复5轮、每轮8个组合。本次RTX 5090 D v2 / PyTorch 2.8.0+cu128 / NCCL 2.27.3上，
35个对照正常，5次同组“非零梯度→None”均丢失前一参数的梯度项，只留下权重衰减。
assets/vla-buffer-reference.json保存24个CPU与40个CUDA组合。所有检查都只用微小矩阵，
不加载模型、不验证多卡训练，也不保证所有硬件/完成时序结果相同。实际后端仍需单独回归。

可选两／三进程VLA补齐与候选修复对照（另需PyTorch CPU/Gloo）：
  python -B probe_official_vla_padding.py \
    --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py \
    --world-size 2 --output padding-two.json
  python -B probe_official_vla_padding.py \
    --official-file /path/to/Pion/VLA/VLAAdapter/pion_optim/muon.py \
    --world-size 3 --output padding-three.json
最后一批参数不足进程数时，补齐分支也会清零接收缓冲；全部真实参数有梯度仍可触发。
脚本保留真实Gloo通信，区分自然异步返回和返回句柄前等待完成的受控时序。
两种类都重定向CUDA缓冲分配到CPU，并使用共享原存储的扁平Gloo输出视图。
original类不改动；private_zero候选在内存中显式将缺失梯度与补齐两处g赋值包成
 torch.zeros_like(update_buffer_views[self.rank])。
候选不是未修改的官方实现；两个类都不重写源文件，哈希在运行前后核对。
每组连续三步覆盖配置梯度、零梯度保留动量、缺失梯度仅衰减；同时核对独立正确结果
与跨进程一致性，并逐步核对动量状态归属、初始化和EMA值。副本相等不能单独证明正确。
运行失败或超时不会写出结果文件；
--output的父目录需已存在，成功时原子替换指定结果文件；拒绝覆盖官方源文件及其别名。
assets/vla-padding-reference.json保存48个组合、360次进程内step调用。
其中候选24组均符合独立三步更新；原类24组中，本次10组有偏差。
自然异步观察可随完成时序改变。只验证小型CPU/Gloo，不运行模型、不验证多卡NCCL
或性能；这是一处存储冲突的候选修复对照，不能作为生产训练器的完整验收。

可选绘图：
  .venv/bin/python -m pip install matplotlib==3.10.9
  .venv/bin/python -B make_figures.py
  .venv/bin/python -B make_rank_figures.py

输出assets/spectral-polynomials-v2.png。谱响应直接来自论文系数与固定构造谱，
不是训练曲线；图底对低于1e-12的系数只做显示截断并标倒三角，JSON保留原值。
平坦谱图另输出assets/flat-spectrum-rms.png，读取标准库参考JSON，使用Decimal对数；
BF16与Float64格式最小正数仅是表示范围参考，不预测实际内核的下溢位置。
两图复现环境为Matplotlib 3.10.9；不同版本的字体与PNG字节可能不同。

有限多项式是平滑谱变换，不是硬top-k。数值极小或浮点下溢不证明精确秩为零。
纯抑制也可能压小所有方向；它不保证自动区分噪声与有效梯度。
