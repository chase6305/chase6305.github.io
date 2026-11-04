Muon：两篇文章的可复现实例
================================

文章：
https://chase6305.github.io/posts/ai/muon-optimizer/
https://chase6305.github.io/posts/ai/muon-learning-rate-batch-size/

这份下载包含标量公式、Float64 矩阵参考和一个完整的 PyTorch CPU
训练/恢复例子。不下载模型或数据集，不进行 GPU 或分布式训练。
各脚本输出构造值或集成检查，不构成优化器性能排名。

解压后保持 optimizer/、batch/、checks/ 和 muon-download.json 的位置。
以下命令在解压目录执行；从其他目录调用时，给脚本加上实际路径。

1. 只用 Python 标准库
---------------------

建议 Python 3.10+：

  python3 -B optimizer/muon_math.py
  python3 -B batch/batch_math.py
  python3 -B checks/test_muon_articles.py

最后一条会跳过需要 NumPy 的矩阵检查。它仍会检查 Decimal 多项式、
归一化约定、形状因子、学习率/衰减换算、滤波权重、独立高斯积分、
噪声估计和累计收缩。默认参考输出在两份 assets/math-reference.json。

batch_math.migration_example() 对应文章中的 Batch 迁移计算器：
学习率依赖平均场与曲率条件；动量只匹配旧状态保留比例；两种衰减
换算分别匹配每步收缩或同样本量的累计收缩。它不保持整个参数轨迹。

2. Float64 矩阵参考
-------------------

可建立独立环境。下面使用 Python 3.12，Linux/macOS 命令；Windows
将 .venv/bin/python 换成 .venv\Scripts\python.exe。

  python3.12 -m venv .venv
  .venv/bin/python -m pip install -r requirements-numpy.txt
  .venv/bin/python -B optimizer/matrix_update.py
  .venv/bin/python -B optimizer/matrix_update.py --transform polar
  .venv/bin/python -B batch/mean_field_probe.py
  .venv/bin/python -B checks/test_muon_articles.py --numpy

matrix_update.py 默认采用 EMA、Nesterov、clamp 归一化、五步快速 NS、
Moonlight 尺寸因子与解耦衰减。polar 选项用 SVD，并显式选择数值秩：
默认相对阈值是 max(shape) * Float64 epsilon，不是精确实数秩判定器。
零梯度继续作用于已有状态；缺失梯度跳过该矩阵。函数不修改输入。
参考轨迹分别在 optimizer/assets/matrix-ns-reference.json 与
optimizer/assets/matrix-polar-reference.json。该参考不模拟 BF16。

batch/mean_field_probe.py 在同一参数位置对两个构造高斯梯度分布各做
20 万次独立 SVD，不执行训练。每个 Batch 梯度元素的噪声方差为 0.05；
它比较平均极因子与 Gram、标量两级近似。区间为均值正负 1.96 个
样本标准误，仅是近似蒙特卡洛区间。结果在
batch/assets/mean-field-probe-reference.json。

3. 原生 PyTorch CPU 训练与恢复
------------------------------

在同一个独立环境安装 CPU 版本：

  .venv/bin/python -m pip install -r requirements-torch.txt
  .venv/bin/python -B optimizer/toy_training.py
  .venv/bin/python -B checks/test_muon_training.py

requirements-torch.txt 使用 PyTorch 官方 CPU wheel 源，固定 2.14.1。
如果已经使用其他环境，可直接运行脚本；toy_training.py 检查接口是否
存在，但可复核的源码哈希测试固定在 2.14.1。PyTorch 官方安装入口：
https://pytorch.org/get-started/previous-versions/

本例以 512 条训练数据、128 条留出数据做合成回归。只有隐藏线性层的
权重交给 Muon；输入、输出权重与全部 bias 用 AdamW。默认 Batch=32、
microbatch=13，按 13+13+6 归约，三次反向后才调用一次优化器更新。

默认 160 次更新在中间保存临时 checkpoint，随后比较连续训练和完整
恢复的最终参数。还会单独检查丢失 Muon 动量的分支。保存文件留在
临时目录并自动删除，不覆盖当前目录的用户文件。

如需真的退出进程再恢复，可以使用一个尚不存在的路径：

  .venv/bin/python -B optimizer/toy_training.py --save-checkpoint run.pt --steps 80
  .venv/bin/python -B optimizer/toy_training.py --resume run.pt --steps 160

steps 是累计完成的更新次数。保存拒绝覆盖任何已有文件；恢复会
保持文件中记录的 Batch 与 microbatch。输出中的 parameter_sha256
可以与同一环境中默认 160 步连续训练的结果比较。独立检查会真的
启动新 Python 进程来验证这一点。

默认 CPU 参考输出见 optimizer/assets/toy-training-reference.json。
它记录 PyTorch 2.14.1+cpu、CPU float32 参数；该实现内部 NS 使用 BF16。
留出集损失只是小例子的观测，不是通用收敛保证，也不是与 AdamW
同等调参预算的对照。跨硬件或版本的数值未承诺逐位一致。

这个模型不含 dropout、随机增强、AMP 或学习率调度器。加入它们后，
还需要恢复对应随机数、scaler 和 scheduler 等状态。仅保存本例的
采样器状态不足以覆盖所有随机来源。

4. 固定参数位置采集梯度
-----------------------

安装过同一 CPU PyTorch 环境后，可在解压目录运行：

  .venv/bin/python -B optimizer/gradient_noise.py --save-records gradients.json
  python3 -I -S -B optimizer/gradient_noise.py --replay gradients.json
  .venv/bin/python -B checks/test_muon_noise.py

默认固定在上述模型的初始化位置，记录 hidden.weight 的 64 份梯度；
每份独立、有放回地抽取 16 行数据，损失同时平均行与输出坐标。
这里 K=64 是重复观测次数，b=16 是每份记录的样本数，不执行训练更新。

autograd.grad 返回原始梯度，不向已有 .grad 累积；采集前后核对全部
参数指纹。Float32 模型求导后才转换成 Python 浮点数做统计，转换
不会把前向/反向变成 Float64。模型没有 dropout 或运行统计；扩展到
其他模型时，还需固定相应状态并明确采样与损失目标。

记录含参数形状、抽样索引、原始梯度和参数指纹。保存拒绝覆盖已有
文件；回放只读，不重新采样。第二条命令禁用所有第三方包，统计
仍可复算。这个脚本需要完整包中的本地 batch/batch_math.py；
不会搜索父级 checkout 或 PYTHONPATH 来补齐缺失文件。

默认输出见 optimizer/assets/gradient-noise-reference.json。它是当前
矩阵、当前参数位置的探索性噪声尺度估计，不自动确定最优学习率。
记录格式的检查也不能证明真实采样独立或真实模型状态已固定。

5. 可选重画图表
---------------

  .venv/bin/python -m pip install -r requirements-plots.txt
  .venv/bin/python -B optimizer/make_figures.py
  .venv/bin/python -B batch/make_figures.py

图像写入各自 assets/。构造数据和局部模型不会变成实测训练结果。
概念图不在此包内；文章已提供对应文字、图注和可放大的图片。

6. 独立验证的范围
-----------------

数值检查包括高精度 Decimal、显式 EMA/Nesterov 权重、独立参数更新、
噪声符号穷举、固定网格 Simpson、高斯闭式二阶矩和 NumPy SVD。
PyTorch 检查还验证模块分组、不等 microbatch 的解析梯度、共享角色
冲突、零/缺失梯度、两套状态的恢复与固定版本的源码 SHA-256。
梯度采集检查另含独立线性导数、已有梯度/参数/随机状态保留、记录
契约、已有文件保留，以及不加载任何第三方包的独立进程回放。

不验证真实数据的独立性、真实训练中的噪声识别与统计稳定性、模型
训练优劣、GPU 精度、并行通信、真实吞吐或通用最优 Batch/学习率。
