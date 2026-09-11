RTC：独立 CPU 教学实验
=====================

rtc_lab.py 检查时间对齐、前缀权重、VJP、训练条件、动作尺度与命令队列。
这是本文编写的确定性数学与接口练习，不加载策略权重，不训练模型，
不运行 Kinetix，也不复现三篇论文的任务成功率或延迟。
本次验证环境：Python 3.10.0、NumPy 2.2.6，CPU。

解压后进入 real-time-chunking-lab 目录。Linux/macOS 示例：

  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install -r requirements.txt
  python rtc_lab.py

安装完依赖后，运行不需要网络、GPU、JAX 或完整博客仓库。
末尾应打印 All RTC teaching checks passed.
不提供源码参数时，source parity: False 表示没有进行额外源码对照，
不是练习失败。预期的过期块和异常输入会被程序内部捕获。

代表性结果：
- exp 权重约为 [1, 1, 0.488, 0.189, 0.041, 0, 0, 0]。
- 终点 VJP 为 2.5；仅对速度求导为 3；标量伪逆结果为 0.4。
- 观测步 100、当前步 106、不可撤回边界 108：从新块索引 8 接续。
- 实验内还检查软掩码 W 与 W²、终止动作补零、损失平均方式、
  随机延迟的监督覆盖率、限速后的前缀及反标准化的链式法则。

若本机已有以下固定源码，可以额外运行：

  python rtc_lab.py --source-model /path/to/real-time-chunking-kinetix/src/model.py

参考提交：
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/tree/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b

这一可选操作提取 get_prefix_weights，并以 NumPy 执行数组公式，
不是 JAX 自动微分或性能验证。正常时应显示 source parity: True。
此压缩包不包含官方源码、模型、数据或文章配图生成器。
