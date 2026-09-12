X-VLA：独立 CPU 教学实验
========================

这些程序解释博客中的公式和接口，不加载 X-VLA 权重、不使用机器人数据，
也不复现论文成功率。它们是本文编写的教学代码，不是官方推理程序。
本次验证环境：Python 3.10.0、NumPy 2.2.6，CPU。

解压后进入 x-vla-lab 目录。Linux/macOS 示例：

  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install -r requirements.txt
  python action_generation_lab.py
  python rotation_lab.py
  python timing_lab.py

其中 action_generation_lab.py 和 timing_lab.py 只需要 Python 标准库，
可以跳过 NumPy 安装单独运行。安装完依赖后，实验运行不需要网络或 GPU。

程序与预期结果：
- action_generation_lab.py：重混合结果 89/36，反向 Euler 结果 39/16；
  还检查噪声配对、参数量、采样权重、单位换算与输入屏蔽。
  成功时最后打印 All exact-arithmetic checks passed.
- rotation_lab.py：两种 6D 排列分别往返通过；故意混用排列的例子产生
  约 83.29 度误差。检查退化旋转、6D 插值边界、夹爪软标签，以及基座／工具坐标和 SE(3) 增量组合。
- timing_lab.py：1 秒、4 秒和 0.4 秒窗口的目标间隔分别为
  1/30、2/15、1/75 秒；10/12 秒起始的两个 4 秒窗口共享 15 个未来标签时间。
  成功时打印 All timing checks passed.

预期的退化输入会被程序内部捕获；它们不是实验运行失败。
可选执行 python timing_lab.py --write-figure，在当前实验目录的 assets/
下重建 action-time-grid.svg。图片由精确时间坐标生成，不是模型轨迹。

参考源码固定为：
https://github.com/2toinf/X-VLA/tree/6bc2513f5f1cbec715cc668b414392a6cae5c671
该源码不包含在本压缩包中。无需下载它即可运行上述实验。
