VLA 与世界模型数据来源：随文实验与阅读索引
核对日期：2026-09-14

核心实验要求：Python 3.10+，仅标准库，无需 GPU、模型或机器人。
在解压目录运行：
  python data_contract_lab.py
  python make_figures.py

Linux 上可先核对下载内容：
  sha256sum -c SHA256SUMS
该清单用于检测文件是否发生变化，不是第三方签名或论文复现认证。

data_contract_lab.py：30 项人工示例检查，其中一项遍历 400 个规则动作窗口。
lab-results.json：本次实际运行结果。脚本输出可与此文件比较。
make_figures.py：重新生成 assets/ 下三张原创 SVG。
source-map.json：36 项论文／项目阅读索引、两个固定源码提交及设备文档入口。
source_probe.py：可选的固定源码实验，额外需要 NumPy 和两个本地 Git 仓库。
source-probe-results.json：固定源码探针的实际输出及所读文件哈希。
SHA256SUMS：实验包内文件的内容校验；不包含自身。

可选源码探针：
  python source_probe.py --xr-root /path/to/XRoboToolkit-Teleop-Sample-Python --umi-root /path/to/universal_manipulation_interface

两个仓库必须包含 source-map.json 中指定的 Git 提交。探针用 git show 读取固定
对象，不依赖当前分支或工作区文件，不加载任何硬件控制器。它演示 DataLogger
是否补写时间，以及 UMI rel / relative 两种表示混用时的结果。
NumPy 版本变化可能让结果 JSON 的版本字段或浮点末位不同，数值应在容差内比较。

本实验验证：
- 已到达观测、采集时间与时钟换算；
- 未来动作作为训练标签，与不可偷看的未来观测之间的区别；
- 位置目标零阶保持、记录覆盖与跟踪重置；
- 声明的整数时间网格避免浮点边界伪差；
- 缺失动作的有效性屏蔽及分母；
- 混合来源时统一有效元素均值、分别取均值与显式频次加权的区别；
- 源示范家族和新场景评测的划分；
- 真实命令、状态差分、人体重定向与仿真来源的元数据区别。

边界：
这不是 RLDS/LeRobot 的完整数据验证器，不会发现所有视觉近重复。
来源检查只能验证声明的元数据，不能证明命令已被物理执行。
时间量化不会增加传感器精度，也不能替代多设备同步与标定。
位置目标保持不适用于把增量位姿当作持续重复的命令。
没有复现论文训练、成功率、硬件时延或机器人闭环。
图中的时间是人工例子，不是设备测量。

在线文章：
https://chase6305.github.io/posts/ai/vla-world-model-data/
