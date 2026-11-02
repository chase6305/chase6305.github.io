# Pion 文章三小时优化复核

工作从 2026-10-30 10:27:42 UTC 开始。完成时间、持续时长与最终文件哈希记录在 [本轮审阅记录](../../pion-optimization-20261030-threehour.json)。本文的原发布日期、草稿状态和原有 49 个页面 ID 均保留，其他文章的既有修改保持不变。

本轮补充了矩阵／分头更新流程、交互谱实验、固定 ε、CPU BF16 对照、两个头的幅度反例，以及具有已知干净信号的方向对照。对多项式调度顺序和 Frobenius 幅度预算给出了直接推导，并核对了固定官方代码的参数路由、尺寸缩放与训练停止条件。来源和适用范围见 [来源复核](../../pion-source-audit-20261030.json)。

数值检查包括 10 项标准库测试和 3 项 NumPy 测试；标准库部分在 Python 3.10、3.11、3.12 运行，NumPy 部分在 1.26.4 和 2.2.6 运行。JavaScript 检查包含 900 组变换不变量、124 组 Python 对照与 23 组非法输入。下载包解压后也运行同一组检查。

网页证据汇总在 `browser-evidence.json.gz`：覆盖窄屏、放大文字、三种主题、有／无脚本、键盘输入与焦点、绘图失败与恢复、触屏滑动、图片查看、搜索以及 A4／Letter 原生打印。图片 `mobile-number-200.png` 展示了科学计数法在 200% 文字下保留完整指数的换行。长公式的打印检查另覆盖 200% 文字；并列等式分行后不再超出纸面。

`official-head-axes.json` 是固定 RL 源码在 CPU 上进行 11 项小矩阵检查的记录。可用以下命令复算；官方文件必须来自提交 `87965014b360bde604632cba0aa12e423c6c63c6`，脚本先核对哈希：

```bash
CUDA_VISIBLE_DEVICES='' python3 -B probe_official_head_axes.py \
  --official-file /path/to/Pion/RL/verl/verl/utils/muon.py
```

这些检查针对数学、未分片小矩阵与网页行为，没有复现 VLA／RLVR 训练、GPU 性能、FSDP 参数布局或机器人试验。触屏检查使用 Chrome 的触控事件模拟，没有替代真实移动设备测试。CPU BF16 结果也不是跨设备逐位一致的要求。

新流程图使用内置图像生成工具，提示、编辑指令与逐项检查记录在 [流程图来源](../../image-generation/pion-flow-20261030.json)。定量图使用 Matplotlib，重新运行后与原图逐字节相同；原论文面板的裁剪、出处与许可记录在 [原图及定量图来源](../../image-generation/pion-20261030.json)。
