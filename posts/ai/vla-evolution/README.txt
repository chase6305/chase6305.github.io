VLA 文章随文实验
================

文章：https://chase6305.github.io/posts/ai/vla-evolution/

建议顺序
--------
1. 先运行标准库算例，检查单位、编码、Flow 时间方向和动作时钟。
2. 再重画解析双峰流场，观察同一条件下不同初始噪声的结果。
3. 最后训练一个 4,609 参数的小网络，对照学习误差与积分误差。

这里没有机器人连接、模型权重下载或真机评测。第二步使用已知的数学
流场；第三步才实际训练神经网络。两者都不是 pi0 的复现。

一、标准库算例（Python 3.8+）
----------------------------
进入解压后的 vla-labs 文件夹：

  python3 -B vla_lab.py
  python3 -B vla_lab.py --self-test

默认输出 JSON；--self-test 执行 8 组检查。部分对照结果：

  uniform_quantization.index = 153
  uniform_quantization.reconstructed_m = 0.019921875
  oracle_flow_correct_endpoint ≈ 2
  oracle_flow_wrong_endpoint ≈ -4
  clocks.action_period_s = 0.02
  clocks.chunk_coverage_s = 1.0
  clocks.prefix_duration_s = 0.5
  clocks.ideal_replans_per_s = 2.0
  reward_and_advantage.ten_step_advantage = 20

浮点输出可能包含很小的舍入误差。量化算例使用 256 个等宽区间，
不是 OpenVLA tokenizer 的逐行复刻；oracle_flow 使用已知的目标速度，
只检查方向。bimodal_flow 才展示同一个解析速度场如何保留两种模式。

二、重绘解析双峰图（另需 Matplotlib）
-----------------------------------
推荐 Python 3.10+，在独立环境中安装：

  python3 -m venv .venv-flow
  .venv-flow/bin/pip install matplotlib==3.10.9
  .venv-flow/bin/python -B plot_flow_modes.py --output analytic-flow.svg

图来自概率各半的 -1 / +1 两点分布。没有训练网络。积分停在 s=0.02，
避开离散数据终点的奇异边界。请不要把这个步数当成 pi0 的采样预算。

三、真正训练一个小速度网络（Python 3.10+）
-----------------------------------------
继续使用上面的独立环境，安装已验证的 CPU 依赖：

  .venv-flow/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
  .venv-flow/bin/python -B flow_matching_lab.py \
    --output local-results.json --figure-dir local-figures

默认使用三个训练种子 17、29、43，每个种子训练 6,000 步，batch=256。
网络输入是一个混合数值、流时间与一个标量条件，输出一个速度值。
不包含视觉编码器、Transformer 或机器人动力学。

生成文件：

  local-results.json                         完整数值与实验配置
  local-figures/flow-learned-distribution.png 输出是否保留两个峰
  local-figures/flow-solver-comparison.png    更多积分与拟合误差的关系
  local-figures/flow-sample-paths.png         不同初始噪声的生成路径

也可以先用 --steps 100 --seeds 17 检查安装与运行过程。这样的快速运行
改变了训练预算与种子数量，不应期待得到文章中的结果。

assets/flow-learning-results.json 是文章的参考记录，包含 30 行条件／
种子／求解器组合。在两个已检查的 CPU PyTorch 版本上得到相同指标，
但不同平台或未来依赖版本不保证逐位一致。

如何读结果
----------
- learned_quantile_W1：生成分布相对已知目标的一维距离，越小越接近。
- exact_field_quantile_W1：使用精确速度场时仍存在的有限步采样误差。
- right_mode_fraction：落在条件中心右侧的比例，单独不能证明模式覆盖。
- gap_fraction_abs_center_lt_0_4：落在两峰之间低密度区域的比例。
- field_validation.sample_target_mse：相对抽样路径速度标签的误差。
- field_validation.exact_field_mse：相对精确条件速度场本身的误差。

后两项的监督目标不同。精确场对抽样标签的误差也可以非零，因为
给定混合输入后仍有多条可能的训练路径。它不是机器人失败率。

保持比较公平
------------
比较求解器时固定 checkpoint、条件、初始噪声以及 NFE。一次 Heun
更新调用两次速度场，一次 Euler 更新调用一次；步数相同不代表预算相同。
脚本使用固定高斯分位点作为初值，不在重复比较时重新抽一批噪声。

这些脚本不会计算真实机器人成功率、硬件推理延迟或最佳控制频率。
文章中的时钟交互也只做算术；真实系统还需记录推理、网络与执行调度。
