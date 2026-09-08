MPC / WBC 教程实验包
===================

环境：Python 3.11+；基础实验使用 NumPy 2.3.5。
所有命令在解压后的 control-lab 目录执行。
这些实验不连接机器人硬件。

1. 基础安装
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt

2. 原子 QP 与滚动闭环
   python atomic_control.py
   输出 results-atomic/report.json 和 rollout.csv。
   检查正向/反向/零目标、WBC 标量与双关节耦合任务冲突、不可行边界和 100 步闭环。

3. 状态偏移与短时域反例
   python feedback_demo.py --output results-feedback
   保持 atomic_control.py 与该脚本在同一目录。
   输出 report.json、replay.csv、feedback.csv。
   3 s 注入 -0.08 rad 位置偏移；回放组最终误差约 0.08 rad，反馈组低于 1e-5 rad。
   horizon_trap 字段展示初始两步计划可行、下一拍无可行延续的情况。
   这是状态偏移实验，不是物理冲量仿真。

4. 可选绘图
   python -m pip install matplotlib==3.10.6
   python plot_feedback.py --input results-feedback --output results-feedback/feedback-comparison.png
   绘图不是控制实验的运行依赖。

5. 可选 WholeBodyX 集成
   需另外准备与文章 API 一致的 WholeBodyX 源码，本包不包含该项目。
   将下面路径替换为本机源码路径：
   python -m pip install /path/to/WholeBodyX
   python -B wholebodyx_demo.py --output results-control
   python -B reference_failures.py
   前者运行固定基座六关节对照；后者只检查参考生命周期，不执行机器人。
   WholeBodyX 可能解析额外依赖。文章参考文件 SHA-256 位于博客仓库
   docs/wholebodyx-blog-reference.json；这些哈希用于识别源码，不能代替源码下载。

结果解释
- 时间列是模拟时间，不是计算耗时。
- 默认脚本的断言针对默认配置。改参数后应重新推导预期值，保留硬边界检查。
- 运动学限位、QP 数值精度不能证明动态平衡、摩擦、碰撞或力矩可行性。
- 初始状态满足逐项边界，也可能已无法在边界内制动。
- 请正常运行脚本，不要用 python -O 禁用参考失效测试中的断言。

博客仓库维护者
- 在仓库根目录执行 python scripts/package_control_lab.py 重建下载包。
- 执行 python scripts/package_control_lab.py --check 检查 ZIP 成员与源码逐字节一致。
- 安装 NumPy 后执行 python scripts/package_control_lab.py --check --test，
  从临时解压目录运行原子与反馈实验，并检查 JSON 报告与标准输出一致。
- control-lab.yml 自动检查上述 NumPy 实验，不包含 WholeBodyX 集成或绘图。
