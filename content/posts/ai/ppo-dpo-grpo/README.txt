PPO / DPO / GRPO CPU 实验包
文章：https://chase6305.github.io/posts/ai/ppo-dpo-grpo/

1. 安装与运行

将 ZIP 中全部文件解压到同一个目录，在该目录打开终端。
本文验证环境为 Python 3.10、PyTorch 2.8.0；推荐用独立虚拟环境。
以下命令针对本文实测的 Linux x86_64 环境；安装依赖需要网络，之后核心实验可离线运行。

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
    python -B -m unittest -v test_rl_lab.py

当前版本应运行 19 项测试，并以 OK 结束。测试包括三种算法、三个 seed、
两步 PPO、EOS/PAD、loss/梯度方向与停止梯度；只打印 loss 不代表验收通过。
Windows 可用 py -3.10 -m venv .venv 创建环境，并在 PowerShell 执行
.venv\Scripts\Activate.ps1；环境激活后使用相同的 python 命令。

2. 三层实验，各自回答一个问题

原子模块与一步训练：
    python -B rl_lab.py --algorithm atoms
    python -B rl_lab.py --algorithm all --output results

    atoms 打印 GAE、clipping、DPO 和 group 优势的可手算数值。
    all 运行四上下文、三动作的一步任务，生成 ppo/dpo/grpo.csv 和 JSON。
    初始 expected_reward 约 0.2333，最优动作概率为 1/3；观察训练后上升。
    三种方法的数据与更新预算不同，不能据此排序真实 LLM 能力。

多步信用分配：
    python -B ppo_chain.py --output results-chain

    每回合两步，第一步无奖励，第二步结束时给奖励。默认 gamma=1、lambda=0.95。
    输出 ppo-chain.csv 和 JSON；expected_return 从 0.3 向 1 提升。
    JSON 的 first_rollout_trace 保存前四个回合，可手算优势是否跨回合串联。
    value_start 是 critic 估计，可能略超出奖励范围，不等于精确评估回报。

序列目标：
    python -B token_objectives.py --output results-tokens.json

    五个 token 的 bigram 表，对两条固定回答分别做一次 DPO/GRPO 更新。
    检查有效长度 [2,3]、初始 DPO loss≈0.6931、初始 GRPO loss≈0 但梯度非零。
    EOS 与 PAD 共用编号，真实 EOS 仍计分；这不是在线采样的 GRPO 训练。

3. 复现与改参数

    python -B rl_lab.py --algorithm grpo --seed 19 --group-size 4 --output results-seed19-g4
    python -B ppo_chain.py --seed 7 --gae-lambda 0 --output results-chain-lambda0
    python -B ppo_chain.py --seed 7 --gae-lambda 1 --output results-chain-lambda1

DPO 标签对照，保持用于评估的原始奖励表不变：
    python -B rl_lab.py --algorithm dpo --preference-flips 0 --output results-dpo-clean
    python -B rl_lab.py --algorithm dpo --preference-flips 3 --output results-dpo-flips3
    python -B rl_lab.py --algorithm dpo --preference-flips 12 --output results-dpo-reversed

--preference-flips 为 0～12 的整数，按 seed 选择固定 N 对标签交换，并记录实际数据。
比较更新后的 dpo_training_loss、dpo_clean_loss 与 expected_reward：
全量反转时，训练 loss 仍下降，但原始标签 loss 上升，真实奖励下降。
clean loss 使用同一批上下文的原始标签，并非独立验证集。

GRPO 固定每轮 128 个动作，比较分组方式：
    python -B rl_lab.py --algorithm grpo --group-size 2 --grpo-prompts 64 --output results-grpo-g2
    python -B rl_lab.py --algorithm grpo --group-size 8 --grpo-prompts 16 --output results-grpo-g8

--grpo-prompts 是每轮抽取的 prompt 组数，可重复；采样动作数为组数乘 group-size。
固定动作数不代表固定上下文覆盖、FLOPs 或运行时间。零方差组全对、全错都会出现，
因此要同时观察 expected_reward，不能只凭 zero_group_fraction 判定训练失效。

同名输出会覆盖，改 seed 或设置时请换目录。先保持其余配置不变，再比较一个因素。
JSON 保存环境、设置和首末指标；CSV 保存迭代记录。默认 120 轮，--steps 3 可快速
检查命令入口，但不应用 3 轮结果判断收敛。平台差异可能改变浮点末位。
λ=0 只使初始零 critic 下的第一步优势为零，critic 学到后仍可通过 TD bootstrap 学习。

4. 可选绘图

    python -m pip install "matplotlib==3.10.6"
    python -B plot_results.py --input results --output training-curves.svg

plot_results.py 读取一步实验的三份 CSV，不能直接用于两步实验输出。
Matplotlib 不属于核心运行依赖；不绘图时无需安装。
图片与文章中的参考数据未放进 ZIP；实际运行会在上面的输出路径重新生成数据。

5. 常见运行问题

找不到 rl_lab / ppo_chain / token_objectives：完整解压所有文件，确认当前目录正确。
找不到 torch：确认在运行脚本的同一个 python 环境中使用 python -m pip 安装依赖。
无法安装固定 PyTorch 版本：先核对 Python 版本、操作系统架构与下载源访问情况；
本文实测为 Linux x86_64 / Python 3.10，不要求 GPU，也不下载预训练模型。
测试失败：保存完整 traceback、python --version 及 python -m pip show torch 输出，
先恢复 ZIP 原始文件验证，再检查自己的改动；不要仅调低断言阈值。

6. 仓库维护者

在仓库根目录修改源码后，运行：
    python scripts/package_rl_lab.py
    python scripts/package_rl_lab.py --check --test

第一条重建 ZIP；第二条逐文件核对一致性，再从临时目录解压运行测试和命令入口。
该维护脚本属于仓库，不在下载包内。独立 GitHub Actions 检查使用相同命令。
