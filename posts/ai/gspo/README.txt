GSPO CPU 教学实验（不是 Qwen3 / MoE 复现）

Python 3.10+。在解压后的 gspo-lab 目录执行：
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
  python -B -m unittest -v test_gspo_lab.py
  python gspo_lab.py --output results.json
  python gspo_update.py --output update-results.json
  python gspo_online.py --output online-results.json

默认在线实验：seed 0/1/2 × beta 0/1；20 轮，每轮 8 组×8 条，3 次更新。
每次运行采样 1280 条回答 / 2560 token，共 60 次更新。
  python gspo_online.py --help
  python gspo_online.py --seeds 7 --betas 0.2 --rounds 10 --groups 8 --group-size 4 --updates 2 --learning-rate 0.1 --epsilon 0.1 --output custom-results.json

可选绘图（只读结果，不训练）：
  python -m pip install matplotlib==3.10.6
  python plot_results.py --input results.json --output figures
  python plot_online.py --input online-results.json --output online-training.png

assets/ 保留文章生成时的参考 JSON；请将自己的输出写到其他路径。
浮点末位及随机采样跨 PyTorch 版本可能变化。先检查测试、配置和预算，
不要要求不同环境下每个 JSON 字节完全相同。

gspo_lab.py：固定 logps，比较目标和梯度。
gspo_update.py：枚举均衡批次，展示单批更新与裁剪平台。
gspo_online.py：随机采样、多轮刷新 old、固定 reference 的精确玩具 KL。
三个脚本使用同一组目标函数，但验证的是不同问题。
