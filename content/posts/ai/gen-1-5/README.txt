GEN-1.5 阅读配套统计练习

这是自拟算例，不包含 GEN-1.5 权重、SDK、推理或机器人控制。
仅需要 Python 3.10+ 标准库，无需安装第三方依赖。
在解压后的 gen-1-5-lab 目录执行：
  python3 -B -m unittest -v test_prompt_eval.py
  python3 prompt_eval.py

程序打印 JSON。assets/example-results.json 为文章使用的参考结果。
四项练习：窗口预算、任务等权/试验等权平均、Wilson 区间、配对结果。
自拟结果应包括：
  used_seconds=26，free_seconds=4，nominal_action_samples=2600
  macro=0.7，micro=19/30
  paired: both=2，left_only=3，right_only=1，neither=2
  success_rate_difference=0.25（25 个百分点）

调用 paired_outcomes 前，要在实验日志中验证两侧对应同一配对场景；
函数只能检查长度和布尔结果，不能根据结果本身证明配对正确。
Wilson 区间假设独立同分布的二项试验；不要用本例补造官方试验次数。
