Jev 决策层：离线门控与置信度路由算例
==================================

Python 3.10+；核心计算只用标准库，无 API 密钥、无网络请求、无机器人动作。
门控和路由的逐条记录均为人为构造，不能当作 Jev 准确率或机器人成功率。
结果内的文献重算项仅检验已报告聚合数字的算术，不复现原模型实验。

1. 验证决策接口和观测时效

   python -B decision_gate.py

   13 组检查覆盖无效响应、概率分布、观测年龄、独立请求预算、
   目标版本、提交前重检和 DONE / BLOCKED。输出 execute:... 仅为字符串。
   合成阈值和时限不是真机参数建议，也不构成安全控制器。

2. 复现阈值、成本和校准反例

   python -B routing_lab.py --output routing-results.json \
     --samples routing-fixtures.json --csv routing-fixtures.csv

   200 条记录分为验证 40、留出 80、分布变化 80 条；ID 不重复。
   只在验证组、固定候选阈值上选择阈值，结果为 0.8。
   接受部分的错误率分别为 5%、10%、25%。分布变化组阈值为 0.9 时为 40%。
   阈值为 1 时无人接受，错误率未定义，JSON 中使用 null。

   调用成本单位是任意尺度，不是美元。每条先支付选择器费用，
   拒绝后再支付上层费用；不含执行、重试或上层结果，不能推导最终任务成功率。
   二元概率校准与 Wilson 区间是独立算例，不是路由分数的概率解释，
   更不是构造记录或 Jev 的统计保证。

3. 重画曲线（可选）

   仅此步骤需要 Matplotlib；在单独环境中安装即可。
   python -B routing_lab.py --figure routing-tradeoff.png

   发布曲线由 Python 3.10.0 / Matplotlib 3.10.9 生成。
   绘图文件的字体或压缩可能随环境变化，核心计数和公式不依赖 Matplotlib。

assets/ 保存文章发布时的固定结果、逐条 JSON/CSV 与曲线。
上述命令默认不覆盖 assets/；可用自己的输出文件逐项核对。

文章： https://chase6305.github.io/posts/ai/jev-decision-layer/
