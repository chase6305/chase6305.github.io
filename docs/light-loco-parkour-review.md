# Light-Loco-Parkour 文章核对记录

核对日期：2026-09-12。文章路径为 `content/posts/ai/light-loco-parkour/index.md`。

## 来源与范围

- 用户指定的本地仓库对应 `lucidrains/light-loco-parkour`，固定提交 `963a6ec3b8b42eb29dd6b9dfed34ededd3c64c7b`，包版本 0.0.29。分析前后均确认第三方工作区干净，未修改其实现。
- 原论文为 arXiv `2608.02653v1`（2026-08-01）；阅读了完整 PDF，直接检查 Table V 页图及 Table VII 的评测定义。本文没有复制论文 PDF 或原图到站点。
- 明确区分论文完整系统、WIP 第三方组件、独立教学算例及源码实测。原始参考增强、IsaacLab 场景、机器人模型与部署适配器不包含在当前第三方实现中。
- 引用的多 critic、SPR 与 NextLat 工作是相关研究或仓库扩展来源，没有把它们的全部机制归入原论文。

## 环境与执行

使用独立 Python 3.10.0 CPU 虚拟环境。主要依赖：PyTorch 2.9.0+cpu、NumPy 2.2.6、assoc-scan 0.0.5、mean-conc-beta 0.0.8、torch-einops-utils 0.1.25。完整依赖为随文 `source-constraints.txt`；`pip check` 通过。

- `tests/test_agent.py` 与 `tests/test_next_latent_prediction.py`：33 passed。
- 全量 `--collect-only`：33 tests collected, 1 error；`tests/test_light_loco.py` 导入不存在的 `validate_transition`。没有补造模块或将这次结果报告为全套通过。
- `source_probe.py`：固定源码上的 CPU 接口探针通过，结果为随文 `source-results.json`。
- `parkour_lab.py`：独立标准库数学练习通过，结果为 `teaching-results.json`。
- `make_figures.py --check`：4 张 SVG 与生成器一致。
- 固定 Pendulum 脚本执行 3 轮、每轮 256 步、2 个并行环境、每轮 1 个 PPO epoch，batch_size 64、use_rnn True。平均 episode reward 为约 −1251.28、−1247.76、−1197.08；仅为入口 smoke check，没有宣称收敛。

## 重点核对

- 脚部超限加速度累积量非负；源码默认权重 +0.01，论文 Table I 为 −0.01。给出显式覆盖示例。
- `calc_gae` 返回价值目标；终止传播 mask 与有效样本 mask 分工不同。
- 蒸馏按动作维平均，样本权重后按有效位置平均，不按权重和归一化；验证 14 与 7 的区别。
- 堆帧只针对当前调用；单步保留 GRU 隐藏状态不等于自动保存原始帧窗口。
- `Actor` 的潜变量目标默认 None 为自动启用，False 才关闭；与原论文扫描重建区分。
- 不同阶段训练先验不等于部署 actor 切换；检查严格触发边界、平滑权重顺序要求与共享判别器行为。
- 随机张量 e2e 测试中，直接减去 detached AMP 标量不会给策略提供该项梯度；需通过奖励和优势接入。
- 论文 75 cm 攀爬教师 98.6%、深度学生 33.4%；98% 衔接是独立评测口径。按原表报告 99.9 等值，不反推整数试验记录或置信区间。

## 图像与站点

imagegen 概念插画生成后去掉边角装饰文字，保留通用人形机器人接触姿态及单策略信息流；不是 Lightbot 0 实机照片。提示与修订记录见 `light-loco-parkour-image-prompts.json`。

四张 SVG 用标准库生成：训练与部署关系、脚部冲击记忆、阶段先验权重、论文师生性能差距。精确曲线及表格值与正文对应。

先通过 Hugo 草稿预览，再更改发布状态。浏览器覆盖 320、390、768、1440 像素宽度和浅色、深色、暖色三主题，共 12 组，检查 5 张图、80 处公式、目录链接、图片缩放、手机菜单及下载。生产构建使用独立临时目录，避免已有 public/ 中的历史开发产物影响检查。

新增 `light-loco-lab.yml` 只运行标准库教学练习和 SVG 一致性检查，不运行第三方模型训练。GitHub Pages 继续使用原有 `hugo.yml` 部署。

最终生产检查覆盖 75 篇审阅记录、72 篇已发布文章结构化数据、237 个 HTML 页面及 16,471 处本地引用；未发现错误、未渲染公式或 Python 片段告警。

站点发现性检查通过：72 篇已发布文章、14 组筛选状态，以及搜索分享链接、输入法、历史导航、无 JavaScript 回退、键盘操作和新标签页。
