# 三篇 AI 文章配图与复核记录

日期：2026-10-14。使用内置 Imagegen 生成与编辑；完整提示词和编辑指令见同目录 `three-ai-articles-2026-10-14.json`。

视觉参考：`content/posts/ai/transformer-attention/assets/` 的 training-lifecycle.webp、transformer-block-overview.webp、training-vs-kv-cache-inference.webp。沿用白底、圆角细线、蓝色数据、紫色模型、橙色动作或不确定量、绿色环境或结果。手部左右侧另用蓝／橙区分，质量可靠性只在质量矩阵内以绿／橙表示。

## Behavior Prompting

- 资产：`content/posts/ai/behavior-prompting/assets/behavior-prompting-flow.webp`。
- 依据：[论文 v2，第 3.2 节与图 2](https://arxiv.org/html/2606.30457v2#S3.SS2)。
- 最终源图：`exec-e4c5dd60-a321-4b62-809c-5ef7f94ad605.png`。
- 复核：示教片段汇总为提示 token；当前观测提供 Query，提示提供 K/V；当前状态也进入动作解码器。动作执行形成观测闭环。测试固定权重，提示嵌入可缓存。
- 省略：视觉编码器、扩散时间条件、网络层数、动作执行窗口。示教缩略场景只用于表示输入形式。
- 修改：将 K/V 箭头起点改到提示 token，补入当前状态到解码器的直接分支。

## StableHand

- 资产：`content/posts/ai/stable-hand/assets/stablehand-pipeline.webp`。
- 依据：[论文 v1，第 3 节与图 2](https://arxiv.org/html/2605.18553v1#S3)。
- 最终源图：`exec-b50c6c00-0991-43d7-a66b-5b5ec7bb66a3.png`。
- 复核：噪声与观测分别输入质量感知初始化；观测另作为 DiT 条件；四通道质量同时作用于初始化与生成器。输出为 T×108 双手状态，不是机器人命令。
- 省略：QN 检测标志与相机输入、误差到质量的标定、ODE 步数、损失项。质量矩阵无实测数值；绿色／橙色只示意可靠性差异。
- 修改：删除噪声进入观测框的错误箭头，将噪声连至初始化；把 Observation condition 的起点修正为观测框。

## 视觉 Transformer

- 资产：`content/posts/ai/vision-transformers/assets/vision-transformer-map.webp`。
- 依据：文章引用的 ViT、Swin、CMT、MAE、DINO、CLIP、DiT、SiT 原论文；表示可组合的设计维度，不是模型谱系。
- 最终源图：`exec-ecec89e5-1b77-493a-ab28-941f9c88d97c.png`。
- 复核：骨干／目标／任务分列，无一一对应箭头；时间条件进入生成骨干；采样重复更新潜变量，最终 VAE 解码。
- 省略：采样器更新作为连线的一部分，CFG、VAE 训练、文本编码器。分类小数为输出形式示例；采样下标表示相邻步，不限定时间参数化。

所有源图位于 `/home/ubuntu/.codex/generated_images/01a138db-b2b2-7270-9a61-96b20b62b2fa/`，最终资源以 WebP quality=92 无缩放转换。原 SVG 保留；文章引用更新为 WebP。转换只涉及存储格式，图中内容通过 Imagegen 修改。

## 集成验证

- Hugo 0.165.0 生产构建通过；最终输出为全新的 `/tmp/three-ai-articles-release-20261014/`。
- 发布校验开启 Python 与结构化代码片段检查，无错误。
- 文章筛选测试通过。
- 三篇 × 320/1440 像素 × 明/暗主题共 12 组浏览器检查通过，包括加载、图片尺寸和响应式资源、缩放、公式解析、目录目标、页面溢出、完整图注，以及窄屏表格焦点和方向键滚动。
- 浏览器报告与截图：`/tmp/three-ai-articles-browser/`。已人工查看 BPP 窄屏明色、StableHand 窄屏暗色、视觉 Transformer 桌面截图。
- 修正 BPP 与 StableHand 表格属性前的空行，使 `table-readable` 实际应用；BPP 两张长文本对照表也启用该属性。保留紧凑数字表的原有设计。

## 最终发布复核

- 明确 BPP 图 1 中示教片段先编码缓存、当前观测再查询的先后关系，并补全当前状态进入动作解码器的说明。
- 视觉 Transformer 快速目录增加自测问题与答案入口；保持既有章节标题与锚点。
- 学习路径将“视角运动恢复”明确为“第一视角手部运动恢复”。
- 使用 Hugo 0.165.0 构建到全新 `/tmp/ai-articles-deploy-20261014`；严格公式、Python 和结构化片段发布校验无错误、无警告；筛选与目录测试通过。
- 第一张与第二张图各完成三篇 × 两种宽度 × 两种主题的检查，共 24 组，覆盖图像加载、缩放、尺寸、图注、公式、章节链接、页面溢出与窄屏表格键盘滚动。
- 浏览器结果：`/tmp/three-ai-articles-round3-browser/results.json`、`/tmp/three-ai-articles-final-second-figures/results.json`。
