# 两篇 Muon 文章验收记录

2026-11-03。在已部署提交 `70d583e3857fe88a3fac571c5cf41804064a9323` 上增加两篇通用文章，不给新文章添加应用场景标签。既有 96 条验收记录、原文章内容和学习路径内既有文章的顺序保持不变，见 `preservation.json`。

- `muon-optimizer`：谱范数约束、紧致极分解、有限步 NS、四种形状因子、学习率与衰减换算、Moonlight、参数分组、聚合顺序和 QK-Clip。
- `muon-learning-rate-batch-size`：局部最优步长、两次平均场近似与特殊曲率假设、平方根到饱和、各向异性与反例、动量、累计衰减、梯度累积、噪声估计和比较预算。

两幅概念图的内置 Imagegen 提示词、修订与文件哈希见 `../../image-generation/muon-articles-20261103.json`。其余六幅定量图由各文章的 `make_figures.py` 计算；图注区分构造值、局部模型与训练实证。图表原件、标签、箭头、数值和 PDF 页面已经目视检查，见 `visual-review.json` 和 `visual/`。

先用 `hugo server -D` 预览，再将两篇新文章设为公开。最终检查使用全新生产目录 `/tmp/chase-muon-articles-20261103-a3ush9p0/final-production`，由单独的 loopback HTTP 服务提供。浏览器串行连接私有 Chrome CDP 9244，不修改用户原有浏览器标签。

| 检查 | 最终结果 |
| --- | --- |
| Hugo 0.165.0 生产构建与严格验证 | 324 HTML、33,294 本地引用、592 验收检查；无错误与未渲染公式 |
| 数值例子 | 标准库运行 10 项通过、1 项可选跳过；加 NumPy 后 11 项全部通过 |
| 独立数值依据 | 65 位 Decimal、显式滤波权重、独立参数更新、穷举无偏性、固定网格 Simpson、高斯闭式二阶矩、NumPy SVD |
| PyTorch 单步示例 | 从发布 Markdown 提取并执行，PyTorch 2.14.1 CPU，参数有限 |
| 阅读与原生回退 | 48 组布局、104 个新文档片段入口；320/1440px、100%/200%、三主题、脚本开关均通过 |
| 图片查看 | 两篇所有图片，共 32 组手机/桌面、明/暗主题放大、焦点和位置恢复通过 |
| 键盘 | 16 组表格上下文，14 次实际方向键滚动；12 次精确异步复制；220 次章节导航通过 |
| 打印布局 | 100% 与 200% 各 12 组；三主题、脚本开关、公式/表格/代码边界及状态恢复通过 |
| 原生 PDF | 8 份 A4/Letter 导出均带标签、保留正文和焦点/位置，源 HTML 哈希见 `native-exports.json` |
| 独立 axe 4.14.0 | 16 组，0 违规、0 丢失 ID 引用；关闭弹窗的 incomplete 另行核对 |
| TOC、文章筛选、搜索、资产完整性 | 通过，见对应报告 |

这些检查验证文章公式、构造例子和阅读行为，不验证完整模型训练效果、GPU 数值稳定性、分布式优化器或普遍的最优 Batch 规律。CI 仅新增标准库与可选 NumPy 两条随文数学检查，不增加运行时依赖。

复算命令：

```bash
python3 -B scripts/test_muon_articles.py
python3 -B scripts/test_muon_articles.py --numpy
python3 -B content/posts/ai/muon-optimizer/muon_math.py
python3 -B content/posts/ai/muon-learning-rate-batch-size/batch_math.py
```

浏览器检查脚本位于本目录，参数依次为生产构建的 loopback URL、私有 CDP 端口、输出目录；`check_semantics.cjs` 还需传入固定版本 axe 文件。压缩报告保留完整布局和引用检查，8 份原生 PDF 留在上述临时工作目录的 `final-exports/`，可由 `export_article.cjs` 重新导出。
