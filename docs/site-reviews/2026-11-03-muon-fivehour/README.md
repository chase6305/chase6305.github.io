# Muon 两篇文章的持续修订记录

2026-11-03。基线提交为 014ccd5e05c94eb9a3938bc55a6ccfc113687d03。按用户追加的 3 小时继续复核；本记录先保存已完成的验收快照。本轮完善已有的 Muon 原理与学习率/Batch Size 两篇文章；保留公开地址、原章节锚点、发布日期、draft 状态与标签，不增加应用场景标签。其他 96 条文章验收记录保持不变，见 preservation.json。

## 讲解与复现

- 原理篇补充瞬时 Shampoo 恒等式与历史累积反例、旋转矩阵与逐元素符号的区别、两种 ε 分母、RMS 算子范数、NS 的算术成本与内存、Adam 更新 RMS 的近似来源。
- Float64 矩阵例子逐项返回动量、Nesterov 输入、谱变换、尺寸因子与衰减；SVD 数值秩与精确实数秩分别说明。归一化避免原值平方引起的上溢、下溢。
- PyTorch 2.14.1 CPU 回归例子按模块角色分配 Muon/AdamW，按 13+13+6 归约 Batch 32，保存两套状态与采样位置。检查真正的新 Python 进程恢复，并单独展示丢失 Muon 动量的结果；不进行优化器性能排名。
- Batch 篇补充精确二次步长例子、普通 EMA 方差与旧状态记忆的不同匹配、加权/相关样本的有效数量、精确一维高斯符号期望、矩阵 Gram 近似与实际 SVD 均值、旧状态记忆与新梯度时间权重的区别，以及固定成本下的局部收益率。
- 随机矩阵实验在固定参数位置，对两种信号谱分别做 20 万次独立高斯 SVD。误差条仅为均值的近似蒙特卡洛区间。
- Batch 交互换算保留三个不同目标：条件平均场学习率、旧动量保留比例、每步或累计权重收缩。输入修改后清除旧结果，脚本不可用时显示静态默认值，延迟增强保留当前阅读位置。
- 两篇均增加阅读指引、折叠自测答案和完整下载包。ZIP 在独立解压目录执行检查，并拒绝用父目录或 PYTHONPATH 中的文件补齐缺失成员。

## 验证快照

| 检查 | 结果 |
| --- | --- |
| 全新 Hugo 生产构建与严格验证 | 324 HTML、33,341 本地引用、604 验收检查；无错误、未渲染公式或片段语法警告 |
| 标准库与 NumPy | 25 项注册；无 NumPy 时 14 项通过、11 项明确跳过；NumPy 环境全部通过 |
| PyTorch CPU | 8 项通过，含实际子进程恢复、已有文件保留、不等长 microbatch 解析梯度和固定源码哈希 |
| 无 NumPy 的 PyTorch 环境 | 8 项训练检查与独立解压检查通过；PyTorch 自身的可选 NumPy 提示保留在日志中 |
| JavaScript 算术 | 与独立 Python 的 207 组输入、边界与分量不变量一致 |
| 定量图 | 7 幅在独立解压目录重画，固定绘图环境下逐字节一致 |
| 整篇阅读 | 48 组布局、140 个片段入口；320/1440px、100%/200%、三主题、脚本开关通过 |
| 图片 | 全部 10 张，40 次手机/桌面、明/暗主题查看及焦点/位置恢复通过 |
| 正文键盘 | 24 组表格上下文，18 次实际方向键滚动、32 次精确异步复制、248 次章节跳转通过 |
| 计算器重排 | 48 组，含放大文字和额外字间距；11 项行为、3 项回退通过 |
| 计算器键盘 | 360 个焦点样式、36 组 Tab 正反进出；最低轮廓对比度约 4.94:1 |
| 计算器原生 PDF | 20 份 A4/Letter，含待计算、无效、无脚本与强制颜色；140 组定义标签和值保持同页 |
| 整篇打印布局 | 普通/放大文字、普通/强制颜色各 12 组，共 48 组；公式、代码、表格边界和状态恢复通过 |
| 整篇原生 PDF | 8 份带标签导出，包含新公式、代码、图注与正文，打印后保留焦点、位置和折叠状态 |
| axe 4.14.0 | 18 个状态，0 违规、0 丢失引用；incomplete 仅涉及关闭的菜单/弹窗，另以引用和实际交互检查 |
| 整站放大与可读性 | 1,962 组重排、312 组对比度、263,409 个文字样本、18 组打印；无失败 |
| 新补充公式键盘 | 4 个实际方向键滚动上下文、4 个无需横向滚动的上下文，放宽后保留当前焦点 |
| 全站轮廓回归 | 702 个正常/强制颜色焦点样式通过 |
| TOC、筛选、搜索、资产完整性 | 通过；筛选与资产检查使用全新生产目录 |

工具版本见 tool-versions.json，一手资料与实现哈希见 sources.json。概念图使用内置 Imagegen；提示词、来源、尺寸和哈希保存在 ../../image-generation/muon-fivehour-20261103.json。定量图采用固定 NumPy/Matplotlib 环境；构造数值与训练实证在正文中分别注明。选取的 PDF 页面和屏幕样例位于 visual/。

## 复算

标准库：

    python3 -B scripts/test_muon_articles.py
    node scripts/test_muon_batch.cjs
    python3 -B scripts/package_muon_examples.py --check --test

隔离 NumPy 环境：

    python -B scripts/test_muon_articles.py --numpy
    python -B scripts/package_muon_examples.py --check --test --numpy

文章的 PyTorch 2.14.1 CPU 环境：

    python -B scripts/test_muon_training.py
    python -B scripts/package_muon_examples.py --check --test --numpy --torch

浏览器驱动位于本目录；参数依次为新生产构建的 loopback URL、私有 CDP 端口、输出目录。check_semantics.cjs 最后再传入固定版本的 axe 文件。浏览器套件串行执行，不使用用户已有标签页。check_batch_print.cjs 调用 inspect_pdf_cards.py 检查真正 PDF 的文字边界；不是仅检查屏幕截图。

本轮生产输出位于 /tmp/chase-muon-fivehour-20261103-7235a3f5/final-production，原生 PDF 保存在同一临时工作目录，未把生成站点或完整 PDF 提交到仓库。压缩报告保留检查细节；文件哈希与来源由 files.json、plot-reproduction.json、native-exports.json 和 visual-review.json 记录。

这些检查验证公式、构造分布、示例集成和阅读行为，不验证 GPU/分布式训练、真实样本的独立性、自动噪声尺度识别、通用最优学习率或完整训练效果排名。
