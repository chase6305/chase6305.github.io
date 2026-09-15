# X-VLA 与 RTC 发布前核对

核对日期：2026-09-12。范围为两篇文章、随文图片与教学实验、学习路径和下载包检查工作流。

## 内容与来源

- X-VLA 源码固定为 `6bc2513f5f1cbec715cc668b414392a6cae5c671`，论文为 `2510.10274v1`。
- RTC 源码固定为 `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`；三篇 PDF 分别为 `2506.07339v2`、`2512.05964v2`、`2608.01880v2`。
- 四篇论文的 arXiv 标题、作者与版本日期已核对。源码链接的文件与行号在固定版本中检查。
- 论文报告值、代码行为、本文数学推导和教学结果在正文中分别标注。未运行完整 VLA 训练、模型服务或机器人闭环，不将 CPU 算例当作论文任务复现。
- 两篇文章分别加入 AI 学习路径，文章清单同步保存阅读元数据及验收问题。

## 图片

- X-VLA：5 张 imagegen 概念图与 1 张由精确时间坐标生成的 SVG。
- RTC：2 张 imagegen 概念图与 3 张由公式生成的 SVG。
- 新增 WAM 视频／动作联合预测示意，明确未来预测画面不代替传感器观测。
- imagegen 提示和图片用途分别记录在 `x-vla-image-prompts.json` 与 `rtc-image-prompts.json`。
- Raster 图片继续使用站点已有的响应式 WebP 生成流程，未改动主题子模块。

## 实验与下载包

两个 ZIP 共包含 4 个教学程序。已在新建的 Python 3.10 虚拟环境中安装 NumPy 2.2.6，并从临时解压目录运行全部程序。

```bash
python3 scripts/package_xvla_rtc_labs.py --check --test
```

修改随文程序或包内说明后，重新执行不带 `--check` 的打包命令。检查比较成员列表与解压内容，不依赖不同 zlib 版本产生完全相同的压缩字节。已验证它会拒绝源码过期及额外成员，同时接受压缩方法不同、解压内容相同的包。

`.github/workflows/xvla-rtc-labs.yml` 沿用现有实验包工作流的结构：只读仓库权限，安装固定依赖，检查包与源码一致，再运行解压后的程序。该工作流不运行机器人模型，也不单独部署站点。

## LIBERO 图像处理对照

模型配置固定为 `2toINF/X-VLA-Libero` 的 revision `129e71460678b7236cee6fc9707f09d9fa0c3590`。使用其 `preprocessor_config.json` 构建 `CLIPImageProcessor`，对比固定 X-VLA 读取器关闭 ColorJitter 后的变换。

- 4 个合成 RGB 输入：黑、白、三通道渐变、高频棋盘格，覆盖横图、竖图及非整齐尺寸。
- 两路输出均为 `(3, 224, 224)`；本次最大绝对差均为 0。
- 修改均值或将 bicubic 改为 bilinear 的对照会被检查程序拒绝。
- 环境与完整结果位于 `content/posts/ai/x-vla/processor-check.json`；程序为同目录 `processor_check.py`。
- 这一可选检查使用额外的 PyTorch、torchvision、Transformers 与 Pillow，未加入轻量 NumPy 实验包。它没有验证 HTTP、tokenizer、多视角打包或策略输出。

## 站点验证

发布前使用新的临时输出目录，避免已有 `public/` 混入旧版页面和开发预览的 LiveReload 引用：

```bash
xvla_rtc_build_dir="$(mktemp -d /tmp/xvla-rtc-production.XXXXXX)"
hugo --minify --destination "$xvla_rtc_build_dir"
python3 scripts/validate_blog.py \
  --public "$xvla_rtc_build_dir" \
  --python-snippets --structured-snippets \
  --strict-math posts/ai/x-vla/index.html \
                posts/ai/real-time-chunking/index.html
```

已检查文章清单、学习路径、中文结构化元数据、本地引用、代码片段语法与公式。浏览器检查覆盖两篇文章各 12 组宽度／主题组合、11 张图片、图片缩放、手机菜单和下载入口；另外检查文章筛选与学习路径导航。完整模型训练和论文性能不在这些站点检查的覆盖范围内。

站点通过推送 `main` 触发已有 `hugo.yml` 工作流部署 GitHub Pages。生产构建输出不加入 Git 提交。

## 2026-09-15：训练预算与学习率补充

- X-VLA 第 8.7—8.9 节增加 `iters` 与余弦终点的关系、框架接口对照、checkpoint 比较及拟议的独立衰减参数；数据来源文章第 12.7 节增加交叉引用。
- 调度公式同时核对官方固定提交与本地 `xvla2/models/xvla/train_xvla_bf16.py`；后者文件 SHA-256 为 `df74b3416ea3ab2095ccdb4823bbc3a17fd4c6a0e69d130a4bd8774409ee1099`。官方默认关闭余弦，所核对派生入口默认开启，正文明确区分。
- 从两份源码提取纯调度函数，检查冻结、warmup、50k／100k、衰减终点及超出终点的取值。10 万步与 20 万步计划在 50k／100k 的数值与表格、绘图函数一致；补充调度输入从 0 开始、checkpoint 按已完成更新数编号的边界。
- 新增精确 SVG 学习率曲线及可下载的 `lr_schedule_plot.py`，依赖 Matplotlib；图注注明它是公式曲线。该可选绘图程序独立于原轻量实验 ZIP，原包内容未变。
- 框架依据采用官方 Transformers v4.57.1 Trainer 源码、调度接口文档、PyTorch 2.9 文档、Megatron-LM 配置源码及 Bridge 配置文档。另以 PyTorch 调度器验证超过 `T_max` 后会回升，避免将 X-VLA 的最低值截断推广到其他框架。
- 新增三个小节及公式、三张表格、图 7，在 320／390／768／1440 像素宽度检查，覆盖浅色、深色和暖色主题。检查无页面横向溢出、公式错误、断裂页内锚点或浏览器异常；宽表格保留可聚焦的横向滚动区域，图片与脚本链接可访问。
- 使用新的临时目录完成 Hugo 生产构建，并运行文章清单、Python／结构化代码片段、本地链接与两篇文章的严格公式检查。本轮未运行策略训练，不将学习率算例作为任务成功率证据。
