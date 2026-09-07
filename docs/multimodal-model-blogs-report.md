# RynnBrain 与 InternVL 3.5 新文章交付记录

核对日期：2026-09-05。内容、来源版本、完整 imagegen 提示词及生成图片路径见 [multimodal-model-blogs.json](multimodal-model-blogs.json)。

下方保留首轮交付的测试计数；后续内容与计数变化记录在文末“持续优化记录”，不回写为首轮结果。

## 内容与范围

- [RynnBrain](../content/posts/ai/rynnbrain/index.md)：14 个主体章节，另附自测与参考材料。以当前 1.1 为主线，保留 1.0 的 CoP 历史背景；覆盖空间协议、接触点、3D、跨本体 VLA、训练和机器人评测边界。
- [InternVL 3.5](../content/posts/ai/internvl-3-5/index.md)：16 个主体章节，另附自测与参考材料。覆盖模型集合、动态分辨率、视觉 token、MPO/GSPO、ViCO/ViR、DvD、显存与加载路线。
- 两篇文章均先以草稿预览并检查页面，再设为 `draft: false`；接入学习路线、相关文章、归档与检索。
- 原 66 条文章记录与原草稿状态保持不变；当前清单为 68 篇，正式发布 65 篇，原有草稿 3 篇。
- 未修改主题子模块、工作流或站点配置，未提交或推送本次改动，未执行部署脚本。

## 配图

使用内置 `image_gen`，延续 `content/posts/ai/transformer-attention/assets/transformer-block-overview.webp` 的浅色圆角技术图风格。生成 PNG 经格式转换保存为 WebP，三张合计约 340 KiB；完整提示词与编辑提示词保存在 JSON 清单。

| 文件 | 图意与人工复核重点 |
| --- | --- |
| `content/posts/ai/rynnbrain/assets/embodied-foundation-model.webp` | 基础模型输出与 VLA 动作策略分开；3D 的 2B/9B 范围单独标注 |
| `content/posts/ai/internvl-3-5/assets/visual-token-pipeline.webp` | tile、ViT patch、1024→256 与 Flash 的 256/64 分支分开 |
| `content/posts/ai/internvl-3-5/assets/cascade-rl-training.webp` | CPT→SFT→MPO→GSPO；再到 ViCO→ViR→Flash；HF 不是训练阶段 |

训练图首版出现 GSPO 直接连到 Flash 的误导箭头，已重新编辑并复核最终顺序。所有配图均明确标为教学示意，不作为论文原图、模型输出或实验结果。

## 构建与静态验收

Hugo `0.165.0`；普通生产构建 `hugo --minify` 成功，另使用独立输出目录检查正式版和含草稿版。

| 检查项 | 正式版 | 含草稿版 |
| --- | ---: | ---: |
| BlogPosting 元数据 | 65 | 68 |
| 校验 HTML 页面 | 203 | 213 |
| 校验站内引用 | 12879 | 13386 |
| 校验集合页面 | 193 | 203 |
| 文章筛选索引 | 7 | 7 |
| 错误 / Python 片段警告 | 0 / 0 | 0 / 0 |

文章源文件共检查 68 篇、10 条学习路线、139 条阅读验收题、348 个代码块；语法检查覆盖 Python、Bash/Sh、JSON/JSONC 和 XML。

可重复命令，均从仓库根目录运行：

```bash
hugo --minify
hugo --minify --destination /tmp/chase-model-blogs-production --baseURL http://127.0.0.1:13146/
hugo -D --minify --destination /tmp/chase-model-blogs-validation-drafts --baseURL http://127.0.0.1:13145/
python3 scripts/validate_blog.py --public /tmp/chase-model-blogs-production --python-snippets --structured-snippets
python3 scripts/validate_blog.py --public /tmp/chase-model-blogs-validation-drafts --include-drafts --python-snippets --structured-snippets
node scripts/test_blog_filter.cjs /tmp/chase-model-blogs-production
node scripts/test_blog_filter.cjs /tmp/chase-model-blogs-validation-drafts --include-drafts
python3 -B scripts/test_model_blog_examples.py
git diff --check
```

## Python 与 Processor 验证

`scripts/test_model_blog_examples.py` 的 10 个测试组通过，包括两篇附带脚本的 17 个基础算例、100 组相机投影往返、token 计数性质、非法输入拒绝、CLI 参数处理和固定 revision 检查。

另外在两个独立临时环境运行 `scripts/check_model_processors.py`，复用 PyTorch `2.8.0+cu128` 运行时，但输入处理仅使用 CPU。测试直接调用文章推理脚本的 `build_inputs`，不是另一份未关联的伪造输入实现。

| 模型 | Transformers | 原生实现 | 真实输入形状 |
| --- | --- | --- | --- |
| RynnBrain1.1-2B | 5.2.0 | Qwen3_5ForConditionalGeneration | ids `[1,1526]`；pixels `[6032,1536]`；grid `[1,3]` |
| InternVL3_5-8B-HF | 4.55.0 | InternVLForConditionalGeneration | ids `[1,2319]`；pixels `[9,3,448,448]` |

固定测试图片是上述 Transformer 结构图，提示词为 `Describe the diagram.`。检查浮点张量有限，并验证 RynnBrain 关闭 Thinking 时的模板前缀。RynnBrain 的模板渲染与图像处理拆分，避免 5.2.0 将模板专用参数继续传给 Processor 的警告。

命令分别在文章指定的环境中运行：

```bash
python -B scripts/check_model_processors.py rynnbrain
python -B scripts/check_model_processors.py internvl-3-5
```

以上两条命令只下载固定版本配置、Tokenizer 与 Processor 文件；解析原生模型类但不实例化模型参数。没有下载模型权重、执行模型前向、训练模型、测量 GPU 性能或运行真实机器人。CPU 检查通过不构成能力复现；完整权重推理脚本仍需用户在兼容硬件上验证。

## 浏览器验收

使用 Chrome 的回环 CDP，先启动 Hugo 草稿预览与正式版静态服务。UI 套件在同一浏览器上串行运行，避免共享键盘焦点干扰；页面检查脚本使用自己新建的标签页，不导航现有标签页。

- 全 68 篇：390 / 1440 宽度、图片、KaTeX、移动目录、关联阅读、结构化数据及学习路线检查通过。
- 新增两篇：320 / 390 / 768 / 1440 × light / dark / warm，共 24 种组合检查通过；另保存标题、配图和公式截图并人工复核。
- 交互：图片放大、菜单、代码复制与拒绝权限降级、键盘访问、站内检索（含 RynnBrain 与 InternVL）通过。
- 检索页面：正式与草稿清单分别验证跨页查找、URL 状态、IME、排序、无 JavaScript 回退、非法元数据回退和集合导航。

复测命令按顺序执行：

```bash
node scripts/check_blog_browser.cjs http://127.0.0.1:13145 9230 /tmp/chase-model-blogs-browser
node scripts/check_blog_discovery.cjs http://127.0.0.1:13146 9230 /tmp/chase-model-blogs-discovery-production
node scripts/check_blog_discovery.cjs http://127.0.0.1:13145 9230 /tmp/chase-model-blogs-discovery-drafts --include-drafts
```

截图及原始浏览器结果保存在上述三个临时目录，不计入站点内容。`public/` 是忽略的生成目录；未加入待提交文件。

## 持续优化记录：2026-09-05

本轮仅继续打磨两篇新增模型博客；原有文章状态保持不变，三张教学配图保留，没有重复生成装饰图。

### 内容改进

- RynnBrain：增加主动旋转/列向量约定、`Rz @ Ry @ Rx`、八角点构造、相机投影和完整数值算例；补充缩放裁剪与相机内参的对应关系，区分演示视场角与真实标定。
- InternVL：将固定 8B-HF 配置对应到原生代码的逐层张量形状；解释 CLS 去除、空间/通道重排、视觉占位替换；用四种正负优势情况解释 GSPO clipping。
- 两篇：增加分层阅读跳转、CPU 预检查入口和故障排查表。将新增长公式拆成多行，减轻手机阅读时的横向滚动。
- 新增两条验收题，当前全站 68 篇共 141 条。

### 实际代码改进与测试边界

`infer_image.py --prepare-only` 输出真实 CPU 输入的 JSON 报告；`--local-files-only` 与 `HF_HUB_OFFLINE=1` 可验证已缓存文件的无网络路径。两个脚本均固定原来的模型 revision 和 Transformers 版本。

因为部分上游图像加载路径会执行 EXIF 旋转，本例明确拒绝非 1 的 EXIF 方向，要求先规范化图片与坐标标注/相机约定。不会静默旋转后继续报告原始坐标系。

检查脚本的 `--exercise-cli` 路径会禁止模型加载器被调用，并验证：

1. 无 CUDA 时仍能完成 prepare-only；图像仍是固定 Transformer 图，输入分别为 1526 / 2319 token。
2. 输入上限设为 1 时，在加载权重前拒绝。
3. 不可解码图片、非标准 EXIF 方向、错误库版本，在调用 Processor 加载器前拒绝。
4. 正常推理模式在缺少 CUDA 时提前失败，不自动下载权重后才报错。

EXIF 和库版本异常使用受控替身构造；有效输入使用真实 Processor。未执行模型前向、权重推理、GPU 吞吐或真实机器人实验。

标准库测试扩展为 13 组，含 25 个基础算例、100 组相机投影往返、100 组旋转矩阵正交与行列式检查，以及旋转次序、八角点、clipping 符号与溢出拒绝测试。

在各自已安装匹配依赖并缓存小文件的环境中复测：

```bash
python3 -B scripts/test_model_blog_examples.py
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python -B scripts/check_model_processors.py rynnbrain --local-files-only --exercise-cli
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python -B scripts/check_model_processors.py internvl-3-5 --local-files-only --exercise-cli
```

### 构建隔离与浏览器检查

`hugo --minify` 构建通过。不过默认 `public/` 留有历史预览的草稿和 LiveReload 页面，因此没有用该混合目录的校验结果宣称生产验收通过，也没有删除用户的历史产物。正式验收使用新建目录 `/tmp/chase-model-blogs-refinement-production.3QnL6H`；草稿预览使用 `/tmp/chase-model-blogs-refinement-preview`。

本轮静态检查覆盖 354 个代码块，其中 Python 151 个、Bash 104 个；两种构建的站内链接、目录锚点与文章清单检查通过。正式文章仍为 65 篇，含草稿为 68 篇。

浏览器结果保存在 `/tmp/chase-model-blogs-refinement-browser`：全 68 篇页面检查、新文章 24 种宽度/主题组合，以及新增六个细节章节的 12 张手机/桌面截图。检查范围包括新公式、张量表、CPU 示例、图片加载与目录导航。
