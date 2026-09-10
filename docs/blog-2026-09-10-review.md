# 2026-09-10 全站回归与 ELMP 论文文章

本轮覆盖 72 篇源文章（含 3 篇草稿）的结构、图片、公式与导航检查。实质内容修订集中在 5 篇旧文和 1 篇新文；没有逐篇重写其余文章，也没有重新运行所有硬件或深度学习案例。

## 旧文修订

- PyTorch 运动学：修复坐标变换公式被 Markdown 块语法截断的问题。
- Jacobian、动力学参数辨识：修复独占一行的等号被识别为标题下划线，导致矩阵公式未渲染的问题。
- 阻抗控制：修复独立等号及行首空组上标与 Markdown 块属性的冲突，保留原数学含义。
- 强化学习：依据 PPO 原论文公式 7、9 修正裁剪项括号，统一最大化目标与最小化损失符号；区分采样 horizon 与 mini-batch，并删除“必须联合训练就能稳定”的过度表述。

## ELMP 新文

`content/posts/thesis/elmp/index.md` 单独成篇，加入“规划、轨迹与控制”学习路线与全站文章索引。解释工具点云、BC、APG/BPTT、SDF 梯度、训练和部署差异，以及表 I–IV 的指标。

原图为用户提供 PDF 第 3 页图 2 主体，提取时保留两阶段结构。页面中的来源 JSON 保存原文件 SHA-256、引用范围与提取区域。没有转载整份 PDF，没有把独立教学代码称为作者实现。

文章专门解释了旋转弦距离并非在 SO(3) 上全局严格凸、碰撞球半径的距离约定、CPU-hours 与并行墙钟时间，以及真机成功子集误差与仿真阈值不能混用的疑点。

## 验证

| 项目 | 结果 |
| --- | --- |
| 正式 Hugo 构建 | 通过，69 个发布文章结构化数据 |
| 含草稿构建 | 通过，72 个文章结构化数据 |
| 公式严格检查 | 正式与草稿均 0 未渲染公式警告 |
| 正式站内引用 | 14,726 处通过；不代表遍历验证所有外部网站 |
| 草稿站内引用 | 15,558 处通过 |
| 围栏代码 | 401 块；174 Python、124 Bash、8 sh、4 JSON、6 JSONC、1 XML 语法检查通过；没有执行正文 shell 命令 |
| 全站浏览器 | 72 篇，390/1440 px，图片、目录、导读、自测、专题导航通过，无 JS 异常 |
| ELMP 专项浏览器 | 390/1440 px × 浅色/深色，27 处公式，无横向页面溢出、坏图或失效页内锚点 |
| 筛选索引 | 69 篇正式文章、10 个专题，查询、排序、URL 状态与覆盖检查通过 |
| ELMP 原子练习 | 有限差分最大误差 7.030e-11；球半径、旋转导数与时间口径检查通过 |

截图保存在 `/tmp/chase-all-browser` 与 `/tmp/chase-elmp-browser`。长公式使用既有局部横向滚动，页面本身不横向溢出。

复验命令：

```bash
hugo --minify --destination /tmp/chase-all-production
python3 scripts/validate_blog.py --public /tmp/chase-all-production \
  --python-snippets --structured-snippets --strict-math
hugo --minify -D --destination /tmp/chase-all-preview
python3 scripts/validate_blog.py --public /tmp/chase-all-preview --include-drafts \
  --python-snippets --structured-snippets --strict-math
node scripts/test_blog_filter.cjs /tmp/chase-all-production
python3 -B content/posts/thesis/elmp/elmp_math_checks.py
# 在已启动预览与 Chrome 本地 CDP 后运行：
node scripts/check_blog_browser.cjs http://127.0.0.1:13157 9247 /tmp/chase-all-browser
```

没有修改主题、站点配置或部署工作流。没有安装 ELMP 依赖、训练神经网络或执行机器人实验。

## 后续整体阅读优化：长公式键盘访问

共享阅读脚本为实际溢出的公式增加可见操作提示、Tab 停靠点、区域名称与提示关联。公式能完整显示时移除提示和停靠点，避免所有公式都占用键盘导航。窗口调整、字体加载及折叠章节展开后重新测量宽度；使用 ResizeObserver 跟踪容器变化。关闭 JavaScript 时保留原有 CSS 横向滚动，不依赖脚本展示公式正文。

ELMP 原子教程补充了实际终端输出与断言失败排查方向，没有改变算法或数学结论。

更新 `check_blog_browser.cjs`，逐篇检查残留的显示公式分隔符及滚动公式的键盘入口；新 `check_blog_math.cjs` 实际发送方向键，验证滚动位移、390/1440 px 下提示和 Tab 停靠点切换、浅色/深色主题，以及折叠公式展开后更新。

验证结果：72 篇全站浏览器回归无失败、无 JS 异常；公式专项通过；生产构建与严格数学检查通过，14,726 处站内引用有效。ELMP 原子练习复跑通过。截图在 `/tmp/chase-reading-math` 和 `/tmp/chase-reading-browser`。

```bash
node scripts/check_blog_math.cjs http://127.0.0.1:13157 9247 /tmp/chase-reading-math
node scripts/check_blog_browser.cjs http://127.0.0.1:13157 9247 /tmp/chase-reading-browser
```
