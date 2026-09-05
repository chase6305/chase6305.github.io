# 博客第二轮优化记录

后续更新见 [第三轮优化记录](blog-third-pass-report.md)，本文件保留第二轮完成时的历史范围与验证数字。

日期：2026-09-05。延续第一轮修改，未提交、推送或部署；保留原 URL、发布日期及 3 篇草稿状态。

## 全站改进

- 66 篇正文均新增“阅读自测与验收”，共 132 条针对具体主题的检查建议，覆盖输入假设、结果验证、故障边界与适用条件。逐篇清单见 [blog-second-pass.json](blog-second-pass.json)。这些是读者的验收指引，不冒充已经完成的实机实验。
- 窄屏与平板增加原生折叠目录，选择章节后收起，并把键盘焦点交给标题；桌面继续使用原侧栏目录。
- 长代码块限制显示高度，支持水平/垂直滚动与键盘访问；溢出的数据表可以通过键盘聚焦滚动。
- 触屏和键盘操作时显示复制按钮。复制改为读取 textContent，保留原代码空行和多行字符串，不再通过删除双换行修正文本。
- 为博客输出 BlogPosting JSON-LD，包括标题、摘要、中文语言、作者及发布/更新时间，并补齐 4 篇缺少的作者字段。不添加虚构评分或浏览量。
- 搜索索引移除重复的 MathML/TeX 表达，并在去除 HTML 标签之后再解码代码中的实体；正文公式渲染保持不变。

没有更换主题、修改主题子模块或站点配置。第一轮的 7 张生成图继续保留；本轮新增内容主要适合公式、代码和检查项，没有额外生成装饰图。

## 重点文章修订

| 文章 | 第二轮处理 |
| --- | --- |
| CasADi | 删除重复导入、绘图和逐行复述；两种约束共用独立求解函数，增加状态、有限值、边界、残差和解析答案检查 |
| C++ 智能指针 | 用一个可编译的所有权实验替代重复类定义，补充 move、weak_ptr、控制块、循环引用和构造异常边界 |
| 强化学习 | 修正 PPO 公式括号、目标与梯度混淆、负优势裁剪方向和“硬限制比率/KL”的错误解释 |
| 强化学习 | 区分早期 SAC 与无独立 V 网络的版本，补齐温度、终止掩码、tanh/动作缩放后的概率密度及其梯度测试 |
| 强化学习 | 更正 Off-Policy 与离线学习的混用，补充表格收敛条件及随机梯度、基线和训练稳定性的限制 |
| Open3D | 消除 KD 树复杂度与自动平衡的矛盾表述，明确近邻查询返回距离平方，并与暴力距离排序核对 |
| NVIDIA Warp | 明确脚本片段的执行顺序与 CPU/CUDA 设备；加入固定随机源、NumPy 数值对照及后端依赖说明 |

关键结论核对了 CasADi 文档、PPO/SAC 论文、C++ 工作草案及 Open3D 文档；引用保留在相应文章中。算法例子仍是教学实现，不等价于完整工业控制或训练框架。

## 运行与构建验证

| 项目 | 本轮结果 |
| --- | --- |
| Hugo 生产构建 | 通过；另在独立目录检查，避免旧 public 文件干扰 |
| 全站静态检查 | 66 篇、132 条验收项、358 个代码块、195 个 HTML、6,024 处站内引用，无错误 |
| 结构化数据 | 63 篇正式文章 BlogPosting 完整；草稿不进入生产输出 |
| 第一轮回归 | A* 400 个随机组合、SPSC 100,000 个值、SRS 800 组、Pinocchio、PID、Modbus、CMake 和 TCP/UDP 回环均再次通过 |
| 新 C++ 所有权例子 | 编译与移动、弱引用、异常栈展开断言通过 |
| CasADi 3.8.0 | 两个最优点约 (0.00000666, 0.99999334) 和 (0.5, 0.5)，目标值约 2 与 2.5，边界与残差检查通过 |
| PyTorch 2.8.0 | 高斯 score-function 梯度、PPO 四种裁剪方向通过；SAC 手写密度与 TransformedDistribution 在 float64 下对照通过 |
| Warp 1.17.0 CPU | 向量加法、矩阵乘法、原子计数、粒子积分与自动微分片段运行通过 |
| Open3D 0.19.0 | 近邻距离平方与暴力排序一致；八叉树查询通过，没有启动图形窗口 |
| 浏览器 | 66 篇逐页检查无溢出、坏图、公式解析或 JavaScript 异常；验证目录、搜索、图片放大、菜单及代码复制 |

浏览器检查包含 390 px 窄屏、1440 px 桌面和明暗主题，额外测试“逆运动学”、PPO、CasADi、SPSC 查询及无匹配查询。目录跳转后正确移动焦点，代码复制内容与原始 DOM 文本一致。

截图与逐页结果位于本机 `/tmp/chase-blog-round2-browser/`。新增依赖仅安装在 `/tmp/chase-blog-round2-venv/`，未改动项目依赖或现有 Conda 环境。Warp 数值测试使用 CPU；未验证所有 GPU、GUI、MATLAB 或真实设备示例。

## 复验入口

- `scripts/validate_blog.py`：元数据、草稿状态、全部自检章节、图片、代码语法、结构化数据与站内引用。
- `scripts/test_blog_examples.py --network`：第一轮回归及新所有权程序。
- `scripts/test_blog_round2.py casadi|warp|torch|open3d`：按已安装依赖分别运行本轮库测试。
- `scripts/check_blog_browser.cjs`：目录、复制、搜索等浏览器回归，需要先启动本机 Hugo 与 Chrome 调试端口。

```bash
hugo --minify --destination /tmp/chase-blog-round2-production
python3 scripts/validate_blog.py --python-snippets --public /tmp/chase-blog-round2-production
python3 scripts/test_blog_examples.py --network
# 在安装了对应库的 Python 环境中分别执行：
python scripts/test_blog_round2.py casadi
python scripts/test_blog_round2.py warp
python scripts/test_blog_round2.py torch
python scripts/test_blog_round2.py open3d
```
