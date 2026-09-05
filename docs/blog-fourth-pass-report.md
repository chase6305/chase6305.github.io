# 博客第四轮优化记录

后续整体查找与导航改进见 [第五轮优化记录](blog-fifth-pass-report.md)，本文件保留第四轮完成时的范围与数字。

日期：2026-09-05。延续前三轮，未提交、推送或部署。覆盖 66 篇的共享阅读体验与全量回归，重点深修 7 篇，并修正另外 2 篇的代码标签或设备路径；不是宣称本轮重新改写全部 66 篇正文。逐篇范围见 [blog-fourth-pass.json](blog-fourth-pass.json)。

## 全站改进

- 代码复制使用原始文本，保留缩进、空行和多行字符串。复制权限拒绝、API 不存在和同步异常均有回退：选中代码，在代码块上方提示手动复制；状态通过 aria-live 通知，中文文章使用中文标签，失败后可重试。
- 复制进行中暂时禁用按钮；成功状态定时复原，连续操作先取消旧计时器，避免旧状态覆盖新提示。
- 静态检查增加转义代码围栏、Bash/sh 语法、JSON/JSONC/XML 和重复 HTML id。JSONC 解析保护带 URL、注释形状或逗号的字符串。shell 仅执行语法检查，不运行文章命令。
- 保留前三轮导读、132 条针对性自测、相关内容、10 条学习路线、响应式图片及结构化数据，URL、发布日期和 3 篇草稿状态不变。

## 正文修订

| 文章 | 本轮修改 |
| --- | --- |
| Jacobian | 修复两段转义围栏；基础脚本为后文提供变量；明确 WORLD 线速度的参考原点、任务秩、操作空间惯量与阻尼零空间泄漏；增加二连杆 URDF 复现入口 |
| f-string | 五组独立断言，修正填充宽度；固定时区样例；解释显示精度、字符宽度、调试 repr、日志求值与安全边界 |
| Unicode | 注入真实任务函数并保留异常；ASCII 回退与单行状态；UTF-8 增量解码；C++17/20 的 u8 类型与字节数断言 |
| watch | 区分快照和区间采样；修正 -e 说明；tmux 所有拆分明确目标，拒绝覆盖已有会话；设备轮询保留为需现场确认的独立步骤 |
| PySide6 UI | 提供最小 robot-dialog.ui 与目录结构；完整按钮/标签业务逻辑；重新生成界面不损失信号槽 |
| PySide6 + Matplotlib | 动态更新复用曲线，定时器归属窗口并在关闭时停止；说明自动缩放、事件循环及 offscreen 限制 |
| MeshCat | 圆柱局部 Y 轴旋转为世界 Z 轴；核对 0.3.2 服务启动，说明本地 URL 与实际监听不同，SSH 隧道不等于限制原端口访问 |
| udev | 统一示例设备路径；终端输出、规则内容与 shell 命令分开标注；改进 5 张原截图的替代文本 |
| NVIDIA 排障 | 4 处历史日志/输出改为 text，不把终端记录当作可执行 Bash；未修改系统驱动或安装方案 |

本轮不新增生成图片：已有 Jacobian 概念图、GUI 和 MeshCat 历史截图能支撑说明，新增内容以真实代码、断言与运行边界为主。前三轮按 transformer-attention/assets 风格生成的 8 张说明图保留；未生成模拟实验截图。

## 验证结果

| 检查 | 本轮结果 |
| --- | --- |
| Hugo | hugo --minify、独立生产目录和含草稿预览均通过 |
| 正式输出 | 66 篇源文章、63 个 BlogPosting、196 个 HTML、6,155 处站内引用，无错误 |
| 草稿预览 | 66 个 BlogPosting、206 个 HTML、6,311 处站内引用，无错误 |
| 代码静态检查 | 342 个围栏；149 个 Python、96 个 Bash、8 个 sh、3 个 JSON、6 个 JSONC、1 个 XML 通过相应语法检查 |
| 浏览器 | 66 篇分别检查 390 px 与 1440 px；图片、公式、导航、目录、搜索、主题、复制、图片放大与菜单检查通过 |
| 复制异常 | Promise 拒绝、缺失 clipboard、同步抛错三种分支通过；选区准确、无横向溢出、可重试，成功提示正常复原 |
| Python / C++ | 5 组 f-string 与 3 组 Unicode Python 示例通过；失败异常未被吞掉；C++17 和 C++20 分别编译并运行通过 |
| Pinocchio 4.1.0 | 文章 7 个 Python 片段顺序执行；20 个随机姿态、3 种参考系；位置差分最大误差约 1.64e-9，Jacobian 时间差分误差约 1.60e-9 |
| 零空间 | 合成 6×7 任务验证 JN≈0、零空间维度为 1；阻尼逆产生约 0.0299 的 JN 范数，证实不能声称完全不影响主任务 |
| PySide6 6.11.2 / Matplotlib 3.10.9 | UI 转换、两次点击与重新生成后的信号通过；3 次创建绘图窗口，各 50 次手动更新及真实定时器事件，关闭后定时器停止 |
| MeshCat 0.3.2 | 11 条命令经过真实协议对象序列化；圆柱两端 z 为 0 与 0.6；仅替换为记录接收端，没有启动服务 |
| tmux 3.2a | 私有测试 socket 内创建 3 个 pane；旁观会话不变；再次运行拒绝创建且已有 pane 不变 |
| 前轮回归 | A* 400 组、SPSC 100,000 个值、CMake、PID、SRS 800 组、Pinocchio、Modbus/TCP/UDP，以及轨迹、标定、tqdm、PyTorch 注意力和运动学回归再次通过 |

本轮 DDPM 仅复跑 20 步冒烟测试，未重复第三轮 3,000 步训练；原实际结果图未改动。语法检查不代表全部代码都完成运行验证：Matlab、CUDA、硬件命令及其他工程片段不在本轮新运行范围内。

## 测试环境与边界

- Qt 使用 offscreen 验证控件、绘制与生命周期，不冒充目标桌面的字体、平台插件或显示验收。
- MeshCat 测试验证客户端几何与序列化，不冒充实际 WebSocket 或远程浏览器连通测试。
- tmux 二进制和缺失库仅解包到临时目录，未安装到系统。测试使用独立 socket，watch 替换为占位进程；不接入实际设备，也未操作用户会话。
- PySide6 与 MeshCat 安装在 /tmp/chase-blog-round4-venv；既有 Pinocchio 与第三轮环境用于相应回归。没有修改主题子模块、配置、工作流或项目依赖。
- 仓库 public 中存在此前预览遗留文件，未做破坏性清理；正式/草稿隔离的结论来自独立构建目录，不能以混合旧文件的 public 作为发布验收依据。

逐页浏览器结果和截图位于 /tmp/chase-blog-round4-browser。重点截图：mobile-copy-fallback-dark.png、desktop-copy-fallback-light.png；前三轮截图保留在对应临时目录，便于对照。临时截图不是网站资源，不依赖它们提供页面功能。

## 复验命令

在仓库根目录执行；带可选依赖的组使用安装了对应依赖的 Python 环境：

```bash
hugo --minify
hugo --minify --destination /tmp/chase-blog-round4-production
python3 scripts/validate_blog.py --public /tmp/chase-blog-round4-production \
  --python-snippets --structured-snippets
python3 scripts/test_blog_round4.py --groups core
python scripts/test_blog_round4.py --groups jacobian
python scripts/test_blog_round4.py --groups qt meshcat
python scripts/test_blog_round4.py --groups tmux --tmux-binary /path/to/tmux
python3 scripts/test_blog_examples.py --network
python scripts/test_blog_round3.py
# 先启动包含草稿的 Hugo 与本机 Chrome 调试端口。
node scripts/check_blog_browser.cjs http://127.0.0.1:13142 9229 /tmp/chase-blog-round4-browser
```

新增回归脚本是 [test_blog_round4.py](../scripts/test_blog_round4.py)；完整前轮记录见 [第三轮报告](blog-third-pass-report.md)。
