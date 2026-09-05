# 博客优化记录

完成日期：2026-09-05。

本文件保留第一轮记录。后续见 [第二轮](blog-second-pass-report.md)、[第三轮](blog-third-pass-report.md)、[第四轮](blog-fourth-pass-report.md)；最新整体筛选、导航和回归见 [第五轮优化记录](blog-fifth-pass-report.md)。

## 范围与保留项

覆盖全部 66 篇文章：63 篇已发布、3 篇草稿。保留原发布日期、URL、草稿状态及已有有效图片；没有提交、推送或部署。仓库实际使用 Hextra，本次使用仓库自有模板覆盖，没有修改主题子模块。

每篇文章均更新摘要与 description，增加针对性的前置知识、阅读重点和两篇延伸阅读。逐篇内容及配图决策见 [blog-editorial-review.json](blog-editorial-review.json)。原本较完善的 Transformer、扩散模型、动力学等长文保留主体，重点完善阅读入口和图片加载；问题较集中的教程重整示例、步骤和适用边界。

## 主要内容修订

| 主题 | 修订重点 |
| --- | --- |
| A* | 提供独立 C++17 示例，统一邻接、边权、启发式、穿角检查和失败返回，用 Dijkstra 对照路径代价 |
| 强化学习 / Gymnasium | 修复代码缩进，区分 terminated 与 truncated，修正 Double Q 演示、DQN 标量目标、策略梯度采样和 GAE 权重 |
| 运动学 / 标定 | 统一位姿与雅可比参考系，补充 IK 失败状态、限位与 FK 验证；明确 DH、手眼标定、零位辨识的模型假设 |
| 队列 / 通信 | 区分 Python 加锁队列与 C++ SPSC；增加 TCP 分帧、UDP 截断处理、串口完整事务和 CRC 校验 |
| 规划 / 控制 | 澄清 TO、MPC、WBC 的关系；补齐 PID 采样时间、输出限幅和抗积分饱和 |
| Python / C++ / CMake | 更正标准版本、所有权、运行时库和解释器选择；提供可本地复现的 ExternalProject 示例 |
| 图形 / 驱动 / 系统排错 | 优先定位实际插件、库路径和显示会话，移除无条件改系统链接、宽泛权限及混装驱动等建议 |
| 论文笔记 | 区分论文报告、概念图与个人解读，删除缺少依据的固定提升比例和参数断言 |

保留的历史实验截图已在相关重写示例旁说明来源，避免读者误认为它们是新代码本次运行的结果。硬件相关片段明确依赖厂商手册、模型和实机安全验证。

## 配图与阅读体验

使用内置 imagegen 生成 7 张概念教学图，样式参考用户指定的 `content/posts/ai/transformer-attention/assets`：浅暖底色、蓝绿橙紫配色、圆角模块和简洁连线。全部图片经过目视检查，其中 SRS 几何、SPSC 发布顺序和 IK 概览做过修订。没有生成伪造的性能曲线或实验照片。

| 文章 | 新图 |
| --- | --- |
| 七自由度 SRS 运动学 | 固定位姿、肘圆与臂角 |
| 并发队列 | SPSC 数据发布、读取与槽位复用 |
| TO / MPC / WBC | 参考运动、滚动规划、全身控制和状态反馈 |
| PD / PID | P、I、D 分支与限幅反馈 |
| Pinocchio 数值 IK | 目标、FK、误差、阻尼求解和最终验证 |
| CUDA / NVIDIA Warp | 32 线程执行组与 Python 库的区别 |
| Being-0 | 高层模型、Connector 和低层技能 |

7 个 WebP 源资产合计 667,700 字节，约 652 KiB。完整生成提示词、修订提示词和最终文件路径见 [blog-image-prompts.json](blog-image-prompts.json)。其他文章按情况保留原图，或使用更适合代码、命令和参数比较的文字/表格，没有机械地逐篇插入装饰图。

图片渲染统一支持尺寸声明、懒加载、异步解码及可用尺寸的 WebP srcset；保留 SVG、原图和点击放大能力。原 AI 长文中的 39 处 HTML 图片调用接入同一渲染入口，已有图注继续保留。Hugo 构建统计为 366 个处理图像输出，此数量不是新增原创图数量。

新增文章阅读提示和延伸阅读卡片，适配窄屏及深色主题；列表展示阅读时长。同时修复博客列表页缺少移动侧栏导致的 JavaScript 异常和菜单失效。阅读时长只是包括代码在内的估计，不等于完成实验所需时间。

## 验证结果

### 构建与静态检查

- `hugo --minify` 成功，无模板、短代码或资源处理错误。
- 在全新临时目录进行生产构建，避免旧 `public/` 文件干扰。
- 覆盖 66 篇、363 个代码块、195 个生成 HTML 文件。
- 检查 5,036 处站内链接、锚点和资源引用：无失效目标。
- 所有标记为 Python 的代码块通过语法解析；伪代码、接口签名和终端输出使用相应文本标记。
- 校验逐篇元数据、图片引用和发布状态；3 篇草稿没有进入生产输出。
- `git diff --check` 通过。`public/`、Hugo 图片缓存和 Python 检查缓存不纳入版本控制。

### 示例运行回归

| 检查 | 结果 |
| --- | --- |
| A* | 基本边界测试及 400 个随机地图/邻接组合与 Dijkstra 对照通过 |
| C++ SPSC | 双线程按序传递 100,000 个值，总和 4,999,950,000 |
| C++ 标准库 | 2 个独立示例编译运行通过 |
| CMake ExternalProject | 全新及增量构建通过，程序输出 42 |
| PID | 限幅、抗积分饱和、非法采样周期测试通过 |
| RL | SARSA/Q-Learning 悬崖环境运行通过；n-step 真终止与截断测试通过 |
| SRS IK | 100 个目标 × 8 类分支，共 800 组 FK→IK→FK；最大变换矩阵元素误差约 9.0×10⁻¹⁵ |
| Pinocchio 4.1.0 | 30 个近初值可达目标成功，最大位置/旋转残差约 8.83×10⁻⁵（分别按米/弧度检查）；不可达目标正确报告失败 |
| Modbus 假串口 | CRC、逐字节读取、写回显、并发事务、错误响应和超时通过 |
| TCP / UDP | 仅本机回环：TCP 回显/碎片化，UDP 回显/零长度/超长数据报拒绝通过 |

这不代表所有博客代码均执行过。依赖真实设备、GPU、驱动安装、GUI、MATLAB 或外部模型的示例未做完整运行验证；语法通过也不等于算法在任意模型或初值下正确、收敛或实时安全。

### 浏览器检查

使用 `hugo server -D` 和本机 Chromium 检查全部 66 篇（包括草稿），窄屏宽度 390 px；另外检查 1440 px 桌面、明暗主题、图片放大、延伸阅读和文章/列表页移动导航。

最终结果：无页面横向溢出、无损坏正文图片、无 KaTeX 错误、无未捕获 JavaScript 异常。桌面与手机截图经过目视复核。截图和逐页浏览器结果保存在本机 `/tmp/chase-blog-browser/`；它们是验证产物，不会发布为博客内容。

## 复验方式

静态检查只依赖 Python 标准库：

```bash
blog_build="$(mktemp -d /tmp/chase-blog-build.XXXXXX)"
hugo --minify --destination "$blog_build"
python3 scripts/validate_blog.py --python-snippets --public "$blog_build"
```

示例回归需要 Python 3.10+、NumPy、Pinocchio、g++ 和 CMake；`--network` 只使用回环地址和临时端口：

```bash
python3 scripts/test_blog_examples.py --network
```

浏览器脚本使用 Node.js 22 的内置 WebSocket。先启动本机 Hugo 预览（默认端口 13139）及开启本机调试端口 9229 的 Chrome，再运行：

```bash
node scripts/check_blog_browser.cjs http://127.0.0.1:13139 9229 /tmp/chase-blog-browser
```

无需更改站点配置或部署流程。正常发布仍由用户决定是否提交和推送。
