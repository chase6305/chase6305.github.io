# 2026-09-28 第二轮全站优化

本轮于 2026-09-28 13:31:18—15:31:51（Asia/Shanghai）完成，持续超过两小时。重点完善 14 篇文章，新增或替换 11 张配图，补充 12 个可下载程序。产物清单、文件摘要和检查计数见 [验证数据](blog-2026-09-28-round2-review.json)。本文件记录第二轮增量，不覆盖[上一轮报告](blog-2026-09-28-review.md)。

修改留在本地工作区，未提交、推送或部署。站点仍有 81 篇文章：78 篇公开、3 篇草稿；内容目录外的私有任务规划稿保持隔离。原始发布日期、草稿状态、URL 和既有章节锚点保留。TCP 文章从单文件移到同名页面包，URL 不变。

## 文章改进

| 文章 | 本轮改进 |
| --- | --- |
| [PD/PID](../content/posts/pid/index.md) | 增加可下载控制器、条件积分与饱和恢复实验；区分积分状态单位、测量微分、控制器限幅和执行器实际输出 |
| [Ruckig](../content/posts/trajectory/ruckig/index.md) | 验证逼近速度上限时的状态可行性、时间/相位同步、非零终端速度及采样时刻与精确终点的区别 |
| [TOPPRA](../content/posts/trajectory/toppra/index.md) | 用分段多项式驻点和单侧端点检查连续峰值，展示离散网格上的约束不等于段内处处达标，并解释统一放慢的条件 |
| [TCP](../content/posts/network-protocol/c++_tcp/index.md) | 新增长度分帧图与 C++17 程序，覆盖共享绝对截止时间、部分 EOF、空消息、拼接、拆分和慢速分段到达 |
| [UDP](../content/posts/network-protocol/c++_udp.md) | 补充 MTU/头部预算、模序号比较、重启会话和数据年龄；修正接收超时文字，避免把每次等待误写成总截止时间 |
| [队列](../content/posts/queue/index.md) | 用同一生产/消费模型比较三种拥塞策略，区分队列容量、丢弃方式、消费者排空与 TTL；明确它不是并发性能测试 |
| [智能指针](../content/posts/cpp/smart-pointer/index.md) | 增加不可变快照图与 C++20 双线程程序，说明强引用持有旧版本、发布新版本以及引用计数与实时性的边界 |
| [零位标定](../content/posts/calibration/zero/index.md) | 用三连杆接触同一未知点的例子，说明零残差、固定规范自由度与绝对坐标准确性之间的区别 |
| [CasADi](../content/posts/casadi/index.md) | 比较非零目标梯度的约束最优点、低代价但不可行的候选、两个成功收敛的局部极小值；检查 KKT 与求解状态 |
| [分布式训练](../content/posts/ai/distributed-training-memory/index.md) | 新增两进程 Gloo 实验，对照全局有效 token 均值、局部均值平均、空 Rank、空窗口和不足步数的末尾窗口 |
| [LeRobot / ACT](../content/posts/ai/lerobot-act/index.md) | 推导动作元素损失与样本 KL 的不同全局分母，区分目标函数推导与固定版本的实际实现 |
| [RTC](../content/posts/ai/real-time-chunking/index.md) | 清理介绍图示制作过程的措辞，保留 CPU 算例的验证范围 |
| [机器人滤波](../content/posts/robotics/control/robot-filters/index.md) | 增加低通位置与闭环极点实验，说明独立降噪效果不能代替反馈稳定性判断，并与 PID 文章连接 |
| [动力学参数辨识](../content/posts/robotics/dynamics/parameter-identification/index.md) | 修正协方差、RLS 尺度、TLS 假设、参考/实测运动与滤波关系；增加相关噪声覆盖率实验，修正旧图和合成数据描述 |

## 图与下载材料

新增或替换的概念图采用 Imagegen：TCP 分帧、不可变快照、完整参数与基参数、离线辨识与在线候选跟踪。提示词与来源记录放在 `docs/image-generation/`，文章只解释概念。新的图使用白底、细圆角边框、克制的蓝/紫/橙/绿配色；逐项检查维度、箭头和标签。

- [TCP 分帧提示词](image-generation/tcp-framing-20260928-prompt.txt)
- [不可变快照提示词](image-generation/immutable-snapshot-20260928-prompt.txt)
- [基参数提示词](image-generation/dynamics-base-parameters-20260928-prompt.txt)
- [离线/在线流程提示词](image-generation/dynamics-offline-online-20260928-prompt.txt)

七张数值图由对应程序生成：PID 饱和、Ruckig 同步、队列新鲜度、TOPPRA 连续峰值、求解器候选、滤波闭环稳定性、相关噪声下的参数估计分布。动力学文章原有的诊断曲线缺少可复算的数据依据，正文改用新实验图；旧文件保留。旧基参数图中的矩阵/向量等号和旧在线/离线图中的参数空间与流程顺序也得到修正。

本轮共新增或替换 11 张配图（4 张概念图、7 张数值图），新增 12 个可下载程序：10 个 Python、2 个 C++。每个程序明确了所测模型、单位和运行方式；数值结果文件随文章保存，已与构建产物和本地 HTTP 下载内容比对。

## 实验结果与边界

| 项目 | 复算结果 |
| --- | --- |
| PID 恒定扰动 | PD 的解析稳态偏差为 0.25 m；给定理想模型下 PID 在 20 s 时偏差约 1.03e-5 m |
| PID 饱和恢复 | 条件积分将本例的积分贡献峰值从约 36.674 N 降至 2.986 N；恢复时间定义和记录窗口在文中说明 |
| Ruckig | 向上速度余量不足的状态被拒绝；相位同步保持本例关节直线，但经非线性 FK 后 TCP 仍走曲线；`Finished` 不等于速度为零 |
| TOPPRA | 11 点网格的段内速度峰值约为限值的 1.04864 倍，加速度峰值约 2.06930 rad/s²；加密和统一放慢分别检查，未把有限采样当作连续证明 |
| 队列 | 相同 200/50 Hz 与容量 8 条件下，FIFO 拒绝新样本的最大交付年龄为 355 ms，替换最旧样本为 35 ms；TTL 方案明确拒绝过期交付 |
| 智能指针 | 两线程发布 100,000 个不可变状态，字段一致性与所有权检查通过；观察数量依调度变化，不要求读到每个版本 |
| 未知接触点 | 40 个训练姿态的联合 Jacobian 秩为 4/5；11 个留出姿态内部残差接近零，但与独立外部点的误差约 25.96 mm |
| CasADi | 约束最优点的目标梯度非零而 KKT 残差小；不可行候选被拒绝；两个成功收敛的局部极小值仍有不同代价 |
| DDP token 均值 | 正确归一化与单进程参考的首窗口梯度最大差约 2.78e-17；错误平均局部均值的差异约 0.0880；空 Rank 与全局空窗口通过 |
| 测量低通 | 本例 1 Hz 反馈低通的最大极点模约 1.003462，出现增长振荡；同一低通放在参考输入时约 0.985326。均为明确结构下的线性计算 |
| 参数区间 | 2,500 次相关噪声记录中，错误独立假设下的名义 95% 区间覆盖率为 37.84%/39.64%；完整已知协方差下为 95.20%/95.12% |
| 独立辨识工程 | 本地 `RobotServer` 固定版本 `3f27a1cb4dda0f107f0ff7ff16553f42ad8a7813`、干净的跟踪工作树，在临时目录完成 Debug 构建与 8 项 CTest；合成示例恢复四个参数 |

上述实验没有控制真实机器人，没有重新训练论文模型，也没有测量 GPU 性能。已知协方差、无噪声几何或线性系统等条件已在对应文章中说明。TOPPRA 浮点驻点求解是指定多项式的数值验证，不是任意轨迹的形式化认证；两线程检查不等于实时调度保证。

C++17 TCP 程序和 C++20 快照程序额外通过 AddressSanitizer/UndefinedBehaviorSanitizer。快照程序发现本机 GCC 11 标准库缺少所需特化，实际使用 GCC 12.3 验证，并在源码中加入功能宏检查。没有修改系统编译器默认项。

## 目录、兼容性与站点验证

桌面和移动目录现在复用 Markdown 已渲染的公式标题。旧的美元符号猜测方式会混淆转义货币符号、代码跨度和公式；新规则只处理标题中的实际 KaTeX，并保留原锚点。

`scripts/test_blog_toc.py` 在临时目录构建两套目录，覆盖美元公式、括号公式、货币符号、代码、混合格式、链接标题和中文锚点，同时检查自动生成的栏目页。实时 Hugo 预览也检查了同一锚点从公式改成代码文字、再改回公式的过程，没有残留旧标题。测试已加入 GitHub Actions，维护方法写入 `AGENTS.md`；没有修改主题子模块。

检查使用 Hugo 0.165.0，以及生产与含草稿两种全新输出目录。全站浏览器检查覆盖 81 篇文章；后续修改页面再按 320、390、1440 px 和浅色、深色、暖色主题检查，包含新图放大、图注、公式和横向溢出。筛选检查覆盖中文输入、共享查询 URL、历史返回、分页、键盘操作、无 JavaScript 与无效元数据回退。

最终检查结果：

- 生产版 78 篇公开文章、草稿版 81 篇文章；230 项文章验收检查通过。语法检查覆盖 193 个 Python 代码块及 Bash、JSON、JSONC、Shell、XML 片段；这不等于全部 C++ 片段均已编译。
- 生产版检查 268 个 HTML 页面与 19,983 处本地引用，草稿版检查 279 个 HTML 页面与 20,954 处本地引用；无校验错误或未渲染公式警告。
- 全站浏览器检查覆盖 81 篇；14 篇修改文章完成 126 组宽度/主题组合与 22 次新图放大检查。最后修正文句后的 PID 与动力学页面各另检查 9 组布局，分别完成 2 次与 6 次放大。
- 保留 81 篇文章原有的 2,367 个页面 ID 与 1,727 个标题锚点。30 个新增下载资源的源文件、两种构建产物和本地 HTTP 内容完全一致。
- 本轮新增的 14 个外链均返回 HTTP 200；私有稿未进入两种构建共 1,010 个受检文本输出。原有头像、配置与主题子模块未变。

旧锚点对照、下载摘要、外链访问结果和私有稿隔离结果保存在本轮 JSON。历史受限外链仍与真正的失效地址分开记录，没有把 403、429、证书错误或超时直接当成页面删除。

复验命令：

```bash
hugo --minify --destination /tmp/blog-round2-production
python3 -B scripts/validate_blog.py --public /tmp/blog-round2-production \
  --python-snippets --structured-snippets --strict-math
node scripts/test_blog_filter.cjs /tmp/blog-round2-production

hugo --minify -D --destination /tmp/blog-round2-drafts
python3 -B scripts/validate_blog.py --public /tmp/blog-round2-drafts \
  --include-drafts --python-snippets --structured-snippets --strict-math
node scripts/test_blog_filter.cjs /tmp/blog-round2-drafts --include-drafts
python3 -B scripts/test_blog_toc.py
```

本轮日志与截图目录：`/tmp/chase-blog-audit-20260928-round2-rd03rcpx/`。与第一轮共用的隔离科学计算环境没有写入博客内容目录，构建输出也没有加入 Git。
