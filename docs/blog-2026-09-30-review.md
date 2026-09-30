# 2026-09-30 全站文章优化记录

本轮于 2026-09-30 11:39:51 至 13:40:07（Asia/Shanghai）完成，持续 7216 秒，达到本次两小时优化时长。范围、文件指纹和检查摘要见[验证数据](blog-2026-09-30-review.json)。

范围为 82 篇文章：79 篇公开文章、3 篇草稿。50 篇文章文件发生修改，其中 25 篇调整了讲解或内容结构；33 篇的 95 张表格改善了窄屏列宽，两组有重叠。全站共用的章节定位、图片占位和长标题换行同时改进。原始发布日期、草稿状态、公开地址和已有章节 ID 保持不变。本轮优化结束时，改动留在本地，尚未提交、推送或部署；私有任务规划稿仍排除在生产版、草稿预览和搜索索引之外。

## 主要内容改进

| 范围 | 本轮调整 |
| --- | --- |
| [强化学习](../content/posts/rl/index.md) | 重新串联状态历史、策略、轨迹、回报、MC 与 TD；补齐终点状态，用同一组奖励手算更新；替换不能支持结论的图，加入 20 种子 Cliff Walking 和最大化偏差实验；解释 PPO 的本轮数据复用、DDPG 的梯度路径和 off-policy 与离线训练的区别 |
| [Open3D](../content/posts/open3d/introduction/index.md) | 用完整点云流程连起读写、无效点清理、降采样、去噪、法线和变换；在同一平面比较几何 ICP、均匀颜色与纹理颜色；重写 Poisson 重建解释；用五个点区分样本重心与体素中心 |
| [六轴运动学](../content/posts/robotics/kinematics/six-dof-kinematics/index.md)、[七轴运动学](../content/posts/robotics/kinematics/seven-dof-kinematics/index.md) | 从 World、Base、Flange、TCP 串起坐标链，给出变换方向和逆变换的数值检查；由双球交圆推导臂角，说明 Jacobian 秩、退化和解分支连续性 |
| [CasADi](../content/posts/casadi/index.md) | 重画与代码一致的两个可行域，分清变量边界、等式交集、不可行的无约束极小点和数值解；解释新增约束为什么可能使最优值变大 |
| [CoACD](../content/posts/coacd/index.md)、[CCTag](../content/posts/calibration/cctag/index.md) | 用 U 形开口解释单个凸包与多个凸组件的区别；用像素算例区分重复性、均值偏差和 RMSE，说明像素到毫米换算的条件 |
| [A*](../content/posts/ai/algorithms/Astar/Astar-introduction/index.md)、[Ruckig](../content/posts/trajectory/ruckig/index.md)、[TOPPRA](../content/posts/trajectory/toppra/index.md) | 手算目标首次入队与有效弹出的区别；通过 10 ms 加速度变化解释 jerk；通过路径导数说明各关节限制怎样约束路径速度 |
| [Gymnasium](../content/posts/ai/gymnasium/1/index.md)、[Pink](../content/posts/pink/index.md)、[工作空间](../content/posts/robotics/workspace/whole-workspace/index.md) | 区分终止、截断与错误重置状态；把 QP 位移增量、时间步和速度限制连起来；区分位置覆盖、指定姿态、可行路径和控制能力 |
| 开发与排障文章 | 完善 GCC 工具链与运行库、Python ABI、Qt 插件来源、EGL/GLX、进程 PID、VS Code 断点与双环境说明；补齐 OpenGL 材质和缩放回调，并统一终端构建与调试参数 |

完整文件分类见验证数据的 `inventory.rows`。其余文章保留已经合理的内容，本轮进行了结构、引用、公式、代码语法和阅读检查；部分配套实验也重新运行，不表示所有论文都完成了训练或评测复现。

## 全站阅读体验

- 章节锚点移到标题开头，避免多行标题被顶部导航遮挡；保留原 ID 和显式退出选项。
- 图片在延迟加载前保留正确的宽高比，避免打开深层链接后被上方图片挤离目标。支持栅格图片、SVG 小数及科学计数法 `viewBox`、换行分隔和 pt 尺寸；小图片保留其宽度上限。
- 长标题在 320 px 屏幕内换行。对实测挤成竖排的说明表格逐一使用现有 `.table-readable` 样式，没有统一放大所有数值表。
- [AGENTS.md](../AGENTS.md)补充图片尺寸、章节兼容和窄屏表格检查规则；新增[阅读回归检查](../scripts/check_blog_reading.cjs)及[图片夹具检查](../scripts/test_blog_images.py)，扩展原有目录和示例验证，加入重复图号检查。

## 新增配图与可复现实验

五张概念配图使用内置 image_gen。提示词、编辑记录、源文件与 SHA-256 均保存在仓库文档中，正文只解释技术内容。

| 资源 | 提示词与验收记录 |
| --- | --- |
| [坐标链](../content/posts/robotics/kinematics/six-dof-kinematics/assets/frame-chain.webp) | [prompt](image-generation/frame-chain-20260930-prompt.txt)、[记录](image-generation/frame-chain-20260930.json) |
| [点云处理流程](../content/posts/open3d/introduction/assets/point-cloud-pipeline.webp) | [prompt](image-generation/open3d-pipeline-20260930-prompt.txt)、[记录](image-generation/open3d-pipeline-20260930.json) |
| [历史状态窗口](../content/posts/rl/assets/markov-history.webp) | [prompt](image-generation/markov-history-20260930-prompt.txt)、[记录](image-generation/markov-history-20260930.json) |
| [SARSA 与 Q-learning 目标](../content/posts/rl/assets/sarsa-q-targets.webp) | [prompt](image-generation/sarsa-targets-20260930-prompt.txt)、[记录](image-generation/sarsa-targets-20260930.json) |
| [凸组件与开口](../content/posts/coacd/assets/convex-parts-opening.webp) | [prompt](image-generation/convex-parts-20260930-prompt.txt)、[记录](image-generation/convex-parts-20260930.json) |

另有五张由 Matplotlib 绘制的定量图，数据或模型来自同目录脚本：

| 图与脚本 | 主要验证内容 |
| --- | --- |
| [Colored ICP](../content/posts/open3d/introduction/assets/colored-icp-comparison.png) · [脚本](../content/posts/open3d/introduction/colored_icp_check.py) | 同一合成平面的纹理提供切向约束；纯几何与均匀颜色对照保留 58.31 mm、约 4.58° 的初始误差 |
| [体素与重心](../content/posts/open3d/introduction/assets/voxel-centroid-comparison.png) · [脚本](../content/posts/open3d/introduction/voxel_centroid_check.py) | 固定同一网格原点，验证五个点的分组、样本索引、均值与格子中心 |
| [Cliff Walking](../content/posts/rl/assets/cliff-comparison.png) · [脚本](../content/posts/rl/cliff_comparison.py) | 20 种子、每种子 500 个训练回合；冻结 Q 后分别用贪心和探索策略评估，保留超时失败及其实际回报 |
| [最大化偏差](../content/posts/rl/assets/maximization-bias.png) · [脚本](../content/posts/rl/maximization_bias.py) | 100,000 次、10 动作独立高斯误差：固定动作、同一估计选择并评价、独立估计评价的差别；不将其称为完整 Double Q 训练实验 |
| [CasADi 可行域](../content/posts/casadi/assets/feasible-sets.png) · [脚本](../content/posts/casadi/plot_feasible_sets.py) | 两个问题的解为近似 `(0,1)` 与 `(0.5,0.5)`，目标值分别为 2 与 2.5；脚本求解函数与正文通过 AST 一致性比较 |

另外新增[点云流程脚本](../content/posts/open3d/introduction/point_cloud_pipeline.py)和[坐标链脚本](../content/posts/robotics/kinematics/six-dof-kinematics/frame_chain_check.py)，本轮共新增七个文章配套 Python 程序。旧图片资源没有删除；替换的是正文引用。

## 验证结果

生产版和含草稿版都使用全新临时输出目录，避免旧预览残留。两次 `hugo --minify` 构建成功，严格公式、结构化代码和本地引用检查均无错误或警告。

| 检查 | 结果 |
| --- | --- |
| 文章结构 | 82 篇、11 条学习路线、254 项文章验收检查；生产版 79 个文章结构化数据对象，草稿版 82 个 |
| 代码与引用 | 扫描 541 个 fenced code blocks；其中 197 个 Python 块以及 Bash、JSON、JSONC、XML 等适用语言通过语法检查；生产版 20,758 个本地引用、草稿版 21,736 个本地引用通过 |
| 深层链接与图片稳定性 | 82 篇 × 320/1440 px × JavaScript 开/关，共 328 组检查通过 |
| 表格 | 95 张表格 × 浅色/深色，共 190 组键盘方向键滚动检查通过 |
| 新图 | 10 张图 × 320/1440 px × 浅色/深色，共 40 组布局检查；10 次手机缩放检查通过 |
| 全站浏览器检查 | 82 篇的图片、公式、章节导航、主题链接与溢出检查通过；另检查 24 组代表性布局、复制回退和移动菜单 |
| 发现入口 | 79 篇公开文章的跨分页筛选、中文输入、URL 状态、浏览历史、无 JavaScript 与损坏数据回退检查通过；草稿未进入生产筛选数据 |
| 模板夹具 | 21 项目录检查；4 种页面模式 × 6 类图片共 24 个图片夹具通过 |
| 下载包 | 8 个打包检查入口通过，归档成员与文章源文件一致；适用包完成解压运行检查 |
| 兼容性 | 79 篇公开文章的原有 HTML ID 全部保留；82 篇的原始日期与 draft 字段不变；私有稿和 `public/` 不在 Git 跟踪内容中 |

本轮实际执行的数值和运行检查还包括：

- A* 与 Dijkstra 的 400 个地图对照，SPSC 的 100,000 条数据传递，所有权与 CMake 四种构建状态；C++20 不可变快照发布 100,000 次，检查保活与字段一致性。四组双 ABI 编译对照按预期成功或失败。
- Markov 历史窗口与矩阵传播对照、无效表拒绝、Cliff Walking 正文/下载函数一致性与冻结评估、REINFORCE 梯度和 TD3 停止梯度路径。
- 800 组 SRS 逆解、Pinocchio 可达与不可达目标、三种 Jacobian 参考系、PyTorch FK/IK 和一阶/二阶梯度检查；手眼、零点和可辨识性实验。
- Ruckig 一轴/七轴及非零终点边界，TOPPRA 多网格与区间极值，阻抗静态刚度、环境外力符号、日志异常输入与验收；滤波器和 MolmoMotion 配套数值图重新运行。
- Open3D 点云流程实测 `656→654→191→188`；3000 点球面 Poisson 重建的顶点密度长度和法线方向核对通过。该合成球面平均径向误差约 0.318 mm，不代表实际扫描精度。
- CoACD 1.0.14 对封闭 L 形网格得到两个凸组件，逐块导出后由独立 Open3D 进程读回；检查三个内部点被覆盖、一个开口点没有被覆盖。未据此声称完成物理引擎碰撞验收。
- Transformer 手写注意力与 SDPA 的输出/梯度、padding 和因果性；KV Cache 非方形 mask；DDIM 参数换算；完整二维 DDPM 的 3000 步 CPU 实验。没有下载或运行论文大模型权重。
- Jev 门控与路由、VLA 数据/时钟契约、LLM 指标、RynnBrain/InternVL Processor 等本地检查，以及相应下载包；这些是接口和数学验证，不是模型性能评测。
- Qt 6.8.3 Essentials 离屏 UI 再生成、信号和三个 Matplotlib 窗口的刷新/关闭；MeshCat 0.3.2 的 11 条序列化指令使用记录端验证，没有启动服务器；OpenGL 茶壶在 Xvfb 软件显示下检查三种窗口宽度。

## 外部引用与验证边界

对起点页面提取的 658 个独立外链进行联网检查并重试：636 个返回 HTTP 200，没有确认的 404/410；批量检查中的另 22 个遇到 403、429、TLS 或超时。随后通过浏览工具读到了其中 6 个文档（Ruckig 两页、MathWorks 三页及 Madgwick 报告），仍有 16 个未能确认。未把这些访问限制当作文章链接失效。批量状态、后续验证与剩余清单分别保存在验证数据中，新增的主要论文和文档引用另行核对。

本轮没有实机操作、硬实时测试或论文基准重训；Qt 离屏不替代原生桌面显示，CoACD 点测试不替代引擎碰撞，合成点云结果不替代真实数据评估。临时输出、截图及完整日志位于 `/tmp/chase-all-posts-20260930-oyc9ty4p`；关键结果已摘要到仓库验证数据，便于后续提交前复查。
