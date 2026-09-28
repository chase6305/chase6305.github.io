# 2026-09-28 全站审阅与优化

执行时间：2026-09-28 10:17—12:17（Asia/Shanghai），持续约 2 小时。

本轮检查 81 篇源文章，其中公开文章 78 篇、草稿 3 篇。全站检查包括元数据、学习路线、标签、站内引用、代码围栏、公式、图片、浏览器导航和公开/草稿索引；重点修改 19 篇文章。没有把结构检查等同于逐篇复现论文实验。

本轮变更保存在工作区，未提交、推送或部署。3 篇源草稿保持 `draft: true`；另行保存的私有任务规划稿继续位于内容目录之外。原文章 URL、发布日期和旧章节锚点保留。

## 阅读入口和维护规则

- 学习路线从 10 条调整为 11 条：把具身策略与机器人学习单独成组，按照示教/ACT、VLA、数据、执行与拓展排列。滤波文章移到 PID 后面。
- 标签索引分为 8 个领域，补齐原先未归类的标签，增加可直接跳转的领域入口和数量。生产站的 160 个标签、草稿预览的 164 个标签均只出现一次。
- 修正 Hugo 标签显示大小写与源标签拼写之间的匹配，保留已有标签 URL。
- 静态校验增加标签完整性检查；浏览器检查增加领域入口、触控尺寸和图注裁切检查。实际截图发现的新图注裁切已通过既有图注组件修正。
- 修复 5 篇文章目录中的公式原文显示：桌面侧栏和手机折叠目录在构建时渲染数学符号，保留原章节链接。使用仓库自己的模板覆盖，没有修改主题子模块。
- `AGENTS.md` 补充标签归类、图注结构与现有验证工具的使用方法。
- `.github/workflows/hugo.yml` 在上传站点产物之前运行严格站点校验和文章筛选测试。没有改变部署目标或触发发布。

## 文章修订

| 文章 | 本轮改进 |
|---|---|
| [CCTag](../content/posts/calibration/cctag/index.md) | 区分检测候选、可靠解码、几何估计与独立验收；补充官方命令、误差与延迟口径 |
| [运动学参数标定](../content/posts/calibration/kinematics/index.md) | 增加标定闭环图、按采集批次留出数据、参数尺度与可辨识性；用小模型展示基座角与关节零位的不可区分性 |
| [手眼标定](../content/posts/calibration/model/index.md) | 增加解释器、包路径与 API 能力诊断；给出经过验证的独立 OpenCV 环境，说明特定版本缺少绑定的问题 |
| [非对称圆标记](../content/posts/calibration/opencv/index.md) | 增加 44 点检测实验与实际检测图，澄清 blob 像素面积、点序和误差单位 |
| [三维 A*](../content/posts/ai/algorithms/Astar/Astar-introduction/index.md) | 区分六邻域与二十六邻域的启发函数，给出不会把 Manhattan 距离误用为下界的数值例子 |
| [工作空间](../content/posts/robotics/workspace/whole-workspace/index.md) | 区分关节均匀采样与工作空间面积均匀采样；新增可复现分布图、解析 CDF 与数值对照 |
| [固定网口 IP](../content/posts/network-protocol/fixed_IP.md) | 增加双网卡直连机器人的配置与路由检查，区分路由选择、邻居解析和应用通信 |
| [Python 版本管理](../content/posts/python/version/index.md) | 增加实际解释器、模块路径、包遮蔽与 ABI/API 差异的诊断顺序 |
| [Being-0](../content/posts/thesis/being_0/index.md) | 修复项目链接；保留消融实验的原始计数和小样本边界，不把局部结果概括为所有任务提升 |
| [Pink](../content/posts/pink/index.md) | 修复仓库与文档链接；增加不下载模型的二连杆差分 IK 示例，说明速度、步长和模拟时间 |
| [Open3D](../content/posts/open3d/introduction/index.md) | 修正依赖构建说明、RANSAC 参数、刚体变换、ICP 验收、三维 Alpha Shape 解释与比较表格；增加无对应点反例和可下载检查脚本 |
| [RynnBrain](../content/posts/ai/rynnbrain/index.md) | 清除验收文字中的制图过程描述，保留角度协议和真实机器人验证边界 |
| [强化学习基础](../content/posts/rl/index.md) | 整理 REINFORCE、TRPO、TD3、SAC 推导；修正折扣权重、GAE 回合边界和梯度路径；替换误导性的流程图与参数推荐图 |
| [CoACD](../content/posts/coacd/index.md) | 复现原生库共同导入时的崩溃；将分解与 Open3D 读取分为两个进程，新增参数/版本清单与无窗口验收 |
| [Gymnasium](../content/posts/ai/gymnasium/1/index.md) | 补齐录像依赖，实际运行随机交互与 MP4 编码/解码；明确窗口显示与录像路径的区别 |
| [六自由度运动学](../content/posts/robotics/kinematics/six-dof-kinematics/index.md) | 统一目标坐标、工具变换与腕心公式；新增 NumPy FK、几何 Jacobian 和腕心关系检查 |
| [TCP/UDP 基础（草稿）](../content/posts/network-protocol/tcp-udp-introduction.md) | 补充字节流分帧、完整解析例子、UDP 消息边界和机器人应用协议约定 |
| [运动学基础（草稿）](../content/posts/robotics/kinematics/kinematics-introduction.md) | 增加二连杆 FK/IK 推导、双分支、不可达判断、Jacobian 与坐标变换示例 |
| [智能算法概述（草稿）](../content/posts/ai/algorithms/overview.md) | 增加问题定义、BFS/Dijkstra 对照、训练验证划分和阅读入口 |

## 新增图与可运行材料

新增 3 张概念流程图：标定验证流程、策略梯度采样与更新、TD3 三条更新路径。提示词、来源路径和文件摘要位于 `docs/image-generation/`，不进入文章正文。

新增 3 张由程序产生的图：圆点检测结果、工作空间采样分布、GAE 残差权重。没有用生成式图像表示实测数据或精确数值。

新增 8 个可下载 Python 文件：

- `calibration/kinematics/calibration_identifiability.py`
- `calibration/opencv/circle_grid_smoke.py`
- `open3d/introduction/icp_verification.py`
- `pink/pink_planar_ik.py`
- `robotics/workspace/whole-workspace/workspace_sampling.py`
- `robotics/kinematics/six-dof-kinematics/dh_fk_check.py`
- `rl/rl_update_checks.py`
- `rl/gae_weights.py`

这些路径均相对于 `content/posts/`。示例分别标明单位、随机种子、运行前提及验证范围。

## 数值与运行验证

| 项目 | 实际检查结果 |
|---|---|
| 标定可辨识性 | 三参数 Jacobian 秩为 2；固定规范自由度后拟合与 12 个留出样本验证通过 |
| 圆点检测 | 合成图的 44 个点全部检出，点序与物理间距断言通过 |
| 工作空间采样 | 两组各 20,000 点；关节采样经验 CDF 对解析 CDF 的最大差约 0.00833 |
| Pink | 88 个模拟步后位置误差约 7.02e-6 m；0.88 s 是模拟时间，不是推理耗时 |
| ICP | 已知刚体变换恢复通过；零对应点时 fitness=0、RMSE=0 的失败结果被正确拒绝 |
| REINFORCE | 穷举四条轨迹后，解析梯度、差分、reward-to-go 与状态基线梯度一致；遗漏外层折扣的反例产生预期偏差 |
| TD3 | 目标停止梯度、终止样本目标、回放动作 Critic 更新与冻结 Critic 后的 Actor 梯度检查通过 |
| GAE | 真终止、时间截断、批次末尾和 λ=0 边界检查通过；跨回合残差未泄漏 |
| 六轴 DH | 100 个随机姿态的几何 Jacobian 对差分最大误差约 2.25e-10；腕心关系通过 |
| CoACD | L 形模型导出 2 个凸且闭合的组件；重新读取后包围盒坐标最大偏差约 9.79e-5；近似分解体积和约 2.999685，输入体积为 3 |
| Gymnasium | LunarLander-v3 随机交互 1,000 步通过；视频时长 2.48 s、50 fps、600×400，首尾帧可解码且不同 |
| PyTorch 运动学 | 已知目标 FK→IK→FK 与自动微分 Jacobian 对照通过 |
| Warp | CPU 上的向量、矩阵、原子加、粒子积分和自动微分通过 |

另运行了现有 C++/Python 核心示例、串口假设备与网络回环测试、SRS/Pinocchio IK、手眼/零位标定、Ruckig/TOPPRA、CasADi、Jacobian/阻抗控制、Attention/扩散短实验、LLM 指标与预算、PPO/DPO/GRPO、GSPO、机器人滤波、MolmoMotion 数据契约、VLA/RTC/ACT 与 ELMP 检查。已有下载 ZIP 的内容与源文件一致。

主要验证环境为 Python 3.10，NumPy 2.2.6、SciPy 1.15.3、OpenCV 4.13.0.92、Open3D CPU 0.19.0、Pinocchio 4.1.0、Pink 4.4.0、Ruckig 0.19.4、TOPPRA 0.6.3、CoACD 1.0.14、Gymnasium 1.3.0、MoviePy 2.2.1、Warp 1.17.0；PyTorch 检查使用已有 2.13.0 环境，pytorch-kinematics 为 0.10.0。新增依赖安装在临时隔离环境中。

以上检查没有执行真实机器人动作，没有重新训练论文模型，没有运行 GPU 性能基准。CoACD 的导入崩溃只对应所测依赖组合，尚未确定其底层符号冲突原因；进程分离已验证能完成本例工作流。

## 构建、浏览器和链接

最终计数与逐篇结果保存在 [本轮验证数据](blog-2026-09-28-review.json)。检查生产和含草稿两种构建，使用全新输出目录以排除旧产物污染。

全站浏览器检查覆盖 81 篇文章的图片加载、公式、目录、导读、自测、学习路线和移动/桌面宽度。修改文章额外检查 320、390、1440 px 与浅色、深色、暖色主题；新图在移动和桌面上均检查放大。筛选测试覆盖中文输入、大小写与全角字符、主题选择、排序、历史状态、无 JavaScript 回退及字面文本查询。

19 篇修改文章完成 171 组宽度与主题组合检查，新图完成 12 次放大/关闭操作；5 篇含目录公式的文章另完成 45 组展开目录检查。323 个原章节锚点全部保留。生产构建验证 19,823 处站内引用，含草稿构建验证 20,794 处；496 个代码围栏完成解析，其中 Python、Shell、JSON/JSONC 和 XML 进行了语法检查。没有把 C++、MATLAB 等依赖完整工程的片段计为全部编译通过。

初次外链检查共 497 个唯一地址：473 个返回 200，4 个确认 404 并已修复，另 20 个因访问限制或超时未能判定。修复的是 Pink 的 3 个旧文档地址和 Being-0 的旧项目地址。403、429 或超时不等于目标已删除，不据此猜测替代地址。受限项主要来自 Physical Intelligence、MathWorks、Ruckig 文档及少量论文站点。

本轮新增或替换的 33 个唯一外链已另行访问，全部返回 200。生产版和草稿版共 1,018 个文本产物未检出私有任务规划稿的路径、原资料文件名或完整标题。

复验使用与工作流一致的 Hugo 0.165.0。最小命令：

```bash
hugo --minify --destination /tmp/blog-production-check
python3 -B scripts/validate_blog.py --public /tmp/blog-production-check \
  --python-snippets --structured-snippets --strict-math
node scripts/test_blog_filter.cjs /tmp/blog-production-check

hugo --minify -D --destination /tmp/blog-draft-check
python3 -B scripts/validate_blog.py --public /tmp/blog-draft-check \
  --include-drafts --python-snippets --structured-snippets --strict-math
node scripts/test_blog_filter.cjs /tmp/blog-draft-check --include-drafts
```

临时运行日志和截图位于 `/tmp/chase-blog-audit-20260928-l_u9_jgq/`。静态语法检查不会执行文章中的系统配置、网络修改或硬件控制命令；涉及桌面插件、实机控制、完整模型训练的内容仍需要对应环境单独验收。
