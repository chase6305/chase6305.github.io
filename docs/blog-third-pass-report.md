# 博客第三轮优化记录

后续更新见 [第四轮优化记录](blog-fourth-pass-report.md)，本文件保留第三轮完成时的范围与验证数字。

日期：2026-09-05。延续前两轮修改，未提交、推送或部署。保持 66 篇原文章的 URL、发布日期与 3 篇草稿状态；新增学习路线页，不将草稿转为正式文章。

## 全站阅读组织

- 全部 66 篇归入 10 条学习路线，覆盖运动学、规划控制、标定、AI、几何处理、C++、Python、GPU、图形排障和 Linux/通信。
- 文章顶部显示专题入口与当前位置，底部提供同专题前后篇；“继续阅读”的跨专题推荐保留。博客单页不再额外叠加按发布日期排序的另一套前后篇。
- 新增 `/learning-paths/`，提供专题跳转与建议顺序，博客列表增加入口。正式输出只列 63 篇；草稿预览列 66 篇并标记草稿。
- 共享数据在 [blog_topics.json](../data/blog_topics.json)。静态检查验证唯一覆盖、专题锚点、相邻文章、草稿隔离与链接。

本轮不是把 66 篇全部重写：重点修订了下表的 10 篇正文，其余正文沿用前两轮成果，并纳入共享导航和回归检查。逐篇记录见 [blog-third-pass.json](blog-third-pass.json)。

## 重点正文修订

| 文章 | 修订与验证 |
| --- | --- |
| Transformer | 全遮蔽行安全 softmax；投影后清零 PAD Query；拒绝空 batch 和全部忽略的监督目标；比较手写/SDPA 输出与梯度、padding 一致性及因果性 |
| 扩散模型 | 修正 100 步线性调度留下过多信号的问题；由 cosine 累计量反推 beta；增加完整 CPU 训练的真实结果图 |
| 分布式训练显存 | 估算器拒绝非法 world_size、负数与非有限值，复核 DDP / ZeRO 理论账本 |
| 手眼标定 | 增加 20 组无噪声合成变换，14 组求解、6 组留出验证；检查旋转合法性、变换方向与 AX=XB |
| 零位标定 | 改为二连杆 FK/Jacobian 零偏恢复，20 组拟合、10 组验证；有限差分核对 Jacobian，检测退化姿态秩不足 |
| PyTorch 运动学 | 提供可下载的无网格二连杆 URDF；执行 FK–IK–FK，并比较 Jacobian 与自动微分 |
| 工作空间 | 向量化采样；用不等长连杆展示内部不可达孔洞；检查解析内外半径并说明凸包会填满空洞 |
| Ruckig | 一轴/七轴共用验证函数；包含 t=0、精确终点、速度/加速度及区间平均 jerk；说明预测状态、真实反馈和非零目标速度 |
| TOPP-RA | 明确 q(s) 与 s(t)，显式选择输出参数化器；比较三种网格与解析 2.5 秒时长；区分求解网格和验证采样 |
| tqdm | 简化安装与重复示例；明确 update 的增量语义，增加独立业务计数和二进制文件扫描；解释非 TTY、异步完成量和多进程显示 |

## 配图及来源

1. [轨迹问题对比图](../content/posts/trajectory/toppra/assets/path-vs-state-trajectory.webp)：1536 × 1024，90,320 字节，约 88 KiB。使用**内置 imagegen**生成并做一次定向修订，参考 `content/posts/ai/transformer-attention/assets` 的白底、蓝绿柔和分区、细线与清晰标签。图中曲线是概念示意，不是实验数据。完整初始提示词和修订提示词保存在 [blog-image-prompts.json](blog-image-prompts.json) 的 `trajectory` 记录中。
2. [二维 DDPM 实际结果图](../content/posts/ai/diffusion-models/assets/ddpm-2d-cosine-cpu.webp)：42,716 字节，约 42 KiB，由文章代码在 CPU 上真实训练、采样后绘制，再编码为 WebP；**不是 AI 生成实验结果**。

两张新图共约 130 KiB，保存在文章 bundle 内，使用相对引用、中文替代文本与图注。Hugo 继续生成响应式尺寸、延迟加载并提供放大查看。前两轮 7 张生成说明图保留，本轮只新增 1 张生成说明图；实际结果图另计。

imagegen 原始输出保留在 `/home/ubuntu/.codex/generated_images/01a07027-d114-7f81-9b3b-f632140bfca3/`。最终项目资产与提示词都已进入工作区，没有引用仅存在于生成目录的图片。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| Hugo | `hugo --minify` 与独立目录生产构建通过，无模板或前置元数据错误 |
| 正式输出 | 66 篇源文章、63 篇 BlogPosting、10 条学习路线、349 个代码块、196 个 HTML、6,203 处站内引用，检查无错误 |
| 草稿预览 | 66 篇结构化数据与专题导航、206 个 HTML、6,359 处站内引用，无错误；3 篇草稿只在预览中出现 |
| 浏览器 | 66 篇分别检查 390 px 手机和 1440 px 桌面宽度；另检查明暗主题、专题页、目录、搜索、复制、菜单和图片放大，无异常 |
| TOPP-RA 0.6.10 | 网格 101/201/401 时长约 2.50020432 / 2.50000037 / 2.50000066 秒，密采样断言通过 |
| Ruckig 0.19.4 | 正向、反向、零位移和七轴目标通过；单位位移时长约 3.174802 秒；非法输入检查通过 |
| OpenCV 4.11.0 | 手眼留出集平移误差最大约 9.29e-16 m；这里只是无噪声合成数据的接口正确性测试 |
| PyTorch 2.8.0 | 注意力输出/梯度、padding、因果性和 LM loss 检查通过；短训练冒烟测试与完整 DDPM 实验分别运行 |
| pytorch-kinematics 0.10.0 | 二连杆位置误差约 1.30e-6 m、旋转误差约 8.75e-6 rad；Jacobian 自动微分对照通过 |
| NumPy / tqdm 4.70.0 | 110,018 个工作空间样本、确定性边界、空任务、增量计数及空文件/UTF-8/二进制计数通过 |
| 前轮回归 | A* 400 组、SPSC 100,000 个值、C++ 所有权、CMake、PID、SRS 800 组、Pinocchio 及 Modbus/TCP/UDP 回环再次通过 |

完整 DDPM 实验固定 seed=0、CPU 两线程、100 个扩散步、3,000 个训练步，生成 4,000 个有限样本，平均半径约 4.04646。结果图仍可见少量簇间过渡点；不据此声称生成分布精确一致，也没有声称在真实机器人、GUI 或所有 GPU 后端完成验证。

浏览器逐页结果和截图位于 `/tmp/chase-blog-round3-browser/`。新增依赖仅安装于 `/tmp/chase-blog-round3-venv/`，继承既有 PyTorch 等依赖，未改动项目依赖、系统驱动、主题子模块、站点配置或部署工作流。

## 复验命令

```bash
hugo --minify --destination /tmp/chase-blog-round3-production
python3 scripts/validate_blog.py --python-snippets --public /tmp/chase-blog-round3-production
python3 scripts/test_blog_examples.py --network
# 在安装对应依赖的 Python 环境中执行 CPU 数值回归：
python scripts/test_blog_round3.py
# 完整训练单独运行，会在新的临时目录保存真实结果图：
python scripts/test_blog_round3.py diffusion
# 先启动 Hugo -D 和本机 Chrome 的调试端口，再运行浏览器回归：
node scripts/check_blog_browser.cjs http://127.0.0.1:13141 9229 /tmp/chase-blog-round3-browser
```
