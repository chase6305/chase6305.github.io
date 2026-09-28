# 2026-09-28 第三轮全站优化

本轮于 2026-09-28 15:53:56 至 17:54:25（Asia/Shanghai）完成，持续 7229 秒，达到本次两小时优化时长。重点完善 13 篇文章，另修正 1 篇文章的引用，增加 15 个可下载 Python 程序和 9 张配图。本文只记录本轮增量，保留前两轮报告；文件清单、摘要和检查结果见[验证数据](blog-2026-09-28-round3-review.json)。

改动留在本地，未提交、推送或部署。81 篇文章的原始发布日期、草稿状态和地址保持不变，生产版仍包含 78 篇公开文章。任务规划私有稿仍在内容目录之外，未进入页面、搜索索引、RSS 或 sitemap。

## 文章改进

| 文章 | 本轮改进 |
| --- | --- |
| [Jacobian](../content/posts/robotics/kinematics/jacobian/index.md) | 用同一机械臂核对 LOCAL、LOCAL_WORLD_ALIGNED、WORLD；明确表达坐标轴与参考点是两件事，补充工具偏移、wrench 对偶、功率和 Jlog 验证 |
| [Pinocchio IK](../content/posts/robotics/kinematics/pinocchio/index.md) | 增加可下载实现和独立验证，位置／旋转残差同时缩放 Jacobian；验证米与毫米换算，完善无效输入和停滞状态 |
| [PyTorch IK](../content/posts/robotics/kinematics/pytorch/index.md) | 加入可微 FK 的解析式、中心差分、gradcheck 和二阶导数对照；说明高阶梯度路径与 URDF 初始浮点精度 |
| [阻抗控制](../content/posts/robotics/control/impedance-control/index.md) | 区分速度与力矩零空间投影；修正奇异构型下柔顺性解释；补充可变刚度的能量账本；统一环境外力、执行器力矩与法向力反馈的符号；修正传感器到控制点的力矩平移方向，并注明旋转误差导数的局部近似 |
| [Open3D](../content/posts/open3d/introduction/index.md) | 通过单平面与三面场景解释 ICP 的可观测性，区分 fitness、欧氏 RMSE 与点到平面目标，核对源码中的信息矩阵用途 |
| [手眼标定](../content/posts/calibration/model/index.md) | 推导 Eye-to-Hand 的两条等价调用路径；用退化运动说明残差接近零并不保证外参正确 |
| [Transformer](../content/posts/ai/transformer-attention/index.md) | 加入 KV Cache 下非方形因果掩码对照，验证分块、单 token、左侧 padding、未填充槽位和未来 key |
| [扩散模型](../content/posts/ai/diffusion-models/index.md) | 将 ε／x₀／v 换算、损失权重和低 SNR 敏感性连接起来；核对跳步 DDIM 的实际两个端点与相邻 DDPM 关系 |
| [LLM 训练指标](../content/posts/ai/llm-training-metrics/index.md) | 加入按独立题组重采样的例子，区分重复记录数、独立信息量、按题／按组加权；更新下载包 |
| [PPO / DPO / GRPO](../content/posts/ai/ppo-dpo-grpo/index.md) | 完整枚举三动作分布，说明 KL 数值恒等式不自动意味着固定样本梯度相同；更新下载包并验证声明的依赖版本 |
| [CMake ExternalProject](../content/posts/cmake/ExternalProject_Add/index.md) | 修复本地依赖源码编辑后外部构建未重新执行的问题；检查首次构建、无修改构建、源码修改与库文件丢失后的重建 |
| [udev](../content/posts/dialout/udev/index.md) | 补充同一序列号的多接口适配器，说明导入属性、父节点匹配、接口号和别名实际指向的验证 |
| [C++ 运行库](../content/posts/cpp/gcc/index.md) | 区分 GLIBCXX 符号版本与 libstdc++ 双 ABI，用四种组合验证“语言标准选项不等于 ABI 选择” |
| [InternVL 3.5](../content/posts/ai/internvl-3-5/index.md) | 仅修复后训练引用：改为官方模型卡实际存在的 Cascade Reinforcement Learning 章节 |

没有为增加篇幅重新改写已验证的 VLA、RTC 等长文；本轮对其关键接口、时钟和比较条件继续复核，保留已有表述。

## 实验与结果

| 实验 | 结果与解释 |
| --- | --- |
| Jacobian 参考点 | LWA 位置中心差分误差约 1.61e-10；错误地把 WORLD 前三行当成工具点速度时误差约 1.57。正确的对偶变换保持关节力矩与功率 |
| Pinocchio IK | 60 个可达目标通过独立 FK 检查；位置最大误差约 9.38e-5 m。几何长度换算到毫米并同步缩放残差后，关节解差约 3.33e-16 |
| 零空间力矩 | 正确动态投影的任务加速度增量约 5.31e-14 m/s²；误用速度投影器时约 16.82 m/s²，投影后逐关节裁剪也重新影响主任务 |
| 静态刚度与能量 | 在指定的刚性二连杆与关节弹簧模型中，接近伸直时一个方向的柔顺性趋近零；固定 10 mm 位移，将刚度从 100 提到 1000 N/m 增加 45 mJ 储能，单纯放慢斜坡不会消除它 |
| 外力残差符号 | 按 Mq̈+h=τ_act+JᵀF_ext，模型力矩减执行器力矩恢复 JᵀF_ext。静止与运动两例误差均低于 1.1e-14 N·m，正向动力学恢复加速度 |
| 可微 FK | 四组关节角下 API Jacobian、自动微分与差分一致；验证标准自动微分路径的二阶导数。十进制几何与加载模型的微小差异来自所测版本先按 FP32 解析变换 |
| ICP | 平面场景存在三个不受点到平面残差约束的方向；平移 0.1 m 的网格仍可取得 fitness=1 和极小 RMSE。该 API 的通用信息矩阵在此例满秩，不能替代点到平面 Hessian |
| 手眼标定 | 18 个训练姿态、6 个留出姿态，两条 Eye-to-Hand 路径一致。纯平移／单轴旋转时，一个偏差 0.1 m 的候选仍可保持近零闭合残差 |
| KV Cache 掩码 | 三种分块方式与完整因果注意力一致，误差约 2.22e-16；错误的左上对齐掩码在同一例中造成约 2.10 的输出最大差异 |
| 扩散参数化 | 7 个噪声水平的换算与损失比例、跳步条件高斯分布、4 个相邻 DDPM 步均一致；低 SNR 图比较的是误差传播，不是两类训练模型的质量 |
| 成组评估 | 固定构造数据的点估计同为 +10 个百分点。错误地把 400 行当独立样本得到 [3.25, 16.75] pp；按 20 个组重采样得到 [-20, 40] pp |
| KL 值与梯度 | 两条计算图的值同为约 0.233220 nats，但一个 logit 的梯度符号相反；精确求和、score-function 分解、可微重要性权重及差分相互核对 |
| CMake | Unix Makefiles 与 Ninja 均通过首次构建、无修改、依赖源码 42→43 和删除库后重建；无修改时没有重新编译库 |
| 双 ABI | GCC 11.4 与 12.3 各运行四种组合；ABI 一致的两组成功，不一致的两组按预期链接失败，未替换系统库 |

这些是明确模型、数据和版本下的数值或程序验证，没有控制真实机器人、重新训练论文模型或测量 GPU 性能。机器精度级残差不代表实机精度；单平面的局部秩不等于全局配准唯一性；固定位置的能量记账不是完整的被动性控制器；20 个组的区间没有经过覆盖率实验。

## 图与下载资源

本轮包含 3 张概念图和 6 张数值图。概念图使用 Imagegen，沿用白底、细边框和克制的蓝／紫／橙配色；数值图直接由程序绘制。图注解释对象、单位和假设，制作记录只放在文档目录。

- [Jacobian 参考点设计](image-generation/jacobian-reference-points-20260928-prompt.txt)：三个面板保持同一机械臂姿态，区分坐标轴旋转与参考点平移。记录明确区分设计稿与提交时的缩写提示。
- [ICP 可观测性设计](image-generation/icp-observability-20260928-prompt.txt)：单平面的三个自由方向与多法向约束。
- [接触力方向设计](image-generation/contact-force-directions-20260928-prompt.txt)及[修订提示](image-generation/contact-force-directions-20260928-edit.txt)：修正反作用箭头，使其从同一接触参考线指向工具一侧。

新增 15 个 Python 下载文件；其中 `frame_ik.py`、`check_ik.py` 提供与正文一致的完整入口，其余用于验证。13 个 JSON 结果与程序一起保存。强化学习和 LLM 指标两个 ZIP 已更新，成员与源文件逐项一致，并从解压目录执行验证。

关键环境：Hugo 0.165.0、Python 3.10、NumPy 2.2.6、Pinocchio 4.1.0、OpenCV 4.13.0、Open3D CPU 0.19.0、pytorch-kinematics 0.10.0、CMake 3.22.1、Ninja 1.13.2。数学绘图使用 Matplotlib 3.10.9。PyTorch 算子例子在现有 2.13.0 环境运行，KV Cache 与 KL 另在独立 2.8.0 CPU 环境检查；RL ZIP 按声明的 PyTorch 2.8.0 执行 19 项测试与全部 CLI 冒烟检查。

## 站点检查与保留项

全站浏览器检查覆盖 81 篇文章。13 篇修改文章共检查 117 组宽度／主题组合；后续阻抗与 Open3D 修订单独复核；InternVL 引用修正与最后的力矩平移公式修订，各另外检查了 9 组布局。视口包括 320、390 和 1440 px，主题包括浅色、深色和暖色，检查图注、图片加载、公式、横向溢出及图片放大。筛选与发现页面另外覆盖中文输入、查询 URL、历史返回、键盘、分页、无 JavaScript 和异常元数据回退。

生产与草稿构建各使用全新目录，正文验收规则保持原有 230 项。旧地址、原始日期、草稿标记及 2,400 个原有页面 ID 保留；头像、配置、主题子模块和前两轮报告未改变。文章源码、构建资源与 HTTP 下载内容逐项比较，防止旧下载包或遗漏资源。

外部链接的检查只代表本次可访问性，不代表逐项复现了来源结论。首批 625 个 URL 中，535 个返回 200；其余为 82 个 429、5 个 403、1 个 503、2 个握手超时，没有确认的 404 或 410。后来增加的 systemd 两条引用与 Pinocchio 外力源码引用均返回 200；最终合并检查 628 个 URL，其中 538 个返回 200。没有把限流或访问限制直接当作链接失效。另核对修改文章引用中的章节锚点，修复 Open3D 的 `Point-to-plane-ICP` 大小写；GitHub 行号链接单独对照源码。进一步检查全站 70 个外部章节链接，发现并修复 InternVL 模型卡缺失的 `post-training` 目标；最终 70 个目标均在返回的 HTML 中找到，分属 36 个页面。这不代表所有外部服务都取消了访问限制。

私有稿继续被 Git 忽略，未纳入文章清单；两种输出中的 HTML、JSON 和 XML 检查未发现私有路径或来源标记。没有修改系统 udev 规则、工具链默认项或已有 Python 环境；可选依赖安装在临时独立环境中。

复验命令（输出目录应为新目录）：

```bash
hugo --minify --destination /tmp/blog-round3-production
python3 -B scripts/validate_blog.py --public /tmp/blog-round3-production \
  --python-snippets --structured-snippets --strict-math
node scripts/test_blog_filter.cjs /tmp/blog-round3-production

hugo --minify -D --destination /tmp/blog-round3-drafts
python3 -B scripts/validate_blog.py --public /tmp/blog-round3-drafts \
  --include-drafts --python-snippets --structured-snippets --strict-math
node scripts/test_blog_filter.cjs /tmp/blog-round3-drafts --include-drafts
python3 -B scripts/test_blog_toc.py
python3 -B scripts/test_llm_metrics.py
python3 -B scripts/package_llm_metrics_lab.py --check
python3 -B scripts/package_rl_lab.py --check --test  # 独立 PyTorch 环境
```

具体数值脚本的命令、依赖与结果均链接在对应文章中。界面检查使用已有的本地 Chrome CDP，按串行运行；本轮临时预览结束后关闭，保留用户原有预览服务。
