本轮全站综合优化已按最新的后续1小时范围完成。

用户于2026-11-03 18:12:32 UTC将后续工作调整为1小时，已于2026-11-03 19:13:07 UTC结束。原8小时计划由这项最新要求替换；本目录沿用任务开始日期命名。所有改动保留在工作区，没有提交、推送或部署。

此前阶段已修复主题菜单的Escape/Shift+Tab焦点顺序、中文目录/归档/搜索/分类与语言菜单的语言标记、12处解释性表格的小屏阅读，并扩大HTML、XML及manifest图标的验证覆盖。全文RSS现在按每篇文章的地址解析导航和媒体URL，保留代码文本、摘要feed与既有条目选择。DDP重叠图及5张Transformer图的错误数值、形状或连线已更正；11处Transformer长公式做了等价分行。各阶段构建和浏览器检查范围在主JSON中分别注明，未将不同快照计数混成一次最终全站测试。

最新一小时重点处理三篇文章：六轴历史C++片段修复半开限位端点、腕部奇异分支的象限及2π周期；七轴SRS示例保留很小但非零水平投影的方位。扩散模型文章替换两张数据流有误的图，明确图像Query、条件Key/Value、预测头以及训练/采样的时间输入；缩短手机端图注，将细节留在紧邻正文。时间嵌入修复dim=1时补零列丢失的问题，并将三处扩散公式与一处七轴几何公式等价分行。

优化结束时的工作区生产构建检查324个HTML、33750条本地引用；草稿构建检查332个HTML、34419条引用，均无错误。27个SRI资产、2个manifest图标、17个主题覆盖及20个本地公式字体通过检查。后续只清理了未发布测试文件的末尾空行；再次全新构建的2324个输出文件与已验收生产构建逐字节相同。

局部数学验证包括2028组实际编译的C++用例、256组七轴分支及54组CPU PyTorch时间嵌入用例。4个故意恢复旧错误的C++变体均被测试拒绝。三篇修改页完成54个重排布局、36个打印布局；两张新图完成24个布局、48次图文检查和4次真实键盘放大。中英文入口的12个关键词检索与4个多词负例通过。语义补测各6个有/无脚本案例没有确定违规或缺失ID；有脚本时的40个关闭弹出层引用的人工判断项（菜单或对话框）原样保留。最终6份带标签PDF覆盖普通A4及无脚本200% Letter，图注与图片同页，焦点、滚动位置和折叠状态恢复。

保留了98篇文章及1个栏目页的全部front matter、标题列表和现有URL。开始时30个脏文件中28个当前仍逐字节相同；AGENTS.md和编辑审核JSON仅按本任务扩展。68文件的源码补丁排除了原有改动，通过独立临时仓库的正反向检查、实际重放与空白检查。owned-source-changes.json记录各文件的前后散列；源补丁位于/tmp/chase-site-eight-hour-20261103/owned-source-changes.patch。

仍有两项需要明确保留：全站额外200% A4打印扫描发现37篇文章的166处既有长公式越界，222/588布局受影响；这些页面的公式标注和样式与初始构建相同，详见hour-large-print-backlog.json与hour-large-print-baseline-proof.json。本轮三篇修改页的专项检查已通过。RSS正确性修复在最终2对受控构建中平均增加约3.81秒Hugo CPU时间，约31.75%；它与较早4对阶段的负载/快照不同，不能据差值宣称加速。

数值检查不认证完整历史六轴IK、原工程阈值调优、碰撞或实机效果；CPU嵌入检查不训练扩散模型。浏览器检查使用Chrome模拟的视口与字号，语义规则不替代真实读屏软件测试。原WebP资源保留，新SVG替换后不再输出6个未引用的响应式派生图。

主记录：../../site-wide-review-2026-11-03-eighthour.json
证据索引：evidence-manifest.json（保留历史失败与当前通过报告）
主要复现命令（Hugo生产输出需全新目录；所有CDP套件串行）：
- python3 -B scripts/test_kinematics_fragments.py
- python -B scripts/test_kinematics_fragments.py --numpy（独立NumPy环境）
- python -B scripts/test_diffusion_embedding.py（独立CPU PyTorch环境）
- python3 -B scripts/render_diffusion_flows.py --check
- node docs/site-reviews/2026-11-03-eighthour/check_diffusion_figures.cjs <baseURL> <CDP-port> <out>
- node docs/site-reviews/2026-11-03-eighthour/check_diffusion_print.cjs <baseURL> <CDP-port> <out> --large-text
- node docs/site-reviews/2026-11-03-eighthour/check_kinematics_reflow.cjs <baseURL> <CDP-port> <out>
- node docs/site-reviews/2026-11-03-eighthour/export_hour_native.cjs <baseURL> <CDP-port> <out>（需Poppler）

提交范围复核：上述优化阶段记录包含当时工作区既有的未提交Muon内容。本次提交已排除这些原有改动，在独立工作树中重新通过生产构建、324个HTML/33742条引用及主题资产检查；具体结果与重叠文档的提取方式见commit-review.json。
