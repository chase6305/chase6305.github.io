# 博客第五轮：整体查找与阅读体验

日期：2026-09-05。延续前四轮，未提交、推送或部署。本轮主要优化整体入口与阅读体验，不重复重写全部文章：66 篇均纳入共享改进与回归；正文及下载脚本的实质修订集中在阻抗控制日志分析。逐篇范围见 [blog-fifth-pass.json](blog-fifth-pass.json)。

## 查找文章

- 博客列表新增关键词、10 个专题和排序组合。关键词按空白拆分、同时匹配，支持中文、大小写与全角字符；C++、A* 按普通文本处理，不执行正则表达式。
- 筛选使用标题、摘要、分类、专题和标签，覆盖全部文章而非当前分页。正文全文搜索仍使用原导航栏搜索，页面明确区分二者。
- 提供最新发布、最早发布和阅读较短三种顺序。发布日期排序不冒充更新时间；阅读时长是估算值，不包含实际练习时间。
- 筛选状态写入当前 URL，可刷新、分享并响应历史返回；普通输入做短暂防抖，中文输入法组词期间不提前筛选。
- 默认仍显示 Hugo 的原生分页。关闭 JavaScript 或筛选数据解析失败时保留原列表；空结果有提示和清除入口，清除后恢复原来页码与键盘焦点。
- 增加专题路线、年份归档和 RSS 订阅入口。博客标题、简介和卡片语言与中文正文保持一致。

索引只包含已有元数据，正式版共 33,458 字节，草稿预览共 35,068 字节；筛选脚本只在博客列表加载。未引入前端框架、额外 CDN、分析埋点或远程查询服务。动态卡片通过 DOM 文本节点创建，不把查询词插入 HTML；构建时解码摘要实体，并逐篇与原摘要比较，避免 C++ 被显示为转义字符串。

## 共享模板与导航修复

博客列表与分类/标签详情页共用 [card.html](../layouts/_partials/blog/card.html)，统一日期格式、阅读时长、摘要、标签与文章语言。草稿在预览单页、卡片、归档和首页最新文章组件中明确标记；正式输出不含 3 篇草稿。

扩大浏览器检查后发现分类、标签详情、归档和项目页缺少 Hextra 移动导航容器，菜单脚本会对空节点调用 removeAttribute。已在仓库自有布局中加入相同导航 partial，验证手机菜单展开、Escape 关闭、aria 状态和桌面显示；没有改主题子模块。

手机筛选区域使用紧凑布局，390 px 下专题与排序并列，320 px 下纵向排列，控件触摸高度至少 44 px。另检查 768 px 与 1440 px，以及浅色、深色、暖色三种主题。

## 日志分析的实质修复

[analyze_impedance_log.py](../content/posts/robotics/control/impedance-control/analyze_impedance_log.py) 原先仅用 actual > limit 判断失败；当指标为 NaN 时比较返回假，可能使无效指标通过验收。本轮：

- CSV 和直接 API 调用均验证最少样本、必要字段、有限数值与严格递增时间戳。
- 拒绝重复表头、缺失数值与没有表头对应的多余单元格。
- NaN、正负无穷等非有限指标一律判为验收失败。
- 正文说明故障注入复验方法，以及本例零基线、平均采样周期和 JSON 阈值文件的适用边界。

这是离线教学日志工具的输入与失败处理修复，不代表在真实机器人上完成控制器安全验收。未改变控制增益、力矩输出、已有仿真结果图或实机配置。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| Hugo | hugo --minify、独立生产构建、含草稿预览均通过 |
| 正式静态检查 | 66 篇源文章，63 个 BlogPosting，196 个 HTML；其中 186 个博客/集合页面、7 个完整筛选索引、12,178 处站内引用通过 |
| 草稿静态检查 | 66 个 BlogPosting，206 个 HTML；196 个博客/集合页面、7 个完整筛选索引、12,685 处站内引用通过 |
| 代码语法 | 342 个围栏；149 Python、96 Bash、8 sh、3 JSON、6 JSONC、1 XML 通过对应语法检查；shell 不执行正文命令 |
| 原浏览器回归 | 66 篇逐页检查 390 px 与 1440 px，图片、公式、导读、自测、目录、专题导航、搜索、复制与图片放大通过 |
| 新筛选回归 | 正式版 63 篇与预览 66 篇分别验证；14 组查询状态、10 个专题、跨分页查询、URL 刷新、历史返回、中文输入、空结果和文字注入测试通过 |
| 降级与无障碍 | 无 JavaScript、损坏 JSON 回退；Tab 顺序、重置后焦点、菜单展开与 Escape 关闭通过 |
| 布局与主题 | 320 / 390 / 768 / 1440 px × 浅色 / 深色 / 暖色，无横向溢出 |
| 站点入口 | 分类、标签、归档、学习路线、中英文首页、中英文项目页和标签详情的双宽度与移动菜单检查通过 |
| 阻抗日志 | 251 条正常样本通过；估计刚度约 1000.000034 N/m；20 组无效 API 输入、10 组无效 CSV、18 组非有限指标均拒绝；CLI 正常/损坏数据退出状态正确 |
| 前轮回归 | C++/CMake、A*、SPSC、PID、SRS、Pinocchio、通信回环、轨迹、标定、PyTorch、tqdm，以及第四轮核心/Jacobian 回归再次通过 |

本轮新增测试未安装依赖。复用现有环境，Qt/硬件/GPU 专属运行未重复验证；DDPM 只复跑原 20 步冒烟测试，不冒充重新进行了完整训练。清理了本轮导入检查产生的一份源码目录字节码及其临时预览副本，并让新测试禁止在文章 bundle 写入字节码；这些是可重新生成的缓存。

public 仍可能含此前预览遗留文件，未作宽泛清理；正式发布隔离的结论来自独立构建目录。配置、工作流、主题子模块和系统设置未改动；站点布局改动包含项目页移动导航，应与博客布局一起审阅。

## 文件与复验

- [blog-filter.js](../assets/js/blog-filter.js)：筛选、URL 状态与渐进增强。
- [test_blog_filter.cjs](../scripts/test_blog_filter.cjs)：纯逻辑与正式/草稿精确覆盖测试。
- [check_blog_discovery.cjs](../scripts/check_blog_discovery.cjs)：交互、无脚本回退、主题与站点入口检查。
- [test_blog_round5.py](../scripts/test_blog_round5.py)：离线日志异常回归。

```bash
hugo --minify
hugo --minify --destination /tmp/chase-blog-round5-production
python3 scripts/validate_blog.py --public /tmp/chase-blog-round5-production \
  --python-snippets --structured-snippets
node scripts/test_blog_filter.cjs /tmp/chase-blog-round5-production
python3 scripts/test_blog_round5.py
python3 scripts/test_blog_examples.py --network
# 使用第三轮依赖环境：
python scripts/test_blog_round3.py
python3 scripts/test_blog_round4.py --groups core jacobian
# 先启动对应 Hugo 预览和 Chrome 本地调试端口：
node scripts/check_blog_browser.cjs http://127.0.0.1:13143 9229 /tmp/chase-blog-round5-browser
node scripts/check_blog_discovery.cjs http://127.0.0.1:13143 9229 \
  /tmp/chase-blog-round5-discovery --include-drafts
```

浏览器 JSON 与截图保存在 /tmp/chase-blog-round5-browser、/tmp/chase-blog-round5-discovery 和 /tmp/chase-blog-round5-production-browser。重点截图为 filter-390-light.png、filter-390-dark.png、filter-1440-warm.png、no-javascript-page-2.png；前四轮截图保留，可作对照。

本轮不新增生成图：改进集中在交互与实际数据验证，保留现有有效配图。后续需要生成说明图时，仍以 transformer-attention/assets 的既有风格为准。
