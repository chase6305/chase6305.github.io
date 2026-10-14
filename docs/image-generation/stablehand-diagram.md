# StableHand 图示来源

- 页面：`content/posts/ai/stable-hand/index.md`
- 资产：`content/posts/ai/stable-hand/assets/stablehand-pipeline.svg`
- 依据：Zeng et al., [StableHand: Quality-Aware Flow Matching for World-Space Dual-Hand Motion Estimation from Egocentric Video](https://arxiv.org/abs/2605.18553), Fig. 2 与第 3 节。
- 制作说明：手工绘制的数据流图，未复用论文插图。图中将冻结感知/几何模块、世界坐标变换、质量网络和质量感知生成器分开；质量网络读世界坐标候选、检测标志与相机位姿，生成器另接手部特征、相机和场景条件。图示省略质量标定细节与损失项。
