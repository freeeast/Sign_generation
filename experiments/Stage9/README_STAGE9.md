# Stage 9 交付索引

将下载的各个 Stage 9 ZIP 解压到现有 `sign_motion_research` 工程根目录。每卷的路径相同、内容互补。Stage 8 的代码、权重配置和基线输出由已有工程提供。

| 交付卷 | 内容 |
|---|---|
| Stage9_research.zip | 八份报告、两个配置、Stage 9 源码、逐指/逐关节审计、等预算比较、RGB 重检测、曲线、SHA256、计时对照 |
| Stage9_motion_trajectories.zip | 前三来源的主实验、四种反馈消融、余量诊断、全部保存的真实动作参数和各提案状态 |
| Stage9_additional_source.zip | DKOz 的 Stage 8 对照、粗起点、Stage 9 参数与留出/RGB 比较 |
| Stage9_visual_demonstrations.zip | 双手独立控制和特写、同步前侧面、优化中间检查点、抖动前后和额外一条来源 |

源视频、受许可约束的模型和现有 Stage 8 原始归档独立复用。所有动作检查点保留 SMPL-X 旋转，可依 `S9_ARTICULATED_CONTROLLER.md` 复现关键点与网格。前三段只在焦点联合/联合几何保留重复的笛卡尔检查点，其他状态用相同 FK 重建即可。RGB 数据保留重检测坐标、置信度和实际使用的帧。

## 入口

- [执行摘要及组会简报](reports/S9_EXECUTIVE_SUMMARY.md)
- [手指控制审计](reports/S9_HAND_CONTROL_AUDIT.md)
- [可复现实装](reports/S9_ARTICULATED_CONTROLLER.md)
- [抖动定位与稳定性消融](reports/S9_TEMPORAL_STABILITY.md)
- [闭环实验、RGB 复测与失败案例](reports/S9_CLOSED_LOOP_RESULTS.md)
- [完全相同顺序和预算的基线](reports/S9_BASELINE_COMPARISON.md)
- [Go/No-Go](reports/S9_GO_NO_GO.md)
- [下一步研究](reports/NEXT_RESEARCH_STAGE.md)
