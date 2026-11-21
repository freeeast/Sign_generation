# Stage 8 视频说明

- `33jxeIIbBnM_21-1-rgb_front_coarse_closed_loop.mp4` 是 49 帧短片。
- `3sZjSNcqXpI_2-1-rgb_front_coarse_closed_loop.mp4` 是 212 帧长片。第一格为真人，其余为粗拟合、纯几何、分阶段第 6/12/18 步。所有格子在同一 24 Hz 时间轴。
- 两个 `*_coarse_front_side.mp4` 是以上同一真实三维状态的正面和侧面骨架，侧面格是虚拟人的 3D 视图。
- `3sZjSNcqXpI_2-1-rgb_front_coarse_algorithm_comparison.mp4` 显示同一长录像的纯几何、分阶段和两个局部手指改进。
- `closed_loop_trajectories.png` 显示三个来源、三档起点的完整轨迹。第 6 步竖线是 C 系列从几何转向专家反馈的位置。

视频叠字的专家分数来自直接分支，held 是所有奇数留出帧的 MediaPipe 综合像素误差，edit 是相对共享初态的最大局部转角改动。真实 RGB 重跑和实际渲染后重新检测的指标见 `../rgb_validation/results.csv`。

`all_methods/` 包含三个来源乘三档初态的九段动画。每段有四列（纯几何、仅专家、分阶段、联合优化）和两行（正面、虚拟侧面）。第 0、6、12、18 步按顺序播放同一时间轴，叠字包含专家响应、留出误差、运动守门违规计数。
