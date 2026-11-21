# 历史基线与可复用代码

历史 Git 快照为 `fd65b0808fca37ca7b4b833077a73f695171bc42`。历史原始报告、配置、数值结果和代码保留在原路径。

## A · 逐视频重建与专家反馈

目的为检验真实录像与同片拟合的姿态专家响应，以及专家能否提供几何之外的细粒度修正。早期五段真实拟合完成匹配输入验证，编码检索 5/5，NLL 配对 16/20；局部指、腕修正方向只命中 2/5、1/5。在 Stage 5–8 建立逐片受约束闭环和 RGB 复评，Stage 7 修正重复右手的特征索引。Stage 9 实现独立双手关节和可微反馈。Stage 10 的固定相机与逐手约束在匹配起点的六段视频中使直接专家距离改善 6/6，RGB 复评改善 2/6。3sZj 第 90 帧左手过度张开，61.85→29.08 px 的局部改善主要由几何重锚定带来。强几何基线、检测器域差及缺少独立语言与强侧视验证限制了结论。

| 模块 | 可复用入口 |
| --- | --- |
| NSA 与 How2Sign | `src/nsa_motion.py`、`scripts/shared_motion_data.py`、`scripts/fetch_how2sign_members.py` |
| 关节序、双手均值、6D 和 SO(3) | `scripts/shared_motion_data.py` 的 ACTIVE/canonical，`src/shared_motion.py` 的 rotations/decode/six |
| FK、COCO-WholeBody、SMPL-X mesh | `src/smplx_to_pose.py` 的 forward，`src/stage9_hand.py` 的可微 FK |
| 约束和反馈 | `src/stage10_guard.py`、`src/stage10_world.py`、`src/stage7_feedback.py`、`scripts/stage10_run.py` |
| RGB、检测与评价 | `src/stage6_render.py`、`src/extract_real_pose.py`、`scripts/stage10_{rgb,fingers,side,visuals}.py` |

各阶段的可执行入口和依赖记录在 [S5](../experiments/Stage5/S5_CLOSED_LOOP_IMPLEMENTATION.md)、[S6](../STAGE6_REPRODUCE.md)、[S7](../STAGE7_REPRODUCE.md)、[S8](../experiments/Stage8/STAGE8_CLOSED_LOOP.md)、[S9](../experiments/Stage9/README_STAGE9.md)、[S10](../experiments/Stage10/README_STAGE10.md)。完整回归需要用户授权的配对缓存、SMPL-X、Uni-Sign，以及 RGB 路径使用的检测模型和 EGL。本轮核对历史代码和报告，未重新运行 Stage 5–10 实验。权重和模型文件路径见 [资产恢复](../CODEX_CLOUD_SETUP.md)。

## B · 共享 GRU 条件预测

从真实历史预测腕指未来，使用 43 个活跃局部关节和 SO(3) 增量，名义 12 Hz。原实验按录像分离 600/75/100 段，三组配置中选定 16 帧种子、3 步训练目标的 165,633 参数 GRU。0.5 秒手部角误差 Hold 22.06°、GRU 20.90°；1 秒 25.61°、23.95°；4 秒反转为 29.72°、35.23°，此时 GRU 手指速度约为 NSA 拟合参考的 7%。四段同起点、同守门、36 提案的闭环比较里，加入 GRU 先验后的最大留出综合几何差约 0.004 px，RGB 复评近乎持平。它因此只作为历史条件预测和运动先验基线。

外部权重位于迁移 ZIP 的 `experiments/SharedMotion/model/`，由 `scripts/restore_migration_assets.py` 恢复。完整预报和闭环命令见 [SharedMotion 复现](../experiments/SharedMotion/README_SHARED_MOTION.md)，评价表见 [闭环记录](../experiments/SharedMotion/feedback/integration.csv)。

| 权重 | SHA-256 |
| --- | --- |
| gru_context16_h3.pt，验证选定 | `be5f569913e288b15cd75246a17dbdde6efe50a7233b946299fa5ced49c495b7` |
| gru_context8_h1.pt | `445e8dfda1b98d5a66459de1fd63ce8d0f84ee3eae9e8a1e020b9006d4096daa` |
| gru_context8_h3.pt | `d399ff19b486c686a18d8a5e0b9830766d8fcbda6004d6954967f901af8c5b47` |
