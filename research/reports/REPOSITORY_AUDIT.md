# 仓库与执行环境

起点 `596d6e9302d70148ca00f4add1edc9b88b133bb6`，新分支 `research/autonomous-motion-loop`。已核对 Git 跟踪目录、入口、历史报告、checkpoint 路径及现存授权资产。Python 3.12.14 / Torch 2.6.0+cpu / NumPy 2.3.5 / SciPy 1.17.0，9 CPU 核、9.7 GiB RAM、约 13 GB 空盘、FFmpeg 和 SMPL-X neutral 可用，CUDA 不可用。

| 区域 | 可复用入口与证据 |
| --- | --- |
| Stage 5–10 | 逐视频 reconstruction、冻结 Uni-Sign 的双手特征、固定相机和受约束细化。仍在 `experiments/Stage*`，见 [历史索引](../../docs/PROJECT_HISTORY.md)。 |
| 数据 | How2Sign 条目与配对工具在 `src/nsa_motion.py`、`scripts/fetch_how2sign_members.py`、`scripts/shared_motion_data.py`；源级 manifests 与 CRC 检查已复用。 |
| 运动 | `src/shared_motion.py` 的轴角/列式 6D、均值和 SO(3)；`src/smplx_to_pose.py` 官方 FK；`src/stage9_hand.py` 的可微 FK 与手指关节；现有生成训练不使用单独 MANO。 |
| GRU | 条件预测、Hold 与短时测试在 `experiments/SharedMotion/`。旧权重 SHA 及权限见历史索引。4 秒手指速度约参考 7%，专家闭环无测得增益。 |
| DDPM | 现存 clean+velocity 权重可加载，短于 96 帧时导出为完整采样的裁剪。新增 Flow 复用 `train.py`、`generate.py`、`motion.py` 及 [历史评价](../../experiments/generative_motion/EVALUATION_REPORT.md)。 |
| 视觉与评价 | `visualize.py` FK 双手和前侧固定视角，`evaluate.py` 聚合评价。新增 `audit_samples.py` 做源分离逐样本审核，输出阈值、近邻覆盖指标和可选的 FK 统计。 |
