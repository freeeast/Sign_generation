# Generative 3D sign motion

当前研究从真实手语动作拟合学习共享分布，并从随机噪声生成连续 SMPL-X 动作。请从 [研究目标](docs/RESEARCH_OBJECTIVE.md)、[生成训练与采样](experiments/generative_motion/README.md) 和 [Codex 交接](docs/CODEX_HANDOFF.md) 开始。旧逐视频重建、专家闭环和 GRU 条件预测见 [历史实验索引](docs/PROJECT_HISTORY.md)。

## 历史迁移入口

本工程保存从真实手语视频出发的 SMPL-X 动作模仿、冻结 Uni-Sign 专家反馈闭环，以及跨录像共享动作预测的既有研究代码和实验记录。

当前工作点是 **SharedMotion**。已实现的短时 GRU 在部分滚动时长改善预测，长时滚动明显过平滑。共享先验接入 Stage 10 受约束专家闭环后，相对同起点、同预算专家对照尚未显示明确增益。结论见 [研究交接](HANDOFF_CONTEXT.md) 和 [最新摘要](experiments/SharedMotion/reports/EXECUTIVE_SUMMARY_CN.md)。

## 在 Codex 接手

1. 阅读 [AGENTS.md](AGENTS.md)、[HANDOFF_CONTEXT.md](HANDOFF_CONTEXT.md) 和 [START_HERE_CN.md](START_HERE_CN.md)。
2. 按 [Codex 环境说明](CODEX_CLOUD_SETUP.md) 安装最小推理依赖。
3. 先运行 `python scripts/check_handoff.py`，恢复权重后再加 `--smoke`，然后使用 [接手提示词](CODEX_START_PROMPT.md)。

仓库包含源代码、配置、历史报告和逐样本数值记录。三个小型 GRU 权重、原始视频、3D 拟合轨迹、衍生图像和完整演示留在原迁移包中。[数据恢复说明](CODEX_CLOUD_SETUP.md)提供按 SHA-256 校验的恢复命令，恢复脚本按完整迁移包的 SHA-256 核验来源。完整闭环还需要按原来源取得 SMPL-X 模型、Uni-Sign 权重及对应检测器。

## 研究入口

| 组件 | 入口 |
| --- | --- |
| 共享动作模型 | [src/shared_motion.py](src/shared_motion.py) |
| 数据与训练 | [scripts/shared_motion_data.py](scripts/shared_motion_data.py)、[scripts/shared_motion_train.py](scripts/shared_motion_train.py) |
| 共享先验接入专家闭环 | [scripts/shared_motion_feedback.py](scripts/shared_motion_feedback.py) |
| 固定相机、局部守门与回滚 | [Stage 10 复现](experiments/Stage10/README_STAGE10.md) |
| 最新定量结果 | [预测结果](experiments/SharedMotion/reports/MOTION_GENERATION_RESULTS.md)、[闭环对照](experiments/SharedMotion/reports/FEEDBACK_INTEGRATION.md) |
| 合并来源与旧版本 | [SOURCE_ARCHIVES.tsv](SOURCE_ARCHIVES.tsv)、[MERGE_CONFLICTS.tsv](MERGE_CONFLICTS.tsv)、`archives/superseded/` |
| 原始安装说明 | [历史 README](archives/superseded/github_migration/README.md)、[Stage 6](STAGE6_REPRODUCE.md)、[Stage 7](STAGE7_REPRODUCE.md) |

研究使用 NSA 拟合代理、How2Sign 真实观测和已冻结的模型。专家响应改善、几何改善与语言正确性分别报告。数据和第三方模型遵循原来源许可，项目迁移未改变其授权条件。
