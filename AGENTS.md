# 项目指令

本工程研究如何让受运动约束的 SMPL-X 化身从真人手语视频和冻结 Uni-Sign 专家反馈中学习模仿。当前工作点是 SharedMotion，最新可用的逐视频闭环来自 Stage 10。先读 START_HERE_CN.md 和 HANDOFF_CONTEXT.md。根 README.md 是早期历史入口。

- 从仓库根目录运行命令。先执行 `python scripts/check_handoff.py`。启用适用依赖后用 `--smoke` 和 `--assets`。检查通过的内容按测试层级报告。
- 保留 `experiments/E*`、`experiments/Stage*` 和 `experiments/SharedMotion` 的归档记录。复现写入 `work/`，新研究使用新的输出目录。保留实际运行命令、依赖版本、配置、随机种子、源级划分、原始指标和拒绝的提案。
- 复用 `src/shared_motion.py`、`scripts/shared_motion_*`、`configs/stage10.json` 和纠正后的双手特征。保留关节约定、手均值补偿、固定相机、时序采样、可见性和每手约束。使用相同起点、提案预算和守门比较闭环方法。
- 按原录像 ID 分隔训练、验证与测试，评价帧不可参与更新选择。NSA 是 3D 拟合代理，留出 2D 观测依然带检测偏差。专家目标、运动质量和语言正确性分别报告。没有真人语义验证时明确说明。
- 保护第三方数据和权重。将模型与环境配置在 `work/` 或本机目录。重用已授权缓存。遵循 NSA、How2Sign 和 SMPL-X 的来源及许可。小规模复验无需完整数据集。
- `parallel/ASL_SMPLX_UniSign_Pilot/` 是独立的文字到动作探索，结果不归入 SharedMotion。旧版同名文件存于 `archives/superseded/`，只用于历史查证。

## GitHub 工作副本

先读 CODEX_CLOUD_SETUP.md。默认 `check_handoff.py` 验证已提交工程，加 `--data` 检查已恢复的研究数据。使用 `scripts/restore_migration_assets.py` 从经校验的完整迁移 ZIP 恢复外部资产。完整资产恢复与最小 GRU 推理的可运行范围分别报告。

GRU 权重从用户提供的原迁移 ZIP 恢复后再运行 --smoke，公开副本未提交二进制权重。
