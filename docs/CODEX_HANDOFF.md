# Codex 交接

当前分支 `research/generative-motion`，历史基点 `fd65b0808fca37ca7b4b833077a73f695171bc42`。新代码位于 `experiments/generative_motion/`，主任务为无条件生成。可复用组件、历史重建、专家反馈与 GRU 对照见 [PROJECT_HISTORY.md](PROJECT_HISTORY.md)。已有 pose、数据与视觉工具原地保留。

## 环境和数据

- Python 3.12.14、Torch 2.6.0、NumPy 2.3.5、SciPy 1.17.0。FK 预览另用 smplx 0.1.28、Matplotlib 3.10.8、FFmpeg/libx264。CPU 已实测，小模型无需 GPU。正式扩展用可用 CUDA 单卡，复核安装与峰值内存再定 batch。
- NSA 官方 How2Sign 拟合快照索引为 train 30,965、val 1,724、test 2,318 段。两个快照 SHA 已核验，旧抽样 600/75/100。模型训练仅消费满足 96 帧的 train/val。三划分按源录像检查。签者级独立性未确认。
- 需要约 1.92 GB 供两个压缩快照，另给拟合 NPZ 和实验产物保留余量。SMPL-X 模板通过用户授权路径获得，默认 `work/models/smplx/SMPLX_NEUTRAL.npz`。Uni-Sign 和 RTMW 仅用于历史闭环。
- 旧 GRU 权重在有授权的完整迁移包，公开 Git 不承载训练权重与实验数据。

## 接手命令

```bash
git fetch origin research/generative-motion
git switch research/generative-motion
python scripts/check_handoff.py
python -m experiments.generative_motion.prepare
```

进入 [生成实验 README](../experiments/generative_motion/README.md)，依次执行解包、训练、噪声采样、独立评估和 FK 预览。恢复某个已中断的训练时，从同一代码提交、设备、配置和数据运行原 `train` 命令并追加 `--resume`。选择新训练配置时给新输出路径。新 Codex 会话可以使用以下提示词。

> 读取 AGENTS.md、docs/CODEX_HANDOFF.md、docs/RESEARCH_OBJECTIVE.md 和新生成报告。复用现有 DDPM 和 SMPL-X 接口，先复现已记录的噪声采样及验证对照。下一里程碑是在已冻结的验证方案上做 clean-pose ± 速度项消融，改善 4 秒动态与质量，同时保留样本间多样性。优先做一个有针对性的同预算对照。训练只用 train，配置选择只用 val，最终锁定再用 test。保存 checkpoint、源划分、真实命令、随机种子和多个种子生成视频。
