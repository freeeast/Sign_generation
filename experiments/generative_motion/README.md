# 无条件 3D 手语动作生成

`prepare.py` 核验官方档案快照。旧 `scripts.shared_motion_data` 解析档案并保留源录像划分。`motion.py` 定义 24 Hz 窗口与 SMPL-X 轴角往返。`diffusion.py`、`train.py` 和 `generate.py` 实现共享 DDPM 与纯噪声采样。`evaluate.py` 计算划分内评价，`visualize.py` 导出双手、前视和侧视 FK 预览、133 点轨迹。模型、视频、样本和完整日志在 `work/`，聚合结果在 `results/`。

## 环境与准备

在仓库根目录。以下以 Python 3.12 和 CPU 为例，CUDA 机器使用与驱动匹配的 Torch 2.6.0 wheel。预览另需系统 FFmpeg/libx264。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu
python -m pip install -r experiments/generative_motion/requirements.txt
python scripts/check_handoff.py
python -m experiments.generative_motion.prepare
sha256sum -c experiments/SharedMotion/analysis/nsa_inputs.sha256
python -m scripts.shared_motion_data \
  --prefix work/shared/nsa_prefix.bin --train-central work/shared/train_central.bin \
  --output work/generative_motion/nsa_pinned --limits 600 75 100
```

将授权的 `SMPLX_NEUTRAL.npz` 放在 `work/models/smplx/`，哈希见 [历史复现](../../STAGE6_REPRODUCE.md)。快照下载入口来自项目既有 NSA 作者链接。两个快照约 1.92 GB。服务器忽略训练目录 Range 时，准备器会停止，读入同哈希的快照可继续。数据、权重和生成资产遵循来源许可，保留在本地或获授权的私有存储。

## 训练、生成与评估

```bash
data=work/generative_motion/nsa_pinned
asset=work/models/smplx/SMPLX_NEUTRAL.npz
run=work/generative_motion/cpu_motion_loss
python -m experiments.generative_motion.train --dataset "$data" --asset "$asset" \
  --config experiments/generative_motion/motion_loss.json --output "$run" --device cpu

# 96 帧和 48 帧均从训练长度 96 的完整噪声窗口采样，48 帧为前半段。
for frames in 48 96; do
  python -m experiments.generative_motion.generate --checkpoint "$run/best.pt" \
    --num-frames "$frames" --seeds 11 22 33 44 55 66 77 88 --output "$run/generated_$frames"
  python -m experiments.generative_motion.evaluate --dataset "$data" --asset "$asset" \
    --samples "$run/generated_$frames" --split val --output "$run/eval_$frames.json"
done
python -m experiments.generative_motion.visualize \
  --motion "$run/generated_96/seed_11.npz" "$run/generated_48/seed_22.npz" \
  --model-dir work/models --output "$run/video"
```

已训练权重用于纯噪声推理，生成命令无需模型模板或数据清单。若已有归档 GRU，给评价追加 `--gru-checkpoint experiments/SharedMotion/model/gru_context16_h3.pt`。对照在 12 Hz 下使用真实历史，单独报告。更换为 `pilot.json` 或 `config.json` 复验 500 步 / 5,000 步噪声预测，分别用 `cpu_pilot` 与 `cpu_baseline` 作为输出。可用 `python -m experiments.generative_motion.compare --dataset "$data" --asset "$asset" --gru-checkpoint experiments/SharedMotion/model/gru_context16_h3.pt --output "$run/comparison"` 导出同长度的 NSA、GRU、Hold 视觉对照。

关于实际跑过的环境、步骤和验证结果见 [训练报告](TRAINING_REPORT.md)。
