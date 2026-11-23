# 复现指南

研究起点 `596d6e9302d70148ca00f4add1edc9b88b133bb6`，新实验在 `research/autonomous-motion-loop`。在仓库根执行，输出写 `work/`。依赖和模型模板路径沿用 [生成器 README](../experiments/generative_motion/README.md)。当前实验使用 Python 3.12.14、Torch 2.6.0+cpu、NumPy 2.3.5、SciPy 1.17.0、smplx 0.1.28、Matplotlib 3.10.8 和 FFmpeg。原模型资产需按原授权获取。

```bash
python scripts/check_handoff.py
python -m experiments.generative_motion.prepare
sha256sum -c experiments/SharedMotion/analysis/nsa_inputs.sha256
python -m scripts.shared_motion_data --prefix work/shared/nsa_prefix.bin \
  --train-central work/shared/train_central.bin --output work/generative_motion/nsa_pinned \
  --limits 600 75 100
```

样本清单 SHA-256 为 `e2a78804f64aa7959d1c9e221d04614010d202454de4091b144c598782b43c60`。若有原准备好的私有缓存，跳过下载。当前轮读取复用了上轮解码后的匹配数据。复验要保留相同官方划分和选样。

```bash
data=work/generative_motion/nsa_pinned
asset=work/models/smplx/SMPLX_NEUTRAL.npz
python research/datasets/make_tiny.py --dataset "$data" --output work/loop/tiny_replay
python -m experiments.generative_motion.train --dataset work/loop/tiny_replay \
  --asset "$asset" --config work/loop/tiny_replay/config.json --output work/loop/tiny_replay_run
for label in baseline e1 e2 e3_control e3_correlated rho95; do
  run=work/loop/replay_${label}
  python -m experiments.generative_motion.train --dataset "$data" --asset "$asset" \
    --config "experiments/generative_motion/flow_${label}.json" --output "$run"
  python -m experiments.generative_motion.generate --checkpoint "$run/best.pt" \
    --seeds $(seq 11 42) --output "$run/samples"
  python -m experiments.generative_motion.audit_samples --dataset "$data" --asset "$asset" \
    --samples "$run/samples" --output "$run/audit" --fk-count 8
  python -m experiments.generative_motion.evaluate --dataset "$data" --asset "$asset" \
    --samples "$run/samples" --output "$run/eval.json"
done
```

在 E0 或 E1 用 `--split test` 重跑审核，标签及测试参数固定。E0 数值积分诊断用 `generate --flow-steps 200` 在新输出目录采样，再跑相同 `audit_samples`。同种子 bootstrap 用 `python research/evaluations/compare_audits.py BASELINE/per_sample.csv CANDIDATE/per_sample.csv --output paired.json`。迭代分支统一使用结束前验证损失最低的 checkpoint。后续可以取 `--resume`，保持同一提交、环境、配置和数据路径语义。

只采样时准备仓库与用户可访问的实验包 checkpoint。视频标签、骨架、双手局部轨迹由现有 FK 工具输出：

```bash
python -m experiments.generative_motion.generate \
  --checkpoint /path/to/AutonomousSignMotion_Loop/runs/flow_e1/best.pt \
  --seeds 28 42 --output work/loop/quick_sample
python -m experiments.generative_motion.visualize \
  --motion work/loop/quick_sample/seed_28.npz work/loop/quick_sample/seed_42.npz \
  --model-dir work/models --output work/loop/quick_video
```

签语视频、原始拟合和授权模板留在用户有权访问的位置。公开 Git 只含代码、记录及聚合实验指标。外部 SignSparK 步骤及当前失败状态见 [复现记录](baselines/SIGNSPARK_REPRODUCTION.md)。
