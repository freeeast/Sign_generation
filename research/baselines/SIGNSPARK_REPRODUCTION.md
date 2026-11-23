# SignSparK 复现状态

已克隆作者 commit `916402d0ecd6322fb4a9e5f513229a94a57bf4b6`。已核对 `sample.py`、`classifier_free.py`、`signspark/{rflow,unet_model,pose_datasets_lmdb}.py` 和推理配置。官方默认 304 帧（padding）、50 次 Euler 调用、三流分开。Hub 模型版本 `b907f25f9bb9f5a0adcaf131a23a155dc471d9a8`。body/face/hand 约 5.5805 / 5.5804 / 5.5817 GB，合计 16.743 GB。How2Sign test LMDB 587,984,896 B，数据 revision `9fe7e04ef59204c787cd6094c747f889efff6ba0`。

| 测试 | 源码查证与控制 | 执行状态 |
| --- | --- | --- |
| A 原条件 | 官方文本和数据 keyframes，经模型和 ODE 每步 clamp | 运行作者命令，启动失败 `ModuleNotFoundError: hydra`。无输出。 |
| B 仅关键帧 | 保留 `obs_mask`，强制 `y.uncond=True` 把文本特征置零；guidance=−1 只关闭 CFG | 已实现启动器，未加载权重 |
| C 仅文本 | `eval.keyframes_mask_all=True` 仅关闭关键帧，真实文本和数据长度仍进入 | 已查证配置，未加载权重 |
| D 纯噪声 | 手工指定 96 帧、全长 mask、零关键帧张量及布尔 mask、空文本与 `uncond=True`。不读取 LMDB。官方文档中的 mask-all 开关单独使用无法实现该测试。 | 启动器预检确认缺 Hydra、torchdiffeq、wandb、lmdb、blobfile。未验证 checkpoint 加载或此条件下质量。 |

官方安装基于 Python 3.10 + Torch 2.10 CUDA；当前为 Py3.12 / Torch2.6 CPU、约 13 GB 空盘。三流官方复现需要另有 CUDA 环境及容得下权重、文本编码器和数据的磁盘、内存。没有把占位张量或零权样本记录成官方结果。无条件模式存在分别丢弃文本和关键帧的训练配置，但发布权重能否做到稳定、协调的双手全身无条件结果尚需跑通。

获得授权及匹配硬件后，在仓库根用已检出的上游绝对路径和权重。

```bash
python research/baselines/run_signspark.py --repo work/upstream/SignSparK \
  --checkpoint /authorized/signspark/checkpoints/hand/ema_0.9999_200000.pt \
  --data-root /authorized/signspark/data --mode A --stream hand \
  --output work/loop/signspark --seed 11 --steps 50
```

将 mode 改成 B/C/D。A–C 上游写至 checkpoint 邻目录，D 写至指定 output，保留原始 upstream features。然后依作者可视化工具合并流，并记录 row rot6d、左右手镜像和帧率处理。该下载及推理命令尚未在官方环境成功。DOI、数据、版本及发布差异见 [作者工程](https://github.com/JianHe0628/SignSparK)。
