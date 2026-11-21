# Stage 4 复现

运行环境为 Linux CPU、4 个 PyTorch 线程。使用之前的五段 SMPL-X、同一 Uni-Sign commit `eed438bcb49e30405cd6ccdfcccca330c134e830` 和 checkpoint。资产与下载步骤见 [STAGE3_REPRODUCE.md](STAGE3_REPRODUCE.md) 的第 2 节。下载真实 RGB 时只需 manifest 中五个含拟合的 clip。模型和 checkpoint 不在本包内。

```bash
pip install 'opencv-python==4.11.0.86' 'mediapipe==0.10.32'
curl -fL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o work/hand_landmarker.task
printf 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1  work/hand_landmarker.task\n' | sha256sum -c
python -m scripts.stage4_geometry --videos work/rgb --mp-model work/hand_landmarker.task --output experiments/Stage4/geometry
python -m scripts.stage4_feedback \
  --repo work/Uni-Sign --checkpoint work/how2sign_pose_only_slt.pth --mt5 work/mt5 \
  --motions work/motions --asset work/models/smplx/SMPLX_NEUTRAL.npz --stage4 experiments/Stage4
python -m scripts.stage4_compare experiments/Stage4
python -m scripts.stage4_review_media experiments/Stage4 --videos work/rgb
```

如只复核排名表，可直接运行 `stage4_compare`。为 4090 启用现有适配器可附加 `--device cuda`，实际报告来自 CPU。可在 [configs/stage4.json](configs/stage4.json) 核对分组、候选种子、误差阈值和重点帧。运行 `stage4_review_media` 还需要 ffmpeg（libx264）。审阅时只分发 `blind_review/`，身份映射在外层文件中。

How2Sign 源视频和叠图仅用于研究，版权及许可见 [How2Sign](https://how2sign.github.io/)。动作输入来源见 [E1 Real](experiments/E1_real/E1_REAL_RESULTS.md)。MediaPipe 结果由原始 RGB 重提。
