# Stage 6 复现

在项目根目录运行。Stage 6 使用原 Stage 5 代码、固定专家权重、已有校准，并新增三个兼容接口。直接分支新增 4 段；RGB 分支另复用此前拳击片段用于配对诊断。Stage 6 归档含全部新报告、源级原始记录、图、视频、盲审材料、拟合参数及配对缓存。历史大型视频和旧实验输出保存在此前 Stage 5 归档中。

## 环境与固定资源

| 资源 | 本次固定版本或 SHA-256 |
| --- | --- |
| Uni-Sign | commit `eed438bcb49e30405cd6ccdfcccca330c134e830` |
| How2Sign pose-only SLT | `1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d` |
| SMPLX_NEUTRAL.npz | `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992` |
| YOLOX tiny 416×416 | `ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250` |
| RTMW 192×256 | `68db326c5c6f3273456f803eae0faf5440478369f2c3ee67258d2f36512e8d76` |
| MediaPipe HandLandmarker | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| How2Sign 原时序验证 TSV | `f6f0cef01c3f891ca624a45c71eb7f7748b77362d7418c47567a984384f1537c` |

实测 Python 3.12.14、Torch 2.6.0+cpu、OpenCV 4.11.0.86、MediaPipe 0.10.32、ONNX Runtime 1.22.1、ModernGL 5.12.0、SMPL-X 0.1.28。需 FFmpeg/libx264 和支持 OpenGL 3.3 的 Linux EGL。实测使用 Mesa llvmpipe 和 3–5 个线程。现有 Torch 验证环境可直接装新增依赖。
模型获取见 [项目 README](README.md)、[STAGE3_REPRODUCE.md](STAGE3_REPRODUCE.md) 及 [STAGE4_REPRODUCE.md](STAGE4_REPRODUCE.md)。SMPL-X 模板与公开 checkpoint 请用已有资产或各自官方途径取得。本包包含小样本的数值拟合参数，未打包预训练权重或 SMPL-X 模板。

```bash
python -m pip install 'onnxruntime==1.22.1' 'moderngl==5.12.0' 'opencv-python==4.11.0.86' 'mediapipe==0.10.32'
python -c "import moderngl; print(moderngl.create_standalone_context(backend='egl').info['GL_RENDERER'])"
```

## 重放已配对输入

下列命令把结果写到独立目录，可保留本包原始记录。根据本地路径调整共用变量。真实 RGB 通过盲审映射恢复精确文件名；拳击 RGB 取自已归档的 RGB 对照。审阅人员只接收 `blind_review/`，不接收外层映射表。

```bash
data=experiments/Stage6
out=work/replay_stage6
asset=work/models/smplx/SMPLX_NEUTRAL.npz
motion="$data/data/source_fits"
detector=$(find work/rtm/detector -name end2end.onnx)
mkdir -p "$out" work/rgb_s6
cp -r "$data/data" "$out/data"
python - <<'PY'
import csv
import shutil
from pathlib import Path
d = Path("experiments/Stage6")
o = Path("work/rgb_s6")
with (d/"blind_review_key.csv").open() as f:
    names = {r["clip"]:r["sample"] for r in csv.DictReader(f)}
for clip, folder in names.items():
    shutil.copyfile(d/"blind_review"/folder/"reference.mp4", o/f"{clip}.mp4")
clip = "3TrMyzNWGpY_4-2-rgb_front"
shutil.copyfile(d/"rgb"/clip/"real.mp4", o/f"{clip}.mp4")
PY
common=(--repo work/Uni-Sign --checkpoint work/how2sign_pose_only_slt.pth --mt5 work/mt5
        --asset "$asset" --motions "$motion" --output "$out")
pose=(--detector "$detector" --pose-model work/rtm/pose/end2end.onnx)
python -m scripts.stage6_direct "${common[@]}" --threads 5
python -m scripts.stage6_rgb "${common[@]}" "${pose[@]}" --videos work/rgb_s6 \
  --prior-motions "$motion" --prior-videos work/rgb_s6 --threads 3
python -m scripts.stage6_rgb_audit "${common[@]}" "${pose[@]}" --threads 3
python -m scripts.stage6_pose_diagnostics --output "$out"
python -m scripts.stage6_summary --output "$out" --asset "$asset" --videos work/rgb_s6 --threads 4
```

如需从源重新配对，可运行现有下载器（24 fps 原始切段）和 `src.extract_real_pose` 后，从 `stage6_prepare` 重建 `data/paired`。四个短片按 SHA 核验。若 RGB 镜像某个请求窗口遇到 HTTP 502，可用同一固定 revision 的更窄行窗口请求同一 `SENTENCE_NAME`。本轮不更换片段，使用窄窗口复取成功。

```bash
ids=(3xni-I6N3EY_3-1-rgb_front bcNtUYh3GNs_0-8-rgb_front EQWFrWeRVjQ_14-5-rgb_front 33YRZ5UZTA4_15-2-rgb_front)
mkdir -p work/nsa_stage6 work/motions_s6 "$out/data/real_pose"
python scripts/fetch_nsa_validation_prefix.py --output work/nsa_stage6 --max-bytes 6500000 -- "${ids[@]}" > work/nsa_stage6/inventory.txt
for id in "${ids[@]}"; do
  python -m src.nsa_motion "work/nsa_stage6/$id.zip" "$id" "work/motions_s6/$id.npz"
  python -m src.extract_real_pose --repo work/Uni-Sign "${pose[@]}" --threads 4 \
    --video "work/rgb_s6/$id.mp4" --output "$out/data/real_pose/$id.npz"
done
python -m scripts.stage6_prepare --inventory work/nsa_stage6/inventory.txt \
  --annotations work/annotations/how2sign_val.csv --asset "$asset" --motions work/motions_s6 \
  --videos work/rgb_s6 --mp-model work/hand_landmarker.task --output "$out"
```

候选自由度、权重和预算见 `configs/stage5.json`、`configs/stage6.json`、`experiments/Stage5/calibration_scales.json`。种子为优化轴 5061、冻结 PyTorch 0、盲审分配 6081。除相同帧格和同一相机外，数值复现受 CPU、ONNX 与 OpenGL 实现差异影响，临界候选可能选择不同更新。视频面向同项目科研审阅，数据出处见 [How2Sign](https://how2sign.github.io/) 与 [Neural Sign Actors](https://guanxiongsun.github.io/projects/neural-sign-actors/)。
