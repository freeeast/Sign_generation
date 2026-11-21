# E1-Real、E3 和 E3.5 复现

在项目根目录执行。环境、固定 Uni-Sign commit、checkpoint 与 mT5 下载命令沿用 [README](README.md) 第一节。下面的 `work/` 路径可替成现有依赖目录。默认 CPU 4 线程，float32，原时序，模型最多 256 帧，生成 beam 4。新数据使用 24 fps，81 对真人句子/姿态均无需抽帧。

## 1. 复跑现有真人结果

这个包已含九段原始取样的姿态数组、文件 SHA-256 和提取覆盖率。本命令输出真人侧 NLL、翻译、编码和排名。

```bash
python -m src.run_real_validation \
  --repo work/Uni-Sign --checkpoint work/how2sign_pose_only_slt.pth --mt5 work/mt5 \
  --manifest configs/validation_pairs.json --pose-dir experiments/E3/real_pose \
  --output work/replay
```

三个对照分别是原始 RTMW 分数、原可见点分数改成 1、仅面部遮罩。原模型的 69 点切片、body crop 尺度、手腕/面部锚点和 mask 逻辑不变。虚拟投影同样使用这套标准预处理。数据归属 How2Sign，科研用途，CC BY-NC 4.0。这里的姿态由视频重新提取。[How2Sign Dataset](https://how2sign.github.io/) · [RGB 镜像](https://huggingface.co/datasets/Flimdejong/how2sign/tree/8ebd757f151ab3cb74741d961abf93f0e36fff40)

## 2. 取五段拟合及精确 RGB

```bash
mkdir -p work/motions work/annotations
curl -fL https://dataverse.csuc.cat/api/access/datafile/45360 -o work/annotations/how2sign_val.csv
curl -fL https://dataverse.csuc.cat/api/access/datafile/45363 -o work/annotations/how2sign_realigned_val.csv

example=_2FBDaOPYig_1-3-rgb_front
extra=(0zvsqf23tmw_16-2-rgb_front 3TrMyzNWGpY_4-2-rgb_front bBHYQCDp_9A_4-3-rgb_front f5EGPzGSCJs_13-8-rgb_front)
python scripts/fetch_nsa_validation_prefix.py --output work/nsa --max-bytes 6500000 -- "${extra[@]}"
python -m src.nsa_motion "work/input/$example.zip" "$example" "work/motions/$example.npz"
for id in "${extra[@]}"; do
  python -m src.nsa_motion "work/nsa/$id.zip" "$id" "work/motions/$id.npz"
done

mapfile -t rgb_clips < <(python -c 'import json; print(*(r["clip_id"] for r in json.load(open("configs/validation_pairs.json"))), sep="\n")')
python scripts/fetch_validation_clips.py --annotations work/annotations/how2sign_val.csv --output work/rgb -- "${rgb_clips[@]}"
```

将用户原附件放到 `work/input/`。官方 Dropbox 把验证 ZIP 作为首个子包直接存储，下载器在固定字节数截断流，仅从条目顺序判断完整 clip，并校验解压长度和 CRC。若上游重打包导致目录顺序变化，重新打印前缀索引或从作者处单独请求上述 clip。拟合输入以完整 `SENTENCE_NAME` 连接，把压缩包的 `cam_trans` 独立存储。

可选重跑原时间证据。区间下载只传输官方 ZIP 的目录与单个 83 MB 原视频。

```bash
python scripts/fetch_how2sign_members.py --output work/raw -- _2FBDaOPYig-3-rgb_front
python -m src.verify_video_timing --raw work/raw/_2FBDaOPYig-3-rgb_front.mp4 \
  --clip "work/rgb/$example.mp4" --original-start 3.15 --realigned-start 4.78 \
  --output work/video_timing_audit.json
```

## 3. 从 RGB 重提 RTMPose（包内已有）

```bash
pip install opencv-python-headless==4.11.0.86 onnxruntime==1.22.1
mkdir -p work/rtm/detector work/rtm/pose
curl -fL https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip -o work/rtm/detector.zip
curl -fL https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip -o work/rtm/pose.zip
unzip -q work/rtm/detector.zip -d work/rtm/detector
unzip -q work/rtm/pose.zip -d work/rtm/pose
detector=$(find work/rtm/detector -name end2end.onnx)
for id in "${rgb_clips[@]}"; do
  python -m src.extract_real_pose --repo work/Uni-Sign --detector "$detector" \
    --pose-model work/rtm/pose/end2end.onnx --video "work/rgb/$id.mp4" \
    --output "work/poses/$id.npz"
done
```

按仓库 vendored 的轻量模式，检测器输入 416×416，RTMW 192×256。原始分数不会在本步截断。每段元数据中可核对原视频哈希，分数大于 1 的比例与有效点比例。

## 4. 用取得的 SMPL-X 模型运行虚拟侧

模型文件放在 `work/models/smplx/SMPLX_NEUTRAL.npz`，与原拟合的模型版本一致为佳。模型下载需官网研究许可注册。[官方入口](https://smpl-x.is.tue.mpg.de/download.php) 

```bash
common=(--repo work/Uni-Sign --checkpoint work/how2sign_pose_only_slt.pth --mt5 work/mt5
        --manifest configs/validation_pairs.json --pose-dir experiments/E3/real_pose
        --motion-dir work/motions --model-dir work/models --hand-mean model)
python -m src.run_real_validation "${common[@]}" --reuse-real experiments/E3 --output work/paired
python -m scripts.audit_paired_alignment --manifest configs/validation_pairs.json \
  --poses experiments/E3/real_pose --videos work/rgb --geometry work/paired/E1_real \
  --asset work/models/smplx/SMPLX_NEUTRAL.npz
python -m scripts.summarize_paired_recognition work/paired/E3
```

原相机逆变换可用 `--camera-dir work/camera` 提供。每个拟合用以完整 ID 命名的 NPY，形状 `(2,3)` 或 `(T,2,3)`，将归一化 crop 坐标映射到归一化图像坐标。缺原相机时通过偶数帧的五个上身点拟合正向 xy 缩放和平移，只在固定相机后度量奇数帧。对齐候选必须保持同一帧栅格。等帧数是对齐的线索，需用 FK 检查实际对应。OSX 预设为公开实现的虚拟相机，包含 5000 焦距和 192×256 crop。

审核 `E1_real/geometry_qc.json` 和 3D、投影图后，复用相同配置可运行局部对照。

```bash
python -m src.run_real_validation "${common[@]}" --reuse-real experiments/E3 --output work/paired --probe-only
python -m scripts.summarize_local_probes work/paired/E3_5
python -m scripts.plot_correction_contrasts work/paired/E3_5
```

第一组命令复用真人 NLL 矩阵和编码，仅为跨域全词表分布计算真人侧所需的额外前向，并生成 `E3/virtual_*` 和 `E3/domain_gap.csv`。第二组读取固定的虚拟骨架与相机，生成局部排序、曲线和反例图，复用真人编码。若只想复核汇总图和表，可以把以上两条汇总脚本的输入替换为包内的 `experiments/E3` 和 `experiments/E3_5`。本轮没有执行 E4。

## 5. 结果入口

- [E1_REAL_RESULTS.md](experiments/E1_real/E1_REAL_RESULTS.md)
- [E3_RESULTS.md](experiments/E3/E3_RESULTS.md)
- [E3_5_RESULTS.md](experiments/E3_5/E3_5_RESULTS.md)
- [GO_NO_GO_DECISION.md](reports/GO_NO_GO_DECISION.md)
- [NEXT_STAGE_PLAN.md](reports/NEXT_STAGE_PLAN.md)
