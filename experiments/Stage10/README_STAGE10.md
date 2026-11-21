# Stage 10 复现与入口

将研究与动作包解压到已有 `sign_motion_research` 根目录。两个 ZIP 的内容互补。Stage 9、Stage 8 及 Stage 7 的主包和运动、侧视分卷提供开发源。Stage 6 主包提供两个锁定源的真实视频、2D 与 NSA 配对缓存。模型权重、素材原始许可证和安装见 `STAGE6_REPRODUCE.md` 与 `STAGE7_REPRODUCE.md`。新增实验与旧 Stage 9 文件分开保存。

命令在工程根执行，默认固定配置 `configs/stage10.json`。建议另用独立输出目录避免覆盖原始 Stage 10 结果。完整运行需要 CPU Torch 2.6、OpenCV、SMPL-X、ONNX Runtime、FFmpeg 和 Mesa EGL。固定 Uni-Sign commit `eed438bcb49e30405cd6ccdfcccca330c134e830`。

```bash
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
out=experiments/Stage10_replay
asset=work/models/smplx/SMPLX_NEUTRAL.npz
repo=work/Uni-Sign
weights=work/how2sign_pose_only_slt.pth
detector=$(find work/rtm/detector -name end2end.onnx)
pose=work/rtm/pose/end2end.onnx
core=(--repo "$repo" --checkpoint "$weights" --asset "$asset" --output "$out")
geometry=(--asset "$asset" --output "$out")
rgb=(--detector "$detector" --pose-model "$pose")
python -m scripts.stage10_audit "${geometry[@]}"
python -m scripts.stage10_run "${core[@]}" --partition development
python -m scripts.stage10_run "${core[@]}" --partition locked
python -m scripts.stage10_control "${core[@]}"
python -m scripts.stage10_rgb "${core[@]}" "${rgb[@]}" --jitter
python -m scripts.stage10_rgb "${core[@]}" "${rgb[@]}" --partition development
python -m scripts.stage10_rgb "${core[@]}" "${rgb[@]}" --partition locked
python -m scripts.stage10_side "${geometry[@]}"
python -m scripts.stage10_fingers "${geometry[@]}"
python -m scripts.stage10_summary --output "$out"
python -m scripts.stage10_visuals "${geometry[@]}"
python -m scripts.stage10_visuals "${geometry[@]}" --sources bcNtUYh3GNs_0-8-rgb_front 33YRZ5UZTA4_15-2-rgb_front
```

全部候选和轨迹保留局部轴角、人体根平移、固定相机及真实参考缓存。在核心 ZIP 中附主要数值、逐帧探测和视频，在动作 ZIP 中附 3D 参数及可重放系数。受限权重和源拟合数据依照旧项目获得。

## 报告

- [执行摘要](reports/S10_EXECUTIVE_SUMMARY.md)
- [相机与根坐标审计](reports/S10_CAMERA_AND_ROOT_AUDIT.md)
- [坐姿与下肢稳定](reports/S10_LOWER_BODY_STABILITY.md)
- [手形约束与第 90 帧失败](reports/S10_HANDSHAPE_CONSTRAINTS.md)
- [专家细化与回滚](reports/S10_EXPERT_REFINEMENT.md)
- [完整闭环与锁定源](reports/S10_CLOSED_LOOP_RESULTS.md)
- [等提案预算基线](reports/S10_BASELINE_COMPARISON.md)
- [Go / No-Go](reports/S10_GO_NO_GO.md)
- [下一阶段](reports/NEXT_RESEARCH_STAGE.md)
