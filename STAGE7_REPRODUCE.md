# Stage 7 复现

在项目根目录执行。需要同 Stage 6 的 Python/Torch、SMPL-X NEUTRAL、Uni-Sign 提交、mT5 本地 tokenizer/config、RTMW 及 MediaPipe 资产。模型地址和 SHA-256 见 [STAGE6_REPRODUCE.md](STAGE6_REPRODUCE.md)。无需安装 PPO 或启动训练。优化运行只使用冻结的官方 Uni-Sign 图分支，审核句子排序时才加载 mT5。实测运行使用 CPU、Mesa EGL、RTMW ONNX、FFmpeg libx264。

首次准备数据使用固定 NSA 6,500,000 字节验证前缀和 How2Sign 原始验证 TSV。正面小片段来自固定镜像 revision `8ebd757f151ab3cb74741d961abf93f0e36fff40`，侧面来自官方 `val_rgb_side_clips.zip`，仅下载精确 ZIP 成员。主包缓存了四段正面原始视频、拟合、RTMW 和 MediaPipe，以及两个 Stage 6 校准源的拟合和检测结果。模型权重与授权 SMPL-X 模板需按上述说明另备。

```bash
python -m pip install onnxruntime==1.22.1 moderngl==5.12.0 opencv-python==4.11.0.86 mediapipe==0.10.32
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
src=experiments/Stage7
out=work/replay_stage7
mkdir -p "$out"
cp -r "$src/data" "$out/data"
det=$(find work/rtm/detector -name end2end.onnx)
common=(--repo work/Uni-Sign --checkpoint work/how2sign_pose_only_slt.pth
        --asset work/models/smplx/SMPLX_NEUTRAL.npz --output "$out")
rtm=(--detector "$det" --pose-model work/rtm/pose/end2end.onnx)
python -m scripts.stage7_optimize "${common[@]}" --mode direct --threads 4
python -m scripts.stage7_optimize "${common[@]}" --mode controller --threads 3
python -m scripts.stage7_optimize "${common[@]}" "${rtm[@]}" --mode rgb --threads 3
python -m scripts.stage7_optimize "${common[@]}" --mode diagnose --threads 3
python -m scripts.stage7_optimize "${common[@]}" "${rtm[@]}" --mode rgb_diagnostic --threads 3
python -m scripts.stage7_evaluate --output "$out" --mode semantic --repo work/Uni-Sign \
  --checkpoint work/how2sign_pose_only_slt.pth --mt5 work/mt5
python -m scripts.stage7_evaluate --output "$out" --mode domains
python -m scripts.stage7_rerank "$out"
python -m scripts.stage7_summary "$out"
python -m scripts.stage7_visuals "$out" --asset work/models/smplx/SMPLX_NEUTRAL.npz
python -m scripts.stage7_domain_audit --repo work/Uni-Sign --checkpoint work/how2sign_pose_only_slt.pth --output "$out"
```

程序按来源、条件和方法保存端点参数、输出轨迹、逐步日志、候选提案及完成 JSON。重启同一命令继续未完成的方法。更改配置时使用新的输出目录，以免读到旧端点。CPU、Mesa 驱动和样本边缘使临界提案有数值差异。几何和专家尺度校准在两个旧源上完成。复现流程首先读或重建这些校准尺度及 RGB 方案选择。

```bash
git clone https://github.com/sign-language-processing/pose-evaluation work/pose-evaluation
git -C work/pose-evaluation checkout 2769b6f49693dd83e2ee81c09355d787afe775c6
python -m pip install pose-format==0.15.0 dtaidistance==2.4.0 fastdtw==0.3.4
python -m scripts.stage7_evaluate --output "$out" --mode toolkit --toolkit work/pose-evaluation

python -m scripts.stage7_multiview_fetch --manifest "$out/data/manifest.json" --output work/side_s7
mkdir -p "$out/multiview/pose"
for f in work/side_s7/*.mp4; do
  python -m src.extract_real_pose --repo work/Uni-Sign "${rtm[@]}" --threads 2 \
    --video "$f" --output "$out/multiview/pose/$(basename "$f" .mp4).npz"
done
python -m scripts.stage7_evaluate --output "$out" --mode side --side-videos work/side_s7 \
  --mp-model work/hand_landmarker.task
```

几何在奇数留出帧评价，具体索引见 RGB `frames.json`。运动旋转轴种子 5061、Torch 冻结载入种子 0、来源级 bootstrap 种子 7061。各新源公共受扰初态由清单顺序产生确定性变化，全方法匹配。人体比例始终沿用原拟合。单独的自然初态控制器实验关闭受控扰动。

## 数据出处

How2Sign 由 Amanda Duarte 等发布 [项目与出处](https://how2sign.github.io/) · [Dataverse DOI](https://doi.org/10.34810/DATA33) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)。原始短视频保留下载字节，pose 与可视化是本项目生成的衍生数据。拟合来自 Baltatzis 等 [Neural Sign Actors](https://baltatzisv.github.io/neural-sign-actors/)。源和派生文件仅供科研使用。
