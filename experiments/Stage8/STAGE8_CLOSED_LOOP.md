# Stage 8 · 闭环实现与复现

## 闭环和方法

1. 缓存同帧真人 RTMW，使用修正后的 Uni-Sign 左右手图激活，构造固定可见支持的真实目标特征和置信度。
2. 用共享 SMPL-X 身材和固定摄像机，从指定初态产生真实 3D 运动。每次提案在 SO(3) 上局部复合、变更 SMPL-X 参数，经过 FK 和投影重新评估。图反馈由空间、时序和 4/12 帧多尺度响应距离加权成一个目标，权重 0.35、0.40、0.25。
3. 主要控制是手指、腕与臂各两个原时间模式。三轮半径 6°、3°、1.5°，每轮六轴，每轴两个方向。24 Hz 旋转和动力学约束遍历整段，提案评分帧以步长 4 取样、最多 44 帧。独立正面几何用全部可用奇数帧的 MediaPipe；每次提案只记录最多 48 个奇数帧的诊断值，检查点重算全部奇数帧。
4. A 只按鲁棒几何选择；B 只按专家响应选择；C 用前 6 次决策建立几何锚点，余下 12 次只接受专家距离下降且通过固定锚点几何守门的更新；D 用相同几何和专家量纲，最小化归一化几何加 0.5 倍归一化专家差。归一化尺度继承 Stage 7 的旧源校准。所有臂按相同次数发提案，可以在同一初态内缓存相同系数的 FK 和专家结果。
5. 运动筛查为系数 ±12°、局部更改 ≤20°、附加速度 ≤2°/帧及加速度 ≤0.5°/帧²。附加宽松的肩、肘、腕、指绝对转角包络，初态原有超限不被扩大。固定真实手部支持，候选支持减少超过 5 个百分点则拒绝。几何守门作用于 C 系列的专家阶段，容差在配置，均用优化观测而非留出观测。
6. C2 保持相同几何阶段，专家阶段换成每只手独立的三个局部时间窗，沿 SMPL-X 手部平均弯曲方向更新。C3 在 C2 上只将专家阶段尝试半径改成 6°/3°。两臂仍然各 36 个提案且持有相同运动守门。
7. RGB 从实际真人像素和渲染网格重跑同一 RTMW。按真人与虚拟共享采样索引打分。候选在相机固定的 RGB 管线复评，不参与直接主实验的优化决策。

## 产物和可恢复性

每臂保存 19 个真实 3D 参数状态，36 个提案的当前及后续系数、接受/拒绝、响应分项、按手及帧匹配的响应曲线、几何和运动量。每源保存真实空间/时序激活。原始虚拟激活用单行提案回放器精确再计算，现场验证结果在 `replay_example/verification.json`。关键帧 0/6/12/18 的高维旋转可从 `motion_trajectory.npz` 和共享 `initial_fit.npz` 无损恢复。完整视图未入包的中间点可运行同一 FK 重建。

实现核心 `scripts/stage8_imitate.py`，粗拟合 `src/stage8_coarse.py`，手控制 `src/stage8_motion.py`，配置 `configs/stage8.json`。从图训练权重重载只读取原模型匹配的参数，左右覆盖修复来自 Stage 7，未训练新识别器。

## 运行命令

将报告/代码包与运动分卷解压到同一项目目录。复用 Stage 7 的原始配对数据和校准尺度。需已有 Python/Torch、EGL、FFmpeg、SMPL-X 模板和 Uni-Sign 权重。版本与资产准备见 `STAGE7_REPRODUCE.md` 及 `STAGE6_REPRODUCE.md`。权重没有打包。以下命令使用新目录保留交付数据。优化支持按所选来源及方法续跑，阶段级别的配置改变应另用新目录。

```bash
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
out=experiments/Stage8_replay
asset=work/models/smplx/SMPLX_NEUTRAL.npz
repo=work/Uni-Sign
checkpoint=work/how2sign_pose_only_slt.pth
det=$(find work/rtm/detector -name end2end.onnx)
common=(--repo "$repo" --checkpoint "$checkpoint" --asset "$asset" --output "$out")
python -m scripts.stage8_imitate "${common[@]}" --levels B A C --threads 3
python -m scripts.stage8_imitate "${common[@]}" --levels B --methods C2_local_hands C3_local_wide
python -m scripts.stage8_rgb "${common[@]}" --detector "$det" --pose-model work/rtm/pose/end2end.onnx
python -m scripts.stage8_evaluate --output "$out"
python -m scripts.stage8_evaluate --output "$out" --secondary --repo "$repo" --checkpoint "$checkpoint" --mt5 work/mt5
python -m scripts.stage8_visualize --root "$out" --asset "$asset"
python -m scripts.stage8_visualize --root "$out" --asset "$asset" --sources 3sZjSNcqXpI_2-1-rgb_front --local
python -m scripts.stage8_animate_all --root "$out" --asset "$asset"
python -m scripts.stage8_replay --asset "$asset" --repo "$repo" --checkpoint "$checkpoint" \
  --root "$out" --output "$out/replay_example" --source 3sZjSNcqXpI_2-1-rgb_front \
  --level B --method C3_local_wide --proposal 25
```

三个源各自固定初始化，坐标方向种子沿用 5061，粗拟合零起点、固定优化步数，专家采用评估模式。CPU 线程变化和 EGL 驱动差异可能影响临界候选或渲染重排。数据沿用 How2Sign 和 Neural Sign Actors 的科研用数据，语言评估与接触验证不由单次研究目标推断。
