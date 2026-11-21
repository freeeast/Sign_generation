# Stage 9 可复现分指控制器

## 控制和反馈

- 每手 15 个方向。食指、中指、无名指、小指各有掌指屈伸、近端屈伸与远端 0.65 耦合、掌指侧展。拇指使用屈伸、侧展与对掌样式三个方向。轴由 SMPL-X 静息关节方向以及手均姿确定。左右手独立。
- 双侧肩、肘和腕各有三个局部旋转通道，共 18 个。48 个方向在三个来源上使用 5/16/16 个三次样条结点，即 240/768/768 个标量系数。先对系数的时间展开做有界映射，再在 SO(3) 上与初始姿态复合。导出时减回 SMPL-X 内部的手均姿。体型和骨骼比例逐片固定。
- 继承 Stage 7 修复后的左右手图。空间分支以腕相对位置输入手图，时序分支融合对应人体腕特征，多尺度版本按取样后的 4 与 12 个响应时间步池化。主损失的三项权重 0.35、0.40、0.25。冻结所有模型权重。Stage 9 在训练帧先缓存真人响应，虚拟试探用可微稀疏 FK 裁取官方十个网格指尖及活动关节，真实检查点由官方全网格 FK 核验。
- 每次 Adam 状态提案经过实际 3D 参数、FK、投影及同一冻结专家复评分。纯几何选择只使用几何和可见支持约束，专家值记录作诊断。层级顺序为臂腕 5、冻结臂腕的手形 14、加重时序正则 6、通过臂腕手形锚点后的专家 10；第一项预处理也计入 36 提案。联合组采用 35 次端到端迭代和同样的初步稳定化，联合几何对照用完全相同的更新顺序和学习率。被拒的提案恢复运动系数并降低步长，Adam 动量继承。
- 每次接受前检查全 24 Hz 的活动关节局部旋转和相邻帧增量。可见支持下降超过 5 个百分点会拒绝候选。允许保留起点已有的局部旋转或增量峰值。诊断修正 `art_guarded` 根据初态到上限的余量，在五帧邻域内抑制过临界关节的增量，保留其它关节自由度并按原边界审查。截止阈值 0.25°、线性过渡区 10°，此参数用同三片做诊断，外部泛化待测。更细的解剖限位、接触和碰撞未加测。所有手形来自真实视频的 2D 观测，未使用 3D 标签。
- 连续状态记录包括系数轨迹、提案/接受/拒绝和响应分项 CSV、全帧 3D 检查点、真实特征缓存。新增诊断运行也保存被拒试探系数和余量遮罩。基准首轮的拒绝项保存候选分数与拒绝原因，未保存其完整候选系数。

## 复现

在现有 Stage 8 工程根目录运行。保留 `experiments/Stage8/` 原始输出，使用 Stage 7 数据、校准和相同官方冷冻权重。提交号和资产哈希见 [清单](../summary/assets_sha256.json)。Uni-Sign 提交 `eed438bcb49e30405cd6ccdfcccca330c134e830`，权重 `how2sign_pose_only_slt.pth`；Python/Torch 和 RTMW、ONNX、SMPL-X、EGL 和 FFmpeg 安装沿用 `STAGE7_REPRODUCE.md`。RTMW 端点视频用最多 14 帧复评，完整 212 帧只用于抖动分层对照；侧视同步依赖 Stage 7 对应真人侧视视频。

```bash
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
out=experiments/Stage9_replay
asset=work/models/smplx/SMPLX_NEUTRAL.npz
repo=work/Uni-Sign
checkpoint=work/how2sign_pose_only_slt.pth
det=work/rtm/detector/end2end.onnx
pose=work/rtm/pose/end2end.onnx
focus=3sZjSNcqXpI_2-1-rgb_front
side=experiments/Stage7/multiview/videos/$focus.mp4
common=(--repo "$repo" --checkpoint "$checkpoint" --asset "$asset" --output "$out")
python -m scripts.stage9_audit --asset "$asset" --output "$out/audit"
python -m scripts.stage9_run "${common[@]}" --levels B A C
python -m scripts.stage9_run "${common[@]}" --sources "$focus" --methods art_raw
python -m scripts.stage9_run "${common[@]}" --sources "$focus" --methods art_hier --feedback spatial temporal multiscale
python -m scripts.stage9_run "${common[@]}" --levels A C B --methods art_guarded
python -m scripts.stage9_run "${common[@]}" --levels B --methods art_joint_geo
python -m scripts.stage9_run "${common[@]}" --levels A C --methods art_guarded_geo
python -m scripts.stage9_rgb "${common[@]}" --detector "$det" --pose-model "$pose"
python -m scripts.stage9_rgb "${common[@]}" --detector "$det" --pose-model "$pose" --jitter
python -m scripts.stage9_verify "${common[@]}" --detector "$det" --pose-model "$pose"
python -m scripts.stage9_summary --asset "$asset" --output "$out"
python -m scripts.stage9_visuals --asset "$asset" --output "$out" --side-video "$side"
mkdir -p "$out/compute"
python -m scripts.stage9_compute --repo "$repo" --checkpoint "$checkpoint" --asset "$asset" --output "$out/compute"
```

焦点的最后接受轨迹重新导出的局部参数差为 **0**；可微所选关节与官方导出网格差 <0.001 mm。全输入（包含优化期间固定的脸部点）的最大投影差为 0.608 px，专家差小于 0.000025。选定高梯度的拇指屈伸自由度，自动梯度 −0.002905，有限差分约 −0.00299 至 −0.00305，方向相同。两次 CPU 搜索在角度保护边界附近的末值有 <0.0004 响应差，以主运行记录表和实际视频为准。重检测、EGL 和 CPU 线程会影响时延。

额外来源 DKOz 来自 Stage 7 数据包。先恢复 `data/inputs`、`data/paired`、`data/real_pose` 下的该来源，随后在新目录运行 Stage 8 的 B 级 A、C、C3，再按 `configs/stage9_new_source.json` 运行 Stage 9。复跑时改该配置的 `upstream` 到本次 Stage 8 对照输出；计分前固定 Stage 9 配置。

```bash
sid=DKOzSnHwSFI_7-5-rgb_front
ext=experiments/Stage9/new_source
python -m scripts.stage8_imitate --repo "$repo" --checkpoint "$checkpoint" --asset "$asset" \
  --output "$ext/stage8_controls" --sources "$sid" --levels B --methods A_geometry C_sequential C3_local_wide --threads 2
python -m scripts.stage9_run --repo "$repo" --checkpoint "$checkpoint" --asset "$asset" --output "$ext" \
  --config configs/stage9_new_source.json --levels B --methods art_geo art_hier art_joint art_joint_geo
python -m scripts.stage9_rgb --repo "$repo" --checkpoint "$checkpoint" --asset "$asset" --output "$ext" \
  --config configs/stage9_new_source.json --detector "$det" --pose-model "$pose"
python -m scripts.stage9_visuals --asset "$asset" --output "$ext" --config configs/stage9_new_source.json --only front
```
