# S5 · 闭环实现

每次候选都走同一条执行路径。

```text
真人 RTMW 偶数帧 → 冻结 Uni-Sign → 参考表示（每段缓存一次）
当前六维系数 → 局部 SMPL-X 旋转 → FK（133 点） → 固定相机投影
            → 同一 Uni-Sign → 虚拟表示 → 与参考表示的余弦距离
候选评分 → 接受能降低目标且满足限幅的更新 → 从更新后姿态再提候选
```

## 参数、观测和预算

- 选择上色作为校准来源，踢拳和诊断为留出来源。两者在 Stage 4 已出现，本轮仅对归一化尺度和优化配置留出。沿用原配对、原相机、SMPL-X 体型和表情，保持 NLL 模型权重冻结。相同方法提取真人、虚拟关键点的后处理和专家表示。真人输入是存档 RTMW 坐标与分数，虚拟分数固定为 1。
- 只更新六个标量。指关节、腕朝向、臂轨迹各有两个平滑且起止为零的时间窗系数。各组的局部旋转轴在所有序列固定。共享初始化在这组轴和窗口上加入已知旋转，并叠加小幅非同轴扰动。这使搜索空间可部分修复受控损坏，又不包含精确原拟合作为候选。原始重建仅作为 A 对照和对拟合参考的审计。
- 在 SMPL-X 实际旋转上用 SO(3) 局部复合，手部先加入、导出时减去模型的 hand mean。硬约束为系数 ±12°，每关节相对起点的旋转增量 ≤20°，附加旋转速度 ≤2°/帧，附加加速度 ≤0.5°/帧²。正则限制系数大小及附加加速度。该约束定义在此次改动的局部旋转上，未验证绝对解剖限位、手指接触或碰撞。
- 优化仅看偶数帧的 RTMW 与同样抽样的专家输入（12 Hz）。运动参数与 FK 仍为 24 fps。奇数帧的 RTMW/MediaPipe、3D 轨迹对初始拟合的偏差和骨段/加速度只用于评估。MediaPipe 观测独立于优化损失，但 NSA 原拟合曾使用 MediaPipe，有检测器共享偏差。
- NumPy 预处理及 Stage 4 特征 hook 不构成端到端可微图。本轮用受控坐标搜索，每个系数依次试正负 6°、3°、1.5°，接受严格改进。每臂 18 个更新步、36 个提案，跨臂复用相同系数点的数值计算。记录所有提案、拒绝状态、每步系数和指标。复现无需梯度或训练。

## 比较条件

| 代号 | 优化目标 |
|---|---|
| A | 原始拟合，仅测量 |
| I | 注入扰动的公共起点，仅测量 |
| B | 偶数帧几何加共同运动约束 |
| C 时序（主方法） | B 加左右手时序支路虚实距离 |
| C 空间、C 全局 | B 分别加身体腕点特征融合前的手部空间支路或全局虚实距离 |
| C NLL | B 加真人参考句的虚拟 NLL，诊断已知弱信号 |
| D 时序 | 手部时序虚实距离加相同运动约束和正则 |

几何为可见腕点、全图指尖、腕相对指尖的中位像素误差的平均。几何和反馈各用校准段公共起点的 12 个 ±6° 单轴探针的绝对改变量中位数作尺度。C 的反馈权重预设为 0.5，D 为 1。全部实验臂、权重、尺度口径和步长在评估段运行前固定。

## 执行

在 Stage 4 项目根目录运行，沿用 [Stage 4 复现说明](../../STAGE4_REPRODUCE.md) 的环境、模型、动作和已缓存几何。脚本校验 SMPL-X 资产哈希与 G1 一致。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m scripts.stage5_closed_loop \
  --repo work/Uni-Sign --checkpoint work/how2sign_pose_only_slt.pth --mt5 work/mt5 \
  --asset work/models/smplx/SMPLX_NEUTRAL.npz --motions work/motions --output experiments/Stage5
python -m scripts.stage5_summary experiments/Stage5
python -m scripts.stage5_audit --root experiments/Stage5 --motions work/motions --asset work/models/smplx/SMPLX_NEUTRAL.npz
python -m scripts.stage5_holdout_expert --root experiments/Stage5 --repo work/Uni-Sign \
  --checkpoint work/how2sign_pose_only_slt.pth --mt5 work/mt5
```

运行使用 Linux CPU 4 个 PyTorch 线程。可用 `--device cuda` 切换受支持的环境。视频需要 ffmpeg/libx264。每个视频 A 列表示文件名 `_vs_` 前的方法，B 列表示后者。上下两排分别为正面与侧面；可视化为 3D 骨架投影视图。
