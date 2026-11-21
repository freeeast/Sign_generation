# Text → SMPL-X → Uni-Sign｜24 句 ASL 初步实验

本数据包包含 LLM 编写的动作计划和运动学规则、24 个 SMPL-X 格式 NPZ、近似骨架的 3D/2D 轨迹、真人 / 生成 / 静止三组 Uni-Sign 实际推理结果及汇总图。LLM 根据英文文本编写关键点动作计划，确定性程序生成局部轴角。短语动作和指拼均为未经签者校订的近似。正式 SMPL-X 网格和人工可懂度评审尚待验证。

## 快速查看

```bash
# 安装 AITViewer，并在支持其桌面显示环境的 Python 环境运行
pip install aitviewer
python view_aitviewer.py poses/S20.npz --proxy

# 自有模型目录需含 models/smplx/SMPLX_NEUTRAL.npz
python view_aitviewer.py poses/S20.npz --model-dir /absolute/path/to/models
```

S20 是 44 帧的上下方向动作样例。25 fps。近似骨架可以直接使用包内的 `proxy/S20.npz`。正式模型使用 SMPL-X neutral、整维手部旋转、`use_pca=False`、`flat_hand_mean=True`，以 Y 为上轴。当前验证了文件及近似骨架，正式网格和 AITViewer GUI 尚待在带资产和桌面环境的机器上检查。[SMPL-X 模型](https://smpl-x.is.tue.mpg.de/) · [AITViewer SMPLSequence](https://eth-ait.github.io/aitviewer/renderables/smpl_sequences.html)

## 数据栏位

每个 `poses/Sxx.npz` 都可用 `np.load(..., allow_pickle=False)`。时间维为 T。局部旋转以弧度轴角表示。

| 键 | 数组形状 | 内容 |
|---|---|---|
| `global_orient` | T × 3 | 根朝向 |
| `body_pose` | T × 63 | 身体 21 关节 |
| `left_hand_pose` / `right_hand_pose` | 各 T × 45 | 每只手 15 关节 |
| `jaw_pose` / `expression` | T × 3 / T × 10 | 下颌与表情，本试验固定为零 |
| `transl` / `betas` | T × 3 / 10 | 根位移、中性体型，本试验固定为零 |
| `fps` | 标量 | 25 |

与上传样例的对应关系为 `smplx_root_pose → global_orient`、`smplx_body_pose → body_pose`、`smplx_lhand_pose → left_hand_pose`、`smplx_rhand_pose → right_hand_pose`、`smplx_jaw_pose → jaw_pose`、`smplx_shape → betas`、`smplx_expr → expression`。样例的 `cam_trans` 属相机拟合信息。包内生成动作具有自己的根位移和正交投影，不使用该相机平移。测试集里没有上传样例 ID。其 ID 匹配公开的 How2Sign 重对齐验证集元数据。

`proxy/Sxx.npz` 的 `keypoints`、`scores` 和 `landmarks3d` 为 T × 133 × 2、T × 133、T × 133 × 3，遵循 COCO-WholeBody 格式。实际映射 85 点，包含官方评测使用的 69 点。其余点的置信度为零。近似 FK 使用按身体比例设定的骨长、解析双段 IK 与手指局部旋转。代理脸只提供局部几何锚点，jaw / expression 为零。静止对照复制同一生成片段的首帧并保持帧数、投影和置信度不变。

## 复跑

```bash
# 建议使用 Python 3.12 与匹配的 torch / torchvision
# CPU 在本实验用 torch 2.6.0+cpu 和 torchvision 0.21.0+cpu
DS_BUILD_OPS=0 pip install -r requirements-unisign.txt "scipy>=1.12" "sacrebleu==2.6.0"
mkdir -p work/mt5
git clone https://github.com/ZechengLi19/Uni-Sign.git work/Uni-Sign
git -C work/Uni-Sign checkout eed438bcb49e30405cd6ccdfcccca330c134e830
release=eab251b7fe7e8521afc0e67be98add670ea40a0d
curl -fL "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/$release/how2sign_pose_only_slt.pth" -o work/how2sign_pose_only_slt.pth
mt5=2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f
for name in config.json spiece.model tokenizer_config.json; do
    curl -fL "https://huggingface.co/google/mt5-base/resolve/$mt5/$name" -o "work/mt5/$name"
done
curl -fL 'https://huggingface.co/datasets/aipieces/How2Sign/resolve/a4fb69196612b063e602ab95b8d2099e5eda74fe/how2sign_realigned_test.csv' -o work/how2sign_test.csv
python materialize_sample.py work/how2sign_test.csv work/sample.json
# 需 Bash。-- 允许样本 ID 以短横线开头。
mapfile -t ids < <(python -c 'import json; print(*(r["clip_id"] for r in json.load(open("sample_ids.json"))["clips"]),sep="\n")')
python fetch_official_poses.py --split test --output work/real -- "${ids[@]}"
common=(--sample work/sample.json --repo work/Uni-Sign --checkpoint work/how2sign_pose_only_slt.pth --mt5 work/mt5)

python generate.py --stats results/pose_stats.csv
python project_proxy.py poses proxy --static-dir work/static
python evaluate.py "${common[@]}" --real-dir work/real --output results/real.csv
python evaluate.py "${common[@]}" --generated-dir proxy --label rig_proxy --output results/generated_proxy.csv
python evaluate.py "${common[@]}" --generated-dir work/static --label static_proxy --output results/static_proxy.csv
python score.py work/sample.json
```

如需保留初次运行的结果，请将上面三次推理和图表保存到实验包的副本里。英文目标在锁定清单中按 ID 从原始 TSV 取出。原始真人姿态 PKL 来自固定版本的 Uni-Sign 归档，并由 ZIP CRC 检查。请将本脚本的 PKL 读入仅用于该已知来源。

若正式模型资产已可用，先预览一个动作，检查左右、朝向和手腕的几何，再统一投影。真人的 RTMPose 结果与 SMPL-X 虚拟投影仍可能存在输入分布差异。

```bash
python project_smplx.py poses work/generated_smplx --model-dir /absolute/path/to/models
python evaluate.py "${common[@]}" --generated-dir work/generated_smplx \
  --label official_smplx_fk --output results/generated_smplx.csv
```

更有诊断价值的参照是让同一句真实签者的拟合 SMPL-X 也走这条正式 FK 路径。如果真人拟合转译有明显下降，结合相同相机的渲染→RTMPose，以及固定脸部/置信度的消融定位投影偏差。若真人拟合与真人 RTMPose 表现接近，再扩大生成实验。由熟练 ASL 签者对混排的无字幕视频独立写回语义，并标出手型、方向、接触、双手和非手部线索的错误。

## 来源

- How2Sign，Duarte et al.，CVPR 2021。以 [官方数据页](https://how2sign.github.io/) 和 [固定测试集元数据版本](https://huggingface.co/datasets/aipieces/How2Sign/blob/a4fb69196612b063e602ab95b8d2099e5eda74fe/how2sign_realigned_test.csv) 为采样来源，数据 CC BY-NC 4.0。
- Uni-Sign，Li et al.，ICLR 2025。[论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html) · [发布权重和输入姿态](https://huggingface.co/ZechengLi19/Uni-Sign/tree/eab251b7fe7e8521afc0e67be98add670ea40a0d)。
