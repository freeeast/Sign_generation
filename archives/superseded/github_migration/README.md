# Recognition-guided 3D sign motion

Stage 5 已执行三段实拟合的低维 3D 专家闭环。见 [Stage 5 决策](experiments/Stage5/S5_GO_NO_GO.md) 与 [复现说明](experiments/Stage5/S5_CLOSED_LOOP_IMPLEMENTATION.md)。

Stage 4 对局部反馈表示和跨检测器几何排序做了续跑，结论仍是 E4 NO-GO。见 [Stage 4 决策](experiments/Stage4/STAGE4_GO_NO_GO.md)、[复现](STAGE4_REPRODUCE.md)及配套 A/B 盲评材料。

G1 和 G2 已在 5 段真实拟合上完成，G3 的局部句子 NLL 反馈未通过。判定与两分钟组会讲稿见 [GO_NO_GO_DECISION.md](reports/GO_NO_GO_DECISION.md)，复现命令见 [STAGE3_REPRODUCE.md](STAGE3_REPRODUCE.md)。原 E0–E2 的对照和用法如下。

## 1. 环境和权重

从解压后的项目根目录运行。Linux CPU 验证环境为 Python 3.12、PyTorch 2.6.0 和 torchvision 0.21.0。CUDA 安装时选择匹配本机驱动的 PyTorch 构建，推理命令可加 `--device cuda`。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu' 'torchvision==0.21.0+cpu'
DS_BUILD_OPS=0 pip install -r requirements.txt

mkdir -p work/mt5
git clone https://github.com/ZechengLi19/Uni-Sign.git work/Uni-Sign
git -C work/Uni-Sign checkout eed438bcb49e30405cd6ccdfcccca330c134e830

UNISIGN_RELEASE=eab251b7fe7e8521afc0e67be98add670ea40a0d
MT5_RELEASE=2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f
curl -fL "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/${UNISIGN_RELEASE}/how2sign_pose_only_slt.pth" -o work/how2sign_pose_only_slt.pth
for name in config.json spiece.model tokenizer_config.json; do
  curl -fL "https://huggingface.co/google/mt5-base/resolve/${MT5_RELEASE}/$name" -o "work/mt5/$name"
done
printf '%s\n' '1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d  work/how2sign_pose_only_slt.pth' | sha256sum -c -
```

官方 How2Sign 权重含完整模型状态。适配器以 mT5 配置构建架构，再使用 `strict=True` 加载全部训练权重。原始归档姿态使用 Python pickle，只对已验证来源的样本使用 `--trusted-pickle`。自行生成的输入请用 NPZ。

## 2. 重跑公开姿态实验

```bash
mapfile -t clips < <(python -c 'import json; print(*(r["clip_id"] for r in json.load(open("experiments/E0/public_clips.json"))), sep="\n")')
python scripts/fetch_official_poses.py --output work/poses "${clips[@]}"

common=(--repo work/Uni-Sign --checkpoint work/how2sign_pose_only_slt.pth
        --mt5 work/mt5 --manifest experiments/E0/public_clips.json
        --pose-dir work/poses --pose-suffix .pkl --trusted-pickle --threads 4)
python -m src.evaluate_feedback "${common[@]}" --output work/E0
python -m src.evaluate_feedback "${common[@]}" --perturb --output work/E2
python scripts/plot_feedback.py work/E2/per_clip.csv work/E2/feedback.png

python -m src.smoke_geometry work/E1_geometry
```

范围下载从约 8.47 GB 的分卷归档取目录和指定片段，已在当前环境验证。服务器若不支持 HTTP Range，可沿用上游说明合并分卷，再将所需样本所在目录传给 `--pose-dir`。[Uni-Sign 数据说明](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/docs/DATASET.md)

## 3. 用自己的 3D 序列跑 E1

输入为 canonical axis-angle NPZ。保留原始动作文件，并先用单个文件核对帧数、FPS、模型体型、根关节方向、手部是否使用 PCA 和模型 mean。若数据以旋转矩阵、PCA 或合并向量储存，应按生产者的字段和旋转约定导出成下面的形状。请以现有 aitviewer 设置核对 `--hand-mean`。

| 键 | 形状 | 说明 |
|---|---|---|
| `global_orient` | `(T,3)` | 轴角，弧度 |
| `body_pose` | `(T,63)` | 21 × 3 |
| `left_hand_pose`, `right_hand_pose` | 各 `(T,45)` | 15 × 3 |
| `transl`, `jaw_pose`, `leye_pose`, `reye_pose` | 各 `(T,3)` | 可选，省略时用零 |
| `betas`, `expression` | `(K,)`, `(1,K)` 或 `(T,K)` | 可选，维数须与模型兼容 |

通过官方分发获取 SMPL-X 模型资产。下例的模型应位于 `work/models/smplx/SMPLX_NEUTRAL.npz`。相机轴和置信度为示例参数，按实际拟合约定调整。图中 L/R 表示签者的解剖学左/右。[SMPL-X](https://smpl-x.is.tue.mpg.de/)

```bash
python -m src.smplx_to_pose work/clip.npz work/clip.landmarks.npz \
  --model-dir work/models --gender neutral --hand-mean flat
python -m src.project_keypoints work/clip.landmarks.npz work/virtual/clip.npz \
  --right 1 0 0 --down 0 -1 0 --aspect 1 --scale 1 \
  --confidence .9 --plot work/clip.png

# 替换为真实片段 ID、实际文件名及对应 CSV 原句
cat > work/clips.json <<'JSON'
[{"clip_id":"clip","pose":"clip.npz","reference":"Replace with the corresponding How2Sign sentence."}]
JSON
python -m src.evaluate_feedback --repo work/Uni-Sign \
  --checkpoint work/how2sign_pose_only_slt.pth --mt5 work/mt5 \
  --manifest work/clips.json --pose-dir work/virtual --output work/E1
```

先核对腕点、指尖、面部轮廓和遮挡，再做置信度及画幅消融。可通过 `--mask-face` 加入面部缺失控制，或用 `--score-track` 注入已经按帧对齐的 `(T,133)` 分数。E2 可在同批虚拟投影上用同一个命令加 `--perturb` 重跑。该脚本面向小规模 pilot，片段多时先缩小清单以控制错配评分开销。

## 4. 文件

| 文件 | 内容 |
|---|---|
| `reports/ENVIRONMENT.md`、`reports/MODEL_INPUT_SPEC.md` | 运行环境、来源、输入契约 |
| `reports/E1_RESULTS.md`、`reports/E2_RESULTS.md` | 几何验证、真人扰动实验和限制 |
| `reports/RESEARCH_DECISION.md` | 组会汇报、文献对照和下一步 |
| `experiments/E0/` | 样本清单、逐片段输出、编码特征、复现一致性 |
| `experiments/E1/` | 合成几何测试的验证记录和图 |
| `experiments/E2/` | 逐片段及汇总 CSV、编码特征和扰动图 |
