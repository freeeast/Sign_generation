# 共享手语动作实验的复现与索引

将文件并入现有 sign_motion_research。依赖已有 Stage 7 到 Stage 10 的工具和数据缓存，包括 Stage 9 additional source 里的 DKOz coarse。完整安装与公开模型来源沿用 STAGE6_REPRODUCE.md、STAGE7_REPRODUCE.md 和 experiments/Stage10/README_STAGE10.md。Uni-Sign 和 SMPL-X 模板不重新分发。作者下载和原始数据按非商业研究渠道取得。训练样本清单、官方档案索引、权重、衍生动作、4 个视频与反馈记录都在包内。原始 775 段拟合未重新分发。

## 获取数据、训练与闭环
```bash
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
mkdir -p work/shared
official='https://www.dropbox.com/scl/fo/nc4qxrwqz1ze2b4dy5wce/h'
key='rlkey=zyx550j0lx88kj7wx6nfp3wjr&dl=1'
curl -fsSL "$official?$key" | head -c 1200000000 > work/shared/nsa_prefix.bin
curl -fsSL --range 5711719869-6426140848 "$official/train_poses.zip?$key" -o work/shared/train_central.bin
sha256sum -c experiments/SharedMotion/analysis/nsa_inputs.sha256
work=experiments/SharedMotion_replay
asset=work/models/smplx/SMPLX_NEUTRAL.npz
repo=work/Uni-Sign
expert=work/how2sign_pose_only_slt.pth
det=$(find work/rtm/detector -name end2end.onnx)
pose=work/rtm/pose/end2end.onnx
python -m scripts.shared_motion_data --prefix work/shared/nsa_prefix.bin --train-central work/shared/train_central.bin --output "$work/data"
python -m scripts.shared_motion_train --dataset "$work/data" --asset "$asset" --output "$work/model"
ck="$work/model/gru_context16_h3.pt"
python -m scripts.shared_motion_analysis --dataset "$work/data" --asset "$asset" --model "$work/model" --output "$work/analysis"
python -m scripts.shared_motion_feedback --repo "$repo" --checkpoint "$expert" --shared-checkpoint "$ck" --asset "$asset" --output "$work/feedback"
python -m scripts.shared_motion_rgb --repo "$repo" --checkpoint "$expert" --asset "$asset" --output "$work/feedback" --detector "$det" --pose-model "$pose"
python -m scripts.shared_motion_visuals --dataset "$work/data" --asset "$asset" --checkpoint "$ck" --feedback "$work/feedback" --output "$work/visuals"
```

内容长度与 SHA 使用本轮快照。限流或上游 ZIP 更新时，从官方目录重新验证区间与索引，再记录新数据版本。目录是官方 ZIP 中的条目全统计。实验限于不相交的录像样本。完整受限资源见官方 NSA、How2Sign 和仓库审计，不从同名数据假定权利。

## 入口

| 对象 | 归档路径 |
|---|---|
| 源代码 | scripts/shared_motion_data.py、scripts/shared_motion_train.py、src/shared_motion.py、scripts/shared_motion_feedback.py 等 |
| 模型 | model/gru_context16_h3.pt（选定）、model/selection.json、model/learning.csv |
| 数据与评测 | data/archive_inventory.csv、data/sample.csv、model/test_forecast.csv、analysis/paired_source_bootstrap.csv |
| 运行记录 | feedback/integration.csv、feedback/各源/各方法/trace.csv |
| Avatar 动作 | visuals/motion_exports/*.npz 和 feedback/各源/各方法/final_fit.npz |
| 四个视频 | visuals/Video1_teacher_forced.mp4 至 Video4_feedback_integration.mp4 |
| 文档 | reports/ 下八项指定交付及中文摘要 |
