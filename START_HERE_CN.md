# 迁移到 Codex

## 1. 打开工程

解压 `SignMotion_Codex_Full.zip`。在 Codex 应用里打开解压得到的 `sign_motion_research/`。使用 CLI 时，在该目录启动 `codex`。IDE 或 WSL 也请打开这一级目录，根目录应当包含 `AGENTS.md`。

将 [CODEX_START_PROMPT.md](CODEX_START_PROMPT.md) 里的提示词粘贴给 Codex。提示词、[HANDOFF_CONTEXT.md](HANDOFF_CONTEXT.md) 和源记录提供恢复工作状态所用的上下文。历史对话无需导入。

## 2. 先核验文件，再做最小复验

```bash
python scripts/check_handoff.py
```

已安装 Python 3.12、CPU PyTorch、NumPy 和 SciPy 时，可运行下面的 GRU 权重检查。它用合成历史执行 6 个预测步并验证旋转矩阵。此检查无需渲染、NSA 原始训练集或 Uni-Sign。

```bash
python scripts/check_handoff.py --smoke
```

若需新建 Linux CPU 环境，在项目根执行以下命令。老环境的全依赖及固定下载版本见原 [README.md](README.md)。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
python -m pip install 'numpy==2.3.5' 'scipy==1.17.0'
python scripts/check_handoff.py --smoke
```

## 3. 恢复完整闭环

| 需要的外部资源 | 默认路径 | 获取与核验 |
| --- | --- | --- |
| SMPL-X neutral | `work/models/smplx/SMPLX_NEUTRAL.npz` | 已上传的模型文件（请从原会话附件获取），或通过 SMPL-X 官方许可取得，按 Stage 6 文档核对哈希 |
| Uni-Sign 源码与权重 | `work/Uni-Sign/`、`work/how2sign_pose_only_slt.pth` | [README 安装命令](README.md)，固定代码提交和权重哈希 |
| mT5 tokenizer 与配置 | `work/mt5/` | [README](README.md)，仅在句子生成或 NLL 路径需要 |
| RTMW、人体检测器、渲染依赖 | `work/rtm/pose/end2end.onnx`、`work/rtm/detector/**/end2end.onnx` | [Stage 3](STAGE3_REPRODUCE.md)、[Stage 6](STAGE6_REPRODUCE.md) 和 [Stage 7](STAGE7_REPRODUCE.md)，RGB 路径需要 FFmpeg 与 Linux EGL |
| NSA 完整训练样本 | `work/shared/` 或自定义路径 | [共享实验复现](experiments/SharedMotion/README_SHARED_MOTION.md)。重训或独立扩展抽样时才需要，复验单源用已归档缓存。 |

`python scripts/check_handoff.py --assets` 检查默认路径。该命令发现缺失项时退出码为 2，含按路径才使用的可选资产。其他本机目录可用 `--work-dir` 指定。版本和哈希请与既有文档核对。Windows 可以把 Linux 全链路放在 WSL2，匹配其图形驱动和 EGL 环境。先从 CPU 最小闭环开始。

资产和全依赖到位后，试跑已选开发源的一次冻结专家闭环。已有数值起点与目标缓存随包提供，输出写入新的目录。

```bash
export OPENBLAS_NUM_THREADS=1
python -m scripts.shared_motion_feedback \
  --repo work/Uni-Sign \
  --checkpoint work/how2sign_pose_only_slt.pth \
  --asset work/models/smplx/SMPLX_NEUTRAL.npz \
  --shared-checkpoint experiments/SharedMotion/model/gru_context16_h3.pt \
  --output work/replay_shared_one \
  --sources 3sZjSNcqXpI_2-1-rgb_front
```

对照归档的 [逐源闭环](experiments/SharedMotion/feedback/integration.csv) 与该次新输出的 `integration.csv`、`trace.csv` 和 `proposals.csv`。然后按原复现文档添加 RGB、其他源或训练。共享训练阶段的作者档案索引和范围读取绑定既有快照，需重新核对哈希和数据许可。大规模下载前先确定所需样本范围。

## 4. 直接查看成果

- [共享模型中文摘要](experiments/SharedMotion/reports/EXECUTIVE_SUMMARY_CN.md) 与 [闭环结果](experiments/SharedMotion/reports/FEEDBACK_INTEGRATION.md)。
- [自由滚动演示](experiments/SharedMotion/visuals/Video2_free_running.mp4) 与 [共享先验接入闭环](experiments/SharedMotion/visuals/Video4_feedback_integration.mp4)。
- [Stage 10 守门与固定相机](experiments/Stage10/README_STAGE10.md) 与 [全动作对照](experiments/Stage10/visuals/D_full_signing_comparison.mp4)。
- [合并来源](SOURCE_ARCHIVES.tsv) 与 [历史差异](MERGE_CONFLICTS.tsv)。20 个恢复到的归档已合并。完全相同的文件只存一份，旧版差异保留在 `archives/superseded/`。
- `parallel/ASL_SMPLX_UniSign_Pilot/` 是独立的文字到动作小试验。旧盲评包在 `archives/stage6_blind_review/`。

包内的研究数据、衍生视频与动作应沿用原来源及许可，仅在授权范围内迁移。自行准备虚拟环境、第三方模型和完整 NSA 原始数据。

## GitHub / Codex 在线工作副本

当前 Git 工作副本的最小安装与可选数据恢复见 [CODEX_CLOUD_SETUP.md](CODEX_CLOUD_SETUP.md)。默认自检验证已提交的代码与数值入口；恢复原迁移包中的数据资产后再加 `--data`。根 README 已更新，原 README 安装段落在 [历史 README](archives/superseded/github_migration/README.md)。第三方 Uni-Sign 和 SMPL-X 仍采用原来源及路径。

GRU 冒烟测试前先按 CODEX_CLOUD_SETUP.md 恢复模型权重。公开副本不包含二进制权重。
