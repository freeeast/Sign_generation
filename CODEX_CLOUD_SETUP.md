# Codex 工程恢复

## 最小可运行范围

克隆仓库后即可读取代码、报告、实验配置和数值对照。三个已训练 GRU 权重从原迁移包恢复。最小 CPU 推理需 Python 3.12、PyTorch、NumPy 和 SciPy。

在有安装网络的环境执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
python -m pip install -r requirements-smoke.txt
python scripts/check_handoff.py
# 下方恢复模型权重后，再运行 python scripts/check_handoff.py --smoke
```

本次迁移验证使用 Python 3.12.14、Torch 2.6.0+cpu、NumPy 2.3.5 和 SciPy 1.17.0。权重恢复后的最小测试使用合成历史运行 6 步 GRU，检查输出形状、有限值和旋转矩阵有效性。完整闭环需额外模型和真实配对数据。

## 数据恢复

完整迁移包 SignMotion_Codex_Full.zip（请从原会话附件获取） 约 486 MB。下载后，将文件放到执行环境能读取的目录，例如项目根目录的 `work/`。该文件链接由用户现有账号访问，Codex 执行环境需要实际取得文件字节。

```bash
mkdir -p work
# 把原迁移 ZIP 放到 work/SignMotion_Codex_Full.zip
python scripts/restore_migration_assets.py --archive work/SignMotion_Codex_Full.zip
python scripts/check_handoff.py --data
python scripts/check_handoff.py --smoke
```

恢复器验证整个 ZIP 的大小和 SHA-256，再逐文件校验。遇到不同的本地文件会停止。它只恢复清单中的外部数据资产，代码继续采用 Git 中的版本。可用多个 `--prefix` 缩小恢复范围，例如仅查看最新演示：

```bash
python scripts/restore_migration_assets.py \
  --archive work/SignMotion_Codex_Full.zip \
  --prefix experiments/SharedMotion/visuals/
```

默认 `check_handoff.py` 验证仓库中的 13 项关键入口。加 `--data` 才验证完整复验所需的 55 个缓存、轨迹与视频入口。`--assets` 检查第三方模型默认路径，缺失项的退出码是 2。

## 专家闭环与 RGB

第三方资产路径、固定版本和 SHA 见 [START_HERE_CN.md](START_HERE_CN.md)、[原安装说明](archives/superseded/github_migration/README.md)、[Stage 6 复现](STAGE6_REPRODUCE.md)和[Stage 10 复现](experiments/Stage10/README_STAGE10.md)。完整链路需 SMPL-X neutral、固定版本的 Uni-Sign checkpoint，RGB 路径另需 RTMW/检测器、FFmpeg 与 EGL 渲染依赖。

小样本复验可使用迁移包里的配对缓存。共享模型重训需要按已有清单重新取得 NSA 原始训练/验证/测试数据，保持录像级划分。复现和新增实验统一写入 `work/` 或新的实验目录。

## 仅恢复已有 GRU 权重

在用户提供完整迁移 ZIP 后，可先只恢复三个模型，以执行 CPU 推理。其余数据可以稍后恢复。

```bash
python scripts/restore_migration_assets.py \
  --archive work/SignMotion_Codex_Full.zip \
  --prefix experiments/SharedMotion/model/
python scripts/check_handoff.py --smoke
```

二进制权重与研究数据默认由 .gitignore 排除，公开仓库保存源代码和数值记录。
