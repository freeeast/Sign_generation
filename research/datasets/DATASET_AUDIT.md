# 数据集审计

| 资源 | 来源、观测与许可 | 本轮状态 |
| --- | --- | --- |
| How2Sign / Neural Sign Actors | [How2Sign](https://how2sign.github.io/) 公开 RGB、深度、多视角与分段，网站列原始分发 CC BY-NC 4.0。室内绿屏多视角，库方报告 11 签者。NSA 作者发布 How2Sign 的 SMPL-X 拟合。本项目采用正面伪标签、名义 24 Hz，历史 RGB 缓存 720p。原始姿态缺逐点可靠性及逐帧时间戳。 | 主数据。官方目录有 30,965/1,724/2,318 个连续段、2,192/115/149 个录像源。源录像级隔离，签者独立性未证实。 |
| SignSparK | [作者 LMDB](https://huggingface.co/datasets/LionelLow/SignSparK_data) 为重新优化后的 6D 姿态、文本与分段。代码 Apache-2.0，数据权利按来源。 | 已验证元数据，未下载 7.90 GB How2Sign train LMDB。 |
| SignAvatars | [作者项目](https://signavatars.github.io/) 报告 153 签者、70,000 视频和 8.34M 帧。多数据源自动重建，需分别核对帧率、视角和许可。 | 未用，后续核对签者、源重复和资产授权。 |
| 多视角 3D | How2Sign RGB 可配同源前侧视。MV-Sign 提出跨视角重建。 | 本轮无校准多视角 3D 测试，旧侧视证据保留。 |
| 真值 mocap | [DexAvatar / SGNify](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html) 提供拟合准确度评价。 | 未取得，未报告真实 3D 精度。 |

数据清单 SHA-256 为 `e2a78804f64aa7959d1c9e221d04614010d202454de4091b144c598782b43c60`。两 NSA 快照 SHA 已核对。读取历史抽取的 600/75/100 段，抽取日志无解码拒绝。长度至少 96 帧时剩 421/59/74，短片排除 179/16/26。训练和验证分别有 1,789 / 242 个窗。全段任意关节曾出现 >90° 帧间转角的比例，原 train/val/test 为 464/600、59/75、82/100，长度筛选后为 358/421、51/59、65/74。涵盖所有关节，不代表手部单独率。长片有更多机会遇到极端帧。未按生成模型表现剔除数据。

本轮执行既有 3D 快照到训练。RGB→检测→拟合的复现沿用 [历史索引](../../docs/PROJECT_HISTORY.md)。清单位于 `experiments/SharedMotion/data/`。原始视频、拟合、模型权重与访问凭据留在授权渠道。缺失和遮挡的置信度、接触、网格穿插、跨签者仍待审计。
