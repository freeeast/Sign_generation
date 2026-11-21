# E1-Real · 真实拟合和真人输入审计

**G1 在本批五段验证集样本上通过。** 使用用户提供的中性 SMPL-X 模型完成 582 帧前向运动学和 133 点投影。每个拟合的左右肩方向与真人一致，交换左右手腕会使中位误差由 15.0–25.0 px 上升到 129.6–178.9 px。固定相机后，在同一批奇数帧比较 ±4 帧偏移，5/5 段都在零偏移时误差最低。

## 模型与投影验收

模型有 10,475 个顶点、55 个关节回归项、486 个姿态修正基和 400 个形状/表情基。运行使用拟合提供的 10 维体型、10 维表情及整只手的 45 维轴角，使用 SMPL-X model hand mean。模型核心数组为有限值。资产 SHA-256、形状和运行配置见 [model_asset_audit.json](model_asset_audit.json) 和 [geometry_qc.json](geometry_qc.json)。投影保留原来的 69 个 Uni-Sign 输入点组合。

| 片段 | 帧 | 头肩与腕点对应的身体中位误差 | 腕点中位误差 | 指尖中位误差 | 指尖 P95 |
|---|---:|---:|---:|---:|---:|
| 炸薯条 `_2FBDaOPYig_1` | 126 | 11.4 px | 15.0 px | 12.6 px | 39.0 px |
| 上色 `0zvsqf23tmw_16` | 140 | 15.4 px | 16.8 px | 12.8 px | 33.4 px |
| 踢拳 `3TrMyzNWGpY_4` | 112 | 17.7 px | 24.3 px | 15.6 px | 52.9 px |
| 篮球 `bBHYQCDp_9A_4` | 110 | 20.4 px | 21.1 px | 21.7 px | 86.9 px |
| 诊断 `f5EGPzGSCJs_13` | 94 | 18.7 px | 25.0 px | 18.5 px | 69.0 px |

使用偶数帧的鼻、耳、肩拟合分轴恒定仿射，以奇数帧的 1280 × 720 原图像素和 RTMW 点测量误差。腕点中位约为各段肩宽的 7.4–13.5%，指尖为 6.2–10.7%。相机裁剪逆变换缺失，数据只支持用恒定仿射通过粗对齐。无法区分原始拟合偏差、裁剪变化、手指遮挡和 RTMW 定位误差各自的影响。

**几何失败帧。** 篮球第 9 帧的 SMPL-X 上身整体向图像右侧偏移，手的位置有明显差异，见 [RGB 叠图](bBHYQCDp_9A_4-3-rgb_front/rgb_overlay.png)。诊断片段有 6.8% 的所选点落在画幅外，比分段中位误差显示的风险更高。所有片段保留在后续评分中。各段保存 3D 骨架、2D 投影和 RGB 叠图，逐段偏移和有效区域数据见 [alignment_audit.csv](alignment_audit.csv)。当前验收可以开展跨域诊断，指尖 P95 提示这些拟合仍有局部误差。

## 资源实况

| 资源 | 结果 |
|---|---|
| 真实 SMPL-X、RGB、现场重提 RTMPose | 首轮 5 对候选，完整 ID 与帧数一致，首轮另有 4 段只含真人侧的对照。全部视频为 1280 × 720、24 fps。 |
| How2Sign CSV | train、val、test 原始及 realigned 标注均已读取，分别为 31,165、1,741、2,357 行。验证集两份 CSV 与 Dataverse 官方分发逐字节一致。 |
| SMPL-X 官方模型 | 使用本次上传的 `SMPLX_NEUTRAL.npz`，沿用 `smplx==0.1.28` 和项目既有 FK。 |
| 拟合相机 | 每帧含 `cam_trans`。焦距、主点、原图裁剪逆变换和拟合帧时戳未随帧发布。上传样本的 z 为约 20–22.5 模型单位，符合 OSX 虚拟长焦坐标的量级，完整外参或原图坐标仍待校验。 |
| Uni-Sign 归档与权重 | 姿态目录索引有 train 31,087、test 2,350 个文件，val 无发布。重用了姿态权重，SHA-256 与前一轮一致。9 段 val 用官方代码 vendored 的 lightweight RTMW 192 重提。 |

| 句段 ID | 拟合帧 | 原始 RGB 帧 | 来源 |
|---|---:|---:|---|
| `_2FBDaOPYig_1-3-rgb_front` | 126 | 126 | 用户附件 |
| `0zvsqf23tmw_16-2-rgb_front` | 140 | 140 | 官方 val 前缀 |
| `3TrMyzNWGpY_4-2-rgb_front` | 112 | 112 | 官方 val 前缀 |
| `bBHYQCDp_9A_4-3-rgb_front` | 110 | 110 | 官方 val 前缀 |
| `f5EGPzGSCJs_13-8-rgb_front` | 94 | 94 | 官方 val 前缀 |

## 附件 ID 与时间

附件是 “Today I am going to be showing you how to make Gourmet French Fries cooked in duck fat.”，原始切分 **3.15–8.39 s**，realigned **4.78–9.05 s**。消息中 “arm drill” 参考句的 ID 是 `-d5dN54tH2E_0-1-rgb_front`，真人侧单独评分。只按句子 ID 搜索可能合并不同 take，初查的去重镜像为同一英文句返回 **168 帧**。本次用完整 `SENTENCE_NAME` 取得 **126 帧**。再从 Dataverse 的原视频 ZIP 用区间请求取单个原视频，抽查 7 个画面。每个画面最佳匹配都在 `原视频第 76 帧 + 片段帧号`，图像均值绝对像素差中位数约 **1.09 / 255**，realigned 时间附近约 **31.60 / 255**。这个检查证实 RGB 切段依据，新一轮用上述 FK 叠图及零偏移对照核验了帧对应。

## 参数与接口

- 五段每个帧文件均含根轴角 3、身体轴角 63、左右手各 45、下颌 3、体型 10、表情 10、相机平移 3。帧号无缺口，字段均为有限浮点值。手部维数符合展开后的 15 关节轴角，运行配置采用 OSX 的 model hand mean。OSX 公开代码使用 neutral、`use_pca=False`、默认 model hand mean，眼球旋转补零。
- 原有 133 点映射和 69 点选择复用。把 `cam_trans` 保存在独立字段。OSX 假设投影中只加入一次 `cam_trans`，输入为裁剪坐标。暂以裁剪视角或根据偶数帧的头肩点拟合恒定仿射作诊断，固定后在奇数帧身体、手腕、指尖及脸部度量重投影。它不是原相机标定。面部轮廓、手指关节中心与 RTMPose 表面点存在对应误差。
- 真人侧姿态与抽帧可视化见 `real_rtmpose_keyframes.png`。拟合原始参数轨迹见 `uploaded_fit_parameter_tracks.png`，该图没有运行 FK。虚拟识别结果见 [E3 报告](../E3/E3_RESULTS.md)。

## 存档译文复核

按带 ID 的官方存档与 test 标签比对，**2,349/2,349** 条参考句对应。用官方 `forward` 和 `generate` 在相同 CPU float32 上直接打分，仍仅 **3/9** 与存档译文逐字一致，5 个短序列的差异完全不涉及抽帧。唯一 416 帧片段测试均匀抽帧与 3 个随机种子，NLL 为 **1.382–1.451**，4 个设置均没有还原归档译文。与前轮适配器同采样下的 NLL 最大差 **0.00293**。上游框架的 BF16、批次、原始发布权重或其预处理版本差异仍待隔离。当前实验结论限于固定权重和配置下的相对 NLL，需记录对采样敏感的小边际。

数据来源：[How2Sign](https://how2sign.github.io/) · [原始与 realigned 标注](https://dataverse.csuc.cat/dataset.xhtml?persistentId=doi%3A10.34810%2Fdata33) · [带完整 ID 的 RGB 镜像](https://huggingface.co/datasets/Flimdejong/how2sign/tree/8ebd757f151ab3cb74741d961abf93f0e36fff40) · [Neural Sign Actors](https://baltatzisv.github.io/neural-sign-actors/) · [SMPL-X 官方下载](https://smpl-x.is.tue.mpg.de/download.php) · [OSX](https://github.com/IDEA-Research/OSX) · [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign)
