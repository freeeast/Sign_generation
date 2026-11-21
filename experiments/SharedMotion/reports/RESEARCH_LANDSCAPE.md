# 研究现状与创新边界

截至 2026-11-18。本文区分论文结果、可见代码和本项目实测。

## 手语专用三维重建与生成

| 方法 | 已展示及适配价值 |
|---|---|
| Neural Sign Actors，CVPR 2024 | 从 How2Sign 拟合身体、双手和脸的 SMPL-X，再训练文本条件扩散。重建与跨序列生成是两个阶段。论文报告分布指标、回译及定性比较。现项目用其拟合作伪标签。 [项目及数据](https://baltatzisv.github.io/neural-sign-actors/) |
| SignAvatars / SignVAE，NeurIPS 2024 | 多语料全身双手 SMPL-X。项目汇报约 70K 序列、8.34M 注释。分别对语义和运动建 VQ 码本，再自回归预测动作码。适合作为更广的数据来源与离散基线。 [项目](https://signavatars.github.io/) |
| SignAvatar / ASL3DWord，2024 | Transformer CVAE 做词级 3D 生成及重建，表征含双手和脸，通过 ST-GCN 等评价。可借词级手形和条件采样，跨词平滑与长句仍需额外建模。 [作者论文](https://arxiv.org/abs/2405.07974) · [仓库](https://github.com/dongludeeplearning/SignAvatar) |
| SOKE，ICCV 2025 | 身体部件分流 tokenizer、多头解码和手语词典检索辅助跨语种文本到 3D。可迁移左右手分流与可控语义锚点。发布权重训练域与评估数据须去重。 [论文与源码](https://github.com/2000ZRL/SOKE) |
| HandMDM，CVPR 2026 | 文本描述手形、位置、动作的短片扩散。274 维 SMPL-X 旋转特征包含左右手和 13 个上身关节，作者另提供评价专用的检索模型。BOBSL3DT / ASL 词级来源和 14 帧、25 Hz 的公开设置不能直接外推长 ASL 句。 [项目](https://imagine.enpc.fr/~leore.bensabath/HandMDM/) · [代码](https://github.com/leorebensabath/HandMDM) |
| SignSparK，ECCV 2026 | 以稀疏关键帧为 3D 锚点，条件 flow matching 生成手、身体和脸。公开分流权重和 LMDB，作者注明发布版数据与权重不同于论文版。适合后续研究稀疏条件和动作段补全。 [项目](https://cogvis-cvssp.github.io/papers/signspark/) · [源码](https://github.com/JianHe0628/SignSparK) |
| M3T，BMVC 2026 | SMPL-FX 扩充脸部表征，身体、手和脸采用独立 FSQ-VAE，报告模态码本使用率。迁移重点是检查每种手形是否被离散码保留。当前未核验可复用发布权重。 [论文记录](https://openresearch.surrey.ac.uk/esploro/outputs/conferenceProceeding/M3T-Discrete-Multi-Modal-Motion-Tokens-for/991154594102346) |
| MV-Sign，2026 预印本 | 多视角 3D 信号、身体手部时空 VQ 与伪 gloss／检索增强文本条件。适合参考多视评估与条件设计，未作复现。 [arXiv](https://arxiv.org/abs/2611.07811) |
| SignBot | 以 SMPL-X 手语动作重定向到人形机器人，用仿真跟踪策略执行，并组合 Uni-Sign 与 SOKE。适合未来执行端，其物理控制训练与本项目的候选动作筛选是不同问题。 [源码](https://github.com/qiaoguanren/Signbot) |
| DexAvatar，WACV 2026 | SignBPoser 与 SignHPoser 作为重建先验，后者使用手部动作捕捉。符号域手部先验及可见性约束已有紧邻先例。 [论文](https://openaccess.thecvf.com/content/WACV2026/papers/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.pdf) |

## 通用动作、运动先验和共享控制

- **预测**。GRU 把历史压缩到可延续状态，TCN 提供明确有限接收域，小 Transformer 可以更灵活地取远程上下文。短时强基线是保持末姿。训练看到真人历史、推理只看自身历史，会产生 exposure bias，需独立报告一步和自由滚动。 [残差与静止基线](https://openaccess.thecvf.com/content_cvpr_2017/html/Martinez_On_Human_Motion_CVPR_2017_paper.html) · [TCN 比较](https://arxiv.org/abs/1803.01271)
- **潜空间先验**。HuMoR 学条件 VAE 状态转移，TestOpt 通过可微先验拟合噪声观测，另有自由生成变体。通用 AMASS 权重与本项目双手逐指签语域不直接对应，可借条件转移、潜变量与观测项分离的形式。 [HuMoR](https://openaccess.thecvf.com/content/ICCV2021/html/Rempe_HuMoR_3D_Human_Motion_Model_for_Robust_Pose_Estimation_ICCV_2021_paper.html)
- **离散 token**。T2M-GPT 用 VQ 加动作 Transformer，MotionGPT 用运动词表做统一语言建模。通用身体权重缺少本项目双手的训练通道。重训前先检验独立手指的重建失真，再做分流及码本利用率。 [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) · [MotionGPT](https://github.com/OpenMotionLab/MotionGPT)
- **扩散和流**。MDM 提供无条件、条件和补全代码。PriorMDM 提供长段拼接、前缀和腕控制，迁移方式是用手部表示重训小块，保留关键帧 inpainting。频繁调用冻结专家会导致每个采样步额外成本。 [MDM](https://github.com/GuyTevet/motion-diffusion-model) · [PriorMDM](https://github.com/priorMDM/priorMDM)
- **模仿与修正策略**。摊销优化学习选择时间窗、手指块和幅度。可从真实优化轨迹提取同起点的几何、专家和失败标签，保留弃权。DAgger 提醒应采集模型自己的状态；目前缺独立可靠收益标签，离线 RL、PPO 或偏好奖励还缺关键前提。 [DAgger](https://proceedings.mlr.press/v15/ross11a.html)

## 评估和科学新意

三维可见动作需分别检查腕、双手逐指、速度、手形转折、接触和多视一致性。DTW 可以容许不同节奏，但会掩盖需精确模拟的时机偏差。独立检测器、留出录像、人评和回译回答不同问题。参与优化的 Uni-Sign 得分只衡量其响应，共用 RGB 重检测能检验投影域偏差，语义须用独立签者／识别体系。 [指标研究与人评](https://aclanthology.org/2025.wmt-1.4/) · [评价工具](https://github.com/sign-language-processing/pose-evaluation)

共享生成、运动先验、通用零文本模型已有明确先例，手语专门的双手生成与手先验也已有相近研究。已核实的长句系统多数按文本或运动语义条件报告，不能据此断言其论文已验证了本项目的纯历史设定。训练一个普通共享预测器不足以主张方法创新。可证伪的贡献是检验共享手先验能否在同几何起点、同预算和独立测量下减少专家导致的手形退化，或能否在未见源优先选中真正有收益的动作块。
