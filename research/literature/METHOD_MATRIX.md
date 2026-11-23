# 方法矩阵

作者报告的任务、数据与评价口径各异。配对语义条件下的 DTW 或翻译分数无法用来给无条件自然度排名。已核对作者论文、项目或发布源码，未运行的方法注明。

| 方法与问题 | 数据与监督 | 表示、模型、推理 | 作者指标与报告结果 | 发布与本项目意义 |
| --- | --- | --- | --- | --- |
| SignSparK，文本及稀疏关键帧条件生成 | 多语言拟合；发布版训练 CSL-Daily、How2Sign、BOBSL、CSL-News | 行式 rot6d，body 60、单手 90、face 56；双手为单手网络独立输入；分流时域 U-Net CFM，Euler/Heun | 配对 MPJPE/PA-MPJPE/DTW、回译；论文报告少于 10 步可合成，与当前发布版默认 50 步区分 | Apache-2.0 源码、LMDB 和三个权重可查。作为外部生成器优先检查，条件空化要审源码。[论文](https://arxiv.org/abs/2603.10446) · [代码](https://github.com/JianHe0628/SignSparK) |
| Neural Sign Actors，文本条件生成 | How2Sign RGB 与拟合的 SMPL-X 伪标签 | SMPL-X，解剖姿态编码和自回归去噪扩散 | 作者使用识别网络和视觉比较评价手语生产，未给本任务无条件对照 | 使用其官方拟合快照。未复验作者生成权重。[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html) · [发布](https://baltatzisv.github.io/neural-sign-actors/) |
| SignAvatars / SignVAE，整体手语生成基准 | 70,000 段、153 签者，孤立和连续；视频自动拟合，经时间约束和运动学优化 | 整体 SMPL-X，语义和动作 VQ-VAE 码本，再自回归预测 | 任务级生成与定性结果，多种提示；与本报告的动力学评价不等价 | 作者代码与数据入口存在，使用前核对各来源授权。签者更多可用于外部验证。[项目](https://signavatars.github.io/) |
| HandMDM，语言描述控制单手运动 | 大规模签语单手轨迹，噪声类别和 LLM 动作描述 | 局部手部运动的文本条件扩散 | 报告跨类别、跨签语和非签语的手部生成，主要验证手部 | 可借鉴输入和指部细节。全身和左右手耦合需单独验证。[论文与代码](https://imagine.enpc.fr/~leore.bensabath/HandMDM/) |
| VideoMDM，2D 监督学习 3D 运动先验 | HumanML3D 投影及 Fit3D/NBA 视频 2D | 扩散用骨架重投影监督训练 3D 去噪，推理从噪声开始 | 作者 HumanML3D FID 0.88，对照完全 3D MDM 0.54，真实视频另有人评 | 作者提供代码入口。不是手语专用评估器，需先验证手指遮挡、相机尺度和独立 3D。[项目](https://videomdm.github.io/) |
| DexAvatar，单目视频重建 | 单目 2D 观测与学习的手、身姿态先验，SGNify mocap 评价 | SMPL-X 对输入视频拟合 | 作者报告相较既有重建基线身体和手部估计改善 35.11% | 适合改善确诊的伪标签缺陷，非随机采样基线。[WACV](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html) · [代码](https://github.com/kaustesseract/DexAvatar) |
| SignNMFlow，语义条件下的自然度路径 | 作者报告 PHOENIX-2014T、CSL-Daily、How2Sign | 学习基于自然度和数据拟合的流形路径，再训练生成场 | 预印本的 DTW / 对齐 DTW 指标与用户偏好均为条件生成。独立无条件对照未查到 | arXiv 本轮无法直接取回，搜索索引确认题名及方法描述，权重和源码未验证。待取得作者原文复核。[预印本](https://arxiv.org/abs/2609.33339) |
| M3T，离散多模态签语生产及识别 | How2Sign、CSL-Daily、PHOENIX14T；联合动作、脸和语言 | SMPL-FX 表情、身/手/脸 FSQ-VAE，自回归 Transformer | 作者报告 How2Sign 身/手 DTW-PA-JPE 4.28/2.28（联合任务）。仅作配对任务参考 | 项目标注代码即将发布。未确认权重。离散化代价需通过逐指重建和无条件实验核查。[项目](https://cogvis-cvssp.github.io/papers/m3t/) |
| MV-Sign，较新的文本条件 3D 生成 | 以 How2Sign 前侧视构建联合 3D 重建 | 多视角融合训练标签和改进文字条件 | 作者报告文本生成基准改善，不能替代我们对真实 3D 的独立评估 | 关注 How2Sign-MV3D 可用性和同签者跨集情况，未复验。[预印本](https://arxiv.org/abs/2611.07811) |

无条件实验使用 [Flow Matching](https://arxiv.org/abs/2210.02747) 的直线路径及现有 SMPL-X 双手表示。当前研究数据只使用已核验 NSA 快照。
