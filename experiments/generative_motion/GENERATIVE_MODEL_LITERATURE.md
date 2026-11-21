# 生成模型调研

| 候选 | 公开实现与输入 | 与当前 43 关节任务的关系 |
| --- | --- | --- |
| [MDM](https://github.com/GuyTevet/motion-diffusion-model) | MIT，公开训练、采样、权重，含 unconstrained 分支 | 有成熟的全段噪声采样框架。公开通用人体权重需重训或适配双手。 |
| [Neural Sign Actors](https://baltatzisv.github.io/neural-sign-actors/) | CVPR 2024，SMPL-X 手语分身、以文本为条件的扩散和公开拟合数据入口 | 直接复用数据。当前使用的作者入口可复取数据，完整作者训练与权重本轮未复验。 |
| [SignAvatars / SignVAE](https://github.com/ZhengdiYu/SignAvatars) | 全身手脸拟合及生产基准，分离运动和语义编码 | 可借鉴手部表示、数据筛选和运动先验。需另核验授权、预训练权重及不同姿态分布。 |
| [SignSparK](https://github.com/JianHe0628/SignSparK) | Apache-2.0，公开 6D LMDB、分流 flow-matching 训练、采样及权重 | 最接近的公开手部代码。它使用稀疏关键帧和按 body/hand/face 流训练，需确认左右手约定并移除条件接口后重训。 |
| [HandMDM](https://github.com/leorebensabath/HandMDM) | 作者仓库提供手部文本条件扩散 | 可复核局部手部表示，双手与上身同步需另实现。当前无条件基线不加载其权重。 |
| [MLD](https://github.com/ChenFengYe/motion-latent-diffusion) | 作者训练、推理和模型下载入口，运动 latent 上扩散 | 降低长序列开销需要先验证自编码器保留高速指部运动。 |
| [MotionGPT](https://github.com/OpenMotionLab/MotionGPT) / [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) | 作者代码和权重下载入口，运动 token 架构 | 通用人体 token 缺目标手部细节，适合后续语言条件，需重建分手词表。 |
| [VideoMDM](https://videomdm.github.io/) | 作者页提供论文及代码入口，视频 2D 弱监督的 3D 扩散 | 未来 2D 辅助路线。手指遮挡、相机尺度、多个样本同一重投影需单测，未引入本轮训练。 |

本轮查证了论文/作者站点和发布入口，以及 MDM、SignSparK、MLD 等的公开实现。上游代码许可与数据或模型许可分别处理。本轮无第三方生成模型二进制作为依赖，未执行上游 checkpoint。
