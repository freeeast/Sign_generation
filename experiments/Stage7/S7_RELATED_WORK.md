# 相邻方法与可验证的贡献

| 相关工作 | 与 Stage 7 的关系 |
| --- | --- |
| SGNify，CVPR 2023 | 用手语先验处理孤立手势的手姿态歧义，并用动捕及感知测试检验。本文的几何与外观不足可直接借鉴其数据验证方式。 [论文](https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html) |
| DexAvatar，WACV 2026 | 用学习的手与身体 3D 先验加 2D 重投影约束细化拟合。其先验和更强的手部检测是下一轮有价值的比较对象。 [论文](https://arxiv.org/abs/2512.21054) |
| Neural Sign Actors，CVPR 2024 | 释放了本阶段使用的初始 SMPL-X 轨迹。其原优化已有 MediaPipe 重投影及时间约束，留出 MediaPipe 不能视为完全独立于数据生成。 [论文](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html) |
| SignAvatars，ECCV 2024 | 大规模全身签语动作语料和生成基准，可用于单独的控制器泛化与运动先验比较。 [论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/653_ECCV_2024_paper.php) |
| MotionCritic，ICLR 2025 | 一般人类运动的感知偏好模型说明可以将动作评价和生成分开，仍需验证是否覆盖本任务需要的手型、接触等细粒度信息。 [论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c129741a2451e5fefe447591e39de30e-Abstract-Conference.html) |
| SignLLM，CVPR 2024 | 用签语到文本的表征对齐辅助翻译。对齐信号可借鉴，翻译语义相近与几何可用性仍需分别评价。 [论文](https://openaccess.thecvf.com/content/CVPR2024/html/Gong_LLMs_are_Good_Sign_Language_Translators_CVPR_2024_paper.html) |
| Meaningful Pose-Based Sign Language Evaluation，WMT 2025 | 将关键点、表征和回译区分开，本轮实际调用其 pose-evaluation DTW 度量。 [论文](https://aclanthology.org/2025.wmt-1.4/) · [代码](https://github.com/sign-language-processing/pose-evaluation) |

本轮最可辩护的贡献是可复现的**消融和失效证据**：相同运动、相同预算和固定评估集合下，专家距离、候选接受、正面几何以及可用的侧面几何有何分歧。独立控制并测量可见性和左右手激活提取可复查旧结果。加权损失、分层、几何守门和重排已有通用先例，当前数据不足以成立一个新的重建算法或语义指标。要提出方法贡献，需增加独立专家、多视角复测和适当 3D 先验基线，并预先锁定增益判据。
