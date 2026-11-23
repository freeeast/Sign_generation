# 无条件签语生成研究回合

在来源隔离的 NSA 签语拟合上训练了共享 Flow Matching 模型，可以直接从噪声输出 24 Hz、43 个局部关节的 4 秒 SMPL-X。试验使用 421 个训练录像和 59 个验证录像，32 个固定采样种子、两次独立训练种子，并用独立测试参考窗核对锁定方法。另以复取的 DDPM 及旧 GRU 档案保留历史参照。

主要发现是**将白高斯改为时间相关高斯，可以复验地缓解极端抖动**。两个训练种子均从 32/32 降到 0/32 超阈。rho=0 消融复现原张量与样本，加倍 Euler 步数保持失败。相关噪声的运动仍太快，覆盖代理改善伴随精度代理下降。单种子 rho=.95 在验证上进一步逼近左右手中位速度但低估转角尾部，暂留为候选。没有给出整体动作自然度、跨签者、语言语义或 SOTA 声明。

SignSparK 已审计代码及发布资产。官方尝试因依赖、GPU 和全流存储受阻，未生成官方样本。四种条件模式及下一次准确命令另行记录。真实身体接触、严格生理约束、长段、签者级独立性和高精度 3D 待进一步核对。科研结论限于某一已确认失败项的干预和复验。

- [文献](LITERATURE_REVIEW.md) · [基线](BASELINE_REPRODUCTION_REPORT.md) · [坏例](BAD_CASE_ANALYSIS.md)
- [消融](METHOD_AND_ABLATION_REPORT.md) · [实验账本](EXPERIMENT_LEDGER.md) · [复现](REPRODUCTION_GUIDE.md) · [局限](LIMITATIONS_AND_FUTURE_WORK.md)
