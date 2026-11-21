# 实现选择

主实现采用 MDM 所代表的全段动作扩散路线。我们的局部连续 6D、左右各 15 关节和模型手均值需要独立保留，直接使用仓库内 `src/shared_motion.py` 的 rotation/6D/decode 路径及 `src/smplx_to_pose.py` 的 SMPL-X FK。扩散采用公开的 DDPM 公式，纯 PyTorch 实现，便于在现有 CPU 完成回归。本次未复制或加载上游生成权重。

备选为 rectified flow，优先参考 SignSparK 的手、身和面分流与采样边界。沿用本次数据、上身+双手格式和评估接口，先测试同预算整段生成。若以后转向 2D 弱监督，在共享生成模型的训练目标增加软投影和可见性损失，保留纯噪声采样接口。
