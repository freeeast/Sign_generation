# Uni-Sign How2Sign 姿态输入契约

| 环节 | 实际约定 |
|---|---|
| 官方归档 | pickle 字典，包含 `keypoints` 和 `scores`。实测批次维度为单人，形状分别为 `(T,1,133,2)` 和 `(T,1,133)` |
| 适配器 NPZ | `keypoints: (T,133,2)`，`scores: (T,133)`；xy 分别按画幅宽高归一化，x 向右、y 向下。画幅外坐标可能小于 0 或大于 1 |
| 布局 | COCO-WholeBody。Body 取 0 和 3–10，左手 91–111，右手 112–132，脸部取 23–39（步长 2）、83–90 和 53。零起始索引，共 69 点 |
| 归一化 | Body 用整段中分数大于 0.3 的点建立方形 crop；手部以本手腕为原点，脸部以索引 53 为原点，三者按同一身体尺度归一化。模型输入每点包含 x、y 和 detector score |
| 置信度 | 官方发布的原始分数大于 1 是可观察的现象。本次保留原值交给上游 clipping 和 mask。模型阈值为 `score > 0.3`，缺失 landmark 以 0 表示 |
| 时间 | 保留原帧序。超过 256 帧时，适配器均匀保留含首尾的 256 帧；官方 loader 使用排序后的随机抽帧。9 个公开片段中仅有一个触发适配器的抽帧 |
| 推理 | 复用 `load_part_kp`、pose-only collator、Uni-Sign 模型和 mT5 解码。模型每个部位的输入为 `(B,T,V,3)`，beam size 4，最多 100 个新 token |
| 两种反馈 | 对固定参考句做 teacher forcing，取非 padding token 平均 NLL，含 EOS，默认最多 50 tokens。特征为 mT5 encoder 的 768 维输出，对有效姿态 token 取平均，排除翻译 prompt |

上游验证从官方代码和 9 个真实归档条目交叉确认。公开 README 里的模型结果可能与同一仓库当前提供的权重、输入版本或解码配置不完全一致。本次独立复跑的一致性与作者存档译文一致性分别在 `experiments/E0/reproduction.json` 中列出。[数据处理源码](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/datasets.py) · [模型源码](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/models.py)
