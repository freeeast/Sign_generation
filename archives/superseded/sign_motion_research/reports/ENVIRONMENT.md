# 环境与复现记录

- 执行日 2026-11-16。CPU，4 个 PyTorch 线程，float32。GPU 环境未实测。
- Python 3.12.14，PyTorch 2.6.0+cpu，torchvision 0.21.0+cpu，Transformers 4.40.0，DeepSpeed 0.16.3，SMPL-X Python 包 0.1.28。完整安装命令见 [README](../README.md)。
- Uni-Sign 源码 commit `eed438bcb49e30405cd6ccdfcccca330c134e830`。How2Sign 姿态专用权重和输入归档发布版本 `eab251b7fe7e8521afc0e67be98add670ea40a0d`。[源码](https://github.com/ZechengLi19/Uni-Sign/tree/eed438bcb49e30405cd6ccdfcccca330c134e830) · [权重和归档](https://huggingface.co/ZechengLi19/Uni-Sign/tree/eab251b7fe7e8521afc0e67be98add670ea40a0d)
- 核验 checkpoint 文件 SHA-256 为 `1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d`。使用 `weights_only=True` 读取，`strict=True` 加载。
- 权重、原始姿态归档与 SMPL-X 模型资产通过上述官方渠道获取。公开实验证据包含片段 ID、输入来源、逐片段输出、数值、图和可执行代码。
- 开始时工作目录仅含任务和综述。尝试读取 Neural Sign Actors 官方共享包的局部字节时，服务器返回完整约 7.25 GB 文件的 HTTP 200，未取得小样本。复现实验使用公开的 Uni-Sign RTMPose 关键点。其归档实际有 train 和 test 两个目录，未提供 validation 目录。[Neural Sign Actors 项目](https://baltatzisv.github.io/neural-sign-actors/)
