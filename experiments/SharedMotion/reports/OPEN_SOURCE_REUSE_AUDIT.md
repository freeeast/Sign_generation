# 开源复用审计

具体公开 Git 版本在 [版本表](../analysis/repository_commits.csv)。入口通过实际代码树核对，权重栏指文档给出的渠道，未实际加载第三方大模型。GPU 需求列按已公布依赖和相对成本，未为未知仓库估计精确显存。

| 资源 | 训练、推理、权重与复用入口 | 双手、数据与无文本 | 成本与条款 |
|---|---|---|---|
| Neural Sign Actors | 官方项目提供三套 fitted ZIP；未在项目页核实完整生成代码或扩散权重。实际使用现仓 src/nsa_motion.py 的读取逻辑 | 双手与身体、脸 SMPL-X 拟合，可用于无文本训练 | 数据直接接入；How2Sign 为 CC BY-NC 4.0，SMPL-X 模板分开授权；未核实 NSA 拟合单独的商业许可。[项目](https://baltatzisv.github.io/neural-sign-actors/) |
| SignAvatars | vis.py、visualizer/、datasets/；公开树中未核实 SignVAE 训练、完整采样或权重 | 双手 SMPL-X，标注适合重新建立无监督 loader | 标注非商业申请，旧 PyTorch/CUDA 示例；适配中等。[仓库](https://github.com/ZhengdiYu/SignAvatars) |
| SignAvatar | dataset/dataset.py、train/train_cvae.py、train/train_stgcn.py、generate/generate_sequences_pose.py、generate/generate_sequences_mesh.py；权重经作者申请 | 含双手和脸，CVAE 依赖词条件，无文本需改 label 与 decoder | CC BY-NC 4.0，适配中等。[仓库](https://github.com/dongludeeplearning/SignAvatar) |
| SOKE | train.py、test.py、configs/deto.yaml、configs/soke.yaml、mGPT/。文档列 tokenizer checkpoint | 双手分流 SMPL-X，tokenizer 可脱离文本研究，整体生成按文本条件 | T5 和 GPU 训练较重，仓库条款 CC BY-NC-ND 4.0。[仓库](https://github.com/2000ZRL/SOKE) |
| HandMDM | train.py、inference.py、configs/train_bobsl.yaml、src/data/、src/normalizer.py；列原模型及 THMR 下载 | 双手与上身 274 维，LMDB；条件 dropout 接口可研究，纯无文本长时间训练未核实 | 文档 Python 3.9、Torch 2.4 CUDA 11.8；根目录单独许可证未见，依各资产核查。适配中高。[仓库](https://github.com/leorebensabath/HandMDM) |
| SignSparK | train.py、sample.py、sample_all.py、flow_train_util.py、tools/inspect_lmdb.py；文档发布各流权重与辅助回译 | 关键帧条件，身体／双手／脸 LMDB，纯无条件配置未复现 | Torch 2.10 与 CUDA 12.8 文档环境，Apache 2.0 代码，权重和语料另核。分流中等，全套较高。[仓库](https://github.com/JianHe0628/SignSparK) |
| Sign-VQ | sign_vq/train.py、sign_vq/dataset.py、sign_vq/poses_to_codes.py、codes_to_poses CLI，README 给样例训练；未充分核实公开权重可得性 | MediaPipe pose 与 Masked NumPy ZIP，FSQ 无文本；改为双手 SMPL-X 时需 rot/pose loader | CC BY-NC 4.0，适合改造做压缩消融。[仓库](https://github.com/sign-language-processing/sign-vq) |
| MotionGPT | train.py、demo.py、prepare/download_pretrained_models.sh、configs/config_h3d_stage2.yaml，权重和分阶段指南 | HumanML3D；标准词表不覆盖手指。可无文本训练 tokenizer | MIT 代码，语料另有条款。重训多流 tokenizer 及语言模型成本高。[仓库](https://github.com/OpenMotionLab/MotionGPT) |
| T2M-GPT | train_vq.py、models/vqvae.py、train_t2m_trans.py，README 列权重及推理指导 | HumanML3D/KIT；VQ 阶段可不使用文本，双手需重建表示及损失 | Apache 2.0，适配中高。[仓库](https://github.com/Mael-zys/T2M-GPT) |
| MDM | train/train_mdm.py、sample/generate.py、sample/edit.py、sample/predict.py，有权重与无条件采样 | HumanML3D/KIT、动作数据；手部输出需替换并重训 | Torch/CUDA、CLIP，MIT；训练成本高于 GRU。[仓库](https://github.com/GuyTevet/motion-diffusion-model) |
| PriorMDM | sample/double_take.py、sample/two_person_prefix_completion.py、sample/finetuned_motion_control.py；训练脚本和权重 | 通用身体长段及腕控制；现成模型不覆盖 SMPL-X 双手，补全方法可迁移 | MIT、Torch/CUDA；中高。[仓库](https://github.com/priorMDM/priorMDM) |
| HuMoR | humor/models/humor_model.py、humor/fitting/motion_optimizer.py、humor/train/train_humor.py、get_ckpt.sh，提供无文本训练和采样 | AMASS／SMPL+H；当前预训练目标不包含本任务的逐指手语，转移结构可用 | MIT；SMPL 和 AMASS 授权分开。适配中高。[仓库](https://github.com/davrempe/humor) |
| SignBot | data_process_body.py、dex-retargeting/example/data_process_hand.py、legged_gym/legged_gym/scripts/train.py 与 play.py，文档有示例策略，生成跳转 SOKE | SMPL-X 双手动作到机器人，跟踪策略不以文本为输入 | Isaac Gym、Torch CUDA 12.1、RSL-RL，各子模块条款，成本高。留作机器人端。[仓库](https://github.com/qiaoguanren/Signbot) |
| VIPL-SLP 索引 | README.md、README_old.md，可检索论文和链接，无独立训练、推理、权重 | 没有运动数据接口 | CC0 索引。[仓库](https://github.com/VIPL-SLP/awesome-sign-language-processing) |

**已在本轮复用的代码**为既有 NSA 字段验证、官方 SMPL-X FK 和网格渲染、Stage 10 局部样条、逐手守门、冻结 Uni-Sign，以及 Stage 10 的同 RTMW RGB 复测。第三方候选仅完成源码审计。
