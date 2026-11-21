# 研究状态交接

## 当前工作点

目标是从真人示范出发，通过冻结 Uni-Sign 反馈和可行的 3D 更新让化身逐步模仿签语，逐步把每片独立优化推进为可迁移的动作能力。直接分支通过 SMPL-X FK 和固定相机投影输入姿态专家。RGB 分支把虚拟网格渲染成画面，并对真人与虚拟画面使用同配置 RTMW 和 Uni-Sign。两条分支的专家分数可能不同。源级划分只证明原录像 ID 不重叠，签者级泛化尚未建立。

| 工作 | 已执行的结论 | 查证入口 |
| --- | --- | --- |
| E0–E3.5 | 五段拟合 582 帧进入识别管线。配对编码检索 5/5，配对 NLL 16/20，NLL 在局部指与腕朝向修复上分别 2/5、1/5。 | [实验决策](reports/GO_NO_GO_DECISION.md)，[复现](STAGE3_REPRODUCE.md) |
| Stage 4–8 | 逐片闭环成立，局部双手特征可用，几何是强基线。Stage 7 修正重复右手特征。Stage 8 跑过多种起点和粗 IK。 | [Stage 7](experiments/Stage7/)，[Stage 8](experiments/Stage8/) |
| Stage 9–10 | Stage 9 使用分手、逐指控制。Stage 10 区分固定相机与人体根平移，使用坐姿软稳定和局部手形保护。焦点第 90 帧的 61.85→29.08 px 主要由几何重锚定贡献。6 段中直接专家改善 6/6，同检测器 RGB 2/6，明显的留出几何收益见于其中一段锁定录像。 | [Stage 10 执行摘要](experiments/Stage10/reports/S10_EXECUTIVE_SUMMARY.md)，[复现](experiments/Stage10/README_STAGE10.md) |
| SharedMotion（最新） | NSA 官方档案条目 30,965/1,724/2,318。本轮实用 600/75/100 条，按源录像分开。选中 43 个局部关节、165,633 参数、16 帧历史、3 步训练目标的 SO(3) 增量 GRU，名义 12 Hz。 | [中文摘要](experiments/SharedMotion/reports/EXECUTIVE_SUMMARY_CN.md)，[数据清单](experiments/SharedMotion/data/sample.csv)，[模型选取](experiments/SharedMotion/model/selection.json) |

## 最近一次结果

| 自由滚动时长 | Hold 手部角误差 | GRU 手部角误差 |
| --- | ---: | ---: |
| 一步 | 10.52° | 10.68° |
| 0.5 s | 22.06° | 20.90° |
| 1 s | 25.61° | 23.95° |
| 2 s | 28.85° | 27.73° |
| 4 s | 29.72° | 35.23° |

4 秒处手指速度约为 NSA 拟合参考的 7%，长段动作趋向过平滑。四段闭环开发录像与共享模型拟合源分开，但已用于旧闭环开发。共享先验加专家与无共享先验的同起点、同守门、36 提案专家对照相比，留出综合几何最大差约 0.004 px。RGB 复评近乎持平。证据支持有限短时的跨录像动作预测，尚未显示可辨认的共享先验闭环增益。见[预测结果](experiments/SharedMotion/reports/MOTION_GENERATION_RESULTS.md)和[闭环结果](experiments/SharedMotion/reports/FEEDBACK_INTEGRATION.md)。

**Conditional GO。** 现有可运行状态是 3D 逐片受约束闭环、短时共享预测器和先验正则接入。优先的后续实验是在同一固定切分上训练小型分手连续短块预测，用 Hold、阻尼速度和 GRU 比较 0.5–1 秒的逐指动态和腕局部 3D，达标后把模型用于生成多轨候选并使用 Stage 10 原守门。短块模型与候选生成是拟议工作。详见[下一阶段说明](experiments/SharedMotion/reports/NEXT_STAGE_RESEARCH_PROPOSAL.md)。

## 工程映射

| 对象 | 路径 |
| --- | --- |
| 共享模型与旋转表示 | `src/shared_motion.py` |
| 数据、训练、分析 | `scripts/shared_motion_{data,train,analysis}.py` |
| 闭环、RGB、视觉产物 | `scripts/shared_motion_{feedback,rgb,visuals}.py` |
| Stage 10 提案、守门与固定配置 | `scripts/stage10_run.py`、`scripts/stage10_inputs.py`、`configs/stage10.json` |
| 旧真实配对缓存 | `experiments/Stage6/data/`、`experiments/Stage7/data/` |
| 最新权重和逐源轨迹 | `experiments/SharedMotion/model/`、`experiments/SharedMotion/feedback/` |
| 完整演示和动作导出 | `experiments/SharedMotion/visuals/`、`experiments/Stage10/visuals/` |
| 独立文字生成试验 | `parallel/ASL_SMPLX_UniSign_Pilot/` |
| 归档合并溯源 | `SOURCE_ARCHIVES.tsv`、`MERGE_CONFLICTS.tsv`、`archives/superseded/` |

## 数据答疑

已有研究闭环使用匹配 How2Sign 真人句段的 Neural Sign Actors（NSA）SMPL-X 拟合，在不同阶段也使用从 2D 观测构造的粗 IK 起点和优化后动作。复验已配对的小样本无需读取整个 NSA Train/Val/Test。重新训练共享动作模型时才按固定清单获取训练样本和验证、测试样本，保持官方切分及源分组。本包保留代表性轨迹、配对缓存和选定权重，完整 775 条训练／验证／测试原始拟合需按官方许可重新取得。
