# 方法与消融

直路径 Flow 的输入为标准化 (x_t=(1-t)z+t x)，模型预测 (x-z)。E1 将 (z) 替换为内部的平稳 AR(1) 变换，(z_0=epsilon_0)，(z_t=0.85 z_{t-1}+sqrt{1-0.85^2}epsilon_t)。每关节通道独立抽取 (epsilon)，所有帧的边缘方差仍为一。采样使用同一变换，100 步 Euler。该消融测试噪声时间结构在训练与采样的联合影响。

[迭代一](reports/ITERATION_01.md)涵盖 E0、E1、E2 与第二训练种子 E3。[迭代二](reports/ITERATION_02.md)评价 rho=.95 的单训练种子验证。[反例和数值诊断](bad_cases/ROOT_CAUSE_ANALYSIS.md)解释为何优先测试时序噪声，而将 2D 和手部先验留待独立证据。核心方法实现 [flow.py](../experiments/generative_motion/flow.py)。
