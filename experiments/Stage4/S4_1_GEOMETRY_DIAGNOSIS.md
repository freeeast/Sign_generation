# S4.1 · 手部几何与输入域

**篮球有明显的逐帧平移误差，平移无法解释全部手部偏差。** 使用上一轮 582 帧投影缓存，逐段核验了 RGB SHA-256。新运行 MediaPipe Hand Landmarker（IMAGE 模式），从 940 个输出手实例中接受 931 个。接受的左右手由 RTMW 腕点最近邻分配，并排除距离过大或左右歧义；MediaPipe 的关键点坐标来自独立模型，手的匹配仍部分依赖 RTMW。奇数帧接受 461/582 个可能的手实例。MediaPipe 缺测保留为空。Neural Sign Actors 的原拟合流程也使用过 MediaPipe 关节。这里的观测独立于 Uni-Sign 和 RTMW 的分值，但可能与原拟合共享检测器偏差。[NSA 拟合方法](https://guanxiongsun.github.io/projects/neural-sign-actors/#data-a-3d-version-of-how2sign)

只对已有相机加帧级平移。用偶数帧的鼻、耳、肩的中位残差，线性插值预测奇数帧，所有手点和识别分数均未参与求解。按各段奇数帧的中位误差看，RTMW 的全图指尖误差改善 4/5 段，MediaPipe 改善 4/5 段。两个检测器对踢拳与诊断片段的变化方向不同。相对腕点手型误差对统一平移不变，这是定义上的不变性。

| 重点失败帧，零起始 | RTMW 指尖误差 px | MediaPipe 指尖误差 px | 读图和误差归因 |
|---|---:|---:|---|
| 篮球 9 | 132.0 → 57.5 | 117.8 → 43.1，双手 | 身体和手一起偏右；约 75.3 px 的头肩平移移除了主要全局偏差，手臂仍有残差。 |
| 诊断 13 | 54.6 → 46.8 | 35.8 → 27.8，单手 | 双臂交叉，手腕和手指有遮挡。MediaPipe 只匹配到一只手；RTMW 和 MediaPipe 在可配对手上的腕相对指尖坐标差 15.5 px。 |
| 踢拳 109 | 70.1 → 78.5 | 85.8 → 70.5，单手 | 手肘、前臂和头部的错位不同步。往返两检测器的全局趋势有分歧，RTMW 与 MediaPipe 在共同匹配手上的指尖仅差 4.7 px。 |

现有拟合保留了 `cam_trans`，原始逐帧 crop 逆变换、焦距、主点仍缺。帧级平移只作为本轮可复现的几何对照，见各段 NPZ 的 `shift`。手部保留 COCO/MediaPipe 的 21 点同名拓扑，模型点为 SMPL-X 回归关节或表面顶点。遮挡下的 2D 检测点和模型点没有保证物理同位。RTMW 的置信通道可以大于 1，上游会裁剪；阈值仍有可见点并不保证指尖位置准确。

[三帧对比叠图](geometry/failure_frames.png) · [逐段与逐帧误差](geometry/geometry.csv) · [MediaPipe 手部标注模型](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
