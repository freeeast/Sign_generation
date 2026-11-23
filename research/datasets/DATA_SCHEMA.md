# 数据与导出

`sample.csv` 使用 `split,clip,source,frames,path`，以及现有 QC 的手、根、腿、上身 P99、形状跨度、相机跨度和 >90° 帧步数。路径相对数据根，逐片 NPZ 由 `scripts.shared_motion_data` 生成。官方 ZIP 使用序号连续且 CRC 匹配的逐帧记录，训练仅消费 train 和 val 的完整片。官方验证和测试保留独立。每片内采样 96 帧、步幅 48，附加最后一窗。标准化参数只用训练集。低置信度信息需从 2D 缓存加入，当前生成数据没有统一每帧覆盖。

输入必需 `global_orient[T,3],body_pose[T,63],left_hand_pose[T,45],right_hand_pose[T,45]`。可含 `jaw_pose,leye_pose,reye_pose,betas,expression,cam_trans`。原 cam_trans 为拟合相关偏移，未作生成全局位移使用。导出保留轴角与 fps，且 `transl` 为零。外部官方 SMPL-X 中性模板加载使用 `use_pca=False,flat_hand_mean=False`。SignSparK row rot6d 欲转入本工程 column rot6d，应先按源行规则解码到旋转矩阵，再按本地规则编码并核对左右手和关节顺序。原视频对应How2Sign句段 ID，原坐标、时序和相机缓存仅保留已授权路径。
