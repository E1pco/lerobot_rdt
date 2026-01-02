# 函数解析：RDT 数据采集与落盘（`RDT/`）

本章对应 2.3 的“数据集采集源代码 + 格式说明”，关键文件：
- `RDT/collect_rdt_dataset_teleop.py`
- `RDT/rdt_hdf5.py`
- `RDT/build_rdt_hdf5_from_raw.py`
- `RDT/inspect_rdt_hdf5.py`

## 1. `RDT/rdt_hdf5.py`

### 1.1 `rotmat_to_rot6d(R)`

- 输入：$3\times3$ 旋转矩阵
- 输出：6D 表示（取前两列，按列主序展平）

### 1.2 `UnifiedVector` / `make_unified_vector()` / `fill_slice(vec, sl, data)`

- `UnifiedVector.value`：`float32[128]`
- `UnifiedVector.mask`：`uint8[128]`
- `fill_slice`：写入 slice，并把对应 mask 置 1（自动截断到 slice 宽度）

### 1.3 `RDTHDF5EpisodeWriter`

- `append_step(...)`：追加 1 条 step（检查 images shape 必须匹配）
- `finalize_action_chunks()`：由 `(T,128)` 派生 `action_chunk (T,Ta,128)`
- `close()`：自动 finalize 并关闭文件

## 2. `RDT/collect_rdt_dataset_teleop.py`

### 2.1 `RawEpisodeWriter`

- 作用：写 raw episode（CSV + JPG），便于肉眼检查
- `append_step(...)`：
  - 保存 `timg*ncam` 张图片到 `images/`
  - 追加 proprio/action/mask/timestamp 到 CSV

### 2.2 图像预处理：`pad_to_square_and_resize_rgb(frame_bgr, out_size, pad_bgr)`

- pad 成正方形 → resize 到 `out_size` → 转 RGB

### 2.3 相机抽象：`MultiViewCamera`

- `_open(source, width, height)`：支持 index 或路径；支持 v4l2/gstreamer backend
- `read_bgr(view)`：读取指定 view 的一帧
- `close()`：释放所有 `VideoCapture`

### 2.4 速度估计：`compute_ang_vel_rad_s(R_prev, R_cur, dt)`

- 通过 $dR=R_{prev}^T R_{cur}$ 的旋转向量除以 dt 估计角速度

### 2.5 episode 编号：`_max_existing_episode_idx(out_dir)`

- 扫描 `episode_XXXXXX/` 与 `episode_XXXXXX.hdf5`，返回最大编号
- 用于避免覆盖、实现跨运行递增

### 2.6 采集主类：`JoyConRDTCollector`

- 初始化阶段：
  - 创建双臂 `ArmRig`
  - 可选打开三相机 `MultiViewCamera`
  - 配置 preview 与保存格式
- `run()`（主循环，函数体较长）：
  - 读 JoyCon 控制
  - 生成目标位姿→IK→下发
  - 若录制：构建 unified proprio/action → writer.append_step

## 3. raw→HDF5 与检查

- `build_rdt_hdf5_from_raw.py`：读取 raw 的 CSV/JPG，重建 HDF5 episode
- `inspect_rdt_hdf5.py`：打印 meta/shape，并可抽样检查内容

理论解释（unified/mask/action_chunk）见：`theory_rdt_format.md`。
