# 2.3 基于人机交互的机械臂数据集构建（RDT）+ 手眼标定 + 时序同步

快速跳转：
- 相机与手眼标定：`stage1_2_3_calibration.md`
- 时序同步说明：`stage1_2_3_timesync.md`
- 理论：手眼标定：`theory_handeye.md`
- 理论：RDT 数据格式：`theory_rdt_format.md`
- 函数解析：RDT 采集链路：`ref_rdt_pipeline.md`
- 函数解析：手眼评估：`ref_vision_handeye.md`

对应 `decument/task.txt` 的 2.3 产出物：
- 数据集：`RDT/collect_rdt_dataset_teleop.py` 采集输出（raw/HDF5）
- 采集演示视频：按本文“采集流程”录制
- 数据格式说明：本文“数据格式”
- 手眼标定/时序同步源代码：`vision/` + `RDT/`（见本文“标定与同步”）
- 手眼标定/时序同步设计手册：本文“标定与同步”

## 1. 数据集采集代码入口

- 采集（raw/HDF5）：`RDT/collect_rdt_dataset_teleop.py`
- raw→HDF5 合成：`RDT/build_rdt_hdf5_from_raw.py`
- HDF5 检查：`RDT/inspect_rdt_hdf5.py`
- 格式定义（unified 128 + action_chunk）：`RDT/rdt_hdf5.py`

采集脚本的默认录制按键与数据结构说明，见：`RDT/README.md`。

## 2. 采集流程（建议用于 2.3 演示视频）

### 2.1 采集 raw（CSV + JPG，便于肉眼检查）

示例（双臂 + 三相机）：

```bash
python RDT/collect_rdt_dataset_teleop.py \
  --device right \
  --control-arm right \
  --right-port /dev/right_arm \
  --left-port /dev/left_arm \
  --right-config ./driver/right_arm.json \
  --left-config ./driver/left_arm.json \
  --cam-exterior /dev/video0 \
  --cam-right-wrist /dev/video2 \
  --cam-left-wrist /dev/video4 \
  --out-dir ./rdt_finetune_hdf5 \
  --instruction "your task here"
```

默认会写：`--out-dir/episode_XXXXXX/`（每个 episode 一个目录）。

### 2.2 raw→HDF5 合成

```bash
python RDT/build_rdt_hdf5_from_raw.py ./rdt_finetune_hdf5/episode_000001
```

### 2.3 检查 HDF5（shape/字段/可视化抽样）

```bash
python RDT/inspect_rdt_hdf5.py ./rdt_finetune_hdf5/episode_000001.hdf5
```

## 3. 数据格式（对齐 RDT fine-tuning）

核心约束（以 `RDT/README.md` 与 `RDT/rdt_hdf5.py` 为准）：
- 统一向量：`float32[128]` + `uint8[128] mask`
- `Timg=2`，三相机 `Ncam=3`，默认 `384×384` RGB
- 每个 episode 一个 `.hdf5`；episode 结束派生 `action_chunk (T, Ta=64, 128)`

关键数据集（HDF5 内）：
- `observations/images`：`(T, 2, 3, 384, 384, 3)` `uint8`
- `observations/proprio` / `observations/proprio_mask`：`(T, 128)`
- `actions/action` / `actions/action_mask`：`(T, 128)`
- `actions/action_chunk` / `actions/action_chunk_mask`：`(T, 64, 128)`
- `timestamps_unix_s`：`(T,)`

## 4. 手眼标定与验证（2.3 产出物 4/5）

本仓库标定代码位于 `vision/`（不是 `hand_eye_calibration/`）。

### 4.1 相机内参标定

脚本：`vision/calibrate_camera.py`

- 采集：`python vision/calibrate_camera.py --capture --camid 2`
- 标定：`python vision/calibrate_camera.py --calibrate --image-folder ./vision/calib_images_right`
- 一键：`python vision/calibrate_camera.py --all --camid 2`

输出：会在对应 session 目录生成 `camera_intrinsics*.yaml`（并打印重投影误差报告）。

### 4.2 手眼标定（棋盘格）

- Eye-in-Hand（相机在末端）：`vision/handeye_calibration_eyeinhand.py`
  - `python vision/handeye_calibration_eyeinhand.py --collect --video 0 --port /dev/left_arm`
  - `python vision/handeye_calibration_eyeinhand.py --calibrate`
- Eye-to-Hand（相机固定在环境）：`vision/handeye_calibration_eyetohand.py`
  - `python vision/handeye_calibration_eyetohand.py --collect --video 0 --port /dev/left_arm`
  - `python vision/handeye_calibration_eyetohand.py --calibrate`

一致性评估：`vision/handeye_utils.py` 提供 `evaluate_*_consistency()` 与报告输出。

### 4.3 标定结果验证（蓝色圆跟踪）

脚本：`vision/track_blue_circle_eyetohand.py`
- 作用：检测蓝色圆 → PnP 求 $T_{target}^{cam}$ → 用手眼结果换算到基座系 → 驱动机械臂靠近目标。
- 运行：`python vision/track_blue_circle_eyetohand.py`

## 5. 数据时序同步设计（2.3 产出物 4/5）

## 6. 理论补充（建议写入 2.3 说明文档/答辩）

### 6.1 unified 128 + mask

本项目对齐 RDT fine-tuning 的关键点之一是：每条 proprio/action 都不是“只有数值”，而是“数值 + 可用性 mask”。

- `value`：`float32[128]`，缺失维度填 0
- `mask`：`uint8[128]`，可用维度置 1

实现对应：`RDT/rdt_hdf5.py::UnifiedVector/make_unified_vector/fill_slice`。

### 6.2 action_chunk（Ta=64）的构造

episode 结束时，把 per-step `action[t]` 堆叠为：

$$
\text{action\_chunk}[t,\tau] = \text{action}[t+\tau],\ \tau\in[0,Ta)
$$

越界部分做 0 padding，并把 mask 置 0。

实现对应：`RDT/rdt_hdf5.py::RDTHDF5EpisodeWriter.finalize_action_chunks()`。

更完整的格式推导与张量形状解释见：`theory_rdt_format.md`。

### 5.1 设计目标

- 让每个 control tick 的数据（双臂 proprio、动作、三相机最近 `Timg=2` 帧）在同一条时间轴上可对齐。

### 5.2 当前实现（代码映射）

- 统一采集循环：`RDT/collect_rdt_dataset_teleop.py`
  - 每个 tick：读 JoyCon / 读机械臂状态 / 取多相机帧 / 写入一条 step
- 时间戳：写入 `timestamps_unix_s`（Unix 秒）
- 图像堆叠：每个 tick 保存最近 `Timg=2` 帧（不足则 padding）

### 5.3 已知限制（需要在报告里说明）

- 多相机为软件顺序读取，非硬件触发同步；跨相机存在毫秒级偏差。
- 若需要更严格同步，应使用硬件同步相机或外部触发，并把每帧的独立时间戳写入文件。
