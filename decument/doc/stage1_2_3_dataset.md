# 2.3 基于人机交互的机械臂数据集构建（RDT）

快速跳转：
- 相机与手眼标定：`stage1_2_3_calibration.md`
- 时序同步说明：`stage1_2_3_timesync.md`
- 理论：手眼标定：`theory_handeye.md`
- 理论：RDT 数据格式：`theory_rdt_format.md`
- 函数解析：RDT 采集链路：`ref_rdt_pipeline.md`
- 函数解析：手眼评估：`ref_vision_handeye.md`

对应 `decument/task.txt` 的 2.3 产出物（本页聚焦“采集与格式”）：
- 数据集：`RDT/collect_rdt_dataset_teleop.py` 采集输出（raw/HDF5）
- 采集演示视频：按本文“采集流程”录制
- 数据格式说明：本页 + `theory_rdt_format.md`

标定与同步属于 2.3 的必要前置条件，但其详细流程独立成文：
- 相机/手眼标定：`stage1_2_3_calibration.md`
- 时序同步说明：`stage1_2_3_timesync.md`

## 0. 理论背景 + 目标与验收（论文式开头）

### 0.1 理论背景：为什么必须是 unified 128 + mask + action_chunk

RDT fine-tuning 期望输入是固定维度的向量与图像序列，因此本项目把所有 proprio/action 统一映射到 `float32[128]`，并用 `uint8[128] mask` 标注“哪些维度真实有效”。

同时，为了训练时的未来动作监督，数据中还会构造 `action_chunk`（默认 $T_a=64$），即从当前时刻起未来若干步的动作序列（越界部分做 padding，并把 mask 置 0）。

理论与张量形状的完整说明见：`theory_rdt_format.md`。

### 0.2 本章目标

从“可控的遥操作”出发，采集一个或多个 episode，并把 raw 数据稳定转换为符合 RDT fine-tuning 约束的 HDF5。

### 0.3 你需要交付/证明的结果

- raw：落盘目录存在，图片与 CSV 数量对得上（肉眼可检查）
- HDF5：能被 `RDT/inspect_rdt_hdf5.py` 正确读取，张量 shape/字段合理，并能抽样可视化

**最小验收**：完成 1 个 episode 的“采集→转换→检查”闭环。

## 1. 数据集采集代码入口（你会用到的脚本）

本模块的核心目标是采集符合 RDT Fine-tuning 格式要求的多模态数据。

- **采集主程序**：`RDT/collect_rdt_dataset_teleop.py`
  - **功能**：同时连接双臂（串口）和三路相机（USB），以 30Hz（默认）的频率同步记录数据。
  - **输出**：Raw 格式（CSV + JPG 图片文件夹），便于人工检查和筛选。
- **格式转换工具**：`RDT/build_rdt_hdf5_from_raw.py`
  - **功能**：将 Raw 数据打包为高效的 HDF5 格式，并进行图像预处理（Resize、Padding）。
- **数据检查工具**：`RDT/inspect_rdt_hdf5.py`
  - **功能**：读取生成的 HDF5 文件，打印张量形状，并可视化抽样帧，确保数据无误。
- **格式定义库**：`RDT/rdt_hdf5.py`
  - **功能**：定义了 `UnifiedVector`（统一向量）和 `RDTHDF5EpisodeWriter` 类，封装了 HDF5 的写入逻辑，确保符合 `unified 128 + action_chunk` 规范。

采集脚本的默认录制按键与数据结构说明，见：`RDT/README.md`。

## 2. 前置条件（不满足会导致采集失败或数据不可用）

1) 2.1/2.2 已跑通：能稳定遥操作（建议先完成 `joycon_ik_control_py.py` 的录屏级稳定控制）
2) 设备就绪：
- 机械臂串口：`/dev/left_arm`、`/dev/right_arm`（或 `/dev/ttyACM0` 等）
- 三路相机：`/dev/video*`
3) 配置正确：
- 左臂：`driver/left_arm.json`
- 右臂：`driver/right_arm.json`
4) 标定完成（推荐）：
- 相机内参：由 `vision/calibrate_camera.py` 生成（建议记录你实际使用的 `camera_intrinsics.yaml` 路径）
- 手眼：按 `stage1_2_3_calibration.md` 生成 `handeye_result.yaml`（并完成一致性评估）

> 说明：采集 raw/HDF5 的脚本不一定强制依赖手眼，但如果你的下游训练/验证需要把视觉结果转换到机器人坐标系，标定缺失会让“复现”不严谨。

## 3. 采集流程（建议用于 2.3 演示视频）

### 3.1 采集 raw（CSV + JPG，便于肉眼检查）

**步骤说明**：
1.  **硬件连接**：确保左右臂串口（`/dev/left_arm`, `/dev/right_arm`）和三路相机（`/dev/video*`）已连接。
2.  **启动脚本**：运行以下命令启动采集程序。

示例（双臂 + 三相机）：

```bash
python RDT/collect_rdt_dataset_teleop.py \
  --device right \
  --control-arm right \
  --save-format raw \
  --right-port /dev/right_arm \
  --left-port /dev/left_arm \
  --right-config ./driver/right_arm.json \
  --left-config ./driver/left_arm.json \
  --cam-exterior /dev/video0 \
  --cam-right-wrist /dev/video2 \
  --cam-left-wrist /dev/video4 \
  --out-dir ./rdt_raw \
  --instruction "pick up the red block"
```

**参数详解**：
- `--control-arm`: 指定主控臂（遥操作手柄控制的臂）。
- `--cam-exterior`: 第三人称视角相机（Eye-to-Hand）。
- `--cam-*-wrist`: 手腕相机（Eye-in-Hand）。
- `--instruction`: 当前任务的自然语言描述（将写入 HDF5 元数据）。
- `--save-format raw`: 保存为“CSV + JPG”的 raw episode（便于人工检查）。
- `--out-dir`: 输出根目录；raw 模式下会在其下创建 `episode_000001/`、`episode_000002/`…

> 可选：如果你希望“直接生成 HDF5（跳过 raw→HDF5）”，把 `--save-format` 改为 `hdf5` 即可。

**运行交互**：
- 程序启动后会显示相机画面预览。
- 按 `Space` 键开始/暂停录制。
- 按 `ESC` 键结束当前 Episode 并保存。
- 默认输出路径（raw 模式）：`--out-dir/episode_000001/`（按已有 episode 自动续号）。

**raw 验收点（建议录屏/截图）**：
- `episode_*` 目录内存在图像文件夹与 CSV（或等价日志文件），且数量随录制时长增长
- 能肉眼回看图像序列，确认相机视角正确、曝光正常

### 3.2 raw→HDF5 合成

采集完成后，需要将散乱的图片和 CSV 转换为紧凑的 HDF5 文件。

```bash
# 转换单个 episode
python RDT/build_rdt_hdf5_from_raw.py ./rdt_raw/episode_000001
```

**处理逻辑**：
- 读取 CSV 中的 `proprio` 和 `action` 数据。
- 读取 `images/` 目录下的 JPG 图片。
- 调用 `pad_to_square_and_resize_rgb` 将图片统一调整为 384x384 分辨率。
- 生成 `action_chunk`（未来动作块）。
- 写入 HDF5 文件。

**转换验收点**：默认会在同级生成 `./rdt_raw/episode_000001.hdf5`，且文件大小随 episode 时长增长。

### 3.3 检查 HDF5（shape/字段/可视化抽样）

在投入训练前，务必检查生成的 HDF5 文件是否合法。

```bash
python RDT/inspect_rdt_hdf5.py ./rdt_raw/episode_000001.hdf5
```

**检查项**：
- **Shape**：确认 `images` 是 `(T, 2, 3, 384, 384, 3)`，`action` 是 `(T, 128)`。
- **Mask**：确认 `proprio_mask` 和 `action_mask` 非全零。
- **Visual**：脚本会弹窗显示随机抽取的几帧图像和对应的动作值。

**检查验收点**：
- `T` 与你的录制长度一致或近似（允许因丢帧/起止边界有少量差异）
- `mask` 不应全 0；若全 0，通常是字段未填充或 slice 配置错误（优先看 `ref_rdt_pipeline.md`）

## 4. 数据格式（对齐 RDT fine-tuning，简述 + 索引）

为了适配 RDT 模型的输入要求，我们采用了特定的数据结构。

核心约束（以 `RDT/README.md` 与 `RDT/rdt_hdf5.py` 为准）：
- **统一向量 (Unified Vector)**：所有的物理量（位置、速度、力矩等）都被映射到一个固定的 `float32[128]` 向量中。
- **Mask 机制**：配合 `uint8[128] mask`，标记哪些维度是有效的。例如，如果只控制了 6 个关节，则只有对应的 6 个位置维度 mask 为 1，其余为 0。
- **多帧输入**：`Timg=2`，即模型输入包含当前帧和上一帧。
- **Action Chunking**：采用 `action_chunk` 机制，预测未来 `Ta=64` 步的动作序列。

关键数据集（HDF5 内）：
- `observations/images`：`(T, 2, 3, 384, 384, 3)` `uint8`。维度含义：(时间, 历史帧数, 相机数, 高, 宽, 通道)。
- `observations/proprio` / `observations/proprio_mask`：`(T, 128)`。机器人本体状态。
- `actions/action` / `actions/action_mask`：`(T, 128)`。当前时刻的动作。
- `actions/action_chunk` / `actions/action_chunk_mask`：`(T, 64, 128)`。未来动作块。
- `timestamps_unix_s`：`(T,)`。Unix 时间戳，用于时序对齐分析。

更完整的格式推导与张量形状解释见：`theory_rdt_format.md`。

## 5. 时序同步与标定（作为复现“必要说明”的索引）

- 时序同步：实现与限制见 `stage1_2_3_timesync.md`
- 相机/手眼标定：流程与验收见 `stage1_2_3_calibration.md`

## 6. 理论补充（建议写入 2.3 说明文档/答辩，可选）

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

与同步相关的实现细节与限制请直接引用：`stage1_2_3_timesync.md`（避免在多处重复导致不一致）。

## 7. 常见问题（采集链路排错）

- 相机预览卡顿/丢帧：降低分辨率、减少 USB 带宽占用，或减少同时打开的相机数量
- `mask` 全 0：优先检查 unified vector 的 slice 填充逻辑（见 `ref_rdt_pipeline.md`）
- raw→HDF5 报找不到图片：确认 episode 目录结构与图片文件名是否与脚本预期一致
