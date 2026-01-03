# 3. 基于人机交互的机械臂数据集构建（RDT）

快速跳转：

- 相机与手眼标定：`stage1_3_calibration.md`
- 时序同步说明：`stage1_3_timesync.md`
- 理论：手眼标定：`theory_handeye.md`
- 理论：RDT 数据格式：`theory_rdt_format.md`
- 函数解析：RDT 采集链路：`ref_rdt_pipeline.md`
- 函数解析：手眼评估：`ref_vision_handeye.md`

对应 `decument/task.txt` 的 3. 产出物（本页聚焦“采集与格式”）：

- 数据集：`RDT/collect_rdt_dataset_teleop.py` 采集输出（raw/HDF5）
- 采集演示视频：按本文“采集流程”录制
- 数据格式说明：本页 + `theory_rdt_format.md`

标定与同步属于 3. 的必要前置条件，但其详细流程独立成文：

- 相机/手眼标定：`stage1_3_calibration.md`
- 时序同步说明：`stage1_3_timesync.md`


### 1. 理论背景

RDT fine-tuning 期望输入是固定维度的向量与图像序列，因此本项目把所有 proprio/action 统一映射到 `float32[128]`，并用 `uint8[128] mask` 标注“哪些维度真实有效”。

同时，为了训练时的未来动作监督，数据中还会构造 `action_chunk`（默认 $T_a=64$），即从当前时刻起未来若干步的动作序列（越界部分做 padding，并把 mask 置 0）。

理论与张量形状的完整说明见：`theory_rdt_format.md`。

### 2. 本章目标

从“可控的遥操作”出发，采集一个或多个 episode，并把 raw 数据稳定转换为符合 RDT fine-tuning 约束的 HDF5。

### 3. 数据格式

- raw：落盘目录存在，图片与 CSV 数量对得上（肉眼可检查）
- HDF5：能被 `RDT/inspect_rdt_hdf5.py` 正确读取，张量 shape/字段合理，并能抽样可视化

## 程序设计结构

本章数据管线可抽象为“采集（raw）→ 结构化（HDF5）→ 校验（inspect）”，并通过 `unified 128 + mask + action_chunk` 保证张量维度一致且可解释：

1) **采集层**：以固定频率采样机器人状态（proprio）与相机图像（images），并记录动作（actions）与时间戳
2) **统一表示层**：把 proprio/action 映射到 `float32[128]`，并用 `uint8[128] mask` 标记有效维度
3) **序列监督层**：构造 `action_chunk (Ta=64)` 作为未来动作监督信号（越界 padding + mask=0）
4) **落盘层**：raw 便于人工检查，HDF5 便于训练与批量读取

对应实现索引：格式与张量形状见 `theory_rdt_format.md`，代码链路解析见 `ref_rdt_pipeline.md`。

## 脚本作用

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

## 方法作用

围绕“unified 128 + mask + action_chunk”的最关键方法主要在 `RDT/rdt_hdf5.py` 与 raw→HDF5 脚本里：

### 统一向量构造 (`RDT/rdt_hdf5.py`)

- **`UnifiedVector` 类**
  - **作用**：数据容器。
  - **细节**：维护一个 `float32[128]` 的值向量和一个 `uint8[128]` 的掩码向量。
- **`fill_slice(start_idx, end_idx, data)`**
  - **作用**：填充数据片段。
  - **细节**：将特定模态（如关节角、末端位姿）的数据填入 `UnifiedVector` 的指定切片区间，并自动将对应的 mask 设为 1。
- **`make_unified_vector(q, dq, ...)`**
  - **作用**：工厂方法。
  - **细节**：接收所有可能的输入（关节角、速度、末端位姿、夹爪状态等），按预定义布局组装成完整的 `UnifiedVector` 对象。

### HDF5 写入与 Chunking (`RDTHDF5EpisodeWriter`)

- **`add_step(proprio, action, images, ...)`**
  - **作用**：写入单步数据。
  - **细节**：将当前时刻的观测和动作暂存到内存 buffer 中，等待 episode 结束时统一处理。
- **`finalize_action_chunks(chunk_size=64)`**
  - **作用**：构造动作块（Action Chunking）。
  - **细节**：这是 RDT 训练的关键。对于时刻 $t$，它从 buffer 中提取 $t$ 到 $t+chunk\_size-1$ 的动作序列。如果接近 episode 结尾不足 64 步，则用 0 填充（Padding）并将对应的 mask 置 0。
- **`write_to_disk(path)`**
  - **作用**：落盘 HDF5。
  - **细节**：创建 HDF5 文件，定义数据集（Datasets）和属性（Attributes），将处理好的 chunked actions、images（经过压缩或 resize）和 proprio 写入文件。

### 图像处理与检查

- **`pad_to_square_and_resize_rgb(image, target_size)`**
  - **作用**：图像预处理。
  - **细节**：为了保持长宽比，先对短边进行 Padding 使图像变方，然后 Resize 到目标尺寸（如 224x224 或 300x300），防止图像拉伸变形影响模型训练。
- **`inspect_rdt_hdf5.py` 的可视化逻辑**
  - **作用**：数据验收。
  - **细节**：随机抽取 HDF5 中的一个 step，解码并显示图像，打印 proprio 和 action 的数值及 mask，用于人工确认数据是否对齐、mask 是否正确。

## 2. 前置条件

1) 1./2. 已跑通
2) 设备就绪：

- 机械臂串口：`/dev/left_arm`、`/dev/right_arm`（或 `/dev/ttyACM0` 等）
- 三路相机：`/dev/video*`

3) 配置正确：

- 左臂：`driver/left_arm.json`
- 右臂：`driver/right_arm.json`

4) 标定完成（推荐）：

- 相机内参：由 `vision/calibrate_camera.py` 生成
- 手眼：按 `vision/handeye_calibration_eyeinhand`和`vision/handeye_calibration_eyetohand`生成`handeye_result.yaml`

## 3. 采集流程

### 3.1 采集 raw（CSV + JPG，便于肉眼检查）

**步骤说明**：

1. **硬件连接**：确保左右臂串口（`/dev/left_arm`, `/dev/right_arm`）和三路相机（`/dev/video*`）已连接。
2. **启动脚本**：运行以下命令启动采集程序。

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

```bash
python RDT/inspect_rdt_hdf5.py ./rdt_raw/episode_000001.hdf5
```

**检查项**：

- **Shape**：确认 `images` 是 `(T, 2, 3, 384, 384, 3)`，`action` 是 `(T, 128)`。
- **Mask**：确认 `proprio_mask` 和 `action_mask` 非全零。
- **Visual**：脚本会弹窗显示随机抽取的几帧图像和对应的动作值。

**检查验收点**：

- `T` 与录制长度一致或近似（允许因丢帧/起止边界有少量差异）
- `mask` 不应全 0；若全 0，通常是字段未填充或 slice 配置错误（优先看 `ref_rdt_pipeline.md`）

## 4. 数据格式（对齐 RDT fine-tuning，简述 + 索引）

为了适配 RDT 模型的输入要求，我们采用了特定的数据结构。

核心约束：

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

## 5. 时序同步与标定

- 时序同步：实现与限制见 `stage1_3_timesync.md`
- 相机/手眼标定：流程与验收见 `stage1_3_calibration.md`

## 6. 理论补充

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

与同步相关的实现细节与限制请直接引用：`stage1_3_timesync.md`（避免在多处重复导致不一致）。
