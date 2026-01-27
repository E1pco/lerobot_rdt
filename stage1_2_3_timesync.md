# 3. 数据时序同步说明

本章描述 3. 数据采集链路中“如何对齐同一控制周期的数据”，以及当前实现的限制与可改进方向。

这里的“同步”并不等价于硬件同触发，而是：在离散控制周期（control tick）内，把多模态观测尽可能对齐到同一时间轴，保证下游能解释“这张图/这个 proprio/这个 action 属于同一次决策”。

因此本章关注的是：

- 以 tick 为单位的对齐策略
- 写入 HDF5 的时间戳语义
- 现有软件采集带来的偏差与需要在报告里说明的限制

## 程序设计结构

当前实现采用**单线程单循环**驱动，把每个 tick 的采样顺序固定下来，并把“近似同时刻”的数据写入同一个 step：

1) 读输入（遥操作）
2) 读机器人状态（proprio）
3) 取相机图像（按相机顺序）并堆叠最近 `Timg=2` 帧
4) 写入 action/proprio/images 与 `timestamps_unix_s`

这种结构的优点是实现简单、复现路径清晰；缺点是跨相机与跨模态必然存在毫秒级偏差。

## 脚本作用

- `RDT/collect_rdt_dataset_teleop.py`：采集主循环与 tick 对齐逻辑（决定“同步语义”）。
- `RDT/build_rdt_hdf5_from_raw.py`：把 raw 序列整理成 HDF5。
- `RDT/rdt_hdf5.py`：HDF5 字段定义（包含 `timestamps_unix_s` 等）。
- `RDT/inspect_rdt_hdf5.py`：通过 shape/抽样可视化检查对齐效果是否符合预期。

## 方法作用（对齐在代码里落到哪些“写入点”）

### 采集主循环 (`collect_rdt_dataset_teleop.py`)

- **`main_loop()` 中的顺序执行逻辑**
  - **作用**：定义软同步策略。
  - **细节**：在一个 `while` 循环（Tick）内，严格按照“读手柄 -> 读机械臂 -> 读相机 -> 存数据”的顺序执行。虽然各设备没有硬件触发信号连接，但这种顺序执行保证了在代码逻辑上它们属于同一个“Step”。
- **`time.time()` 调用**
  - **作用**：记录时间戳。
  - **细节**：在读取完所有传感器数据后，调用一次 `time.time()` 获取当前 Unix 时间戳，并将其作为该 Step 的 `timestamp` 写入 CSV/HDF5。这为后续分析对齐误差提供了粗粒度的基准。

### 数据处理与 HDF5 (`rdt_hdf5.py` / `build_rdt_hdf5_from_raw.py`)

- **图像堆叠 (Image Stacking)**
  - **作用**：构造时序输入。
  - **细节**：在写入 HDF5 时，对于时刻 $t$，不仅写入当前帧 $I\_t$，还可能根据配置（`T_img`）写入历史帧 $I\_{t-1}$。若 $t=0$，则复制 $I\_0$ 进行 Padding。这在 `RDTHDF5EpisodeWriter` 中处理。
- **`timestamps_unix_s` 字段**
  - **作用**：数据对齐的验证依据。
  - **细节**：在 HDF5 中作为一个独立 Dataset 存储。在多机采集或多模态融合场景下，可以通过比较不同流的 timestamp 来评估同步偏差。

## 1. 同步目标

在每个 control tick，尽量让以下数据能在同一时间轴上对齐：

- 三相机图像（exterior/right-wrist/left-wrist）最近 `Timg=2` 帧
- 双臂 proprio（关节/末端等，写入 unified 128 + mask）
- 当前动作 action（写入 unified 128 + mask）

## 2. 当前实现（代码映射）

主循环：`RDT/collect_rdt_dataset_teleop.py`

- 单一循环驱动：每个 tick 顺序执行“读输入 → 读状态 → 取图像 → 写 step”。
- 时间戳：写入 `timestamps_unix_s`（Unix 秒）。
- 图像堆叠：每个 tick 保存最近 `Timg=2` 帧（不足则 padding）。

## 3. 已知限制

- 多相机为软件顺序读取：跨相机存在毫秒级偏差（非硬触发）。
- 图像与关节状态并非同一时刻采样：受采集顺序与 USB/串口抖动影响。

## 4. 可改进方向

- 为每路相机单独记录帧级时间戳（每帧一个 timestamp），并在 HDF5 中额外保存。
- 多线程/异步采集相机，主线程按时间最近邻对齐。
- 使用硬件触发同步相机（最严格），或使用同一同步信号写入日志。
