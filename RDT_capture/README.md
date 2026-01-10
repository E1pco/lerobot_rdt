# RDT Fine-tuning HDF5 Collector (Dual-arm + 3 cameras)

本目录提供一套**对齐论文 arXiv:2410.07864v2（RDT / ICLR'25）fine-tuning 数据格式**的采集与落盘工具：

- `rdt_hdf5.py`：HDF5 episode writer + 128 维 unified space（Table 4）工具函数
- `collect_rdt_dataset_teleop.py`：JoyCon 遥操采集脚本（双臂数据 + 三相机）
- `build_rdt_hdf5_from_raw.py`：把 raw episode（CSV+JPG）合成为标准 HDF5 episode

---

## 1. `rdt_hdf5.py`

### 1.1 作用

- 定义论文中的 **128 维统一动作/本体感知空间**（Table 4 对应的 index ranges）。
- 提供 `UnifiedVector(value, mask)`：
  - `value`: `float32[128]`
  - `mask`: `uint8[128]`（0/1，标记哪些维度在当前数据里可用）
- 提供 episode 级 HDF5 落盘：每个 episode 一个 `.hdf5` 文件。
- 在 episode 结束时派生 `action_chunk (T, Ta, 128)`（默认 `Ta=64`）。

### 1.2 主要结构

#### Table 4 slice 常量

用 `slice(start, stop)` 定义 128 维向量每一段的含义（半开区间 `[start, stop)`）。例如：

- Right arm
  - `RIGHT_ARM_JOINT_POS = slice(0, 10)`
  - `RIGHT_GRIPPER_POS = slice(10, 15)`
  - `RIGHT_ARM_JOINT_VEL = slice(15, 25)`
  - `RIGHT_GRIPPER_VEL = slice(25, 30)`
  - `RIGHT_EEF_POS = slice(30, 33)`
  - `RIGHT_EEF_ROT6D = slice(33, 39)`
  - `RIGHT_EEF_LIN_VEL = slice(39, 42)`
  - `RIGHT_EEF_ANG_VEL = slice(42, 45)`

- Left arm
  - `LEFT_ARM_JOINT_POS = slice(50, 60)`
  - `LEFT_GRIPPER_POS = slice(60, 65)`
  - `LEFT_ARM_JOINT_VEL = slice(65, 75)`
  - `LEFT_GRIPPER_VEL = slice(75, 80)`
  - `LEFT_EEF_POS = slice(80, 83)`
  - `LEFT_EEF_ROT6D = slice(83, 89)`
  - `LEFT_EEF_LIN_VEL = slice(89, 92)`
  - `LEFT_EEF_ANG_VEL = slice(92, 95)`

- Base（如需）
  - `BASE_LIN_VEL = slice(100, 102)`
  - `BASE_ANG_VEL = slice(102, 103)`

> 注意：采集脚本里只会填充它“确实能计算/读取到”的段位；其余维度保持 0 且 mask=0。

#### 工具函数

- `rotmat_to_rot6d(R: (3,3)) -> (6,)`
  - 6D 旋转表示：取旋转矩阵前两列并按列主序展开。
- `make_unified_vector() -> UnifiedVector`
  - 创建全零 `value/mask`。
- `fill_slice(vec, sl, data)`
  - 把 `data` 写入 `vec.value[sl]`，同时把对应 `vec.mask` 置 1。

#### `RDTHDF5EpisodeWriter`

一个**append-only** 的 episode writer。

- `append_step(...)`：每个 control tick 写入一次。
- `finalize_action_chunks()`：根据每步 action 派生 `action_chunk`。
- `close()`：自动 finalize，再关闭文件。

### 1.3 HDF5 布局与 shape

写入的关键 datasets：

- `meta/*`：以 attrs 存储元信息（`instruction`, `timg`, `ncam`, `image_size`, `ta`, `control_hz` 等）

- `observations/images`：
  - shape `(T, Timg=2, Ncam=3, H=384, W=384, C=3)`
  - dtype `uint8`
  - camera 顺序固定：`[exterior, right-wrist, left-wrist]`

- `observations/proprio` / `observations/proprio_mask`：
  - shape `(T, 128)`
  - dtype `float32` / `uint8`

- `actions/action` / `actions/action_mask`：
  - shape `(T, 128)`
  - dtype `float32` / `uint8`

- `actions/action_chunk` / `actions/action_chunk_mask`：
  - shape `(T, Ta=64, 128)`

- `timestamps_unix_s`：shape `(T,)`
- `ik_success`：shape `(T,)`（1/0）

---

## 2. `collect_rdt_dataset_teleop.py`

### 2.1 作用

- 用 JoyCon 遥操一只指定手臂（`--control-arm`），并在每个控制周期采集：
  - 三相机（exterior/right-wrist/left-wrist）最近 `Timg=2` 帧（pad-to-square + resize 到 `384×384`）
  - 双臂 proprio（关节角/关节速度/末端位姿/末端速度/夹爪）写入 unified 128（right/left 分段）
  - 本周期的动作（目标关节角/夹爪）写入 unified 128
- 录制时按 episode 写成 HDF5：依赖 `RDTHDF5EpisodeWriter`。

### 2.2 主要结构

#### 图像处理

- `pad_to_square_and_resize_rgb(...)`
  - 将输入 BGR 帧 padding 成正方形，再 resize 到 `--image-size`（默认 384），最后转 RGB。
- `MultiViewCamera`
  - 负责打开/读取 3 路视频源：
    - `--cam-exterior`
    - `--cam-right-wrist`
    - `--cam-left-wrist`

#### 双臂抽象：`ArmRig`

每只手臂维护一份状态与配置：

- 硬件：`ServoController`（可为 None，表示 `--no-robot`）
- 运动学：`create_so101_5dof()` 的 robot 实例
- 关节/传动：`joint_names`, `gear_sign`, `gear_ratio`, `home_pose`
- 状态缓存：`current_q`, `prev_q`, `prev_eef_pos`, `prev_eef_R`
- 参考基座：`base_pos`, `base_rpy`（用于 JoyCon offset）
- 夹爪：`gripper_pos_steps` + `min/max/step`

#### 采集主类：`JoyConRDTCollector`

- 初始化：
  - `right_arm = _init_arm(...)`（从 `--right-port/--right-config`）
  - `left_arm = _init_arm(...)`（从 `--left-port/--left-config`）
  - 可选 home：`Home` 按键会对两臂同时回中，并重连 JoyCon
  - 可选相机：`MultiViewCamera`
- 控制循环 `run()`：
  1. 读取 JoyCon `get_control()`
  2. 生成目标位姿 `T_goal = base_pose + joycon_offset`
     - RPY 符号对齐 `joycon_ik_control_py.py`：`roll/pitch` 取反（`[-pose[3], -pose[4], pose[5]]`）
  3. 对 `--control-arm` 指定的手臂做 IK 并下发舵机目标
  4. 若正在录制：
     - 取 `Timg=2` 的三相机帧（缺失视角会用 pad 色填充）
     - 构建 `proprio`（双臂写入 right/left 对应 slice）
     - 构建 `action`（本周期右/左目标关节角写入对应 slice）
     - `writer.append_step(...)`

### 2.3 CLI 参数（常用）

- 双臂：
  - `--right-port` / `--left-port`
  - `--right-config` / `--left-config`
  - `--baudrate`
- JoyCon：
  - `--device {right,left}`：使用哪只 JoyCon
  - `--control-arm {right,left}`：JoyCon 控制哪只手臂
- 相机：
  - `--cam-exterior`, `--cam-right-wrist`, `--cam-left-wrist`
  - `--cam-width`, `--cam-height`
- 输出：
  - `--out-dir`
  - `--instruction`
- 运行开关：
  - `--no-robot` / `--no-camera` / `--no-home`

---

## 3. 典型运行命令（双臂 + 三相机）

> 你的设备命名：right 是 `/dev/right_arm`，left 是 `/dev/left_arm`。

```bash
python /home/elpco/code/lerobot/lerobot_rdt/RDT/collect_rdt_dataset_teleop.py \
  --device right \
  --control-arm right \
  --right-port /dev/right_arm \
  --left-port /dev/left_arm \
  --right-config /home/elpco/code/lerobot/lerobot_rdt/driver/right_arm.json \
  --left-config /home/elpco/code/lerobot/lerobot_rdt/driver/left_arm.json \
  --cam-exterior /dev/video0 \
  --cam-right-wrist /dev/video2 \
  --cam-left-wrist /dev/video4 \
  --out-dir /home/elpco/code/lerobot/lerobot_rdt/rdt_finetune_hdf5 \
  --instruction "your task here"
```

默认会先保存 **raw（CSV + JPG）** 到 `--out-dir/episode_XXXXXX/` 目录，方便你直接打开查看。

把 raw episode 合成 HDF5：

```bash
python /home/elpco/code/lerobot/lerobot_rdt/RDT/build_rdt_hdf5_from_raw.py \
  /home/elpco/code/lerobot/lerobot_rdt/rdt_finetune_hdf5/episode_000001
```

如需采集时直接写 HDF5（不落 raw）：

```bash
python /home/elpco/code/lerobot/lerobot_rdt/RDT/collect_rdt_dataset_teleop.py \
  --save-format hdf5 \
  ...
```

---

## 4. 录制控制（默认按键）

- `Y`：开始/结束录制（toggle）
- `A`：开启新 episode（并进入录制状态）
- `X`：退出
- `Home`：双臂回中 + JoyCon 重连
- `ZR/R`：夹爪收紧/放松（作用于 `--control-arm` 指定的那只手臂）
- `+/-`：速度调节

---

## 5. 备注与已知行为

- `observations/images` 的 3-view 顺序固定为：exterior、right-wrist、left-wrist。
- 缺失视角会用 `--pad-bgr` 指定颜色填充。
- `--no-robot` 下不会填充 proprio/action 的 slice（mask 会保持 0）。
- `action_chunk` 在 episode `close()` 时生成。
