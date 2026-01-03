# Stage 1 总览



## 程序设计结构

按模块分层，可以把仓库拆成 5 块：

- **驱动层**（协议与串口）：`driver/ftservo_driver.py`
- **控制层**（面向“关节名”的动作接口）：`driver/ftservo_controller.py` + `driver/*.json`
- **运动学层**（FK/IK 与角度↔步数映射）：`ik/robot.py`（及 `ik/`）
- **交互层**（键盘/JoyCon 输入到控制命令）：根目录控制脚本 + `joyconrobotics/`
- **数据/标定层**（采集、格式、标定、验证）：`RDT/` + `vision/`

## 脚本作用（按交付顺序）

### 1. SDK（先验证“能控能读”）

- `arm_keyboard_control.py`：关节级步进控制；用于快速验证串口、关节方向、软限位与回中位。
- `ik_keyboard_realtime.py`：末端级 IK 控制；用于验证 FK/IK、mask 与角度↔步数映射。

### 2. 人机交互（让控制可录屏）

- `joycon_ik_control_py.py`：JoyCon 遥操作到 IK 控制；重点验证速度档、Home 重置、夹爪开合与退出清理。

### 3. 标定与数据（先标定，后采集，再校验）

- `vision/calibrate_camera.py`：相机内参标定，落盘到 `session_*/`。
- `vision/handeye_calibration_eyeinhand.py` / `vision/handeye_calibration_eyetohand.py`：手眼标定与误差评估。
- `vision/track_blue_circle_eyetohand.py`：闭环验证标定有效性。
- `RDT/collect_rdt_dataset_teleop.py`：采集 raw episode。
- `RDT/build_rdt_hdf5_from_raw.py`：raw→HDF5。
- `RDT/inspect_rdt_hdf5.py`：检查张量形状/抽样可视化。

## 方法作用（理解“脚本是怎么连起来的”）

下面的方法是各脚本最核心的“搭积木接口”，理解它们基本就能读懂上层脚本：

### Driver 层（通信与协议）

- **`driver.ftservo_driver.FTServoDriver.sync_write(ids, address, data_list)`**
  - **作用**：向多个舵机同时写入数据（如目标位置）。
  - **细节**：利用串口总线的广播或同步写指令，确保所有关节在同一时刻接收到指令并开始运动，避免“波浪式”延迟。
- **`driver.ftservo_driver.FTServoDriver.sync_read(ids, address, length)`**
  - **作用**：从多个舵机同步读取数据（如当前位置）。
  - **细节**：发送一次指令，让总线上的舵机按顺序返回数据包，极大提高了读取效率，保证状态采样的同时性。

### Controller 层（控制逻辑）

- **`driver.ftservo_controller.ServoController.fast_move_to_pose(joint_targets, time_s)`**
  - **作用**：输入 `{关节名: 目标步数}` 字典，驱动机械臂运动。
  - **细节**：内部先将关节名映射为 ID，进行软限位（Soft Limit）检查，然后调用 `sync_write` 下发。`time_s` 参数可用于指定运动时间（速度控制）。
- **`driver.ftservo_controller.ServoController.read_servo_positions()`**
  - **作用**：读取当前机械臂状态。
  - **细节**：调用 `sync_read` 获取原始步数，并根据 `homing_offset` 和方向配置，将其转换为以“零位”为基准的步数，返回 `{关节名: 步数}` 字典。

### IK 层（运动学解算）

- **`ik.robot.Robot.fkine(q)`**
  - **作用**：正运动学求解。
  - **细节**：输入关节角 $q$（弧度），通过 ETS（Elementary Transform Sequence）链式乘法，计算出末端执行器相对于基座的齐次变换矩阵 $T$（包含位置与姿态）。
- **`ik.robot.Robot.ikine_LM(Tep, q0, mask, ...)`**
  - **作用**：逆运动学求解（Levenberg-Marquardt 数值法）。
  - **细节**：输入目标位姿 $T_{ep}$ 和初值 $q_0$。算法通过迭代最小化误差 $E = ||T(q) - T_{ep}||^2$ 来寻找最优关节角。`mask` 参数用于指定只关注哪些维度（如只控位置 `[1,1,1,0,0,0]` 或全控 `[1,1,1,1,1,1]`）。
- **`ik.robot.Robot.q_to_servo_targets(q)` / `read_joint_angles()`**
  - **作用**：物理空间（弧度）与驱动空间（步数）的双向映射。
  - **细节**：处理减速比、零位偏置和方向符号，是连接 IK 算法与底层驱动的桥梁。

## 验收视频

- 1.：终端启动日志 + 回中位 + 2~3 个动作 + 读取位姿/关节 + 退出后串口关闭。
- 2.：JoyCon/键盘实时控制 + 夹爪动作 + 漂移时 reset（如 `Home`）+ 退出。
- 3.：采集时三相机预览 + 完成一个 episode 落盘 + raw→hdf5 + inspect 输出。
