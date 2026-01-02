# 2.2 基于自设计 SDK 的遥操作/键盘等人机交互

快速跳转：
- 控制映射速查：`stage1_2_2_controls.md`
- 理论推导（FK/IK/LM）：`theory_kinematics_ik.md`

对应 `decument/task.txt` 的 2.2 产出物：
- 源代码：本仓库根目录脚本 + `joyconrobotics/`
- 演示视频：按本文“运行与录屏建议”录制
- 设计/实现手册：本文
- 控制误差分析文档：本文“误差分析”

## 0. 理论背景（像论文一样先定义问题）

本章解决的问题是：给定人类输入（键盘/JoyCon），如何在每个控制周期构造目标末端位姿 $T^*$，并通过数值 IK 求解关节角 $q$，最终下发到舵机步数。

统一控制闭环（与 2.1 保持一致）：
1) 读取当前关节角 $q_t$（由步数反解）
2) FK 得到当前末端位姿 $T(q_t)$
3) 将输入映射为位姿增量，构造 $T^*_{t+1}$
4) LM-IK：求 $q_{t+1}$ 使 $T(q_{t+1})\approx T^*_{t+1}$
5) 角度→步数，下发并做限位裁剪

相关推导（误差定义、mask、LM 更新）集中在：`theory_kinematics_ik.md`。

## 1. 人机交互入口与职责

### 1.1 键盘（关节级）：`arm_keyboard_control.py`

**功能**：
该脚本提供最基础的单关节调试功能。它绕过了复杂的运动学解算，直接对每个舵机的目标步数（steps）进行增量控制。
主要用于：
- 验证串口通信是否正常。
- 确认每个关节的运动方向（`+/-`）是否符合预期。
- 测试机械臂的软限位保护。
- 快速将机械臂复位到“回中位”。

**核心方法与实现**：
- `ArmKeyboardController` 类：
  - `__init__`：初始化 `ServoController`，并读取当前所有舵机的实际位置作为控制起点。若读取失败，则默认使用 `home_pose`。
  - `update_joint(name, delta)`：
    - 计算新步数：`new_pos = current_pos + delta`。
    - 调用 `controller.limit_position` 进行限位裁剪。
    - 组装协议包（0x2A 指令）并通过 `sync_write` 下发。
  - `reset_to_home()`：调用 `controller.soft_move_to_home` 实现带插值的平滑回中。
  - `on_press(key)`：`pynput` 库的回调函数，将按键映射为特定关节的步数增量（默认 `step=100`）。

### 1.2 键盘（末端位姿级 IK）：`ik_keyboard_realtime.py`

**功能**：
该脚本实现了基于逆运动学（IK）的笛卡尔空间控制。用户通过键盘控制机械臂末端在 X/Y/Z 轴移动或 Roll/Pitch/Yaw 旋转，脚本实时解算对应的关节角并驱动舵机。
主要用于：
- 验证 IK 求解器（`ik.solver`）在实机上的表现。
- 检查运动学模型（ETS）与实际机械臂是否一致。
- 体验“所见即所得”的末端控制方式。

**核心方法与实现**：
- `main()` 主循环：
  - **初始化**：连接舵机，先回中，然后读取当前舵机位置并反解为关节角 `q0`，作为 IK 的初始猜测值。
  - **FK 正解**：利用 `ets.fkine(q0)` 计算当前末端的位姿矩阵 $T_{now}$，从中提取出 `(x, y, z, r, p, y)` 作为控制基准。
  - **非阻塞输入**：使用 `select` + `tty` 实现无回车读取键盘输入（`get_key_nonblock`）。
  - **目标构建**：根据按键更新 `(x, y, z, r, p, y)`，调用 `build_target_pose` 生成新的目标矩阵 $T_{target}$。
  - **IK 求解**：调用 `robot.ikine_LM(Tep=T_target, q0=q_current, ...)` 求解目标关节角。
  - **下发控制**：将解算出的弧度转换为步数（`q_to_servo_targets`），并进行限位检查后下发。

### 1.3 JoyCon（末端位姿级 IK）：`joycon_ik_control_py.py`

**功能**：
这是本项目最核心的遥操作脚本，用于 2.3 阶段的数据集采集。它利用 Switch Joy-Con 手柄的摇杆和姿态传感器（IMU）来控制机械臂末端。
主要特点：
- **6-DoF 控制**：结合摇杆（平移）和手柄姿态（旋转），实现灵活的末端控制。
- **平滑体验**：引入了速度调节和死区过滤。
- **所见即所得**：手柄怎么转，机械臂末端就怎么转。

**核心方法与实现**：
- `JoyConIKController` 类：
  - `__init__`：初始化 `ServoController` 和 `JoyconRobotics`（负责 HID 通信与解算）。
  - `run()`：主控制循环。
    - **状态读取**：`joycon.get_control()` 返回摇杆的推量（`stick_l/r`）和手柄的姿态四元数/欧拉角。
    - **增量映射**：
      - 平移：摇杆值 $\times$ 速度系数 $\rightarrow$ `delta_pos`。
      - 旋转：计算手柄当前姿态相对于“基准姿态”的差值 $\rightarrow$ `delta_rpy`。
    - **目标合成**：`target_pos = base_pos + delta_pos`，`target_rpy = base_rpy + delta_rpy`。
    - **IK 与执行**：与键盘 IK 逻辑类似，求解后驱动舵机。
- `_ButtonHelper` 类：
  - 辅助处理按键事件，支持“按下瞬间（rising edge）”和“长按连发（repeat）”检测，用于控制夹爪开合或调整速度。

### 1.4 JoyCon 输入库：`joyconrobotics/`

**功能**：
这是一个独立的 Python 包，封装了对 Joy-Con 设备的底层 HID 通信和数据解析。
主要职责：
- **设备枚举**：自动发现并通过蓝牙/USB 连接 Joy-Con。
- **协议解析**：解析 Joy-Con 的 Input Report（包含按键状态、摇杆模拟量、IMU 6轴数据）。
- **传感器融合**：内置互补滤波或梯度下降算法，将 IMU 的加速度计和陀螺仪数据融合为稳定的姿态（欧拉角/四元数）。

## 2. 键位说明

### 2.1 `arm_keyboard_control.py`（关节步进）

- `q/a`：`shoulder_pan` +/−
- `w/s`：`shoulder_lift` +/−
- `e/d`：`elbow_flex` +/−
- `r/f`：`wrist_flex` +/−
- `t/g`：`wrist_roll` +/−
- `y/h`：`gripper` +/−
- `m`：回中位
- `ESC`：退出

### 2.2 `ik_keyboard_realtime.py`（末端 IK）

- `W/S`：$+Z/-Z$
- `A/D`：$-Y/+Y$
- `I/K`：$+X/-X$
- `J/L`：pitch +/−
- `U/O`：yaw +/−
- `+/-`：速度调节
- `Q`：退出

### 2.3 `joycon_ik_control_py.py`（JoyCon + IK）

- 移动 Joy-Con：控制末端位置/姿态偏移（在“基准位姿”上叠加）
- `ZR`：夹爪收紧一点
- `R`：夹爪松开一点
- `B`：Z 方向手动上移（增大 Z 偏移）
- `Home`：机械臂回中 + 重新连接 JoyCon
- `+/-`：速度调节
- `X`：退出

## 3. 运行与录屏建议（用于 2.2 验证视频）

### 3.1 键盘关节步进

- `python arm_keyboard_control.py`

### 3.2 键盘 IK 控制

- `python ik_keyboard_realtime.py`

### 3.3 JoyCon IK 控制

- 右臂示例：
  - `python joycon_ik_control_py.py --device right --port /dev/right_arm --config ./driver/right_arm.json`
- 左臂示例：
  - `python joycon_ik_control_py.py --device left --port /dev/left_arm --config ./driver/left_arm.json`

录屏建议镜头：
1) 机械臂与操作者同框；
2) 先回中位，再做 2~3 次小幅末端移动；
3) 展示夹爪开合；
4) 最后按 `X` 或 `ESC/Q` 正常退出。

## 4. 控制误差分析（2.2 产出物 4）

本节给出一套“可复现、可量化”的误差分析写法，目标不是追求极限精度，而是：
1) 说明误差来自哪里；2) 给出观测指标；3) 给出你在本仓库实现下的实测结果与改进手段。

### 4.1 建议定义（最小集合）

- **末端重复定位误差（位置）**：对同一目标位姿 $T^*$ 重复到达 $N$ 次，记录末端位置 $p_k$，统计
  $$
  e_p = \sqrt{\frac{1}{N}\sum_{k=1}^N \lVert p_k - \bar{p} \rVert^2}
  $$
- **末端重复定位误差（姿态）**：对同一目标位姿重复到达，记录欧拉角/旋转向量差的均方根（单位 deg）
- **IK 成功率**：控制循环中 `success/total`（失败时记录 `reason`）
- **限位截断次数**：`limit_position()` 触发次数（提示你的目标是否经常越界）

### 4.2 记录方式（推荐你写进实验日志）

- 记录控制周期（tick 频率）、每 tick 的目标增量（XYZ 与 RPY）、`speed` 档位
- 同一动作做 3 组重复实验：低速/中速/高速
- 每组实验至少 30 秒，保证统计稳定

### 4.3 常见结论（写报告时怎么严谨表述）

- 若误差随速度显著增大：更可能是舵机内环/负载/通信抖动引入的动态误差，而非纯运动学模型误差
- 若同一方向总偏：优先怀疑 `homing_offset`、关节方向符号（`gear_sign`）或 DH/ETS 模型与实机不一致
- 若 IK 偶发失败：优先降低每 tick 位姿增量，或调整 `mask`（先只控 xyz，再逐步加入姿态）

## 5. 关键实现链路（从输入到舵机下发）

以 `joycon_ik_control_py.py` 为例，控制链路是：

1) JoyCon 读取姿态/位移偏移：`JoyconRobotics.get_control()`
2) 与“基准位姿”叠加生成目标：
  - `pos = base_pos + joycon_offset_pos`
  - `rpy = base_rpy + joycon_offset_rpy`（该脚本对 roll/pitch 做了符号对齐）
3) 目标位姿矩阵：`build_target_pose()`
4) IK 求解：`robot.ikine_LM(Tep=T_goal, q0=current_q, mask=..., method=...)`
5) 角度→步数：`robot.q_to_servo_targets()`
6) 限位：`ServoController.limit_position()`
7) 同步写：`ServoController.fast_move_to_pose()` → `FTServo.sync_write(0x2A, ...)`

其中 IK 的误差定义（angle-axis）与 LM 更新公式见：`theory_kinematics_ik.md`。

### 4.1 误差来源（从大到小常见排序）

- 机械与舵机：背隙、摩擦、负载变化、舵机内环参数差异。
- 零位/限位标定：`homing_offset` 不准导致 FK/IK 的“零点”错位。
- IK 数值误差：初值偏离、步长过大、`mask` 设置不合理导致收敛到局部最优或失败。
- 传感器与输入：JoyCon 姿态漂移、用户手抖、滤波参数不匹配。
- 控制与通信：串口抖动、控制周期不稳定、速度/加速度过高引发超调。

### 4.2 观测指标与建议记录方式

- 末端重复定位误差：同一目标位姿多次到达后的 XYZ 偏差（mm）与姿态偏差（deg）。
- IK 成功率：连续控制 ticks 中 `success/total`。
- 极限触发次数：`limit_position()` 截断的次数（建议在实验日志中记录）。

### 4.3 常用排查/改进手段

- 降低每 tick 位姿增量、降低 `speed` 上限。
- 先用 `arm_keyboard_control.py` 验证每个关节方向与限位是否正确。
- 确保运行时使用正确的 `driver/*_arm.json`。
- JoyCon 漂移时用 `Home` 重置基准位姿并重新校准。
