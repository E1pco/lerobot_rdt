# 2. 基于自设计 SDK 的遥操作/键盘等人机交互

快速跳转：

- 控制映射速查：`stage1_2_controls.md`
- 理论推导（FK/IK/LM）：`stage1_1_sdk.md`

对应 `decument/task.txt` 的 2. 产出物：

- 源代码：本仓库根目录脚本 + `joyconrobotics/`
- 演示视频：
- 设计/实现手册：本文
- 控制误差分析文档：本文“误差分析”

本章解决的问题是：给定人类输入（键盘/JoyCon），如何在每个控制周期构造目标末端位姿 $T^*$，并通过数值 IK 求解关节角 $q$，最终下发到舵机步数。

统一控制闭环（与 1. 保持一致）：

1) 读取当前关节角 $q\_t$（由步数反解）
2) FK 得到当前末端位姿 $T(q\_t)$
3) 将输入映射为位姿增量，构造 $T^*\_{t+1}$
4) LM-IK：求 $q\_{t+1}$ 使 $T(q\_{t+1})\approx T^*\_{t+1}$
5) 角度→步数，下发并做限位裁剪

相关推导（误差定义、mask、LM 更新）集中在：`stage1_1_sdk.md`。

## 程序设计结构

本章程序结构可以抽象为“输入 → 目标构造 → IK → 下发”的稳定控制环：

1) **输入层**：键盘（`pynput` 或非阻塞 stdin）/ JoyCon（`joyconrobotics/`）输出离散按键或连续姿态/摇杆量
2) **目标构造层**：把输入映射为 $\Delta p$ / $\Delta rpy$，与“基准位姿”合成 $T^*$
3) **求解层**：用 `ik.robot.Robot.ikine_LM` 求 $q$，必要时用 `mask` 只控制子空间
4) **执行层**：`Robot.q_to_servo_targets` 把角度变成步数，`ServoController.limit_position` 截断，`ServoController.fast_move_to_pose` 同步下发
5) **辅助层**：速度档、Home 复位、退出清理与按钮事件处理（夹爪/连发）

## 脚本作用

### 1. 键盘（关节级）：`arm_keyboard_control.py`

该脚本提供最基础的单关节调试功能。它绕过了复杂的运动学解算，直接对每个舵机的目标步数（steps）进行增量控制。
主要用于：验证串口通信、确认关节方向、测试软限位、快速回中。

### 2. 键盘（末端位姿级 IK）：`ik_keyboard_realtime.py`

该脚本实现了基于逆运动学（IK）的笛卡尔空间控制。用户通过键盘控制机械臂末端在 X/Y/Z 轴移动或 Roll/Pitch/Yaw 旋转。
主要用于：验证 IK 求解器表现、检查运动学模型一致性、体验末端控制。

### 3. JoyCon（末端位姿级 IK）：`joycon_ik_control_py.py`

这是本项目最核心的遥操作脚本，用于 3. 阶段的数据集采集。利用 Switch Joy-Con 手柄的摇杆和姿态传感器（IMU）控制机械臂。
主要特点：6-DoF 控制、平滑体验（速度/死区）、所见即所得。

### 4. JoyCon 输入库：`joyconrobotics/`

封装了 HID 通信与原始数据解析，提供面向对象的摇杆/IMU 接口。

## 方法作用

### 关节级控制 (`ArmKeyboardController`)

- **`__init__`**
  - **作用**：初始化控制器。
  - **细节**：初始化 `ServoController`，并读取当前所有舵机的实际位置作为控制起点。若读取失败，则默认使用 `home_pose`。
- **`update_joint(name, delta)`**
  - **作用**：执行单关节运动。
  - **细节**：计算 `new_pos = current_pos + delta`，调用 `controller.limit_position` 进行软限位裁剪，最后组装协议包通过 `sync_write` 下发。
- **`reset_to_home()`**
  - **作用**：回中位。
  - **细节**：调用 `controller.soft_move_to_home` 实现带插值的平滑回中，避免急停急启对机械结构造成冲击。
- **`on_press(key)`**
  - **作用**：按键映射。
  - **细节**：`pynput` 库的回调函数，查表将按键映射为特定关节的步数增量（默认 `step=100`）。

### 实时 IK 控制 (`ik_keyboard_realtime.py` / `joycon_ik_control_py.py`)

- **`main()` / `run()` (主循环)**
  - **作用**：控制系统的核心调度器。
  - **细节**：
    1. **初始化**：连接舵机，回中，读取当前 `q0` 并 FK 得到初始位姿 $T\_{now}$。
    2. **输入处理**：非阻塞读取键盘或 JoyCon 状态（摇杆/IMU）。
    3. **目标构建**：根据输入更新基准位姿 `(x, y, z, r, p, y)`，调用 `build_target_pose` 生成 $T\_{target}$。
    4. **IK 求解**：调用 `robot.ikine_LM(Tep=T_target, q0=q_current, ...)`。
    5. **下发**：`q_to_servo_targets` 转步数 -> 限位 -> 下发。
- **`get_key_nonblock()`**
  - **作用**：键盘无阻塞输入。
  - **细节**：使用 `select` + `tty` (Linux) 修改终端属性，实现按键即读。
- **`JoyConIKController` 类**
  - **作用**：JoyCon 遥操作逻辑封装。
  - **细节**：
    - **状态读取**：`joycon.get_control()` 返回归一化的摇杆推量和 IMU 四元数。
    - **增量映射**：摇杆映射为位置增量 `delta_pos`，IMU 相对姿态映射为 `delta_rpy`。
    - **平滑处理**：引入死区（Deadzone）防止漂移，使用低通滤波平滑输入。
- **`_ButtonHelper` 类**
  - **作用**：按键事件处理。
  - **细节**：支持“按下瞬间（rising edge）”和“长按连发（repeat）”检测，用于精细控制夹爪开合或调整速度档位。

## 2. 键位说明

### 1. `arm_keyboard_control.py`（关节步进）

- `q/a`：`shoulder_pan` +/−
- `w/s`：`shoulder_lift` +/−
- `e/d`：`elbow_flex` +/−
- `r/f`：`wrist_flex` +/−
- `t/g`：`wrist_roll` +/−
- `y/h`：`gripper` +/−
- `m`：回中位
- `ESC`：退出

### 2. `ik_keyboard_realtime.py`（末端 IK）

- `W/S`：$+Z/-Z$
- `A/D`：$-Y/+Y$
- `I/K`：$+X/-X$
- `J/L`：pitch +/−
- `U/O`：yaw +/−
- `+/-`：速度调节
- `Q`：退出

### 3. `joycon_ik_control_py.py`（JoyCon + IK）

- 移动 Joy-Con：控制末端位置/姿态偏移（在“基准位姿”上叠加）
- `ZR`：夹爪收紧一点
- `R`：夹爪松开一点
- `B`：Z 方向手动上移（增大 Z 偏移）
- `Home`：机械臂回中 + 重新连接 JoyCon
- `+/-`：速度调节
- `X`：退出

## 3. 运行与录屏（

### 1. 键盘关节步进

- `python arm_keyboard_control.py`

### 2. 键盘 IK 控制

- `python ik_keyboard_realtime.py`

### 3. JoyCon IK 控制

- 右臂示例：
  - `python joycon_ik_control_py.py --device right --port /dev/right_arm --config ./driver/right_arm.json`
- 左臂示例：
  - `python joycon_ik_control_py.py --device left --port /dev/left_arm --config ./driver/left_arm.json`

录屏建议镜头：

1) 机械臂与操作者同框；
2) 先回中位，再做 2~3 次小幅末端移动；
3) 展示夹爪开合；
4) 最后按 `X` 或 `ESC/Q` 正常退出。

## 4. 控制误差分析

### 4.1 建议定义（最小集合）

- **末端重复定位误差（位置）**：对同一目标位姿 $T^*$ 重复到达 $N$ 次，记录末端位置 $p\_k$，统计
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

其中 IK 的误差定义（angle-axis）与 LM 更新公式见：`stage1_1_sdk.md`。

### 4.1 误差来源

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

```

```
