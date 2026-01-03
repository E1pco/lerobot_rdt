# 2. 控制映射速查（键盘/JoyCon）

## 程序设计结构

映射关系在程序里通常体现为：输入事件 → $\Delta$（步数或位姿增量）→ 目标构造（IK 时）→ 下发。

## 脚本作用

- `arm_keyboard_control.py`：按键 → 关节步数增量（不经 IK）。
- `ik_keyboard_realtime.py`：按键 → 末端位姿增量 → IK → 下发。
- `joycon_ik_control_py.py`：JoyCon 摇杆/IMU → 末端位姿增量 → IK → 下发。

## 方法作用

### 键盘关节级控制 (`ArmKeyboardController`)

- **`on_press(key)`**
  - **作用**：键盘事件回调。
  - **细节**：捕获按键，查表找到对应的关节和方向，调用 `update_joint`。
- **`update_joint(joint_name, delta)`**
  - **作用**：更新单关节目标。
  - **细节**：读取当前值，加上 `delta`（步数增量），做软限位截断，立即下发指令。
- **`reset_to_home()`**
  - **作用**：一键回中。
  - **细节**：调用 `controller.soft_move_to_home()`，以较慢速度平滑回到预设零位。

### 键盘 IK 控制 (`ik_keyboard_realtime.py`)

- **`get_key_nonblock()`**
  - **作用**：非阻塞读取键盘。
  - **细节**：使用 `termios` 和 `sys.stdin` 修改终端属性，实现按键即读，无需回车，保证控制实时性。
- **`build_target_pose(current_pose, keys)`**
  - **作用**：构造目标位姿。
  - **细节**：根据按下的键（如 W/S/A/D），在当前位姿 `(x, y, z, r, p, y)` 上叠加增量 `delta`，返回新的目标齐次变换矩阵 $T^*$。
- **`robot.ikine_LM(Tep, q0, mask)`**
  - **作用**：求解逆运动学。
  - **细节**：以当前关节角 `q0` 为初值，求解目标 $T^*$ 对应的关节角。`mask` 通常设为 `[1,1,1,0,0,0]` (只控位置) 或 `[1,1,1,1,1,0]` (控位置+Pitch/Roll)，忽略 Yaw 以避免 5-DoF 奇异。

### JoyCon IK 控制 (`JoyConIKController`)

- **`JoyconRobotics.get_control()`**
  - **作用**：获取手柄状态。
  - **细节**：返回摇杆推量（归一化 -1~1）和 IMU 姿态（四元数/欧拉角），经过了死区处理和低通滤波。
- **`JoyConIKController.run()`**
  - **作用**：主控制循环。
  - **细节**：以固定频率（如 30Hz）运行，完成“读手柄 -> 算增量 -> 更新目标 -> IK 求解 -> 下发执行”的全过程。
- **`_ButtonHelper.check()`**
  - **作用**：按键状态机。
  - **细节**：区分“按下瞬间”（Rising Edge）和“按住不放”（Holding），用于分别处理触发式动作（如切换模式）和连续动作（如夹爪开合）。

## 1. 键盘关节步进（`arm_keyboard_control.py`）

- `q/a`：`shoulder_pan` +/−
- `w/s`：`shoulder_lift` +/−
- `e/d`：`elbow_flex` +/−
- `r/f`：`wrist_flex` +/−
- `t/g`：`wrist_roll` +/−
- `y/h`：`gripper` +/−
- `m`：回中位
- `ESC`：退出

适用场景：验证通信、限位、关节方向。

## 2. 键盘末端 IK（`ik_keyboard_realtime.py`）

- 平移：`I/K`（X）、`A/D`（Y）、`W/S`（Z）
- 姿态：`J/L`（pitch）、`U/O`（yaw）
- `+/-`：速度调节
- `Q`：退出

适用场景：验证 FK/IK 是否稳定、mask/初值是否合理。

## 3. JoyCon 末端 IK（`joycon_ik_control_py.py`）

- 姿态/位移：移动 Joy-Con（在“基准位姿”上叠加偏移）
- `ZR`：夹爪收紧一点
- `R`：夹爪松开一点
- `B`：Z 手动上移（增加 Z 偏移）
- `Home`：机械臂回中 + JoyCon 重连/校准
- `+/-`：速度调节
- `X`：退出

安全建议：所有动作都从小幅度开始，先把 `speed` 调低，确认方向无误再加速。
