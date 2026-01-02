# 2.2 基于自设计 SDK 的遥操作/键盘等人机交互

快速跳转：
- 控制映射速查：`stage1_2_2_controls.md`
- 理论推导（FK/IK/LM）：`theory_kinematics_ik.md`

对应 `decument/task.txt` 的 2.2 产出物：
- 源代码：本仓库根目录脚本 + `joyconrobotics/`
- 演示视频：按本文“运行与录屏建议”录制
- 设计/实现手册：本文
- 控制误差分析文档：本文“误差分析”

## 1. 人机交互入口与职责

- 键盘（关节级）：`arm_keyboard_control.py`
  - 直接对舵机步数做增量控制，适合验证通讯/限位/回中位。
- 键盘（末端位姿级 IK）：`ik_keyboard_realtime.py`
  - 读取当前关节角作为初值，实时 IK 控制末端 `pos/rpy`。
- JoyCon（末端位姿级 IK，单臂演示）：`joycon_ik_control_py.py`
  - JoyCon 姿态→末端位姿偏移→IK→下发舵机。
- JoyCon（输入库）：`joyconrobotics/`
  - 负责 JoyCon 枚举/传感器融合/按键事件。

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
