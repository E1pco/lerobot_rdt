# 2.2 控制映射速查（键盘/JoyCon）

本页把 2.2 中涉及的控制映射做成“随手查”的表格/清单，便于录屏与写误差分析时引用。

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
