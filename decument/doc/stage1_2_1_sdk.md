# 2.1 基于 Python 的自设计 lerobot 控制 SDK

快速跳转：
- API 速查：`stage1_2_1_sdk_api.md`
- 理论推导：`theory_kinematics_ik.md`
- 函数解析（driver）：`ref_sdk_driver.md`
- 函数解析（ik）：`ref_ik_robot_solver.md`

对应 `decument/task.txt` 的 2.1 产出物：
- 源代码：本仓库 `driver/` + `ik/`（核心 SDK）
- 演示视频：按本文“运行验证”录制
- 设计/实现手册：本文

## 1. SDK 分层与目录映射

- 硬件通信层（串口协议）
  - `driver/ftservo_driver.py`：FTServo 协议封装（ping/read/write/sync_read/sync_write）
- 控制抽象层（面向关节/夹爪）
  - `driver/ftservo_controller.py`：`ServoController`（回中位、限位、单关节/多关节下发、读位置等）
  - `driver/servo_config.json`：默认舵机 ID/零位/限位
  - `driver/left_arm.json`、`driver/right_arm.json`：左右臂标定后的偏置与限位（建议采集/运行时使用各自 json）
- 运动学/IK 层
  - `ik/robot.py`：SO101 机械臂模型构建（FK/IK、角度↔舵机步数映射、读关节角等）
  - `ik/` 目录：DH/ET/求解器等实现细节

## 2. 关键配置（必须确认）

- 串口设备：示例使用 `/dev/left_arm`、`/dev/right_arm`（也可直接用 `/dev/ttyACM0` 等实际端口）
- 波特率：默认 `1000000`
- 舵机配置文件：
  - 单臂控制建议：`driver/left_arm.json` / `driver/right_arm.json`
  - 标定/通用可用：`driver/servo_config.json`

JSON 字段含义（以 `driver/right_arm.json` 为例）：
- `id`：舵机 ID
- `homing_offset`：回中位偏置（影响 `home_pose`）
- `range_min/range_max`：限位（`ServoController.limit_position()` 使用）

## 3. SDK 核心 API（对接遥操作/采集的最小集合）

## 3+. 理论推导（建议写入 2.1 设计文档/答辩）

本项目的“控制闭环”把硬件步数与运动学角度用统一公式连接：

- `ik/robot.py::Robot.q_to_servo_targets()`（角度→步数）：
  - $\text{steps}_i = \text{home\_pose}_i + s_i g_i q_i \cdot \frac{4096}{2\pi}$
- `ik/robot.py::Robot.read_joint_angles()`（步数→角度）：
  - $q_i = \dfrac{s_i(\text{steps}_i-\text{home\_pose}_i)}{(4096/(2\pi))\,g_i}$

其中：
- $s_i$ 对应 `gear_sign[name]`（关节方向）
- $g_i$ 对应 `gear_ratio[name]`（减速比，当前默认 1.0）

更完整的 FK/IK（LM）推导与 mask 权重含义见：`theory_kinematics_ik.md`。

### 3.1 `driver.ftservo_controller.ServoController`

常用能力（以代码为准）：
- 初始化：`ServoController(port, baudrate, config_path=...)`
- 回中位：`move_all_home()` / `soft_move_to_home(...)`
- 单关节下发：`move_servo(name, position, speed=...)`
- 多关节下发：`fast_move_to_pose({name: position, ...}, speed=...)`
- 读位置：`read_position(name)` 或底层 `sync_read` 批量读取
- 限位保护：`limit_position(name, position)`
- 释放资源：`close()`

### 3.2 `ik.robot`（运动学/IK）

- 创建模型：
  - `create_so101_5dof()` / `create_so101_5dof_gripper()`
- FK：`robot.fkine(q)` → $4\times4$ 齐次矩阵
- IK：`robot.ikine_LM(Tep=..., q0=..., ...)`
- 角度→舵机步数：`robot.q_to_servo_targets(q, joint_names, home_pose, ...)`
- 读取关节角：`robot.read_joint_angles(joint_names, home_pose, gear_sign, ...)`

## 4. 运行验证（用于 2.1 演示视频）

### 4.1 直接关节步进（不依赖 IK）

- 运行：
  - `python arm_keyboard_control.py`
- 行为：按键直接给各舵机加/减步数；支持回中位。

### 4.2 实时 IK 控制（末端位姿控制）

- 运行：
  - `python ik_keyboard_realtime.py`
- 行为：读取当前关节角作为起点，实时 IK 控制末端 `pos/rpy`。

建议录制内容（验收友好）：
1) 启动脚本并完成回中位/同步当前位置
2) 在安全空间内做 2~3 个方向的小幅位移
3) 退出并正常关闭串口

## 5. 常见问题

- 串口权限：若报 `Permission denied`，需给当前用户 `dialout` 权限或调整 udev 规则。
- 舵机乱动/超限：优先检查 `driver/*_arm.json` 的 `homing_offset` 与 `range_min/range_max`。
- IK 不收敛：减小每步位姿增量、放宽/调整 `mask`，并确保初值 `q0` 为当前实际关节角。
