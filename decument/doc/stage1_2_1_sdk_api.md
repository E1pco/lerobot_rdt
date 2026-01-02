# 2.1 API 速查（SDK）

本页给出“写遥操作/采集脚本时最常用”的 SDK 接口速查，详细实现以源码为准。

## 1. `driver.ftservo_controller.ServoController`

位置：`driver/ftservo_controller.py`

### 1.1 初始化

- `ServoController(port: str, baudrate: int, config_path: str)`

配置文件建议使用：
- 左臂：`driver/left_arm.json`
- 右臂：`driver/right_arm.json`

### 1.2 常用控制

- 回中位：`move_all_home()` / `soft_move_to_home(step_count=..., interval=...)`
- 单关节：`move_servo(name: str, position_steps: int, speed: int = ...)`
- 多关节：`fast_move_to_pose(pose: dict[str,int], speed: int = ...)`
- 限位：`limit_position(name: str, position_steps: int) -> int`
- 关闭：`close()`

## 2. `ik.robot`（运动学/IK）

位置：`ik/robot.py`

### 2.1 创建模型

- `create_so101_5dof()`
- `create_so101_5dof_gripper()`

### 2.2 FK/IK

- FK：`robot.fkine(q_rad: np.ndarray) -> np.ndarray(4,4)`
- IK：`robot.ikine_LM(Tep=..., q0=..., tol=..., ilimit=..., mask=..., method=...)`

### 2.3 读关节角与下发目标

- 读当前关节角：
  - `robot.read_joint_angles(joint_names, home_pose, gear_sign, verbose=...) -> q_rad`
- 角度→舵机步数：
  - `robot.q_to_servo_targets(q_rad, joint_names, home_pose, gear_ratio, gear_sign) -> dict[name, steps]`

## 3. 典型控制闭环（最小模板）

1) 读当前关节角 `q0`
2) FK 得到当前末端位姿 `T_now`
3) 构造目标位姿 `T_goal`
4) IK 得到 `q1`
5) 转换为舵机步数 `targets`
6) `limit_position()` 截断
7) `fast_move_to_pose()` 下发
