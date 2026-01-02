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

## 0. 理论基础与符号约定（论文式开头）

本项目的“控制闭环”把硬件步数与运动学角度用统一公式连接；后续所有控制/IK/采集都依赖这组约定。

- `ik/robot.py::Robot.q_to_servo_targets()`（角度→步数）：
  - $\text{steps}_i = \text{home\_pose}_i + s_i g_i q_i \cdot \frac{4096}{2\pi}$
- `ik/robot.py::Robot.read_joint_angles()`（步数→角度）：
  - $q_i = \dfrac{s_i(\text{steps}_i-\text{home\_pose}_i)}{(4096/(2\pi))\,g_i}$

其中：
- $s_i$ 对应 `gear_sign[name]`（关节方向）
- $g_i$ 对应 `gear_ratio[name]`（减速比，当前默认 1.0）

更完整的 FK/IK（LM）推导与 mask 权重含义见：`theory_kinematics_ik.md`。

## 1. SDK 分层与目录映射

本 SDK 采用分层架构设计，旨在解耦硬件通信、运动控制与运动学解算，便于后续扩展不同型号的机械臂或更换底层通信协议。

### 1.1 硬件通信层（Driver Layer）
- **职责**：负责与底层硬件（如飞特串行总线舵机）进行原始字节流通信。
- **核心文件**：
  - `driver/ftservo_driver.py`：实现了 FTServo 串口通信协议。
    - 封装了 `ping`（心跳）、`read`（读寄存器）、`write`（写寄存器）等基础指令。
    - 实现了 `sync_read`（同步读）和 `sync_write`（同步写），这是实现多关节同步控制的关键。
    - 处理了协议的校验和（Checksum）计算与验证，确保通信可靠性。

### 1.2 控制抽象层（Controller Layer）
- **职责**：屏蔽底层 ID 和寄存器细节，提供面向“关节名称”的高级控制接口。
- **核心文件**：
  - `driver/ftservo_controller.py`：定义了 `ServoController` 类。
    - 维护 `joint_name` 到 `servo_id` 的映射。
    - 实现了软限位保护（Soft Limit），防止机械臂运动超出物理范围。
    - 提供了“回中位（Home）”逻辑，这是所有运动控制的基准。
  - **配置文件**：
    - `driver/servo_config.json`：定义了出厂默认的舵机 ID、零位脉冲值和物理限位。
    - `driver/left_arm.json` / `driver/right_arm.json`：针对具体机械臂（左/右）标定后的配置文件。**在实际运行中，应优先加载这两个文件以获得准确的零位偏置。**

### 1.3 运动学/IK 层（Kinematics Layer）
- **职责**：实现笛卡尔空间（末端位姿）与关节空间（舵机角度）的相互转换。
- **核心文件**：
  - `ik/robot.py`：构建了 SO101 机械臂的运动学模型。
    - 实现了 `q_to_servo_targets`：将物理角度（弧度）映射为舵机脉冲步数。
    - 实现了 `read_joint_angles`：将读取到的脉冲步数反解为物理角度。
    - 封装了正运动学（FK）和逆运动学（IK）的调用接口。
  - `ik/` 目录下的其他文件：
    - `DH.py` / `et.py`：实现了 DH 参数法和 ET（Elementary Transform）序列法建模。
    - `solver.py`：实现了基于 Levenberg-Marquardt (LM) 算法的数值逆运动学求解器。

## 2. 关键配置（必须确认）

在运行任何控制脚本前，请务必确认以下配置与您的硬件环境一致。

### 2.1 串口设备
- **Linux 设备名**：通常为 `/dev/ttyACM0` 或 `/dev/ttyUSB0`。
- **udev 规则**：建议配置 udev 规则将串口映射为固定名称（如 `/dev/left_arm`），避免插拔后设备号变动。
  - 示例规则：`SUBSYSTEM=="tty", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", SYMLINK+="left_arm"`

### 2.2 波特率
- **默认值**：`1000000` (1Mbps)。
- **注意**：所有舵机的波特率必须一致，否则无法通信。

### 2.3 舵机配置文件详解
以 `driver/right_arm.json` 为例，每个关节的配置项如下：

```json
"shoulder_pan": {
    "id": 1,                // 舵机物理 ID
    "homing_offset": 0,     // 零位偏置（单位：步）。用于修正机械装配误差。
                            // 实际零位 = 理论中点 + homing_offset
    "range_min": 0,         // 最小允许步数（软限位）
    "range_max": 4096       // 最大允许步数（软限位）
}
```

**重要提示**：
- `ServoController` 在初始化时会读取此配置。
- `limit_position()` 方法会严格检查目标步数是否在 `[range_min, range_max]` 范围内，超出则会被截断并打印警告。

## 3. SDK 核心 API（对接遥操作/采集的最小集合）

本节列出了在开发上层应用（如遥操作、数据采集）时最常用的 API。

### 3.1 `driver.ftservo_controller.ServoController`

- **初始化**
  ```python
  controller = ServoController(port="/dev/ttyACM0", baudrate=1000000, config_path="driver/left_arm.json")
  ```

- **回中位（复位）**
  ```python
  # 快速回中（慎用，可能产生冲击）
  controller.move_all_home()
  
  # 平滑回中（推荐，带插值）
  # step_count: 插值点数, interval: 每步间隔秒数
  controller.soft_move_to_home(step_count=20, interval=0.05)
  ```

- **多关节同步控制（核心）**
  ```python
  # targets_dict: {关节名: 目标步数}
  # speed: 舵机内部速度参数（0-1000，0为最快）
  targets = {"shoulder_pan": 2048, "shoulder_lift": 2000}
  controller.fast_move_to_pose(targets, speed=0)
  ```
  *原理：底层使用 `sync_write` 指令，确保所有舵机在同一时刻开始运动。*

- **读取当前位置**
  ```python
  # 返回: {关节名: 当前步数}
  positions = controller.read_servo_positions()
  ```

### 3.2 `ik.robot.Robot`

- **创建模型**
  ```python
  from ik.robot import create_so101_5dof
  robot = create_so101_5dof()
  ```

- **正运动学 (FK)**
  ```python
  # q: 关节角数组（弧度）
  # 返回: 4x4 齐次变换矩阵 (numpy array)
  T_end = robot.fkine(q)
  ```

- **逆运动学 (IK)**
  ```python
  # Tep: 目标末端位姿 (4x4 矩阵)
  # q0: 初始猜测角（通常取当前关节角）
  # mask: 6维权重向量 [x, y, z, rx, ry, rz]，1表示控制该维度，0表示忽略
  q_sol, success, reason = robot.ikine_LM(Tep, q0=current_q, mask=[1,1,1,0,0,0])
  ```

- **角度与步数转换**
  ```python
  # 角度转步数（用于下发控制）
  targets = robot.q_to_servo_targets(q_rad, robot.joint_names, controller.home_pose)
  
  # 步数转角度（用于更新状态）
  q_current = robot.read_joint_angles(robot.joint_names, controller.home_pose)
  ```

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
