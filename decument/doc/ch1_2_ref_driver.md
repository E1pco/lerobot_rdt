# 函数解析：舵机驱动与控制（`driver/`）

本章聚焦 1. SDK 的“硬件通信层 + 控制抽象层”，对应源码：

- `driver/ftservo_driver.py`
- `driver/ftservo_controller.py`

## 程序设计结构

- `FTServo`：协议与串口封装（组包、校验、收发、同步读写）
- `ServoController`：面向关节名的控制抽象（加载配置、软限位、回中、同步下发）

## 脚本作用

- `arm_keyboard_control.py`：直接使用 `ServoController` 做步数级控制与回中。
- `ik_keyboard_realtime.py` / `joycon_ik_control_py.py`：在 IK 之后用 `ServoController` 下发目标并读回当前位置。
- `RDT/collect_rdt_dataset_teleop.py`：采集时读 proprio/下发 action，同样依赖控制层稳定。

## 方法作用

下文按类/方法逐个解释职责、输入输出与常见失败模式。

## 1. `driver.ftservo_driver.FTServo`

定位：串口协议封装，提供「组包→发送→解析应答」能力。

### 1.1 `__init__(port, baudrate, timeout)`

- 作用：打开串口 `serial.Serial(port, baudrate, timeout=...)`
- 失败模式：端口不存在/无权限/波特率不匹配

### 1.2 `build_packet(ID, instruction, params=None)`

- 作用：构造指令包 `HEADER(0xFF,0xFF) + [ID, length, instruction] + params + checksum`
- `length = len(params) + 2`（包含 `instruction` 与 `checksum`）

### 1.3 `checksum(data)`

- 作用：返回按协议定义的校验：

$$
\text{chk}=\sim(\sum data)\ \&\ 0xFF

$$

### 1.4 `send_packet(packet)`

- 作用：向串口写入字节流

### 1.5 `read_response()`

- 作用：读取应答包并返回 dict：
  - `id/length/error/params/checksum/valid`
- 关键点：会校验 checksum（`valid`）
- 失败模式：读取超时、帧头不匹配、长度不足

### 1.6 基本指令（`ping/read_data/write_data/...`）

- `ping(ID)`：发 ping 并读应答
- `read_data(ID, start_addr, length)`：读寄存器片段
- `write_data(ID, start_addr, data_bytes)`：写寄存器片段
- `sync_write(start_addr, data_len, servo_data_dict)`：广播同步写
- `sync_read(start_addr, data_len, ids)`：广播同步读（逐个收应答）

工程上，本项目主要用到的寄存器地址：

- `0x2A`：写入「位置(2B)+时间(2B)+速度(2B)」
- `0x38`：读取当前位置（2B）

## 2. `driver.ftservo_controller.ServoController`

定位：对上提供“按关节名控制”，对下调用 `FTServo`。

### 2.1 初始化与配置加载

- `__init__(port, baudrate, config_path)`：加载 JSON 配置为 `self.config`
- `id_map`：把 `id -> joint_name` 反向映射
- `home_pose`：
  - 非夹爪：取 `(range_min + range_max)//2`
  - 夹爪：取 `range_min`

### 2.2 限位保护：`limit_position(name, target_pos)`

- 输入：关节名、目标步数
- 输出：裁剪到 `[range_min, range_max]` 后的步数
- 副作用：若被裁剪会打印警告

### 2.3 单关节控制：`move_servo(name, target_pos, speed=1000)`

- 组包格式：`[pos_lo,pos_hi, 0,0, speed_lo,speed_hi]`
- 调用：`FTServo.write_data(id, 0x2A, data)`
- 失败模式：串口写失败、舵机无应答、`resp['error']!=0` 或 `valid=False`

### 2.4 多关节同步控制

- `move_group(targets_dict)`：固定 speed=1000 的同步写
- `fast_move_to_pose(target_dict, speed=...)`：支持 int 或 dict 速度

底层都是 `FTServo.sync_write(0x2A, 6, servo_data)`。

### 2.5 回中位与缓动

- `move_to_home(name)`：单关节回中
- `move_all_home()`：全部关节同步回中
- `soft_move_to_home(step_count, interval)`：读当前步数→线性插值→逐步 sync_write
- `soft_move_to_pose(target_dict, step_count, interval)`：同理，但目标由 `target_dict` 指定

### 2.6 读取位置：`read_servo_positions(joint_names=None, verbose=False)`

- 底层：`FTServo.sync_read(0x38, 2, ids)`
- 输出：`{joint_name: position_steps}`

## 3. 最小调用链

- 回中位：`ServoController.move_all_home()`
- 读实际位置：`ServoController.read_servo_positions()`
- 目标步数（来自 IK）：`Robot.q_to_servo_targets()`
- 限位：`ServoController.limit_position()`
- 下发：`ServoController.fast_move_to_pose()`
