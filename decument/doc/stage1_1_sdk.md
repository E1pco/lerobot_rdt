# 1. 基于 Python 的自设计 lerobot 控制 SDK

快速跳转：

- API 速查：`stage1_1_sdk_api.md`
- 函数解析（driver）：`ref_sdk_driver.md`
- 函数解析（ik）：`ref_ik_robot_solver.md`

产出物：

- 源代码：本仓库 `driver/` + `ik/`（核心 SDK）
- 演示视频
- 设计/实现手册：本文

## 理论框架

### 1. 背景与问题定义

**逆运动学（Inverse Kinematics, IK）** 是机器人控制中的核心问题，其数学本质是求解非线性映射的逆。

设机器人的正运动学方程为 $x = f(q)$，其中：

- $q \in \mathbb{R}^n$ 为关节空间变量（关节角）。
- $x \in \mathbb{R}^m$ 为任务空间变量（末端位姿，通常 $m=6$，包含位置和姿态）。

IK 的目标是：给定期望的末端位姿 $x_{des}$，寻找一组关节角 $q$，使得 $f(q) = x_{des}$。

**挑战**：

1. **非线性**：$f(q)$ 包含大量的三角函数，是非线性的。
2. **多解性**：同一个末端位姿可能对应多组关节角（如“肘部朝上”和“肘部朝下”）。
3. **奇异性**：在某些特定构型下，雅可比矩阵秩下降，导致运动能力丧失或数值求解发散。
4. **冗余与欠驱动**：当 $n > m$ 时（冗余），存在无穷多解。当 $n < m$ 时（欠驱动，如本仓库的 5-DoF 机械臂），通常无法精确达到任意 6D 位姿，需要引入权重（Mask）忽略某些维度（如 Yaw 角）。

### 2. 逆运动学方法谱系：解析法 vs 数值法

解决 IK 问题主要有两大类方法：

#### 1.1 解析法（Analytic Methods）

解析法试图通过代数或几何推导，直接得到 $q$ 关于 $x$ 的闭式数学表达式（Closed-form solution）。

- **几何法**：利用机械臂的几何结构（连杆长度、关节偏移），通过余弦定理、三角恒等式构建几何约束方程，逐步解出各个关节角。*优点*：计算速度极快（微秒级），可获得所有可能的解。*缺点*：严重依赖特定构型，推导过程复杂，通用性差。
- **代数法**：将 FK 矩阵方程中的元素提取出来，通过变量代换消元，将超越方程转化为多项式方程求解。

#### 1.2 数值法（Numerical Methods）

数值法将 IK 视为一个非线性优化问题，从一个初始猜测值 $q_0$ 开始，通过迭代更新 $q$ 来最小化末端误差 $e(q) = x_{des} - f(q)$。*优点*：通用性强，适用于任意构型的串联机械臂；易于处理冗余自由度和避障约束。*缺点*：计算量相对较大，可能陷入局部极小值，收敛性受初值影响。

**本 SDK 采用数值法**，因为它能统一处理不同构型的机械臂，且易于集成 Mask 权重功能。

### 3. 数值求解算法详解

数值法的核心在于迭代更新律：$q_{k+1} = q_k + \Delta q$。不同的算法主要区别在于 $\Delta q$ 的计算方式。

基于一阶泰勒展开，末端误差与关节增量的关系为：$e \approx J \Delta q$。

#### 3.1 牛顿-拉夫逊法 (Newton-Raphson)

牛顿法试图一步到位消除线性化后的误差。

$$
\Delta q = J^{-1} e

$$

若 $J$ 不可逆（非方阵或奇异），则使用伪逆 $J^{\dagger}$：

$$
\Delta q = J^{\dagger} e = (J^T J)^{-1} J^T e

$$

- **特点**：收敛速度快（二次收敛），但在奇异点附近（$J^T J$ 接近奇异）极其不稳定，会导致 $\Delta q$ 剧烈震荡。

#### 3.2 梯度下降法 (Gradient Descent)

梯度下降法沿着误差函数的负梯度方向更新，不涉及矩阵求逆。

$$
\Delta q = \alpha J^T e

$$

- **特点**：计算简单，绝对稳定（不会发散），但收敛速度慢（线性收敛），容易在平坦区域“之”字形震荡。

#### 3.3 Levenberg-Marquardt (LM) 算法

LM 算法是牛顿法与梯度下降法的结合体（Damped Least Squares），是工程中最常用的 IK 算法。
它引入了阻尼因子 $\lambda$：

$$
\Delta q = (J^T J + \lambda^2 I)^{-1} J^T e

$$

- **机制**：
  - 当误差较大或接近奇异时，增大 $\lambda$，算法行为接近梯度下降（保证稳定性）。
  - 当误差较小且远离奇异时，减小 $\lambda$，算法行为接近牛顿法（加速收敛）。
- **本仓库实现**：`ik/solver.py` 中实现了标准的 LM 求解器，并支持 `mask` 权重矩阵 $W_e$，实际求解方程为：
  $$
  (J^T W_e J + \lambda^2 I) \Delta q = J^T W_e e

  $$

### 4. 雅可比矩阵 (Jacobian Matrix)

雅可比矩阵 $J(q) \in \mathbb{R}^{m \times n}$ 是数值法的核心，描述了关节速度 $\dot{q}$ 到末端速度 $\dot{x}$ 的线性映射：$\dot{x} = J(q) \dot{q}$。

#### 4.1 数值雅可比 (Numerical Jacobian)

通过有限差分法近似计算偏导数。对于第 $j$ 列 $J_j$：

$$
J_j \approx \frac{f(q + \delta \cdot u_j) - f(q - \delta \cdot u_j)}{2\delta}

$$

其中 $u_j$ 是第 $j$ 个分量为 1 的单位向量，$\delta$ 是微小步长。

- **优点**：实现极其简单，无需推导公式，只要有 FK 函数即可。
- **缺点**：计算量大（计算一列需要运行 FK），精度受步长 $\delta$ 影响（截断误差 vs 舍入误差）。

#### 4.2 解析/几何雅可比 (Analytic/Geometric Jacobian)

基于刚体运动学公式，直接利用连杆的几何信息（旋转轴 $z_i$ 和位置 $p_i$）构造矩阵。
对于旋转关节，第 $i$ 列通常形式为：

$$
J_i = \begin{bmatrix}
z_{i-1} \times (p_e - p_{i-1}) \\
z_{i-1}
\end{bmatrix}

$$

- $z_{i-1}$：第 $i$ 关节轴在基座坐标系下的单位向量。
- $p_e$：末端执行器原点位置。
- $p_{i-1}$：第 $i$ 关节原点位置。
- **优点**：计算精确，效率高（仅涉及向量运算），无数值误差。
- **缺点**：需要准确的运动学模型参数，推导相对繁琐。**本 SDK 优先使用解析法计算雅可比以保证实时性。**

### 5. 舵机步数 ↔ 关节角（弧度）

#### 5.1 编码器计数

项目默认每转编码器计数：

$$
\text{counts\_per\_rev}=4096

$$

因此每弧度计数：

$$
\text{counts\_per\_rad}=\frac{4096}{2\pi}

$$

#### 5.2 home_pose（零位参考）

在 `ServoController` 中，`home_pose` 默认取舵机允许范围的中点（夹爪例外取 `range_min`）：

$$
\text{home\_pose} = \left\lfloor\frac{\text{range\_min}+\text{range\_max}}{2}\right\rfloor

$$

**注意**：仓库里 JSON 里存在 `homing_offset`，但当前 `ServoController.home_pose` 的计算**不使用** `homing_offset`（这是工程选择，便于把“零位参考”和“机械装配标定偏置”分开看）。

#### 5.3 关节角→目标步数（前向映射）

`ik/robot.py::Robot.q_to_servo_targets()` 的核心公式：

设第 $i$ 个关节：

- 关节角 $q_i$（rad）
- 方向符号 $s_i\in\{+1,-1\}$（`gear_sign`）
- 传动比 $g_i$（`gear_ratio`）

则目标步数：

$$
\text{steps}_i = \text{home\_pose}_i + s_i\,g_i\,q_i\,\text{counts\_per\_rad}

$$

#### 5.4 步数→关节角（反向映射）

`ik/robot.py::Robot.read_joint_angles()` 的核心公式：

$$
q_i = \frac{s_i\,(\text{steps}_i-\text{home\_pose}_i)}{\text{counts\_per\_rad}\,g_i}

$$

这也是 2./3. 里把“舵机实际位置”写入 proprio 的基础。

### 6. 正运动学（FK）：ETS 链

本仓库 SO101 的 FK 由 Elementary Transform Sequence（ETS）链表达：

$$
T(q)=\prod_k T_k(q)

$$

在 `ik/robot.py::create_so101_5dof(_gripper)` 中，链由 `ET.tx/ET.tz/ET.Rx/ET.Ry/ET.Rz` 组成。

- 平移：$T_x(a)$、$T_z(d)$
- 旋转：$R_x(\theta)$、$R_y(\theta)$、$R_z(\theta)$

实现入口：

- `Robot.fkine(q)` → `ets.fkine(q)` 返回 $4\times4$ 齐次矩阵

### 7. 逆运动学（IK）：LM 数值法（本仓库实现）

IK 目标：给定期望末端位姿 $T^\*\,(=T_{ep})$，求关节角 $q$ 使 $T(q)\approx T^\*$。

#### 7.1 位姿误差：角轴（angle-axis）形式

`ik/solver.py::_angle_axis(Te, Tep)` 计算 6 维误差：

$$
e = \begin{bmatrix}e_p\\ e_R\end{bmatrix}\in\mathbb{R}^6

$$

- 位置误差：

$$
e_p = p^* - p

$$

- 旋转误差：

$$
R_{err}=R^*R^T

$$

并由 $R_{err}$ 转为轴角向量（代码通过反对称项与迹计算）。

#### 7.2 mask（权重）与代价函数

`mask=[w_x,w_y,w_z,w_{roll},w_{pitch},w_{yaw}]`，在代码中构成对角权重矩阵：

$$
W_e=\mathrm{diag}(\text{mask})

$$

代价函数：

$$
E(q)=\frac{1}{2}e(q)^T W_e\,e(q)

$$

直观理解：mask 为 0 的维度在优化中被忽略；介于 0~1 则是“弱约束”。

#### 7.3 LM 更新公式（实现等价）

`ikine_LM()` 核心更新：

$$
q \leftarrow q + (J^T W_e J + W_n)^{-1}J^T W_e\,e

$$

其中 $J$ 为雅可比：`ets.jacob0(q)`。

阻尼矩阵 $W_n$ 在本仓库支持三种风格：

- `chan`：$W_n=k\,E\,I$
- `wampler`：$W_n=k\,I$
- `sugihara`：$W_n=(E+k)I$

#### 7.4 多次搜索（slimit）

`slimit` 表示允许多次随机初值重试：

- 第 1 次使用 `q0`
- 后续使用 $[-\pi,\pi]$ 的随机初始化

这样能降低陷入局部最优的概率。

## 程序设计结构

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

## 脚本作用

1. 的验收通常不直接看 `driver/` 或 `ik/` 的单元函数，而是用两个最短路径脚本把分层串起来：

- `arm_keyboard_control.py`：

  - 作用：**关节级**步进控制（直接对目标步数做增量），用于验证串口通信、关节方向、软限位与回中位。
  - 适用：第一次接硬件时优先跑它，避免 IK 不收敛等高层问题干扰排错。
- `ik_keyboard_realtime.py`：

  - 作用：**末端级**实时 IK 控制（目标位姿 → IK → 关节角 → 步数下发），用于验证 FK/IK、mask 配置与角度↔步数映射。
  - 适用：在关节级控制稳定后，用它验证“末端位姿控制”链路。

## 方法作用

### Driver 层（通信/协议）

- **`FTServoDriver.ping(id)` / `read(id, addr, len)` / `write(id, addr, val)`**
  - **作用**：基础通信指令。
  - **细节**：直接操作串口发送符合飞特协议的数据包。`ping` 用于检测舵机在线状态；`read/write` 用于读写单个寄存器（如 PID 参数、当前位置）。
- **`FTServoDriver.sync_read(ids, addr, len)`**
  - **作用**：同步读取。
  - **细节**：发送一条指令让总线上指定 ID 的舵机依次返回数据。这是实现高频（如 50Hz+）状态更新的关键，避免了轮询带来的累积延迟。
- **`FTServoDriver.sync_write(ids, addr, data_list)`**
  - **作用**：同步写入。
  - **细节**：广播一条指令，让所有舵机在收到指令的瞬间同时更新寄存器（如目标位置）。保证了多关节运动的协调性。
- **校验和计算 (`_calc_checksum`)**
  - **作用**：数据完整性验证。
  - **细节**：协议规定的 Checksum 算法。在读取数据时，若计算值与接收值不符，驱动层会抛出异常或重试，防止错误数据进入控制环。

### Controller 层（关节名→舵机 ID 与保护逻辑）

- **`ServoController.__init__(config_path)`**
  - **作用**：初始化与配置加载。
  - **细节**：加载 JSON 配置文件，建立 `joint_name` -> `servo_id` 的映射表，并读取 `homing_offset` 和软限位范围。
- **`ServoController.soft_move_to_home(time_s=2.0)`**
  - **作用**：平滑回中。
  - **细节**：规划一条从当前位置到零位的插值轨迹，在 `time_s` 秒内执行完毕。相比直接下发零位，这能有效避免机械臂剧烈抖动。
- **`ServoController.fast_move_to_pose(joint_targets)`**
  - **作用**：快速下发目标。
  - **细节**：接收 `{关节名: 目标步数}`，内部进行 ID 转换和软限位检查，然后调用 `sync_write` 立即执行。
- **`ServoController.read_servo_positions()`**
  - **作用**：获取当前状态。
  - **细节**：调用 `sync_read` 读取原始步数，减去 `homing_offset` 并根据方向配置处理符号，返回物理意义上的关节步数。
- **`ServoController.limit_position(name, pos)`**
  - **作用**：软限位保护。
  - **细节**：检查目标步数是否在 `[range_min, range_max]` 范围内。若越界，则强制截断到边界值，并打印警告。

### IK 层（角度↔位姿↔步数）

- **`Robot.fkine(q)`**
  - **作用**：正运动学。
  - **细节**：输入关节角 $q$（弧度），通过连乘各连杆的变换矩阵，计算末端坐标系相对于基座的位姿 $T$。
- **`Robot.ikine_LM(Tep, q0, mask, ...)`**
  - **作用**：逆运动学（LM 算法）。
  - **细节**：求解 $\min_q ||T(q) - T_{ep}||^2$。利用雅可比矩阵 $J$ 迭代更新 $q$。`mask` 向量（如 `[1,1,1,1,1,0]`）用于在误差计算中屏蔽掉不需要控制的自由度（如 5-DoF 机械臂无法独立控制 Yaw）。
- **`Robot.read_joint_angles()`**
  - **作用**：读取物理角度。
  - **细节**：先调用 `controller.read_servo_positions` 得到步数，再利用 `(steps - zero) / steps_per_rad` 公式转换为弧度。
- **`Robot.q_to_servo_targets(q)`**
  - **作用**：角度转步数。
  - **细节**：IK 输出的是理论弧度，该方法将其转换为舵机能识别的脉冲步数，是下发控制前的最后一步变换。

## 关键配置

### 1. 串口设备

- **Linux 设备名**：通常为 `/dev/ttyACM0` 。
- **udev 规则**：建议配置 udev 规则将串口映射为固定名称（如 `/dev/left_arm`），避免插拔后设备号变动。
  - 示例规则：`SUBSYSTEM=="tty", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="xxxx", SYMLINK+="left_arm"`

### 2. 波特率

- **默认值**：`1000000` (1Mbps)。
- **注意**：所有舵机的波特率必须一致，否则无法通信。

### 3. 舵机配置文件详解

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

## 4. 运行验证

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
