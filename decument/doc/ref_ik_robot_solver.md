# 函数解析：运动学与 IK（`ik/`）

本章对应 1./2. 的“算法层 SDK”，主要文件：

- `ik/robot.py`
- `ik/solver.py`

运动学/IK 层要解决的是：

- **正运动学**：给定关节角 $q$，计算末端位姿 $T(q)$
- **逆运动学**：给定目标末端位姿 $T^*$，求一个关节角解 $q$ 使 $T(q)\approx T^*$
- **工程闭环**：把 $q$ 与舵机步数互相转换，保证 IK 输出能真正驱动硬件并能从硬件读回更新状态

## 程序设计结构

- `ik.robot.Robot`：面向工程使用的封装（FK/IK 调用 + 角度↔步数转换）
- `ik.solver`：数值法 IK 求解器（LM/GN/NR/QP 等）
- SO101 模型构建：`create_so101_*` 把 ETS 参数、限位、符号约定组合成可用模型

## 脚本作用

- `ik_keyboard_realtime.py`：末端 IK 控制的最小验证脚本。
- `joycon_ik_control_py.py`：遥操作脚本（需要 IK 在实时循环内收敛）。
- `RDT/collect_rdt_dataset_teleop.py`：采集时会持续读回关节角/末端状态写入 unified vector。

## 方法作用

下文按类/函数解释各方法在“读状态→构造目标→求解→下发”的链路中承担的职责。

## 1. `ik.robot.Robot`

### 1.1 `q_to_servo_targets(q_rad, ..., counts_per_rev=4096, gear_ratio, gear_sign)`

- 作用：关节角（rad）→ 舵机目标步数（steps）
- 公式见：`stage1_1_sdk.md`
- 关键依赖：`home_pose`（若未传入则从 `ServoController.home_pose` 取）

常见错误：

- 未传 `home_pose` 且 `ServoController` 未能初始化 → 抛 `ValueError`

### 1.2 `read_joint_angles(joint_names=None, home_pose=None, gear_sign=None, gear_ratio=None)`

- 作用：读舵机当前步数 → 反解关节角（rad）
- 底层依赖：`ServoController.read_servo_positions()`

### 1.3 `fkine(q)`

- 作用：FK，返回 $4\times4$ 齐次矩阵
- 实现：`return self.ets.fkine(q)`

### 1.4 `ikine_LM(Tep, q0=None, ..., mask=None, k=..., method='chan')`

- 作用：LM 数值 IK
- 真实实现：调用 `ik/solver.py::ikine_LM(self.ets, ...)`
- 关键参数：
  - `q0` 初值（强烈建议传入当前关节角）
  - `mask`（位置/姿态权重）
  - `ilimit/slimit/tol`（迭代/多次搜索/收敛阈值）

同类接口：`ikine_GN/ikine_NR/ikine_QP`。

## 2. `ik.solver.ikine_LM`（核心求解器）

### 2.1 `_angle_axis(Te, Tep)`

- 输入：当前位姿 $T\_e$ 与目标位姿 $T\_{ep}$
- 输出：6D 误差 $e=[\Delta p,\ \Delta R]$
  - $\Delta p = p^*-p$
  - $\Delta R$ 用 $R\_{err}=R^*R^T$ 转为轴角向量

### 2.2 代价函数与权重

- `mask` → $W\_e=\mathrm{diag}(mask)$
- 代价：$E=\frac12 e^T W\_e e$

### 2.3 LM 更新

- 雅可比：`J = ets.jacob0(q)`
- 阻尼：`Wn` 随 `method` 不同而不同（chan/wampler/sugihara）
- 更新：$q \leftarrow q + (J^T W\_e J + W\_n)^{-1} J^T W\_e e$

### 2.4 多次搜索（slimit）

- 第 1 次用 `q0`
- 后续随机初始化重试

失败模式：

- 矩阵奇异（`LinAlgError`）导致中断当前搜索
- 迭代耗尽：返回 `IKResult(False, q, 'iteration limit reached')`

## 3. SO101 模型构建函数

- `create_so101_5dof()`：5DOF ETS + `gear_sign/gear_ratio/qlim`
- `create_so101_5dof_gripper()`：在 5DOF 末端后额外增加固定平移段（工具长度）
