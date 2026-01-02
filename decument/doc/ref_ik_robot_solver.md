# 函数解析：运动学与 IK（`ik/`）

本章对应 2.1/2.2 的“算法层 SDK”，主要文件：
- `ik/robot.py`
- `ik/solver.py`

## 1. `ik.robot.Robot`

### 1.1 `q_to_servo_targets(q_rad, ..., counts_per_rev=4096, gear_ratio, gear_sign)`

- 作用：关节角（rad）→ 舵机目标步数（steps）
- 公式见：`theory_kinematics_ik.md`
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

- 输入：当前位姿 $T_e$ 与目标位姿 $T_{ep}$
- 输出：6D 误差 $e=[\Delta p,\ \Delta R]$
  - $\Delta p = p^*-p$
  - $\Delta R$ 用 $R_{err}=R^*R^T$ 转为轴角向量

### 2.2 代价函数与权重

- `mask` → $W_e=\mathrm{diag}(mask)$
- 代价：$E=\frac12 e^T W_e e$

### 2.3 LM 更新

- 雅可比：`J = ets.jacob0(q)`
- 阻尼：`Wn` 随 `method` 不同而不同（chan/wampler/sugihara）
- 更新：$q \leftarrow q + (J^T W_e J + W_n)^{-1} J^T W_e e$

### 2.4 多次搜索（slimit）

- 第 1 次用 `q0`
- 后续随机初始化重试

失败模式：
- 矩阵奇异（`LinAlgError`）导致中断当前搜索
- 迭代耗尽：返回 `IKResult(False, q, 'iteration limit reached')`

## 3. SO101 模型构建函数

- `create_so101_5dof()`：5DOF ETS + `gear_sign/gear_ratio/qlim`
- `create_so101_5dof_gripper()`：在 5DOF 末端后额外增加固定平移段（工具长度）

工程建议：
- 控制脚本优先使用 `create_so101_5dof_gripper()`（末端位置更贴近实际夹爪）。
