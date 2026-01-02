# 理论推导：舵机步数↔关节角、FK/IK（LM）

本章为 2.1/2.2 提供必要的理论推导，对应实现主要来自：
- 舵机控制：`driver/ftservo_controller.py`
- 运动学与 IK：`ik/robot.py`、`ik/solver.py`、`ik/et.py`

## 1. 舵机步数 ↔ 关节角（弧度）

### 1.1 编码器计数

项目默认每转编码器计数：

$$
\text{counts\_per\_rev}=4096
$$

因此每弧度计数：

$$
\text{counts\_per\_rad}=\frac{4096}{2\pi}
$$

### 1.2 home_pose（零位参考）

在 `ServoController` 中，`home_pose` 默认取舵机允许范围的中点（夹爪例外取 `range_min`）：

$$
\text{home\_pose} = \left\lfloor\frac{\text{range\_min}+\text{range\_max}}{2}\right\rfloor
$$

**注意**：仓库里 JSON 里存在 `homing_offset`，但当前 `ServoController.home_pose` 的计算**不使用** `homing_offset`（这是工程选择，便于把“零位参考”和“机械装配标定偏置”分开看）。

### 1.3 关节角→目标步数（前向映射）

`ik/robot.py::Robot.q_to_servo_targets()` 的核心公式：

设第 $i$ 个关节：
- 关节角 $q_i$（rad）
- 方向符号 $s_i\in\{+1,-1\}$（`gear_sign`）
- 传动比 $g_i$（`gear_ratio`）

则目标步数：

$$
\text{steps}_i = \text{home\_pose}_i + s_i\,g_i\,q_i\,\text{counts\_per\_rad}
$$

### 1.4 步数→关节角（反向映射）

`ik/robot.py::Robot.read_joint_angles()` 的核心公式：

$$
q_i = \frac{s_i\,(\text{steps}_i-\text{home\_pose}_i)}{\text{counts\_per\_rad}\,g_i}
$$

这也是 2.2/2.3 里把“舵机实际位置”写入 proprio 的基础。

## 2. 正运动学（FK）：ETS 链

本仓库 SO101 的 FK 由 Elementary Transform Sequence（ETS）链表达：

$$
T(q)=\prod_k T_k(q)
$$

在 `ik/robot.py::create_so101_5dof(_gripper)` 中，链由 `ET.tx/ET.tz/ET.Rx/ET.Ry/ET.Rz` 组成。

- 平移：$T_x(a)$、$T_z(d)$
- 旋转：$R_x(\theta)$、$R_y(\theta)$、$R_z(\theta)$

实现入口：
- `Robot.fkine(q)` → `ets.fkine(q)` 返回 $4\times4$ 齐次矩阵

## 3. 逆运动学（IK）：LM 数值法（本仓库实现）

IK 目标：给定期望末端位姿 $T^\*\,(=T_{ep})$，求关节角 $q$ 使 $T(q)\approx T^\*$。

### 3.1 位姿误差：角轴（angle-axis）形式

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

### 3.2 mask（权重）与代价函数

`mask=[w_x,w_y,w_z,w_{roll},w_{pitch},w_{yaw}]`，在代码中构成对角权重矩阵：

$$
W_e=\mathrm{diag}(\text{mask})
$$

代价函数：

$$
E(q)=\frac{1}{2}e(q)^T W_e\,e(q)
$$

直观理解：mask 为 0 的维度在优化中被忽略；介于 0~1 则是“弱约束”。

### 3.3 LM 更新公式（实现等价）

`ikine_LM()` 核心更新（代码注释同款）：

$$
q \leftarrow q + (J^T W_e J + W_n)^{-1}J^T W_e\,e
$$

其中 $J$ 为雅可比：`ets.jacob0(q)`。

阻尼矩阵 $W_n$ 在本仓库支持三种风格：
- `chan`：$W_n=k\,E\,I$
- `wampler`：$W_n=k\,I$
- `sugihara`：$W_n=(E+k)I$

### 3.4 多次搜索（slimit）

`slimit` 表示允许多次随机初值重试：
- 第 1 次使用 `q0`
- 后续使用 $[-\pi,\pi]$ 的随机初始化

这样能降低陷入局部最优的概率。

## 4. 实战建议（和文档 2.1/2.2 对应）

- 首次验证先用 `arm_keyboard_control.py` 确认“关节方向 + 限位 + 回中”正确，再上 IK。
- IK 控制每 tick 的位姿增量要小（尤其 yaw/pitch/roll）。
- `mask` 建议从“只控位置”开始：`[1,1,1,0,0,0]`，稳定后再放开姿态。
