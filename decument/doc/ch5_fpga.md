# 5. FPGA 逆解加速

本章介绍如何利用 FPGA 硬件加速 Levenberg-Marquardt 逆运动学求解器，实现相比纯软件执行约 **10倍** 的单步迭代性能提升。

## 理论背景

### 为什么需要硬件加速

逆运动学（IK）求解是机械臂控制的核心环节，决定了末端执行器从当前位姿运动到目标位姿时各关节应旋转的角度。在实时控制场景（如遥操作、轨迹跟踪）中，IK 求解需要在毫秒级完成，而纯软件实现的迭代式 LM 算法在资源受限的嵌入式平台（如 PYNQ-Z2 的 ARM 处理器）上往往无法满足性能需求。

FPGA 通过其高度并行的硬件架构，可以将 LM 算法中的矩阵运算（雅可比计算、Cholesky 分解、线性方程求解等）映射为流水线电路，显著降低单步迭代延迟。本项目在 PYNQ-Z2 板卡上实现的硬件加速 IP 核，将单步迭代速度提升约 **10 倍**，平均迭代时间减少 **7.9 倍**，总体加速比达到 **2.5 倍**。

### Levenberg-Marquardt 算法回顾

LM 算法是一种迭代优化方法，通过最小化位姿误差来求解关节角。核心更新公式为：

$$
\left( J^T J + \lambda I \right) \Delta q = -J^T e

$$

其中：

- $q \in \mathbb{R}^5$：关节角向量（SO-101 为 5 自由度）
- $e \in \mathbb{R}^6$：位姿误差（3D 位置 + 3D 姿态，angle-axis 表示）
- $J \in \mathbb{R}^{6 \times 5}$：几何雅可比矩阵（末端速度对关节角速度的偏导）
- $\lambda$：阻尼系数（正则化参数，防止奇异）

**单步迭代流程**：

1. **正运动学（FK）**：计算当前关节角 $q$ 对应的末端位姿 $T(q)$
2. **误差计算**：$e = \text{pose\_error}(T\_{\text{goal}}, T(q))$
3. **雅可比计算**：$J = \frac{\partial v}{\partial \dot{q}}$（6 行 5 列）
4. **构造正规方程**：$A = J^T J + \lambda I$，$b = -J^T e$
5. **Cholesky 求解**：分解 $A = L L^T$，求解 $\Delta q$
6. **更新关节角**：$q \leftarrow q + \Delta q$

本项目的硬件加速核心在于 **步骤 3-5**，这三步涉及大量矩阵运算，占据了 LM 算法 80% 以上的计算时间。

### 硬件加速的关键计算

在 LM 迭代中，最耗时的操作包括：

1. **雅可比矩阵计算**（`compute_fk_jacobian_so101`）

   - 输入：$\sin(q)$、$\cos(q)$（预计算，避免 FPGA 实现超越函数）
   - 输出：$J \in \mathbb{R}^{6 \times 5}$
   - 复杂度：涉及 30+ 次浮点乘法、加法（旋转矩阵连乘与叉积）
   - 优化：UNROLL factor=3，部分展开平衡 DSP 使用与延迟
2. **矩阵乘法**（调用 Vitis Solver L1 库）

   - $A = J^T J$：$(5 \times 6) \times (6 \times 5) \rightarrow (5 \times 5)$
   - $b = J^T e$：$(5 \times 6) \times (6 \times 1) \rightarrow (5 \times 1)$
   - 复杂度：$O(N^3)$ 或 $O(N^2 M)$（$N=5$，$M=6$）
   - 优化：使用 Xilinx L1 库的高度优化实现（PIPELINE + DATAFLOW）
3. **Cholesky 分解与三角求解**（调用 Vitis Solver L1 库）

   - 分解：$A = L L^T$（$L$ 为下三角矩阵）
   - 前向替换：$L y = b$
   - 后向替换：$L^T \Delta q = y$
   - 复杂度：$O(N^3)$
   - 优化：**完全展开**（UNROLL），利用 DSP 流水线（延迟最小化）

FPGA 可以针对这些操作进行流水线优化，并利用 BRAM 和 DSP 资源实现高效的浮点运算。相比 ARM 处理器的顺序执行，FPGA 的并行计算能力在小矩阵密集运算场景下优势显著。

---

## 系统架构

### 硬件平台

- **开发板**：PYNQ-Z2（Xilinx Zynq-7020 SoC）
  - **PS 端**（Processing System）：双核 ARM Cortex-A9 @ 650 MHz
    - 运行 Python 代码（PYNQ 框架、Jupyter Notebook）
    - 高层控制逻辑（误差计算、收敛判断、阻尼调整）
    - 通过 AXI-Lite 接口访问 PL 端 IP 核
  - **PL 端**（Programmable Logic）：Artix-7 FPGA
    - 部署 IK 求解器 IP 核（`xf_solver_lm_so101`）
    - 硬件资源：53200 LUT、106400 FF、220 DSP、140 BRAM
- **通信接口**：AXI-Lite
- **编程框架**：
  - **Vitis HLS 2024.2**：高层次综合（C++ → RTL）
  - **Vivado 2024.2**：FPGA 综合与布局布线
  - **PYNQ 2.7+**：Python 到 FPGA 的桥接层（Overlay 机制）

### 软硬件协同架构

```
┌─────────────────────────────────────────────────────┐
│              Jupyter Notebook 层                     │
│  （用户交互、可视化、演示）                           │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│           Python 应用层                              │
│  • robot.py      - 机器人模型                        │
│  • solvers.py    - 软件 IK 求解器                    │
│  • hw_solver_wrapper.py - 硬件求解器封装             │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│      硬件抽象层（HAL）                                │
│  • PYNQ Overlay  - 比特流加载                        │
│  • MMIO          - 寄存器访问                        │
│  • driver        - 舵机控制                          │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│              FPGA 硬件层                             │
│  • xf_solver_lm_so101 IP 核                         │
│  • AXI-Lite Slave 接口                              │
└─────────────────────────────────────────────────────┘
```

### IP 核接口设计

硬件求解器通过 AXI-Lite 寄存器接口与 PS 端交互，主要寄存器包括：

#### 输入寄存器


| 偏移地址    | 名称    | 类型  | 描述                 |
| ------------- | --------- | ------- | ---------------------- |
| 0x10        | sin0    | float | 关节 0 的正弦值      |
| 0x18        | sin1    | float | 关节 1 的正弦值      |
| 0x20        | sin2    | float | 关节 2 的正弦值      |
| 0x28        | sin3    | float | 关节 3 的正弦值      |
| 0x30        | sin4    | float | 关节 4 的正弦值      |
| 0x38        | cos0    | float | 关节 0 的余弦值      |
| 0x40        | cos1    | float | 关节 1 的余弦值      |
| 0x48        | cos2    | float | 关节 2 的余弦值      |
| 0x50        | cos3    | float | 关节 3 的余弦值      |
| 0x58        | cos4    | float | 关节 4 的余弦值      |
| 0x60 - 0x88 | e0 - e5 | float | 位姿误差向量（6 维） |
| 0x90        | lambda  | float | 阻尼系数             |

#### 输出寄存器


| 偏移地址 | 名称   | 类型  | 描述                   |
| ---------- | -------- | ------- | ------------------------ |
| 0x98     | d0     | float | 关节角增量$\Delta q\_0$ |
| 0xA8     | d1     | float | 关节角增量$\Delta q\_1$ |
| 0xB8     | d2     | float | 关节角增量$\Delta q\_2$ |
| 0xC8     | d3     | float | 关节角增量$\Delta q\_3$ |
| 0xD8     | d4     | float | 关节角增量$\Delta q\_4$ |
| 0xE8     | status | uint  | 执行状态码             |

#### 控制寄存器


| 偏移地址 | 名称 | 位域 | 描述                         |
| ---------- | ------ | ------ | ------------------------------ |
| 0x00     | CTRL | [0]  | 启动计算（写 1 触发）        |
| 0x00     | STAT | [1]  | 完成标志（读取时检查 bit 1） |

---

## 程序设计结构

本项目采用软硬件协同设计，明确划分计算边界：**误差计算在 PS，增量求解在 PL**。

### 硬件层次（FPGA PL 端）

#### 1. IP 核实现：`lm_solver_so101.hpp`

**核心数据结构**：

```cpp
struct SO101_Params {
    // 5个关节的几何参数（基于 ET 序列）
    static constexpr float j1_tx = 0.0612f;  // Joint 1: Rz
    static constexpr float j1_tz = 0.0598f;
    static constexpr float j2_tx = 0.02943f; // Joint 2: Ry
    // ...
};
```

**三大核心函数**：

1. **`compute_fk_jacobian_so101<T>(sin_q, cos_q, p_end, J)`**

   - **输入**：$\sin(q\_i)$、$\cos(q\_i)$ [5]（PS 端预计算）
   - **输出**：末端位置 $p$ [3]，雅可比 $J$ [6×5]
   - **实现**：串联 5 个 ET 变换（Rz → Ry → Ry → Ry → Rx）
   - **优化**：`#pragma HLS UNROLL factor=3`（部分展开叉积计算）
2. **`lm_solve_step_so101<T>(sin_q, cos_q, error, lambda, delta)`**

   - **输入**：误差向量 $e$ [6]，阻尼系数 $\lambda$
   - **输出**：关节增量 $\Delta q$ [5]
   - **流程**：
     1. 调用 `compute_fk_jacobian_so101` 获取 $J$
     2. 矩阵乘法：$A = J^T J$（调用 Vitis L1 库）
     3. 矩阵乘法：$b = -J^T e$
     4. 添加阻尼：$A \leftarrow A + \lambda I$
     5. Cholesky 分解：$A = L L^T$（调用 Vitis L1 库）
     6. 三角求解：$L y = b$，$L^T \Delta q = y$（**完全展开**）
3. **Cholesky 分解与三角求解优化**

   - **对角线倒数预计算**：避免除法流水线停顿
     ```cpp
     T L_diag_inv[N];
     for(int i = 0; i < N; i++) {
         L_diag_inv[i] = T(1.0) / L[i][i];
     }
     ```
   - **前向替换展开**（以 $N=5$ 为例）：
     ```cpp
     y[0] = b[0] * L_diag_inv[0];
     y[1] = (b[1] - L[1][0]*y[0]) * L_diag_inv[1];
     y[2] = (b[2] - L[2][0]*y[0] - L[2][1]*y[1]) * L_diag_inv[2];
     // ...
     ```
   - **后向替换展开**：从 $\Delta q\_4$ 到 $\Delta q\_0$ 逆序计算

#### 2. Vivado 工程：`src/overlay/`

- **Block Design**：AXI 互连、时钟域转换、IP 核实例化
- **比特流输出**：`notebook/design_1.bit` + `design_1.hwh`（硬件描述）
- **资源占用**（综合后）：
  - LUT: 33743 / 53200 (63%)
  - DSP: 200 / 220 (90%)
  - BRAM: 8 / 140 (6%)

### 软件层次（Python PS 端）

#### 1. 硬件抽象层：`hw_solver_wrapper.py`

**`HWSolverIterator` 类**：封装 AXI-Lite 寄存器访问

- **写入**：通过 `solver_ip.mmio.write(offset, value)` 传递 sin/cos、误差、lambda
- **读取**：通过 `solver_ip.mmio.read(offset)` 获取 $\Delta q$、状态码
- **同步**：轮询控制寄存器的 Done 标志位（bit 1）

**`HWSolver` 类**：与 `robot.ikine_LM()` 兼容的高层接口

- **软件部分**：FK 误差计算（调用 `robot.fkine(q)`）
- **硬件部分**：增量求解（调用 `hw_iterator.solve_step()`）
- **迭代控制**：收敛判断、阻尼调整（Wampler/Sugihara 方法）

#### 2. 工厂函数：`create_hw_solver()`

```python
from pynq import Overlay

overlay = Overlay('design_1.bit')
solver_ip = overlay.xf_solver_lm_so101_0
hw_solver = HWSolver(robot, HWSolverIterator(solver_ip, robot))
```

### 数据流图

```
┌─────────────────────────────────────────────────────┐
│              PS 端（Python）                         │
│  1. 计算 FK: T_current = robot.fkine(q)             │
│  2. 计算误差: e = pose_error(T_goal, T_current)     │
│  3. 计算三角函数: sin_q, cos_q = sin(q), cos(q)     │
│  4. 通过 AXI 写入: sin_q, cos_q, e, lambda → FPGA   │
└────────────────┬────────────────────────────────────┘
                 │ AXI-Lite 总线
┌────────────────▼────────────────────────────────────┐
│              PL 端（FPGA IP 核）                     │
│  5. 计算雅可比: J = compute_jacobian(sin_q, cos_q)  │
│  6. 矩阵乘法: A = J^T * J, b = J^T * e              │
│  7. Cholesky: A = L * L^T                           │
│  8. 三角求解: delta = solve(L, b)                   │
│  9. 通过 AXI 返回: delta → PS                       │
└────────────────┬────────────────────────────────────┘
                 │ AXI-Lite 总线
┌────────────────▼────────────────────────────────────┐
│              PS 端（Python）                         │
│  10. 更新关节角: q = q + delta                      │
│  11. 检查收敛: ||e|| < tol ?                        │
└─────────────────────────────────────────────────────┘
```

---

## 方法作用

本节深入解析 HLS C++ 代码中的核心函数实现与优化策略。

### 1. `compute_fk_jacobian_so101()` - 正运动学与雅可比计算

**函数原型**：

```cpp
template <typename T>
void compute_fk_jacobian_so101(
    const T sin_q[5],      // 输入：sin(q0), ..., sin(q4)
    const T cos_q[5],      // 输入：cos(q0), ..., cos(q4)
    T p_end[3],            // 输出：末端位置 [x, y, z]
    T J[6][5]              // 输出：几何雅可比 [6行×5列]
);
```

**功能**：基于 SO-101 的 ETS（Elementary Transform Sequence）串联计算正运动学和几何雅可比矩阵。

**几何参数**（硬编码在 `SO101_Params` 结构体）：

```cpp
struct SO101_Params {
    // Joint 1 (Rz): Base → J1
    static constexpr float j1_tx = 0.0612f;
    static constexpr float j1_tz = 0.0598f;
  
    // Joint 2 (Ry): J1 → J2
    static constexpr float j2_tx = 0.02943f;
    static constexpr float j2_tz = 0.05504f;
  
    // Joint 3 (Ry): J2 → J3
    static constexpr float j3_tx = 0.02798f;
    static constexpr float j3_tz = 0.1127f;
  
    // Joint 4 (Ry): J3 → J4
    static constexpr float j4_tx = 0.15504f;
    static constexpr float j4_tz = 0.00519f;
  
    // Joint 5 (Rx): J4 → End
    static constexpr float j5_tx = 0.0593f;
    static constexpr float j5_tz = 0.00996f;
};
```

**核心算法**：

1. **初始化**：旋转矩阵 $R = I\_3$，位置 $p = [t\_{x1}, 0, t\_{z1}]^T$
2. **逐关节串联**（5 个关节）：
   - 更新旋转：$R \leftarrow R \cdot \text{Rot}(\text{axis}, q\_i)$（axis 为 X/Y/Z）
   - 更新位置：$p \leftarrow p + R \cdot [t\_x, 0, t\_z]^T$
   - 保存关节信息：$z\_i = R[:, \text{axis}]$（旋转轴），$p\_i = p$（关节位置）
3. **雅可比计算**：
   - 线速度列：$J\_v^i = z\_i \times (p\_{\text{end}} - p\_i)$（叉积）
   - 角速度列：$J\_\omega^i = z\_i$

**HLS 优化**：

```cpp
#pragma HLS INLINE off  // 防止自动内联（便于性能分析）
#pragma HLS UNROLL factor=3  // 部分展开雅可比列计算（平衡DSP与延迟）
#pragma HLS ARRAY_PARTITION variable=J dim=2 complete  // 雅可比列完全分区
#pragma HLS ARRAY_PARTITION variable=J dim=1 complete  // 雅可比行完全分区
```

**数学推导**（以 Joint 2 为例）：

- 旋转轴为 Y 轴：$z\_1 = R[:, 1] = [R\_{01}, R\_{11}, R\_{21}]^T$
- 位置差：$\Delta p = p\_{\text{end}} - p\_1$
- 叉积：
  $$
  J_v^1 = \begin{bmatrix} z_{1y} \Delta p_z - z_{1z} \Delta p_y \\ z_{1z} \Delta p_x - z_{1x} \Delta p_z \\ z_{1x} \Delta p_y - z_{1y} \Delta p_x \end{bmatrix}

  $$

---

### 2. `lm_solve_step_so101()` - LM 单步求解

**函数原型**：

```cpp
template <typename T>
int lm_solve_step_so101(
    const T sin_q[5],      // 输入：sin(q)
    const T cos_q[5],      // 输入：cos(q)
    const T error[6],      // 输入：位姿误差 e
    T lambda,              // 输入：阻尼系数
    T delta[5]             // 输出：关节增量 Δq
);
```

**功能**：完成 LM 算法的单步迭代，从误差到关节增量。

**计算流程**：

1. **计算雅可比**（调用 `compute_fk_jacobian_so101`）

   ```cpp
   T J[6][5];
   #pragma HLS ARRAY_PARTITION variable=J dim=0 complete
   compute_fk_jacobian_so101<T>(sin_q, cos_q, p_end, J);
   ```
2. **矩阵乘法 $A = J^T J$**（调用 Vitis Solver L1 库）

   ```cpp
   T A[5][5];
   #pragma HLS ARRAY_PARTITION variable=A dim=0 complete
   typedef matrixMultiplyTraits<Transpose, NoTranspose, 6, 5, 6, 5, T, T> JtJ_Traits;
   matrixMultiplyTop<Transpose, NoTranspose, 6, 5, 6, 5, 5, 5, JtJ_Traits, T, T>(J, J, A);
   ```

   - **Transpose 模式**：自动转置第一个矩阵
   - **流水线深度**：库内部已优化（II=1，吞吐率最大化）
3. **矩阵乘法 $b = J^T e$**

   ```cpp
   T b[5];
   #pragma HLS ARRAY_PARTITION variable=b complete
   typedef matrixMultiplyTraits<Transpose, NoTranspose, 6, 5, 6, 1, T, T> Jte_Traits;
   matrixMultiplyTop<Transpose, NoTranspose, 6, 5, 6, 1, 5, 1, Jte_Traits, T, T>(J, e_mat, b_mat);
   ```
4. **添加阻尼 $A \leftarrow A + \lambda I$**

   ```cpp
   for(int i = 0; i < 5; i++) {
       #pragma HLS UNROLL
       A[i][i] += lambda;
   }
   ```

   - **UNROLL**：完全展开（5 个加法并行执行）
5. **Cholesky 分解 $A = L L^T$**（调用 Vitis Solver L1 库）

   ```cpp
   T L[5][5];
   #pragma HLS ARRAY_PARTITION variable=L dim=0 complete
   typedef choleskyTraits<true, 5, T, T> CholeskyConfig;
   int chol_ret = choleskyTop<true, 5, CholeskyConfig, T, T>(A, L);
   ```

   - **Lower Triangle 模式**：只计算下三角（节省 50% 运算）
   - **失败处理**：若 $A$ 非正定，返回错误码 1
6. **三角求解**（**完全展开实现**）

   - **前向替换 $L y = b$**：
     ```cpp
     T L_diag_inv[5];
     for(int i = 0; i < 5; i++) {
         L_diag_inv[i] = T(1.0) / L[i][i];  // 预计算对角线倒数
     }
     y[0] = b[0] * L_diag_inv[0];
     y[1] = (b[1] - L[1][0]*y[0]) * L_diag_inv[1];
     y[2] = (b[2] - L[2][0]*y[0] - L[2][1]*y[1]) * L_diag_inv[2];
     y[3] = (b[3] - L[3][0]*y[0] - L[3][1]*y[1] - L[3][2]*y[2]) * L_diag_inv[3];
     y[4] = (b[4] - L[4][0]*y[0] - L[4][1]*y[1] - L[4][2]*y[2] - L[4][3]*y[3]) * L_diag_inv[4];
     ```
   - **后向替换 $L^T \Delta q = y$**：
     ```cpp
     delta[4] = y[4] * L_diag_inv[4];
     delta[3] = (y[3] - L[4][3]*delta[4]) * L_diag_inv[3];
     delta[2] = (y[2] - L[3][2]*delta[3] - L[4][2]*delta[4]) * L_diag_inv[2];
     delta[1] = (y[1] - L[2][1]*delta[2] - L[3][1]*delta[3] - L[4][1]*delta[4]) * L_diag_inv[1];
     delta[0] = (y[0] - L[1][0]*delta[1] - L[2][0]*delta[2] - L[3][0]*delta[3] - L[4][0]*delta[4]) * L_diag_inv[0];
     ```

---

### 3. Python 硬件封装：`hw_solver_wrapper.py`

**`HWSolverIterator` 类**：AXI-Lite 寄存器映射


| 寄存器偏移 | 名称      | 数据类型 | 方向 | 描述                   |
| ------------ | ----------- | ---------- | ------ | ------------------------ |
| 0x00       | CTRL/STAT | uint32   | R/W  | [0]=启动，[1]=完成标志 |
| 0x10-0x30  | sin0-sin4 | float32  | W    | $\sin(q\_i)$            |
| 0x38-0x58  | cos0-cos4 | float32  | W    | $\cos(q\_i)$            |
| 0x60-0x88  | e0-e5     | float32  | W    | 误差向量$e$            |
| 0x90       | lambda    | float32  | W    | 阻尼系数$\lambda$      |
| 0x98-0xD8  | d0-d4     | float32  | R    | 关节增量$\Delta q$     |
| 0xE8       | status    | uint32   | R    | 执行状态码             |

**核心方法**：

```python
def solve_step(self, q, error, lambda_damping=0.01):
    sin_q, cos_q = np.sin(q), np.cos(q)
  
    # 写入输入
    for i in range(5):
        self.solver_ip.mmio.write(0x10 + i*8, sin_q[i].astype(np.float32))
        self.solver_ip.mmio.write(0x38 + i*8, cos_q[i].astype(np.float32))
    for i in range(6):
        self.solver_ip.mmio.write(0x60 + i*8, error[i].astype(np.float32))
    self.solver_ip.mmio.write(0x90, np.float32(lambda_damping))
  
    # 启动计算
    self.solver_ip.mmio.write(0x00, 0x01)
  
    # 轮询等待（超时保护）
    for _ in range(10000):
        if self.solver_ip.mmio.read(0x00) & 0x02:
            break
  
    # 读取输出
    delta = np.array([self.solver_ip.mmio.read(0x98 + i*0x10) for i in range(5)], dtype=np.float32)
    status = self.solver_ip.mmio.read(0xE8)
    return delta, status
```

**MMIO 开销分析**：

- 写入 23 个寄存器（sin×5 + cos×5 + error×6 + lambda×1）：约 **50 μs**
- 硬件计算：约 **100 μs**（包括雅可比、矩阵乘法、Cholesky）
- 读取 6 个寄存器（delta×5 + status×1）：约 **15 μs**
- **总延迟**：约 **165 μs/步**（相比纯软件的 ~1500 μs，提升 **9×**）

---

## 性能数据

### 性能提升总结

经对比测试，硬件加速相比纯软件实现的性能改善如下：


| 性能指标         | 纯软件（PS） | 硬件加速（PL） | 提升倍数   |
| ------------------ | -------------- | ---------------- | ------------ |
| **单步迭代速度** | 基准         | 改善           | **~10×**  |
| **平均迭代时间** | 基准         | 减少           | **~7.9×** |
| **吞吐率**       | 基准         | 提升           | **~9.2×** |
| **总体加速比**   | 1.0×        | 加速           | **~2.5×** |

> **注**：具体性能数据会因测试场景和目标位姿而异。详细图表和数据见 `notebook/test_results/`。

### 资源利用率

FPGA 资源消耗（Zynq-7020）：


| 资源类型 | 使用量 | 总量   | 占比  |
| ---------- | -------- | -------- | ------- |
| LUT      | 28456  | 53200  | 53.5% |
| FF       | 31204  | 106400 | 29.3% |
| BRAM     | 48     | 140    | 34.3% |
| DSP      | 134    | 220    | 60.9% |

**分析**：

- **DSP 资源占用较高**（60.9%）：用于浮点乘法、乘加操作（矩阵乘法、Cholesky 分解）
- **BRAM 用于存储中间矩阵**：雅可比 $J$ [6×5]、正规矩阵 $A$ [5×5]、Cholesky 因子 $L$ [5×5]
- **LUT/FF 用于控制逻辑与数据路径**：AXI-Lite 接口、状态机、浮点运算单元
- **优化空间**：若需支持更高自由度（如 6-DOF），可考虑时分复用 DSP（降低资源占用，增加延迟）

---

## 优化策略

### HLS 优化指令

在 `lm_solver_so101.hpp` 中应用的关键优化：

1. **PIPELINE**（流水线）

   ```cpp
   #pragma HLS PIPELINE II=1
   ```

   - **作用**：使循环各迭代重叠执行，目标启动间隔（II）为 1 周期
   - **应用场景**：矩阵乘法内层循环、Cholesky 分解迭代
   - **效果**：将延迟从 $O(N^2)$ 降至 $O(N)$（对于可流水化的循环）
2. **UNROLL**（循环展开）

   ```cpp
   #pragma HLS UNROLL factor=3  // 部分展开
   #pragma HLS UNROLL           // 完全展开
   ```

   - **部分展开**（factor=3）：用于雅可比列计算（5 个关节 → 2 个并行单元）
     - 平衡并行度与资源占用
   - **完全展开**：用于小循环（如 5×5 矩阵对角线操作）
     - 消除循环开销，所有迭代并行执行
3. **ARRAY_PARTITION**（数组分区）

   ```cpp
   #pragma HLS ARRAY_PARTITION variable=J complete dim=1  // 行完全分区
   #pragma HLS ARRAY_PARTITION variable=J complete dim=2  // 列完全分区
   #pragma HLS ARRAY_PARTITION variable=A dim=0 complete  // 全维度完全分区
   ```

   - **作用**：将数组分散到多个 BRAM 块或寄存器
   - **效果**：消除访问冲突，允许并行读写
   - **代价**：BRAM 数量增加（$5 \times 5 = 25$ 个存储单元）

### Cholesky 算子集成

本项目借鉴了初赛阶段优化的 Cholesky 算子（性能提升 **4-5 倍**），用于求解正规方程 $\left( J^T J + \lambda I \right) \Delta q = -J^T e$：

**输入**：对称正定矩阵 $A = J^T J + \lambda I \in \mathbb{R}^{5 \times 5}$

**输出**：下三角矩阵 $L$（满足 $A = L L^T$）

**后续**：通过前向/后向回代求解 $\Delta q$

**优化关键**：

1. **消除循环依赖**（通过数据重排）

   - 原始算法：$L\_{ij} = (A\_{ij} - \sum\_{k=1}^{j-1} L\_{ik} L\_{jk}) / L\_{jj}$
   - 优化算法：预计算对角线倒数，用乘法替代除法
2. **BRAM 带宽优化**（双端口访问）

   - 分区策略：按行分区（row-wise partition）
   - 并行度：同时读取 $L\_{ik}$ 和 $L\_{jk}$
3. **浮点流水线深度调优**

   - 乘法器延迟：8 周期
   - 加法器延迟：11 周期
   - 除法器延迟：28 周期
   - **优化**：用倒数乘法（延迟 8+8=16 周期）替代除法

**性能数据**（Cholesky 算子单独测试）：


| 性能指标                | 优化前 | 优化后 | 改善幅度 |
| ------------------------- | -------- | -------- | ---------- |
| **延迟 (Latency)**      | 4919   | 991    | 79.9%    |
| **执行时间**            | 30871  | 4731   | 84.7%    |
| **启动间隔 (II)**       | 696    | 124    | 82.2%    |
| **吞吐率 (Throughput)** | 2.3e5  | 1.6e6  | 596%     |

---

## 总结

FPGA 硬件加速为实时逆运动学求解提供了显著的性能提升，使得在资源受限的嵌入式平台上也能实现高频控制（>100 Hz）。本章介绍的软硬件协同架构具有良好的可扩展性，可进一步优化以支持更复杂的机器人（如 6-DOF、7-DOF 冗余臂）或更高精度的求解算法（如二阶 LM、SQP）。

**关键要点**：

1. **性能提升**：

   - 单步迭代加速约 **10 倍**（通过流水线与并行优化）
   - 总体求解加速约 **2.5 倍**（含 MMIO 通信开销）
2. **软硬件分工**：

   - **PS 端**：FK/误差计算、收敛判断、阻尼调整（需要条件分支与浮点超越函数）
   - **PL 端**：雅可比计算、矩阵乘法、Cholesky 分解（高度并行的线性代数）
3. **接口设计**：

   - **AXI-Lite 寄存器映射**：输入/输出寄存器清晰分离
   - **三角函数预计算**：避免 FPGA 实现复杂的 CORDIC 算法
4. **优化经验**：

   - **Cholesky 算子优化**：延迟降低 **79.9%**，吞吐率提升 **596%**
   - **三角求解完全展开**：延迟从 $O(N^2)$ 降至 $O(N)$
   - **数组分区策略**：小矩阵（$N \leq 5$）完全分区到寄存器
5. **可扩展性**：

   - 修改 `SO101_Params` 即可适配不同机器人（仅需重新综合）
   - 支持更高自由度：增加 $N$ 和矩阵维度（需评估资源占用）
   - 支持其他 IK 算法：替换 `lm_solve_step_so101` 内核（如 GN、NR、QP）
