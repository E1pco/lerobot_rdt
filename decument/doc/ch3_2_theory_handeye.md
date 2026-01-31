# 3.2 理论：手眼标定（Hand-Eye Calibration）

## 目录

1. [概述](#1-概述)
2. [眼在手上标定 (Eye-in-Hand)](#2-眼在手上标定-eye-in-hand)
3. [眼在手外标定 (Eye-to-Hand)](#3-眼在手外标定-eye-to-hand)
4. [数学原理](#4-数学原理)
5. [标定流程](#5-标定流程)
6. [常见算法](#6-常见算法)
7. [标定质量评估](#7-标定质量评估)
8. [实际应用建议](#8-实际应用建议)

---

## 1. 概述

### 1.1 什么是手眼标定？

**手眼标定**是机器人视觉系统中的关键技术，用于确定相机坐标系与机器人坐标系之间的空间变换关系。根据相机安装位置的不同，手眼标定分为两种主要类型。

### 1.2 基本概念

| 概念 | 说明 |
|-----|------|
| **坐标系统** | 机器人系统涉及多个坐标系：基座坐标系、末端执行器坐标系、相机坐标系、标定板坐标系 |
| **齐次变换矩阵** | 使用 $4 \times 4$ 矩阵表示三维空间中的旋转和平移 |
| **标定目标** | 通常使用棋盘格标定板或圆点标定板 |

### 1.3 应用场景

| 场景 | 需求 |
|-----|------|
| 视觉引导抓取 | 相机检测物体位置 → 转换为机器人可执行的目标位姿 |
| 视觉伺服 | 实时跟踪目标 → 闭环控制机器人运动 |
| 数据采集 | 记录图像时同步记录对应的机器人状态 |
| 三维重建 | 多视角融合时需要精确的相机位姿 |

---

## 2. 眼在手上标定 (Eye-in-Hand)

### 2.1 配置说明

眼在手上配置是指**相机固定安装在机械臂的末端执行器上**，相机随机械臂一起运动。

```
机器人基座 → 机械臂关节 → 末端执行器 → 相机
                                    ↓
                              标定板(固定在工作台)
```

### 2.2 坐标关系

在眼在手上配置中，需要求解的是**末端执行器到相机的变换矩阵** ${}^E\mathbf{T}_C$（Hand-to-Eye transformation）。

**关键坐标系**：
- **Base** ($B$)：机器人基座坐标系
- **End** ($E$)：机械臂末端执行器坐标系
- **Camera** ($C$)：相机坐标系
- **Target** ($T$)：标定板坐标系（固定）

### 2.3 变换关系推导

坐标系之间的完整变换链为：

$${}^B\mathbf{T}_T = {}^B\mathbf{T}_E \cdot {}^E\mathbf{T}_C \cdot {}^C\mathbf{T}_T$$

其中 ${}^A\mathbf{T}_B$ 表示从坐标系 $B$ 到坐标系 $A$ 的齐次变换矩阵。

由于标定板位置固定，当机械臂移动到不同位姿时：

$${}^B\mathbf{T}_E^{(1)} \cdot {}^E\mathbf{T}_C \cdot {}^C\mathbf{T}_T^{(1)} = {}^B\mathbf{T}_E^{(2)} \cdot {}^E\mathbf{T}_C \cdot {}^C\mathbf{T}_T^{(2)}$$

两边同时左乘 $({}^B\mathbf{T}_E^{(1)})^{-1}$ 和右乘 $({}^C\mathbf{T}_T^{(1)})^{-1}$，整理得到标准的 $\mathbf{AX} = \mathbf{XB}$ 问题：

$$\mathbf{A} \cdot \mathbf{X} = \mathbf{X} \cdot \mathbf{B}$$

其中：
- $\mathbf{A} = ({}^B\mathbf{T}_E^{(1)})^{-1} \cdot {}^B\mathbf{T}_E^{(2)} = {}^{E_1}\mathbf{T}_{E_2}$ （末端执行器的相对运动）
- $\mathbf{X} = {}^E\mathbf{T}_C$ （待求解的手眼关系）
- $\mathbf{B} = ({}^C\mathbf{T}_T^{(2)})^{-1} \cdot {}^C\mathbf{T}_T^{(1)} = {}^{C_2}\mathbf{T}_{C_1}$ （相机观察到的标定板相对运动）

### 2.4 应用场景

1. **机器人抓取**：相机跟随机械臂观察目标物体
2. **精密装配**：相机提供实时视觉反馈
3. **移动检测**：相机需要从多个角度观察目标
4. **焊接/喷涂**：相机监控作业过程

### 2.5 优势与劣势

| 优势 | 劣势 |
|-----|------|
| 相机视野随机械臂移动，可从多角度观察目标 | 相机增加末端负载 |
| 适合需要近距离观察的精密作业 | 线缆管理复杂 |
| 工作空间灵活性高 | 相机振动可能影响图像质量 |

---

## 3. 眼在手外标定 (Eye-to-Hand)

### 3.1 配置说明

眼在手外配置是指**相机固定安装在机器人工作空间的外部**（如支架、天花板等），相机位置固定不动。

```
        相机(固定)
          ↓
    机器人基座 → 机械臂关节 → 末端执行器 → 标定板
```

### 3.2 坐标关系

在眼在手外配置中，需要求解的是**机器人基座到相机的变换矩阵** ${}^C\mathbf{T}_B$（Base-to-Camera transformation）。

**关键坐标系**：
- **Base** ($B$)：机器人基座坐标系
- **End** ($E$)：机械臂末端执行器坐标系
- **Camera** ($C$)：相机坐标系（固定）
- **Target** ($T$)：标定板坐标系（固定在末端）

### 3.3 变换关系推导

坐标系之间的完整变换链为：

$${}^C\mathbf{T}_T = {}^C\mathbf{T}_B \cdot {}^B\mathbf{T}_E \cdot {}^E\mathbf{T}_T$$

当机械臂移动到不同位姿时：

$${}^C\mathbf{T}_T^{(1)} = {}^C\mathbf{T}_B \cdot {}^B\mathbf{T}_E^{(1)} \cdot {}^E\mathbf{T}_T$$
$${}^C\mathbf{T}_T^{(2)} = {}^C\mathbf{T}_B \cdot {}^B\mathbf{T}_E^{(2)} \cdot {}^E\mathbf{T}_T$$

同样可以转化为 $\mathbf{AX} = \mathbf{XB}$ 问题：

其中：
- $\mathbf{A} = {}^C\mathbf{T}_T^{(2)} \cdot ({}^C\mathbf{T}_T^{(1)})^{-1}$ （相机观察到的标定板运动）
- $\mathbf{X} = {}^C\mathbf{T}_B$ （待求解的相机到基座关系）
- $\mathbf{B} = {}^B\mathbf{T}_E^{(2)} \cdot ({}^B\mathbf{T}_E^{(1)})^{-1}$ （末端执行器的运动）

### 3.4 应用场景

1. **物料分拣**：固定相机俯视工作台
2. **质量检测**：大视野监控生产线
3. **装配引导**：相机观察整个装配区域
4. **码垛作业**：固定相机监控托盘区域

### 3.5 优势与劣势

| 优势 | 劣势 |
|-----|------|
| 不增加机械臂负载 | 视野固定，无法观察复杂角度 |
| 线缆布置简单 | 可能存在遮挡问题 |
| 图像稳定性好 | 需要较大的安装空间 |
| 可以观察整个工作区域 | 对于小物体，分辨率可能不足 |

---

## 4. 数学原理

### 4.1 齐次变换矩阵

三维空间中的齐次变换矩阵表示为：

$$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in SE(3)$$

其中：
- $\mathbf{R} \in SO(3)$ 是 $3 \times 3$ 旋转矩阵，满足 $\mathbf{R}^T\mathbf{R} = \mathbf{I}$ 且 $\det(\mathbf{R}) = 1$
- $\mathbf{t} \in \mathbb{R}^3$ 是 $3 \times 1$ 平移向量

对于点 $\mathbf{p} = [x, y, z, 1]^T$，变换后的点为：

$$\mathbf{p}' = \mathbf{T} \cdot \mathbf{p}$$

### 4.2 $\mathbf{AX} = \mathbf{XB}$ 问题

这是手眼标定的**核心数学问题**，需要从多组观测数据中求解 $\mathbf{X}$：

$$\mathbf{A}_i \cdot \mathbf{X} = \mathbf{X} \cdot \mathbf{B}_i, \quad i = 1, 2, \ldots, n$$

将齐次变换矩阵分解为旋转和平移两部分：

$$\mathbf{A}_i = \begin{bmatrix} \mathbf{R}_A^{(i)} & \mathbf{t}_A^{(i)} \\ \mathbf{0}^T & 1 \end{bmatrix}, \quad \mathbf{X} = \begin{bmatrix} \mathbf{R}_X & \mathbf{t}_X \\ \mathbf{0}^T & 1 \end{bmatrix}, \quad \mathbf{B}_i = \begin{bmatrix} \mathbf{R}_B^{(i)} & \mathbf{t}_B^{(i)} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

**旋转部分**：

$$\mathbf{R}_A^{(i)} \cdot \mathbf{R}_X = \mathbf{R}_X \cdot \mathbf{R}_B^{(i)}$$

**平移部分**：

$$\mathbf{R}_A^{(i)} \cdot \mathbf{t}_X + \mathbf{t}_A^{(i)} = \mathbf{R}_X \cdot \mathbf{t}_B^{(i)} + \mathbf{t}_X$$

整理得：

$$(\mathbf{R}_A^{(i)} - \mathbf{I}) \mathbf{t}_X = \mathbf{R}_X \mathbf{t}_B^{(i)} - \mathbf{t}_A^{(i)}$$

### 4.3 旋转表示方法

#### 4.3.1 旋转矩阵 (Rotation Matrix)

$3 \times 3$ 正交矩阵 $\mathbf{R} \in SO(3)$，满足约束：$\mathbf{R}^T\mathbf{R} = \mathbf{I}$，$\det(\mathbf{R}) = 1$

#### 4.3.2 欧拉角 (Euler Angles)

使用三个角度 $(\alpha, \beta, \gamma)$ 表示绕三个轴的旋转。常见的 ZYX 欧拉角：

$$\mathbf{R} = \mathbf{R}_z(\alpha) \mathbf{R}_y(\beta) \mathbf{R}_x(\gamma)$$

> ⚠️ **注意**：存在万向节锁问题 (Gimbal Lock)

#### 4.3.3 四元数 (Quaternion)

四元数 $\mathbf{q} = [q_w, q_x, q_y, q_z]^T \in \mathbb{H}$，满足：

$$q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1$$

旋转矩阵与四元数的转换：

$$\mathbf{R} = \begin{bmatrix} 
1-2(q_y^2+q_z^2) & 2(q_xq_y-q_wq_z) & 2(q_xq_z+q_wq_y) \\
2(q_xq_y+q_wq_z) & 1-2(q_x^2+q_z^2) & 2(q_yq_z-q_wq_x) \\
2(q_xq_z-q_wq_y) & 2(q_yq_z+q_wq_x) & 1-2(q_x^2+q_y^2)
\end{bmatrix}$$

#### 4.3.4 轴角表示 (Axis-Angle)

使用旋转轴 $\mathbf{k} = [k_x, k_y, k_z]^T$ ($\|\mathbf{k}\| = 1$) 和旋转角度 $\theta$。

**罗德里格斯公式** (Rodrigues' formula)：

$$\mathbf{R} = \mathbf{I} + \sin\theta \cdot [\mathbf{k}]_\times + (1-\cos\theta) \cdot [\mathbf{k}]_\times^2$$

其中 $[\mathbf{k}]_\times$ 是反对称矩阵：

$$[\mathbf{k}]_\times = \begin{bmatrix} 0 & -k_z & k_y \\ k_z & 0 & -k_x \\ -k_y & k_x & 0 \end{bmatrix}$$

---

## 5. 标定流程

### 5.1 准备工作

#### 5.1.1 硬件准备

| 设备 | 要求 |
|-----|------|
| **标定板** | 高精度棋盘格或圆点阵列，平整度 < 0.1 mm |
| **相机** | 分辨率至少 640×480，尽量选择低畸变镜头 |
| **安装支架** | 稳定、刚性好，避免振动 |
| **照明** | 均匀柔和，避免反光和阴影 |

#### 5.1.2 标定板尺寸选择

标定板方格大小的选择公式：

$$d_{square} = \frac{f \cdot W_{image}}{Z_{working} \cdot W_{sensor}} \cdot k$$

其中：
- $f$：镜头焦距
- $W_{image}$：期望的方格图像宽度（像素）
- $Z_{working}$：工作距离
- $W_{sensor}$：相机传感器宽度
- $k$：安全系数，通常取 0.1 - 0.15

### 5.2 标定步骤

#### 第一步：相机内参标定

在进行手眼标定前，必须先完成相机内参标定。相机内参矩阵：

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

需要标定的参数：
- 焦距：$(f_x, f_y)$
- 主点：$(c_x, c_y)$
- 畸变系数：$(k_1, k_2, p_1, p_2, k_3, \ldots)$

#### 第二步：采集标定数据

**眼在手上标定**：
1. 将标定板固定在工作台上
2. 移动机械臂到 $n$ 个不同位姿（通常 $n \geq 15$）
3. 在每个位姿 $i$ 记录：
   - 机械臂末端位姿：${}^B\mathbf{T}_E^{(i)}$（从正运动学获得）
   - 相机拍摄的标定板图像，通过 PnP 算法计算：${}^C\mathbf{T}_T^{(i)}$

**眼在手外标定**：
1. 将标定板固定在机械臂末端
2. 移动机械臂到 $n$ 个不同位姿
3. 在每个位姿 $i$ 记录同上数据

#### 第三步：位姿选择原则

为了获得良好的标定结果，位姿选择应遵循以下原则：

| 原则 | 建议 |
|-----|------|
| **覆盖工作空间** | 位姿分布应覆盖实际工作区域 |
| **旋转多样性** | 包含绕 X、Y、Z 轴的不同旋转，每轴跨度 ≥ 30° |
| **平移多样性** | 位移范围覆盖工作空间的 50% 以上 |
| **避免奇异位姿** | 避免接近关节极限和机械臂奇异点 |
| **图像质量** | 标定板占据图像面积的 20% - 60% |

#### 第四步：构建方程组

对于 $n$ 组观测数据，构建 $\mathbf{AX} = \mathbf{XB}$ 方程组。

**眼在手上**：

$$\mathbf{A}_i = ({}^B\mathbf{T}_E^{(1)})^{-1} \cdot {}^B\mathbf{T}_E^{(i+1)}$$
$$\mathbf{B}_i = ({}^C\mathbf{T}_T^{(i+1)})^{-1} \cdot {}^C\mathbf{T}_T^{(1)}$$
$$\mathbf{X} = {}^E\mathbf{T}_C$$

#### 第五步：求解手眼关系

使用标定算法求解 $\mathbf{X}$（详见下一章节）。

#### 第六步：验证标定结果

**重投影误差**：

$$e_{reproj} = \frac{1}{n \cdot m} \sum_{i=1}^{n} \sum_{j=1}^{m} \|\mathbf{p}_{ij} - \hat{\mathbf{p}}_{ij}\|_2$$

通常要求：$e_{reproj} < 1$ 像素

---

## 6. 常见算法

### 6.1 Tsai-Lenz 方法 (1989)

**原理**：分两步求解，先求旋转后求平移。

**旋转求解**：从 $\mathbf{R}_A^{(i)} \mathbf{R}_X = \mathbf{R}_X \mathbf{R}_B^{(i)}$ 出发，使用轴角表示构建线性方程组求解。

**平移求解**：已知 $\mathbf{R}_X$ 后，从 $(\mathbf{R}_A^{(i)} - \mathbf{I})\mathbf{t}_X = \mathbf{R}_X\mathbf{t}_B^{(i)} - \mathbf{t}_A^{(i)}$ 用最小二乘法求解 $\mathbf{t}_X$。

**特点**：
- 计算效率高：$O(n)$ 复杂度
- 对噪声较敏感
- 需要至少 3 组数据

### 6.2 Park-Martin 方法 (1994)

**原理**：使用四元数表示旋转，避免万向节锁问题。

**四元数约束**：

$$\mathbf{q}_A^{(i)} \otimes \mathbf{q}_X = \mathbf{q}_X \otimes \mathbf{q}_B^{(i)}$$

转化为矩阵形式，$\mathbf{q}_X$ 是 $\mathbf{M}^T\mathbf{M}$ 对应最小特征值的特征向量。

**特点**：
- 数值稳定性好
- 避免万向节锁
- 适合大旋转角度

### 6.3 Horaud-Dornaika 方法 (1995)

**原理**：同时优化旋转和平移，使用非线性最小化。

**目标函数**：

$$\min_{\mathbf{X}} \sum_{i=1}^{n} \|\mathbf{A}_i \mathbf{X} - \mathbf{X} \mathbf{B}_i\|_F^2$$

使用 Levenberg-Marquardt 方法迭代优化。

**特点**：
- 精度较高
- 计算复杂度较高
- 需要良好的初始值

### 6.4 Daniilidis 方法 (1999)

**原理**：使用对偶四元数统一处理旋转和平移。

**对偶四元数表示**：

$$\hat{\mathbf{q}} = \mathbf{q}_r + \epsilon\mathbf{q}_d$$

其中 $\mathbf{q}_r$ 表示旋转，$\mathbf{q}_d$ 与平移相关，$\epsilon^2 = 0$。

**特点**：
- 数学形式优雅
- 全局闭式解
- 旋转和平移统一处理
- 数值稳定性好

### 6.5 OpenCV 实现

OpenCV 提供了现成的手眼标定函数：

```python
import cv2
import numpy as np

# 准备数据 (n组观测)
R_gripper2base = [...]  # list of 3x3 rotation matrices
t_gripper2base = [...]  # list of 3x1 translation vectors
R_target2cam = [...]    # list of 3x3 rotation matrices
t_target2cam = [...]    # list of 3x1 translation vectors

# 眼在手上标定
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base,
    t_gripper2base,
    R_target2cam,
    t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI  # 选择算法
)

# 构建齐次变换矩阵
T_cam2gripper = np.eye(4)
T_cam2gripper[:3, :3] = R_cam2gripper
T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
```

**支持的算法选项**：

| 算法 | OpenCV 常量 |
|-----|-------------|
| Tsai-Lenz | `cv2.CALIB_HAND_EYE_TSAI` |
| Park-Martin | `cv2.CALIB_HAND_EYE_PARK` |
| Horaud-Dornaika | `cv2.CALIB_HAND_EYE_HORAUD` |
| Andreff | `cv2.CALIB_HAND_EYE_ANDREFF` |
| Daniilidis | `cv2.CALIB_HAND_EYE_DANIILIDIS` |

---

## 7. 标定质量评估

### 7.1 一致性误差

标定完成后，需要评估外参的精度。核心思想：**如果外参正确，则所有采样应该自洽**。

#### Eye-in-Hand 评估

计算残差矩阵：

$$\Delta_{ij} = (A_{ij} \cdot X) \cdot (X \cdot B_{ij})^{-1}$$

理想情况下 $\Delta_{ij} = I$（单位矩阵）。

#### Eye-to-Hand 评估

计算标定板相对末端的位姿：

$$T_{TG}^{(i)} = (T_{GB}^{(i)})^{-1} \cdot T_{CB} \cdot T_{TC}^{(i)}$$

比较不同采样之间的差异：

$$\Delta_{ij} = T_{TG}^{(i)} \cdot (T_{TG}^{(j)})^{-1}$$

### 7.2 误差度量

从残差矩阵 $\Delta$ 中提取误差：

**平移误差**：

$$e_t = \| t_\Delta \| \quad (\text{单位: mm})$$

**旋转误差**（使用旋转向量）：

$$e_r = \| \text{rotvec}(R_\Delta) \| \cdot \frac{180}{\pi} \quad (\text{单位: deg})$$

### 7.3 质量等级标准

| 等级 | 重投影误差 (pixel) | 平移误差 (mm) | 旋转误差 (deg) | 适用场景 |
|-----|-------------------|--------------|---------------|---------|
| **优秀** | < 0.5 | < 1.0 | < 0.2 | 精密装配 |
| **良好** | 0.5 - 1.0 | 1.0 - 2.0 | 0.2 - 0.5 | 一般抓取 |
| **可接受** | 1.0 - 2.0 | 2.0 - 5.0 | 0.5 - 1.0 | 粗定位 |
| **需改进** | > 2.0 | > 5.0 | > 1.0 | 需重新标定 |

### 7.4 一致性检验

进行 $N$ 次独立标定，计算变异系数（CV）：

$$\text{CV} = \frac{\sigma}{\mu} \times 100\%$$

| CV 范围 | 评价 |
|--------|------|
| < 5% | 稳定可靠 |
| 5% - 10% | 基本可接受 |
| ≥ 10% | 需要改进 |

---

## 8. 实际应用建议

### 8.1 提高标定精度的技巧

#### 数据采集优化

| 项目 | 建议 |
|-----|------|
| **采样数量** | 最少 3 组，推荐 15-20 组，高精度要求 > 30 组 |
| **旋转分散度** | $D_{rot} > 30°$ |
| **平移分散度** | $D_{trans} > 0.3 \times L_{workspace}$ |
| **图像质量** | 标定板占比 20%-60%，避免运动模糊 |

#### 硬件选择

**空间分辨率估计**：

$$\Delta x = \frac{Z \cdot p}{f}$$

对于 ±1 mm 的定位精度，建议：$\Delta x < 0.5$ mm

### 8.2 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 标定精度不高 | 相机内参误差大 | 重新标定相机，增加标定图像 |
| | 位姿数量不足 | 增加到 ≥ 15 组 |
| | 位姿多样性不足 | 增加旋转和平移覆盖范围 |
| 标定板检测失败 | 光照问题 | 使用漫反射光源，避免过曝 |
| | 标定板尺寸不当 | 调整标定板大小 |
| 结果不稳定 | 固定性不足 | 确保标定板或相机固定牢固 |
| | 振动 | 降低移动速度，增加延时 |
| 实际应用误差大 | 工作区域不一致 | 标定区域应覆盖实际工作区域 |
| | 温度变化 | 控制 ΔT < 5°C，定期重新标定 |

### 8.3 维护与更新

| 频率 | 检查项 |
|-----|--------|
| **每天** | 检查相机/标定板固定，快速精度测试 |
| **每周** | 使用标准测试位姿验证精度，记录误差趋势 |
| **每月** | 完整重新标定（或精度下降 > 20% 时） |

### 8.4 配置选择建议

| 特性 | 眼在手上 | 眼在手外 |
|------|----------|----------|
| **灵活性** | 高 - 相机可移动观察 | 低 - 固定视野 |
| **精度** | 受振动影响 | 图像稳定 |
| **负载** | 增加末端负载 | 无额外负载 |
| **工作范围** | 受机械臂限制 | 大视野覆盖 |
| **典型应用** | 抓取、装配、检测 | 分拣、码垛、监控 |

---

## 9. 代码实现参考

### 9.1 完整标定流程

```python
import cv2
import numpy as np

def calibrate_eye_in_hand(robot_poses, camera_poses, method=cv2.CALIB_HAND_EYE_TSAI):
    """
    Eye-in-Hand 手眼标定
    
    Args:
        robot_poses: list of 4x4 T_GB matrices (gripper to base)
        camera_poses: list of 4x4 T_TC matrices (target to camera)
        method: OpenCV 标定算法
    
    Returns:
        T_CG: 4x4 camera to gripper transformation
    """
    # 提取旋转和平移
    R_gripper2base = [T[:3, :3] for T in robot_poses]
    t_gripper2base = [T[:3, 3:4] for T in robot_poses]
    R_target2cam = [T[:3, :3] for T in camera_poses]
    t_target2cam = [T[:3, 3:4] for T in camera_poses]
    
    # OpenCV 标定
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=method
    )
    
    # 构造齐次变换矩阵
    T_CG = np.eye(4)
    T_CG[:3, :3] = R_cam2gripper
    T_CG[:3, 3] = t_cam2gripper.flatten()
    
    return T_CG

def evaluate_consistency(robot_poses, camera_poses, T_CG):
    """
    评估 Eye-in-Hand 标定一致性
    
    Returns:
        trans_errors: 平移误差列表 (mm)
        rot_errors: 旋转误差列表 (deg)
    """
    trans_errors = []
    rot_errors = []
    
    n = len(robot_poses)
    for i in range(n):
        for j in range(i + 1, n):
            # 构造 A 和 B
            A = np.linalg.inv(robot_poses[j]) @ robot_poses[i]
            B = camera_poses[j] @ np.linalg.inv(camera_poses[i])
            
            # 计算残差
            AX = A @ T_CG
            XB = T_CG @ B
            Delta = AX @ np.linalg.inv(XB)
            
            # 提取误差
            t_err = np.linalg.norm(Delta[:3, 3]) * 1000  # mm
            r_err = np.linalg.norm(cv2.Rodrigues(Delta[:3, :3])[0]) * 180 / np.pi  # deg
            
            trans_errors.append(t_err)
            rot_errors.append(r_err)
    
    return trans_errors, rot_errors
```

### 9.2 仓库对应文件

| 功能 | 文件 |
|-----|------|
| Eye-in-Hand 标定 | `vision/handeye_calibration_eyeinhand.py` |
| Eye-to-Hand 标定 | `vision/handeye_calibration_eyetohand.py` |
| 一致性评估 | `vision/handeye_utils.py` |

---

## 10. 参考文献

1. **Tsai, R. Y., & Lenz, R. K.** (1989). "A new technique for fully autonomous and efficient 3D robotics hand/eye calibration." *IEEE Transactions on Robotics and Automation*, 5(3), 345-358.

2. **Park, F. C., & Martin, B. J.** (1994). "Robot sensor calibration: solving AX= XB on the Euclidean group." *IEEE Transactions on Robotics and Automation*, 10(5), 717-721.

3. **Horaud, R., & Dornaika, F.** (1995). "Hand-eye calibration." *The International Journal of Robotics Research*, 14(3), 195-210.

4. **Daniilidis, K.** (1999). "Hand-eye calibration using dual quaternions." *The International Journal of Robotics Research*, 18(3), 286-298.

5. **OpenCV Documentation**: [Hand-Eye Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

---

## 附录：数学符号表

| 符号 | 含义 |
|------|------|
| $\mathbf{T}$ | 齐次变换矩阵 ($4 \times 4$) |
| $\mathbf{R}$ | 旋转矩阵 ($3 \times 3$) |
| $\mathbf{t}$ | 平移向量 ($3 \times 1$) |
| ${}^A\mathbf{T}_B$ | 从坐标系 $B$ 到 $A$ 的变换 |
| $\mathbf{q}$ | 四元数 ($4 \times 1$) |
| $\otimes$ | 四元数乘法 |
| $[\mathbf{v}]_\times$ | 向量的反对称矩阵 |
| $SO(3)$ | 三维旋转群 |
| $SE(3)$ | 三维刚体运动群 |
| $\|\cdot\|_2$ | 欧几里得范数 |
| $\|\cdot\|_F$ | Frobenius 范数 |
