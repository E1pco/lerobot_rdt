# 3.2 理论：手眼标定（Hand-Eye Calibration）

## 1. 引言

### 1.1 什么是手眼标定？

**手眼标定**（Hand-Eye Calibration）是机器人视觉系统中的核心问题：确定相机与机器人末端执行器（或基座）之间的**空间变换关系**。

简单来说，手眼标定回答的问题是：

> "相机看到的目标在哪里？" → "机器人应该去哪里抓取？"

这个转换需要一个精确的**外参矩阵**，它描述了相机坐标系与机器人坐标系之间的刚体变换。

### 1.2 为什么需要手眼标定？

| 场景 | 需求 |
|-----|------|
| 视觉引导抓取 | 相机检测物体位置 → 转换为机器人可执行的目标位姿 |
| 视觉伺服 | 实时跟踪目标 → 闭环控制机器人运动 |
| 数据采集 | 记录图像时同步记录对应的机器人状态 |
| 三维重建 | 多视角融合时需要精确的相机位姿 |

### 1.3 两种标定配置

根据相机安装位置，手眼标定分为两种配置：

```
┌─────────────────────────────────────────────────────────────────┐
│                     Eye-in-Hand（眼在手上）                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     [基座 B]                                                    │
│        │                                                        │
│        │ T_GB（机器人运动学）                                    │
│        ▼                                                        │
│     [末端 G] ◄──── T_CG（待标定）──── [相机 C]                   │
│                                          │                      │
│                                          │ T_TC（PnP求解）       │
│                                          ▼                      │
│                                      [标定板 T]（固定在环境中）  │
│                                                                 │
│  特点：相机随末端移动，标定板固定                                │
│  求解：X = T_CG（相机相对于末端的变换）                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Eye-to-Hand（眼在手外）                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     [基座 B] ◄──── T_CB（待标定）──── [相机 C]（固定在环境中）   │
│        │                                  │                     │
│        │ T_GB（机器人运动学）              │ T_TC（PnP求解）      │
│        ▼                                  ▼                     │
│     [末端 G] ────── T_TG（固定）────► [标定板 T]                 │
│                                                                 │
│  特点：相机固定在环境中，标定板随末端移动                        │
│  求解：X = T_CB（相机相对于基座的变换）                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 坐标系与符号定义

### 2.1 坐标系

| 符号 | 名称 | 说明 |
|-----|------|------|
| $\{B\}$ | 基座坐标系 (Base) | 机器人世界坐标系，通常固定不动 |
| $\{G\}$ | 末端坐标系 (Gripper) | 机器人末端执行器坐标系 |
| $\{C\}$ | 相机坐标系 (Camera) | 相机光心坐标系 |
| $\{T\}$ | 标定板坐标系 (Target) | 棋盘格/AprilTag 的参考坐标系 |

### 2.2 变换矩阵

变换矩阵 $T_{ab} \in SE(3)$ 表示从坐标系 $\{b\}$ 到坐标系 $\{a\}$ 的刚体变换：

$$
T_{ab} = \begin{bmatrix} R_{ab} & t_{ab} \\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

其中：
- $R_{ab} \in SO(3)$：旋转矩阵（3×3）
- $t_{ab} \in \mathbb{R}^3$：平移向量

**物理意义**：如果点 $p$ 在坐标系 $\{b\}$ 中的坐标为 $p_b$，则在坐标系 $\{a\}$ 中的坐标为：

$$
p_a = T_{ab} \cdot p_b
$$

### 2.3 变换链

变换可以链式组合：

$$
T_{ac} = T_{ab} \cdot T_{bc}
$$

读作："从 $c$ 到 $a$" = "从 $b$ 到 $a$" × "从 $c$ 到 $b$"

---

## 3. Eye-in-Hand 标定

### 3.1 问题建模

**已知**：
- $T_{GB}^{(i)}$：第 $i$ 次采样时末端相对于基座的位姿（从机器人正运动学获得）
- $T_{TC}^{(i)}$：第 $i$ 次采样时标定板相对于相机的位姿（从 PnP 算法获得）

**未知**：
- $X = T_{CG}$：相机相对于末端的位姿（待标定的外参）

### 3.2 AX = XB 方程推导

由于标定板固定在环境中，其在基座坐标系下的位置 $T_{TB}$ 恒定：

$$
T_{TB} = T_{GB}^{(i)} \cdot T_{CG} \cdot T_{TC}^{(i)} = \text{常数}
$$

对于相邻两次采样 $(i)$ 和 $(j)$：

$$
T_{GB}^{(i)} \cdot T_{CG} \cdot T_{TC}^{(i)} = T_{GB}^{(j)} \cdot T_{CG} \cdot T_{TC}^{(j)}
$$

整理得：

$$
\underbrace{(T_{GB}^{(j)})^{-1} \cdot T_{GB}^{(i)}}_{A} \cdot \underbrace{T_{CG}}_{X} = \underbrace{T_{CG}}_{X} \cdot \underbrace{T_{TC}^{(j)} \cdot (T_{TC}^{(i)})^{-1}}_{B}
$$

即经典的 **AX = XB** 方程：

$$
\boxed{A X = X B}
$$

其中：
- $A = (T_{GB}^{(j)})^{-1} \cdot T_{GB}^{(i)}$：机器人末端的相对运动
- $B = T_{TC}^{(j)} \cdot (T_{TC}^{(i)})^{-1}$：标定板在相机下的相对运动

### 3.3 求解方法

经典求解方法包括：

| 方法 | 特点 | OpenCV 函数 |
|-----|------|-------------|
| Tsai-Lenz | 分离旋转和平移求解 | `cv2.CALIB_HAND_EYE_TSAI` |
| Park | 李群方法 | `cv2.CALIB_HAND_EYE_PARK` |
| Horaud | 四元数方法 | `cv2.CALIB_HAND_EYE_HORAUD` |
| Andreff | 同时估计旋转和平移 | `cv2.CALIB_HAND_EYE_ANDREFF` |
| Daniilidis | 对偶四元数方法 | `cv2.CALIB_HAND_EYE_DANIILIDIS` |

**OpenCV 调用示例**：

```python
import cv2
import numpy as np

# R_gripper2base: list of 3x3 rotation matrices
# t_gripper2base: list of 3x1 translation vectors
# R_target2cam: list of 3x3 rotation matrices  
# t_target2cam: list of 3x1 translation vectors

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

# 构造 4x4 齐次变换矩阵
T_CG = np.eye(4)
T_CG[:3, :3] = R_cam2gripper
T_CG[:3, 3] = t_cam2gripper.flatten()
```

---

## 4. Eye-to-Hand 标定

### 4.1 问题建模

**已知**：
- $T_{GB}^{(i)}$：第 $i$ 次采样时末端相对于基座的位姿
- $T_{TC}^{(i)}$：第 $i$ 次采样时标定板相对于相机的位姿

**未知**：
- $X = T_{CB}$：相机相对于基座的位姿（待标定的外参）

### 4.2 约束条件

由于标定板固定在末端，其相对于末端的位置 $T_{TG}$ 恒定：

$$
T_{TG} = (T_{GB}^{(i)})^{-1} \cdot T_{CB} \cdot T_{TC}^{(i)} = \text{常数}
$$

### 4.3 AX = XB 形式

同样可以转化为 AX = XB 形式。定义：

$$
A = T_{GB}^{(j)} \cdot (T_{GB}^{(i)})^{-1}
$$

$$
B = T_{TC}^{(j)} \cdot (T_{TC}^{(i)})^{-1}
$$

则：

$$
A X = X B, \quad X = T_{CB}
$$

**OpenCV 调用**（注意参数顺序）：

```python
R_base2gripper = [R.T for R in R_gripper2base]  # 取逆
t_base2gripper = [-R.T @ t for R, t in zip(R_gripper2base, t_gripper2base)]

R_cam2base, t_cam2base = cv2.calibrateHandEye(
    R_base2gripper, t_base2gripper,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)
```

---

## 5. 标定质量评估

### 5.1 一致性误差

标定完成后，需要评估外参的精度。核心思想：**如果外参正确，则所有采样应该自洽**。

#### Eye-in-Hand 评估

计算残差矩阵：

$$
\Delta_{ij} = (A_{ij} \cdot X) \cdot (X \cdot B_{ij})^{-1}
$$

理想情况下 $\Delta_{ij} = I$（单位矩阵）。

#### Eye-to-Hand 评估

计算标定板相对末端的位姿：

$$
T_{TG}^{(i)} = (T_{GB}^{(i)})^{-1} \cdot T_{CB} \cdot T_{TC}^{(i)}
$$

比较不同采样之间的差异：

$$
\Delta_{ij} = T_{TG}^{(i)} \cdot (T_{TG}^{(j)})^{-1}
$$

### 5.2 误差度量

从残差矩阵 $\Delta$ 中提取误差：

**平移误差**：

$$
e_t = \| t_\Delta \| \quad (\text{单位: mm})
$$

**旋转误差**（使用旋转向量）：

$$
e_r = \| \text{rotvec}(R_\Delta) \| \cdot \frac{180}{\pi} \quad (\text{单位: deg})
$$

### 5.3 质量等级

| 等级 | 平移误差 (mm) | 旋转误差 (deg) | 适用场景 |
|-----|--------------|---------------|---------|
| 优秀 | < 1.0 | < 0.5 | 精密装配 |
| 良好 | 1.0 - 3.0 | 0.5 - 1.0 | 一般抓取 |
| 可接受 | 3.0 - 5.0 | 1.0 - 2.0 | 粗定位 |
| 需改进 | > 5.0 | > 2.0 | 需重新标定 |

---

## 6. 标定最佳实践

### 6.1 数据采集建议

1. **采样数量**：至少 15-20 组，建议 30+ 组
2. **位姿多样性**：
   - 覆盖工作空间的不同区域
   - 末端旋转角度变化 > 60°
   - 避免只做平移运动
3. **标定板要求**：
   - 平整无弯曲
   - 特征点清晰可见
   - 占据图像 30%-70% 面积

### 6.2 常见问题

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 平移误差大 | 标定板检测不准 | 提高图像质量，更大的标定板 |
| 旋转误差大 | 位姿变化不足 | 增加旋转多样性 |
| 结果不稳定 | 数据量不足 | 增加采样数量 |
| 无法收敛 | 数据存在离群点 | 检查并剔除异常数据 |

### 6.3 验证方法

标定完成后，可通过以下方式验证：

1. **重投影验证**：将已知3D点投影到图像，检查与检测点的偏差
2. **闭环验证**：让机器人移动到视觉检测的目标位置，检查实际偏差
3. **交叉验证**：用部分数据标定，用剩余数据验证

---

## 7. 代码实现参考

### 7.1 Eye-in-Hand 完整流程

```python
import cv2
import numpy as np

def calibrate_eye_in_hand(robot_poses, camera_poses):
    """
    Eye-in-Hand 手眼标定
    
    Args:
        robot_poses: list of 4x4 T_GB matrices (gripper to base)
        camera_poses: list of 4x4 T_TC matrices (target to camera)
    
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
        method=cv2.CALIB_HAND_EYE_TSAI
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

### 7.2 仓库对应文件

| 功能 | 文件 |
|-----|------|
| Eye-in-Hand 标定 | `vision/handeye_calibration_eyeinhand.py` |
| Eye-to-Hand 标定 | `vision/handeye_calibration_eyetohand.py` |
| 一致性评估 | `vision/handeye_utils.py` |

---

## 8. 参考文献

1. Tsai, R. Y., & Lenz, R. K. (1989). A new technique for fully autonomous and efficient 3D robotics hand/eye calibration. *IEEE Transactions on Robotics and Automation*.
2. Park, F. C., & Martin, B. J. (1994). Robot sensor calibration: Solving AX=XB on the Euclidean group. *IEEE Transactions on Robotics and Automation*.
3. Horaud, R., & Dornaika, F. (1995). Hand-eye calibration. *The International Journal of Robotics Research*.
4. OpenCV Documentation: [Hand-Eye Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b)
