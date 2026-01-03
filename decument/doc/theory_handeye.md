# 理论推导：手眼标定（Eye-in-Hand / Eye-to-Hand）与一致性评估

本章对应 3. 的“手眼标定设计思路/实现思路”，并与仓库实现一致：

- `vision/handeye_calibration_eyeinhand.py`
- `vision/handeye_calibration_eyetohand.py`
- `vision/handeye_utils.py`


手眼标定要解决的是把“相机看到的几何”与“机器人末端的几何”对齐：通过多组运动与观测，求一个固定外参 $X$，使两条链路在同一坐标系下自洽。

- Eye-in-Hand：用 $AX=XB$ 求 $X=T_{CG}$
- Eye-to-Hand：求 $X=T_{CB}$，使标定板相对末端 $T_{TG}$ 在多次采样中保持一致

一致性评估则把“是否自洽”量化成平移（mm）与旋转（deg）误差。

## 程序设计结构

手眼标定在工程上分三步：

1) 采集：同步记录 $T_{GB}^{(i)}$ 与 $T_{TC}^{(i)}$
2) 求解：用 $AX=XB$（Eye-in-Hand）或“标定板相对末端恒定”（Eye-to-Hand）得到外参 $X$
3) 评估：把 $X$ 代回数据集，输出平移/旋转一致性误差（mm/deg）

## 脚本作用

- `vision/handeye_calibration_eyeinhand.py`：采集与求解 $T_{CG}$。
- `vision/handeye_calibration_eyetohand.py`：采集与求解 $T_{CB}$。
- `vision/handeye_utils.py`：一致性评估与报告输出。

## 方法作用

- Eye-in-Hand 评估：`evaluate_eye_in_hand_consistency(...)`（构造 $A,B$ 并统计 $\Delta=(AX)(XB)^{-1}$）。
- Eye-to-Hand 评估：`evaluate_eye_to_hand_consistency(...)`（构造 $T_{TG}^{(i)}$ 并两两比较）。
- 报告输出：`print_consistency_report(...)`（把误差统计转成“质量等级”文本）。

## 1. 坐标系与符号

- $\{B\}$：基座（base）
- $\{G\}$：末端（gripper/end-effector）
- $\{C\}$：相机（camera）
- $\{T\}$：标定板（target）

变换记号：

- $T_{ab}$ 表示“坐标系 $\{b\}$ 到 $\{a\}$ 的齐次变换”（实现里使用 $4\times4$ 矩阵）。

## 2. Eye-in-Hand：AX = XB

场景：相机安装在末端。

未知量：

$$
X = T_{CG}

$$

采集第 $i$ 次：

- 末端在基座下：$T_{GB}^{(i)}$
- 标定板在相机下（PnP 得到）：$T_{TC}^{(i)}$

对相邻两次 $(i,i+1)$ 构造：

$$
A = T_{GB}^{(i+1)}\,(T_{GB}^{(i)})^{-1}

$$

$$
B = (T_{TC}^{(i+1)})^{-1}\,T_{TC}^{(i)}

$$

则满足：

$$
A X = X B

$$

仓库一致性评估（`vision/handeye_utils.py::evaluate_eye_in_hand_consistency`）做的是：

$$
\Delta = (A X)\,(X B)^{-1}

$$

并统计：

- 平移误差：$\|\Delta_{0:3,3}\|$（换算 mm）
- 旋转误差：$\|\mathrm{rotvec}(\Delta_{0:3,0:3})\|$（换算 deg）

## 3. Eye-to-Hand：标定板相对末端恒定

场景：相机固定在环境（不动），标定板固定在末端。

未知量：

$$
X = T_{CB}

$$

对每次采集 $i$，标定板相对末端应为常量：

$$
T_{TG}^{(i)} = (T_{GB}^{(i)})^{-1}\,T_{CB}\,T_{TC}^{(i)}

$$

若标定正确，则不同 $i$ 的 $T_{TG}^{(i)}$ 应非常接近。

仓库一致性评估（`vision/handeye_utils.py::evaluate_eye_to_hand_consistency`）做的是两两比较：

$$
\Delta_{ij}=T_{TG}^{(i)}\,(T_{TG}^{(j)})^{-1}

$$

并统计平移/旋转误差的均值与最大值。
