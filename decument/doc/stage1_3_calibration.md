# 3. 相机与手眼标定（代码 + 流程）

理论推导：`theory_handeye.md`

本章对应 3. 产出物中的“手眼标定、数据时序同步源代码 + 设计手册”的标定部分。


手眼标定的核心是把“相机观测到的标定板位姿”（来自 PnP）与“机械臂末端位姿”（来自 FK/编码器）关联起来，求解固定的坐标变换：

- Eye-in-Hand：相机装在末端，求 $T_{CG}$（相机 $\{C\}$ 到夹爪/末端 $\{G\}$）
- Eye-to-Hand：相机固定在环境，求 $T_{CB}$（相机 $\{C\}$ 到基座 $\{B\}$）

在采集多组姿态后，通过经典形式 $AX=XB$ 求解 $X$（详见：`theory_handeye.md`）。

本页的目标是把“理论变量”落到“脚本输入/输出”：你将明确每一步会生成哪些文件、应达到什么误差水平、失败时优先查哪里。

## 程序设计结构

标定链路可以抽象为“内参 → PnP → 手眼（AX=XB）→ 评估 → 验证闭环”：

1) **相机内参**：用棋盘格角点拟合得到 $K$ 与畸变；这是 PnP 与手眼的共同前提
2) **PnP 位姿**：每次观测把棋盘格/目标在相机坐标系下的位姿估计出来（生成 $T_{TC}$ 或等价量）
3) **手眼求解**：结合机器人末端位姿 $T_{GB}$ 与观测位姿，解 $AX=XB$ 得到固定外参（Eye-in-Hand 的 $T_{CG}$ 或 Eye-to-Hand 的 $T_{CB}$）
4) **一致性评估**：用多组采样对求解结果做平移/旋转误差统计
5) **闭环验证**：把视觉检测结果通过手眼外参换算到机器人坐标系，驱动机械臂做“看得见→够得着”的动作

## 脚本作用

- `vision/calibrate_camera.py`：相机内参标定（采集/计算/一键 all），输出 `camera_intrinsics.yaml` 与报告。
- `vision/handeye_calibration_eyeinhand.py`：Eye-in-Hand 数据采集与解算，输出 `handeye_result.yaml`（含 $T_{CG}$）。
- `vision/handeye_calibration_eyetohand.py`：Eye-to-Hand 数据采集与解算，输出 `handeye_result.yaml`（含 $T_{CB}$）。
- `vision/handeye_utils.py`：一致性评估与误差统计（用于把标定质量量化进报告）。
- `vision/track_blue_circle_eyetohand.py`：闭环验证脚本（用于证明外参“可用且有效”）。

## 方法作用

### 相机内参标定 (`calibrate_camera.py`)

- **`detect_checkerboard(image)`**
  - **作用**：角点检测。
  - **细节**：调用 `cv2.findChessboardCorners` 寻找棋盘格内角点，若成功则进一步调用 `cv2.cornerSubPix` 进行亚像素级精细化。
- **`calibrate_intrinsics(objpoints, imgpoints, image_size)`**
  - **作用**：计算内参。
  - **细节**：调用 `cv2.calibrateCamera`，输入世界坐标系下的点（假设在 Z=0 平面）和图像坐标系下的点，输出内参矩阵 $K$ 和畸变系数 $D$。
- **`save_calibration_result(...)`**
  - **作用**：结果持久化。
  - **细节**：将 $K$ 和 $D$ 保存为 YAML 文件，并生成包含重投影误差统计的文本报告。

### 手眼标定 (`handeye_calibration_*.py`)

- **`get_board_pose_in_camera(image, K, D)`**
  - **作用**：PnP 位姿估计。
  - **细节**：检测棋盘格角点，利用已知的棋盘格尺寸和内参，调用 `cv2.solvePnP` 计算标定板相对于相机的位姿 $T_{target}^{cam}$。
- **`record_sample(robot_pose, image)`**
  - **作用**：采集数据对。
  - **细节**：同步记录当前的机械臂末端位姿 $T_{end}^{base}$ 和相机图像（或计算出的 $T_{target}^{cam}$），存入列表用于后续解算。
- **`cv2.calibrateHandEye(...)`**
  - **作用**：求解手眼方程 $AX=XB$。
  - **细节**：输入一系列的 $R_{gripper}^{base}, t_{gripper}^{base}$ 和 $R_{target}^{cam}, t_{target}^{cam}$，输出手眼变换矩阵（Eye-in-Hand 为 $T_{cam}^{gripper}$，Eye-to-Hand 为 $T_{cam}^{base}$）。常用算法为 `CALIB_HAND_EYE_TSAI` 或 `CALIB_HAND_EYE_PARK`。

### 验证与工具 (`handeye_utils.py` / `track_blue_circle.py`)

- **`compute_consistency_error(handeye_result, samples)`**
  - **作用**：一致性评估。
  - **细节**：利用求得的手眼矩阵，将所有观测统一转换到同一坐标系下（如基座），计算标定板位置的标准差，衡量标定的一致性精度。
- **`transform_point(point_cam, T_handeye, robot_pose)`**
  - **作用**：坐标变换（闭环验证）。
  - **细节**：将视觉识别到的目标点（相机系）变换为机械臂目标点（基座系），用于驱动机械臂去“够”目标。

## 1. 相机内参标定

**目的**：获取相机的内参矩阵（Intrinsics）和畸变系数（Distortion Coefficients），这是后续 PnP 解算和手眼标定的基础。

**脚本**：`vision/calibrate_camera.py`

> 说明：该脚本在采集模式下会自动创建 `session_YYYYmmdd_HHMMSS/` 子目录，并把图片与标定结果都保存到该 session 内。
> 因此如果你选择“分两步”（先采集后标定），第二步的 `--image-folder` 必须指向具体的 session 目录；否则会因为找不到图片而失败。

### 1.1 采集标定图

使用棋盘格标定板，在不同角度和距离下拍摄多张图片。

```bash
# --capture: 进入采集模式
# --camid: 相机设备 ID (如 /dev/video2 对应 2)
# --image-folder: 图片保存路径
python vision/calibrate_camera.py --capture --camid 2 --image-folder ./vision/calib_images_right
```

*操作提示：按 `s` 键保存当前帧，按 `q` 键退出。采集结束后，终端会打印本次 session 目录路径。*

### 1.2 执行标定计算

读取采集到的图片，检测角点并计算内参。

如果你刚完成 1.1 采集，先找到最新的 session 目录（示例）：

```bash
ls -dt ./vision/calib_images_right/session_* | head -1
```

然后把该目录作为 `--image-folder` 输入：

```bash
# --calibrate: 进入计算模式
python vision/calibrate_camera.py --calibrate --image-folder ./vision/calib_images_right/session_YYYYmmdd_HHMMSS
```

### 1.3 一键模式（采集+标定）

如果想在一个流程内完成：

```bash
python vision/calibrate_camera.py --all --camid 2 --image-folder ./vision/calib_images_right
```

**输出产物**：
脚本会在 session 目录下生成（以实际打印为准）：

- `camera_intrinsics.yaml`：相机内参矩阵 $K$ 与畸变系数
- `camera_intrinsics_report.txt`：详细报告（含每张图的重投影误差）
- `extrinsics.yaml` / `extrinsics.npy`：每张标定图对应的 $T_{target}^{cam}$（用于 PnP 精度诊断/复现）
- `undistorted_test.jpg`：去畸变效果对比用

**关键检查点**：

- 报告中的平均重投影误差建议 < 0.5 px（更小更好）
- `undistorted_test.jpg` 目视检查畸变是否明显减轻

## 2. 手眼标定（棋盘格）

**目的**：求解相机坐标系 $\{C\}$ 与机械臂末端坐标系 $\{G\}$（Eye-in-Hand）或基座坐标系 $\{B\}$（Eye-to-Hand）之间的变换矩阵。

> 注意：本仓库标定代码在 `vision/` 目录下。

### 2.1 Eye-in-Hand（相机在末端）

**场景**：相机安装在机械臂末端，随机械臂运动。求解 $T_{CG}$。

**脚本**：`vision/handeye_calibration_eyeinhand.py`

1. **数据采集**：

   ```bash
   # --collect: 采集模式
   # --video: 相机 ID
   # --port: 机械臂串口
   python vision/handeye_calibration_eyeinhand.py --collect --video 0 --port /dev/left_arm
   ```

   *流程*：

   - 移动机械臂到不同姿态（保持标定板在视野内）。
   - 脚本会自动记录：当前机械臂末端位姿 $T_{GB}$ + 当前相机拍摄的标定板位姿 $T_{TC}$（通过 PnP 解算）。
   - 建议采集 10-15 组数据，覆盖不同的旋转角度。
2. **解算**：

   ```bash
   # --calibrate: 计算模式
   python vision/handeye_calibration_eyeinhand.py --calibrate
   ```

   *输出*：`handeye_result.yaml`，包含 $T_{CG}$ 矩阵。

### 2.2 Eye-to-Hand（相机固定在环境）

**场景**：相机固定不动，拍摄机械臂末端（末端夹持标定板）。求解 $T_{CB}$。

**脚本**：`vision/handeye_calibration_eyetohand.py`

1. **数据采集**：

   ```bash
   python vision/handeye_calibration_eyetohand.py --collect --video 0 --port /dev/left_arm
   ```

   *注意*：此时标定板是固定在机械臂末端的，相机是不动的。
2. **解算**：

   ```bash
   python vision/handeye_calibration_eyetohand.py --calibrate
   ```

### 2.3 一致性评估与误差分析

**工具**：`vision/handeye_utils.py`

在执行标定计算时，脚本会自动调用一致性评估函数：

- Eye-in-Hand：`evaluate_eye_in_hand_consistency()`
- Eye-to-Hand：`evaluate_eye_to_hand_consistency()`

**原理**：
利用 $AX=XB$ 关系，检查每两组数据计算出的 $X$ 是否一致。

- **平移误差**：计算 $\Delta t$ 的欧氏距离（单位：mm）。
- **旋转误差**：计算 $\Delta R$ 的旋转向量模长（单位：deg）。

## 方法作用

本章的关键方法集中解决两件事：**把观测变成变换矩阵**、以及**评估这个矩阵是否可信**。

- PnP 位姿估计：把棋盘格角点与内参 $K$ 结合，求每帧的 $T_{target}^{cam}$（实现位于 `vision/` 脚本内部，输出会写到 session/report 里）。
- 手眼求解（AX=XB）：在采集到多组 $(A_i,B_i)$ 后求解固定外参 $X$，分别对应 Eye-in-Hand 的 $T_{CG}$ 与 Eye-to-Hand 的 $T_{CB}$。
- 一致性评估：
  - Eye-in-Hand：`evaluate_eye_in_hand_consistency()`
  - Eye-to-Hand：`evaluate_eye_to_hand_consistency()`
    输出平移（mm）与旋转（deg）误差，作为“标定是否可用”的量化指标。

## 3. 标定结果验证（蓝色圆追踪闭环）

脚本：`vision/track_blue_circle_eyetohand.py`

- 作用：检测蓝色圆 → PnP 求 $T_{target}^{cam}$ → 用手眼结果换算到基座系 → 驱动机械臂靠近目标。
- 运行：

```bash
python vision/track_blue_circle_eyetohand.py
```
