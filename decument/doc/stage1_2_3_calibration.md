# 2.3 相机与手眼标定（代码 + 流程）

理论推导：`theory_handeye.md`

本章对应 2.3 产出物中的“手眼标定、数据时序同步源代码 + 设计手册”的标定部分。

## 1. 相机内参标定

脚本：`vision/calibrate_camera.py`

- 采集标定图：

```bash
python vision/calibrate_camera.py --capture --camid 2 --image-folder ./vision/calib_images_right
```

- 用已有图像执行标定：

```bash
python vision/calibrate_camera.py --calibrate --image-folder ./vision/calib_images_right
```

- 一键（采集+标定）：

```bash
python vision/calibrate_camera.py --all --camid 2 --image-folder ./vision/calib_images_right
```

输出：脚本会在 session 目录生成 `camera_intrinsics*.yaml`，并打印重投影误差分析。

## 2. 手眼标定（棋盘格）

> 注意：本仓库标定代码在 `vision/`。

### 2.1 Eye-in-Hand（相机在末端）

脚本：`vision/handeye_calibration_eyeinhand.py`

- 采集：

```bash
python vision/handeye_calibration_eyeinhand.py --collect --video 0 --port /dev/left_arm
```

- 计算：

```bash
python vision/handeye_calibration_eyeinhand.py --calibrate
```

### 2.2 Eye-to-Hand（相机固定在环境）

脚本：`vision/handeye_calibration_eyetohand.py`

- 采集：

```bash
python vision/handeye_calibration_eyetohand.py --collect --video 0 --port /dev/left_arm
```

- 计算：

```bash
python vision/handeye_calibration_eyetohand.py --calibrate
```

### 2.3 一致性评估

工具：`vision/handeye_utils.py`
- Eye-in-Hand：`evaluate_eye_in_hand_consistency()`
- Eye-to-Hand：`evaluate_eye_to_hand_consistency()`

采集与计算脚本会在控制台输出一致性报告（平移 mm / 旋转 deg）。

误差定义（与实现一致）：
- 旋转误差：将 $\Delta R$ 转为旋转向量 `rotvec`，取其范数并换算为度。
- 平移误差：取 $\Delta t$ 的欧氏范数并换算为 mm。

## 3. 标定结果验证（蓝色圆追踪闭环）

脚本：`vision/track_blue_circle_eyetohand.py`

- 作用：检测蓝色圆 → PnP 求 $T_{target}^{cam}$ → 用手眼结果换算到基座系 → 驱动机械臂靠近目标。
- 运行：

```bash
python vision/track_blue_circle_eyetohand.py
```

建议录屏：同时录到相机画面（检测/坐标叠字）与机械臂运动，便于证明标定闭环有效。
