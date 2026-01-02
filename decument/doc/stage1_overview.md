# Stage 1 总览

本章给出 Stage 1（2.1/2.2/2.3）的目标拆解、仓库代码映射、推荐执行顺序与验收口径，便于你按“能演示 + 能复现 + 能交付文档”的方式推进。

## 1. 目标与里程碑（对应 `decument/task.txt`）

- 2.1 自设计 SDK：能不依赖 huggingface 官方 lerobot SDK，直接用 Python 控制机械臂并完成演示。
- 2.2 人机交互：在 2.1 基础上实现键盘/遥操作控制，并给出控制误差分析。
- 2.3 数据集构建：在 2.2 基础上采集数据集（raw/HDF5），并补齐手眼标定与时序同步说明。

## 2. 仓库代码映射（你需要交付的“源代码”在哪里）

- SDK（2.1）
  - 硬件驱动：`driver/ftservo_driver.py`
  - 控制抽象：`driver/ftservo_controller.py`
  - 配置：`driver/servo_config.json`、`driver/left_arm.json`、`driver/right_arm.json`
  - 运动学/IK：`ik/robot.py`（及 `ik/` 目录）

- 人机交互（2.2）
  - 键盘关节控制：`arm_keyboard_control.py`
  - 键盘 IK 控制：`ik_keyboard_realtime.py`
  - JoyCon IK 控制：`joycon_ik_control_py.py`
  - JoyCon 输入库：`joyconrobotics/`

- 数据集（2.3）
  - 采集：`RDT/collect_rdt_dataset_teleop.py`
  - raw→hdf5：`RDT/build_rdt_hdf5_from_raw.py`
  - 检查：`RDT/inspect_rdt_hdf5.py`
  - 格式定义：`RDT/rdt_hdf5.py`
  - 标定：`vision/`（相机内参、手眼标定、验证脚本）

## 3. 推荐执行顺序（最省时间的路径）

1) 先跑通 2.1（把“关节步数 ↔ 关节角 ↔ 末端位姿”链路打通）：
- 先用 `arm_keyboard_control.py` 做关节步进，验证串口/限位/回中位与每个关节方向。
- 再用 `ik_keyboard_realtime.py` 做末端 IK 控制，验证 FK/IK、mask 与步数映射。

2) 再跑通 2.2（让控制稳定到“可录屏”）：
- 用 `joycon_ik_control_py.py` 做单臂 JoyCon 遥操作，重点验证：速度档、Home 重置、夹爪开合。

3) 最后跑通 2.3（先标定，后采集，再校验）：
- 先做相机内参标定：`vision/calibrate_camera.py`（推荐直接用 `--all`，会在 `session_*/` 下落盘结果）。
- 再做手眼标定：`vision/handeye_calibration_eyeinhand.py` / `vision/handeye_calibration_eyetohand.py`，并用一致性评估输出误差报告。
- 用 `vision/track_blue_circle_eyetohand.py` 做闭环验证（这是“标定有效”的最强证据）。
- 最后采集：`RDT/collect_rdt_dataset_teleop.py` → raw→HDF5（`RDT/build_rdt_hdf5_from_raw.py`）→ inspect（`RDT/inspect_rdt_hdf5.py`）。

## 4. 验收视频建议（每条视频应包含的“必拍镜头”）

- 2.1 视频：终端启动日志 + 机械臂回中 + 2~3 个动作 + 读取位姿/关节 + 退出后串口关闭。
- 2.2 视频：JoyCon/键盘实时控制 + 夹爪动作 + 发生漂移时如何 reset（如 `Home`）+ 退出。
- 2.3 视频：采集时三相机预览画面 + 完成一个 episode 落盘 + raw→hdf5 + inspect 输出。
