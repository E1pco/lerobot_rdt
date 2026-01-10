# Stage 1 交付文档（GitBook）

本书用于完成 Stage 1 的任务交付

## 程序设计结构

- 文档结构：`README.md`（本页）+ `SUMMARY.md`（目录）+ 每章独立 Markdown
- 代码结构（按分层）：
  - 驱动与控制：`driver/`
  - 运动学与 IK：`ik/`
  - 人机交互脚本：根目录 `*_control*.py`
  - 标定与视觉：`vision/`
  - 数据集与格式：`RDT/`
  - 硬件加速：外部仓库 `Hardware-Accelerated_System_for_IK_Solution_Based_on_LeRobot/`

## 脚本作用

- 1.（SDK 验证）：`arm_keyboard_control.py`、`ik_keyboard_realtime.py`
- 2.（遥操作）：`joycon_ik_control_py.py`（JoyCon），以及键盘脚本
- 3.（采集与检查）：`RDT/collect_rdt_dataset_teleop.py`、`RDT/build_rdt_hdf5_from_raw.py`、`RDT/inspect_rdt_hdf5.py`
- 3.（标定与验证）：`vision/calibrate_camera.py`、`vision/handeye_calibration_*.py`、`vision/track_blue_circle_eyetohand.py`- 5.（FPGA 加速）：`notebook/SO101_Hardware_IK_Demo.ipynb`、`notebook/SO101_IK_HW_vs_Python.ipynb`（需 PYNQ-Z2）

## 章节导航

- Stage 1 总览：`stage1_overview.md`
- 环境与依赖：`env_setup.md`
- 1. 自设计 SDK：`stage1_1_sdk.md`

  - API 速查：`stage1_1_sdk_api.md`
- 2. 人机交互（键盘/JoyCon）：`stage1_2_hci_teleop.md`

  - 控制映射速查：`stage1_2_controls.md`
- 3. 数据集构建（RDT）：`stage1_3_dataset.md`

  - 相机与手眼标定：`stage1_3_calibration.md`
  - 时序同步说明：`stage1_3_timesync.md`

理论与源码解析会在对应章节的开头给出索引：

- 运动学/IK 理论：`stage1_1_sdk.md`
- 手眼标定理论：`theory_handeye.md`
- RDT 数据格式理论：`theory_rdt_format.md`
