# Stage 1 交付文档（GitBook）

本书用于完成 `decument/task.txt` 中 Stage 1 的前三个任务交付（1./2./3.），内容严格对应本仓库当前代码。

## 理论框架

本书按“可复现闭环”的思路组织：先讲清楚每一章的理论框架（为什么要这么做/变量如何定义），再给出程序结构（代码怎么分层/数据怎么流动），最后把结构落到具体脚本与关键方法（怎么跑/调用链是什么）。

Stage 1 的核心闭环是：控制（关节/末端）→ 交互（键盘/JoyCon）→ 标定（相机/手眼/同步）→ 采集（raw/HDF5）→ 校验（inspect）。

## 程序设计结构

- 文档结构：`README.md`（本页）+ `SUMMARY.md`（目录）+ 每章独立 Markdown
- 代码结构（按分层）：
	- 驱动与控制：`driver/`
	- 运动学与 IK：`ik/`
	- 人机交互脚本：根目录 `*_control*.py`
	- 标定与视觉：`vision/`
	- 数据集与格式：`RDT/`

## 脚本作用

- 1.（SDK 验证）：`arm_keyboard_control.py`、`ik_keyboard_realtime.py`
- 2.（遥操作）：`joycon_ik_control_py.py`（JoyCon），以及键盘脚本
- 3.（采集与检查）：`RDT/collect_rdt_dataset_teleop.py`、`RDT/build_rdt_hdf5_from_raw.py`、`RDT/inspect_rdt_hdf5.py`
- 3.（标定与验证）：`vision/calibrate_camera.py`、`vision/handeye_calibration_*.py`、`vision/track_blue_circle_eyetohand.py`

## 方法作用（贯穿全书的“最小公共接口”）

如果你只想快速读懂调用链，优先理解这些方法的输入/输出：

- `driver.ftservo_controller.ServoController.fast_move_to_pose/read_servo_positions/limit_position/close`
- `ik.robot.Robot.fkine/ikine_LM/q_to_servo_targets/read_joint_angles`
- `RDT/rdt_hdf5.py` 中 `UnifiedVector` 与 `RDTHDF5EpisodeWriter.finalize_action_chunks`

## 你将得到什么（面向“可复现”的结果）

- 一条从零开始可跑通的复现主线：环境 → SDK → 遥操作 → 标定 → 数据采集 → HDF5 校验
- 每一步的输入/输出与验收口径（该看什么日志、应生成哪些文件、失败时优先查哪里）
- 每章采用“理论背景 → 方法/流程 → 实现细节 → 验收与排错”的论文式结构（先讲清楚为什么/是什么，再讲怎么做）

## 读者假设（前置知识/设备）

- Linux 环境（能访问串口 `/dev/tty*`、相机 `/dev/video*`、JoyCon `/dev/hidraw*`）
- 已接好机械臂与相机；知道自己在用左臂还是右臂配置（`driver/left_arm.json` / `driver/right_arm.json`）
- 目标是复现 Stage1 交付：能演示、能采集、能解释

## 快速开始（生成静态书）

在 `decument/` 目录下执行：

```bash
gitbook build
```

默认输出到 `decument/_book/`。

如你的环境支持预览服务（可选）：

```bash
gitbook serve
```

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

## 最快复现路径（建议严格按顺序）

1) 环境与权限：按 `env_setup.md` 解决串口/相机/JoyCon 权限与 Python 依赖
2) 1. SDK：先跑 `arm_keyboard_control.py`（关节级，验证方向/限位），再跑 `ik_keyboard_realtime.py`（末端级，验证 FK/IK）
3) 2. 遥操作：跑 `joycon_ik_control_py.py`（形成可录制演示视频的稳定控制）
4) 3. 标定：先相机内参，再手眼（必要时做一致性评估），最后用验证脚本做闭环演示
5) 3. 采集：采 raw → raw→HDF5 → inspect 校验张量形状与抽样可视化

## 常见坑（跑不通先查这几个）

- 跑错配置：左/右臂必须匹配 `--port` 与 `--config`，否则方向/零位会错
- 相机内参标定：采集会生成 `session_*/` 子目录，后续标定必须指向该 session（或直接用 `--all` 一步完成）
- “看起来没更新”：GitBook 输出在 `decument/_book/`，浏览器要强刷（`Ctrl+F5`）

## 建议视频录制清单（按任务产出物）

- 1.：展示“回中位 → 运动 → 读取关节/末端位姿 → 正常退出”的完整流程
- 2.：展示键盘或 JoyCon 的实时控制（含夹爪）
- 3.：展示采集一次 episode（含三相机预览）→ raw→hdf5 合成 → inspect 校验
