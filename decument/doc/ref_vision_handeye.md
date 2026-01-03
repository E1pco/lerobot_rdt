# 函数解析：标定与一致性评估（`vision/`）

本章对应 3. 的“手眼标定源代码 + 说明文档”，重点解析可复用的核心函数：

- `vision/handeye_utils.py`

一致性评估的理论目标是：在手眼标定得到 $T_{CG}

$ 或 $T_{CB}$ 后，用多组采样构造误差指标（mm/deg），回答“这个外参是否可信、是否可用于闭环验证/数据复现”。

## 程序设计结构

- 标定脚本（`vision/handeye_calibration_*.py`）：负责采集 $T_{GB}^{(i)}$ 与 $T_{TC}^{(i)}$ 并求解外参
- 评估函数（`vision/handeye_utils.py`）：负责把外参代入采样对/采样集，输出误差统计与报告

## 脚本作用

- `vision/handeye_calibration_eyeinhand.py` / `vision/handeye_calibration_eyetohand.py`：生成 `handeye_result.yaml`。
- `vision/handeye_utils.py`：在标定解算阶段或单独调用时输出一致性报告。
- `vision/track_blue_circle_eyetohand.py`：使用标定结果做闭环验证（把“评估好”变成“能用”）。

## 方法作用

下文逐个函数说明输入/输出与误差计算方式，便于你在报告中引用“误差如何定义”。

## 1. `evaluate_eye_in_hand_consistency(T_cam_gripper, T_gripper_base_list, T_target_cam_list)`

- 场景：Eye-in-Hand
- 输入：
  - `T_cam_gripper`：$T_{CG}$（标定结果）
  - `T_gripper_base_list`：每次采集的 $T_{GB}^{(i)}$
  - `T_target_cam_list`：每次采集的 $T_{TC}^{(i)}$（PnP 得到）
- 核心计算（相邻对）：
  - $A=T_{GB}^{(i+1)}(T_{GB}^{(i)})^{-1}$
  - $B=(T_{TC}^{(i+1)})^{-1}T_{TC}^{(i)}$
  - $\Delta=(AX)(XB)^{-1}$
- 输出：
  - `trans_errors`（mm）/ `rot_errors`（deg）及均值/最大值

## 2. `evaluate_eye_to_hand_consistency(T_cam_base, T_gripper_base_list, T_target_cam_list)`

- 场景：Eye-to-Hand
- 输入：
  - `T_cam_base`：$T_{CB}$（标定结果）
  - 列表同上
- 核心计算：
  - 每次求 $T_{TG}^{(i)}=(T_{GB}^{(i)})^{-1}T_{CB}T_{TC}^{(i)}$
  - 两两比较 $\Delta_{ij}=T_{TG}^{(i)}(T_{TG}^{(j)})^{-1}$
- 输出：
  - 同上（mm/deg）

## 3. `print_consistency_report(result, title='一致性误差')`

- 作用：把一致性评估结果打印为“平均/最大 + 质量等级”
- 质量阈值：实现里用经验阈值（mm/deg）分级

## 4. 与采集脚本的接口边界

- 标定脚本负责生成：
  - 内参：相机 `K/dist`
  - 手眼结果：$T_{CG}$ 或 $T_{CB}$
- 数据集采集脚本（3.）只负责记录控制与图像，不会自动推导外参；外参应作为实验配置写入报告。
