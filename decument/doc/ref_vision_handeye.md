# 函数解析：标定与一致性评估（`vision/`）

本章对应 2.3 的“手眼标定源代码 + 说明文档”，重点解析可复用的核心函数：
- `vision/handeye_utils.py`

更完整的推导见：`theory_handeye.md`。

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
- 数据集采集脚本（2.3）只负责记录控制与图像，不会自动推导外参；外参应作为实验配置写入报告。
