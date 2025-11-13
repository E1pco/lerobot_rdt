#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------
# File: ik_solver_drive_test.py
# Desc: ET/IK + ServoController 一体化运行示例
# Flow: 回中(软启动) → IK → 打印目标步数 → 按回车执行
# ------------------------------------------------

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof

# -----------------------------
# 构造目标末端位姿 (位置 + 姿态)
# -----------------------------
def build_target_pose(x=0.5, y=0, z=0.1, roll=0.0, pitch=np.pi/4, yaw=0.0):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = [x, y, z]
    return T

# -----------------------------
# 主流程：回中 → IK → 打印 → 回车执行
# -----------------------------
def main():
    # 4.1 初始化底层控制
    controller = ServoController(port="/dev/ttyACM0", baudrate=1_000_000, config_path="./driver/servo_config.json")
    robot = create_so101_5dof()
    q0 = np.zeros(5)
    controller.move_all_home()
    time.sleep(1)
    q0 = robot.read_joint_angles()
    # 计算当前末端位姿
    T_current = robot.fkine(q0)
    print("\n🔍 当前末端位姿矩阵：")
    print(np.round(T_current, 3))
    print(f"当前位置: x={T_current[0,3]:.4f}, y={T_current[1,3]:.4f}, z={T_current[2,3]:.4f},roll={0:.4f}, pitch={1:.4f}, yaw={2:.4f}".format(
        R.from_matrix(T_current[:3, :3]).as_euler('xyz', degrees=False)[0],
        R.from_matrix(T_current[:3, :3]).as_euler('xyz', degrees=False)[1],
        R.from_matrix(T_current[:3, :3]).as_euler('xyz', degrees=False)[2],
    ))

    # 目标末端位姿（可自行调整）
    T_goal = build_target_pose(x=0.3, y=-0.2, z=0.15, roll=np.pi/4, pitch=0, yaw=0)
    print("\n🎯 目标末端位姿矩阵：")
    print(np.round(T_goal, 3))
    print(f"目标位置: x={T_goal[0,3]:.4f}, y={T_goal[1,3]:.4f}, z={T_goal[2,3]:.4f}")
    
    print("\n🔄 开始从当前位置进行逆运动学求解...")
    sol = robot.ikine_LM(
        Tep=T_goal,
        q0=q0,
        ilimit=1000, 
        slimit=50,
        tol=1e-6,
        mask=np.array([1, 1, 1, 0, 0, 0]),  
        k=0.1, 
        method="sugihara"
    )

    if not sol.success:
        print("\n❌ 逆运动学求解失败：", sol.reason)
        controller.close()
        return

    print("\n✅ IK 求解成功")
    print("目标关节角度 q(rad) =", np.round(sol.q, 4))


    servo_targets = robot.q_to_servo_targets(q_rad=sol.q)

    # 电子限位保护（用底层 clamp 一次，双保险）
    for k in list(servo_targets.keys()):
        servo_targets[k] = controller.limit_position(k, servo_targets[k])

    print("\n📋 即将执行的舵机目标步数：")
    for k in robot.joint_names:
        print(f"  - {k:15s} : {servo_targets[k]}")

    input("\n按 Enter 开始平滑执行到目标位姿...")
    controller.soft_move_to_pose(servo_targets, step_count=5, interval=0.08)
    q0 = robot.read_joint_angles()
    # 计算当前末端位姿
    T_current = robot.fkine(q0)
    print("\n🔍 当前末端位姿矩阵：")
    print(np.round(T_current, 3))

    print(f"当前位置: x={T_current[0,3]:.4f}, y={T_current[1,3]:.4f}, z={T_current[2,3]:.4f}")
    pos_error = np.linalg.norm(T_current[:3,3] - T_goal[:3,3])
    print(f"位置误差: {pos_error*1000:.2f} mm")


    print("\n✅ 动作完成，开始监控（Ctrl+C 退出）")
    try:
        controller.monitor_positions(ids=[cfg["id"] for cfg in controller.config.values()], interval=0.3)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
