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

from ftservo_controller import ServoController
from ik.robot import create_so101_5dof

# ========== 应用坐标系转换 ==========
def build_target_pose(robot, x=0, y=0.3, z=0.0, roll=0.0, pitch=-np.pi/4, yaw=0.0):
    """构造目标末端位姿 (用户坐标系)"""
    return robot.build_pose(x, y, z, roll, pitch, yaw)


# -----------------------------
# 3) 角度(rad) → 舵机步数 的映射
#    - home_map: 各关节在“零角”时的舵机步数（你给的中位姿）
#    - counts_per_rev: 每圈脉冲数 (默认 4096)
#    - gear_ratio: 关节减速比 (电机→关节)
#    - gear_sign: 方向 (+1/-1)
# -----------------------------
def q_to_servo_targets(q_rad, joint_names, home_map,
                       counts_per_rev=4096,
                       gear_ratio=None,
                       gear_sign=None):
    if gear_ratio is None:
        gear_ratio = {name: 1.0 for name in joint_names}
    if gear_sign is None:
        gear_sign = {name: +1 for name in joint_names}

    counts_per_rad = counts_per_rev / (2 * np.pi)  # ≈ 651.8986

    targets = {}
    for i, name in enumerate(joint_names):
        q = float(q_rad[i])
        steps = int(round(home_map[name] + gear_sign[name] * gear_ratio[name] * q * counts_per_rad))
        targets[name] = steps
    return targets


# -----------------------------
# 4) 主流程：回中 → IK → 打印 → 回车执行
# -----------------------------
def main():
    # 4.1 初始化底层控制
    controller = ServoController(port="/dev/ttyACM0", baudrate=1_000_000, config_path="servo_config.json")
    
    # 获取home位置（用于计算角度差值）
    home_pose = {}
    for name in controller.config.keys():
        home_pose[name] = controller.get_home_position(name)
    
    print("\n📍 跳过回中位，直接读取当前位置...")

    # 4.4 构建 5DOF 机器人、准备 IK
    robot = create_so101_5dof()
    ets = robot.ets
    gear_sign = {
            "shoulder_pan": +1,
            "shoulder_lift": +1,
            "elbow_flex":   +1,
            "wrist_flex":   -1,
            "wrist_roll":   -1,
        }
    gear_ratio = {
        "shoulder_pan": 1.0,
        "shoulder_lift": 1.0,
        "elbow_flex":   1.0,
        "wrist_flex":   1.0,
        "wrist_roll":   1.0,
    }
    controller.move_all_home()
    time.sleep(1)
    
    # 从控制器读取当前实际步数
    ids = [cfg["id"] for cfg in controller.config.values()]
    resp = controller.servo.sync_read(0x38, 2, ids)

    q0 = np.zeros(5)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll", "wrist_flex"]

    print("\n📊 当前关节状态:")
    for i, name in enumerate(joint_names):
        sid = controller.config[name]["id"]
        cur_pos = resp.get(sid, [home_pose[name] & 0xFF, home_pose[name] >> 8])
        current = cur_pos[0] + (cur_pos[1] << 8)
        delta = current - home_pose[name]
        q0[i] = gear_sign[name] * delta * 0.0015339807878856412
        print(f" {name:15s} : 当前步数={current:4d}, 步数差={delta:+5d} → q0={q0[i]:+.4f} rad ")

    
    # 计算当前末端位姿
    T_current = ets.fkine(q0)
    print("\n🔍 当前末端位姿矩阵（机械臂坐标系）：")
    print(np.round(T_current, 3))
    # 转换到用户坐标系显示
    x_cur, y_cur, z_cur, roll_cur, pitch_cur, yaw_cur = robot.get_user_pose(T_current)
    print(f"当前位置（用户坐标系）: x={x_cur:.4f}, y={y_cur:.4f}, z={z_cur:.4f}")
    print(f"当前姿态: roll={roll_cur:.4f}, pitch={pitch_cur:.4f}, yaw={yaw_cur:.4f}")

    # 目标末端位姿（可自行调整）
    T_goal = build_target_pose(robot, x=0.3, y=0, z=0.115, roll=np.pi/2, pitch=0, yaw=0)
    print("\n🎯 目标末端位姿矩阵：")
    print(np.round(T_goal, 3))
    print(f"目标位置: x={T_goal[0,3]:.4f}, y={T_goal[1,3]:.4f}, z={T_goal[2,3]:.4f}")
    
    print("\n🔄 开始从当前位置进行逆运动学求解...")
    sol = robot.ikine_LM(
        Tep=T_goal,
        q0=q0,
        ilimit=5000, 
        slimit=500,
        tol=1e-3,
        mask=np.array([1, 1, 1, 1,0 , 0]),  
        k=0.1, 
        method="sugihara"
    )


    if not sol.success:
        print("\n❌ 逆运动学求解失败：", sol.reason)
        controller.close()
        return

    print("\n✅ IK 求解成功")
    print("目标关节角度 q(rad) =", np.round(sol.q, 4))
    
    # FK 验证
    T_fk = robot.ets.fkine(sol.q)
    print("\n验证正运动学结果:")
    print(np.round(T_fk, 3))
    # 转换到用户坐标系显示
    x_fk, y_fk, z_fk, roll_fk, pitch_fk, yaw_fk = robot.get_user_pose(T_fk)
    print(f"FK位置（用户坐标系）: x={x_fk:.4f}, y={y_fk:.4f}, z={z_fk:.4f}")
    
    # 从目标位姿提取用户坐标系的坐标进行对比
    x_goal, y_goal, z_goal, _, _, _ = robot.get_user_pose(T_goal)
    pos_error = np.linalg.norm(np.array([x_fk, y_fk, z_fk]) - np.array([x_goal, y_goal, z_goal]))
    print(f"位置误差: {pos_error*1000:.2f} mm")

    # 4.6 角度 → 步数映射（只映射 5 个 IK 关节）
    joint5 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]



    servo_targets = q_to_servo_targets(
        q_rad=sol.q,
        joint_names=joint5,
        home_map=home_pose,
        counts_per_rev=4096,
        gear_ratio=gear_ratio,
        gear_sign=gear_sign
    )

    # 电子限位保护（用底层 clamp 一次，双保险）
    for k in list(servo_targets.keys()):
        servo_targets[k] = controller.limit_position(k, servo_targets[k])

    print("\n📋 即将执行的舵机目标步数：")
    for k in joint5:
        print(f"  - {k:15s} : {servo_targets[k]}")

    input("\n按 Enter 开始平滑执行到目标位姿...")
    controller.soft_move_to_pose(servo_targets, step_count=5, interval=0.08)

    print("\n✅ 动作完成，开始监控（Ctrl+C 退出）")
    try:
        controller.monitor_positions(ids=[cfg["id"] for cfg in controller.config.values()], interval=0.3)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
