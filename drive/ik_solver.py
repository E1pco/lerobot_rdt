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
from lerobot_kinematics.ET import ET

# -----------------------------
# 1) 构建 SO-101 (5DOF) 的 ET 模型
#    关节顺序：base(Rz) → shoulder(Ry) → elbow(Ry) → wrist_pitch(Ry) → wrist_roll(Rx)
# -----------------------------
def create_so101_5dof():
    E1 = ET.Rz()      # shoulder_pan
    E2 = ET.tx(0.0612)
    E3 = ET.tz(0.0598)
    E4 = ET.tx(0.02943)
    E5 = ET.tz(0.05504)
    E6 = ET.Ry()      # shoulder_lift
    E7 = ET.tz(0.1127)
    E8 = ET.tx(0.02798)
    E9 = ET.Ry()      # elbow_flex
    E10 = ET.tx(0.13504)
    E11 = ET.tz(0.00519)
    E12 = ET.Ry()     # wrist_flex
    E13 = ET.tx(0.0593)
    E14 = ET.tz(0.00996)
    E15 = ET.Rx()     # wrist_roll

    robot = E1 * E2 * E3 * E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15

    # 自动同步URDF中的限位
    robot.qlim = np.array([
        [-1.91986, -1.74533, -1.69, -1.65806, -2.74385],
        [ 1.91986,  1.74533,  1.69,  1.65806,  2.84121]
    ])
    return robot

# -----------------------------
# 2) 构造目标末端位姿 (位置 + 姿态)
# -----------------------------
def build_target_pose(x=0.18, y=0.05, z=0.22, roll=0.0, pitch=np.pi/4, yaw=0.0):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = [x, y, z]
    return T


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
    home_pose = {
        "shoulder_pan": 2096,
        "shoulder_lift": 1983,
        "elbow_flex":   2100,
        "wrist_flex":   1954,
        "wrist_roll":   2048,
        "gripper":      2037,   # 抓手不参与 IK，可忽略
    }

    print("\n[HOME] 即将回到中位（软启动）:")
    for k, v in home_pose.items():
        print(f"  - {k:15s} → {v}")
    controller.move_all_home()
    time.sleep(0.6)

    # 4.4 构建 5DOF 机器人、准备 IK
    ets = create_so101_5dof()
    gear_sign = {
        "shoulder_pan": +1,
        "shoulder_lift": +1,
        "elbow_flex":   +1,
        "wrist_flex":   +1,
        "wrist_roll":   +1,
    }
    gear_ratio = {
        "shoulder_pan": 1.0,
        "shoulder_lift": 1.0,
        "elbow_flex":   1.0,
        "wrist_flex":   1.0,
        "wrist_roll":   1.0,
    }
# 从控制器读取当前实际步数
    ids = [cfg["id"] for cfg in controller.config.values()]
    resp = controller.servo.sync_read(0x38, 2, ids)

    q0 = np.zeros(5)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    for i, name in enumerate(joint_names):
        sid = controller.config[name]["id"]
        cur_pos = resp.get(sid, [home_pose[name] & 0xFF, home_pose[name] >> 8])
        current = cur_pos[0] + (cur_pos[1] << 8)
        delta = current - home_pose[name]
        q0[i] = gear_sign[name] * delta * 0.0015339807878856412
        print(f" {name:15s} : 步数差={delta:+d} → q0={q0[i]:+.4f} rad ")

    # 目标末端位姿（可自行调整）
    T_goal = build_target_pose(x=0.1, y=0.1, z=0.15, roll=0, pitch=-np.pi/4, yaw=np.pi/6)
    print("\n🎯 目标末端位姿矩阵：\n", np.round(T_goal, 3))

    # 4.5 IK 求解（LM）
    sol = ets.ikine_LM(
        Tep=T_goal,
        q0=q0,
        ilimit=100, slimit=5, tol=1e-3,
        mask=np.array([1, 1, 1, 0, 0.8, 0.8]),  # 位置+姿态(无绕轴)
        k=0.1, method="sugihara",
        kq=0.0, km=0.0 
    )


    if not sol.success:
        print("\n❌ 逆运动学求解失败：", sol.reason)
        controller.close()
        return

    print("\n✅ IK 求解成功")
    print("q(rad) =", np.round(sol.q, 4))
    # FK 验证
    T_fk = ets.fkine(sol.q).A
    print("FK(T) =\n", np.round(T_fk, 3))

    # 4.6 角度 → 步数映射（只映射 5 个 IK 关节）
    joint5 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll", "wrist_flex"]



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
