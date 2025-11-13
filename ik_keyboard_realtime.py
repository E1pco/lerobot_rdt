#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------
# File: ik_keyboard_realtime_speed.py
# Desc: SO-101 5DOF 实时键盘控制（读取当前舵机角度作为起点）
# ------------------------------------------------

import time
import sys
import termios
import tty
import select
import numpy as np
from scipy.spatial.transform import Rotation as R

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof


# ----------------------------- # 1) 创建 ET 模型
# -----------------------------
# 已使用 ik.robot.create_so101_5dof()


# -----------------------------
# 2) 构造位姿矩阵
# -----------------------------
def build_target_pose(x, y, z, roll, pitch, yaw):
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


# -----------------------------
# 3) 角度→步数映射
# -----------------------------
def q_to_servo_targets(q_rad, joint_names, home_map, counts_per_rev=4096,
                       gear_ratio=None, gear_sign=None):
    if gear_ratio is None:
        gear_ratio = {name: 1.0 for name in joint_names}
    if gear_sign is None:
        gear_sign = {name: +1 for name in joint_names}
    counts_per_rad = counts_per_rev / (2*np.pi)
    targets = {}
    for i, name in enumerate(joint_names):
        steps = int(round(home_map[name] + gear_sign[name]*gear_ratio[name]*q_rad[i]*counts_per_rad))
        targets[name] = steps
    return targets


# -----------------------------
# 4) 非阻塞键盘监听
# -----------------------------
def get_key_nonblock():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


# -----------------------------
# 5) 主程序
# -----------------------------
def main():
    controller = ServoController(port="/dev/ttyACM0", baudrate=1_000_000, config_path="left_arm.json")
    home_pose = {
        "shoulder_pan": 2096,
        "shoulder_lift": 1983,
        "elbow_flex":   2100,
        "wrist_flex":   1954,
        "wrist_roll":   2048,
        "gripper":      2037,
    }
    controller.move_all_home()
    time.sleep(1)
    robot = create_so101_5dof()
    ets = robot.ets
    joint5 = robot.joint_names
    gear_sign = robot.gear_sign
    gear_ratio = robot.gear_ratio


    ids = [controller.config[name]["id"] for name in joint5]
    resp = controller.servo.sync_read(0x38, 2, ids)
    q0 = np.zeros(5)
    counts_per_rad = 4096 / (2*np.pi)

    print("\n📡 读取当前舵机位置：")
    for i, name in enumerate(joint5):
        sid = controller.config[name]["id"]
        cur_pos = resp.get(sid, [home_pose[name] & 0xFF, home_pose[name] >> 8])
        current = cur_pos[0] + (cur_pos[1] << 8)
        delta = current - home_pose[name]
        q0[i] = gear_sign[name] * delta / counts_per_rad
        print(f"  {name:15s}: {current:4d} (Δ={delta:+d}) → {q0[i]:+.4f} rad")

    # ✅ 根据当前角度计算末端实际位姿
    T_now = ets.fkine(q0)
    pos = T_now[:3, 3]
    rpy = R.from_matrix(T_now[:3, :3]).as_euler('xyz')
    print(f"\n✅ 已同步当前机械臂姿态\n   pos={np.round(pos,3)}, rpy(deg)={np.round(np.degrees(rpy),1)}")

    # 控制参数
    speed = 800
    print("\n🎮 键盘控制已启动")
    print("W/S: +Z/-Z | A/D: -Y/+Y | I/K: +X/-X | J/L: pitch | U/O: yaw | +/-: 速度调节 | Q:退出\n")

    # 设置终端 raw 模式
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        while True:
            key = get_key_nonblock()
            if key:
                if key == 'q':
                    print("\n🛑 退出控制")
                    break
                elif key == '+':
                    speed = min(speed + 100, 2000)
                    print(f"\n⚙️ 当前速度 ↑ {speed}")
                elif key == '-':
                    speed = max(speed - 100, 200)
                    print(f"\n⚙️ 当前速度 ↓ {speed}")
                elif key == 'w': pos[2] += 0.005
                elif key == 's': pos[2] -= 0.005
                elif key == 'a': pos[1] -= 0.005
                elif key == 'd': pos[1] += 0.005
                elif key == 'i': pos[0] += 0.005
                elif key == 'k': pos[0] -= 0.005
                elif key == 'j': rpy[1] += np.deg2rad(2)
                elif key == 'l': rpy[1] -= np.deg2rad(2)
                elif key == 'u': rpy[2] += np.deg2rad(2)
                elif key == 'o': rpy[2] -= np.deg2rad(2)

            # IK 求解
            T_goal = build_target_pose(*pos, *rpy)
            sol = robot.ikine_LM(
                Tep=T_goal, 
                q0=q0,
                ilimit=50, 
                tol=1e-3,
                mask=[1,1,1,0,1,1],
                k=0.1, 
                method="sugihara"
            )

            if sol.success:
                q0 = sol.q
                servo_targets = q_to_servo_targets(q0, joint5, home_pose,
                                                   gear_ratio=gear_ratio, gear_sign=gear_sign)
                for k in joint5:
                    servo_targets[k] = controller.limit_position(k, servo_targets[k])
                controller.fast_move_to_pose(servo_targets, speed=speed)
                print(f"\r→ pos={pos.round(3)}, rpy(deg)={np.rad2deg(rpy).round(1)}, speed={speed}", end='')
            else:
                print("\r❌ IK失败，跳过", end='')

            time.sleep(0.04)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        controller.close()
        print("\n舵机已关闭")

if __name__ == "__main__":
    main()
