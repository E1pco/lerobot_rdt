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
from ik.robot import create_so101 ,create_so101_5dof,create_so101_5dof_gripper,create_so101_6dof_gripper

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
    controller = ServoController(port="/dev/right_arm", baudrate=1_000_000, config_path="./driver/right_arm.json")
    robot = create_so101_6dof_gripper()
    
    # 设置舵机控制器到机器人
    robot.set_servo_controller(controller)
    
    q0 = np.zeros(6)
    controller.move_all_home()
    #time.sleep(1)
    
    # 读取当前关节角度
    q0 = robot.read_joint_angles(
        joint_names=robot.joint_names,
        verbose=True
    )
    # 计算当前末端位姿
    T_current = robot.fkine(q0)
    print("\n🔍 当前末端位姿矩阵：")
    print(np.round(T_current, 3))
    print(f"当前位置: x={T_current[0,3]:.4f}, y={T_current[1,3]:.4f}, z={T_current[2,3]:.4f},roll={0:.4f}, pitch={1:.4f}, yaw={2:.4f}".format(
        R.from_matrix(T_current[:3, :3]).as_euler('xyz', degrees=False)[0],
        R.from_matrix(T_current[:3, :3]).as_euler('xyz', degrees=False)[1],
        R.from_matrix(T_current[:3, :3]).as_euler('xyz', degrees=False)[2],
    ))

  
    T_goal = build_target_pose(x=0.3, y=0.3, z=0.3, roll=np.pi/2, pitch=-np.pi/4, yaw=0)#
    print("\n目标末端位姿矩阵：")
    print(np.round(T_goal, 3))
    print(f"目标位置: x={T_goal[0,3]:.4f}, y={T_goal[1,3]:.4f}, z={T_goal[2,3]:.4f}")
    
    print("\n 开始从当前位置进行逆运动学求解...")
    sol = robot.ikine_LM(
        Tep=T_goal,
        q0=q0,
        ilimit=1000, 
        slimit=10,
        tol=1e-3,
        mask=np.array([0.5, 0.5, 0.5, 1, 1, 1]),  
        k=0.1, 
        method="sugihara"
    )

    if not sol.success:
        print("\n❌ 逆运动学求解失败：", sol.reason)
        controller.close()
        return

    print("\n✅ IK 求解成功")
    print("目标关节角度 q(rad) =", np.round(sol.q, 4))
    tar_q_rad = sol.q
    T_tar=robot.fkine(tar_q_rad)
    print("目标末端位姿\r T =", np.round(T_tar, 3))
    pos_error = np.linalg.norm(T_tar[:3,3] - T_goal[:3,3])
    print(f"计算误差: {pos_error*1000:.2f} mm")

    # 获取 home_pose - 需要显式传入
    home_pose = {}
    for name in robot.joint_names:
        home_pose[name] = controller.get_home_position(name)

    servo_targets = robot.q_to_servo_targets(q_rad=sol.q, home_pose=home_pose)

    # 电子限位保护（用底层 clamp 一次，双保险）
    for k in list(servo_targets.keys()):
        servo_targets[k] = controller.limit_position(k, servo_targets[k])

    print("\n📋 即将执行的舵机目标步数：")
    for k in robot.joint_names:
        print(f"  - {k:15s} : {servo_targets[k]},delta={servo_targets[k]-robot.q_to_servo_targets(q0, home_pose=home_pose)[k]}")

    input("\n按 Enter 开始平滑执行到目标位姿...")
    controller.soft_move_to_pose(servo_targets, step_count=5, interval=0.08)
    
    # 等待舵机执行完毕
    time.sleep(1)
    
    # 读取执行后的实际关节角度
    q0 = robot.read_joint_angles(
        joint_names=robot.joint_names,
        verbose=True
    )
    # 计算当前末端位姿
    T_current = robot.fkine(q0)
    print("\n🔍 当前末端位姿矩阵：")
    print(np.round(T_current, 3))

    print(f"当前位置: x={T_current[0,3]:.4f}, y={T_current[1,3]:.4f}, z={T_current[2,3]:.4f}")
    pos_error = np.linalg.norm(T_current[:3,3] - T_goal[:3,3])
    print(f"位置误差: {pos_error*1000:.2f} mm")


    print("\n✅ 动作完成，开始监控（Ctrl+C 退出）")
    try:
        while True:
            q_m=robot.read_joint_angles()
            T_m=robot.fkine(q_m)
            print("\r当前位置: x={:.4f}, y={:.4f}, z={:.4f}".format(
                T_m[0,3],T_m[1,3],T_m[2,3]
            ),end='')
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n🛑 退出监控")
    finally:
        controller.close()
        print("舵机已关闭")


if __name__ == "__main__":
    main()
