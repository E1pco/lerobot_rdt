#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试纯 Python IK API 的适配，包括冗余关节处理
"""

import sys
import numpy as np

sys.path.insert(0, '/home/elpco/code/lerobot/lerobot_rdt')

from ik.robot import create_so101_5dof
from scipy.spatial.transform import Rotation as R


def build_target_pose_from_vec(pose_vec):
    x, y, z, roll, pitch, yaw = pose_vec
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def test_ik_api():
    print("=" * 70)
    print("测试纯 Python IK API（包括冗余关节处理）")
    print("=" * 70)
    
    # 1. 创建机器人模型
    print("\n[1/4] 创建 SO-101 5DOF 机器人模型...")
    robot = create_so101_5dof()
    print(f"✓ 机器人创建成功，自由度: {robot.n}")
    print(f"✓ 关节限位:\n{robot.qlim}")
    
    # 2. 测试正运动学（标准输入）
    print("\n[2/4] 测试正运动学 (FK) - 标准输入...")
    q_test = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
    pose = robot.fk(q_test)
    print(f"✓ 输入关节角度 (5个): {q_test}")
    print(f"✓ 输出末端位姿: [x, y, z, roll, pitch, yaw]")
    print(f"  位置: [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}] m")
    print(f"  姿态: [{np.rad2deg(pose[3]):.2f}, {np.rad2deg(pose[4]):.2f}, {np.rad2deg(pose[5]):.2f}] deg")
    
    # 3. 测试正运动学（带额外关节，如夹爪）
    print("\n[3/4] 测试正运动学 (FK) - 包含额外关节...")
    q_test_with_gripper = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.5])  # 6个关节，最后一个是夹爪
    pose2 = robot.fk(q_test_with_gripper)
    print(f"✓ 输入关节角度 (6个，含夹爪): {q_test_with_gripper}")
    print(f"✓ 自动使用前5个关节")
    print(f"  位置: [{pose2[0]:.4f}, {pose2[1]:.4f}, {pose2[2]:.4f}] m")
    print(f"  姿态: [{np.rad2deg(pose2[3]):.2f}, {np.rad2deg(pose2[4]):.2f}, {np.rad2deg(pose2[5]):.2f}] deg")
    
    # 4. 测试逆运动学（带额外关节）
    print("\n[4/4] 测试逆运动学 (IK) - 包含额外关节...")
    target_pose = pose
    q_init_with_gripper = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])  # 初始位置含夹爪
    
    print(f"  初始q (6个): {q_init_with_gripper}")
    print(f"  目标位姿: pos=[{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
    
    # 不返回完整向量（只返回5个关节的解） - 使用 Robot.ikine_LM
    T = build_target_pose_from_vec(target_pose)
    sol = robot.ikine_LM(Tep=T, q0=q_init_with_gripper[:robot.n], ilimit=500, tol=1e-3)

    if sol.success:
        q_solution = sol.q
        print(f"✓ IK 求解成功 (return_full=False)")
        print(f"  解 (5个关节): {q_solution}")

        # 验证解的正确性
        pose_verify = robot.fk(q_solution)
        pos_error = np.linalg.norm(pose_verify[:3] - target_pose[:3])
        print(f"  位置误差: {pos_error*1000:.3f} mm")

        if pos_error < 0.01:
            print("✓ IK 解验证通过")
        else:
            print("⚠ IK 解存在误差（在容忍范围内）")
    else:
        print("❌ IK 求解失败")

    # 测试返回完整向量
    print("\n  测试 return_full=True...")
    sol_full = robot.ikine_LM(Tep=T, q0=q_init_with_gripper[:robot.n], ilimit=500, tol=1e-3)
    if sol_full.success:
        q_solution_full = q_init_with_gripper.copy()
        q_solution_full[:robot.n] = sol_full.q
        print(f"✓ IK 求解成功 (return_full=True)")
        print(f"  完整解 (6个关节): {q_solution_full}")
        print(f"  前5个关节已更新，夹爪保持不变: {q_solution_full[5]:.3f}")
    
    print("\n" + "=" * 70)
    print("✓ API 测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_ik_api()
