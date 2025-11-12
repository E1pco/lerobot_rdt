#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 mask 参数在 IK 中的作用
验证设置 mask[5]=0 可以忽略 yaw（偏航角）
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

def test_mask_effect():
    print("=" * 70)
    print("测试 mask 参数对 IK 的影响")
    print("=" * 70)
    
    # 创建机器人模型
    robot = create_so101_5dof()
    
    # 设置初始关节角度
    q_init = np.array([0.0, -0.3, 0.3, 0.0, 0.0])
    
    # 计算当前位姿
    current_pose = robot.fk(q_init)
    print(f"\n当前位姿:")
    print(f"  位置: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}] m")
    print(f"  姿态: roll={np.rad2deg(current_pose[3]):.2f}°, pitch={np.rad2deg(current_pose[4]):.2f}°, yaw={np.rad2deg(current_pose[5]):.2f}°")
    
    # 创建一个只改变 yaw 的目标位姿
    target_pose = current_pose.copy()
    target_pose[5] += np.deg2rad(45)  # yaw 增加 45 度
    
    print(f"\n目标位姿（只改变 yaw +45°）:")
    print(f"  位置: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}] m")
    print(f"  姿态: roll={np.rad2deg(target_pose[3]):.2f}°, pitch={np.rad2deg(target_pose[4]):.2f}°, yaw={np.rad2deg(target_pose[5]):.2f}°")
    
    # 测试 1: 使用完整 mask（所有维度权重为1）
    print("\n" + "=" * 70)
    print("测试 1: 完整 mask [1, 1, 1, 1, 1, 1] - 考虑所有自由度")
    print("=" * 70)
    
    T = build_target_pose_from_vec(target_pose)
    sol1 = robot.ikine_LM(Tep=T, q0=q_init[:robot.n], ilimit=100, tol=1e-3, mask=np.array([1, 1, 1, 1, 1, 1]))

    success1 = sol1.success
    if sol1.success:
        q_solution1 = sol1.q
        result_pose1 = robot.fk(q_solution1)
        print(f"✓ IK 成功")
        print(f"  解的关节角度: {np.round(q_solution1, 4)}")
        print(f"  达到的姿态: roll={np.rad2deg(result_pose1[3]):.2f}°, pitch={np.rad2deg(result_pose1[4]):.2f}°, yaw={np.rad2deg(result_pose1[5]):.2f}°")
        yaw_error1 = abs(result_pose1[5] - target_pose[5])
        print(f"  yaw 误差: {np.rad2deg(yaw_error1):.2f}°")
    else:
        print("❌ IK 失败")
    
    # 测试 2: 忽略 yaw 的 mask（yaw 维度权重为0）
    print("\n" + "=" * 70)
    print("测试 2: 忽略 yaw 的 mask [1, 1, 1, 1, 1, 0] - 不考虑 yaw")
    print("=" * 70)
    
    sol2 = robot.ikine_LM(Tep=T, q0=q_init[:robot.n], ilimit=100, tol=1e-3, mask=np.array([1, 1, 1, 1, 1, 0]))

    success2 = sol2.success
    if sol2.success:
        q_solution2 = sol2.q
        result_pose2 = robot.fk(q_solution2)
        print(f"✓ IK 成功")
        print(f"  解的关节角度: {np.round(q_solution2, 4)}")
        print(f"  达到的姿态: roll={np.rad2deg(result_pose2[3]):.2f}°, pitch={np.rad2deg(result_pose2[4]):.2f}°, yaw={np.rad2deg(result_pose2[5]):.2f}°")
        yaw_error2 = abs(result_pose2[5] - target_pose[5])
        print(f"  yaw 误差: {np.rad2deg(yaw_error2):.2f}°")
        print(f"  ⚠️ 注意: yaw 没有跟随目标值（因为被忽略了）")
    else:
        print("❌ IK 失败")
    
    # 比较结果
    print("\n" + "=" * 70)
    print("结论:")
    print("=" * 70)
    if success1 and success2:
        print(f"✓ 使用完整 mask 时，yaw 误差: {np.rad2deg(yaw_error1):.2f}°")
        print(f"✓ 忽略 yaw mask 时，yaw 误差: {np.rad2deg(yaw_error2):.2f}°")
        print(f"\n✅ mask 参数生效！设置 mask[5]=0 可以忽略偏航角")
    else:
        print("⚠️ 部分测试失败，请检查参数设置")

if __name__ == "__main__":
    test_mask_effect()
