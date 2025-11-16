#!/usr/bin/env python3
"""
测试 SO-101 DH 模型与 ET 模型的一致性

对比两种运动学表示：
1. 基于 ET (Elementary Transforms) 的模型
2. 基于 DH (Denavit-Hartenberg) 参数的模型
"""

import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, '/home/elpco/code/lerobot/lerobot_rdt')

from ik.robot import create_so101_5dof as create_so101_et
from ik.DH import create_so101_5dof as create_so101_dh


def compare_models():
    """比较 ET 模型和 DH 模型"""
    
    print("=" * 80)
    print("SO-101 5DOF 机械臂运动学模型比较")
    print("=" * 80)
    
    # 创建两个模型
    print("\n1️⃣  创建模型...")
    robot_et = create_so101_et()
    robot_dh = create_so101_dh()
    
    print(f"   ✅ ET 模型: {robot_et.n} 个关节")
    print(f"   ✅ DH 模型: {robot_dh.n} 个关节")
    
    # 测试多个关节配置
    test_configs = [
        np.zeros(5),
        np.array([0.1, -0.2, 0.15, 0.1, -0.05]),
        np.array([np.pi/4, -np.pi/6, 0, np.pi/8, np.pi/4]),
        np.array([-np.pi/4, 0.3, -0.2, -0.1, -np.pi/3]),
    ]
    
    print("\n2️⃣  正运动学比较...")
    print("-" * 80)
    
    errors = []
    for idx, q in enumerate(test_configs):
        print(f"\n   测试配置 {idx + 1}: q = {np.array2string(np.degrees(q), precision=2, separator=', ')}")
        
        # 计算 FK
        T_et = robot_et.fkine(q)
        T_dh = robot_dh.fkine(q)
        
        # 计算误差
        error_pos = np.linalg.norm(T_et[:3, 3] - T_dh[:3, 3])
        
        # 计算旋转矩阵的差异（使用 Frobenius 范数）
        error_rot = np.linalg.norm(T_et[:3, :3] - T_dh[:3, :3])
        
        print(f"      ET 末端位置:   {T_et[:3, 3]}")
        print(f"      DH 末端位置:   {T_dh[:3, 3]}")
        print(f"      位置误差:     {error_pos * 1000:.4f} mm")
        print(f"      旋转误差:     {error_rot:.6f}")
        
        errors.append((error_pos, error_rot))
        
        if error_pos > 1e-6:
            print(f"      ⚠️  位置误差较大!")
        if error_rot > 1e-6:
            print(f"      ⚠️  旋转误差较大!")
    
    # 统计误差
    print("\n3️⃣  误差统计...")
    print("-" * 80)
    
    pos_errors = np.array([e[0] for e in errors])
    rot_errors = np.array([e[1] for e in errors])
    
    print(f"   位置误差统计:")
    print(f"      平均值: {np.mean(pos_errors) * 1000:.4f} mm")
    print(f"      最大值: {np.max(pos_errors) * 1000:.4f} mm")
    print(f"      最小值: {np.min(pos_errors) * 1000:.4f} mm")
    
    print(f"\n   旋转误差统计:")
    print(f"      平均值: {np.mean(rot_errors):.6f}")
    print(f"      最大值: {np.max(rot_errors):.6f}")
    print(f"      最小值: {np.min(rot_errors):.6f}")
    
    # 关节信息
    print("\n4️⃣  关节信息...")
    print("-" * 80)
    
    print(f"\n   ET 模型关节:")
    if hasattr(robot_et, 'name'):
        for i, name in enumerate(robot_et.name):
            print(f"      {i}: {name}")
    
    print(f"\n   DH 模型关节:")
    print(f"      DH 参数矩阵 (theta, d, r, alpha):")
    for i, (theta, d, r, alpha) in enumerate(robot_dh.dh_params):
        print(f"      {i}: theta={np.degrees(theta):7.2f}° d={d:8.6f}m r={r:8.6f}m alpha={np.degrees(alpha):7.2f}°")
    
    print(f"\n      关节限位:")
    for i in range(robot_dh.n):
        q_min = np.degrees(robot_dh.qlim[0, i])
        q_max = np.degrees(robot_dh.qlim[1, i])
        print(f"      关节 {i}: [{q_min:7.2f}°, {q_max:7.2f}°]")
    
    # 验证模型
    print("\n5️⃣  模型验证...")
    print("-" * 80)
    
    if np.max(pos_errors) < 1e-4 and np.max(rot_errors) < 1e-4:
        print("   ✅ 两个模型运动学完全一致!")
    elif np.max(pos_errors) < 1e-3 and np.max(rot_errors) < 1e-3:
        print("   ⚠️  两个模型运动学基本一致，误差很小")
    else:
        print("   ❌ 两个模型存在显著差异，需要检查!")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    return np.max(pos_errors) < 1e-3 and np.max(rot_errors) < 1e-3


if __name__ == "__main__":
    try:
        success = compare_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
