#!/usr/bin/env python3
"""
验证 ET 模型与 URDF 的一致性
通过比较零位时的末端位置来验证
"""
import numpy as np
import sys
sys.path.insert(0, '/home/elpco/code/lerobot/lerobot_rdt')

from ik.robot import create_so101_5dof

# 创建机器人模型
robot = create_so101_5dof()

# 测试零位姿态
q_zero = np.zeros(5)
T_zero = robot.fkine(q_zero)

print("="*60)
print("SO-101 ET 模型验证")
print("="*60)
print(f"\n关节数量: {robot.n}")
print(f"关节名称: {robot.joint_names}")
print(f"\n关节限位 (rad):")
print(f"  下限: {robot.qlim[0]}")
print(f"  上限: {robot.qlim[1]}")

print(f"\n零位姿态 q = {q_zero}")
print(f"\n末端位姿矩阵 T(零位):")
print(T_zero)

pos = T_zero[:3, 3]
print(f"\n末端位置 (零位):")
print(f"  x = {pos[0]:.6f} m")
print(f"  y = {pos[1]:.6f} m")
print(f"  z = {pos[2]:.6f} m")

# 测试一些典型姿态
test_configs = [
    ("零位", np.array([0.0, 0.0, 0.0, 0.0, 0.0])),
    ("肩抬30°", np.array([0.0, 0.524, 0.0, 0.0, 0.0])),
    ("肘弯45°", np.array([0.0, 0.0, 0.785, 0.0, 0.0])),
    ("组合姿态", np.array([0.3, -0.5, 0.5, 0.2, 0.0])),
]

print(f"\n{'='*60}")
print("典型姿态测试")
print(f"{'='*60}")

for name, q in test_configs:
    T = robot.fkine(q)
    pos = T[:3, 3]
    print(f"\n{name}:")
    print(f"  q = {q}")
    print(f"  末端: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    
    # 计算到零位的距离
    dist = np.linalg.norm(pos - T_zero[:3, 3])
    print(f"  距离零位: {dist*1000:.1f} mm")

# 使用 fk() 方法测试（返回 6D 位姿）
print(f"\n{'='*60}")
print("6D 位姿输出测试 (fk 方法)")
print(f"{'='*60}")

q_test = np.array([0.0, -0.3, 0.5, 0.0, 0.0])
pose_6d = robot.fk(q_test)
print(f"\n关节角度: {q_test}")
print(f"6D 位姿: [X, Y, Z, Yaw, Pitch, Roll]")
print(f"  位置: [{pose_6d[0]:.4f}, {pose_6d[1]:.4f}, {pose_6d[2]:.4f}] m")
print(f"  姿态: [{np.degrees(pose_6d[3]):.2f}°, {np.degrees(pose_6d[4]):.2f}°, {np.degrees(pose_6d[5]):.2f}°]")

print(f"\n{'='*60}")
print("✅ 验证完成")
print(f"{'='*60}")
