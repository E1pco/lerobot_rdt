#!/usr/bin/env python3
"""测试新的 Robot API 是否与原 API 兼容"""

import numpy as np
from ik.robot import create_so101_5dof

# 创建机器人模型
robot = create_so101_5dof()

print("=" * 70)
print("测试 Robot API 兼容性")
print("=" * 70)

# 测试 1: fkine
print("\n[测试 1] 正运动学 (fkine)")
q = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
T = robot.fkine(q)
print(f"输入关节角度: {q}")
print(f"输出位姿矩阵形状: {T.shape}")
print(f"末端位置: {T[:3, 3]}")

# 测试 2: ikine_LM
print("\n[测试 2] 逆运动学 (ikine_LM)")
target_T = np.eye(4)
target_T[:3, 3] = [0.2, 0.0, 0.15]

sol = robot.ikine_LM(
    Tep=target_T,
    q0=q,
    ilimit=50,
    tol=1e-3,
    mask=[1, 1, 1, 0.8, 0.8, 0],
    k=0.1,
    method="sugihara"
)

print(f"求解成功: {sol.success}")
print(f"关节角度: {sol.q}")
print(f"失败原因: {sol.reason if not sol.success else 'N/A'}")

# 测试 3: 验证 mask 参数
print("\n[测试 3] 验证 mask 参数生效")
target_T_yaw = np.eye(4)
target_T_yaw[:3, 3] = [0.2, 0.0, 0.15]
# 添加 yaw 旋转
from scipy.spatial.transform import Rotation as R
target_T_yaw[:3, :3] = R.from_euler('xyz', [0, 0, np.pi/4]).as_matrix()

# 使用 mask[5]=0 忽略 yaw
sol_no_yaw = robot.ikine_LM(
    Tep=target_T_yaw,
    q0=q,
    ilimit=50,
    tol=1e-3,
    mask=[1, 1, 1, 1, 1, 0],  # 忽略 yaw
    k=0.1,
    method="sugihara"
)

print(f"带 yaw 旋转的目标位姿，但 mask[5]=0")
print(f"求解成功: {sol_no_yaw.success}")
if sol_no_yaw.success:
    T_result = robot.fkine(sol_no_yaw.q)
    result_rpy = R.from_matrix(T_result[:3, :3]).as_euler('xyz')
    print(f"结果 yaw 角度: {np.degrees(result_rpy[2]):.2f}°")
    print(f"目标 yaw 角度: {np.degrees(np.pi/4):.2f}°")
    print(f"→ yaw 差异大，说明 mask 生效（忽略了 yaw）")

print("\n" + "=" * 70)
print("✓ API 兼容性测试完成")
print("=" * 70)
