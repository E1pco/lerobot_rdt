#!/usr/bin/env python3
"""
测试 ik_solver_dh.py 中的 DH 机器人包装类和 IK 求解
不需要实际的硬件连接
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

sys.path.insert(0, '/home/elpco/code/lerobot/lerobot_rdt')

from ik.DH import create_so101_5dof


def build_target_pose(x=0.5, y=0, z=0.1, roll=0.0, pitch=np.pi/4, yaw=0.0):
    """构造目标末端位姿"""
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def test_dh_model():
    """测试 DH 模型的基本功能"""
    
    print("=" * 70)
    print("测试 DH 模型基本功能")
    print("=" * 70)
    
    # 创建 DH 模型
    print("\n1️⃣  创建 DH 模型...")
    dh_robot = create_so101_5dof()
    print(f"   ✅ DH 模型: {dh_robot.n} DOF")
    print(f"   关节类型: {dh_robot.joint_type}")
    print(f"   DH 约定: {dh_robot.convention}")
    
    # 关节限位
    print(f"\n2️⃣  关节限位信息:")
    for i in range(dh_robot.n):
        q_min = np.degrees(dh_robot.qlim[0, i])
        q_max = np.degrees(dh_robot.qlim[1, i])
        print(f"   关节 {i}: [{q_min:7.2f}°, {q_max:7.2f}°]")
    
    # 测试 FK
    print(f"\n3️⃣  正运动学测试:")
    test_configs = [
        ("零位", np.zeros(5)),
        ("随机配置1", np.array([0.1, -0.2, 0.15, 0.1, -0.05])),
        ("随机配置2", np.array([np.pi/4, -np.pi/6, 0, np.pi/8, np.pi/4])),
    ]
    
    for name, q in test_configs:
        T = dh_robot.fkine(q)
        print(f"\n   {name}:")
        print(f"      q(°) = {np.degrees(q)}")
        print(f"      末端位置: [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}] m")
    
    # 测试 IK
    print(f"\n4️⃣  逆运动学测试:")
    
    # 从一个已知的 FK 结果反向求解
    q_target = np.array([0.1, -0.2, 0.15, 0.1, -0.05])
    T_target = dh_robot.fkine(q_target)
    
    print(f"\n   目标关节角(°): {np.degrees(q_target)}")
    print(f"   目标末端位置: [{T_target[0,3]:.4f}, {T_target[1,3]:.4f}, {T_target[2,3]:.4f}] m")
    
    # 使用 ETS IK 求解
    print(f"\n   使用 ETS IK 求解器...")
    from ik.solvers import IK_LM
    
    ik_solver = IK_LM(dh_robot.ets, ilimit=5000, slimit=250, tol=1e-5, k=0.1)
    
    q0 = np.zeros(5)  # 初始猜测
    mask = np.array([1, 1, 1, 0, 0, 0])  # 仅求解位置
    
    try:
        result = ik_solver.solve(T_target, q0=q0, mask=mask)
        
        print(f"\n   求解结果:")
        if hasattr(result, 'success'):
            print(f"      成功: {result.success}")
            print(f"      关节角(°): {np.degrees(result.q)}")
            
            # 验证结果
            T_verify = dh_robot.fkine(result.q)
            pos_error = np.linalg.norm(T_verify[:3, 3] - T_target[:3, 3])
            print(f"      末端位置: [{T_verify[0,3]:.4f}, {T_verify[1,3]:.4f}, {T_verify[2,3]:.4f}] m")
            print(f"      位置误差: {pos_error*1000:.4f} mm")
        else:
            print(f"      结果: {result}")
            pos_error = np.linalg.norm(dh_robot.fkine(result)[:3, 3] - T_target[:3, 3])
            print(f"      位置误差: {pos_error*1000:.4f} mm")
            
    except Exception as e:
        print(f"      ❌ IK 求解失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_dh_model()
