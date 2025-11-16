#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------
# File: ik_solver_dh.py
# Desc: DH 参数模型 + IK 求解 + ServoController 一体化运行示例
# Flow: 回中(软启动) → IK → 打印目标步数 → 按回车执行
# 
# 说明：
#   - 使用 DH.create_so101_5dof() 创建的 DH 模型
#   - 与 ik_solver_py.py 保持相同的 API
#   - 支持与硬件控制器集成
# ------------------------------------------------

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from driver.ftservo_controller import ServoController
from ik.DH import create_so101_5dof
from ik.solvers import IK_LM

# 注意：DHRobot 没有 ikine_LM 方法，需要封装或使用 ETS 接口
# 这里我们创建一个包装类来统一接口


class DHRobotWrapper:
    """
    DH 机器人的包装类，提供与 Robot 类相同的接口
    支持 IK 求解、伺服控制集成等
    """
    
    def __init__(self, dh_robot, servo_controller=None, joint_names=None, gear_sign=None):
        """
        初始化 DH 机器人包装类
        
        Parameters
        ----------
        dh_robot : DHRobot
            DH 模型的机器人
        servo_controller : ServoController, optional
            舵机控制器实例
        joint_names : list, optional
            关节名称列表
        gear_sign : dict, optional
            关节方向符号 {name: +1 or -1}
        """
        self.dh_robot = dh_robot
        self.n = dh_robot.n
        self.servo_controller = servo_controller
        self.counts_per_rad = 4096 / (2 * np.pi)  # 舵机转换系数
        
        if joint_names is None:
            self.joint_names = [f"joint_{i}" for i in range(self.n)]
        else:
            self.joint_names = joint_names
        
        # 关节方向符号（默认都是正向）
        if gear_sign is None:
            self.gear_sign = {name: 1 for name in self.joint_names}
        else:
            self.gear_sign = gear_sign
    
    def set_servo_controller(self, controller):
        """设置舵机控制器"""
        self.servo_controller = controller
    
    def fkine(self, q: np.ndarray) -> np.ndarray:
        """
        正运动学求解
        
        Parameters
        ----------
        q : np.ndarray
            关节角度向量（弧度）
            
        Returns
        -------
        np.ndarray
            4x4 齐次变换矩阵
        """
        return self.dh_robot.fkine(q)
    
    def read_joint_angles(self, joint_names=None, verbose=False):
        """
        读取当前关节角度
        
        Parameters
        ----------
        joint_names : list, optional
            要读取的关节名称列表
        verbose : bool
            是否打印详细信息
            
        Returns
        -------
        np.ndarray
            关节角度向量
        """
        if self.servo_controller is None:
            raise RuntimeError("ServoController 未设置")
        
        if joint_names is None:
            joint_names = self.joint_names
        
        # 读取舵机步数
        positions = self.servo_controller.read_servo_positions(joint_names=joint_names, verbose=False)
        
        q = np.zeros(len(joint_names))
        
        if verbose:
            print("\n📡 读取关节角度:")
        
        for i, name in enumerate(joint_names):
            pos_steps = positions[name]
            home_pos = self.servo_controller.get_home_position(name)
            delta = pos_steps - home_pos
            q[i] = self.gear_sign[name] * delta / self.counts_per_rad
            
            if verbose:
                print(f"   {name:15s} : 步数={pos_steps:4d}, Δ={delta:+5d} → q={q[i]:+.4f} rad ({np.degrees(q[i]):+7.2f}°)")
        
        return q
    
    def q_to_servo_targets(self, q_rad: np.ndarray, home_pose: dict) -> dict:
        """
        将关节角度转换为舵机目标步数
        
        Parameters
        ----------
        q_rad : np.ndarray
            关节角度向量（弧度）
        home_pose : dict
            home 位置字典 {joint_name: steps}
            
        Returns
        -------
        dict
            舵机目标步数 {joint_name: steps}
        """
        servo_targets = {}
        for i, name in enumerate(self.joint_names):
            # 公式：steps = home_pose + gear_sign * q_rad * counts_per_rad
            delta = self.gear_sign[name] * q_rad[i] * self.counts_per_rad
            servo_targets[name] = int(np.round(home_pose[name] + delta))
        return servo_targets
    
    def ikine_LM(self, Tep: np.ndarray, q0: np.ndarray, 
                 ilimit: int = 5000, slimit: int = 250,
                 tol: float = 1e-5, mask: np.ndarray = None,
                 k: float = 0.1, method: str = "sugihara"):
        """
        Levenberg-Marquardt 逆运动学求解
        
        Parameters
        ----------
        Tep : np.ndarray
            目标末端位姿 (4x4 齐次矩阵)
        q0 : np.ndarray
            初始关节角度
        ilimit : int
            最大迭代次数
        slimit : int
            最大步长限制
        tol : float
            收敛容差
        mask : np.ndarray
            求解掩码 [1,1,1,0,0,0] 表示只求解位置
        k : float
            阻尼因子
        method : str
            求解方法
            
        Returns
        -------
        IKResult
            逆运动学求解结果
        """
        # 使用 ETS 进行 IK 求解（DH 的 ETS 表示）
        ik_solver = IK_LM(
            self.dh_robot.ets,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            k=k
        )
        
        if mask is None:
            mask = np.ones(6)
        
        # 执行 IK 求解
        sol = ik_solver.solve(Tep, q0=q0, mask=mask, method=method)
        
        return sol


# 兼容的 IKResult 类（与 robot.py 中的结构一致）
class IKResult:
    """IK 求解结果封装类"""
    def __init__(self, success, q, reason=""):
        self.success = success
        self.q = q
        self.reason = reason


# 转换 ETS IK 结果为兼容格式
def convert_ik_result(ets_result):
    """将 ETS IK 结果转换为兼容的 IKResult 格式"""
    if hasattr(ets_result, 'success'):
        return IKResult(ets_result.success, ets_result.q, 
                       getattr(ets_result, 'reason', ''))
    else:
        # 假设是数组，表示求解成功
        return IKResult(True, ets_result, "")


# ------------------------------------------------
# 构造目标末端位姿 (位置 + 姿态)
# ------------------------------------------------
def build_target_pose(x=0.5, y=0, z=0.1, roll=0.0, pitch=np.pi/4, yaw=0.0):
    """
    构造目标末端位姿
    
    Parameters
    ----------
    x, y, z : float
        位置坐标（米）
    roll, pitch, yaw : float
        欧拉角（弧度）
        
    Returns
    -------
    np.ndarray
        4x4 齐次变换矩阵
    """
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = [x, y, z]
    return T


# ------------------------------------------------
# 主流程：回中 → IK → 打印 → 回车执行
# ------------------------------------------------
def main():
    """
    主程序流程：
    1. 初始化舵机控制器和机器人模型
    2. 机器人回中
    3. 读取当前关节角度
    4. IK 求解目标位姿
    5. 执行平滑移动
    6. 实时监控位置
    """
    
    print("=" * 70)
    print("SO-101 DH 模型 IK 求解示例")
    print("=" * 70)
    
    # 1. 初始化底层控制
    print("\n📱 初始化舵机控制器...")
    try:
        controller = ServoController(
            port="/dev/ttyACM0",
            baudrate=1_000_000,
            config_path="./driver/servo_config.json"
        )
    except Exception as e:
        print(f"❌ 舵机控制器初始化失败: {e}")
        return
    
    print("📐 创建 DH 机器人模型...")
    dh_robot = create_so101_5dof()
    print(f"   ✅ DH 模型已创建: {dh_robot.n} DOF")
    
    print("   关节名称: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll")
    
    # 2. 创建包装类，统一接口
    robot = DHRobotWrapper(
        dh_robot,
        servo_controller=controller,
        joint_names=[
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll"
        ],
        gear_sign={
            "shoulder_pan": 1,
            "shoulder_lift": 1,
            "elbow_flex": 1,
            "wrist_flex": 1,
            "wrist_roll": 1
        }
    )
    
    # 3. 机器人回中
    print("\n🏠 执行回中动作...")
    controller.move_all_home()
    time.sleep(1)
    
    # 4. 读取当前关节角度
    print("\n📍 读取当前关节角度...")
    q0 = robot.read_joint_angles(joint_names=robot.joint_names, verbose=True)
    
    # 计算当前末端位姿
    T_current = robot.fkine(q0)
    print("\n🔍 当前末端位姿：")
    print(f"   位置: x={T_current[0,3]:.4f}, y={T_current[1,3]:.4f}, z={T_current[2,3]:.4f} (m)")
    euler = R.from_matrix(T_current[:3, :3]).as_euler('xyz', degrees=False)
    print(f"   欧拉角: roll={np.degrees(euler[0]):.2f}°, pitch={np.degrees(euler[1]):.2f}°, yaw={np.degrees(euler[2]):.2f}°")
    
    # 5. 定义目标末端位姿（可自行调整）
    print("\n🎯 定义目标末端位姿...")
    T_goal = build_target_pose(
        x=0.0, 
        y=-0.25, 
        z=0.25,
        roll=np.pi/4,
        pitch=-np.pi/6,
        yaw=0
    )
    print(f"   位置: x={T_goal[0,3]:.4f}, y={T_goal[1,3]:.4f}, z={T_goal[2,3]:.4f} (m)")
    
    # 6. IK 求解
    print("\n🔄 执行逆运动学求解...")
    print("   算法: Levenberg-Marquardt")
    print("   掩码: [1, 1, 1, 0, 0, 0] (仅求解位置)")
    
    try:
        sol = robot.ikine_LM(
            Tep=T_goal,
            q0=q0,
            ilimit=5000,
            slimit=250,
            tol=1e-5,
            mask=np.array([1, 1, 1, 0, 0, 0]),
            k=0.1,
            method="sugihara"
        )
        
        # 处理不同的结果格式
        if hasattr(sol, 'success'):
            success = sol.success
            q_sol = sol.q if hasattr(sol, 'q') else sol[0]
            reason = getattr(sol, 'reason', '')
        else:
            # 假设数组表示成功
            success = True
            q_sol = sol
            reason = ""
        
        if not success:
            print(f"\n❌ 逆运动学求解失败: {reason}")
            controller.close()
            return
        
        print("\n✅ 逆运动学求解成功!")
        print(f"   目标关节角(°): {np.degrees(q_sol)}")
        print(f"   目标关节角(rad): {q_sol}")
        
        # 验证求解结果
        T_tar = robot.fkine(q_sol)
        print(f"\n   验证结果:")
        print(f"   末端位置: x={T_tar[0,3]:.4f}, y={T_tar[1,3]:.4f}, z={T_tar[2,3]:.4f}")
        pos_error = np.linalg.norm(T_tar[:3, 3] - T_goal[:3, 3])
        print(f"   位置误差: {pos_error*1000:.2f} mm")
        
    except Exception as e:
        print(f"❌ IK 求解异常: {e}")
        import traceback
        traceback.print_exc()
        controller.close()
        return
    
    # 7. 转换为舵机目标步数
    print("\n🔧 转换为舵机目标步数...")
    home_pose = {}
    for name in robot.joint_names:
        home_pose[name] = controller.get_home_position(name)
    
    servo_targets = robot.q_to_servo_targets(q_rad=q_sol, home_pose=home_pose)
    
    # 电子限位保护
    for k in list(servo_targets.keys()):
        servo_targets[k] = controller.limit_position(k, servo_targets[k])
    
    print("\n📋 即将执行的舵机目标步数:")
    current_targets = robot.q_to_servo_targets(q0, home_pose=home_pose)
    for k in robot.joint_names:
        delta = servo_targets[k] - current_targets[k]
        print(f"   {k:15s} : {servo_targets[k]:5d} steps (delta: {delta:+6d})")
    
    # 8. 等待用户确认
    input("\n⏸️  按 Enter 开始平滑执行到目标位姿...")
    
    # 9. 执行平滑移动
    print("\n🚀 执行平滑移动...")
    controller.soft_move_to_pose(servo_targets, step_count=5, interval=0.08)
    
    # 等待舵机执行完毕
    time.sleep(1)
    
    # 10. 读取执行后的实际关节角度
    print("\n✓ 动作完成，读取最终关节角度...")
    q_final = robot.read_joint_angles(joint_names=robot.joint_names, verbose=True)
    
    T_final = robot.fkine(q_final)
    print("\n🔍 最终末端位姿:")
    print(f"   位置: x={T_final[0,3]:.4f}, y={T_final[1,3]:.4f}, z={T_final[2,3]:.4f}")
    pos_error = np.linalg.norm(T_final[:3, 3] - T_goal[:3, 3])
    print(f"   与目标的误差: {pos_error*1000:.2f} mm")
    
    # 11. 实时监控
    print("\n📊 开始实时监控（Ctrl+C 退出）...")
    try:
        while True:
            q_m = robot.read_joint_angles()
            T_m = robot.fkine(q_m)
            print("\r   当前位置: x={:.4f}, y={:.4f}, z={:.4f} m".format(
                T_m[0, 3], T_m[1, 3], T_m[2, 3]
            ), end='', flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\n🛑 退出监控")
    finally:
        controller.close()
        print("✓ 舵机已安全关闭")


if __name__ == "__main__":
    main()
