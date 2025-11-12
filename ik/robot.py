"""
纯 Python 机器人运动学实现

提供 SO100 和 SO101 机械臂的正逆运动学计算，无需 C++ 扩展。
"""

import numpy as np
import math
from scipy.spatial.transform import Rotation as R

# 支持直接运行和模块导入
try:
    from .et import ET, ETS
    from .solvers import IK_LM, IK_GN, IK_NR, IK_QP
except ImportError:
    # 直接运行时使用绝对导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from ik.et import ET, ETS
    from ik.solvers import IK_LM, IK_GN, IK_NR, IK_QP


def atan2(first, second):
    """保留3位小数的 atan2"""
    return round(math.atan2(first, second), 3)


def sin(radians_angle):
    """保留3位小数的 sin"""
    return round(math.sin(radians_angle), 3)


def cos(radians_angle):
    """保留3位小数的 cos"""
    return round(math.cos(radians_angle), 3)


def acos(value):
    """保留3位小数的 acos"""
    return round(math.acos(value), 3)

class IKResult:
    """IK 求解结果封装类，兼容 roboticstoolbox 接口"""
    def __init__(self, success, q, reason=""):
        self.success = success
        self.q = q
        self.reason = reason


class Robot:
    """
    机器人封装类，提供与 roboticstoolbox 兼容的 API
    
    Attributes
    ----------
    ets : ETS
        Elementary Transform Sequence
    n : int
        关节数量
    qlim : np.ndarray
        关节限位 (2, n)
    """
    
    def __init__(self, ets, qlim=None):
        """
        初始化机器人模型
        
        Parameters
        ----------
        ets : ETS
            机器人运动学链
        qlim : np.ndarray, optional
            关节限位 (2, n)，第一行为下限，第二行为上限
        """
        self.ets = ets
        self.n = ets.n
        self.qlim = qlim
        
        # 将 qlim 设置到 ETS 对象上（IK solver 会从 ets.qlim 读取）
        if qlim is not None:
            self.ets.qlim = qlim
    
    # ========== 坐标系转换方法 ==========
    def user_to_robot(self, x, y, z):
        """用户坐标 → 机械臂坐标 (对调 X 和 Y)"""
        return y, x, z
    
    def robot_to_user(self, x, y, z):
        """机械臂坐标 → 用户坐标 (对调 X 和 Y)"""
        return y, x, z
    
    def build_pose(self, x, y, z, roll=0, pitch=0, yaw=0):
        """
        构造用户坐标系下的目标位姿矩阵
        
        Parameters
        ----------
        x, y, z : float
            用户坐标系中的位置
        roll, pitch, yaw : float
            欧拉角（弧度）
            
        Returns
        -------
        np.ndarray
            机械臂坐标系下的 4x4 齐次变换矩阵
        """
        x_robot, y_robot, z_robot = self.user_to_robot(x, y, z)
        r = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = [x_robot, y_robot, z_robot]
        return T
    
    def get_user_pose(self, T):
        """
        从机械臂坐标系的齐次变换矩阵获取用户坐标系的位姿
        
        Parameters
        ----------
        T : np.ndarray
            机械臂坐标系下的 4x4 齐次变换矩阵
            
        Returns
        -------
        tuple
            (x, y, z, roll, pitch, yaw) 用户坐标系下的位姿
        """
        # 从机械臂坐标系转换到用户坐标系
        x_robot, y_robot, z_robot = T[0, 3], T[1, 3], T[2, 3]
        x, y, z = self.robot_to_user(x_robot, y_robot, z_robot)
        
        # 提取欧拉角
        rpy = R.from_matrix(T[:3, :3]).as_euler('xyz')
        
        return x, y, z, rpy[0], rpy[1], rpy[2]
    
    def fkine(self, q):
        """
        正运动学计算
        
        Parameters
        ----------
        q : array_like
            关节角度
            
        Returns
        -------
        np.ndarray
            4x4 齐次变换矩阵
        """
        return self.ets.fkine(q)

    def fk(self, qpos_data, joint_indices=None):
        """
        并返回末端执行器位姿向量 [X, Y, Z, Roll, Pitch, Yaw]
        注：对调 X 和 Y 以满足右手系

        Parameters
        ----------
        qpos_data : np.ndarray
            关节角度向量（可以比机器人关节多，会根据 joint_indices 提取）
        joint_indices : list or np.ndarray, optional
            要使用的关节索引。如果为 None，则使用前 n 个关节

        Returns
        -------
        np.ndarray
            末端执行器位姿 [X, Y, Z, Roll, Pitch, Yaw]
        """
        # 如果提供了关节索引，使用索引提取关节角度
        if joint_indices is not None:
            if max(joint_indices) >= len(qpos_data):
                raise Exception(
                    f"Joint index {max(joint_indices)} out of range for qpos_data "
                    f"with length {len(qpos_data)}"
                )
            q = qpos_data[joint_indices]
        else:
            # 否则，检查长度并提取前 n 个
            if len(qpos_data) < self.n:
                raise Exception(
                    f"The dimensions of qpos_data ({len(qpos_data)}) "
                    f"is less than the robot joint dimensions ({self.n})"
                )
            q = qpos_data[:self.n]

        # 计算正运动学，获取齐次变换矩阵
        T = self.fkine(q)

        # 提取位置并对调 X 和 Y（满足右手系）
        X, Y, Z = T[0, 3], T[1, 3], T[2, 3]
        X, Y = Y, X  # 对调 X 和 Y

        # 提取旋转矩阵并计算欧拉角 (XYZ -> Roll, Pitch, Yaw)
        R_mat = T[:3, :3]

        beta = atan2(-R_mat[2, 0], math.sqrt(R_mat[0, 0]**2 + R_mat[1, 0]**2))

        if cos(beta) != 0:
            alpha = atan2(R_mat[1, 0] / cos(beta), R_mat[0, 0] / cos(beta))
            gamma = atan2(R_mat[2, 1] / cos(beta), R_mat[2, 2] / cos(beta))
        else:
            # 万向节锁情况
            alpha = 0
            gamma = atan2(R_mat[0, 1], R_mat[1, 1])

        return np.array([X, Y, Z, gamma, beta, alpha])
    
    def ikine_LM(self, Tep, q0=None, ilimit=100, slimit=10, tol=1e-3, mask=None, 
                 k=1.0, method='chan'):
        """
        使用 Levenberg-Marquardt 方法求解逆运动学
        
        Parameters
        ----------
        Tep : np.ndarray
            目标位姿 (4x4 齐次变换矩阵)
        q0 : array_like, optional
            初始关节角度，默认为零向量
        ilimit : int
            最大迭代次数
        slimit : int
            搜索次数限制
        tol : float
            收敛容差
        mask : array_like, optional
            位姿权重 [x, y, z, roll, pitch, yaw]，0 表示忽略该维度
        k : float
            LM 阻尼系数
        method : str
            LM 更新方法 ('chan', 'wampler', 'sugihara')
            
        Returns
        -------
        IKResult
            求解结果，包含 .success, .q, .reason 属性
        """
        if q0 is None:
            q0 = np.zeros(self.n)
        
        solver = IK_LM(ilimit=ilimit, slimit=slimit, tol=tol, k=k, method=method)
        
        # 构建默认 mask
        if mask is None:
            mask = np.ones(6)
        else:
            mask = np.asarray(mask)
        
        # 求解
        sol = solver.solve(self.ets, Tep, q0=q0, mask=mask)

        return IKResult(sol.success, sol.q, sol.reason)
    
    def ikine_GN(self, Tep, q0=None, ilimit=50, tol=1e-3, mask=None, pinv=False):
        """
        使用 Gauss-Newton 方法求解逆运动学
        
        Parameters
        ----------
        Tep : np.ndarray
            目标位姿 (4x4 齐次变换矩阵)
        q0 : array_like, optional
            初始关节角度
        ilimit : int
            最大迭代次数
        tol : float
            收敛容差
        mask : array_like, optional
            位姿权重
        pinv : bool
            是否使用伪逆
            
        Returns
        -------
        IKResult
            求解结果
        """
        if q0 is None:
            q0 = np.zeros(self.n)
        
        solver = IK_GN(ilimit=ilimit, tol=tol, pinv=pinv)
        
        if mask is None:
            mask = np.ones(6)
        else:
            mask = np.asarray(mask)
        
        sol = solver.solve(self.ets, Tep, q0=q0, mask=mask)
        return IKResult(sol.success, sol.q, sol.reason)
    
    def ikine_NR(self, Tep, q0=None, ilimit=50, tol=1e-3, mask=None, pinv=False):
        """使用 Newton-Raphson 方法求解逆运动学"""
        if q0 is None:
            q0 = np.zeros(self.n)
        
        solver = IK_NR(ilimit=ilimit, tol=tol, pinv=pinv)
        
        if mask is None:
            mask = np.ones(6)
        else:
            mask = np.asarray(mask)
        
        sol = solver.solve(self.ets, Tep, q0=q0, mask=mask)
        return IKResult(sol.success, sol.q, sol.reason)
    
    def ikine_QP(self, Tep, q0=None, ilimit=50, tol=1e-3, mask=None, 
                 kj=0.01, ks=1.0):
        """使用二次规划方法求解逆运动学"""
        if q0 is None:
            q0 = np.zeros(self.n)
        
        solver = IK_QP(ilimit=ilimit, tol=tol, kj=kj, ks=ks)
        
        if mask is None:
            mask = np.ones(6)
        else:
            mask = np.asarray(mask)
        
        sol = solver.solve(self.ets, Tep, q0=q0, mask=mask)
        return IKResult(sol.success, sol.q, sol.reason)


def create_so101_5dof():
    """
    创建 SO-101 5自由度机械臂模型
    
    Returns
    -------
    Robot
        封装后的机器人对象，具有 .fkine() 和 .ikine_*() 方法
    """
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

    ets = E1 * E2 * E3 * E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15

    # 自动同步URDF中的限位
    qlim = np.array([
        [-1.91986, -1.74533, -1.69, -1.65806, -2.74385],
        [ 1.91986,  1.74533,  1.69,  1.65806,  2.84121]
    ])
    
    return Robot(ets, qlim)




def get_robot(robot="so101"):
    """
    获取指定的机器人模型
    
    Parameters
    ----------
    robot : str
        机器人类型： 'so101_5dof'
        
    Returns
    -------
    ETS or None
        机器人的运动学模型
    """

    if robot == "so101_5dof":
        return create_so101_5dof()
    else:
        print(f"Sorry, we don't support {robot} robot now")
        return None

def smooth_joint_motion(q_now, q_new, robot, max_joint_change=0.1):
    """
    平滑关节运动，限制单步最大变化量
    
    Parameters
    ----------
    q_now : np.ndarray
        当前关节角度
    q_new : np.ndarray
        新的关节角度
    robot : ETS
        机器人运动学模型
    max_joint_change : float
        单步允许的最大关节变化量
        
    Returns
    -------
    np.ndarray
        平滑后的关节角度
    """
    q_smoothed = q_new.copy()
    
    for i in range(len(q_new)):
        delta = q_new[i] - q_now[i]
        if abs(delta) > max_joint_change:
            delta = np.sign(delta) * max_joint_change
        q_smoothed[i] = q_now[i] + delta
    
    return q_smoothed
if __name__ == "__main__":
    robot = create_so101_5dof()
    qpos_data = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
    T = robot.fkine(qpos_data)
    print(T)
