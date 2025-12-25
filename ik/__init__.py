"""
纯 Python 逆运动学求解器库


主要特性：
- ET/ETS 类：基本变换表示和运动学链
- Robot 类：多种求解算法（LM, GN, NR, QP）直接内置
- 支持冗余机械臂（>6 DoF）
- 清晰的 API 和详细的文档
"""

# ET 和 ETS 类
from .et import ET, ETS, SE3_to_xyzrpy, xyzrpy_to_SE3

# DH 参数建模
from .DH import DHRobot, create_dh_robot, create_puma560, create_stanford_arm, create_so101_5dof

# IK 求解器
from .solver import IKResult, ikine_LM, ikine_GN, ikine_NR, ikine_QP

# 工具函数
from .utils import angle_axis, p_servo

# 机器人模型
from .robot import (
    create_so101_5dof,
    get_robot,
    smooth_joint_motion,
    Robot,
)

__version__ = "0.1.0"

__all__ = [
    # ET/ETS
    "ET",
    "ETS",
    "SE3_to_xyzrpy",
    "xyzrpy_to_SE3",
    
    # DH 参数建模
    "DHRobot",
    "create_dh_robot",
    "create_puma560",
    "create_stanford_arm",
    "create_so101_5dof",
    
    # IK 求解器
    "IKResult",
    "ikine_LM",
    "ikine_GN",
    "ikine_NR",
    "ikine_QP",
    
    # 工具函数
    "angle_axis",
    "p_servo",
    
    # 机器人模型
    "create_so101_5dof",
    "get_robot",
    "Robot",
    "smooth_joint_motion",
]
