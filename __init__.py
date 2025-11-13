"""
LeRobot RDT - Robot Demonstration Toolkit

主要模块:
  - driver: 舵机控制器 (ServoController)
  - ik: 机器人运动学 (create_so101_5dof)
  - joyconrobotics: Joy-Con 手柄控制
  - vision: 视觉相关功能
"""

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof, Robot

__version__ = "1.0.0"
__all__ = [
    "ServoController",
    "create_so101_5dof",
    "Robot",
]
