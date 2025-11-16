#!/usr/bin/env python3
"""
ROS2 关节状态发布器 - 用于控制 SO-101 机械臂仿真
可以发布预定义的轨迹或者从 IK 求解器获取的关节角度
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np
import time


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('so101_joint_publisher')
        
        # 创建发布器
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # 定义关节名称（必须与 URDF 中的 joint 名称一致）
        self.joint_names = [
            'shoulder_pan',
            'shoulder_lift', 
            'elbow_flex',
            'wrist_flex',
            'wrist_roll'
        ]
        
        # 定时器：以 50Hz 频率发布
        self.timer = self.create_timer(0.02, self.publish_joint_states)
        
        self.get_logger().info('SO-101 Joint State Publisher 已启动')
        self.get_logger().info(f'关节列表: {self.joint_names}')
        
        # 初始化时间和轨迹参数
        self.start_time = time.time()
        self.trajectory_mode = 'sine'  # 可选: 'sine', 'static', 'custom'
        
    def publish_joint_states(self):
        """发布关节状态"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        # 根据模式生成关节角度
        if self.trajectory_mode == 'sine':
            # 正弦波轨迹（演示用）
            t = time.time() - self.start_time
            msg.position = [
                0.2 * np.sin(0.5 * t),           # shoulder_pan
                0.3 * np.sin(0.8 * t),           # shoulder_lift
                0.3 * np.sin(1.0 * t + 0.5),     # elbow_flex
                0.2 * np.sin(1.2 * t - 0.3),     # wrist_flex
                0.1 * np.sin(1.5 * t)            # wrist_roll
            ]
        elif self.trajectory_mode == 'static':
            # 静态姿态
            msg.position = [0.0, -0.5, 0.5, 0.0, 0.0]
        else:
            # 默认零位
            msg.position = [0.0] * len(self.joint_names)
        
        # 速度和力矩（可选，留空）
        msg.velocity = []
        msg.effort = []
        
        self.publisher.publish(msg)
    
    def set_joint_positions(self, positions):
        """
        设置自定义关节位置
        
        Parameters
        ----------
        positions : list or np.ndarray
            关节角度列表（rad）
        """
        if len(positions) != len(self.joint_names):
            self.get_logger().error(f'关节数量不匹配: 期望{len(self.joint_names)}, 得到{len(positions)}')
            return
        
        self.trajectory_mode = 'custom'
        self.custom_positions = positions


def main(args=None):
    rclpy.init(args=args)
    
    node = JointStatePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
