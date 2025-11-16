#!/usr/bin/env python3
"""
目标位姿发布演示脚本
定期发布不同的目标位姿，使 IK 求解器计算对应的关节角度
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np


class TargetPosePublisher(Node):
    def __init__(self):
        super().__init__('target_pose_publisher')
        
        self.pub = self.create_publisher(PoseStamped, 'target_pose', 10)
        
        # 定义演示轨迹
        self.targets = [
            {
                'name': '目标 1',
                'x': 0.0, 'y': -0.25, 'z': 0.15,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'duration': 5  # 保持 5 秒
            },
            {
                'name': '目标 2',
                'x': 0.1, 'y': -0.20, 'z': 0.20,
                'roll': 0.0, 'pitch': np.pi/6, 'yaw': 0.0,
                'duration': 5
            },
            {
                'name': '目标 3',
                'x': -0.1, 'y': -0.30, 'z': 0.10,
                'roll': 0.0, 'pitch': -np.pi/6, 'yaw': np.pi/4,
                'duration': 5
            },
        ]
        
        self.current_target_idx = 0
        self.target_start_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        self.timer = self.create_timer(0.5, self.publish_target)
        
        self.get_logger().info('🎯 目标位姿发布器已启动')
        self.get_logger().info('   发布话题: /target_pose')
        self.get_logger().info(f'   演示轨迹: {len(self.targets)} 个目标位姿')
    
    def publish_target(self):
        """发布当前目标位姿"""
        target = self.targets[self.current_target_idx]
        
        # 检查是否需要切换到下一个目标
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = current_time - self.target_start_time
        
        if elapsed > target['duration']:
            self.current_target_idx = (self.current_target_idx + 1) % len(self.targets)
            self.target_start_time = current_time
            target = self.targets[self.current_target_idx]
            self.get_logger().info(f"🔄 切换到 {target['name']}")
        
        # 构建位姿消息
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        # 位置
        msg.pose.position = Point(
            x=float(target['x']),
            y=float(target['y']),
            z=float(target['z'])
        )
        
        # 姿态（欧拉角转四元数）
        quat = R.from_euler('xyz', [
            target['roll'],
            target['pitch'],
            target['yaw']
        ]).as_quat()
        
        msg.pose.orientation = Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3])
        )
        
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetPosePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
