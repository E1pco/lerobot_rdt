#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ik.robot import create_so101_5dof


class IKSolverNode(Node):
    def __init__(self):
        super().__init__('so101_ik_solver')
        
        # 创建机器人模型
        self.robot = create_so101_5dof()
        self.get_logger().info(f'✅ 机器人模型已加载: {self.robot.n} 自由度')
        self.get_logger().info(f'   关节: {self.robot.joint_names}')
        
        # 初始化 IK 参数
        self.q_current = np.zeros(5)
        self.T_target = None
        
        # 发布器
        self.pub_joint_states = self.create_publisher(
            JointState, 'joint_states_ik', 10
        )
        self.pub_markers = self.create_publisher(
            MarkerArray, 'visualization_marker_array', 10
        )
        
        # 订阅器
        self.sub_target_pose = self.create_subscription(
            PoseStamped, 'target_pose', self.target_pose_callback, 10
        )
        
        # 创建一个定时器来持续发布
        self.timer = self.create_timer(0.1, self.publish_visualization)
        
        self.get_logger().info('🚀 IK 求解器节点已启动')
        self.get_logger().info('   订阅话题: /target_pose (geometry_msgs/PoseStamped)')
        self.get_logger().info('   发布话题: /joint_states_ik (sensor_msgs/JointState)')
        self.get_logger().info('   发布话题: /visualization_marker_array (visualization_msgs/MarkerArray)')
    
    def target_pose_callback(self, msg):
        """接收目标位姿并求解 IK"""
        try:
            # 提取位置
            pos = msg.pose.position
            x, y, z = pos.x, pos.y, pos.z
            
            # 提取姿态（四元数转欧拉角）
            quat = msg.pose.orientation
            R_mat = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
            roll, pitch, yaw = R.from_matrix(R_mat).as_euler('xyz')
            
            # 构建目标变换矩阵
            self.T_target = np.eye(4)
            self.T_target[:3, :3] = R_mat
            self.T_target[:3, 3] = [x, y, z]
            
            self.get_logger().info(f'📍 收到目标位姿:')
            self.get_logger().info(f'   位置: ({x:.4f}, {y:.4f}, {z:.4f}) m')
            self.get_logger().info(f'   姿态: RPY=({np.degrees(roll):.2f}°, {np.degrees(pitch):.2f}°, {np.degrees(yaw):.2f}°)')
            
            # 运行 IK 求解（只关心位置，忽略姿态）
            self.get_logger().info('🔄 运行 IK 求解...')
            
            sol = self.robot.ikine_LM(
                Tep=self.T_target,
                q0=self.q_current,
                ilimit=3000,
                slimit=150,
                tol=1e-5,
                mask=np.array([1, 1, 1, 0, 0, 0]),  # 只约束位置
                k=0.1,
                method="sugihara"
            )
            
            if sol.success:
                self.q_current = sol.q
                
                # 验证结果
                T_result = self.robot.fkine(self.q_current)
                pos_error = np.linalg.norm(T_result[:3, 3] - self.T_target[:3, 3])
                
                self.get_logger().info(f'✅ IK 求解成功')
                self.get_logger().info(f'   关节角: {np.round(np.degrees(self.q_current), 2)}°')
                self.get_logger().info(f'   末端位置误差: {pos_error*1000:.2f} mm')
                
                # 发布关节状态
                self.publish_joint_states()
                
                # 发布可视化标记
                self.publish_visualization()
            else:
                self.get_logger().warn(f'❌ IK 求解失败: {sol.reason}')
                
        except Exception as e:
            self.get_logger().error(f'❌ 处理目标位姿出错: {e}', once=True)
    
    def publish_joint_states(self):
        """发布关节状态"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.robot.joint_names
        msg.position = self.q_current.tolist()
        msg.velocity = [0.0] * len(self.robot.joint_names)
        msg.effort = [0.0] * len(self.robot.joint_names)
        
        self.pub_joint_states.publish(msg)
    
    def publish_visualization(self):
        """发布可视化标记"""
        markers = MarkerArray()
        
        # 计算末端位姿
        T_ee = self.robot.fkine(self.q_current)
        
        # 1. 末端执行器位置
        marker_ee = Marker()
        marker_ee.header = Header()
        marker_ee.header.stamp = self.get_clock().now().to_msg()
        marker_ee.header.frame_id = 'base_link'
        marker_ee.id = 0
        marker_ee.type = Marker.SPHERE
        marker_ee.action = Marker.ADD
        marker_ee.pose.position = Point(x=T_ee[0, 3], y=T_ee[1, 3], z=T_ee[2, 3])
        marker_ee.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        marker_ee.scale.x = 0.02
        marker_ee.scale.y = 0.02
        marker_ee.scale.z = 0.02
        marker_ee.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # 绿色
        markers.markers.append(marker_ee)
        
        # 2. 目标位置
        if self.T_target is not None:
            marker_target = Marker()
            marker_target.header = Header()
            marker_target.header.stamp = self.get_clock().now().to_msg()
            marker_target.header.frame_id = 'base_link'
            marker_target.id = 1
            marker_target.type = Marker.CUBE
            marker_target.action = Marker.ADD
            marker_target.pose.position = Point(x=self.T_target[0, 3], y=self.T_target[1, 3], z=self.T_target[2, 3])
            marker_target.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            marker_target.scale.x = 0.015
            marker_target.scale.y = 0.015
            marker_target.scale.z = 0.015
            marker_target.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # 红色
            markers.markers.append(marker_target)
            
            # 3. 连接线（从末端到目标）
            marker_line = Marker()
            marker_line.header = Header()
            marker_line.header.stamp = self.get_clock().now().to_msg()
            marker_line.header.frame_id = 'base_link'
            marker_line.id = 2
            marker_line.type = Marker.LINE_STRIP
            marker_line.action = Marker.ADD
            marker_line.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            marker_line.scale.x = 0.005  # 线宽
            marker_line.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.7)  # 黄色
            
            marker_line.points.append(Point(x=T_ee[0, 3], y=T_ee[1, 3], z=T_ee[2, 3]))
            marker_line.points.append(Point(x=self.T_target[0, 3], y=self.T_target[1, 3], z=self.T_target[2, 3]))
            
            markers.markers.append(marker_line)
        
        # 4. 末端坐标轴
        self.add_axis_marker(markers, T_ee, marker_id=3)
        
        self.pub_markers.publish(markers)
    
    def add_axis_marker(self, markers, T, marker_id=3):
        """添加坐标轴标记"""
        axis_size = 0.05
        
        # X 轴 (红色)
        marker_x = Marker()
        marker_x.header = Header()
        marker_x.header.stamp = self.get_clock().now().to_msg()
        marker_x.header.frame_id = 'base_link'
        marker_x.id = marker_id
        marker_x.type = Marker.ARROW
        marker_x.action = Marker.ADD
        marker_x.pose.position = Point(x=T[0, 3], y=T[1, 3], z=T[2, 3])
        marker_x.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        marker_x.scale.x = axis_size
        marker_x.scale.y = 0.005
        marker_x.scale.z = 0.005
        marker_x.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
        markers.markers.append(marker_x)
        
        # Y 轴 (绿色)
        marker_y = Marker()
        marker_y.header = Header()
        marker_y.header.stamp = self.get_clock().now().to_msg()
        marker_y.header.frame_id = 'base_link'
        marker_y.id = marker_id + 1
        marker_y.type = Marker.ARROW
        marker_y.action = Marker.ADD
        marker_y.pose.position = Point(x=T[0, 3], y=T[1, 3], z=T[2, 3])
        marker_y.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        marker_y.scale.x = axis_size
        marker_y.scale.y = 0.005
        marker_y.scale.z = 0.005
        marker_y.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.7)
        markers.markers.append(marker_y)
        
        # Z 轴 (蓝色)
        marker_z = Marker()
        marker_z.header = Header()
        marker_z.header.stamp = self.get_clock().now().to_msg()
        marker_z.header.frame_id = 'base_link'
        marker_z.id = marker_id + 2
        marker_z.type = Marker.ARROW
        marker_z.action = Marker.ADD
        marker_z.pose.position = Point(x=T[0, 3], y=T[1, 3], z=T[2, 3])
        marker_z.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        marker_z.scale.x = axis_size
        marker_z.scale.y = 0.005
        marker_z.scale.z = 0.005
        marker_z.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.7)
        markers.markers.append(marker_z)


def main(args=None):
    rclpy.init(args=args)
    node = IKSolverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
