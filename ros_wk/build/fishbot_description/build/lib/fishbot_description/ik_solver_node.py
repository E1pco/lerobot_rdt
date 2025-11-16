#!/usr/bin/env python3
"""
SO-101 机械臂 IK 求解器 ROS2 节点（改进版）
在 ros_wk 工作空间中运行
- 订阅目标位姿话题
- 运行 IK 求解
- 发布关节状态和可视化标记到 RViz
- 提供服务和命令行交互允许用户输入目标位姿
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import threading

# 添加 lerobot_rdt 到路径
sys.path.insert(0, '/home/elpco/code/lerobot/lerobot_rdt')

from ik.robot import create_so101_5dof


class SO101IKSolverNode(Node):
    def __init__(self):
        super().__init__('so101_ik_solver')
        
        # 创建机器人模型
        self.robot = create_so101_5dof()
        self.get_logger().info(f'✅ 机器人模型已加载: {self.robot.n} 自由度')
        self.get_logger().info(f'   关节: {self.robot.joint_names}')
        
        # 初始化状态
        self.q_current = np.zeros(5)
        self.T_current = self.robot.fkine(self.q_current)
        self.T_target = None
        self.T_last_target = None
        self.trajectory_points = []
        
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
        
        # 定时器：定期发布可视化
        self.timer = self.create_timer(0.1, self.publish_visualization)
        
        # 启动用户交互线程
        self.user_input_thread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.user_input_thread.start()
        
        self.get_logger().info('🚀 IK 求解器节点已启动')
        self.get_logger().info('   订阅话题: /target_pose (geometry_msgs/PoseStamped)')
        self.get_logger().info('   发布话题: /joint_states_ik (sensor_msgs/JointState)')
        self.get_logger().info('   发布话题: /visualization_marker_array (visualization_msgs/MarkerArray)')
        self.get_logger().info('')
        self.get_logger().info('   使用方法（3种）:')
        self.get_logger().info('   1. ROS2 话题发布 (自动):')
        self.get_logger().info('      ros2 topic pub /target_pose geometry_msgs/PoseStamped ')
        self.get_logger().info('        "{header: {frame_id: base_link}, pose: {position: {x: 0.0, y: -0.3, z: 0.15}, ')
        self.get_logger().info('        orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"')
        self.get_logger().info('   2. 命令行交互 (输入 \'i\' 或 \'help\'):')
        self.get_logger().info('   3. 预定义位姿 (输入 \'p\'):')
        self.get_logger().info('')
    
    def target_pose_callback(self, msg):
        """接收目标位姿并求解 IK"""
        try:
            # 提取位置
            pos = msg.pose.position
            x, y, z = pos.x, pos.y, pos.z
            
            # 提取姿态（四元数转旋转矩阵）
            quat = msg.pose.orientation
            R_mat = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
            
            # 构建目标变换矩阵
            self.T_target = np.eye(4)
            self.T_target[:3, :3] = R_mat
            self.T_target[:3, 3] = [x, y, z]
            
            # 转换为欧拉角显示
            roll, pitch, yaw = R.from_matrix(R_mat).as_euler('xyz')
            
            self.get_logger().info(f'📍 收到目标位姿:')
            self.get_logger().info(f'   位置: X={x:.4f}, Y={y:.4f}, Z={z:.4f} m')
            self.get_logger().info(f'   姿态: R={np.degrees(roll):.2f}°, P={np.degrees(pitch):.2f}°, Y={np.degrees(yaw):.2f}°')
            
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
                self.T_current = self.robot.fkine(self.q_current)
                
                # 验证结果
                pos_error = np.linalg.norm(self.T_current[:3, 3] - self.T_target[:3, 3])
                
                self.get_logger().info(f'✅ IK 求解成功')
                self.get_logger().info(f'   关节角(°): {np.round(np.degrees(self.q_current), 2)}')
                self.get_logger().info(f'   末端误差: {pos_error*1000:.2f} mm')
                
                # 添加到轨迹
                self.trajectory_points.append(self.T_current[:3, 3].copy())
                if len(self.trajectory_points) > 100:
                    self.trajectory_points.pop(0)
                
                # 立即发布关节状态
                self.publish_joint_states()
                self.publish_visualization()
            else:
                self.get_logger().warn(f'❌ IK 求解失败: {sol.reason}')
                
        except Exception as e:
            self.get_logger().error(f'❌ 处理目标位姿出错: {str(e)}')
            import traceback
            traceback.print_exc()
    
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
        """发布可视化标记到 RViz"""
        markers = MarkerArray()
        marker_id = 0
        
        # 1. 当前末端执行器位置（绿色球体）
        marker_ee = Marker()
        marker_ee.header = Header()
        marker_ee.header.stamp = self.get_clock().now().to_msg()
        marker_ee.header.frame_id = 'base_link'
        marker_ee.id = marker_id
        marker_id += 1
        marker_ee.type = Marker.SPHERE
        marker_ee.action = Marker.ADD
        marker_ee.pose.position = Point(
            x=float(self.T_current[0, 3]), 
            y=float(self.T_current[1, 3]), 
            z=float(self.T_current[2, 3])
        )
        marker_ee.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        marker_ee.scale = Vector3(x=0.025, y=0.025, z=0.025)
        marker_ee.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # 绿色
        marker_ee.text = "EE"
        markers.markers.append(marker_ee)
        
        # 2. 目标位置（红色立方体）
        if self.T_target is not None:
            marker_target = Marker()
            marker_target.header = Header()
            marker_target.header.stamp = self.get_clock().now().to_msg()
            marker_target.header.frame_id = 'base_link'
            marker_target.id = marker_id
            marker_id += 1
            marker_target.type = Marker.CUBE
            marker_target.action = Marker.ADD
            marker_target.pose.position = Point(
                x=float(self.T_target[0, 3]), 
                y=float(self.T_target[1, 3]), 
                z=float(self.T_target[2, 3])
            )
            marker_target.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            marker_target.scale = Vector3(x=0.02, y=0.02, z=0.02)
            marker_target.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # 红色
            markers.markers.append(marker_target)
            
            # 3. 连接线（黄色）
            marker_line = Marker()
            marker_line.header = Header()
            marker_line.header.stamp = self.get_clock().now().to_msg()
            marker_line.header.frame_id = 'base_link'
            marker_line.id = marker_id
            marker_id += 1
            marker_line.type = Marker.LINE_STRIP
            marker_line.action = Marker.ADD
            marker_line.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            marker_line.scale = Vector3(x=0.008, y=0.0, z=0.0)
            marker_line.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.7)  # 黄色
            
            marker_line.points.append(Point(
                x=float(self.T_current[0, 3]), 
                y=float(self.T_current[1, 3]), 
                z=float(self.T_current[2, 3])
            ))
            marker_line.points.append(Point(
                x=float(self.T_target[0, 3]), 
                y=float(self.T_target[1, 3]), 
                z=float(self.T_target[2, 3])
            ))
            
            markers.markers.append(marker_line)
        
        # 4. 轨迹（青色线）
        if len(self.trajectory_points) > 1:
            marker_traj = Marker()
            marker_traj.header = Header()
            marker_traj.header.stamp = self.get_clock().now().to_msg()
            marker_traj.header.frame_id = 'base_link'
            marker_traj.id = marker_id
            marker_id += 1
            marker_traj.type = Marker.LINE_STRIP
            marker_traj.action = Marker.ADD
            marker_traj.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            marker_traj.scale = Vector3(x=0.003, y=0.0, z=0.0)
            marker_traj.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.5)  # 青色
            
            for point in self.trajectory_points:
                marker_traj.points.append(Point(
                    x=float(point[0]), 
                    y=float(point[1]), 
                    z=float(point[2])
                ))
            
            markers.markers.append(marker_traj)
        
        # 5. 末端坐标轴
        axis_size = 0.08
        axis_colors = [
            (1.0, 0.0, 0.0),  # X 轴 红色
            (0.0, 1.0, 0.0),  # Y 轴 绿色
            (0.0, 0.0, 1.0),  # Z 轴 蓝色
        ]
        
        for axis_idx in range(3):
            marker_axis = Marker()
            marker_axis.header = Header()
            marker_axis.header.stamp = self.get_clock().now().to_msg()
            marker_axis.header.frame_id = 'base_link'
            marker_axis.id = marker_id
            marker_id += 1
            marker_axis.type = Marker.ARROW
            marker_axis.action = Marker.ADD
            
            # 箭头起点
            marker_axis.pose.position = Point(
                x=float(self.T_current[0, 3]), 
                y=float(self.T_current[1, 3]), 
                z=float(self.T_current[2, 3])
            )
            
            # 箭头方向（沿坐标轴）
            direction = self.T_current[:3, axis_idx]
            end_point = self.T_current[:3, 3] + axis_size * direction
            
            # 使用旋转矩阵计算四元数
            z_axis = direction / np.linalg.norm(direction)
            x_axis = np.array([1, 0, 0]) if axis_idx != 0 else np.array([0, 1, 0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            R_axis = np.column_stack([x_axis, y_axis, z_axis])
            quat = R.from_matrix(R_axis).as_quat()
            
            marker_axis.pose.orientation = Quaternion(
                x=float(quat[0]), 
                y=float(quat[1]), 
                z=float(quat[2]), 
                w=float(quat[3])
            )
            
            marker_axis.scale = Vector3(x=axis_size, y=0.006, z=0.006)
            r, g, b = axis_colors[axis_idx]
            marker_axis.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=0.8)
            
            markers.markers.append(marker_axis)
        
        self.pub_markers.publish(markers)
    
    def user_input_loop(self):
        """用户交互线程：处理命令行输入"""
        print("\n" + "="*70)
        print("SO-101 IK 求解器 - 交互模式")
        print("="*70)
        print("输入 'help' 显示帮助信息")
        print("="*70 + "\n")
        
        while True:
            try:
                # 读取用户输入
                cmd = input("请输入命令 (help/i/p/q): ").strip().lower()
                
                if cmd in ['help', 'h', '?']:
                    self.show_help()
                
                elif cmd in ['input', 'i']:
                    self.input_target_pose()
                
                elif cmd in ['preset', 'p']:
                    self.show_presets()
                
                elif cmd in ['quit', 'q', 'exit']:
                    print("👋 退出程序...")
                    break
                
                elif cmd == '':
                    continue
                
                else:
                    print(f"❌ 未知命令: '{cmd}'，输入 'help' 查看帮助")
                    
            except KeyboardInterrupt:
                print("\n👋 收到退出信号，正在关闭...")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    def show_help(self):
        """显示帮助信息"""
        print("\n" + "="*70)
        print("📖 帮助信息")
        print("="*70)
        print("命令列表:")
        print("  help, h, ?     - 显示此帮助信息")
        print("  input, i       - 手动输入目标位姿 (x, y, z, roll, pitch, yaw)")
        print("  preset, p      - 从预定义位姿列表中选择")
        print("  quit, q, exit  - 退出程序")
        print("\n位姿单位:")
        print("  位置 (x, y, z): 米 (m)")
        print("  姿态 (roll, pitch, yaw): 度 (°)")
        print("\n示例:")
        print("  输入目标位置: x=-0.1, y=-0.3, z=0.1")
        print("  输入欧拉角: roll=45, pitch=-30, yaw=0")
        print("="*70 + "\n")
    
    def input_target_pose(self):
        """交互式输入目标位姿"""
        print("\n" + "="*70)
        print("📍 手动输入目标位姿")
        print("="*70)
        
        try:
            # 输入位置
            print("\n📌 输入末端位置 (单位: 米)")
            x = float(input("  x = "))
            y = float(input("  y = "))
            z = float(input("  z = "))
            
            # 输入姿态
            print("\n🔄 输入欧拉角 (单位: 度)")
            roll = float(input("  roll (°) = "))
            pitch = float(input("  pitch (°) = "))
            yaw = float(input("  yaw (°) = "))
            
            # 转换为弧度
            roll_rad = np.radians(roll)
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            
            # 构建目标位姿
            r = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad], degrees=False)
            T_goal = np.eye(4)
            T_goal[:3, :3] = r.as_matrix()
            T_goal[:3, 3] = [x, y, z]
            
            # 显示输入的位姿
            print(f"\n✅ 已输入目标位姿:")
            print(f"   位置: ({x:.4f}, {y:.4f}, {z:.4f}) m")
            print(f"   姿态: roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°")
            
            # 立即执行 IK 求解
            self.solve_ik(T_goal)
            
        except ValueError as e:
            print(f"❌ 输入错误: {e}，请输入有效的数字")
        except Exception as e:
            print(f"❌ 异常: {e}")
    
    def show_presets(self):
        """显示预定义位姿"""
        presets = {
            '1': {
                'name': '初始位置 (Home)',
                'x': 0.0, 'y': -0.25, 'z': 0.25,
                'roll': 0, 'pitch': 0, 'yaw': 0
            },
            '2': {
                'name': '左前方',
                'x': -0.1, 'y': -0.3, 'z': 0.1,
                'roll': 45, 'pitch': -30, 'yaw': 45
            },
            '3': {
                'name': '右前方',
                'x': 0.1, 'y': -0.2, 'z': 0.2,
                'roll': -45, 'pitch': 30, 'yaw': -45
            },
            '4': {
                'name': '正上方',
                'x': 0.0, 'y': -0.25, 'z': 0.35,
                'roll': 0, 'pitch': -90, 'yaw': 0
            },
            '5': {
                'name': '侧面',
                'x': 0.3, 'y': 0.0, 'z': 0.2,
                'roll': 0, 'pitch': 0, 'yaw': 90
            }
        }
        
        print("\n" + "="*70)
        print("🎯 预定义位姿")
        print("="*70)
        for key, pose in presets.items():
            print(f"\n{key}. {pose['name']}")
            print(f"   位置: ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}) m")
            print(f"   姿态: R={pose['roll']:.0f}°, P={pose['pitch']:.0f}°, Y={pose['yaw']:.0f}°")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice in presets:
            pose = presets[choice]
            print(f"\n✅ 已选择: {pose['name']}")
            
            # 构建目标位姿
            roll_rad = np.radians(pose['roll'])
            pitch_rad = np.radians(pose['pitch'])
            yaw_rad = np.radians(pose['yaw'])
            
            r = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad], degrees=False)
            T_goal = np.eye(4)
            T_goal[:3, :3] = r.as_matrix()
            T_goal[:3, 3] = [pose['x'], pose['y'], pose['z']]
            
            # 执行 IK 求解
            self.solve_ik(T_goal)
        else:
            print(f"❌ 无效选择: {choice}")
    
    def solve_ik(self, T_target):
        """执行 IK 求解并发布结果"""
        try:
            self.T_target = T_target
            
            roll, pitch, yaw = R.from_matrix(T_target[:3, :3]).as_euler('xyz')
            self.get_logger().info(f'📍 开始求解 IK:')
            self.get_logger().info(f'   位置: X={T_target[0,3]:.4f}, Y={T_target[1,3]:.4f}, Z={T_target[2,3]:.4f} m')
            self.get_logger().info(f'   姿态: R={np.degrees(roll):.2f}°, P={np.degrees(pitch):.2f}°, Y={np.degrees(yaw):.2f}°')
            self.get_logger().info('🔄 运行 IK 求解...')
            
            sol = self.robot.ikine_LM(
                Tep=T_target,
                q0=self.q_current,
                ilimit=3000,
                slimit=150,
                tol=1e-5,
                mask=np.array([1, 1, 1, 0, 0, 0]),
                k=0.1,
                method="sugihara"
            )
            
            if sol.success:
                self.q_current = sol.q
                self.T_current = self.robot.fkine(self.q_current)
                
                pos_error = np.linalg.norm(self.T_current[:3, 3] - T_target[:3, 3])
                
                self.get_logger().info(f'✅ IK 求解成功!')
                self.get_logger().info(f'   关节角(°): {np.round(np.degrees(self.q_current), 2)}')
                self.get_logger().info(f'   末端误差: {pos_error*1000:.2f} mm')
                
                print(f"\n✅ IK 求解成功!")
                print(f"   关节角(°): {np.round(np.degrees(self.q_current), 2)}")
                print(f"   末端误差: {pos_error*1000:.2f} mm\n")
                
                self.trajectory_points.append(self.T_current[:3, 3].copy())
                if len(self.trajectory_points) > 100:
                    self.trajectory_points.pop(0)
                
                self.publish_joint_states()
                self.publish_visualization()
            else:
                self.get_logger().warn(f'❌ IK 求解失败: {sol.reason}')
                print(f"❌ IK 求解失败: {sol.reason}\n")
                
        except Exception as e:
            self.get_logger().error(f'❌ 求解出错: {str(e)}')
            print(f"❌ 求解出错: {e}\n")


def main(args=None):
    rclpy.init(args=args)
    node = SO101IKSolverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
