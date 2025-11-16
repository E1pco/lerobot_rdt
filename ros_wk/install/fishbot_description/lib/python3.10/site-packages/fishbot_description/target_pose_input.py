#!/usr/bin/env python3
"""
SO-101 目标位姿交互输入程序
允许用户在终端中交互式输入目标位姿，然后发布到 ROS2 话题

使用方法:
  ros2 run fishbot_description target_pose_input
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading


class TargetPoseInputNode(Node):
    """目标位姿输入节点"""
    
    def __init__(self):
        super().__init__('target_pose_input')
        
        # 发布器
        self.pub_target_pose = self.create_publisher(
            PoseStamped, 'target_pose', 10
        )
        
        self.get_logger().info('🎯 目标位姿输入节点已启动')
        self.get_logger().info('   发布话题: /target_pose (geometry_msgs/PoseStamped)')
        self.get_logger().info('')
        
        # 启动交互线程
        self.input_thread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.input_thread.start()
    
    def user_input_loop(self):
        """用户交互循环"""
        print("\n" + "="*70)
        print("🎯 SO-101 目标位姿交互输入")
        print("="*70)
        print("输入命令:")
        print("  help        - 显示帮助信息")
        print("  input       - 手动输入目标位姿")
        print("  preset      - 选择预定义位姿")
        print("  quit/exit   - 退出程序")
        print("="*70 + "\n")
        
        while True:
            try:
                cmd = input("请输入命令: ").strip().lower()
                
                if cmd in ['help', 'h', '?']:
                    self.show_help()
                
                elif cmd in ['input', 'i']:
                    self.input_target_pose_interactive()
                
                elif cmd in ['preset', 'p']:
                    self.show_and_select_presets()
                
                elif cmd in ['quit', 'q', 'exit']:
                    print("\n👋 正在退出...")
                    break
                
                elif cmd == '':
                    continue
                
                else:
                    print(f"❌ 未知命令: '{cmd}'，输入 'help' 查看帮助\n")
                    
            except KeyboardInterrupt:
                print("\n\n👋 收到退出信号")
                break
            except Exception as e:
                print(f"❌ 错误: {e}\n")
    
    def show_help(self):
        """显示帮助信息"""
        print("\n" + "="*70)
        print("📖 帮助信息")
        print("="*70)
        print("\n命令说明:")
        print("  help/h/?    - 显示此帮助信息")
        print("  input/i     - 手动输入目标位姿 (x, y, z, roll, pitch, yaw)")
        print("  preset/p    - 从预定义位姿列表中选择")
        print("  quit/q/exit - 退出程序")
        print("\n位姿单位:")
        print("  位置 (x, y, z)    : 米 (m)")
        print("  姿态 (roll, pitch, yaw) : 度 (°)")
        print("\n示例:")
        print("  位置: x=0.0, y=-0.25, z=0.25 (米)")
        print("  姿态: roll=45, pitch=-30, yaw=0 (度)")
        print("\n提示:")
        print("  - 位置：机械臂末端在基座坐标系中的笛卡尔坐标")
        print("  - 姿态：欧拉角 (绕X, Y, Z轴的旋转)")
        print("  - 输入完成后会自动发布到 /target_pose 话题")
        print("  - IK 求解器会自动接收并求解逆运动学")
        print("="*70 + "\n")
    
    def input_target_pose_interactive(self):
        """交互式输入目标位姿"""
        print("\n" + "="*70)
        print("📍 手动输入目标位姿")
        print("="*70)
        
        try:
            # 输入位置
            print("\n📌 输入末端位置 (单位: 米)")
            print("   提示：典型范围 x=[-0.3,0.3], y=[-0.4,-0.1], z=[0.0,0.4]")
            x = float(input("  x (m) = "))
            y = float(input("  y (m) = "))
            z = float(input("  z (m) = "))
            
            # 输入姿态
            print("\n🔄 输入欧拉角 (单位: 度)")
            print("   提示：典型范围 ±180°")
            roll = float(input("  roll (°) = "))
            pitch = float(input("  pitch (°) = "))
            yaw = float(input("  yaw (°) = "))
            
            # 发布目标位姿
            self.publish_target_pose(x, y, z, roll, pitch, yaw)
            
        except ValueError as e:
            print(f"❌ 输入错误: 请输入有效的数字\n")
        except Exception as e:
            print(f"❌ 异常: {e}\n")
    
    def show_and_select_presets(self):
        """显示预定义位姿并让用户选择"""
        presets = {
            '1': {
                'name': '初始位置 (Home)',
                'desc': '机械臂在回中位置',
                'x': 0.0, 'y': -0.25, 'z': 0.25,
                'roll': 0, 'pitch': 0, 'yaw': 0
            },
            '2': {
                'name': '左前下方',
                'desc': '伸向左前下',
                'x': -0.15, 'y': -0.30, 'z': 0.10,
                'roll': 45, 'pitch': -30, 'yaw': 45
            },
            '3': {
                'name': '右前上方',
                'desc': '伸向右前上',
                'x': 0.15, 'y': -0.20, 'z': 0.30,
                'roll': -45, 'pitch': 30, 'yaw': -45
            },
            '4': {
                'name': '正上方',
                'desc': '机械臂向上伸',
                'x': 0.0, 'y': -0.25, 'z': 0.40,
                'roll': 0, 'pitch': -90, 'yaw': 0
            },
            '5': {
                'name': '正前方',
                'desc': '机械臂向前伸',
                'x': 0.0, 'y': -0.40, 'z': 0.20,
                'roll': 0, 'pitch': 0, 'yaw': 0
            },
            '6': {
                'name': '侧面水平',
                'desc': '机械臂向右侧伸',
                'x': 0.25, 'y': -0.15, 'z': 0.25,
                'roll': 0, 'pitch': 0, 'yaw': 90
            },
            '7': {
                'name': '抓取位置',
                'desc': '适合抓取的低位置',
                'x': 0.0, 'y': -0.25, 'z': 0.05,
                'roll': 0, 'pitch': -45, 'yaw': 0
            }
        }
        
        print("\n" + "="*70)
        print("🎯 预定义位姿库")
        print("="*70)
        for key, pose in presets.items():
            print(f"\n{key}. {pose['name']}")
            print(f"   描述: {pose['desc']}")
            print(f"   位置: ({pose['x']:6.2f}, {pose['y']:6.2f}, {pose['z']:6.2f}) m")
            print(f"   姿态: roll={pose['roll']:6.1f}°, pitch={pose['pitch']:6.1f}°, yaw={pose['yaw']:6.1f}°")
        
        print(f"\n0. 返回主菜单")
        choice = input("\n请选择 (0-{0}): ".format(len(presets))).strip()
        
        if choice == '0':
            return
        
        if choice in presets:
            pose = presets[choice]
            print(f"\n✅ 已选择: {pose['name']}")
            print(f"   位置: ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}) m")
            print(f"   姿态: R={pose['roll']:.0f}°, P={pose['pitch']:.0f}°, Y={pose['yaw']:.0f}°")
            
            # 发布目标位姿
            self.publish_target_pose(
                pose['x'], pose['y'], pose['z'],
                pose['roll'], pose['pitch'], pose['yaw']
            )
        else:
            print(f"❌ 无效选择: {choice}\n")
    
    def publish_target_pose(self, x, y, z, roll, pitch, yaw):
        """
        发布目标位姿到 ROS2 话题
        
        Parameters
        ----------
        x, y, z : float
            位置坐标 (米)
        roll, pitch, yaw : float
            欧拉角 (度)
        """
        try:
            # 转换欧拉角为弧度
            roll_rad = np.radians(roll)
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            
            # 欧拉角转四元数
            r = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad], degrees=False)
            quat = r.as_quat()  # [x, y, z, w]
            
            # 构建 PoseStamped 消息
            msg = PoseStamped()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            
            # 位置
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.position.z = float(z)
            
            # 姿态（四元数）
            msg.pose.orientation.x = float(quat[0])
            msg.pose.orientation.y = float(quat[1])
            msg.pose.orientation.z = float(quat[2])
            msg.pose.orientation.w = float(quat[3])
            
            # 发布
            self.pub_target_pose.publish(msg)
            
            print(f"\n✅ 已发布目标位姿到 /target_pose:")
            print(f"   位置: ({x:.4f}, {y:.4f}, {z:.4f}) m")
            print(f"   姿态: roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°")
            print(f"   四元数: ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
            print(f"   💡 IK 求解器正在处理...\n")
            
            self.get_logger().info(f"📢 已发布目标位姿: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            
        except Exception as e:
            print(f"❌ 发布失败: {e}\n")
            self.get_logger().error(f"发布失败: {e}")


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = TargetPoseInputNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
