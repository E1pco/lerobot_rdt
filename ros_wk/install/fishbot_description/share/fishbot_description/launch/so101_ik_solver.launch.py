#!/usr/bin/env python3
"""
ROS2 launch 文件 - SO-101 机械臂 IK 求解可视化
启动: robot_state_publisher, RViz2, joint_state_publisher, IK 求解器
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    pkg_path = get_package_share_directory('fishbot_description')
    urdf_path = os.path.join(pkg_path, 'urdf', 'so101_new_calib.urdf')
    rviz_config_path = os.path.join(pkg_path, 'config', 'so101_ik.rviz')
    
    # 确保 URDF 文件存在
    if not os.path.exists(urdf_path):
        # 使用备用路径
        urdf_path = '/home/elpco/code/lerobot/lerobot_rdt/ros_wk/fishbot_description/urdf/fishbot_base.urdf'
    
    print(f"[Launch] URDF: {urdf_path}")
    print(f"[Launch] RViz: {rviz_config_path}")
    
    # 读取 URDF 文件
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()
    
    # 1. robot_state_publisher 节点
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': False,
        }],
        remappings=[
            ('joint_states', 'joint_states_ik'),  # 使用 IK 发布的关节状态
        ]
    )
    
    # 2. RViz2 节点
    rviz_config = os.path.join(
        get_package_share_directory('fishbot_description'),
        'config',
        'so101_ik.rviz'
    )
    
    if not os.path.exists(rviz_config):
        rviz_config = None
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config] if rviz_config else [],
    )
    
    # 3. Joint State Publisher GUI（可选，用于手动控制）
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
    )
    
    # 4. IK 求解器节点
    ik_solver_node = Node(
        package='fishbot_description',
        executable='ik_solver_node',
        name='so101_ik_solver',
        output='screen',
    )
    
    # 5. 目标位姿发布器（演示用）
    target_publisher_node = Node(
        package='fishbot_description',
        executable='target_pose_publisher',
        name='target_pose_publisher',
        output='screen',
    )
    
    return LaunchDescription([
        robot_state_publisher_node,
        rviz_node,
        joint_state_publisher_gui_node,
        ik_solver_node,
        target_publisher_node,
    ])
