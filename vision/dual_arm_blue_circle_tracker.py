#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双臂蓝色圆形识别跟踪脚本
=====================================
功能:
  1. 同时使用左右两个相机检测蓝色圆形
  2. 使用手眼标定矩阵将相机坐标转换为机器人坐标
  3. 控制对应的机械臂进行跟踪
  4. 支持双臂协同操作

相机配置:
  - video0: 左手相机
  - video2: 右手相机
  
机械臂配置:
  - /dev/left_arm: 左臂串口
  - /dev/right_arm: 右臂串口

使用方法:
  python vision/dual_arm_blue_circle_tracker.py
"""

import sys
import os
import cv2
import numpy as np
import time
import yaml
import threading
from scipy.spatial.transform import Rotation as R

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof_gripper


class DualArmBlueCircleTracker:
    """双臂蓝色圆形跟踪器"""
    
    def __init__(self, circle_diameter=0.05):
        """
        Parameters
        ----------
        circle_diameter : float
            圆形直径 (米)，默认 5cm
        """
        self.circle_diameter = circle_diameter
        self.circle_radius = circle_diameter / 2.0
        
        # 相机和机械臂配置
        self.camera_configs = {
            'left': {
                'camera_id': 0,
                'device_path': '/dev/left_arm',
                'intrinsic_file': 'config_data/camera_intrinsics_left.yaml',
                'handeye_file': 'config_data/handeye_result_left.npy'
            },
            'right': {
                'camera_id': 2,
                'device_path': '/dev/right_arm', 
                'intrinsic_file': 'config_data/camera_intrinsics_right.yaml',
                'handeye_file': 'config_data/handeye_result_right.npy'
            }
        }
        
        # 初始化双臂系统
        self.arms = {}
        self.cameras = {}
        self.intrinsics = {}
        self.handeye_matrices = {}
        
        self.setup_dual_arm_system()
        
        # 控制参数
        self.tracking_enabled = True
        self.detection_results = {'left': None, 'right': None}
        
    def setup_dual_arm_system(self):
        """初始化双臂系统"""
        print("🔧 初始化双臂系统...")
        
        for arm_name, config in self.camera_configs.items():
            print(f"\n--- 初始化 {arm_name.upper()} 臂 ---")
            
            try:
                # 初始化机械臂控制器
                print(f"连接机械臂: {config['device_path']}")
                controller = ServoController(
                    port=config['device_path'],
                    baudrate=1_000_000,
                    config_path="../driver/servo_config.json"
                )
                
                # 创建机器人模型
                robot = create_so101_5dof_gripper()
                robot.set_servo_controller(controller)
                
                self.arms[arm_name] = {
                    'controller': controller,
                    'robot': robot
                }
                
                # 初始化相机
                print(f"连接相机: /dev/video{config['camera_id']}")
                cap = cv2.VideoCapture(config['camera_id'])
                if not cap.isOpened():
                    raise RuntimeError(f"无法打开相机 {config['camera_id']}")
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cameras[arm_name] = cap
                
                # 加载相机内参
                print(f"加载相机内参: {config['intrinsic_file']}")
                self.load_camera_intrinsics(arm_name, config['intrinsic_file'])
                
                # 加载手眼标定结果
                print(f"加载手眼矩阵: {config['handeye_file']}")
                self.load_handeye_calibration(arm_name, config['handeye_file'])
                
                print(f"✅ {arm_name.upper()} 臂初始化成功")
                
            except Exception as e:
                print(f"❌ {arm_name.upper()} 臂初始化失败: {e}")
                raise
    
    def load_camera_intrinsics(self, arm_name, intrinsic_file):
        """加载相机内参"""
        try:
            # 使用OpenCV读取YAML文件（支持OpenCV格式）
            fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
            
            # 读取相机矩阵和畸变系数
            camera_matrix = fs.getNode('K').mat()
            dist_coeffs = fs.getNode('distCoeffs').mat().flatten()
            
            fs.release()
            
            self.intrinsics[arm_name] = {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs
            }
            
            print(f"   相机内参矩阵:")
            print(f"{camera_matrix}")
            
        except Exception as e:
            print(f"❌ 无法加载 {arm_name} 相机内参: {e}")
            raise
    
    def load_handeye_calibration(self, arm_name, handeye_file):
        """加载手眼标定结果"""
        try:
            handeye_matrix = np.load(handeye_file)
            self.handeye_matrices[arm_name] = handeye_matrix
            
            print(f"   手眼标定矩阵:")
            print(f"{handeye_matrix}")
            
        except Exception as e:
            print(f"❌ 无法加载 {arm_name} 手眼标定结果: {e}")
            raise
    
    def detect_blue_circle(self, frame):
        """检测蓝色圆形"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 蓝色HSV范围 (调整为更宽松的范围)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # 创建蓝色掩码
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_circle = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 过滤小轮廓
                continue
            
            # 计算轮廓的圆度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.5:  # 圆度阈值
                # 计算最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # 评分: 圆度 + 大小合理性
                size_score = 1.0 - abs(radius - 30) / 30  # 期望半径约30像素
                total_score = circularity * 0.7 + max(0, size_score) * 0.3
                
                if total_score > best_score:
                    best_score = total_score
                    best_circle = {
                        'center': center,
                        'radius': radius,
                        'score': total_score,
                        'contour': contour
                    }
        
        return best_circle, mask
    
    def calculate_3d_position(self, arm_name, circle_center, circle_radius_pixels):
        """使用PnP算法计算3D位置"""
        if arm_name not in self.intrinsics:
            return None
        
        # 3D圆形关键点 (圆心在z=0平面)
        object_points = np.array([
            [0, 0, 0],                              # 圆心
            [self.circle_radius, 0, 0],             # 右
            [0, self.circle_radius, 0],             # 上
            [-self.circle_radius, 0, 0],            # 左
            [0, -self.circle_radius, 0]             # 下
        ], dtype=np.float32)
        
        # 2D图像点 (假设圆形垂直于相机)
        cx, cy = circle_center
        image_points = np.array([
            [cx, cy],                               # 圆心
            [cx + circle_radius_pixels, cy],        # 右
            [cx, cy - circle_radius_pixels],        # 上
            [cx - circle_radius_pixels, cy],        # 左
            [cx, cy + circle_radius_pixels]         # 下
        ], dtype=np.float32)
        
        camera_matrix = self.intrinsics[arm_name]['camera_matrix']
        dist_coeffs = self.intrinsics[arm_name]['dist_coeffs']
        
        # 解PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs
        )
        
        if success:
            return {
                'translation': tvec.flatten(),
                'rotation': rvec.flatten(),
                'success': True
            }
        else:
            return {'success': False}
    
    def camera_to_robot_coords(self, arm_name, camera_position):
        """将相机坐标转换为机器人坐标"""
        if arm_name not in self.handeye_matrices:
            return None
        
        # 相机坐标系下的位姿
        camera_point = np.array([camera_position[0], camera_position[1], camera_position[2], 1])
        
        # 使用手眼标定矩阵转换
        handeye_matrix = self.handeye_matrices[arm_name]
        robot_point = handeye_matrix @ camera_point
        
        return robot_point[:3]  # 返回xyz坐标
    
    def process_arm_detection(self, arm_name):
        """处理单个机械臂的检测"""
        if arm_name not in self.cameras:
            return
        
        cap = self.cameras[arm_name]
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ {arm_name} 相机读取失败")
            return
        
        # 检测蓝色圆形
        circle_result, mask = self.detect_blue_circle(frame)
        
        # 可视化
        display_frame = frame.copy()
        
        if circle_result:
            center = circle_result['center']
            radius = circle_result['radius']
            score = circle_result['score']
            
            # 绘制检测结果
            cv2.circle(display_frame, center, radius, (0, 255, 0), 2)
            cv2.circle(display_frame, center, 2, (0, 255, 0), -1)
            
            # 计算3D位置
            pos_3d = self.calculate_3d_position(arm_name, center, radius)
            
            if pos_3d and pos_3d['success']:
                camera_pos = pos_3d['translation']
                
                # 转换为机器人坐标
                robot_pos = self.camera_to_robot_coords(arm_name, camera_pos)
                
                if robot_pos is not None:
                    # 显示信息
                    info_text = f"{arm_name.upper()}: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})"
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 保存检测结果
                    self.detection_results[arm_name] = {
                        'camera_pos': camera_pos,
                        'robot_pos': robot_pos,
                        'pixel_center': center,
                        'pixel_radius': radius,
                        'score': score
                    }
                else:
                    cv2.putText(display_frame, f"{arm_name.upper()}: 坐标转换失败", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, f"{arm_name.upper()}: PnP失败", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, f"{arm_name.upper()}: 未检测到圆形", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.detection_results[arm_name] = None
        
        # 显示图像
        cv2.imshow(f'{arm_name.upper()} Camera', display_frame)
        cv2.imshow(f'{arm_name.upper()} Mask', mask)
    
    def control_arm_movement(self, arm_name, target_position, approach_height=0.1):
        """控制机械臂移动到目标位置"""
        if arm_name not in self.arms:
            return False
        
        try:
            robot = self.arms[arm_name]['robot']
            
            # 目标位置（稍微抬高一些接近）
            target_pos = target_position.copy()
            target_pos[2] += approach_height  # 在目标上方approach_height米
            
            # 构建目标变换矩阵（末端垂直向下）
            target_transform = np.eye(4)
            target_transform[:3, 3] = target_pos
            target_transform[:3, :3] = R.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
            
            # 求解逆运动学
            current_q = robot.read_joint_angles(verbose=False)
            result = robot.ikine_LM(target_transform, q0=current_q)
            
            if result.success:
                # 执行移动
                home_pose = {name: robot.servo.get_home_position(name) 
                           for name in robot.joint_names}
                
                servo_targets = robot.q_to_servo_targets(
                    q_rad=result.q,
                    home_pose=home_pose
                )
                
                robot.servo.soft_move_to_pose(servo_targets, step_count=8, interval=0.05)
                
                print(f"✅ {arm_name.upper()} 臂移动到目标位置: {target_pos}")
                return True
            else:
                print(f"❌ {arm_name.upper()} 臂逆运动学求解失败")
                return False
                
        except Exception as e:
            print(f"❌ {arm_name.upper()} 臂移动失败: {e}")
            return False
    
    def run_dual_arm_tracking(self):
        """运行双臂跟踪"""
        print("🚀 开始双臂蓝色圆形跟踪")
        print("按键操作:")
        print("  q - 退出程序")
        print("  l - 左臂移动到检测位置")
        print("  r - 右臂移动到检测位置")
        print("  b - 双臂同时移动")
        print("  h - 双臂回到初始位置")
        
        try:
            while self.tracking_enabled:
                # 处理左右两个相机
                self.process_arm_detection('left')
                self.process_arm_detection('right')
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 退出程序")
                    break
                elif key == ord('l'):
                    # 左臂移动
                    if self.detection_results['left'] is not None:
                        target = self.detection_results['left']['robot_pos']
                        self.control_arm_movement('left', target)
                    else:
                        print("⚠️  左臂未检测到目标")
                elif key == ord('r'):
                    # 右臂移动
                    if self.detection_results['right'] is not None:
                        target = self.detection_results['right']['robot_pos']
                        self.control_arm_movement('right', target)
                    else:
                        print("⚠️  右臂未检测到目标")
                elif key == ord('b'):
                    # 双臂同时移动
                    left_ok = right_ok = False
                    
                    if self.detection_results['left'] is not None:
                        target = self.detection_results['left']['robot_pos']
                        left_ok = self.control_arm_movement('left', target)
                    
                    if self.detection_results['right'] is not None:
                        target = self.detection_results['right']['robot_pos']
                        right_ok = self.control_arm_movement('right', target)
                    
                    if not (left_ok or right_ok):
                        print("⚠️  双臂都未检测到目标")
                elif key == ord('h'):
                    # 双臂回到初始位置
                    print("🏠 双臂回到初始位置...")
                    for arm_name in ['left', 'right']:
                        if arm_name in self.arms:
                            self.arms[arm_name]['controller'].move_all_home()
                
        except KeyboardInterrupt:
            print("\\n🛑 用户中断")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理资源...")
        
        # 关闭相机
        for arm_name, cap in self.cameras.items():
            if cap is not None:
                cap.release()
        
        # 关闭机械臂控制器
        for arm_name, arm in self.arms.items():
            if 'controller' in arm:
                arm['controller'].close()
        
        cv2.destroyAllWindows()
        print("✅ 资源清理完成")
    
    def __del__(self):
        self.cleanup()


def main():
    """主函数"""
    try:
        # 创建双臂跟踪器
        tracker = DualArmBlueCircleTracker(circle_diameter=0.05)
        
        # 运行跟踪
        tracker.run_dual_arm_tracking()
        
    except Exception as e:
        print(f"❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()