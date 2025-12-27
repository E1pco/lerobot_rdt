#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼在手外 (Eye-to-Hand) 手眼标定脚本
=====================================
功能:
  1. 控制机械臂移动到不同位姿
  2. 在每个位姿采集图像并检测棋盘格
  3. 使用PnP计算棋盘格在相机坐标系下的位姿 (T_target_cam)
  4. 使用正运动学计算末端在基座标系下的位姿 (T_gripper_base)
  5. 使用手眼标定算法求解相机到基座的变换矩阵 (T_cam_base)

使用方法:
  python handeye_calibration_eyetohand.py --collect --camera 0 --port /dev/left_arm --output-dir ./calib_data
  python handeye_calibration_eyetohand.py --calibrate --output-dir ./calib_data
  python handeye_calibration_eyetohand.py --all --camera 0 --port /dev/left_arm --output-dir ./calib_data

坐标系定义:
  - base: 机械臂基座坐标系
  - gripper/end-effector: 机械臂末端坐标系
  - cam: 相机坐标系 (固定在环境中)
  - target: 标定板坐标系

眼在手外方程: AX = YB
  - A: 相邻两个末端位姿的相对变换
  - B: 相邻两个标定板位姿的相对变换
  - X: 相机相对于基座的变换 (T_cam_base) - 待求
  - Y: 末端相对于基座的变换 (T_gripper_base)
"""

import os
import sys
import cv2
import numpy as np
import time
import yaml
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof, create_so101_5dof_gripper


class EyeToHandCalibrator:
    """眼在手外手眼标定器"""
    
    def __init__(self, 
                 board_size=(4, 4),
                 square_size=0.00983,  # 25mm
                 camera_id=0,
                 port="/dev/left_arm",
                 camera_params_file="./config_data/camera_intrinsics_environment.yaml",
                 output_dir="./handeye_data_environment"):
        """
        Parameters
        ----------
        board_size : tuple
            棋盘格内角点数量 (cols-1, rows-1)
        square_size : float
            棋盘格方格边长 (米)
        camera_id : int
            相机设备ID
        port : str
            机械臂串口路径
        camera_params_file : str
            相机内外参文件路径 (OpenCV YAML格式)
        output_dir : str
            数据保存目录
        """
        self.board_size = board_size
        self.square_size = square_size
        self.camera_id = camera_id
        self.port = port
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化相机
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开相机 {camera_id}")
        
        # 设置相机分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 初始化机械臂
        print(f"连接机械臂: {port}")
        self.controller = ServoController(
            port=port,
            baudrate=1_000_000,
            config_path="../driver/servo_config.json"
        )
        
        # 创建机器人模型
        self.robot = create_so101_5dof_gripper()
        self.robot.set_servo_controller(self.controller)
        
        # 加载相机参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_extrinsics = None
        
        if camera_params_file:
            self.load_camera_params(camera_params_file)
        
        # 生成棋盘格3D点
        self.generate_board_corners()
        
        # 数据存储
        self.robot_poses = []       # T_gripper_base
        self.target_poses = []      # T_target_cam
        self.images = []
        
        print(f"✅ 眼在手外标定器初始化完成")
        print(f"   棋盘格尺寸: {board_size}")
        print(f"   方格大小: {square_size*1000:.1f}mm")
        print(f"   相机ID: {camera_id}")
        print(f"   机械臂: {port}")
        print(f"   输出目录: {output_dir}")
    
    def load_camera_params(self, camera_params_file):
        """加载相机内外参数 (OpenCV YAML格式)"""
        try:
            if not os.path.exists(camera_params_file):
                print(f"❌ 相机参数文件不存在: {camera_params_file}")
                return

            fs = cv2.FileStorage(camera_params_file, cv2.FILE_STORAGE_READ)
            
            if not fs.isOpened():
                print(f"❌ 无法打开相机参数文件: {camera_params_file}")
                return

            # 1. 加载内参矩阵 K
            camera_matrix_node = fs.getNode('K')
            if camera_matrix_node.empty():
                camera_matrix_node = fs.getNode('camera_matrix')
            
            if not camera_matrix_node.empty():
                self.camera_matrix = camera_matrix_node.mat()
            
            # 2. 加载畸变系数 distCoeffs
            dist_coeffs_node = fs.getNode('distCoeffs')
            if dist_coeffs_node.empty():
                dist_coeffs_node = fs.getNode('distortion_coefficients')
            
            if not dist_coeffs_node.empty():
                self.dist_coeffs = dist_coeffs_node.mat().flatten()

            # 3. 尝试加载棋盘格参数 (如果文件中有)
            square_size_node = fs.getNode('square_size')
            if not square_size_node.empty():
                file_square_size = square_size_node.real()
                if abs(file_square_size - self.square_size) > 1e-6:
                    print(f"ℹ️  使用文件中的方格大小: {file_square_size*1000:.2f}mm (原设置: {self.square_size*1000:.2f}mm)")
                    self.square_size = file_square_size
                    # 重新生成棋盘格3D点
                    self.generate_board_corners()

            cols_node = fs.getNode('board_size_cols')
            rows_node = fs.getNode('board_size_rows')
            if not cols_node.empty() and not rows_node.empty():
                cols = int(cols_node.real())
                rows = int(rows_node.real())
                if (cols, rows) != self.board_size and (rows, cols) != self.board_size:
                     print(f"ℹ️  使用文件中的棋盘格尺寸: {cols}x{rows} (原设置: {self.board_size})")
                     self.board_size = (cols, rows)
                     self.generate_board_corners()

            # 尝试加载外参 (可选)
            extrinsics_node = fs.getNode('camera_extrinsics')
            if not extrinsics_node.empty():
                self.camera_extrinsics = extrinsics_node.mat()
            
            fs.release()
            
            print(f"✅ 加载相机参数: {camera_params_file}")
            if self.camera_matrix is not None:
                print(f"   相机矩阵:\\n{self.camera_matrix}")
            if self.dist_coeffs is not None:
                print(f"   畸变系数: {self.dist_coeffs}")
            if self.camera_extrinsics is not None:
                print(f"   外参矩阵:\\n{self.camera_extrinsics}")
                
        except Exception as e:
            print(f"❌ 无法加载相机参数: {e}")
            print("   将使用自动标定或默认参数")
    
    def load_camera_intrinsics(self, intrinsic_file):
        """加载相机内参 (兼容性方法)"""
        try:
            if intrinsic_file.endswith('.yaml') or intrinsic_file.endswith('.yml'):
                # OpenCV YAML格式
                fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
                self.camera_matrix = fs.getNode('camera_matrix').mat()
                self.dist_coeffs = fs.getNode('distortion_coefficients').mat().flatten()
                fs.release()
            else:
                # NumPy格式
                data = np.load(intrinsic_file, allow_pickle=True).item()
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
            
            print(f"✅ 加载相机内参: {intrinsic_file}")
            print(f"   相机矩阵:\\n{self.camera_matrix}")
            print(f"   畸变系数: {self.dist_coeffs}")
            
        except Exception as e:
            print(f"❌ 无法加载相机内参: {e}")
            print("   将使用自动标定或默认参数")
    
    def load_camera_extrinsics(self, extrinsic_file):
        """加载相机外参 (兼容性方法)"""
        try:
            if extrinsic_file.endswith('.yaml') or extrinsic_file.endswith('.yml'):
                with open(extrinsic_file, 'r') as f:
                    data = yaml.safe_load(f)
                self.camera_extrinsics = np.array(data['camera_extrinsics']).reshape(4, 4)
            else:
                self.camera_extrinsics = np.load(extrinsic_file)
            
            print(f"✅ 加载相机外参: {extrinsic_file}")
            print(f"   外参矩阵:\\n{self.camera_extrinsics}")
            
        except Exception as e:
            print(f"❌ 无法加载相机外参: {e}")
    
    def generate_board_corners(self):
        """生成棋盘格3D角点"""
        self.board_corners = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.board_corners[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        self.board_corners *= self.square_size
    
    def detect_chessboard(self, image):
        """检测棋盘格"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测角点
        ret, corners = cv2.findChessboardCorners(
            gray, self.board_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            return True, corners.reshape(-1, 2)
        else:
            return False, None
    
    def calculate_target_pose(self, corners):
        """使用PnP计算标定板在相机坐标系下的位姿"""
        if self.camera_matrix is None:
            print("❌ 相机内参未加载，无法计算目标位姿")
            return None
        
        # 解PnP
        success, rvec, tvec = cv2.solvePnP(
            self.board_corners, corners, 
            self.camera_matrix, self.dist_coeffs
        )
        
        if success:
            # 转换为变换矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = tvec.flatten()
            
            return transform
        else:
            return None
    
    def get_robot_pose(self):
        """获取机器人末端位姿"""
        try:
            # 读取当前关节角度
            current_q = self.robot.read_joint_angles(verbose=False)
            
            # 计算正运动学
            T_gripper_base = self.robot.fkine(current_q)
            
            return T_gripper_base
            
        except Exception as e:
            print(f"❌ 获取机器人位姿失败: {e}")
            return None
    
    def capture_calibration_data(self):
        """采集标定数据"""
        print("🎯 开始采集手眼标定数据")
        print("操作说明:")
        print("  空格键 - 采集当前位姿的数据")
        print("  r键 - 删除最后一个数据点")
        print("  s键 - 保存数据")
        print("  q键 - 退出采集")
        print("  h键 - 机械臂回到初始位置")
        print()
        print("请移动机械臂到不同位置，确保相机能看到标定板...")
        
        # 创建本次采集的会话目录
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"session_{session_timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        print(f"📂 数据将实时保存到: {session_dir}")
        
        pose_count = 0
        
        try:
            while True:
                # 读取图像
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 相机读取失败")
                    break
                
                # 检测棋盘格
                found, corners = self.detect_chessboard(frame)
                
                # 可视化
                display_frame = frame.copy()
                
                if found:
                    # 绘制检测到的角点
                    cv2.drawChessboardCorners(display_frame, self.board_size, corners, found)
                    
                    # 如果有相机内参，计算并显示坐标轴
                    if self.camera_matrix is not None:
                        target_pose = self.calculate_target_pose(corners)
                        if target_pose is not None:
                            # 绘制坐标轴
                            axis_points = np.array([
                                [0, 0, 0],
                                [0.05, 0, 0],  # X轴 - 红色
                                [0, 0.05, 0],  # Y轴 - 绿色
                                [0, 0, -0.05]  # Z轴 - 蓝色
                            ], dtype=np.float32)
                            
                            axis_2d, _ = cv2.projectPoints(
                                axis_points, 
                                cv2.Rodrigues(target_pose[:3, :3])[0],
                                target_pose[:3, 3],
                                self.camera_matrix, 
                                self.dist_coeffs
                            )
                            
                            # 转换为整数坐标并绘制坐标轴
                            axis_2d = axis_2d.reshape(-1, 2)
                            pts = np.int32(axis_2d).reshape(-1, 2)
                            origin = tuple(pts[0].tolist())
                            pt_x = tuple(pts[1].tolist())
                            pt_y = tuple(pts[2].tolist())
                            pt_z = tuple(pts[3].tolist())
                            
                            # 使用line代替arrowedLine以避免类型问题
                            cv2.line(display_frame, origin, pt_x, (0, 0, 255), 3)  # X - 红
                            cv2.line(display_frame, origin, pt_y, (0, 255, 0), 3)  # Y - 绿
                            cv2.line(display_frame, origin, pt_z, (255, 0, 0), 3)  # Z - 蓝
                    
                    # 显示状态
                    cv2.putText(display_frame, f"Chessboard: FOUND", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Chessboard: NOT FOUND", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示已采集的数据点数量
                cv2.putText(display_frame, f"Poses: {pose_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Hand-Eye Calibration (Eye-to-Hand)', display_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # 空格 - 采集数据
                    if found and self.camera_matrix is not None:
                        # 计算标定板位姿
                        target_pose = self.calculate_target_pose(corners)
                        robot_pose = self.get_robot_pose()
                        
                        if target_pose is not None and robot_pose is not None:
                            # 保存数据到内存
                            self.target_poses.append(target_pose)
                            self.robot_poses.append(robot_pose)
                            self.images.append(frame.copy())
                            
                            # 立即保存到磁盘
                            img_filename = os.path.join(session_dir, f"image_{pose_count:03d}.jpg")
                            
                            # 保存带有坐标轴的图像 (可选，如果用户想要保存带轴的图)
                            # 但通常标定需要原始图。用户说"在采集的时候都在图像加上三维坐标轴"，可能是指显示，也可能是指保存。
                            # 如果是指保存，我们应该保存 display_frame。
                            # 但为了标定准确性，原始图像必须是干净的。
                            # 也许用户只是想在界面上看到。
                            # 既然界面上已经有了，那可能是用户觉得不够明显或者没看到？
                            # 或者用户希望保存下来的图片也有坐标轴用于检查？
                            # 让我们保存一份带坐标轴的副本用于调试。
                            
                            cv2.imwrite(img_filename, frame) # 保存原始图用于标定
                            cv2.imwrite(os.path.join(session_dir, f"vis_{pose_count:03d}.jpg"), display_frame) # 保存可视化图
                            
                            pose_filename = os.path.join(session_dir, f"pose_{pose_count:03d}.npz")
                            np.savez(pose_filename, 
                                     robot_pose=robot_pose, 
                                     target_pose=target_pose)
                            
                            pose_count += 1
                            
                            # 计算欧拉角以便显示
                            r_robot = R.from_matrix(robot_pose[:3, :3])
                            euler_robot = r_robot.as_euler('xyz', degrees=True)
                            
                            print(f"✅ 采集位姿 {pose_count}")
                            print(f"   机器人位置: {robot_pose[:3, 3]}")
                            print(f"   机器人姿态(Euler XYZ): {euler_robot}")
                            print(f"   标定板位姿: {target_pose[:3, 3]}")
                            print(f"   已保存: {img_filename}")
                        else:
                            print("❌ 位姿计算失败")
                    else:
                        if not found:
                            print("❌ 未检测到棋盘格")
                        if self.camera_matrix is None:
                            print("❌ 相机内参未加载")
                
                elif key == ord('r'):  # r - 删除最后一个数据点
                    if pose_count > 0:
                        self.target_poses.pop()
                        self.robot_poses.pop()
                        self.images.pop()
                        pose_count -= 1
                        print(f"🗑️  删除最后一个数据点，剩余: {pose_count}")
                    else:
                        print("❌ 没有数据点可删除")
                
                elif key == ord('s'):  # s - 保存数据
                    if pose_count > 0:
                        self.save_calibration_data()
                        print(f"💾 已保存 {pose_count} 个数据点")
                    else:
                        print("❌ 没有数据可保存")
                
                elif key == ord('h'):  # h - 回到初始位置
                    print("🏠 机械臂回到初始位置...")
                    self.controller.move_all_home()
                    time.sleep(2)
                
                elif key == ord('q'):  # q - 退出
                    print("🛑 退出采集")
                    break
        
        except KeyboardInterrupt:
            print("\\n🛑 用户中断")
        
        finally:
            cv2.destroyAllWindows()
        
        print(f"📊 采集完成，共获得 {pose_count} 个有效数据点")
        return pose_count > 0
    
    def save_calibration_data(self):
        """保存标定数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存位姿数据
        poses_file = os.path.join(self.output_dir, f"calibration_poses_{timestamp}.npz")
        np.savez(poses_file,
                robot_poses=np.array(self.robot_poses),
                target_poses=np.array(self.target_poses))
        
        # 保存图像
        images_dir = os.path.join(self.output_dir, f"images_{timestamp}")
        os.makedirs(images_dir, exist_ok=True)
        
        for i, img in enumerate(self.images):
            img_file = os.path.join(images_dir, f"image_{i:03d}.jpg")
            cv2.imwrite(img_file, img)
        
        print(f"💾 数据已保存到:")
        print(f"   位姿数据: {poses_file}")
        print(f"   图像数据: {images_dir}")
    
    def load_calibration_data(self, poses_file):
        """加载标定数据"""
        try:
            data = np.load(poses_file)
            self.robot_poses = data['robot_poses'].tolist()
            self.target_poses = data['target_poses'].tolist()
            
            print(f"✅ 加载标定数据: {poses_file}")
            print(f"   数据点数量: {len(self.robot_poses)}")
            return True
            
        except Exception as e:
            print(f"❌ 加载标定数据失败: {e}")
            return False
    
    def calibrate_eye_to_hand(self):
        """执行眼在手外标定"""
        if len(self.robot_poses) < 3:
            print("❌ 数据点不足，至少需要3个位姿")
            return None
        
        print(f"🔧 开始眼在手外标定，数据点数量: {len(self.robot_poses)}")
        
        try:
            # 准备数据
            R_gripper2base = []
            t_gripper2base = []
            R_target2cam = []
            t_target2cam = []
            
            R_base2gripper = []
            t_base2gripper = []
            
            R_cam2target = []
            t_cam2target = []
            
            for i in range(len(self.robot_poses)):
                # 1. 机器人末端到基座 (Standard FK)
                T_gripper_base = self.robot_poses[i]
                R_gripper2base.append(T_gripper_base[:3, :3])
                t_gripper2base.append(T_gripper_base[:3, 3])
                
                # 2. 基座到机器人末端 (Inverted FK)
                T_base_gripper = np.linalg.inv(T_gripper_base)
                R_base2gripper.append(T_base_gripper[:3, :3])
                t_base2gripper.append(T_base_gripper[:3, 3])
                
                # 3. 标定板到相机 (Standard PnP)
                T_target_cam = self.target_poses[i]
                R_target2cam.append(T_target_cam[:3, :3])
                t_target2cam.append(T_target_cam[:3, 3])
                
                # 4. 相机到标定板 (Inverted PnP)
                T_cam_target = np.linalg.inv(T_target_cam)
                R_cam2target.append(T_cam_target[:3, :3])
                t_cam2target.append(T_cam_target[:3, 3])
            
            # 数据质量检查
            print("\\n📊 数据质量检查:")
            self.analyze_data_quality(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
            
            # 定义不同的输入组合策略
            strategies = [
                {
                    "name": "Strategy 1: Base2Gripper + Target2Cam",
                    "R_gripper": R_base2gripper, "t_gripper": t_base2gripper,
                    "R_target": R_target2cam, "t_target": t_target2cam
                },
                {
                    "name": "Strategy 2: Gripper2Base + Cam2Target",
                    "R_gripper": R_gripper2base, "t_gripper": t_gripper2base,
                    "R_target": R_cam2target, "t_target": t_cam2target
                },
                {
                    "name": "Strategy 3: Target2Cam (as Robot) + Gripper2Base (as Target)",
                    "R_gripper": R_target2cam, "t_gripper": t_target2cam,
                    "R_target": R_gripper2base, "t_target": t_gripper2base
                }
            ]
            
            methods = [
                (cv2.CALIB_HAND_EYE_TSAI, "Tsai-Lenz"),
                (cv2.CALIB_HAND_EYE_PARK, "Park"),
                (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
                (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
                (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis")
            ]
            
            best_result = None
            best_score = float('inf')
            
            for strategy in strategies:
                print(f"\\n🔄 尝试策略: {strategy['name']}")
                
                for method, method_name in methods:
                    try:
                        # 执行标定
                        R_calib, t_calib = cv2.calibrateHandEye(
                            strategy["R_gripper"], strategy["t_gripper"],
                            strategy["R_target"], strategy["t_target"],
                            method=method
                        )
                        
                        # 验证结果
                        error = self.evaluate_calibration(R_calib, t_calib, 
                                                        R_gripper2base, t_gripper2base,
                                                        R_target2cam, t_target2cam)
                        
                        print(f"   {method_name}: {error:.6f} mm")
                        
                        if error < best_score and not (np.isnan(error) or np.isinf(error)):
                            best_score = error
                            best_result = (R_calib, t_calib, method_name, strategy['name'])
                        
                    except Exception as e:
                        print(f"   {method_name} 失败: {e}")
                        continue
            
            # 尝试非线性优化
            if best_result is not None:
                print(f"\\n🔄 尝试非线性优化 (基于 {best_result[2]})...")
                try:
                    R_opt, t_opt, error_opt = self.optimize_calibration(
                        best_result[0], best_result[1],
                        R_gripper2base, t_gripper2base,
                        R_target2cam, t_target2cam
                    )
                    print(f"   Optimization: {error_opt:.6f} mm")
                    
                    if error_opt < best_score:
                        best_score = error_opt
                        best_result = (R_opt, t_opt, "Optimization", "Non-linear Least Squares")
                except Exception as e:
                    print(f"   优化失败: {e}")

            if best_result is None:
                print("❌ 所有标定算法都失败了")
                return None
            
            R_cam2base, t_cam2base, best_method, best_strategy = best_result
            
            # 构建变换矩阵
            T_cam2base = np.eye(4)
            T_cam2base[:3, :3] = R_cam2base
            T_cam2base[:3, 3] = t_cam2base.flatten()

            # 有些输入组合/文献定义会返回“逆”的外参（例如得到 T_base_cam 而不是 T_cam_base）。
            # 这里用数据一致性自动判别：选择能让 T_target_gripper 更稳定的那个方向。
            try:
                score_direct = self.evaluate_calibration(
                    T_cam2base[:3, :3],
                    T_cam2base[:3, 3].reshape(3, 1),
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                )
                T_inv = np.linalg.inv(T_cam2base)
                score_inv = self.evaluate_calibration(
                    T_inv[:3, :3],
                    T_inv[:3, 3].reshape(3, 1),
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                )
                if np.isfinite(score_inv) and (score_inv + 1e-9) < score_direct:
                    print(f"\nℹ️  检测到结果可能为逆变换：一致性 {score_direct:.6f} -> {score_inv:.6f} mm，已自动取逆")
                    T_cam2base = T_inv
                    best_strategy = f"{best_strategy} (auto-inverted)"
                    best_score = score_inv
                else:
                    best_score = score_direct
            except Exception as _e:
                # 若评估失败，不阻断主流程
                pass
            
            print(f"\\n✅ 眼在手外标定完成")
            print(f"   最佳策略: {best_strategy}")
            print(f"   最佳算法: {best_method}")
            print(f"   一致性误差: {best_score:.6f} mm")
            print(f"\\n🎯 相机到基座变换矩阵 (T_cam_base):")
            print(T_cam2base)
            
            # 分析结果
            self.analyze_calibration_result(T_cam2base)
            
            # 一致性评估
            self.evaluate_calibration_consistency(T_cam2base)
            
            # 保存结果
            self.save_calibration_result(T_cam2base, best_method, best_score)
            
            return T_cam2base
            
        except Exception as e:
            print(f"❌ 标定失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def optimize_calibration(self, R_init, t_init, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
        """使用非线性最小二乘优化标定结果"""
        from scipy.optimize import least_squares
        
        # 1. 初始化 T_cam_base (X)
        T_cam_base = np.eye(4)
        T_cam_base[:3, :3] = R_init
        T_cam_base[:3, 3] = t_init.flatten()
        
        # 2. 初始化 T_gripper_target (Z)
        # Z = mean( inv(T_base_gripper) * inv(T_cam_base) * T_cam_target ) ?
        # No, T_cam_target = T_cam_base * T_base_gripper * T_gripper_target
        # So T_gripper_target = inv(T_base_gripper) * inv(T_cam_base) * T_cam_target
        # Wait, T_base_gripper is T_gripper_base (in my code variable name)
        # My code: T_gripper_base variable holds T_base_gripper (Pose of gripper in base)
        
        T_gripper_targets = []
        for i in range(len(R_gripper2base)):
            T_bg = np.eye(4)
            T_bg[:3, :3] = R_gripper2base[i]
            T_bg[:3, 3] = t_gripper2base[i]
            
            T_tc = np.eye(4)
            T_tc[:3, :3] = R_target2cam[i]
            T_tc[:3, 3] = t_target2cam[i]
            
            # T_gt = inv(T_bg) * inv(T_cb) * T_tc
            T_gt = np.linalg.inv(T_bg) @ np.linalg.inv(T_cam_base) @ T_tc
            T_gripper_targets.append(T_gt)
        
        # Average T_gripper_target
        # Simple averaging for translation, proper averaging for rotation
        t_gt_mean = np.mean([T[:3, 3] for T in T_gripper_targets], axis=0)
        R_gt_mean_obj = R.from_matrix([T[:3, :3] for T in T_gripper_targets]).mean()
        T_gripper_target = np.eye(4)
        T_gripper_target[:3, :3] = R_gt_mean_obj.as_matrix()
        T_gripper_target[:3, 3] = t_gt_mean
        
        # 3. Optimization parameters: [rx_X, ry_X, rz_X, tx_X, ty_X, tz_X, rx_Z, ry_Z, rz_Z, tx_Z, ty_Z, tz_Z]
        x0 = np.concatenate([
            R.from_matrix(T_cam_base[:3, :3]).as_rotvec(),
            T_cam_base[:3, 3],
            R.from_matrix(T_gripper_target[:3, :3]).as_rotvec(),
            T_gripper_target[:3, 3]
        ])
        
        def residuals(params):
            # Reconstruct X and Z
            r_X = params[0:3]
            t_X = params[3:6]
            r_Z = params[6:9]
            t_Z = params[9:12]
            
            T_X = np.eye(4)
            T_X[:3, :3] = R.from_rotvec(r_X).as_matrix()
            T_X[:3, 3] = t_X
            
            T_Z = np.eye(4)
            T_Z[:3, :3] = R.from_rotvec(r_Z).as_matrix()
            T_Z[:3, 3] = t_Z
            
            res = []
            for i in range(len(R_gripper2base)):
                T_bg = np.eye(4)
                T_bg[:3, :3] = R_gripper2base[i]
                T_bg[:3, 3] = t_gripper2base[i]
                
                T_tc_obs = np.eye(4)
                T_tc_obs[:3, :3] = R_target2cam[i]
                T_tc_obs[:3, 3] = t_target2cam[i]
                
                # Predicted T_tc = X * T_bg * Z
                T_tc_pred = T_X @ T_bg @ T_Z
                
                # Error in translation
                diff_t = T_tc_pred[:3, 3] - T_tc_obs[:3, 3]
                res.extend(diff_t)
                
                # Error in rotation (angle-axis)
                diff_R = T_tc_pred[:3, :3] @ T_tc_obs[:3, :3].T
                diff_r = R.from_matrix(diff_R).as_rotvec()
                res.extend(diff_r * 0.1) # Weight rotation less (approx 0.1m per radian)
            
            return np.array(res)
            
        res = least_squares(residuals, x0, verbose=0)
        
        # Extract optimized X
        params_opt = res.x
        r_X_opt = params_opt[0:3]
        t_X_opt = params_opt[3:6]
        
        R_opt = R.from_rotvec(r_X_opt).as_matrix()
        t_opt = t_X_opt.reshape(3, 1)
        
        # Calculate error
        error = self.evaluate_calibration(R_opt, t_opt, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
        
        return R_opt, t_opt, error

    
    def analyze_data_quality(self, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
        """分析数据质量"""
        print("   检查数据完整性...")
        
        # 检查平移量变化
        translations = np.array(t_gripper2base)
        translation_range = np.ptp(translations, axis=0)
        print(f"   机器人平移范围: X={translation_range[0]:.3f}m, Y={translation_range[1]:.3f}m, Z={translation_range[2]:.3f}m")
        
        # 检查旋转量变化
        rotations = []
        for rot_matrix in R_gripper2base:
            r = R.from_matrix(rot_matrix)
            rotations.append(r.as_euler('xyz', degrees=True))
        
        rotations = np.array(rotations)
        rotation_range = np.ptp(rotations, axis=0)
        print(f"   机器人旋转范围: Roll={rotation_range[0]:.1f}°, Pitch={rotation_range[1]:.1f}°, Yaw={rotation_range[2]:.1f}°")
        
        # 建议
        if np.any(translation_range < 0.05):
            print("   ⚠️  建议增加平移变化量 (>5cm)")
        if np.any(rotation_range < 10):
            print("   ⚠️  建议增加旋转变化量 (>10°)")
    
    def evaluate_calibration(self, R_cam2base, t_cam2base, 
                           R_gripper2base, t_gripper2base,
                           R_target2cam, t_target2cam):
        """评估标定结果 (使用一致性标准差作为指标)"""
        # 构建 T_cam_base
        T_cam_base = np.eye(4)
        T_cam_base[:3, :3] = R_cam2base
        T_cam_base[:3, 3] = t_cam2base.flatten()
        
        # 计算所有位姿下的 T_target_gripper
        # T_target_gripper = inv(T_gripper_base) * T_cam_base * T_target_cam
        
        target_gripper_translations = []
        
        for i in range(len(R_gripper2base)):
            T_gripper_base = np.eye(4)
            T_gripper_base[:3, :3] = R_gripper2base[i]
            T_gripper_base[:3, 3] = t_gripper2base[i]
            
            T_target_cam = np.eye(4)
            T_target_cam[:3, :3] = R_target2cam[i]
            T_target_cam[:3, 3] = t_target2cam[i]
            
            T_target_gripper = np.linalg.inv(T_gripper_base) @ T_cam_base @ T_target_cam
            target_gripper_translations.append(T_target_gripper[:3, 3])
            
        # 计算平移的标准差 (mm)
        translations = np.array(target_gripper_translations)
        std_dev = np.std(translations, axis=0)
        mean_std_dev = np.mean(std_dev) * 1000.0  # 转换为mm
        
        return mean_std_dev
    
    def analyze_calibration_result(self, T_cam2base):
        """分析标定结果"""
        print("\\n📋 标定结果分析:")
        
        # 相机位置
        cam_pos = T_cam2base[:3, 3]
        print(f"   相机位置: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}] m")
        
        # 相机姿态
        r = R.from_matrix(T_cam2base[:3, :3])
        cam_euler = r.as_euler('xyz', degrees=True)
        print(f"   相机姿态: Roll={cam_euler[0]:.1f}°, Pitch={cam_euler[1]:.1f}°, Yaw={cam_euler[2]:.1f}°")
        
        # 与预期的比较 (如果有外参参考)
        if self.camera_extrinsics is not None:
            pos_diff = np.linalg.norm(cam_pos - self.camera_extrinsics[:3, 3])
            print(f"   与参考外参位置差异: {pos_diff*1000:.1f}mm")
    
    def save_calibration_result(self, T_cam2base, method, error):
        """保存标定结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存numpy格式
        result_file = os.path.join(self.output_dir, f"camera_extrinsics_{timestamp}.npy")
        np.save(result_file, T_cam2base)
        
        # 保存YAML格式
        yaml_file = os.path.join(self.output_dir, f"camera_extrinsics_{timestamp}.yaml")
        result_data = {
            'calibration_info': {
                'method': method,
                'error': float(error),
                'timestamp': timestamp,
                'n_poses': len(self.robot_poses)
            },
            'camera_extrinsics': T_cam2base.tolist()
        }
        
        with open(yaml_file, 'w') as f:
            yaml.dump(result_data, f, default_flow_style=False)
        
        print(f"\\n💾 标定结果已保存:")
        print(f"   {result_file}")
        print(f"   {yaml_file}")
    
    def evaluate_calibration_consistency(self, T_cam_base):
        """评估标定结果的一致性 (仿照 Eye-in-Hand)"""
        if T_cam_base is None:
            return
        
        print("\n📊 标定结果一致性评估")
        print("="*70)
        
        errors = []
        
        # 对于眼在手外 (Eye-to-Hand)，标定板固定在机械臂末端
        # 因此 T_target_gripper 应该是恒定的
        # T_target_gripper = inv(T_gripper_base) * T_cam_base * T_target_cam
        
        T_target_grippers = []
        
        for i in range(len(self.robot_poses)):
            T_gb = self.robot_poses[i]
            T_tc = self.target_poses[i]
            
            # 计算 T_target_gripper
            T_tg = np.linalg.inv(T_gb) @ T_cam_base @ T_tc
            T_target_grippers.append(T_tg)
            
        # 计算两两之间的误差
        for i in range(len(T_target_grippers)):
            for j in range(i + 1, len(T_target_grippers)):
                T_tg1 = T_target_grippers[i]
                T_tg2 = T_target_grippers[j]
                
                # 相对误差 T_diff = T_tg1 * inv(T_tg2)
                T_diff = T_tg1 @ np.linalg.inv(T_tg2)
                
                error_trans = np.linalg.norm(T_diff[:3, 3]) * 1000  # mm
                error_rot = np.linalg.norm(R.from_matrix(T_diff[:3, :3]).as_rotvec()) * 180 / np.pi  # deg
                
                errors.append({
                    'pair': (i, j),
                    'trans_error': error_trans,
                    'rot_error': error_rot
                })
        
        if not errors:
            print("   没有足够的数据进行评估")
            return

        # 统计
        trans_errors = [e['trans_error'] for e in errors]
        rot_errors = [e['rot_error'] for e in errors]
        
        print(f"\n一致性误差 (标定板相对于末端的一致性):")
        print(f"   平移误差: 平均={np.mean(trans_errors):.2f}mm, 最大={np.max(trans_errors):.2f}mm")
        print(f"   旋转误差: 平均={np.mean(rot_errors):.2f}°, 最大={np.max(rot_errors):.2f}°")
        
        # 质量评估
        if np.mean(trans_errors) < 10 and np.mean(rot_errors) < 2:
             print("\n   ✅ 标定质量: 优秀")
        elif np.mean(trans_errors) < 20 and np.mean(rot_errors) < 5:
             print("\n   ⚠️  标定质量: 良好")
        else:
             print("\n   ❌ 标定质量: 一般/较差")
             
        print("="*70)
    
    def load_session_data(self, session_dir):
        """从会话目录加载数据"""
        try:
            import glob
            pose_files = sorted(glob.glob(os.path.join(session_dir, "pose_*.npz")))
            
            if not pose_files:
                print(f"❌ 会话目录中没有数据文件: {session_dir}")
                return False
            
            self.robot_poses = []
            self.target_poses = []
            
            for f in pose_files:
                data = np.load(f)
                self.robot_poses.append(data['robot_pose'])
                self.target_poses.append(data['target_pose'])
            
            print(f"✅ 从会话目录加载数据: {session_dir}")
            print(f"   数据点数量: {len(self.robot_poses)}")
            return True
            
        except Exception as e:
            print(f"❌ 加载会话数据失败: {e}")
            return False

    def run_calibration_workflow(self, mode="all"):
        """运行完整的标定流程"""
        if mode in ["collect", "all"]:
            print("🎬 步骤1: 采集标定数据")
            success = self.capture_calibration_data()
            if not success:
                print("❌ 数据采集失败")
                return False
        
        if mode in ["calibrate", "all"]:
            print("\\n🎬 步骤2: 执行标定计算")
            
            # 如果是仅标定模式，尝试加载最新的数据
            if mode == "calibrate" and not self.robot_poses:
                # 查找最新的数据文件
                import glob
                
                # 1. 尝试查找聚合文件 (旧格式)
                pose_files = glob.glob(os.path.join(self.output_dir, "calibration_poses_*.npz"))
                
                # 2. 尝试查找会话目录 (新格式)
                session_dirs = sorted(glob.glob(os.path.join(self.output_dir, "session_*")))
                
                if pose_files:
                    latest_file = max(pose_files, key=os.path.getctime)
                    print(f"ℹ️  发现聚合数据文件: {latest_file}")
                    if not self.load_calibration_data(latest_file):
                        print("❌ 无法加载标定数据")
                        return False
                elif session_dirs:
                    latest_session = session_dirs[-1]
                    print(f"ℹ️  发现最新会话目录: {latest_session}")
                    if not self.load_session_data(latest_session):
                        print("❌ 无法加载会话数据")
                        return False
                else:
                    print("❌ 未找到标定数据文件 (既无聚合文件也无会话目录)")
                    return False
            
            result = self.calibrate_eye_to_hand()
            if result is None:
                print("❌ 标定计算失败")
                return False
            
            print("\\n✅ 眼在手外标定完成!")
        
        return True
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'controller'):
            self.controller.close()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="眼在手外手眼标定")
    
    # 运行模式
    parser.add_argument("--collect", action="store_true", help="仅采集数据")
    parser.add_argument("--calibrate", action="store_true", help="仅执行标定")
    parser.add_argument("--all", action="store_true", help="采集数据+执行标定")
    
    # 硬件配置
    parser.add_argument("--camera", type=int, default=0, help="相机设备ID")
    parser.add_argument("--port", default="/dev/left_arm", help="机械臂串口")
    parser.add_argument("--output-dir", default="./handeye_data_environment", help="输出目录")
    # 文件配置
    parser.add_argument("--camera-params", default="./config_data/camera_intrinsics_environment.yaml", help="相机内外参文件 (OpenCV YAML格式)")

    args = parser.parse_args()
    
    # 确定运行模式
    if not any([args.collect, args.calibrate, args.all]):
        args.all = True  # 默认运行完整流程
    
    if args.collect:
        mode = "collect"
    elif args.calibrate:
        mode = "calibrate"
    else:
        mode = "all"
    
    try:
        # 创建标定器
        calibrator = EyeToHandCalibrator(
            camera_id=args.camera,
            port=args.port,
            camera_params_file=args.camera_params,
            output_dir=args.output_dir
        )
        
        # 运行标定流程
        success = calibrator.run_calibration_workflow(mode)
        
        if success:
            print("\\n🎉 眼在手外标定成功完成!")
        else:
            print("\\n❌ 眼在手外标定失败!")
            return 1
    
    except KeyboardInterrupt:
        print("\\n🛑 用户中断")
        return 1
    except Exception as e:
        print(f"\\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())