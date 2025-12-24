#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼在手上 (Eye-in-Hand) 手眼标定脚本
=====================================
功能:
  1. 控制机械臂移动到不同位姿
  2. 在每个位姿采集图像并检测棋盘格
  3. 使用PnP计算棋盘格在相机坐标系下的位姿 (T_target_cam)
  4. 使用正运动学计算末端在基座标系下的位姿 (T_gripper_base)
  5. 使用Tsai-Lenz方法求解手眼变换矩阵 (T_cam_gripper)

使用方法:
  python handeye_calibration_eyeinhand.py --collect   # 采集数据
  python handeye_calibration_eyeinhand.py --calibrate # 标定计算
  python handeye_calibration_eyeinhand.py --all       # 采集+标定

坐标系定义:
  - base: 机械臂基座坐标系
  - gripper/end-effector: 机械臂末端坐标系
  - cam: 相机坐标系
  - target: 标定板坐标系

眼在手上方程: AX = XB
  - A: 相邻两个末端位姿的相对变换
  - B: 相邻两个标定板位姿的相对变换  
  - X: 相机相对于末端的变换 (T_cam_gripper)
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof, create_so101_5dof_gripper


class HandEyeCalibrator:
    """眼在手上手眼标定器"""
    
    def __init__(self, 
                 board_size=(11, 8),
                 square_size=0.02073,  # 20.73mm
                 intrinsic_file=None,  # 默认使用脚本目录下的文件
                 output_dir='./handeye_data'):
        """
        Parameters
        ----------
        board_size : tuple
            棋盘格内角点数 (cols, rows)
        square_size : float
            棋盘格方格大小 (米)
        intrinsic_file : str
            相机内参文件路径，默认使用脚本目录下的 camera_intrinsics.yaml
        output_dir : str
            数据保存目录
        """
        self.board_size = board_size
        self.square_size = square_size
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果未指定内参文件，使用脚本所在目录的文件
        if intrinsic_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            intrinsic_file = os.path.join(script_dir, 'camera_intrinsics.yaml')
        
        # 加载相机内参
        self.load_camera_intrinsics(intrinsic_file)
        
        # 构造棋盘格3D点
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 初始化机器人和控制器
        self.robot = None
        self.controller = None
        
        # 存储标定数据
        self.T_target_cam_list = []  # 标定板在相机坐标系下的位姿
        self.T_gripper_base_list = []  # 末端在基座标系下的位姿
        self.images = []
        
        # PnP稳定性相关
        self.pose_buffer = []  # 用于多帧平均
        self.pose_buffer_size = 5  # 缓冲区大小
        
        print("="*70)
        print("🤖 眼在手上 (Eye-in-Hand) 手眼标定工具")
        print("="*70)
        print(f"\n棋盘格参数:")
        print(f"  内角点: {board_size[0]} × {board_size[1]}")
        print(f"  方格大小: {square_size*1000:.2f} mm")
        print(f"\n数据保存目录: {os.path.abspath(output_dir)}")
        print("="*70)
    
    def load_camera_intrinsics(self, yaml_path):
        """加载相机内参"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"未找到相机内参文件: {yaml_path}")
        
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        self.K = fs.getNode('K').mat()
        self.dist = fs.getNode('distCoeffs').mat().flatten()
        
        # 焦距修正 - 根据实际测量结果修正
        # 原始测量 647mm，实际 600mm，修正系数 = 600/647
        correction_factor = 67/70
        K_original_fx = self.K[0, 0]
        K_original_fy = self.K[1, 1]
        self.K[0, 0] *= correction_factor  # fx
        self.K[1, 1] *= correction_factor  # fy
        
        print(f"\n📷 焦距修正:")
        print(f"   原始: fx={K_original_fx:.1f}, fy={K_original_fy:.1f}")
        print(f"   修正系数: {correction_factor:.4f} (600/647)")
        print(f"   修正后: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
        
        # 尝试读取文件中保存的方格大小
        square_size_node = fs.getNode('square_size')
        if not square_size_node.empty():
            file_square_size = square_size_node.real()
            if abs(file_square_size - self.square_size) > 0.0001:
                print(f"\n⚠️  警告: 方格大小不一致!")
                print(f"   内参文件中: {file_square_size*1000:.2f} mm")
                print(f"   当前设置: {self.square_size*1000:.2f} mm")
                print(f"   使用内参文件中的值...")
                self.square_size = file_square_size
                # 重新构造objp
                self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
                self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
                self.objp *= self.square_size
        
        fs.release()
        
        print(f"\n📷 已加载相机内参: {os.path.abspath(yaml_path)}")
        print(f"   fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
        print(f"   cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}")
        print(f"   方格大小: {self.square_size*1000:.2f} mm")
    
    def init_robot(self, port="/dev/ttyACM0", baudrate=1_000_000):
        """初始化机器人和控制器"""
        print("\n🤖 初始化机器人...")
        
        self.controller = ServoController(
            port=port, 
            baudrate=baudrate, 
            config_path=os.path.join(os.path.dirname(__file__), "../driver/servo_config.json")
        )
        self.robot = create_so101_5dof_gripper()
        self.robot.set_servo_controller(self.controller)
        
        print("✅ 机器人初始化完成")
        return True
    
    def read_robot_pose(self, verbose=True):
        """
        读取机器人当前末端位姿
        
        Returns
        -------
        T_gripper_base : np.ndarray
            4x4 末端在基座标系下的位姿矩阵
        q : np.ndarray
            关节角度 (弧度)
        """
        # 读取关节角度
        q = self.robot.read_joint_angles(
            joint_names=self.robot.joint_names,
            verbose=verbose
        )
        
        # 正运动学计算末端位姿
        T_gripper_base = self.robot.fkine(q)
        
        if verbose:
            pos = T_gripper_base[:3, 3]
            euler = R.from_matrix(T_gripper_base[:3, :3]).as_euler('xyz', degrees=True)
            print(f"\n📍 末端位姿:")
            print(f"   位置: x={pos[0]*1000:.1f}mm, y={pos[1]*1000:.1f}mm, z={pos[2]*1000:.1f}mm")
            print(f"   姿态: roll={euler[0]:.1f}°, pitch={euler[1]:.1f}°, yaw={euler[2]:.1f}°")
        
        return T_gripper_base, q
    
    def detect_chessboard(self, frame, refine_pose=True):
        """
        检测棋盘格并计算其在相机坐标系下的位姿 (使用RANSAC PnP方法)
        
        经过测试验证，RANSAC PnP方法在稳定性和精度上表现最佳。
        
        Parameters
        ----------
        frame : np.ndarray
            输入图像
        refine_pose : bool
            是否使用LM优化精化位姿
        
        Returns
        -------
        success : bool
            是否成功检测到棋盘格
        T_target_cam : np.ndarray or None
            4x4 标定板在相机坐标系下的位姿矩阵
        corners : np.ndarray or None
            检测到的角点
        reproj_error : float
            重投影误差 (像素)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测棋盘格 - 增加FAST_CHECK加速
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE + 
                cv2.CALIB_CB_FAST_CHECK)
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        
        if not found:
            return False, None, None, float('inf')
        
        # 亚像素精化 - 使用更严格的终止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # PnP求解 - 使用多种方法，选择最佳结果
        imgp = corners.reshape(-1, 2).astype(np.float32)
        
        best_result = None
        best_reproj_error = float('inf')
        
        # 尝试多种PnP方法
        pnp_methods = [
            ('RANSAC', None),
            ('ITERATIVE', cv2.SOLVEPNP_ITERATIVE),
            ('EPNP', cv2.SOLVEPNP_EPNP),
            ('IPPE', cv2.SOLVEPNP_IPPE),
            ('SQPNP', cv2.SOLVEPNP_SQPNP),
        ]
        
        for method_name, method_flag in pnp_methods:
            try:
                if method_name == 'RANSAC':
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        self.objp, imgp, self.K, self.dist,
                        iterationsCount=1000,
                        reprojectionError=2.0,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        self.objp, imgp, self.K, self.dist, flags=method_flag
                    )
                
                if not success:
                    continue
                
                # LM优化精化位姿
                if refine_pose:
                    try:
                        rvec, tvec = cv2.solvePnPRefineLM(
                            self.objp, imgp, self.K, self.dist, rvec, tvec
                        )
                    except:
                        pass
                
                # 计算重投影误差
                reproj_pts, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.dist)
                reproj_error = np.sqrt(np.mean(np.sum((imgp - reproj_pts.reshape(-1, 2))**2, axis=1)))
                
                # 选择重投影误差最小的结果
                if reproj_error < best_reproj_error:
                    best_reproj_error = reproj_error
                    best_result = (rvec.copy(), tvec.copy())
                    
            except Exception as e:
                continue
        
        # 如果没有成功的结果
        if best_result is None:
            return False, None, corners, float('inf')
        
        rvec, tvec = best_result
        
        # 构造4x4变换矩阵
        R_mat, _ = cv2.Rodrigues(rvec)
        T_target_cam = np.eye(4)
        T_target_cam[:3, :3] = R_mat
        T_target_cam[:3, 3] = tvec.squeeze()
        
        return True, T_target_cam, corners, best_reproj_error
    
    def get_stable_pose(self, frame, num_samples=5, max_std_trans=5.0, max_std_rot=2.0):
        """
        获取稳定的PnP位姿 (多次采样取平均)
        
        Parameters
        ----------
        frame : np.ndarray
            输入图像 (实际会重新从相机采集多帧)
        num_samples : int
            采样次数
        max_std_trans : float
            允许的最大平移标准差 (mm)
        max_std_rot : float
            允许的最大旋转标准差 (度)
        
        Returns
        -------
        success : bool
            是否成功获取稳定位姿
        T_avg : np.ndarray
            平均位姿矩阵
        std_info : dict
            标准差信息
        """
        # 直接用传入的帧进行多次检测（模拟多帧，实际应该用cap多次读取）
        poses = []
        
        success, T, corners, err = self.detect_chessboard(frame)
        if not success:
            return False, None, {'error': 'detection_failed'}
        
        # 由于是静态场景，我们进行多次PnP求解来评估稳定性
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgp = corners.reshape(-1, 2).astype(np.float32)
        
        for _ in range(num_samples):
            # 使用不同的PnP方法求解
            methods = [
                cv2.SOLVEPNP_ITERATIVE,
                cv2.SOLVEPNP_EPNP,
                cv2.SOLVEPNP_IPPE,
                cv2.SOLVEPNP_SQPNP,
            ]
            
            for method in methods:
                try:
                    success, rvec, tvec = cv2.solvePnP(
                        self.objp, imgp, self.K, self.dist, flags=method
                    )
                    if success:
                        # LM优化
                        rvec, tvec = cv2.solvePnPRefineLM(
                            self.objp, imgp, self.K, self.dist, rvec, tvec
                        )
                        poses.append((rvec.copy(), tvec.copy()))
                except:
                    continue
        
        if len(poses) < 3:
            return False, None, {'error': 'insufficient_samples'}
        
        # 计算平均位姿
        tvecs = np.array([p[1].flatten() for p in poses])
        rvecs = np.array([p[0].flatten() for p in poses])
        
        # 平移的均值和标准差
        t_mean = np.mean(tvecs, axis=0)
        t_std = np.std(tvecs, axis=0) * 1000  # 转为mm
        
        # 旋转的标准差 (简化处理)
        r_std = np.std(rvecs, axis=0) * 180 / np.pi  # 转为度
        
        std_info = {
            't_std_mm': t_std,
            'r_std_deg': r_std,
            't_std_norm': np.linalg.norm(t_std),
            'r_std_norm': np.linalg.norm(r_std)
        }
        
        # 检查稳定性
        if np.linalg.norm(t_std) > max_std_trans or np.linalg.norm(r_std) > max_std_rot:
            return False, None, std_info
        
        # 构造平均位姿矩阵
        r_mean = np.mean(rvecs, axis=0)
        R_mat, _ = cv2.Rodrigues(r_mean)
        T_avg = np.eye(4)
        T_avg[:3, :3] = R_mat
        T_avg[:3, 3] = t_mean
        
        return True, T_avg, std_info
    
    def update_pose_buffer(self, T):
        """更新位姿缓冲区用于滑动平均"""
        self.pose_buffer.append(T.copy())
        if len(self.pose_buffer) > self.pose_buffer_size:
            self.pose_buffer.pop(0)
    
    def get_averaged_pose(self):
        """从缓冲区获取平均位姿"""
        if len(self.pose_buffer) < 3:
            return None
        
        # 平均平移
        translations = np.array([T[:3, 3] for T in self.pose_buffer])
        t_avg = np.mean(translations, axis=0)
        
        # 平均旋转 (使用四元数)
        quats = []
        for T in self.pose_buffer:
            q = R.from_matrix(T[:3, :3]).as_quat()
            quats.append(q)
        quats = np.array(quats)
        
        # 简单平均四元数 (对于小角度变化足够)
        q_avg = np.mean(quats, axis=0)
        q_avg /= np.linalg.norm(q_avg)  # 归一化
        
        R_avg = R.from_quat(q_avg).as_matrix()
        
        T_avg = np.eye(4)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = t_avg
        
        return T_avg
    
    def collect_data_interactive(self, cam_id=0):
        """
        交互式采集标定数据 (增强版 - 带PnP稳定性检测)
        
        按键:
          SPACE - 采集当前位姿
          'h'   - 机械臂回中
          's'   - 显示/隐藏稳定性信息
          'q'   - 退出采集
        """
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ 无法打开相机")
            return False
        
        print("\n📸 开始交互式数据采集 (增强版)")
        print("="*70)
        print("\n⌨️  快捷键:")
        print("   SPACE - 采集当前位姿数据")
        print("   'h'   - 机械臂回中位")
        print("   's'   - 显示/隐藏稳定性信息")
        print("   'q'   - 退出采集")
        print("\n📖 采集指南:")
        print("   1. 手动移动机械臂到不同位姿")
        print("   2. 确保棋盘格在相机视野内")
        print("   3. 等待位姿稳定(绿色)后按SPACE采集")
        print("   4. 建议采集 10-20 组数据")
        print("   5. 尽量让机械臂姿态多样化")
        print("="*70 + "\n")
        
        sample_count = 0
        show_stability = True
        
        # 清空位姿缓冲区
        self.pose_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            
            # 检测棋盘格并获取重投影误差
            success, T_target_cam, corners, reproj_error = self.detect_chessboard(frame)
            
            # 判断稳定性
            is_stable = False
            stability_info = ""
            
            if success and corners is not None:
                cv2.drawChessboardCorners(display, self.board_size, corners, True)
                
                # 更新位姿缓冲区
                self.update_pose_buffer(T_target_cam)
                
                # 计算位姿稳定性
                if len(self.pose_buffer) >= 3:
                    translations = np.array([T[:3, 3] for T in self.pose_buffer])
                    t_std = np.std(translations, axis=0) * 1000  # mm
                    t_std_norm = np.linalg.norm(t_std)
                    
                    # 判断是否稳定
                    is_stable = t_std_norm < 3.0 and reproj_error < 1.0
                    
                    if show_stability:
                        stability_info = f"Std: {t_std_norm:.1f}mm, ReprojErr: {reproj_error:.2f}px"
                
                # 显示标定板距离
                distance = np.linalg.norm(T_target_cam[:3, 3]) * 1000
                
                # 根据稳定性选择颜色
                color = (0, 255, 0) if is_stable else (0, 255, 255)
                status_text = "STABLE - Press SPACE" if is_stable else "Detecting..."
                
                cv2.putText(display, f"Distance: {distance:.0f}mm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, status_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if show_stability and stability_info:
                    cv2.putText(display, stability_info, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(display, "Chessboard NOT FOUND", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # 清空缓冲区
                self.pose_buffer = []
            
            # 显示采集数量
            cv2.putText(display, f"Samples: {sample_count}", (display.shape[1]-150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Hand-Eye Calibration', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 退出采集")
                break
            
            elif key == ord('h'):
                print("\n🏠 机械臂回中...")
                self.controller.move_all_home()
                time.sleep(1)
            
            elif key == ord('s'):
                show_stability = not show_stability
                print(f"{'显示' if show_stability else '隐藏'}稳定性信息")
            
            elif key == ord(' '):
                if not success:
                    print("⚠️  未检测到棋盘格，无法采集")
                    continue
                
                if not is_stable:
                    print("⚠️  位姿不稳定，建议等待稳定后再采集")
                    # 仍然允许采集，但给出警告
                
                # 使用平均位姿 (更稳定)
                T_avg = self.get_averaged_pose()
                if T_avg is not None:
                    T_to_save = T_avg
                    print("   使用平均位姿")
                else:
                    T_to_save = T_target_cam
                    print("   使用单帧位姿")
                
                sample_count += 1
                print(f"\n📸 采集数据 #{sample_count}")
                
                # 读取机器人位姿
                T_gripper_base, q = self.read_robot_pose(verbose=True)
                
                # 保存数据
                self.T_target_cam_list.append(T_to_save.copy())
                self.T_gripper_base_list.append(T_gripper_base.copy())
                self.images.append(frame.copy())
                
                # 保存到文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                np.savez(
                    os.path.join(self.output_dir, f"pose_{sample_count:02d}_{timestamp}.npz"),
                    T_target_cam=T_to_save,
                    T_gripper_base=T_gripper_base,
                    q=q,
                    reproj_error=reproj_error
                )
                cv2.imwrite(
                    os.path.join(self.output_dir, f"image_{sample_count:02d}_{timestamp}.jpg"),
                    frame
                )
                
                print(f"✅ 已保存数据 #{sample_count}")
                print(f"   标定板距离: {np.linalg.norm(T_to_save[:3,3])*1000:.1f} mm")
                print(f"   重投影误差: {reproj_error:.2f} px")
                
                # 清空缓冲区，避免下一帧使用旧数据
                self.pose_buffer = []
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 共采集 {sample_count} 组数据")
        return sample_count >= 3
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 共采集 {sample_count} 组数据")
        return sample_count >= 3
    
    def load_collected_data(self):
        """从文件加载已采集的数据"""
        import glob
        
        pose_files = sorted(glob.glob(os.path.join(self.output_dir, "pose_*.npz")))
        
        if not pose_files:
            print(f"❌ 未找到标定数据: {self.output_dir}")
            return False
        
        self.T_target_cam_list = []
        self.T_gripper_base_list = []
        
        # 临时创建机器人模型用于重算FK
        temp_robot = create_so101_5dof_gripper()
        
        print(f"\n📂 加载标定数据...")
        for f in pose_files:
            data = np.load(f)
            self.T_target_cam_list.append(data['T_target_cam'])
            
            # 如果有保存关节角度，重新计算FK (以防运动学参数有更新)
            if 'q' in data:
                q = data['q']
                T_gb = temp_robot.fkine(q)
                self.T_gripper_base_list.append(T_gb)
                # print(f"   ✅ {os.path.basename(f)} (Re-computed FK)")
            else:
                self.T_gripper_base_list.append(data['T_gripper_base'])
                # print(f"   ✅ {os.path.basename(f)}")
            print(f"   ✅ {os.path.basename(f)}")
        
        print(f"\n共加载 {len(self.T_target_cam_list)} 组数据")
        return True
    
    def calibrate(self, method=cv2.CALIB_HAND_EYE_PARK):
        """
        执行手眼标定
        
        Parameters
        ----------
        method : int
            手眼标定方法，可选:
            - cv2.CALIB_HAND_EYE_TSAI
            - cv2.CALIB_HAND_EYE_PARK (默认)
            - cv2.CALIB_HAND_EYE_HORAUD
            - cv2.CALIB_HAND_EYE_ANDREFF
            - cv2.CALIB_HAND_EYE_DANIILIDIS
        
        Returns
        -------
        T_cam_gripper : np.ndarray
            4x4 相机在末端坐标系下的位姿矩阵
        """
        if len(self.T_target_cam_list) < 3:
            print("❌ 数据不足，至少需要 3 组数据")
            return None
        
        print("\n🔄 开始手眼标定...")
        print(f"   数据组数: {len(self.T_target_cam_list)}")
        
        # 准备数据
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        for T_gb, T_tc in zip(self.T_gripper_base_list, self.T_target_cam_list):
            R_gripper2base.append(T_gb[:3, :3])
            t_gripper2base.append(T_gb[:3, 3].reshape(3, 1))
            R_target2cam.append(T_tc[:3, :3])
            t_target2cam.append(T_tc[:3, 3].reshape(3, 1))
        
        # 执行手眼标定
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=method
        )
        
        # 构造4x4变换矩阵
        T_cam_gripper = np.eye(4)
        T_cam_gripper[:3, :3] = R_cam2gripper
        T_cam_gripper[:3, 3] = t_cam2gripper.squeeze()
        
        print("\n✅ 手眼标定完成!")
        print("\n📊 结果 (T_cam_gripper - 相机相对于末端的变换):")
        print("-"*70)
        
        # 平移
        t = t_cam2gripper.squeeze() * 1000  # 转换为mm
        print(f"平移向量 (mm):")
        print(f"   tx = {t[0]:8.2f}")
        print(f"   ty = {t[1]:8.2f}")
        print(f"   tz = {t[2]:8.2f}")
        
        # 旋转
        euler = R.from_matrix(R_cam2gripper).as_euler('xyz', degrees=True)
        quat = R.from_matrix(R_cam2gripper).as_quat()
        print(f"\n旋转 (欧拉角, 度):")
        print(f"   roll  = {euler[0]:8.2f}")
        print(f"   pitch = {euler[1]:8.2f}")
        print(f"   yaw   = {euler[2]:8.2f}")
        print(f"\n四元数 (x, y, z, w):")
        print(f"   {quat}")
        
        print("-"*70)
        
        return T_cam_gripper
    
    def evaluate_calibration(self, T_cam_gripper):
        """评估标定结果的一致性"""
        if T_cam_gripper is None:
            return
        
        print("\n📊 标定结果评估")
        print("="*70)
        
        errors = []
        
        # 计算AX=XB的一致性误差
        for i in range(len(self.T_gripper_base_list)):
            for j in range(i + 1, len(self.T_gripper_base_list)):
                # 相邻两帧的相对运动
                T_gb1 = self.T_gripper_base_list[i]
                T_gb2 = self.T_gripper_base_list[j]
                T_tc1 = self.T_target_cam_list[i]
                T_tc2 = self.T_target_cam_list[j]
                
                # A = T_g2_g1 = inv(T_b_g2) * T_b_g1 (在Gripper坐标系下的相对运动)
                A = np.linalg.inv(T_gb2) @ T_gb1
                
                # B = T_c2_c1 = T_c2_t * T_t_c1 = T_c_t2 * inv(T_c_t1) (在Camera坐标系下的相对运动)
                B = T_tc2 @ np.linalg.inv(T_tc1)
                
                # AX 和 XB 应该相等
                AX = A @ T_cam_gripper
                XB = T_cam_gripper @ B
                
                # 计算误差
                error_T = AX @ np.linalg.inv(XB)
                error_trans = np.linalg.norm(error_T[:3, 3]) * 1000  # mm
                error_rot = np.linalg.norm(R.from_matrix(error_T[:3, :3]).as_rotvec()) * 180 / np.pi  # deg
                
                errors.append({
                    'pair': (i, j),
                    'trans_error': error_trans,
                    'rot_error': error_rot
                })
        
        # 统计
        trans_errors = [e['trans_error'] for e in errors]
        rot_errors = [e['rot_error'] for e in errors]
        
        print(f"\n一致性误差 (AX=XB):")
        print(f"   平移误差: 平均={np.mean(trans_errors):.2f}mm, 最大={np.max(trans_errors):.2f}mm")
        print(f"   旋转误差: 平均={np.mean(rot_errors):.2f}°, 最大={np.max(rot_errors):.2f}°")
        
        # 质量评估
        if np.mean(trans_errors) < 30 and np.mean(rot_errors) < 5:
            print("\n   ✅ 标定质量: 优秀")
        elif np.mean(trans_errors) < 50 and np.mean(rot_errors) < 10:
            print("\n   ⚠️  标定质量: 一般")
        else:
            print("\n   ❌ 标定质量: 较差，建议重新采集数据")
        
        print("="*70)
    
    def save_result(self, T_cam_gripper, filename='handeye_result.yaml'):
        """保存标定结果"""
        if T_cam_gripper is None:
            return
        
        # 保存为YAML
        filepath = os.path.join(self.output_dir, filename)
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        fs.write('T_cam_gripper', T_cam_gripper)
        
        # 分解保存
        R_mat = T_cam_gripper[:3, :3]
        t_vec = T_cam_gripper[:3, 3]
        euler = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
        quat = R.from_matrix(R_mat).as_quat()
        
        fs.write('rotation_matrix', R_mat)
        fs.write('translation_vector', t_vec.reshape(3, 1))
        fs.write('euler_angles_deg', np.array(euler).reshape(3, 1))
        fs.write('quaternion_xyzw', np.array(quat).reshape(4, 1))
        fs.write('calibration_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        fs.write('num_samples', len(self.T_target_cam_list))
        fs.release()
        
        print(f"\n💾 标定结果已保存: {filepath}")
        
        # 同时保存为npy (兼容)
        npy_path = os.path.join(self.output_dir, 'handeye_result.npy')
        np.save(npy_path, T_cam_gripper)
        print(f"💾 标定结果已保存: {npy_path}")
        
        # 复制到vision目录根目录
        root_yaml = os.path.join(os.path.dirname(__file__), 'handeye_result.yaml')
        root_npy = os.path.join(os.path.dirname(__file__), 'handeye_result.npy')
        
        import shutil
        shutil.copy(filepath, root_yaml)
        shutil.copy(npy_path, root_npy)
        print(f"💾 已复制到: {root_yaml}")
    
    def close(self):
        """关闭控制器"""
        if self.controller:
            self.controller.close()
            print("🔌 控制器已关闭")


def main():
    parser = argparse.ArgumentParser(description='眼在手上手眼标定工具')
    parser.add_argument('--collect', action='store_true', help='采集标定数据')
    parser.add_argument('--calibrate', action='store_true', help='执行标定计算')
    parser.add_argument('--all', action='store_true', help='采集+标定')
    parser.add_argument('--output-dir', default='./handeye_data', help='数据保存目录')
    parser.add_argument('--intrinsic', default='camera_intrinsics.yaml', help='相机内参文件')
    parser.add_argument('--square-size', type=float, default=20.73, help='棋盘格方格大小(mm)')
    parser.add_argument('--port', default='/dev/ttyACM0', help='串口')
    
    args = parser.parse_args()
    
    calibrator = HandEyeCalibrator(
        board_size=(11, 8),
        square_size=args.square_size / 1000.0,
        intrinsic_file=args.intrinsic,
        output_dir=args.output_dir
    )
    
    try:
        if args.collect or args.all:
            # 初始化机器人
            calibrator.init_robot(port=args.port)
            
            # 回中
            print("\n🏠 机械臂回中...")
            # 采集数据
            calibrator.collect_data_interactive()
        
        if args.calibrate or args.all or (not args.collect and not args.all):
            # 加载数据
            if not calibrator.T_target_cam_list:
                calibrator.load_collected_data()
            
            if calibrator.T_target_cam_list:
                # 执行标定
                T_cam_gripper = calibrator.calibrate()
                
                if T_cam_gripper is not None:
                    # 评估
                    calibrator.evaluate_calibration(T_cam_gripper)
                    
                    # 保存
                    calibrator.save_result(T_cam_gripper)
    
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        calibrator.close()


if __name__ == '__main__':
    main()
