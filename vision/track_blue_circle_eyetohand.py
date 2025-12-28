#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝色圆形抓取脚本 (眼在手外 Eye-to-Hand)
=====================================
功能:
  1. 检测蓝色圆形
  2. 使用PnP计算圆形在相机坐标系下的位置
  3. 结合眼在手外标定结果，计算圆形在基座坐标系下的位置
  4. 控制机械臂末端靠近蓝色圆形

使用方法:
  python vision/track_blue_circle_eyetohand.py
"""

import sys
import os
import cv2
import numpy as np
import time

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof_gripper


class BlueCircleTrackerEyeToHand:
    """蓝色圆形跟踪器（眼在手外版）"""
    
    def __init__(self, circle_diameter=0.05):
        """
        Parameters
        ----------
        circle_diameter : float
            圆形直径 (米)，默认 5cm
        """
        self.circle_diameter = circle_diameter
        self.circle_radius = circle_diameter / 2.0
        
        # 路径配置
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_data')
        self.intrinsic_file = os.path.join(self.config_dir, 'camera_intrinsics_environment.yaml')
        # 手眼标定结果优先使用 handeye_result_envir.npy（新流程产物），回退 camera_extrinsics.npy（旧命名，内容可能是 T_cam_base）
        self.handeye_candidates = [
            os.path.join(self.config_dir, 'handeye_result_envir.npy'),
            os.path.join(self.config_dir, 'camera_extrinsics.npy'),
        ]
        
        # 加载参数
        self.load_camera_intrinsics(self.intrinsic_file)
        self.load_handeye_calibration(self.handeye_candidates)
        
        # 初始化机器人
        self.robot = None
        self.controller = None
        
        # HSV 蓝色范围
        self.hsv_lower1 = np.array([100, 80, 80])
        self.hsv_upper1 = np.array([120, 255, 255])

        # 运行时状态
        self._k_scaled = False
        
        print("="*60)
        print("🔵 蓝色圆形抓取系统 (Eye-to-Hand)")
        print("="*60)
        print(f"圆形直径: {circle_diameter*1000:.0f} mm")
        print("="*60)
    
    def load_camera_intrinsics(self, yaml_path):
        """加载相机内参"""
        if not os.path.exists(yaml_path):
            print(f"❌ 未找到相机内参文件: {yaml_path}")
            # 尝试使用默认值或抛出异常
            raise FileNotFoundError(f"未找到相机内参文件: {yaml_path}")
        
        try:
            fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
            
            # 尝试不同的键名
            camera_matrix_node = fs.getNode('camera_matrix')
            if camera_matrix_node.empty():
                camera_matrix_node = fs.getNode('K')
            
            dist_coeffs_node = fs.getNode('distortion_coefficients')
            if dist_coeffs_node.empty():
                dist_coeffs_node = fs.getNode('distCoeffs')
            
            self.K = camera_matrix_node.mat()
            self.dist = dist_coeffs_node.mat().flatten()

            # 记录标定分辨率（若yaml里有）
            w_node = fs.getNode('image_width')
            h_node = fs.getNode('image_height')
            self.intrinsic_width = int(w_node.real()) if not w_node.empty() else None
            self.intrinsic_height = int(h_node.real()) if not h_node.empty() else None

            self.K0 = self.K.copy()
            fs.release()
            
            print(f"📷 已加载相机内参: {os.path.basename(yaml_path)}")
            print(f"   fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
            if self.intrinsic_width and self.intrinsic_height:
                print(f"   标定分辨率: {self.intrinsic_width}x{self.intrinsic_height}")
            
        except Exception as e:
            print(f"❌ 加载相机内参失败: {e}")
            raise
    
    def load_handeye_calibration(self, npy_path):
        """加载手眼标定结果 (T_cam_base)

        Parameters
        ----------
        npy_path : str | list[str]
            单个路径或候选路径列表（按顺序尝试）。
        """
        candidates = npy_path if isinstance(npy_path, (list, tuple)) else [npy_path]

        self.T_cam_base = None
        self.handeye_path = None

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    self.T_cam_base = np.load(path)
                    self.handeye_path = path
                    print(f"✅ 已加载手眼标定: {os.path.basename(path)}")
                    break
                except Exception as e:
                    print(f"❌ 加载手眼标定参数失败: {path} ({e})")

        if self.T_cam_base is None:
            print("⚠️ 未找到可用的手眼标定文件，将无法进行坐标转换")

    def _maybe_scale_K_to_frame(self, frame):
        """若相机实际分辨率与标定分辨率不同，按比例缩放 K。"""
        if self._k_scaled:
            return
        if frame is None:
            return
        h, w = frame.shape[:2]
        if not self.intrinsic_width or not self.intrinsic_height:
            print(f"ℹ️  当前相机分辨率: {w}x{h} (yaml未提供标定分辨率，跳过K缩放)")
            self._k_scaled = True
            return
        print(f"ℹ️  当前相机分辨率: {w}x{h}")
        if (w, h) == (self.intrinsic_width, self.intrinsic_height):
            self._k_scaled = True
            return
        sx = w / float(self.intrinsic_width)
        sy = h / float(self.intrinsic_height)
        K = self.K0.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        self.K = K
        self._k_scaled = True
        print(f"⚠️  分辨率不一致，已缩放K: sx={sx:.4f}, sy={sy:.4f}")
    
    def init_robot(self, port="/dev/left_arm", baudrate=1_000_000):
        """初始化机器人"""
        print("\n🤖 初始化机器人...")
        try:
            self.controller = ServoController(
                port=port, 
                baudrate=baudrate, 
                config_path=os.path.join(os.path.dirname(__file__), "../driver/servo_config.json")
            )
            self.robot = create_so101_5dof_gripper()
            self.robot.set_servo_controller(self.controller)
            print("✅ 机器人初始化完成")
            return True
        except Exception as e:
            print(f"❌ 机器人初始化失败: {e}")
            return False
    
    def detect_blue_circle(self, frame):
        """检测蓝色圆形"""
        # 转换到HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 蓝色掩码
        mask = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        
        # 形态学处理
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 霍夫圆检测
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=250
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            best_circle = max(circles, key=lambda c: c[2])
            center = (best_circle[0], best_circle[1])
            radius = best_circle[2]
            return True, center, radius, mask
        
        # 轮廓检测备选
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > 800:
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:
                        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                        return True, (int(cx), int(cy)), int(radius), mask
        
        return False, None, None, mask
    
    def estimate_pose_from_circle(self, center, radius_px):
        """从圆形的像素坐标和半径估算3D位置 (相机坐标系)"""
        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        # 估算深度 Z = f * R / r
        f = (fx + fy) / 2.0
        Z = f * self.circle_radius / float(radius_px)

        # 去畸变反投影：undistortPoints 输出归一化坐标 (x, y)，使 X=x*Z, Y=y*Z
        u, v = float(center[0]), float(center[1])
        pts = np.array([[[u, v]]], dtype=np.float32)
        und = cv2.undistortPoints(pts, self.K, self.dist)  # (1,1,2), normalized
        x_n, y_n = float(und[0, 0, 0]), float(und[0, 0, 1])
        X = x_n * Z
        Y = y_n * Z
        return np.array([X, Y, Z])
    
    def read_robot_pose(self, verbose=False):
        """读取机器人当前末端位姿"""
        q = self.robot.read_joint_angles(
            joint_names=self.robot.joint_names,
            verbose=verbose
        )
        T_gripper_base = self.robot.fkine(q)
        return T_gripper_base, q
    
    def run(self):
        """运行跟踪循环"""
        if not self.init_robot():
            return
        
        # 打开相机 (通常眼在手外使用不同的相机ID，这里假设为0，需根据实际情况调整)
        cap = cv2.VideoCapture(0) 
        if not cap.isOpened():
            print("❌ 无法打开相机 0，尝试相机 2...")
            cap = cv2.VideoCapture(2)
            if not cap.isOpened():
                print("❌ 无法打开相机")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 用首帧做一次K缩放/诊断
        ret0, frame0 = cap.read()
        if ret0:
            self._maybe_scale_K_to_frame(frame0)
        
        print("\n🎮 控制说明:")
        print("  'f' - 开启/关闭 跟随模式")
        print("  'h' - 机械臂回中")
        print("  'q' - 退出")
        print("="*60)
        
        following = False
        gain = 0.8
        step_limit = 0.080

        last_print_t = 0.0
        print_interval_s = 0.2
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                display = frame.copy()
                
                # 检测蓝色圆形
                success, center, radius_px, mask = self.detect_blue_circle(frame)
                
                if success:
                    # 绘制检测结果
                    cv2.circle(display, center, radius_px, (0, 255, 0), 2)
                    cv2.circle(display, center, 3, (0, 255, 255), -1)
                    
                    # 1. 计算相机坐标系下的位置
                    pos_cam = self.estimate_pose_from_circle(center, radius_px)
                    pos_cam_mm = pos_cam * 1000

                    now = time.time()
                    do_print = (now - last_print_t) >= print_interval_s
                    if do_print:
                        last_print_t = now
                        print(f"Cam:  [{pos_cam_mm[0]:7.1f}, {pos_cam_mm[1]:7.1f}, {pos_cam_mm[2]:7.1f}] mm")
                    
                    cv2.putText(display, f"PnP(Cam): [{pos_cam_mm[0]:.0f}, {pos_cam_mm[1]:.0f}, {pos_cam_mm[2]:.0f}]", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 2. 转换到基座坐标系 (Eye-to-Hand)
                    if self.T_cam_base is not None:
                        # P_base = T_handeye @ P_cam
                        pos_cam_homo = np.append(pos_cam, 1.0)
                        pos_target_base = (self.T_cam_base @ pos_cam_homo)[:3]
                        pos_target_mm = pos_target_base * 1000

                        if do_print:
                            print(f"Base: [{pos_target_mm[0]:7.1f}, {pos_target_mm[1]:7.1f}, {pos_target_mm[2]:7.1f}] mm")
                        
                        cv2.putText(display, f"Target(Base): [{pos_target_mm[0]:.0f}, {pos_target_mm[1]:.0f}, {pos_target_mm[2]:.0f}]", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        if following:
                            # 获取当前机械臂位置
                            T_gripper_base, q_curr = self.read_robot_pose()
                            pos_gripper_base = T_gripper_base[:3, 3]
                            
                            # 计算误差向量
                            error_base = pos_target_base - pos_gripper_base
                            
                            # 控制量
                            delta_base = error_base * gain
                            if np.linalg.norm(delta_base) > step_limit:
                                delta_base = delta_base / np.linalg.norm(delta_base) * step_limit
                            
                            pos_gripper_des = pos_gripper_base + delta_base
                            
                            # IK求解
                            R_gripper_des = T_gripper_base[:3, :3] # 保持当前姿态
                            T_gripper_base_des = np.eye(4)
                            T_gripper_base_des[:3, :3] = R_gripper_des
                            T_gripper_base_des[:3, 3] = pos_gripper_des
                            
                            ik_res = self.robot.ikine_LM(
                                T_gripper_base_des, 
                                q0=q_curr,
                                mask=np.array([1, 1, 1, 0.5, 0.5, 0])
                            )
                            
                            if ik_res.success:
                                q_new = ik_res.q
                                if np.linalg.norm(q_new - q_curr) < 1.5:
                                    targets = self.robot.q_to_servo_targets(q_new)
                                    self.controller.fast_move_to_pose(targets, speed=200)
                                    cv2.putText(display, "TRACKING", (10, 90), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display, "No Calibration", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 状态显示
                status = "FOLLOW ON" if following else "FOLLOW OFF (Press 'f')"
                cv2.putText(display, status, (10, display.shape[0]-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if following else (0, 255, 255), 2)
                
                cv2.imshow("Eye-to-Hand Tracking", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    following = not following
                    print(f"Follow mode: {following}")
                elif key == ord('h'):
                    self.controller.move_all_home()
                    following = False
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    tracker = BlueCircleTrackerEyeToHand(circle_diameter=0.05)
    tracker.run()


if __name__ == "__main__":
    main()
