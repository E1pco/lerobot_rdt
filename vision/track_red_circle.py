#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蓝色圆形抓取脚本
=====================================
功能:
  1. 检测直径5cm的蓝色圆形（运动形变鲁棒）
  2. 使用PnP计算圆形在相机坐标系下的位置
  3. 控制机械臂末端靠近蓝色圆形进行抓取

使用方法:
  python vision/track_blue_circle.py
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


class BlueCircleTracker:
    """蓝色圆形跟踪器（运动形变鲁棒）"""
    
    def __init__(self, circle_diameter=0.05, intrinsic_file=None):
        """
        Parameters
        ----------
        circle_diameter : float
            圆形直径 (米)，默认 5cm
        intrinsic_file : str
            相机内参文件路径
        """
        self.circle_diameter = circle_diameter
        self.circle_radius = circle_diameter / 2.0
        
        # 加载相机内参
        if intrinsic_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            intrinsic_file = os.path.join(script_dir, 'camera_intrinsics.yaml')
        self.load_camera_intrinsics(intrinsic_file)
        
        # 加载手眼标定结果
        self.load_handeye_calibration()
        
        # 初始化机器人
        self.robot = None
        self.controller = None
        
        # HSV 蓝色范围
        self.hsv_lower1 = np.array([100, 80, 80])
        self.hsv_upper1 = np.array([120, 255, 255])
        self.hsv_lower2 = None
        self.hsv_upper2 = None
        
        print("="*60)
        print("🔵 蓝色圆形抓取系统（运动形变鲁棒）")
        print("="*60)
        print(f"圆形直径: {circle_diameter*1000:.0f} mm")
        print("="*60)
    
    def load_camera_intrinsics(self, yaml_path):
        """加载相机内参"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"未找到相机内参文件: {yaml_path}")
        
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        self.K = fs.getNode('K').mat()
        self.dist = fs.getNode('distCoeffs').mat().flatten()
        fs.release()
        
        # 焦距修正
        correction_factor = 67/70
        self.K[0, 0] *= correction_factor
        self.K[1, 1] *= correction_factor
        
        print(f"📷 已加载相机内参")
        print(f"   fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
    
    def load_handeye_calibration(self):
        """加载手眼标定结果"""
        calib_file = os.path.join(os.path.dirname(__file__), 'handeye_result.npy')
        if not os.path.exists(calib_file):
            print("⚠️ 未找到手眼标定文件，将只显示检测结果")
            self.T_cam_gripper = None
        else:
            self.T_cam_gripper = np.load(calib_file)
            print("✅ 已加载手眼标定参数")
    
    def init_robot(self, port="/dev/right_arm", baudrate=1_000_000):
        """初始化机器人"""
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
    
    def detect_blue_circle(self, frame):
        """
        检测蓝色圆形（增强鲁棒性，支持形变）
        
        使用两种方法：
        1. 霍夫圆检测（优先）
        2. 轮廓检测+椭圆拟合（备选，处理形变）
        
        Returns
        -------
        success : bool
            是否检测到圆形
        center : tuple
            圆心像素坐标 (u, v)
        radius : float
            圆形像素半径（椭圆时取平均）
        mask : np.ndarray
            蓝色掩码图像
        """
        # 转换到HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 绿色掩码
        mask = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        
        # 形态学处理 - 更强的降噪
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 方法1: 霍夫圆检测（使用更严格参数）
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,       # 提高阈值，只检测明显的圆
            minRadius=15,
            maxRadius=250
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            best_circle = max(circles, key=lambda c: c[2])
            center = (best_circle[0], best_circle[1])
            radius = best_circle[2]
            return True, center, radius, mask
        
        # 方法2: 严格的轮廓拟合圆形
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None, None, mask
        
        # 找最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # 面积过滤（更严格）
        if area < 800:  # 增加最小面积要求
            return False, None, None, mask
        
        # 计算圆度（判断是否接近圆形）
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # 更严格的圆度要求（0.7 表示接近圆形）
        if circularity > 0.7:
            # 尝试圆拟合（而不是椭圆）
            if len(largest_contour) >= 5:
                # 使用最小二乘法拟合圆
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(cx), int(cy))
                radius = int(radius)
                
                # 验证拟合质量：计算轮廓点到圆的距离偏差
                points = largest_contour.reshape(-1, 2).astype(np.float32)
                distances = np.abs(np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius)
                mean_error = np.mean(distances)
                std_error = np.std(distances)
                
                # 只有当拟合误差较小时才接受
                if mean_error < radius * 0.15 and std_error < radius * 0.2:  # 误差在15%以内
                    return True, center, radius, mask
        
        # 不符合严格条件，拒绝
        return False, None, None, mask
    
    def estimate_pose_from_circle(self, center, radius_px):
        """
        从圆形的像素坐标和半径估算3D位置
        
        使用针孔相机模型：
        - 已知圆形实际半径 R (米)
        - 已知像素半径 r (像素)
        - 焦距 f (像素)
        - 深度 Z = f * R / r
        
        Returns
        -------
        pos_cam : np.ndarray
            圆心在相机坐标系下的3D位置 [X, Y, Z]
        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        # 估算深度 (使用平均焦距)
        f = (fx + fy) / 2
        Z = f * self.circle_radius / radius_px
        
        # 反投影得到3D坐标
        u, v = center
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
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
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n🎮 控制说明:")
        print("  'f' - 开启/关闭 跟随模式")
        print("  'h' - 机械臂回中")
        print("  'q' - 退出")
        print("="*60)
        
        following = False
        gain = 0.8             # 高增益快速跟踪
        step_limit = 0.080     # 80mm 每步最大移动
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                display = frame.copy()
                
                # 检测蓝色圆形
                success, center, radius_px, mask = self.detect_blue_circle(frame)
                
                if success:
                    # 绘制检测结果（绿色圆圈）
                    cv2.circle(display, center, radius_px, (0, 255, 0), 2)
                    cv2.circle(display, center, 3, (0, 255, 255), -1)
                    
                    # 估算3D位置
                    pos_cam = self.estimate_pose_from_circle(center, radius_px)
                    pos_cam_mm = pos_cam * 1000
                    
                    cv2.putText(display, f"Circle (Cam): [{pos_cam_mm[0]:.0f}, {pos_cam_mm[1]:.0f}, {pos_cam_mm[2]:.0f}] mm", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, f"Distance: {pos_cam[2]*1000:.0f} mm", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    if following and self.T_cam_gripper is not None:
                        # 获取机械臂位姿
                        T_gripper_base, q_curr = self.read_robot_pose(verbose=False)
                        
                        # 计算目标在Base坐标系下的位置
                        # ^B p = ^B T_G @ ^G T_C @ ^C p
                        pos_cam_homo = np.array([pos_cam[0], pos_cam[1], pos_cam[2], 1.0])
                        T_base_cam = T_gripper_base @ self.T_cam_gripper
                        pos_target_base = (T_base_cam @ pos_cam_homo)[:3]
                        
                        pos_gripper_base = T_gripper_base[:3, 3]
                        
                        # 直接向目标位置靠近（抓取模式）
                        error_base = pos_target_base - pos_gripper_base
                        
                        # 构造控制量 - 全部轴都向目标靠近
                        delta_base = error_base * gain
                        
                        # 限幅
                        norm_delta = np.linalg.norm(delta_base)
                        if norm_delta > step_limit:
                            delta_base = delta_base / norm_delta * step_limit
                        
                        # 计算目标位置
                        pos_gripper_des = pos_gripper_base + delta_base
                        
                        # 显示信息
                        pos_target_mm = pos_target_base * 1000
                        pos_curr_mm = pos_gripper_base * 1000  # 当前位置
                        pos_dest_mm = pos_gripper_des * 1000   # 即将到达的目标位置
                        error_mm = error_base * 1000
                        
                        cv2.putText(display, f"Target (Base): [{pos_target_mm[0]:.0f}, {pos_target_mm[1]:.0f}, {pos_target_mm[2]:.0f}] mm", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(display, f"Curr: [{pos_curr_mm[0]:.0f}, {pos_curr_mm[1]:.0f}, {pos_curr_mm[2]:.0f}] -> Dest: [{pos_dest_mm[0]:.0f}, {pos_dest_mm[1]:.0f}, {pos_dest_mm[2]:.0f}]", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        
                        # 日志输出
                        print(f"Target(Base): [{pos_target_mm[0]:7.1f}, {pos_target_mm[1]:7.1f}, {pos_target_mm[2]:7.1f}] | " +
                              f"Curr: [{pos_curr_mm[0]:7.1f}, {pos_curr_mm[1]:7.1f}, {pos_curr_mm[2]:7.1f}] | " +
                              f"Dest: [{pos_dest_mm[0]:7.1f}, {pos_dest_mm[1]:7.1f}, {pos_dest_mm[2]:7.1f}] | " +
                              f"Error: [{error_mm[0]:7.1f}, {error_mm[1]:7.1f}, {error_mm[2]:7.1f}]")
                        
                        # IK求解并执行
                        R_gripper_des = T_gripper_base[:3, :3]
                        T_gripper_base_des = np.eye(4)
                        T_gripper_base_des[:3, :3] = R_gripper_des
                        T_gripper_base_des[:3, 3] = pos_gripper_des
                        
                        ik_res = self.robot.ikine_LM(
                            T_gripper_base_des, 
                            q0=q_curr,
                            ilimit=100,
                            slimit=3,
                            tol=1e-3,
                            mask=np.array([1, 1, 1, 0.5, 0.5, 0]),
                            k=0.1,
                            method="sugihara"
                        )
                        
                        if ik_res.success:
                            q_new = ik_res.q
                            diff = np.linalg.norm(q_new - q_curr)
                            if diff < 1.5:
                                targets = self.robot.q_to_servo_targets(q_new)
                                self.controller.fast_move_to_pose(targets, speed=200)
                                cv2.putText(display, "TRACKING", (10, 150), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            else:
                                cv2.putText(display, f"Move too large: {diff:.2f}", (10, 150), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(display, "IK Failed", (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display, "No blue circle detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 状态显示
                status = "FOLLOW ON" if following else "FOLLOW OFF (Press 'f')"
                color = (0, 255, 0) if following else (0, 255, 255)
                cv2.putText(display, status, (10, display.shape[0]-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 显示mask (调试用)
                mask_small = cv2.resize(mask, (320, 180))
                mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                display[0:180, display.shape[1]-320:display.shape[1]] = mask_bgr
                
                cv2.imshow("Blue Circle Tracking", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    following = not following
                    print(f"Follow mode: {following}")
                elif key == ord('h'):
                    print("Returning to home...")
                    self.controller.move_all_home()
                    self.controller.move_servo("gripper", 3050)
                    self.controller.move_servo("wrist_roll", 850)
                    following = False
                    
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    tracker = BlueCircleTracker(circle_diameter=0.05)  # 5cm直径
    tracker.run()


if __name__ == "__main__":
    main()
