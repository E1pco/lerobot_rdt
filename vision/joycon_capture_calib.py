#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# 添加上一级目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from joyconrobotics import JoyconRobotics
from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof


# ==============================================================
# 工具函数
# ==============================================================

def build_T(x, y, z, roll, pitch, yaw):
    """构造齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def solve_pnp(img, K, distCoeffs, pattern_size=(11, 8), square_size=0.022):
    """检测棋盘格角点并求解外参"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ret:
        print("⚠️ 棋盘格检测失败，请调整角度/光照。")
        return None, None, None

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    ret, rvec, tvec = cv2.solvePnP(objp, corners2, K, distCoeffs)
    if not ret:
        print("⚠️ solvePnP失败。")
        return None, None, None

    R_cam, _ = cv2.Rodrigues(rvec)
    T_target_cam = np.eye(4)
    T_target_cam[:3, :3] = R_cam
    T_target_cam[:3, 3] = tvec.squeeze()

    # 可视化角点
    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners2, True)
    cv2.imshow("Chessboard Detected", vis)
    cv2.waitKey(200)

    return ret, corners2, T_target_cam


# ==============================================================
# 主类：控制 + 拍照
# ==============================================================

class JoyConCapture:
    def __init__(self, port='/dev/ttyACM0', baudrate=1_000_000,
                 config_path='servo_config.json', cam_id=0,
                 save_dir='dataset'):

        # 相机内参（从 camera_intrinsics.yaml 硬编码）
        self.K = np.array([[664.44701044,0.,658.891941  ],
                           [0.,654.89004383, 406.58738455],
                           [0.,0.,1.]], dtype=np.float32)
        
        # 畸变系数
        self.distCoeffs = np.array([
            -0.22848866657422115, -0.24286465556211895,
            -0.0041375613727195667, -0.0214093589304933, 0.67109732798343458
        ], dtype=np.float32)

        # 舵机控制初始化
        self.controller = ServoController(port=port, baudrate=baudrate, config_path=config_path)
        self.robot = create_so101_5dof()
        # 关键：绑定舵机控制器到机器人对象
        self.robot.set_servo_controller(self.controller)

        # JoyCon 初始化
        self.joycon = JoyconRobotics(device='right', without_rest_init=False)

        # 相机初始化
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("❌ 无法打开相机")

        # 数据保存路径
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.img_idx = 0

        print("\n✅ 初始化完成：")
        print(f"相机内参:\n{self.K}")
        print(f"畸变系数: {self.distCoeffs}")
        print(f"数据保存目录: {save_dir}")
        print("\n📋 采集建议:")
        print("  1. 采集 8-12 组数据，确保足够多样性")
        print("  2. 在不同位置和姿态移动机械臂（X/Y/Z/Rx/Ry/Rz 各方向至少变化 5-10cm 或 30-45°）")
        print("  3. 确保棋盘格在图像中清晰可见")
        print("  4. 避免频繁移动导致的震动影响")
        print("  5. 确保机械臂稳定后再拍照\n")
        print("按 [A] 拍照保存，按 [X] 退出。\n")

    def _get_robot_pose(self):
        """读取当前舵机位置并计算 FK"""
        
        # 读取关节角度（传入必要参数确保 ServoController 可用）
        q = self.robot.read_joint_angles(
            joint_names=self.robot.joint_names,
            verbose=False
        )
        
        # ✅ 使用 fk() 方法返回 [X, Y, Z, roll, pitch, yaw]
        pose_6d = self.robot.fk(q)
        # fk() 返回 [X, Y, Z, gamma, beta, alpha] = [X, Y, Z, Yaw, Pitch, Roll]
        x, y, z = pose_6d[0:3]
        yaw, pitch, roll = pose_6d[3:6]
        
        # 构造 4x4 齐次变换矩阵
        T_gripper_base = np.eye(4)
        T_gripper_base[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        T_gripper_base[:3, 3] = [x, y, z]
        print(f"\n✅ FK 计算结果:")
        print(f"   末端位置: ({x:.4f}, {y:.4f}, {z:.4f}) m")
        print(f"   末端姿态: R={np.degrees(roll):.2f}° P={np.degrees(pitch):.2f}° Y={np.degrees(yaw):.2f}°")
        print(f"   T_gripper^base =\n{T_gripper_base}")
        
        return T_gripper_base, q

    def _normalize_transform(self, T):
        """规范化变换矩阵（确保旋转矩阵正交）"""
        T_norm = T.copy()
        U, _, Vt = np.linalg.svd(T[:3, :3])
        T_norm[:3, :3] = U @ Vt
        return T_norm

    def run(self):
        print("🎮 开始采集，移动机械臂对准棋盘格...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            display = frame.copy()
            cv2.putText(display, f"Image #{self.img_idx}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Camera", display)

            self.joycon.update()
            # A 键拍照
            if self.joycon.button.a == 1:
                print(f"\n📸 拍照 #{self.img_idx} 中...")
                time.sleep(0.5)  # 等待机械臂稳定，避免振动影响
                
                # 【关键】先读取机械臂位姿确保同步！
                print("📍 读取机械臂位姿中...")
                poses = []
                qs = []
                for _ in range(3):
                    T, q = self._get_robot_pose()
                    poses.append(T)
                    qs.append(q)
                    time.sleep(0.05)
                
                # 对多次读取的位姿进行平均
                T_gripper_base = np.mean(poses, axis=0)
                T_gripper_base = self._normalize_transform(T_gripper_base)
                q = np.mean(qs, axis=0)  # 关节角度也取平均
                
                # 【然后】立即拍照
                print("📷 立即拍照...")
                time.sleep(0.1)
                
                # 连续拍摄多帧，取最清晰的
                frames_to_capture = 3
                best_frame = None
                best_sharpness = 0
                
                for _ in range(frames_to_capture):
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                    
                    # 计算图像清晰度（Laplacian 方差）
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame = frame
                    
                    time.sleep(0.05)
                
                if best_frame is None:
                    continue
                
                frame = best_frame
                ret_pnp, corners, T_target_cam = solve_pnp(frame, self.K, self.distCoeffs)
                if not ret_pnp:
                    print("⚠️ 未检测到棋盘格，请重试。")
                    continue
                
                # 保存数据
                np.savez(os.path.join(self.save_dir, f"pose_{self.img_idx:02d}.npz"),
                         T_target_cam=T_target_cam,
                         T_gripper_base=T_gripper_base,
                         q=q,  # 保存原始关节角度以便诊断
                         image=frame)
                cv2.imwrite(os.path.join(self.save_dir, f"img_{self.img_idx:02d}.jpg"), frame)

                print(f"✅ 已保存 pose_{self.img_idx:02d}.npz 和对应图片")
                print(f"   图像清晰度: {best_sharpness:.2f}")
                self.img_idx += 1
                time.sleep(0.5)

            # X 键退出
            if self.joycon.button.x == 1:
                print("\n🛑 退出采集。")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# ==============================================================
# 主入口
# ==============================================================

if __name__ == "__main__":
    collector = JoyConCapture(
        port='/dev/ttyACM0',
        baudrate=1_000_000,
        config_path='servo_config.json',
        cam_id=0,
        save_dir='dataset_eyeinhand'
    )
    collector.run()
