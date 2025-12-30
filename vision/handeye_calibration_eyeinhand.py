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
import glob
import numpy as np
import time
import argparse
import shutil
from datetime import datetime
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof_gripper
from handeye_utils import (
    evaluate_eye_in_hand_consistency,
    print_consistency_report,
)


class HandEyeCalibrator:
    """眼在手上手眼标定器"""

    def __init__(
        self,
        board_size=(11, 8),
        square_size=0.02073,
        intrinsic_file="camera_intrinsics.yaml",
        output_dir="./handeye_data",
    ):
        self.board_size = board_size
        self.square_size = square_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.load_camera_intrinsics(intrinsic_file)

        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        self.robot = None
        self.controller = None

        self.T_target_cam_list = []
        self.T_gripper_base_list = []
        self.images = []

        self.pose_buffer = []
        self.pose_buffer_size = 5

        print("=" * 70)
        print("🤖 眼在手上 (Eye-in-Hand) 手眼标定工具")
        print("=" * 70)
        print(f"\n棋盘格参数:")
        print(f"  内角点: {board_size[0]} × {board_size[1]}")
        print(f"  方格大小: {square_size * 1000:.2f} mm")
        print(f"\n数据保存目录: {os.path.abspath(output_dir)}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # 内参加载
    # ------------------------------------------------------------------
    def load_camera_intrinsics(self, yaml_path):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"未找到相机内参文件: {yaml_path}")
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        self.K = fs.getNode("K").mat()
        self.dist = fs.getNode("distCoeffs").mat().flatten()
        fs.release()
        print(f"\n📷 已加载相机内参: {yaml_path}")
        print(f"   fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
        print(f"   cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}")

    # ------------------------------------------------------------------
    # 机器人初始化 & 读取位姿
    # ------------------------------------------------------------------
    def init_robot(self, port="/dev/ttyACM0", baudrate=1_000_000):
        print("\n🤖 初始化机器人...")
        self.controller = ServoController(
            port=port,
            baudrate=baudrate,
            config_path=os.path.join(os.path.dirname(__file__), "../driver/servo_config.json"),
        )
        self.robot = create_so101_5dof_gripper()
        self.robot.set_servo_controller(self.controller)
        print("✅ 机器人初始化完成")
        return True

    def read_robot_pose(self, verbose=True):
        q = self.robot.read_joint_angles(joint_names=self.robot.joint_names, verbose=verbose)
        T_gripper_base = self.robot.fkine(q)
        if verbose:
            pos = T_gripper_base[:3, 3]
            euler = R.from_matrix(T_gripper_base[:3, :3]).as_euler("xyz", degrees=True)
            print(f"\n📍 末端位姿:")
            print(f"   位置: x={pos[0]*1000:.1f}mm, y={pos[1]*1000:.1f}mm, z={pos[2]*1000:.1f}mm")
            print(f"   姿态: roll={euler[0]:.1f}°, pitch={euler[1]:.1f}°, yaw={euler[2]:.1f}°")
        return T_gripper_base, q

    # ------------------------------------------------------------------
    # 棋盘格检测 & PnP
    # ------------------------------------------------------------------
    def detect_chessboard(self, frame, use_ransac=True, refine_pose=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        if not found:
            return False, None, None, float("inf")

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgp = corners.reshape(-1, 2).astype(np.float32)

        if use_ransac:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.objp, imgp, self.K, self.dist, iterationsCount=1000, reprojectionError=2.0
            )
            if inliers is not None and len(inliers) < len(self.objp) * 0.8:
                return False, None, corners, float("inf")
        else:
            success, rvec, tvec = cv2.solvePnP(self.objp, imgp, self.K, self.dist)

        if not success:
            return False, None, corners, float("inf")

        if refine_pose:
            rvec, tvec = cv2.solvePnPRefineLM(self.objp, imgp, self.K, self.dist, rvec, tvec)

        reproj, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.dist)
        reproj_error = np.sqrt(np.mean(np.sum((imgp - reproj.reshape(-1, 2)) ** 2, axis=1)))
        if reproj_error > 2.0:
            return False, None, corners, reproj_error

        Rmat, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = tvec.squeeze()
        return True, T, corners, reproj_error

    # ------------------------------------------------------------------
    # 位姿缓冲 & 平均
    # ------------------------------------------------------------------
    def update_pose_buffer(self, T):
        self.pose_buffer.append(T.copy())
        if len(self.pose_buffer) > self.pose_buffer_size:
            self.pose_buffer.pop(0)

    def get_averaged_pose(self):
        if len(self.pose_buffer) < 3:
            return None
        translations = np.array([T[:3, 3] for T in self.pose_buffer])
        t_avg = np.mean(translations, axis=0)
        quats = np.array([R.from_matrix(T[:3, :3]).as_quat() for T in self.pose_buffer])
        q_avg = np.mean(quats, axis=0)
        q_avg /= np.linalg.norm(q_avg)
        R_avg = R.from_quat(q_avg).as_matrix()
        T_avg = np.eye(4)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = t_avg
        return T_avg

    # ------------------------------------------------------------------
    # 交互式采集
    # ------------------------------------------------------------------
    def collect_data_interactive(self, cam_id=0):
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("❌ 无法打开相机")
            return False

        print("\n📸 开始交互式数据采集")
        print("=" * 70)
        print("   SPACE - 采集 | 'h' - 回中 | 's' - 显示/隐藏稳定性 | 'q' - 退出")
        print("=" * 70)

        sample_count = 0
        show_stability = True
        self.pose_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            success, T_target_cam, corners, reproj_error = self.detect_chessboard(frame)
            is_stable = False

            if success and corners is not None:
                cv2.drawChessboardCorners(display, self.board_size, corners, True)
                self.update_pose_buffer(T_target_cam)

                if len(self.pose_buffer) >= 3:
                    t_std = np.std([T[:3, 3] for T in self.pose_buffer], axis=0) * 1000
                    t_std_norm = np.linalg.norm(t_std)
                    is_stable = t_std_norm < 3.0 and reproj_error < 1.0
                    if show_stability:
                        cv2.putText(
                            display,
                            f"Std: {t_std_norm:.1f}mm, Err: {reproj_error:.2f}px",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                dist_mm = np.linalg.norm(T_target_cam[:3, 3]) * 1000
                color = (0, 255, 0) if is_stable else (0, 255, 255)
                cv2.putText(display, f"Distance: {dist_mm:.0f}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(
                    display,
                    "STABLE - SPACE" if is_stable else "Detecting...",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            else:
                cv2.putText(display, "Chessboard NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.pose_buffer = []

            cv2.putText(
                display, f"Samples: {sample_count}", (display.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
            cv2.imshow("Hand-Eye Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n👋 退出采集")
                break
            elif key == ord("h"):
                print("\n🏠 机械臂回中...")
                self.controller.move_all_home()
                time.sleep(1)
            elif key == ord("s"):
                show_stability = not show_stability
            elif key == ord(" "):
                if not success:
                    print("⚠️  未检测到棋盘格")
                    continue
                if not is_stable:
                    print("⚠️  位姿不稳定，建议等待稳定后再采集")

                T_to_save = self.get_averaged_pose() if self.get_averaged_pose() is not None else T_target_cam
                sample_count += 1
                print(f"\n📸 采集 #{sample_count}")

                T_gripper_base, q = self.read_robot_pose(verbose=True)
                self.T_target_cam_list.append(T_to_save.copy())
                self.T_gripper_base_list.append(T_gripper_base.copy())
                self.images.append(frame.copy())

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                np.savez(
                    os.path.join(self.output_dir, f"pose_{sample_count:02d}_{ts}.npz"),
                    T_target_cam=T_to_save,
                    T_gripper_base=T_gripper_base,
                    q=q,
                    reproj_error=reproj_error,
                )
                cv2.imwrite(os.path.join(self.output_dir, f"image_{sample_count:02d}_{ts}.jpg"), frame)
                print(f"   距离: {np.linalg.norm(T_to_save[:3,3])*1000:.1f}mm | 重投影: {reproj_error:.2f}px")
                self.pose_buffer = []

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 共采集 {sample_count} 组数据")
        return sample_count >= 3

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------
    def load_collected_data(self):
        pose_files = sorted(glob.glob(os.path.join(self.output_dir, "pose_*.npz")))
        if not pose_files:
            print(f"❌ 未找到标定数据: {self.output_dir}")
            return False
        self.T_target_cam_list = []
        self.T_gripper_base_list = []
        print(f"\n📂 加载标定数据...")
        for f in pose_files:
            data = np.load(f)
            self.T_target_cam_list.append(data["T_target_cam"])
            self.T_gripper_base_list.append(data["T_gripper_base"])
            print(f"   ✅ {os.path.basename(f)}")
        print(f"\n共加载 {len(self.T_target_cam_list)} 组数据")
        return True

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------
    def calibrate(self, method=cv2.CALIB_HAND_EYE_TSAI):
        if len(self.T_target_cam_list) < 3:
            print("❌ 数据不足，至少需要 3 组")
            return None

        print(f"\n🔄 开始手眼标定 (数据组数: {len(self.T_target_cam_list)})")

        R_g2b = [T[:3, :3] for T in self.T_gripper_base_list]
        t_g2b = [T[:3, 3].reshape(3, 1) for T in self.T_gripper_base_list]
        R_t2c = [T[:3, :3] for T in self.T_target_cam_list]
        t_t2c = [T[:3, 3].reshape(3, 1) for T in self.T_target_cam_list]

        R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)

        T_cam_gripper = np.eye(4)
        T_cam_gripper[:3, :3] = R_c2g
        T_cam_gripper[:3, 3] = t_c2g.squeeze()

        t_mm = t_c2g.squeeze() * 1000
        euler = R.from_matrix(R_c2g).as_euler("xyz", degrees=True)
        print("\n✅ 手眼标定完成 (T_cam_gripper)")
        print("-" * 70)
        print(f"平移 (mm): tx={t_mm[0]:.2f}, ty={t_mm[1]:.2f}, tz={t_mm[2]:.2f}")
        print(f"旋转 (°): roll={euler[0]:.2f}, pitch={euler[1]:.2f}, yaw={euler[2]:.2f}")
        print("-" * 70)
        return T_cam_gripper

    # ------------------------------------------------------------------
    # 一致性评估 (调用公共模块)
    # ------------------------------------------------------------------
    def evaluate_calibration(self, T_cam_gripper):
        print("\n📊 标定结果评估")
        print("=" * 70)
        result = evaluate_eye_in_hand_consistency(
            T_cam_gripper, self.T_gripper_base_list, self.T_target_cam_list
        )
        print_consistency_report(result, "一致性误差 (AX=XB)")
        print("=" * 70)

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    def save_result(self, T_cam_gripper, filename="handeye_result.yaml"):
        if T_cam_gripper is None:
            return
        filepath = os.path.join(self.output_dir, filename)
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        fs.write("T_cam_gripper", T_cam_gripper)
        R_mat = T_cam_gripper[:3, :3]
        t_vec = T_cam_gripper[:3, 3]
        euler = R.from_matrix(R_mat).as_euler("xyz", degrees=True)
        quat = R.from_matrix(R_mat).as_quat()
        fs.write("rotation_matrix", R_mat)
        fs.write("translation_vector", t_vec.reshape(3, 1))
        fs.write("euler_angles_deg", np.array(euler).reshape(3, 1))
        fs.write("quaternion_xyzw", np.array(quat).reshape(4, 1))
        fs.write("calibration_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        fs.write("num_samples", len(self.T_target_cam_list))
        fs.release()
        print(f"\n💾 已保存: {filepath}")

        npy_path = os.path.join(self.output_dir, "handeye_result.npy")
        np.save(npy_path, T_cam_gripper)
        print(f"💾 已保存: {npy_path}")

        # 复制到 vision 根目录
        root_yaml = os.path.join(os.path.dirname(__file__), "handeye_result.yaml")
        root_npy = os.path.join(os.path.dirname(__file__), "handeye_result.npy")
        shutil.copy(filepath, root_yaml)
        shutil.copy(npy_path, root_npy)
        print(f"💾 已复制: {root_yaml}")

    def close(self):
        if self.controller:
            self.controller.close()
            print("🔌 控制器已关闭")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="眼在手上手眼标定工具")
    parser.add_argument("--collect", action="store_true", help="采集标定数据")
    parser.add_argument("--calibrate", action="store_true", help="执行标定计算")
    parser.add_argument("--all", action="store_true", help="采集+标定")
    parser.add_argument("--output-dir", default="./handeye_data_right", help="数据保存目录")
    parser.add_argument("--intrinsic", default="./config_data/camera_intrinsics_right.yaml", help="相机内参文件")
    parser.add_argument("--square-size", type=float, default=20.73, help="棋盘格方格大小(mm)")
    parser.add_argument("--port", default="/dev/ttyACM0", help="串口")
    parser.add_argument("--video", type=int, default=0, help="视频设备ID")
    args = parser.parse_args()

    calibrator = HandEyeCalibrator(
        board_size=(11, 8),
        square_size=args.square_size / 1000.0,
        intrinsic_file=args.intrinsic,
        output_dir=args.output_dir,
    )

    try:
        if args.collect or args.all:
            calibrator.init_robot(port=args.port)
            print("\n🏠 机械臂回中...")
            calibrator.collect_data_interactive(cam_id=args.video)

        if args.calibrate or args.all or (not args.collect and not args.all):
            if not calibrator.T_target_cam_list:
                calibrator.load_collected_data()
            if calibrator.T_target_cam_list:
                T = calibrator.calibrate()
                if T is not None:
                    calibrator.evaluate_calibration(T)
                    calibrator.save_result(T)
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        calibrator.close()


if __name__ == "__main__":
    main()
