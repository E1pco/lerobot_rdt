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
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof_gripper


class EyeToHandCalibrator:
    """眼在手外 (Eye-to-Hand) 手眼标定器

    结构对齐 `handeye_calibration_eyeinhand.py`：
    - __init__ 仅负责参数/内参/棋盘点准备
    - init_robot/collect_data_interactive 负责硬件与采集
    - load_collected_data/calibrate/evaluate/save_result 提供离线流程
    """

    def __init__(
        self,
        board_size=(7, 5),
        square_size=0.018,  # meters
        intrinsic_file="./config_data/camera_intrinsics_environment.yaml",
        output_dir="./handeye_data_environment",
    ):
        self.board_size = board_size
        self.square_size = square_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 相机内参
        self.K = None
        self.dist = None
        self.load_camera_intrinsics(intrinsic_file)

        # 棋盘格 3D 点
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # 机器人
        self.robot = None
        self.controller = None

        # 采集数据
        self.T_target_cam_list = []
        self.T_gripper_base_list = []
        self.images = []

        # PnP 稳定性
        self.pose_buffer = []
        self.pose_buffer_size = 5

        print("=" * 70)
        print("🤖 眼在手外 (Eye-to-Hand) 手眼标定工具")
        print("=" * 70)
        print("\n棋盘格参数:")
        print(f"  内角点: {board_size[0]} × {board_size[1]}")
        print(f"  方格大小: {square_size * 1000:.2f} mm")
        print(f"\n数据保存目录: {os.path.abspath(output_dir)}")
        print("=" * 70)
    
    def load_camera_intrinsics(self, yaml_path):
        """加载相机内参 - 仅读取 K 与 distCoeffs (与 Eye-in-Hand 对齐)"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"未找到相机内参文件: {yaml_path}")

        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"无法打开相机内参文件: {yaml_path}")

        K_node = fs.getNode("K")
        if K_node.empty():
            K_node = fs.getNode("camera_matrix")

        dist_node = fs.getNode("distCoeffs")
        if dist_node.empty():
            dist_node = fs.getNode("distortion_coefficients")

        self.K = None if K_node.empty() else K_node.mat()
        self.dist = None if dist_node.empty() else dist_node.mat().flatten()
        fs.release()

        if self.K is None or self.dist is None:
            raise ValueError(f"相机内参文件缺少 K/distCoeffs: {yaml_path}")

        print(f"\n📷 已加载相机内参: {yaml_path}")
        print(f"   fx={self.K[0, 0]:.1f}, fy={self.K[1, 1]:.1f}")
        print(f"   cx={self.K[0, 2]:.1f}, cy={self.K[1, 2]:.1f}")
        print(f"   棋盘格尺寸: {self.board_size} (由脚本设定)")
        print(f"   方格大小: {self.square_size * 1000:.2f} mm (由脚本设定)")

    def init_robot(self, port="/dev/left_arm", baudrate=1_000_000):
        """初始化机器人和控制器"""
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
        """读取机器人当前末端位姿"""
        q = self.robot.read_joint_angles(joint_names=self.robot.joint_names, verbose=verbose)
        T_gripper_base = self.robot.fkine(q)

        if verbose:
            pos = T_gripper_base[:3, 3]
            euler = R.from_matrix(T_gripper_base[:3, :3]).as_euler("xyz", degrees=True)
            print("\n📍 末端位姿:")
            print(
                f"   位置: x={pos[0] * 1000:.1f}mm, y={pos[1] * 1000:.1f}mm, z={pos[2] * 1000:.1f}mm"
            )
            print(
                f"   姿态: roll={euler[0]:.1f}°, pitch={euler[1]:.1f}°, yaw={euler[2]:.1f}°"
            )

        return T_gripper_base, q
    
    def detect_chessboard(self, frame, use_ransac=True, refine_pose=True):
        """检测棋盘格并计算其在相机坐标系下的位姿 (与 Eye-in-Hand 对齐的鲁棒版本)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        if not found:
            return False, None, None, float("inf")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgp = corners.reshape(-1, 2).astype(np.float32)

        if use_ransac:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.objp,
                imgp,
                self.K,
                self.dist,
                iterationsCount=1000,
                reprojectionError=2.0,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if inliers is not None and len(inliers) < len(self.objp) * 0.8:
                return False, None, corners, float("inf")
        else:
            success, rvec, tvec = cv2.solvePnP(
                self.objp,
                imgp,
                self.K,
                self.dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not success:
            return False, None, corners, float("inf")

        if refine_pose:
            rvec, tvec = cv2.solvePnPRefineLM(self.objp, imgp, self.K, self.dist, rvec, tvec)

        reproj_pts, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.dist)
        reproj_error = np.sqrt(
            np.mean(np.sum((imgp - reproj_pts.reshape(-1, 2)) ** 2, axis=1))
        )
        if reproj_error > 2.0:
            return False, None, corners, reproj_error

        R_mat, _ = cv2.Rodrigues(rvec)
        T_target_cam = np.eye(4)
        T_target_cam[:3, :3] = R_mat
        T_target_cam[:3, 3] = tvec.squeeze()
        return True, T_target_cam, corners, reproj_error

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
    
    def collect_data_interactive(self, cam_id=0, width=1280, height=720):
        """交互式采集标定数据 (与 Eye-in-Hand 对齐)

        按键:
          SPACE - 采集当前位姿
          'h'   - 机械臂回中
          's'   - 显示/隐藏稳定性信息
          'q'   - 退出采集
        """

        # 每次采集创建一个新的 session 目录（与旧版一致）
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"session_{session_timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir

        print(f"📂 本次采集 session: {os.path.abspath(session_dir)}")

        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if not cap.isOpened():
            print("❌ 无法打开相机")
            return False

        print("\n📸 开始交互式数据采集")
        print("=" * 70)
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
        print("=" * 70 + "\n")

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
            stability_info = ""

            if success and corners is not None:
                cv2.drawChessboardCorners(display, self.board_size, corners, True)

                self.update_pose_buffer(T_target_cam)
                if len(self.pose_buffer) >= 3:
                    translations = np.array([T[:3, 3] for T in self.pose_buffer])
                    t_std = np.std(translations, axis=0) * 1000
                    t_std_norm = np.linalg.norm(t_std)
                    is_stable = t_std_norm < 3.0 and reproj_error < 1.0
                    if show_stability:
                        stability_info = f"Std: {t_std_norm:.1f}mm, ReprojErr: {reproj_error:.2f}px"

                distance = np.linalg.norm(T_target_cam[:3, 3]) * 1000
                color = (0, 255, 0) if is_stable else (0, 255, 255)
                status_text = "STABLE - Press SPACE" if is_stable else "Detecting..."
                cv2.putText(
                    display,
                    f"Distance: {distance:.0f}mm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                cv2.putText(
                    display,
                    status_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                if show_stability and stability_info:
                    cv2.putText(
                        display,
                        stability_info,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
            else:
                cv2.putText(
                    display,
                    "Chessboard NOT FOUND",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                self.pose_buffer = []

            cv2.putText(
                display,
                f"Samples: {sample_count}",
                (display.shape[1] - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Hand-Eye Calibration (Eye-to-Hand)", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n👋 退出采集")
                break
            if key == ord("h"):
                print("\n🏠 机械臂回中...")
                self.controller.move_all_home()
                time.sleep(1)
                continue
            if key == ord("s"):
                show_stability = not show_stability
                print(f"{'显示' if show_stability else '隐藏'}稳定性信息")
                continue
            if key == ord(" "):
                if not success:
                    print("⚠️  未检测到棋盘格，无法采集")
                    continue
                if not is_stable:
                    print("⚠️  位姿不稳定，建议等待稳定后再采集")

                T_avg = self.get_averaged_pose()
                if T_avg is not None:
                    T_to_save = T_avg
                    print("   使用平均位姿")
                else:
                    T_to_save = T_target_cam
                    print("   使用单帧位姿")

                sample_count += 1
                print(f"\n📸 采集数据 #{sample_count}")
                T_gripper_base, q = self.read_robot_pose(verbose=True)

                self.T_target_cam_list.append(T_to_save.copy())
                self.T_gripper_base_list.append(T_gripper_base.copy())
                self.images.append(frame.copy())

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                np.savez(
                    os.path.join(session_dir, f"pose_{sample_count:03d}.npz"),
                    T_target_cam=T_to_save,
                    T_gripper_base=T_gripper_base,
                    q=q,
                    reproj_error=reproj_error,
                )
                cv2.imwrite(
                    os.path.join(session_dir, f"image_{sample_count:03d}.jpg"),
                    frame,
                )

                # 同时保存可视化图，便于回看
                cv2.imwrite(
                    os.path.join(session_dir, f"vis_{sample_count:03d}.jpg"),
                    display,
                )

                print(f"✅ 已保存数据 #{sample_count}")
                print(f"   标定板距离: {np.linalg.norm(T_to_save[:3, 3]) * 1000:.1f} mm")
                print(f"   重投影误差: {reproj_error:.2f} px")

                self.pose_buffer = []

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 共采集 {sample_count} 组数据")
        return sample_count >= 3
    
    def load_collected_data(self, session_dir=None):
        """从文件加载已采集的数据 (优先最新 session_*/pose_*.npz)

        Parameters
        ----------
        session_dir : str | None
            指定 session 目录；为 None 时自动选择最新 session_*/，若不存在则回退 output_dir 根目录。
        """
        import glob

        base_dir = self.output_dir
        if session_dir is None:
            session_dirs = sorted(glob.glob(os.path.join(self.output_dir, "session_*")))
            if session_dirs:
                base_dir = session_dirs[-1]
        else:
            base_dir = session_dir

        pose_files = sorted(glob.glob(os.path.join(base_dir, "pose_*.npz")))
        if not pose_files:
            print(f"❌ 未找到标定数据: {base_dir}")
            return False

        self.T_target_cam_list = []
        self.T_gripper_base_list = []

        print(f"\n📂 加载标定数据: {base_dir}")
        for f in pose_files:
            data = np.load(f)
            self.T_target_cam_list.append(data["T_target_cam"])
            self.T_gripper_base_list.append(data["T_gripper_base"])
            print(f"   ✅ {os.path.basename(f)}")

        print(f"\n共加载 {len(self.T_target_cam_list)} 组数据")
        return True
    
    def calibrate(self):
        """执行眼在手外标定，返回 T_cam_base"""
        if len(self.T_gripper_base_list) < 3 or len(self.T_target_cam_list) < 3:
            print("❌ 数据不足，至少需要 3 组数据")
            return None

        print("\n🔧 开始眼在手外标定...")
        print(f"   数据组数: {len(self.T_gripper_base_list)}")

        # 准备数据
        R_gripper2base = []
        t_gripper2base = []
        R_base2gripper = []
        t_base2gripper = []

        R_target2cam = []
        t_target2cam = []
        R_cam2target = []
        t_cam2target = []

        for T_gb, T_tc in zip(self.T_gripper_base_list, self.T_target_cam_list):
            R_gripper2base.append(T_gb[:3, :3])
            t_gripper2base.append(T_gb[:3, 3])
            T_bg = np.linalg.inv(T_gb)
            R_base2gripper.append(T_bg[:3, :3])
            t_base2gripper.append(T_bg[:3, 3])

            R_target2cam.append(T_tc[:3, :3])
            t_target2cam.append(T_tc[:3, 3])
            T_ct = np.linalg.inv(T_tc)
            R_cam2target.append(T_ct[:3, :3])
            t_cam2target.append(T_ct[:3, 3])

        print("\n📊 数据质量检查:")
        self.analyze_data_quality(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

        strategies = [
            {
                "name": "Strategy 1: Base2Gripper + Target2Cam",
                "R_gripper": R_base2gripper,
                "t_gripper": t_base2gripper,
                "R_target": R_target2cam,
                "t_target": t_target2cam,
            },
            {
                "name": "Strategy 2: Gripper2Base + Cam2Target",
                "R_gripper": R_gripper2base,
                "t_gripper": t_gripper2base,
                "R_target": R_cam2target,
                "t_target": t_cam2target,
            },
            {
                "name": "Strategy 3: Target2Cam (as Robot) + Gripper2Base (as Target)",
                "R_gripper": R_target2cam,
                "t_gripper": t_target2cam,
                "R_target": R_gripper2base,
                "t_target": t_gripper2base,
            },
        ]

        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai-Lenz"),
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
        ]

        best_result = None
        best_score = float("inf")

        for strategy in strategies:
            print(f"\n🔄 尝试策略: {strategy['name']}")
            for method, method_name in methods:
                try:
                    R_calib, t_calib = cv2.calibrateHandEye(
                        strategy["R_gripper"],
                        strategy["t_gripper"],
                        strategy["R_target"],
                        strategy["t_target"],
                        method=method,
                    )

                    error = self.evaluate_calibration(
                        R_calib,
                        t_calib,
                        R_gripper2base,
                        t_gripper2base,
                        R_target2cam,
                        t_target2cam,
                    )
                    print(f"   {method_name}: {error:.6f} mm")

                    if error < best_score and not (np.isnan(error) or np.isinf(error)):
                        best_score = error
                        best_result = (R_calib, t_calib, method_name, strategy["name"])

                except Exception as e:
                    print(f"   {method_name} 失败: {e}")

        if best_result is None:
            print("❌ 所有标定算法都失败了")
            return None

        print(f"\n🔄 尝试非线性优化 (基于 {best_result[2]})...")
        try:
            R_opt, t_opt, error_opt = self.optimize_calibration(
                best_result[0],
                best_result[1],
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
            )
            print(f"   Optimization: {error_opt:.6f} mm")
            if error_opt < best_score:
                best_score = error_opt
                best_result = (R_opt, t_opt, "Optimization", "Non-linear Least Squares")
        except Exception as e:
            print(f"   优化失败: {e}")

        R_cam2base, t_cam2base, best_method, best_strategy = best_result
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3] = t_cam2base.flatten()

        # 自动判别是否需要取逆
        try:
            score_direct = self.evaluate_calibration(
                T_cam2base[:3, :3],
                T_cam2base[:3, 3].reshape(3, 1),
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
            )
            T_inv = np.linalg.inv(T_cam2base)
            score_inv = self.evaluate_calibration(
                T_inv[:3, :3],
                T_inv[:3, 3].reshape(3, 1),
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
            )
            if np.isfinite(score_inv) and (score_inv + 1e-9) < score_direct:
                print(
                    f"\nℹ️  检测到结果可能为逆变换：一致性 {score_direct:.6f} -> {score_inv:.6f} mm，已自动取逆"
                )
                T_cam2base = T_inv
                best_strategy = f"{best_strategy} (auto-inverted)"
                best_score = score_inv
            else:
                best_score = score_direct
        except Exception:
            pass

        print("\n✅ 眼在手外标定完成!")
        print(f"   最佳策略: {best_strategy}")
        print(f"   最佳算法: {best_method}")
        print(f"   一致性误差: {best_score:.6f} mm")
        print("\n🎯 相机到基座变换矩阵 (T_cam_base):")
        print("-" * 70)
        print(T_cam2base)
        print("-" * 70)

        return T_cam2base

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
    
    def save_result(self, T_cam_base, filename="handeye_result_envir.yaml"):
        """保存标定结果 (与 Eye-in-Hand 风格对齐)"""
        if T_cam_base is None:
            return

        filepath = os.path.join(self.output_dir, filename)
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        fs.write("T_cam_base", T_cam_base)

        R_mat = T_cam_base[:3, :3]
        t_vec = T_cam_base[:3, 3]
        euler = R.from_matrix(R_mat).as_euler("xyz", degrees=True)
        quat = R.from_matrix(R_mat).as_quat()

        fs.write("rotation_matrix", R_mat)
        fs.write("translation_vector", t_vec.reshape(3, 1))
        fs.write("euler_angles_deg", np.array(euler).reshape(3, 1))
        fs.write("quaternion_xyzw", np.array(quat).reshape(4, 1))
        fs.write("calibration_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        fs.write("num_samples", len(self.T_target_cam_list))
        fs.release()

        print(f"\n💾 标定结果已保存: {filepath}")

        npy_path = os.path.join(self.output_dir, "handeye_result_envir.npy")
        np.save(npy_path, T_cam_base)
        print(f"💾 标定结果已保存: {npy_path}")
    
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

        n = min(len(self.T_gripper_base_list), len(self.T_target_cam_list))
        for i in range(n):
            T_gb = self.T_gripper_base_list[i]
            T_tc = self.T_target_cam_list[i]
            
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

    def close(self):
        """关闭控制器"""
        if self.controller:
            self.controller.close()
            print("🔌 控制器已关闭")


def main():
    parser = argparse.ArgumentParser(description="眼在手外手眼标定工具")
    parser.add_argument("--collect", action="store_true", help="采集标定数据")
    parser.add_argument("--calibrate", action="store_true", help="执行标定计算")
    parser.add_argument("--all", action="store_true", help="采集+标定")

    parser.add_argument("--output-dir", default="./handeye_data_environment", help="数据保存目录")
    parser.add_argument(
        "--intrinsic",
        default="./config_data/camera_intrinsics_environment.yaml",
        help="相机内参文件 (OpenCV YAML, 仅读 K/distCoeffs)",
    )
    parser.add_argument("--square-size", type=float, default=18.0, help="棋盘格方格大小(mm)")
    parser.add_argument("--port", default="/dev/left_arm", help="串口")
    parser.add_argument("--video", type=int, default=0, help="视频设备ID")
    parser.add_argument("--width", type=int, default=1280, help="相机采集宽度")
    parser.add_argument("--height", type=int, default=720, help="相机采集高度")

    # 兼容旧参数
    parser.add_argument("--camera", type=int, help="(兼容) 相机设备ID，等同于 --video")
    parser.add_argument(
        "--camera-params",
        help="(兼容) 相机参数文件，等同于 --intrinsic (本脚本仅读取内参)",
    )

    args = parser.parse_args()

    if args.camera is not None:
        args.video = args.camera
    if args.camera_params is not None:
        args.intrinsic = args.camera_params

    calibrator = EyeToHandCalibrator(
        board_size=(7, 5),
        square_size=args.square_size / 1000.0,
        intrinsic_file=args.intrinsic,
        output_dir=args.output_dir,
    )

    try:
        if args.collect or args.all:
            calibrator.init_robot(port=args.port)
            print("\n🏠 机械臂回中...")
            calibrator.controller.move_all_home()
            time.sleep(1)
            calibrator.collect_data_interactive(cam_id=args.video, width=args.width, height=args.height)

        if args.calibrate or args.all or (not args.collect and not args.all):
            if not calibrator.T_target_cam_list:
                calibrator.load_collected_data()

            if calibrator.T_target_cam_list:
                T_cam_base = calibrator.calibrate()
                if T_cam_base is not None:
                    calibrator.evaluate_calibration_consistency(T_cam_base)
                    calibrator.save_result(T_cam_base)

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        calibrator.close()


if __name__ == "__main__":
    main()