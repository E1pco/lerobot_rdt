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
  python handeye_calibration_eyetohand.py --collect --video 0 --port /dev/left_arm
  python handeye_calibration_eyetohand.py --calibrate
  python handeye_calibration_eyetohand.py --all --video 0 --port /dev/left_arm

坐标系定义:
  - base: 机械臂基座坐标系
  - gripper/end-effector: 机械臂末端坐标系
  - cam: 相机坐标系 (固定在环境中)
  - target: 标定板坐标系
"""

import os
import sys
import cv2
import glob
import numpy as np
import time
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof_gripper
from handeye_utils import (
    evaluate_eye_to_hand_consistency,
    print_consistency_report,
)


class EyeToHandCalibrator:
    """眼在手外 (Eye-to-Hand) 手眼标定器"""

    def __init__(
        self,
        board_size=(7, 5),
        square_size=0.018,
        intrinsic_file="./config_data/camera_intrinsics_environment.yaml",
        output_dir="./handeye_data_environment",
    ):
        self.board_size = board_size
        self.square_size = square_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.K = None
        self.dist = None
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
        print("🤖 眼在手外 (Eye-to-Hand) 手眼标定工具")
        print("=" * 70)
        print(f"\n棋盘格参数: {board_size[0]}×{board_size[1]}, {square_size*1000:.2f}mm")
        print(f"数据保存目录: {os.path.abspath(output_dir)}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # 内参加载
    # ------------------------------------------------------------------
    def load_camera_intrinsics(self, yaml_path):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"未找到相机内参文件: {yaml_path}")
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
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
        print(f"   fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}, cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}")

    # ------------------------------------------------------------------
    # 机器人
    # ------------------------------------------------------------------
    def init_robot(self, port="/dev/left_arm", baudrate=1_000_000):
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
        T = self.robot.fkine(q)
        if verbose:
            pos = T[:3, 3]
            euler = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
            print(f"\n📍 末端: x={pos[0]*1000:.1f}mm y={pos[1]*1000:.1f}mm z={pos[2]*1000:.1f}mm | "
                  f"r={euler[0]:.1f}° p={euler[1]:.1f}° y={euler[2]:.1f}°")
        return T, q

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
        T_avg = np.eye(4)
        T_avg[:3, :3] = R.from_quat(q_avg).as_matrix()
        T_avg[:3, 3] = t_avg
        return T_avg

    # ------------------------------------------------------------------
    # 交互式采集
    # ------------------------------------------------------------------
    def collect_data_interactive(self, cam_id=0, width=1280, height=720):
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"session_{session_ts}")
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        print(f"📂 Session: {os.path.abspath(session_dir)}")

        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if not cap.isOpened():
            print("❌ 无法打开相机")
            return False

        print("\n📸 采集: SPACE采集 | h回中 | s显示/隐藏稳定性 | q退出")

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
                        cv2.putText(display, f"Std:{t_std_norm:.1f}mm Err:{reproj_error:.2f}px",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                dist_mm = np.linalg.norm(T_target_cam[:3, 3]) * 1000
                color = (0, 255, 0) if is_stable else (0, 255, 255)
                cv2.putText(display, f"Dist: {dist_mm:.0f}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, "STABLE" if is_stable else "Detecting...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(display, "Chessboard NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.pose_buffer = []

            cv2.putText(display, f"Samples: {sample_count}", (display.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Eye-to-Hand Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n👋 退出")
                break
            elif key == ord("h"):
                self.controller.move_all_home()
                time.sleep(1)
            elif key == ord("s"):
                show_stability = not show_stability
            elif key == ord(" "):
                if not success:
                    print("⚠️  未检测到棋盘格")
                    continue

                T_to_save = self.get_averaged_pose() if self.get_averaged_pose() is not None else T_target_cam
                sample_count += 1
                print(f"\n📸 #{sample_count}")

                T_gripper_base, q = self.read_robot_pose(verbose=True)
                self.T_target_cam_list.append(T_to_save.copy())
                self.T_gripper_base_list.append(T_gripper_base.copy())
                self.images.append(frame.copy())

                np.savez(
                    os.path.join(session_dir, f"pose_{sample_count:03d}.npz"),
                    T_target_cam=T_to_save,
                    T_gripper_base=T_gripper_base,
                    q=q,
                    reproj_error=reproj_error,
                )
                cv2.imwrite(os.path.join(session_dir, f"image_{sample_count:03d}.jpg"), frame)
                print(f"   距离: {np.linalg.norm(T_to_save[:3,3])*1000:.1f}mm | 重投影: {reproj_error:.2f}px")
                self.pose_buffer = []

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 共采集 {sample_count} 组")
        return sample_count >= 3

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------
    def load_collected_data(self, session_dir=None):
        base_dir = self.output_dir
        if session_dir is None:
            sessions = sorted(glob.glob(os.path.join(self.output_dir, "session_*")))
            if sessions:
                base_dir = sessions[-1]
        else:
            base_dir = session_dir

        pose_files = sorted(glob.glob(os.path.join(base_dir, "pose_*.npz")))
        if not pose_files:
            print(f"❌ 未找到数据: {base_dir}")
            return False

        self.T_target_cam_list = []
        self.T_gripper_base_list = []
        print(f"\n📂 加载: {base_dir}")
        for f in pose_files:
            data = np.load(f)
            self.T_target_cam_list.append(data["T_target_cam"])
            self.T_gripper_base_list.append(data["T_gripper_base"])
        print(f"   共 {len(self.T_target_cam_list)} 组")
        return True

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------
    def calibrate(self):
        if len(self.T_gripper_base_list) < 3:
            print("❌ 数据不足")
            return None

        print(f"\n🔧 开始标定 (数据: {len(self.T_gripper_base_list)} 组)")

        # 准备数据
        R_g2b, t_g2b, R_b2g, t_b2g = [], [], [], []
        R_t2c, t_t2c, R_c2t, t_c2t = [], [], [], []

        for T_gb, T_tc in zip(self.T_gripper_base_list, self.T_target_cam_list):
            R_g2b.append(T_gb[:3, :3])
            t_g2b.append(T_gb[:3, 3])
            T_bg = np.linalg.inv(T_gb)
            R_b2g.append(T_bg[:3, :3])
            t_b2g.append(T_bg[:3, 3])

            R_t2c.append(T_tc[:3, :3])
            t_t2c.append(T_tc[:3, 3])
            T_ct = np.linalg.inv(T_tc)
            R_c2t.append(T_ct[:3, :3])
            t_c2t.append(T_ct[:3, 3])

        self._analyze_data_quality(t_g2b, R_g2b)

        strategies = [
            ("Base2Gripper+Target2Cam", R_b2g, t_b2g, R_t2c, t_t2c),
            ("Gripper2Base+Cam2Target", R_g2b, t_g2b, R_c2t, t_c2t),
            ("Target2Cam+Gripper2Base", R_t2c, t_t2c, R_g2b, t_g2b),
        ]
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
        ]

        best_result, best_score = None, float("inf")

        for s_name, r_grip, t_grip, r_tgt, t_tgt in strategies:
            print(f"\n🔄 {s_name}")
            for method, m_name in methods:
                try:
                    R_c, t_c = cv2.calibrateHandEye(r_grip, t_grip, r_tgt, t_tgt, method=method)
                    err = self._eval_score(R_c, t_c, R_g2b, t_g2b, R_t2c, t_t2c)
                    print(f"   {m_name}: {err:.4f}mm")
                    if err < best_score and np.isfinite(err):
                        best_score = err
                        best_result = (R_c, t_c, m_name, s_name)
                except Exception as e:
                    print(f"   {m_name} 失败: {e}")

        if best_result is None:
            print("❌ 所有算法失败")
            return None

        # 非线性优化
        print(f"\n🔄 非线性优化 (基于 {best_result[2]})...")
        try:
            R_opt, t_opt, err_opt = self._optimize(best_result[0], best_result[1], R_g2b, t_g2b, R_t2c, t_t2c)
            print(f"   Opt: {err_opt:.4f}mm")
            if err_opt < best_score:
                best_score = err_opt
                best_result = (R_opt, t_opt, "Opt", best_result[3])
        except Exception as e:
            print(f"   优化失败: {e}")

        R_cb, t_cb, m_name, s_name = best_result
        T_cam_base = np.eye(4)
        T_cam_base[:3, :3] = R_cb
        T_cam_base[:3, 3] = t_cb.flatten()

        # 自动判别是否需要取逆
        score_direct = self._eval_score(T_cam_base[:3, :3], T_cam_base[:3, 3], R_g2b, t_g2b, R_t2c, t_t2c)
        T_inv = np.linalg.inv(T_cam_base)
        score_inv = self._eval_score(T_inv[:3, :3], T_inv[:3, 3], R_g2b, t_g2b, R_t2c, t_t2c)
        if np.isfinite(score_inv) and score_inv + 1e-9 < score_direct:
            print(f"\nℹ️  自动取逆: {score_direct:.4f} -> {score_inv:.4f}mm")
            T_cam_base = T_inv
            s_name += " (inv)"
            best_score = score_inv

        print(f"\n✅ 标定完成: {s_name} / {m_name} / err={best_score:.4f}mm")
        print("-" * 70)
        print(T_cam_base)
        print("-" * 70)
        return T_cam_base

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _analyze_data_quality(self, t_list, R_list):
        trans = np.array(t_list)
        t_range = np.ptp(trans, axis=0)
        rots = np.array([R.from_matrix(r).as_euler("xyz", degrees=True) for r in R_list])
        r_range = np.ptp(rots, axis=0)
        print(f"   平移范围: X={t_range[0]:.3f}m Y={t_range[1]:.3f}m Z={t_range[2]:.3f}m")
        print(f"   旋转范围: R={r_range[0]:.1f}° P={r_range[1]:.1f}° Y={r_range[2]:.1f}°")
        if np.any(t_range < 0.05):
            print("   ⚠️  建议增加平移变化量 (>5cm)")
        if np.any(r_range < 10):
            print("   ⚠️  建议增加旋转变化量 (>10°)")

    def _eval_score(self, R_cb, t_cb, R_g2b, t_g2b, R_t2c, t_t2c):
        """一致性评估: T_target_gripper 的平移标准差 (mm)"""
        T_cb = np.eye(4)
        T_cb[:3, :3] = R_cb
        T_cb[:3, 3] = np.array(t_cb).flatten()

        trans = []
        for i in range(len(R_g2b)):
            T_gb = np.eye(4)
            T_gb[:3, :3] = R_g2b[i]
            T_gb[:3, 3] = t_g2b[i]
            T_tc = np.eye(4)
            T_tc[:3, :3] = R_t2c[i]
            T_tc[:3, 3] = t_t2c[i]
            T_tg = np.linalg.inv(T_gb) @ T_cb @ T_tc
            trans.append(T_tg[:3, 3])
        return float(np.mean(np.std(trans, axis=0)) * 1000)

    def _optimize(self, R_init, t_init, R_g2b, t_g2b, R_t2c, t_t2c):
        """非线性优化"""
        T_cb = np.eye(4)
        T_cb[:3, :3] = R_init
        T_cb[:3, 3] = np.array(t_init).flatten()

        # 初始化 T_gripper_target
        T_gts = []
        for i in range(len(R_g2b)):
            T_gb = np.eye(4)
            T_gb[:3, :3] = R_g2b[i]
            T_gb[:3, 3] = t_g2b[i]
            T_tc = np.eye(4)
            T_tc[:3, :3] = R_t2c[i]
            T_tc[:3, 3] = t_t2c[i]
            T_gts.append(np.linalg.inv(T_gb) @ np.linalg.inv(T_cb) @ T_tc)

        t_gt = np.mean([T[:3, 3] for T in T_gts], axis=0)
        R_gt = R.from_matrix([T[:3, :3] for T in T_gts]).mean().as_matrix()
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt

        x0 = np.concatenate([
            R.from_matrix(T_cb[:3, :3]).as_rotvec(), T_cb[:3, 3],
            R.from_matrix(T_gt[:3, :3]).as_rotvec(), T_gt[:3, 3],
        ])

        def residuals(params):
            T_X = np.eye(4)
            T_X[:3, :3] = R.from_rotvec(params[:3]).as_matrix()
            T_X[:3, 3] = params[3:6]
            T_Z = np.eye(4)
            T_Z[:3, :3] = R.from_rotvec(params[6:9]).as_matrix()
            T_Z[:3, 3] = params[9:12]

            res = []
            for i in range(len(R_g2b)):
                T_bg = np.eye(4)
                T_bg[:3, :3] = R_g2b[i]
                T_bg[:3, 3] = t_g2b[i]
                T_tc_obs = np.eye(4)
                T_tc_obs[:3, :3] = R_t2c[i]
                T_tc_obs[:3, 3] = t_t2c[i]
                T_tc_pred = T_X @ T_bg @ T_Z
                res.extend(T_tc_pred[:3, 3] - T_tc_obs[:3, 3])
                diff_R = T_tc_pred[:3, :3] @ T_tc_obs[:3, :3].T
                res.extend(R.from_matrix(diff_R).as_rotvec() * 0.1)
            return np.array(res)

        sol = least_squares(residuals, x0, verbose=0)
        R_opt = R.from_rotvec(sol.x[:3]).as_matrix()
        t_opt = sol.x[3:6].reshape(3, 1)
        err = self._eval_score(R_opt, t_opt, R_g2b, t_g2b, R_t2c, t_t2c)
        return R_opt, t_opt, err

    # ------------------------------------------------------------------
    # 一致性评估 (调用公共模块)
    # ------------------------------------------------------------------
    def evaluate_calibration(self, T_cam_base):
        print("\n📊 标定结果一致性评估")
        print("=" * 70)
        result = evaluate_eye_to_hand_consistency(
            T_cam_base, self.T_gripper_base_list, self.T_target_cam_list
        )
        print_consistency_report(result, "一致性误差 (标定板相对末端)")
        print("=" * 70)

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    def save_result(self, T_cam_base, filename="handeye_result_envir.yaml"):
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
        print(f"\n💾 已保存: {filepath}")

        npy_path = os.path.join(self.output_dir, "handeye_result_envir.npy")
        np.save(npy_path, T_cam_base)
        print(f"💾 已保存: {npy_path}")

    def close(self):
        if self.controller:
            self.controller.close()
            print("🔌 控制器已关闭")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="眼在手外手眼标定工具")
    parser.add_argument("--collect", action="store_true", help="采集数据")
    parser.add_argument("--calibrate", action="store_true", help="执行标定")
    parser.add_argument("--all", action="store_true", help="采集+标定")
    parser.add_argument("--output-dir", default="./handeye_data_environment")
    parser.add_argument("--intrinsic", default="./config_data/camera_intrinsics_environment.yaml")
    parser.add_argument("--square-size", type=float, default=18.0, help="棋盘格方格大小(mm)")
    parser.add_argument("--port", default="/dev/left_arm")
    parser.add_argument("--video", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    # 兼容旧参数
    parser.add_argument("--camera", type=int, help="(兼容) 等同于 --video")
    parser.add_argument("--camera-params", help="(兼容) 等同于 --intrinsic")
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
