#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的相机内参标定脚本
=======================
针对PnP精度问题的优化措施:
1. 增加标定图像质量检查
2. 计算详细的重投影误差分析
3. 检测图像姿态多样性
4. 提供标定质量评估报告
5. 支持多种棋盘格类型

使用方法:
  python calibrate_camera_improved.py --capture   # 实时拍照采集
  python calibrate_camera_improved.py --calibrate # 使用已有图像标定
  python calibrate_camera_improved.py --all       # 采集+标定
"""

import cv2
import numpy as np
import glob
import os
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation as R


class CameraCalibrator:
    """改进的相机标定器"""
    
    def __init__(self, 
                 board_size=(11, 8),
                 square_size=0.02073,  # 20.73mm
                 image_folder="./calib_images_right"):
        """
        Parameters
        ----------
        board_size : tuple
            棋盘格内角点数 (列, 行)
        square_size : float
            棋盘格方格边长 (单位: 米)
        image_folder : str
            标定图像存放文件夹 (如果是采集模式，将作为根目录创建会话子目录)
        """
        self.board_size = board_size
        self.square_size = square_size
        self.image_folder = image_folder
        
        # 默认输出前缀 (将在 capture_images 中更新，或在 calibrate 中使用当前 image_folder)
        self.output_prefix = os.path.join(image_folder, "camera_intrinsics")
        
        # 构造世界坐标点
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 存储标定数据
        self.objpoints = []
        self.imgpoints = []
        self.image_files = []
        self.image_size = None
        
        # 标定结果
        self.K = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_errors = []
        
        print("="*70)
        print("📷 相机内参标定工具 (改进版)")
        print("="*70)
        print(f"\n棋盘格参数:")
        print(f"  内角点: {board_size[0]} × {board_size[1]}")
        print(f"  方格大小: {square_size*1000:.1f} mm")
        print(f"  棋盘物理宽度: {(board_size[0]-1)*square_size*1000:.1f} mm")
        print(f"  棋盘物理高度: {(board_size[1]-1)*square_size*1000:.1f} mm")
        print(f"\n图像文件夹: {os.path.abspath(image_folder)}")
        print("="*70 + "\n")
        
        # 如果不是采集模式，确保目录存在
        if not os.path.exists(image_folder):
            try:
                os.makedirs(image_folder, exist_ok=True)
            except:
                pass
    
    def capture_images(self, cam_id=2, min_images=15, max_images=30):
        """
        交互式采集标定图像
        
        改进点:
        1. 自动创建带时间戳的会话目录
        2. 实时显示角点检测状态
        3. 检查图像姿态多样性
        """
        # 创建本次采集的会话目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.image_folder, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        # 更新当前工作的图像目录和输出前缀
        self.image_folder = session_dir
        self.output_prefix = os.path.join(session_dir, "camera_intrinsics")
        
        print(f"📂 创建采集会话目录: {session_dir}")
        
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ 无法打开相机")
            return False
        
        print("📸 开始采集标定图像")
        print("="*70)
        print("\n📖 采集指南 (确保标定精度的关键!):")
        print("  1. 棋盘格应覆盖图像的不同区域 (中心、四角、边缘)")
        print("  2. 变换棋盘格的倾斜角度 (至少 ±30°)")
        print("  3. 调整棋盘格距离 (近、中、远)")
        print("  4. 保持棋盘格完全在视野内")
        print("  5. 确保光线均匀，避免反光")
        print(f"\n  建议采集 {min_images}-{max_images} 张图像")
        print("\n⌨️  快捷键: SPACE=拍照, Q=退出\n")
        
        captured_poses = []  # 记录已采集的姿态
        count = 0
        
        while count < max_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()
            
            # 检测棋盘格
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
            
            if found:
                # 亚像素精化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 绘制角点
                cv2.drawChessboardCorners(display, self.board_size, corners, found)
                
                # 计算棋盘格姿态 (用于检查多样性)
                success, rvec, tvec = cv2.solvePnP(
                    self.objp, corners, 
                    np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float32),
                    None
                )
                
                if success:
                    euler = R.from_rotvec(rvec.flatten()).as_euler('xyz', degrees=True)
                    distance = np.linalg.norm(tvec) * 1000
                    
                    # 显示当前姿态
                    cv2.putText(display, f"Distance: {distance:.0f}mm", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Angle: {euler[0]:.0f}, {euler[1]:.0f}, {euler[2]:.0f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 检查姿态是否新颖
                    is_novel = self._check_pose_novelty(captured_poses, rvec, tvec)
                    if not is_novel:
                        cv2.putText(display, "Pose similar to existing - try different angle!", (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                cv2.putText(display, "DETECTED - Press SPACE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "NOT DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 显示进度
            h, w = display.shape[:2]
            cv2.putText(display, f"Captured: {count}/{max_images} (min: {min_images})", 
                       (w-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示区域覆盖提示
            self._draw_coverage_guide(display, captured_poses)
            
            cv2.imshow('Camera Calibration', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and found:
                # 保存图像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.image_folder, f"calib_{count:02d}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                
                # 记录姿态
                captured_poses.append({
                    'rvec': rvec.copy() if success else None,
                    'tvec': tvec.copy() if success else None,
                    'corners': corners.copy()
                })
                
                count += 1
                print(f"✅ 已保存: {filename} ({count}/{max_images})")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if count < min_images:
            print(f"\n⚠️  警告: 仅采集 {count} 张图像，建议至少 {min_images} 张")
        
        return count >= min_images
    
    def _check_pose_novelty(self, captured_poses, rvec, tvec, 
                            rot_threshold=15, trans_threshold=0.05):
        """检查当前姿态是否与已有姿态有足够差异"""
        if not captured_poses:
            return True
        
        for pose in captured_poses:
            if pose['rvec'] is None:
                continue
            
            # 旋转差异
            rot_diff = np.linalg.norm(rvec - pose['rvec'])
            rot_diff_deg = np.degrees(rot_diff)
            
            # 平移差异
            trans_diff = np.linalg.norm(tvec - pose['tvec'])
            
            if rot_diff_deg < rot_threshold and trans_diff < trans_threshold:
                return False
        
        return True
    
    def _draw_coverage_guide(self, display, captured_poses):
        """绘制图像区域覆盖引导"""
        h, w = display.shape[:2]
        
        # 将图像分成 3x3 区域
        regions = np.zeros((3, 3), dtype=bool)
        
        for pose in captured_poses:
            if pose['corners'] is not None:
                center = np.mean(pose['corners'], axis=0)[0]
                col = int(center[0] / (w / 3))
                row = int(center[1] / (h / 3))
                col = min(2, max(0, col))
                row = min(2, max(0, row))
                regions[row, col] = True
        
        # 绘制区域网格
        cell_w, cell_h = w // 3, h // 3
        for i in range(3):
            for j in range(3):
                x, y = j * cell_w, i * cell_h
                color = (0, 255, 0) if regions[i, j] else (0, 0, 255)
                cv2.rectangle(display, (x+2, y+2), (x+cell_w-2, y+cell_h-2), color, 1)
        
        # 统计覆盖率
        coverage = np.sum(regions) / 9 * 100
        cv2.putText(display, f"Coverage: {coverage:.0f}%", (w-150, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def load_images(self):
        """加载标定图像并检测角点"""
        images = sorted(glob.glob(os.path.join(self.image_folder, "*.jpg")) +
                       glob.glob(os.path.join(self.image_folder, "*.png")))
        
        if not images:
            print(f"❌ 未找到标定图像: {self.image_folder}")
            return False
        
        print(f"📁 找到 {len(images)} 张标定图像")
        
        self.objpoints = []
        self.imgpoints = []
        self.image_files = []
        
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                print(f"  ⚠️ 无法读取: {fname}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.image_size is None:
                self.image_size = gray.shape[::-1]
            
            # 检测角点
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
            
            if found:
                # 亚像素精化 (关键步骤!)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                self.image_files.append(fname)
                print(f"  ✅ {os.path.basename(fname)}")
            else:
                print(f"  ❌ 未检测到角点: {os.path.basename(fname)}")
        
        print(f"\n成功检测 {len(self.imgpoints)}/{len(images)} 张图像")
        return len(self.imgpoints) >= 3
    
    def calibrate(self):
        """执行相机标定"""
        if not self.imgpoints:
            print("❌ 没有可用的标定数据")
            return False
        
        print("\n📷 开始相机标定...")
        print("-"*70)
        
        # 使用改进的标定参数
        flags = 0
        # 可选: 固定某些参数
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO  # 固定纵横比
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST  # 忽略切向畸变
        # flags |= cv2.CALIB_FIX_K3  # 固定 k3
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        
        ret, self.K, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, 
            None, None, flags=flags, criteria=criteria
        )
        
        print(f"\n✅ 标定完成!")
        print(f"\n内参矩阵 K:")
        print(f"  fx = {self.K[0,0]:.2f}")
        print(f"  fy = {self.K[1,1]:.2f}")
        print(f"  cx = {self.K[0,2]:.2f}")
        print(f"  cy = {self.K[1,2]:.2f}")
        
        print(f"\n畸变系数:")
        print(f"  k1 = {self.dist[0,0]:.6f}")
        print(f"  k2 = {self.dist[0,1]:.6f}")
        print(f"  p1 = {self.dist[0,2]:.6f}")
        print(f"  p2 = {self.dist[0,3]:.6f}")
        print(f"  k3 = {self.dist[0,4]:.6f}")
        
        return True
    
    def evaluate_calibration(self):
        """详细评估标定质量"""
        if self.K is None:
            print("❌ 请先执行标定")
            return
        
        print("\n📊 标定质量评估")
        print("="*70)
        
        # 1. 计算每张图像的重投影误差
        total_error = 0
        self.reprojection_errors = []
        per_image_errors = []
        
        for i in range(len(self.objpoints)):
            imgpoints_proj, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], self.K, self.dist
            )
            error = cv2.norm(self.imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
            per_image_errors.append(error)
            total_error += error
            self.reprojection_errors.append({
                'image': self.image_files[i] if i < len(self.image_files) else f"Image {i}",
                'error': error
            })
        
        mean_error = total_error / len(self.objpoints)
        
        print(f"\n📏 重投影误差 (像素):")
        print(f"   平均误差: {mean_error:.4f}")
        print(f"   最大误差: {max(per_image_errors):.4f}")
        print(f"   最小误差: {min(per_image_errors):.4f}")
        print(f"   标准差:   {np.std(per_image_errors):.4f}")
        
        # 质量评级
        if mean_error < 0.3:
            quality = "优秀 ⭐⭐⭐"
        elif mean_error < 0.5:
            quality = "良好 ⭐⭐"
        elif mean_error < 1.0:
            quality = "一般 ⭐"
        else:
            quality = "较差 ⚠️"
        
        print(f"\n   质量评级: {quality}")
        
        # 2. 分析畸变程度
        print(f"\n📐 畸变分析:")
        k1 = self.dist[0,0]
        if abs(k1) > 0.3:
            print(f"   ⚠️ 径向畸变较大 (k1={k1:.4f})，建议使用去畸变后的图像")
        elif abs(k1) > 0.1:
            print(f"   中等径向畸变 (k1={k1:.4f})")
        else:
            print(f"   径向畸变较小 (k1={k1:.4f})")
        
        # 3. 检查焦距一致性
        fx, fy = self.K[0,0], self.K[1,1]
        aspect_ratio = fx / fy
        print(f"\n📏 焦距分析:")
        print(f"   fx/fy 比值: {aspect_ratio:.4f}")
        if abs(aspect_ratio - 1.0) > 0.01:
            print(f"   ⚠️ 焦距不对称，可能存在传感器非正方形像素或标定问题")
        
        # 4. 检查主点位置
        cx, cy = self.K[0,2], self.K[1,2]
        img_center_x, img_center_y = self.image_size[0] / 2, self.image_size[1] / 2
        offset_x = abs(cx - img_center_x)
        offset_y = abs(cy - img_center_y)
        
        print(f"\n📍 主点分析:")
        print(f"   主点位置: ({cx:.1f}, {cy:.1f})")
        print(f"   图像中心: ({img_center_x:.1f}, {img_center_y:.1f})")
        print(f"   偏移量: ({offset_x:.1f}, {offset_y:.1f}) 像素")
        
        if offset_x > 50 or offset_y > 50:
            print(f"   ⚠️ 主点偏离图像中心较远，可能影响精度")
        
        # 5. 列出误差最大的图像
        print(f"\n📋 各图像重投影误差:")
        sorted_errors = sorted(self.reprojection_errors, key=lambda x: x['error'], reverse=True)
        for item in sorted_errors[:5]:  # 显示前5个
            status = "⚠️" if item['error'] > 0.5 else "✅"
            print(f"   {status} {os.path.basename(item['image'])}: {item['error']:.4f} px")
        
        # 建议
        print(f"\n💡 优化建议:")
        if mean_error > 0.5:
            print("   1. 重新采集标定图像，确保角点清晰")
            print("   2. 增加图像数量和姿态多样性")
            print("   3. 检查棋盘格是否平整")
        if abs(k1) > 0.3:
            print("   4. 考虑使用更高阶的畸变模型")
        
        return mean_error
    
    def save_results(self):
        """保存标定结果"""
        if self.K is None:
            print("❌ 没有标定结果可保存")
            return
        
        # 保存 OpenCV YAML 格式 (内参)
        yaml_file = f"{self.output_prefix}.yaml"
        fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_WRITE)
        fs.write("K", self.K)
        fs.write("distCoeffs", self.dist)
        fs.write("image_width", self.image_size[0])
        fs.write("image_height", self.image_size[1])
        fs.write("board_size_cols", self.board_size[0])
        fs.write("board_size_rows", self.board_size[1])
        fs.write("square_size", self.square_size)
        fs.write("calibration_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 计算并保存平均重投影误差
        if self.reprojection_errors:
            errors = [e['error'] for e in self.reprojection_errors]
            fs.write("mean_reprojection_error", np.mean(errors))
        
        fs.release()
        print(f"\n💾 内参已保存: {yaml_file}")
        
        # 保存外参到YAML (每张图像的 T_target_cam)
        if self.rvecs is not None:
            extrinsic_yaml_file = os.path.join(self.image_folder, "extrinsics.yaml")
            fs_ext = cv2.FileStorage(extrinsic_yaml_file, cv2.FILE_STORAGE_WRITE)
            
            fs_ext.write("num_images", len(self.rvecs))
            fs_ext.write("board_size_cols", self.board_size[0])
            fs_ext.write("board_size_rows", self.board_size[1])
            fs_ext.write("square_size", self.square_size)
            
            extrinsics = []
            for i, (rvec, tvec) in enumerate(zip(self.rvecs, self.tvecs)):
                R_mat, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R_mat
                T[:3, 3] = tvec.squeeze()
                extrinsics.append(T)
                
                # 写入每个外参矩阵
                fs_ext.write(f"T_target_cam_{i}", T)
                fs_ext.write(f"rvec_{i}", rvec)
                fs_ext.write(f"tvec_{i}", tvec)
                
                # 写入对应的图像文件名
                if i < len(self.image_files):
                    fs_ext.write(f"image_{i}", os.path.basename(self.image_files[i]))
            
            fs_ext.release()
            print(f"💾 外参已保存 (YAML): {extrinsic_yaml_file}")
            
            # 同时保存npy格式以兼容旧代码
            npy_file = os.path.join(self.image_folder, "extrinsics.npy")
            np.save(npy_file, np.array(extrinsics))
            print(f"💾 外参已保存 (NPY): {npy_file}")
        
        # 保存详细报告
        report_file = f"{self.output_prefix}_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("相机标定报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("棋盘格参数:\n")
            f.write(f"  内角点: {self.board_size[0]} × {self.board_size[1]}\n")
            f.write(f"  方格大小: {self.square_size*1000:.1f} mm\n\n")
            
            f.write("内参矩阵 K:\n")
            f.write(f"  [[{self.K[0,0]:.6f}, {self.K[0,1]:.6f}, {self.K[0,2]:.6f}],\n")
            f.write(f"   [{self.K[1,0]:.6f}, {self.K[1,1]:.6f}, {self.K[1,2]:.6f}],\n")
            f.write(f"   [{self.K[2,0]:.6f}, {self.K[2,1]:.6f}, {self.K[2,2]:.6f}]]\n\n")
            
            f.write("畸变系数:\n")
            f.write(f"  k1={self.dist[0,0]:.8f}\n")
            f.write(f"  k2={self.dist[0,1]:.8f}\n")
            f.write(f"  p1={self.dist[0,2]:.8f}\n")
            f.write(f"  p2={self.dist[0,3]:.8f}\n")
            f.write(f"  k3={self.dist[0,4]:.8f}\n\n")
            
            if self.reprojection_errors:
                f.write("重投影误差:\n")
                for item in self.reprojection_errors:
                    f.write(f"  {os.path.basename(item['image'])}: {item['error']:.4f} px\n")
                errors = [e['error'] for e in self.reprojection_errors]
                f.write(f"\n  平均: {np.mean(errors):.4f} px\n")
                f.write(f"  标准差: {np.std(errors):.4f} px\n")
        
        print(f"💾 报告已保存: {report_file}")
    
    def undistort_test(self):
        """测试去畸变效果"""
        if self.K is None:
            print("❌ 请先执行标定")
            return
        
        if not self.image_files:
            print("❌ 没有可用的测试图像")
            return
        
        # 使用第一张图像测试
        img = cv2.imread(self.image_files[0])
        h, w = img.shape[:2]
        
        # 计算最优新相机矩阵
        new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
        
        # 去畸变
        undistorted = cv2.undistort(img, self.K, self.dist, None, new_K)
        
        # 裁剪有效区域
        x, y, roi_w, roi_h = roi
        if roi_w > 0 and roi_h > 0:
            undistorted_cropped = undistorted[y:y+roi_h, x:x+roi_w]
        else:
            undistorted_cropped = undistorted
        
        # 显示对比
        comparison = np.hstack([
            cv2.resize(img, (640, 480)),
            cv2.resize(undistorted, (640, 480))
        ])
        
        cv2.putText(comparison, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Undistorted", (650, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Undistortion Test', comparison)
        print("\n📷 显示去畸变对比图，按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存去畸变图像
        save_path = os.path.join(self.image_folder, 'undistorted_test.jpg')
        cv2.imwrite(save_path, undistorted)
        print(f"💾 去畸变测试图像已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='相机内参标定工具')
    parser.add_argument('--capture',                                                action='store_true', help='采集标定图像')
    parser.add_argument('--calibrate', action='store_true', help='执行标定')
    parser.add_argument('--all', action='store_true', help='采集+标定')
    parser.add_argument('--camid', type=int, default=0, help='相机ID')
    parser.add_argument('--image-folder', default='./calib_images_right', help='图像文件夹')
    
    args = parser.parse_args()
    
    calibrator = CameraCalibrator(
        image_folder=args.image_folder
    )
    
    if args.capture or args.all:
        calibrator.capture_images(cam_id=args.camid)
    
    if args.calibrate or args.all or (not args.capture and not args.all):
        if calibrator.load_images():
            if calibrator.calibrate():
                calibrator.evaluate_calibration()
                calibrator.save_results()
                calibrator.undistort_test()


if __name__ == '__main__':
    main()
