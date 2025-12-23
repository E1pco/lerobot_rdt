#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PnP 精度测试工具 - 棋盘格版本
==============================
功能：
  1. 实时相机预览
  2. 按 SPACE 拍照
  3. 自动检测棋盘格 (11×8)
  4. 计算 PnP 得到棋盘的位姿和到相机的距离
  5. 输出结果供用户用尺子验证

按键：
  SPACE - 拍照
  'c'   - 显示相机参数
  'r'   - 重置
  'e'   - 导出结果
  'q'   - 退出
"""

import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from datetime import datetime


class ChessboardPnPTester:
    def __init__(self, square_size_mm=20.73):
        """
        初始化棋盘格 PnP 测试工具
        
        Parameters
        ----------
        square_size_mm : float
            棋盘格方格大小（毫米）
        """
        # 加载相机内参
        self.load_camera_intrinsics()
        
        # 棋盘格参数
        self.board_size = (11, 8)  # 11×8
        self.square_size = square_size_mm / 1000.0  # 转换为米
        
        print(f"📐 棋盘格参数:")
        print(f"   尺寸: {self.board_size[0]}×{self.board_size[1]}")
        print(f"   方格大小: {square_size_mm} mm")
        print(f"   总宽度: {(self.board_size[0]-1)*square_size_mm} mm")
        print(f"   总高度: {(self.board_size[1]-1)*square_size_mm} mm\n")
        
        # 相机捕获
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.frame = None
        self.image_count = 0
        self.results = []
        
        print("="*70)
        print("🎥 棋盘格 PnP 精度测试工具")
        print("="*70)
        print("\n📖 说明:")
        print("  1. 将棋盘格 (11×8) 放在相机前")
        print("  2. 按 SPACE 拍照")
        print("  3. 工具会自动检测棋盘格角点并计算 PnP")
        print("  4. 输出棋盘到相机的距离和姿态")
        print("  5. 用尺子测量实际距离与计算值进行对比")
        print("\n⌨️  快捷键:")
        print("  SPACE - 拍照")
        print("  'c'   - 显示相机参数")
        print("  'r'   - 重置数据")
        print("  'e'   - 导出结果")
        print("  'q'   - 退出")
        print("="*70 + "\n")
    
    def load_camera_intrinsics(self):
        """加载相机内参"""
        yaml_path = 'camera_intrinsics.yaml'
        
        if not os.path.exists(yaml_path):
            print(f"⚠️  警告：未找到 {yaml_path}")
            print("   使用默认内参")
            # 默认内参（标准 USB 摄像头）
            self.K = np.array([
                [800, 0, 640],
                [0, 800, 360],
                [0, 0, 1]
            ], dtype=np.float32)
            self.dist = np.zeros(5)
        else:
            try:
                # 使用 OpenCV 读取格式
                fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
                self.K = fs.getNode('K').mat()
                self.dist = fs.getNode('distCoeffs').mat().flatten()
                fs.release()
                
                print(f"✅ 已加载相机内参 (OpenCV 格式): {yaml_path}")
                print(f"   焦距: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
                print(f"   主点: cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}\n")
            except Exception as e:
                print(f"⚠️  加载 YAML 失败: {e}")
                print("   使用默认内参\n")
                # 默认内参
                self.K = np.array([
                    [800, 0, 640],
                    [0, 800, 360],
                    [0, 0, 1]
                ], dtype=np.float32)
                self.dist = np.zeros(5)
    
    def show_camera_params(self):
        """显示相机参数"""
        print("\n" + "="*70)
        print("📷 相机内参")
        print("="*70)
        print("\n相机矩阵 K:")
        print(self.K)
        print("\n畸变系数:")
        print(self.dist)
        print("="*70 + "\n")
    
    def detect_chessboard(self, frame):
        """
        检测棋盘格角点
        
        Returns
        -------
        tuple
            (found, corners) - 是否找到、角点坐标
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测棋盘格角点
        found, corners = cv2.findChessboardCorners(
            gray, self.board_size, None,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if found:
            # 精细化角点
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return found, corners
    
    def draw_chessboard(self, frame, corners, found, rvec=None, tvec=None):
        """在图像上绘制棋盘格和坐标轴"""
        if found and corners is not None:
            # 绘制棋盘格
            frame = cv2.drawChessboardCorners(frame, self.board_size, corners, found)
            
            # 绘制坐标轴
            if rvec is not None and tvec is not None:
                axis_length = 0.05  # 5cm
                axis_points = np.float32([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, axis_length]
                ])
                
                img_points, _ = cv2.projectPoints(
                    axis_points, rvec, tvec, self.K, self.dist
                )
                img_points = img_points.astype(int)
                
                origin = tuple(img_points[0].ravel())
                x_end = tuple(img_points[1].ravel())
                y_end = tuple(img_points[2].ravel())
                z_end = tuple(img_points[3].ravel())
                
                cv2.line(frame, origin, x_end, (0, 0, 255), 3)  # X 红
                cv2.line(frame, origin, y_end, (0, 255, 0), 3)  # Y 绿
                cv2.line(frame, origin, z_end, (255, 0, 0), 3)  # Z 蓝
        
        return frame
    
    def calculate_pnp_chessboard(self, corners):
        """
        使用 PnP 计算棋盘格的位姿
        
        Parameters
        ----------
        corners : np.ndarray
            棋盘格角点坐标
            
        Returns
        -------
        dict or None
            包含 rvec, tvec 等信息
        """
        if corners is None or len(corners) == 0:
            return None
        
        # 定义棋盘格的 3D 点（在棋盘坐标系中）
        objp = np.zeros((self.board_size[0]*self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # 缩放为实际尺寸
        
        # 2D 图像坐标
        image_points = corners.reshape(-1, 2).astype(np.float32)
        
        # 求解 PnP
        success, rvec, tvec = cv2.solvePnP(
            objp,
            image_points,
            self.K,
            self.dist,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            distance = np.linalg.norm(tvec)
            rotation_mat, _ = cv2.Rodrigues(rvec)
            euler_angles = R.from_matrix(rotation_mat).as_euler('xyz', degrees=True)
            
            return {
                'success': success,
                'rvec': rvec,
                'tvec': tvec,
                'distance': distance,
                'euler': euler_angles,
                'rotation_mat': rotation_mat,
                'objp': objp,
                'imgp': image_points
            }
        
        return None
    
    def print_result(self, result):
        """打印 PnP 结果"""
        if not result or not result['success']:
            print(f"❌ PnP 求解失败")
            return
        
        print(f"\n✅ 棋盘格 PnP 结果:")
        print("-" * 70)
        
        # 位置（相对于相机）
        tvec = result['tvec']
        print(f"📍 位置 (相对相机):")
        print(f"   X = {tvec[0,0]*1000:8.2f} mm")
        print(f"   Y = {tvec[1,0]*1000:8.2f} mm")
        print(f"   Z = {tvec[2,0]*1000:8.2f} mm (深度)")
        print(f"   距离 = {result['distance']*1000:8.2f} mm")
        
        # 姿态（欧拉角）
        euler = result['euler']
        print(f"\n🔄 姿态 (欧拉角):")
        print(f"   Roll  (X轴) = {euler[0]:8.2f}°")
        print(f"   Pitch (Y轴) = {euler[1]:8.2f}°")
        print(f"   Yaw   (Z轴) = {euler[2]:8.2f}°")
        
        print("-" * 70)
    
    def run(self):
        """主循环"""
        print("🎬 启动相机预览...\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 无法读取摄像头")
                break
            
            # 检测棋盘格
            found, corners = self.detect_chessboard(frame)
            
            # 计算 PnP
            rvec = None
            tvec = None
            if found and corners is not None:
                pnp_result = self.calculate_pnp_chessboard(corners)
                if pnp_result:
                    rvec = pnp_result['rvec']
                    tvec = pnp_result['tvec']
            
            # 绘制结果
            display_frame = self.draw_chessboard(frame.copy(), corners, found, rvec, tvec)
            
            # 显示说明
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, "Press SPACE to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'c' for camera params, 'q' to quit", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 显示棋盘格检测状态
            if found:
                cv2.putText(display_frame, "Chessboard: DETECTED", (w-250, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Chessboard: NOT FOUND", (w-250, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Chessboard PnP Tester', display_frame)
            
            # 键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 退出程序")
                break
            
            elif key == ord(' '):
                # 拍照
                self.image_count += 1
                print(f"\n📸 拍照 #{self.image_count}")
                
                if not found or corners is None:
                    print("   ⚠️  未检测到棋盘格")
                    continue
                
                # 保存图像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = f"chessboard_pnp_{timestamp}_{self.image_count}.jpg"
                cv2.imwrite(img_path, frame)
                print(f"   ✅ 图像已保存: {img_path}")
                
                # 计算并显示 PnP 结果
                pnp_result = self.calculate_pnp_chessboard(corners)
                if pnp_result:
                    self.print_result(pnp_result)
                    self.results.append({
                        'image_num': self.image_count,
                        'result': pnp_result
                    })
                    
                    # 提示用户测量
                    print("\n📏 请用尺子测量:")
                    print("   1. 棋盘格到相机的距离")
                    print("   2. 棋盘格在图像中的位置")
                    print("   3. 与计算值对比")
                    print("   4. 记录误差\n")
            
            elif key == ord('c'):
                self.show_camera_params()
            
            elif key == ord('r'):
                print("\n🔄 重置数据")
                self.results = []
                self.image_count = 0
            
            elif key == ord('e'):
                self.export_results()
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def export_results(self):
        """导出结果到文件"""
        if not self.results:
            print("   ⚠️  没有数据可导出")
            return
        
        filename = f"chessboard_pnp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("棋盘格 PnP 精度测试结果\n")
            f.write(f"棋盘格参数: {self.board_size[0]}×{self.board_size[1]}, " +
                   f"方格大小: {self.square_size*1000:.1f} mm\n")
            f.write("="*70 + "\n\n")
            
            for record in self.results:
                f.write(f"图像 #{record['image_num']}\n")
                f.write("-"*70 + "\n")
                
                result = record['result']
                tvec = result['tvec']
                
                f.write(f"位置 (相对相机, 单位: mm):\n")
                f.write(f"  X = {tvec[0,0]*1000:8.2f}\n")
                f.write(f"  Y = {tvec[1,0]*1000:8.2f}\n")
                f.write(f"  Z = {tvec[2,0]*1000:8.2f}\n")
                f.write(f"  距离 = {result['distance']*1000:8.2f}\n\n")
                
                euler = result['euler']
                f.write(f"姿态 (欧拉角, 单位: 度):\n")
                f.write(f"  Roll  = {euler[0]:8.2f}\n")
                f.write(f"  Pitch = {euler[1]:8.2f}\n")
                f.write(f"  Yaw   = {euler[2]:8.2f}\n\n")
        
        print(f"✅ 结果已导出: {filename}\n")


def main():
    tester = ChessboardPnPTester(square_size_mm=20.73)
    tester.run()


if __name__ == '__main__':
    main()
