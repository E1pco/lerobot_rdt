#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PnP鲁棒性测试脚本
==================
测试并对比不同PnP方法的稳定性

功能:
  1. 实时显示多种PnP方法的结果
  2. 统计位姿稳定性 (标准差)
  3. 显示重投影误差
  4. 多帧平均滤波效果对比

使用方法:
  python robust_pnp_test.py --camera 0 --square-size 20.73
"""

import cv2
import numpy as np
import argparse
import yaml
from collections import deque
from scipy.spatial.transform import Rotation as R


class RobustPnPTester:
    """PnP鲁棒性测试器"""
    
    def __init__(self, board_size=(11, 8), square_size=0.02073, intrinsics_file=None):
        self.board_size = board_size
        self.square_size = square_size
        
        # 加载相机内参
        if intrinsics_file:
            self.load_intrinsics(intrinsics_file)
        
        # 生成棋盘格3D点
        self.objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 位姿历史 (用于统计稳定性)
        self.history_size = 30
        self.pose_history = {
            'ITERATIVE': deque(maxlen=self.history_size),
            'RANSAC': deque(maxlen=self.history_size),
            'EPNP': deque(maxlen=self.history_size),
            'IPPE': deque(maxlen=self.history_size),
            'SQPNP': deque(maxlen=self.history_size),
            'AVERAGED': deque(maxlen=self.history_size),
        }
        
        # PnP方法映射
        self.pnp_methods = {
            'ITERATIVE': cv2.SOLVEPNP_ITERATIVE,
            'EPNP': cv2.SOLVEPNP_EPNP,
            'IPPE': cv2.SOLVEPNP_IPPE,
            'SQPNP': cv2.SOLVEPNP_SQPNP,
        }
        
    def load_intrinsics(self, filepath):
        """加载相机内参 (支持OpenCV格式和标准YAML格式)"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # 检查是否是OpenCV格式
        if content.startswith('%YAML'):
            # OpenCV格式，使用cv2.FileStorage
            fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
            K = fs.getNode('K').mat()
            dist = fs.getNode('distCoeffs').mat()
            fs.release()
            
            if K is not None:
                self.K = K
                self.dist = dist.flatten() if dist is not None else np.zeros(5)
                print(f"✅ 加载相机内参(OpenCV格式): fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
            else:
                raise ValueError("无法从OpenCV格式文件中读取相机内参")
        else:
            # 标准YAML格式
            data = yaml.safe_load(content)
            
            self.K = np.array([
                [data['fx'], 0, data['cx']],
                [0, data['fy'], data['cy']],
                [0, 0, 1]
            ], dtype=np.float64)
            
            self.dist = np.array([
                data['k1'], data['k2'], 
                data.get('p1', 0), data.get('p2', 0), 
                data.get('k3', 0)
            ])
            
            print(f"✅ 加载相机内参: fx={data['fx']:.1f}, fy={data['fy']:.1f}")
        
    def detect_corners(self, frame):
        """检测棋盘格角点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE + 
                cv2.CALIB_CB_FAST_CHECK)
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        
        if not found:
            return None
        
        # 亚像素精化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return corners
    
    def solve_pnp(self, corners, method_name):
        """使用指定方法求解PnP"""
        imgp = corners.reshape(-1, 2).astype(np.float32)
        
        if method_name == 'RANSAC':
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.objp, imgp, self.K, self.dist,
                iterationsCount=1000,
                reprojectionError=2.0,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return None, None, float('inf')
        else:
            method = self.pnp_methods.get(method_name, cv2.SOLVEPNP_ITERATIVE)
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self.objp, imgp, self.K, self.dist, flags=method
                )
            except:
                return None, None, float('inf')
            
            if not success:
                return None, None, float('inf')
        
        # LM优化
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                self.objp, imgp, self.K, self.dist, rvec, tvec
            )
        except:
            pass
        
        # 计算重投影误差
        reproj_pts, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.dist)
        reproj_error = np.sqrt(np.mean(np.sum((imgp - reproj_pts.reshape(-1, 2))**2, axis=1)))
        
        return rvec, tvec, reproj_error
    
    def get_averaged_pose(self, corners):
        """获取多方法平均位姿"""
        poses = []
        
        for method_name in ['ITERATIVE', 'EPNP', 'IPPE', 'SQPNP']:
            rvec, tvec, err = self.solve_pnp(corners, method_name)
            if rvec is not None and err < 2.0:
                poses.append((rvec.flatten(), tvec.flatten()))
        
        if len(poses) < 2:
            return None, None, float('inf')
        
        # 平均
        tvecs = np.array([p[1] for p in poses])
        rvecs = np.array([p[0] for p in poses])
        
        t_mean = np.mean(tvecs, axis=0)
        r_mean = np.mean(rvecs, axis=0)
        
        # 计算重投影误差
        imgp = corners.reshape(-1, 2).astype(np.float32)
        reproj_pts, _ = cv2.projectPoints(self.objp, r_mean, t_mean, self.K, self.dist)
        reproj_error = np.sqrt(np.mean(np.sum((imgp - reproj_pts.reshape(-1, 2))**2, axis=1)))
        
        return r_mean, t_mean, reproj_error
    
    def compute_stability(self, history):
        """计算位姿历史的稳定性"""
        if len(history) < 5:
            return {'t_std': float('inf'), 'r_std': float('inf')}
        
        tvecs = np.array([h['tvec'] for h in history])
        rvecs = np.array([h['rvec'] for h in history])
        
        t_std = np.std(tvecs, axis=0) * 1000  # mm
        r_std = np.std(rvecs, axis=0) * 180 / np.pi  # deg
        
        return {
            't_std': np.linalg.norm(t_std),
            'r_std': np.linalg.norm(r_std),
            't_std_xyz': t_std,
            'r_std_xyz': r_std
        }
    
    def run(self, cam_id=2):
        """运行测试"""
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ 无法打开相机")
            return
        
        print("\n🔬 PnP鲁棒性测试")
        print("="*70)
        print("按键: 'q' - 退出, 'c' - 清空历史, 'r' - 显示报告")
        print("="*70)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            corners = self.detect_corners(frame)
            
            if corners is not None:
                cv2.drawChessboardCorners(display, self.board_size, corners, True)
                
                results = {}
                
                # 测试各种方法
                for method_name in ['ITERATIVE', 'RANSAC', 'EPNP', 'IPPE', 'SQPNP']:
                    rvec, tvec, err = self.solve_pnp(corners, method_name)
                    if rvec is not None:
                        results[method_name] = {
                            'rvec': rvec.flatten(),
                            'tvec': tvec.flatten(),
                            'reproj_error': err
                        }
                        self.pose_history[method_name].append({
                            'rvec': rvec.flatten(),
                            'tvec': tvec.flatten()
                        })
                
                # 多方法平均
                r_avg, t_avg, err_avg = self.get_averaged_pose(corners)
                if r_avg is not None:
                    results['AVERAGED'] = {
                        'rvec': r_avg,
                        'tvec': t_avg,
                        'reproj_error': err_avg
                    }
                    self.pose_history['AVERAGED'].append({
                        'rvec': r_avg,
                        'tvec': t_avg
                    })
                
                # 显示结果
                y_offset = 30
                for method_name, res in results.items():
                    dist = np.linalg.norm(res['tvec']) * 1000
                    stability = self.compute_stability(self.pose_history[method_name])
                    
                    # 根据稳定性选择颜色
                    if stability['t_std'] < 2.0:
                        color = (0, 255, 0)  # 绿色 - 稳定
                    elif stability['t_std'] < 5.0:
                        color = (0, 255, 255)  # 黄色 - 一般
                    else:
                        color = (0, 0, 255)  # 红色 - 不稳定
                    
                    text = f"{method_name}: {dist:.1f}mm (std:{stability['t_std']:.1f}mm, err:{res['reproj_error']:.2f}px)"
                    cv2.putText(display, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 22
                
                # 在右上角显示推荐方法
                best_method = min(results.keys(), 
                                 key=lambda m: self.compute_stability(self.pose_history[m])['t_std'])
                cv2.putText(display, f"Best: {best_method}", 
                           (display.shape[1]-200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            else:
                cv2.putText(display, "Chessboard NOT FOUND", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('PnP Robustness Test', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # 清空历史
                for k in self.pose_history:
                    self.pose_history[k].clear()
                print("✅ 已清空位姿历史")
            elif key == ord('r'):
                # 打印报告
                self.print_report()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 最终报告
        self.print_report()
    
    def print_report(self):
        """打印稳定性报告"""
        print("\n" + "="*70)
        print("📊 PnP稳定性报告")
        print("="*70)
        print(f"{'方法':<12} {'样本数':<8} {'平移std(mm)':<14} {'旋转std(deg)':<14} {'评价':<10}")
        print("-"*70)
        
        for method_name, history in self.pose_history.items():
            if len(history) < 3:
                continue
            
            stability = self.compute_stability(history)
            
            if stability['t_std'] < 2.0:
                rating = "⭐⭐⭐ 优秀"
            elif stability['t_std'] < 5.0:
                rating = "⭐⭐ 良好"
            elif stability['t_std'] < 10.0:
                rating = "⭐ 一般"
            else:
                rating = "❌ 差"
            
            print(f"{method_name:<12} {len(history):<8} {stability['t_std']:<14.2f} {stability['r_std']:<14.2f} {rating}")
        
        print("="*70)
        print("\n💡 建议:")
        
        # 找出最稳定的方法
        valid_methods = [(m, self.compute_stability(h)['t_std']) 
                        for m, h in self.pose_history.items() 
                        if len(h) >= 3 and self.compute_stability(h)['t_std'] < float('inf')]
        
        if valid_methods:
            best = min(valid_methods, key=lambda x: x[1])
            print(f"   推荐使用 '{best[0]}' 方法 (平移标准差: {best[1]:.2f}mm)")
        
        if any(self.compute_stability(h)['t_std'] > 5.0 for h in self.pose_history.values() if len(h) >= 3):
            print("   ⚠️  检测到不稳定，建议:")
            print("      - 确保棋盘格在图像中央且大小适中")
            print("      - 检查光照是否均匀")
            print("      - 确保棋盘格完全平整")
            print("      - 避免相机运动模糊")


def main():
    parser = argparse.ArgumentParser(description='PnP鲁棒性测试')
    parser.add_argument('--camera', type=int, default=2, help='相机ID')
    parser.add_argument('--square-size', type=float, default=20.73, help='棋盘格方格边长(mm)')
    parser.add_argument('--intrinsics', type=str, default=None, help='相机内参文件路径')
    parser.add_argument('--board-cols', type=int, default=11, help='棋盘格内角点列数')
    parser.add_argument('--board-rows', type=int, default=8, help='棋盘格内角点行数')
    
    args = parser.parse_args()
    
    # 默认使用当前目录的内参文件
    intrinsics_file = args.intrinsics
    if intrinsics_file is None:
        import os
        default_path = os.path.join(os.path.dirname(__file__), 'config_data/camera_intrinsics_right.yaml')
        if os.path.exists(default_path):
            intrinsics_file = default_path
    
    tester = RobustPnPTester(
        board_size=(args.board_cols, args.board_rows),
        square_size=args.square_size / 1000.0,  # 转为米
        intrinsics_file=intrinsics_file
    )
    
    tester.run(args.camera)


if __name__ == '__main__':
    main()
