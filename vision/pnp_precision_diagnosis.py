#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PnP 精度诊断与测试工具
=======================
针对手眼标定中PnP精度问题的详细诊断工具

功能:
1. 多种PnP方法对比测试
2. 重投影误差分析
3. 距离误差统计
4. 内参敏感性分析
5. 噪声鲁棒性测试

使用方法:
  python pnp_precision_diagnosis.py
"""

import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import json


class PnPDiagnosticTool:
    """PnP精度诊断工具"""
    
    # 支持的PnP方法
    PNP_METHODS = {
        'ITERATIVE': cv2.SOLVEPNP_ITERATIVE,
        'P3P': cv2.SOLVEPNP_P3P,
        'AP3P': cv2.SOLVEPNP_AP3P,
        'EPNP': cv2.SOLVEPNP_EPNP,
        'IPPE': cv2.SOLVEPNP_IPPE,
        'IPPE_SQUARE': cv2.SOLVEPNP_IPPE_SQUARE,
        'SQPNP': cv2.SOLVEPNP_SQPNP,
    }
    
    def __init__(self, square_size_mm=20.73):
        """
        Parameters
        ----------
        square_size_mm : float
            棋盘格方格大小(毫米)
        """
        self.board_size = (11, 8)
        self.square_size = square_size_mm / 1000.0
        
        # 加载相机参数
        self.K = None
        self.dist = None
        self.load_camera_params()
        
        # 相机
        self.cap = None
        
        # 测试结果
        self.test_results = []
        
        print("="*70)
        print("🔬 PnP 精度诊断工具")
        print("="*70)
        print(f"\n棋盘格: {self.board_size[0]}×{self.board_size[1]}, 方格: {square_size_mm}mm")
        print("\n功能:")
        print("  1. 多种PnP方法对比")
        print("  2. 重投影误差分析")
        print("  3. 距离测量验证")
        print("  4. 畸变影响分析")
        print("="*70)
    
    def load_camera_params(self):
        """加载相机参数"""
        yaml_path = 'camera_intrinsics.yaml'
        
        if os.path.exists(yaml_path):
            try:
                fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
                self.K = fs.getNode('K').mat()
                self.dist = fs.getNode('distCoeffs').mat().flatten()
                fs.release()
                
                print(f"\n✅ 加载相机参数: {yaml_path}")
                print(f"   fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
                print(f"   cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}")
                print(f"   畸变: k1={self.dist[0]:.4f}, k2={self.dist[1]:.4f}")
            except Exception as e:
                print(f"⚠️ 加载失败: {e}")
                self._use_default_params()
        else:
            print(f"⚠️ 未找到 {yaml_path}")
            self._use_default_params()
    
    def _use_default_params(self):
        """使用默认参数"""
        print("   使用默认相机参数 (精度可能受影响!)")
        self.K = np.array([
            [800, 0, 640],
            [0, 800, 360],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist = np.zeros(5, dtype=np.float64)
    
    def get_object_points(self):
        """获取棋盘格3D点"""
        objp = np.zeros((self.board_size[0]*self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def detect_chessboard(self, frame, refine=True):
        """检测棋盘格角点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE + 
                cv2.CALIB_CB_FAST_CHECK)
        
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        
        if found and refine:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return found, corners, gray
    
    def solve_pnp_all_methods(self, objp, imgp, use_ransac=False):
        """使用所有PnP方法求解并对比"""
        results = {}
        
        for name, method in self.PNP_METHODS.items():
            try:
                if use_ransac and method in [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP]:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objp, imgp, self.K, self.dist,
                        iterationsCount=1000,
                        reprojectionError=2.0,
                        flags=method
                    )
                    inlier_ratio = len(inliers) / len(objp) if inliers is not None else 0
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        objp, imgp, self.K, self.dist, flags=method
                    )
                    inlier_ratio = 1.0
                
                if success:
                    # 计算重投影误差
                    reproj_pts, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.dist)
                    reproj_error = np.sqrt(np.mean(np.sum((imgp - reproj_pts.reshape(-1, 2))**2, axis=1)))
                    
                    distance = np.linalg.norm(tvec) * 1000  # mm
                    
                    results[name] = {
                        'success': True,
                        'rvec': rvec,
                        'tvec': tvec,
                        'distance_mm': distance,
                        'reproj_error': reproj_error,
                        'inlier_ratio': inlier_ratio
                    }
                else:
                    results[name] = {'success': False}
                    
            except Exception as e:
                results[name] = {'success': False, 'error': str(e)}
        
        return results
    
    def compute_reprojection_error_detailed(self, objp, imgp, rvec, tvec):
        """详细的重投影误差分析"""
        # 投影
        proj_pts, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.dist)
        proj_pts = proj_pts.reshape(-1, 2)
        
        # 逐点误差
        errors = np.sqrt(np.sum((imgp - proj_pts)**2, axis=1))
        
        # 误差分布在图像不同区域
        h, w = 720, 1280  # 假设分辨率
        regions = {'center': [], 'edge': [], 'corner': []}
        
        for i, pt in enumerate(imgp):
            x, y = pt
            # 判断区域
            dist_to_center = np.sqrt((x - w/2)**2 + (y - h/2)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            
            if dist_to_center < max_dist * 0.3:
                regions['center'].append(errors[i])
            elif dist_to_center < max_dist * 0.7:
                regions['edge'].append(errors[i])
            else:
                regions['corner'].append(errors[i])
        
        return {
            'mean': np.mean(errors),
            'max': np.max(errors),
            'min': np.min(errors),
            'std': np.std(errors),
            'per_point': errors,
            'regions': {
                'center': np.mean(regions['center']) if regions['center'] else 0,
                'edge': np.mean(regions['edge']) if regions['edge'] else 0,
                'corner': np.mean(regions['corner']) if regions['corner'] else 0
            }
        }
    
    def analyze_distortion_effect(self, frame, corners):
        """分析畸变对精度的影响"""
        objp = self.get_object_points()
        imgp = corners.reshape(-1, 2)
        
        # 1. 使用原始图像点 + 畸变系数
        success1, rvec1, tvec1 = cv2.solvePnP(
            objp, imgp, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # 2. 使用去畸变的图像点 + 零畸变
        imgp_undist = cv2.undistortPoints(
            imgp.reshape(-1, 1, 2), self.K, self.dist, P=self.K
        ).reshape(-1, 2)
        
        success2, rvec2, tvec2 = cv2.solvePnP(
            objp, imgp_undist, self.K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success1 and success2:
            dist1 = np.linalg.norm(tvec1) * 1000
            dist2 = np.linalg.norm(tvec2) * 1000
            
            # 计算两种方法的重投影误差
            reproj1, _ = cv2.projectPoints(objp, rvec1, tvec1, self.K, self.dist)
            error1 = np.sqrt(np.mean(np.sum((imgp - reproj1.reshape(-1, 2))**2, axis=1)))
            
            reproj2, _ = cv2.projectPoints(objp, rvec2, tvec2, self.K, np.zeros(5))
            error2 = np.sqrt(np.mean(np.sum((imgp_undist - reproj2.reshape(-1, 2))**2, axis=1)))
            
            return {
                'with_distortion': {
                    'distance_mm': dist1,
                    'reproj_error': error1,
                    'tvec': tvec1.flatten()
                },
                'without_distortion': {
                    'distance_mm': dist2,
                    'reproj_error': error2,
                    'tvec': tvec2.flatten()
                },
                'difference_mm': abs(dist1 - dist2)
            }
        
        return None
    
    def run_single_test(self, frame, corners, ground_truth_mm=None):
        """执行单次测试"""
        objp = self.get_object_points()
        imgp = corners.reshape(-1, 2).astype(np.float32)
        
        print("\n" + "="*70)
        print("📊 PnP 方法对比测试")
        print("="*70)
        
        # 1. 多方法对比
        results = self.solve_pnp_all_methods(objp, imgp)
        
        print("\n方法对比 (距离单位: mm, 误差单位: pixel):")
        print("-"*70)
        print(f"{'方法':<15} {'距离':>10} {'重投影误差':>12} {'状态':>10}")
        print("-"*70)
        
        distances = []
        for name, result in results.items():
            if result['success']:
                dist = result['distance_mm']
                err = result['reproj_error']
                distances.append(dist)
                status = "✅"
                print(f"{name:<15} {dist:>10.2f} {err:>12.4f} {status:>10}")
            else:
                print(f"{name:<15} {'N/A':>10} {'N/A':>12} {'❌':>10}")
        
        if distances:
            print("-"*70)
            print(f"{'距离范围':<15} {min(distances):>10.2f} ~ {max(distances):.2f} mm")
            print(f"{'距离标准差':<15} {np.std(distances):>10.2f} mm")
        
        # 2. 畸变影响分析
        print("\n📐 畸变影响分析:")
        dist_analysis = self.analyze_distortion_effect(frame, corners)
        if dist_analysis:
            print(f"   带畸变补偿: {dist_analysis['with_distortion']['distance_mm']:.2f} mm")
            print(f"   无畸变补偿: {dist_analysis['without_distortion']['distance_mm']:.2f} mm")
            print(f"   差异: {dist_analysis['difference_mm']:.2f} mm")
        
        # 3. 详细误差分析 (使用ITERATIVE方法)
        if 'ITERATIVE' in results and results['ITERATIVE']['success']:
            rvec = results['ITERATIVE']['rvec']
            tvec = results['ITERATIVE']['tvec']
            
            error_detail = self.compute_reprojection_error_detailed(objp, imgp, rvec, tvec)
            
            print("\n📏 重投影误差分布 (像素):")
            print(f"   平均: {error_detail['mean']:.4f}")
            print(f"   标准差: {error_detail['std']:.4f}")
            print(f"   最大: {error_detail['max']:.4f}")
            print(f"   图像中心区域: {error_detail['regions']['center']:.4f}")
            print(f"   图像边缘区域: {error_detail['regions']['edge']:.4f}")
            print(f"   图像角落区域: {error_detail['regions']['corner']:.4f}")
            
            # 检查是否有异常
            if error_detail['regions']['corner'] > error_detail['regions']['center'] * 2:
                print("\n   ⚠️ 警告: 角落区域误差明显偏大，可能是畸变校正不准")
        
        # 4. 与真实值对比
        if ground_truth_mm is not None:
            print(f"\n📏 真实值对比:")
            print(f"   输入真实距离: {ground_truth_mm:.1f} mm")
            if 'ITERATIVE' in results and results['ITERATIVE']['success']:
                measured = results['ITERATIVE']['distance_mm']
                error_mm = measured - ground_truth_mm
                error_pct = abs(error_mm) / ground_truth_mm * 100
                print(f"   测量距离: {measured:.2f} mm")
                print(f"   误差: {error_mm:+.2f} mm ({error_pct:.2f}%)")
                
                if error_pct > 5:
                    print(f"   ⚠️ 误差超过5%，建议检查:")
                    print(f"      1. 棋盘格方格大小是否正确 (当前: {self.square_size*1000:.1f}mm)")
                    print(f"      2. 相机内参是否准确")
                    print(f"      3. 测量距离是否从相机光心算起")
        
        return results
    
    def interactive_test(self):
        """交互式测试"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            print("❌ 无法打开相机")
            return
        
        print("\n🎬 启动交互式测试...")
        print("\n⌨️  快捷键:")
        print("   SPACE  - 执行测试")
        print("   'g'    - 输入真实距离")
        print("   'u'    - 测试去畸变效果")
        print("   'r'    - 使用RANSAC")
        print("   'e'    - 导出结果")
        print("   'q'    - 退出\n")
        
        ground_truth_mm = None
        use_ransac = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            display = frame.copy()
            found, corners, gray = self.detect_chessboard(frame)
            
            if found:
                cv2.drawChessboardCorners(display, self.board_size, corners, found)
                
                # 快速计算距离
                objp = self.get_object_points()
                imgp = corners.reshape(-1, 2)
                success, rvec, tvec = cv2.solvePnP(
                    objp, imgp, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    distance = np.linalg.norm(tvec) * 1000
                    
                    # 绘制坐标轴
                    axis = np.float32([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05], [0, 0, 0]])
                    axis_pts, _ = cv2.projectPoints(axis, rvec, tvec, self.K, self.dist)
                    axis_pts = axis_pts.astype(int)
                    
                    origin = tuple(axis_pts[3].ravel())
                    cv2.line(display, origin, tuple(axis_pts[0].ravel()), (0, 0, 255), 3)
                    cv2.line(display, origin, tuple(axis_pts[1].ravel()), (0, 255, 0), 3)
                    cv2.line(display, origin, tuple(axis_pts[2].ravel()), (255, 0, 0), 3)
                    
                    # 显示距离
                    cv2.putText(display, f"Distance: {distance:.1f} mm", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if ground_truth_mm is not None:
                        error = distance - ground_truth_mm
                        cv2.putText(display, f"Error: {error:+.1f} mm", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                cv2.putText(display, "Press SPACE to test", (10, display.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(display, "Chessboard NOT FOUND", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 显示模式
            mode_text = f"RANSAC: {'ON' if use_ransac else 'OFF'}"
            cv2.putText(display, mode_text, (display.shape[1]-150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('PnP Diagnostic', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' ') and found:
                self.run_single_test(frame, corners, ground_truth_mm)
            
            elif key == ord('g'):
                print("\n请输入真实距离(mm): ", end='')
                try:
                    ground_truth_mm = float(input())
                    print(f"✅ 已设置真实距离: {ground_truth_mm} mm")
                except:
                    print("❌ 输入无效")
            
            elif key == ord('r'):
                use_ransac = not use_ransac
                print(f"\n{'✅' if use_ransac else '❌'} RANSAC模式: {'开启' if use_ransac else '关闭'}")
            
            elif key == ord('u') and found:
                self.test_undistortion_effect(frame, corners)
            
            elif key == ord('e'):
                self.export_results()
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def test_undistortion_effect(self, frame, corners):
        """测试去畸变效果"""
        h, w = frame.shape[:2]
        
        # 计算最优新相机矩阵
        new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
        
        # 去畸变图像
        undist_frame = cv2.undistort(frame, self.K, self.dist, None, new_K)
        
        # 在去畸变图像上重新检测
        found, new_corners, _ = self.detect_chessboard(undist_frame)
        
        print("\n📐 去畸变测试:")
        
        if found:
            objp = self.get_object_points()
            
            # 原图PnP
            imgp1 = corners.reshape(-1, 2)
            success1, rvec1, tvec1 = cv2.solvePnP(
                objp, imgp1, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # 去畸变图PnP (使用新内参，无畸变)
            imgp2 = new_corners.reshape(-1, 2)
            success2, rvec2, tvec2 = cv2.solvePnP(
                objp, imgp2, new_K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success1 and success2:
                dist1 = np.linalg.norm(tvec1) * 1000
                dist2 = np.linalg.norm(tvec2) * 1000
                
                print(f"   原始图像 PnP: {dist1:.2f} mm")
                print(f"   去畸变图像 PnP: {dist2:.2f} mm")
                print(f"   差异: {abs(dist1-dist2):.2f} mm")
                
                # 显示对比
                vis1 = frame.copy()
                vis2 = undist_frame.copy()
                cv2.drawChessboardCorners(vis1, self.board_size, corners, True)
                cv2.drawChessboardCorners(vis2, self.board_size, new_corners, True)
                
                comparison = np.hstack([
                    cv2.resize(vis1, (640, 480)),
                    cv2.resize(vis2, (640, 480))
                ])
                
                cv2.putText(comparison, f"Original: {dist1:.1f}mm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(comparison, f"Undistorted: {dist2:.1f}mm", (650, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Undistortion Comparison', comparison)
        else:
            print("   ⚠️ 去畸变图像上未能检测到棋盘格")
    
    def export_results(self):
        """导出测试结果"""
        if not self.test_results:
            print("⚠️ 没有测试结果可导出")
            return
        
        filename = f"pnp_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"✅ 结果已导出: {filename}")


def generate_diagnosis_report():
    """生成完整的诊断报告"""
    print("\n" + "="*70)
    print("📋 PnP精度问题诊断报告")
    print("="*70)
    
    # 检查相机参数文件
    yaml_path = 'camera_intrinsics.yaml'
    if os.path.exists(yaml_path):
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        K = fs.getNode('K').mat()
        dist = fs.getNode('distCoeffs').mat().flatten()
        fs.release()
        
        print("\n📷 当前相机参数分析:")
        print(f"   内参矩阵 K:")
        print(f"     fx = {K[0,0]:.2f}, fy = {K[1,1]:.2f}")
        print(f"     cx = {K[0,2]:.2f}, cy = {K[1,2]:.2f}")
        print(f"     fx/fy = {K[0,0]/K[1,1]:.4f}")
        
        print(f"\n   畸变系数:")
        print(f"     k1 = {dist[0]:.6f}")
        print(f"     k2 = {dist[1]:.6f}")
        print(f"     p1 = {dist[2]:.6f}")
        print(f"     p2 = {dist[3]:.6f}")
        print(f"     k3 = {dist[4]:.6f}")
        
        # 诊断
        print("\n🔍 诊断结果:")
        
        issues = []
        
        # 检查焦距
        if abs(K[0,0]/K[1,1] - 1.0) > 0.01:
            issues.append("焦距不对称 (fx≠fy)，可能是标定问题或传感器问题")
        
        # 检查畸变
        if abs(dist[0]) > 0.3:
            issues.append(f"径向畸变较大 (k1={dist[0]:.4f})，需要更多标定图像")
        
        # 检查主点
        cx_expected = 640  # 1280/2
        cy_expected = 360  # 720/2
        if abs(K[0,2] - cx_expected) > 50 or abs(K[1,2] - cy_expected) > 50:
            issues.append(f"主点偏离图像中心较远")
        
        if issues:
            print("   ⚠️ 发现以下问题:")
            for i, issue in enumerate(issues, 1):
                print(f"      {i}. {issue}")
        else:
            print("   ✅ 相机参数看起来正常")
        
        print("\n💡 建议:")
        print("   1. 重新标定相机，采集15-25张多角度图像")
        print("   2. 确保标定图像覆盖整个视野")
        print("   3. 确认棋盘格方格大小准确测量")
        print("   4. 使用高质量的标定板")
    else:
        print("\n❌ 未找到相机参数文件!")
        print("   请先运行 calibrate_camera_improved.py 进行标定")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PnP精度诊断工具')
    parser.add_argument('--square-size', type=float, default=20.73, help='棋盘格方格大小(mm)')
    parser.add_argument('--report', action='store_true', help='生成诊断报告')
    
    args = parser.parse_args()
    
    if args.report:
        generate_diagnosis_report()
    else:
        tool = PnPDiagnosticTool(square_size_mm=args.square_size)
        tool.interactive_test()


if __name__ == '__main__':
    main()
