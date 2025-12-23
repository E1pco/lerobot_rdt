#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手眼标定效果测试脚本 - 视觉跟随
=====================================
功能:
  1. 加载手眼标定结果
  2. 识别棋盘格
  3. 控制机械臂末端跟随棋盘格 (保持固定距离和姿态)

使用方法:
  python vision/test_handeye_tracking.py
"""

import sys
import os
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.handeye_calibration_eyeinhand import HandEyeCalibrator
from ik.robot import create_so101_5dof_gripper

def main():
    # 1. 加载标定结果
    calib_file = os.path.join(os.path.dirname(__file__), 'handeye_result.npy')
    if not os.path.exists(calib_file):
        print(f"❌ 未找到标定文件: {calib_file}")
        return
    
    T_cam_gripper = np.load(calib_file)
    print(f"✅ 已加载手眼标定参数 T_cam_gripper:\n{T_cam_gripper}")
    
    # 2. 初始化 (复用 HandEyeCalibrator 的初始化逻辑)
    # 注意：这里我们不需要保存数据，只是利用它的检测和机器人控制功能
    calibrator = HandEyeCalibrator(output_dir='/tmp')
    if not calibrator.init_robot():
        return

    # 3. 控制循环
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n🤖 机器人视觉跟随测试")
    print("========================================")
    print("按键说明:")
    print("  'f' - 开启/关闭 跟随模式 (Follow)")
    print("  'h' - 机械臂回中 (Home)")
    print("  'q' - 退出")
    print("========================================")
    
    following = False
    target_distance = 0.50  # 目标距离 30cm
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            display = frame.copy()
            
            # 检测棋盘格
            success, T_target_cam, corners, err = calibrator.detect_chessboard(frame, refine_pose=True)
            
            if success:
                # 绘制角点
                cv2.drawChessboardCorners(display, calibrator.board_size, corners, True)
                # 绘制坐标轴
                cv2.drawFrameAxes(display, calibrator.K, calibrator.dist, 
                                 T_target_cam[:3, :3], T_target_cam[:3, 3], 0.05)
                
                # --- 核心逻辑: 计算目标在基座标系下的位姿 ---
                # 1. 获取当前机械臂位姿 T_gripper_base
                T_gripper_base, q_curr = calibrator.read_robot_pose(verbose=False)
                
                # 2. 计算目标在基座标系下的位姿 T_target_base
                # 链式法则: Base -> Gripper -> Camera -> Target
                T_target_base = T_gripper_base @ T_cam_gripper @ T_target_cam
                
                # 显示目标坐标
                pos_target = T_target_base[:3, 3] * 1000
                cv2.putText(display, f"Target (Base): [{pos_target[0]:.0f}, {pos_target[1]:.0f}, {pos_target[2]:.0f}] mm", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if following:
                    # --- 视觉伺服控制 (增量式 PBVS) ---
                    # 在相机坐标系下计算误差，然后转换到基座标系进行移动
                    # 目标: 让棋盘格中心位于相机坐标系的 [0, 0, target_distance]
                    
                    # 1. 计算相机坐标系下的误差
                    # T_target_cam[:3, 3] 是目标在相机坐标系下的当前位置 [x, y, z]
                    # 我们希望它变成 [0, 0, target_distance]
                    target_pos_in_cam = T_target_cam[:3, 3]
                    desired_pos_in_cam = np.array([0, 0, target_distance])
                    
                    # 误差向量 (相机需要移动的方向)
                    # 如果目标在相机右边 (x>0)，相机需要向右移动 (+x) 才能追上
                    # 所以 error = target - desired
                    error_in_cam = target_pos_in_cam - desired_pos_in_cam
                    
                    # 2. 将误差转换到基座标系
                    # T_cam_base = T_gripper_base @ T_cam_gripper
                    T_cam_base = T_gripper_base @ T_cam_gripper
                    R_cam_base = T_cam_base[:3, :3]
                    
                    error_in_base = R_cam_base @ error_in_cam
                    
                    # 3. 计算新的期望相机位置 (增量式)
                    # 使用比例增益 (Gain) 控制速度
                    gain = 0.1  # 降低增益以更安全
                    
                    # 限制单步最大移动量 (例如 2cm)，防止飞车
                    step_limit = 0.02
                    
                    # 计算 Base 系下的位移增量
                    delta_base = gain * error_in_base
                    
                    # 调试打印
                    print(f"Err(Cam): [{error_in_cam[0]*1000:.1f}, {error_in_cam[1]*1000:.1f}, {error_in_cam[2]*1000:.1f}] -> "
                          f"Delta(Base): [{delta_base[0]*1000:.1f}, {delta_base[1]*1000:.1f}, {delta_base[2]*1000:.1f}]")

                    # 限幅
                    norm_delta = np.linalg.norm(delta_base)
                    if norm_delta > step_limit:
                        delta_base = delta_base / norm_delta * step_limit
                    
                    pos_cam_curr = T_cam_base[:3, 3]
                    pos_cam_des = pos_cam_curr + delta_base
                    
                    # --- 调试: 锁定 Base Y 轴 ---
                    # 用户反馈 Y 轴一直漂移，先锁定 Y 轴看 X 和 Z (距离) 是否正常
                    # 如果 X/Z 正常，说明是 Y 轴方向反了或者标定旋转有误
                    # pos_cam_des[1] = pos_cam_curr[1] 
                    # 暂时不完全锁定，而是尝试反转 Y 轴的修正方向 (假设是镜像问题)
                    # 如果之前是 "一直向正方向"，说明是正反馈，我们需要负反馈
                    # Uncomment below to lock Y:
                    pos_cam_des[1] = pos_cam_curr[1]
                    
                    # 4. 保持当前姿态 (暂时不旋转)
                    # 这样可以避免 "LookAt" 造成的旋转发散问题
                    # 如果需要旋转跟随，可以在此基础上增加旋转控制
                    R_cam_des = R_cam_base 
                    
                    # 构建期望的相机位姿矩阵
                    T_cam_base_des = np.eye(4)
                    T_cam_base_des[:3, :3] = R_cam_des
                    T_cam_base_des[:3, 3] = pos_cam_des
                    
                    # 计算期望的末端位姿 T_gripper_base_des
                    # T_gripper_base = T_cam_base * inv(T_cam_gripper)
                    T_gripper_base_des = T_cam_base_des @ np.linalg.inv(T_cam_gripper)
                    
                    print(f"Err(Cam): {error_in_cam*1000} mm -> Err(Base): {error_in_base*1000} mm")
                    print(f"Desired Gripper Pos (Base): {T_gripper_base_des[:3,3]*1000}")
                    # IK 求解
                    # 参考 ik_solver_py.py 的参数配置
                    ik_res = calibrator.robot.ikine_LM(
                        T_gripper_base_des, 
                        q0=q_curr,
                        ilimit=300,  # 增加迭代次数
                        slimit=3,    # 增加搜索次数
                        tol=1e-2,     # 提高精度要求
                        mask=np.array([1, 1, 1, 0.5, 0.5, 0]),
                        k=0.1,        # 阻尼系数
                        method="sugihara" # 使用 sugihara 方法
                    )
                    
                    if not ik_res.success:
                        # 失败尝试: 仅位置 (忽略所有旋转)
                        print(f"⚠️ IK (Pos+Rot) failed: {ik_res.reason}. Trying Pos only...")
                        ik_res = calibrator.robot.ikine_LM(
                            T_gripper_base_des, 
                            q0=q_curr,
                            ilimit=300, 
                            slimit=3,
                            tol=1e-3,
                            mask=np.array([1, 1, 1, 0.8, 0.8, 0]),
                            k=0.1,
                            method="sugihara"
                        )
                    
                    if ik_res.success:
                        q_new = ik_res.q
                        
                        # 安全检查: 防止剧烈运动
                        diff = np.linalg.norm(q_new - q_curr)
                        if diff > 1.5: # 弧度阈值 (放宽一点)
                            cv2.putText(display, f"Movement too large: {diff:.2f}", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # 执行运动
                            # 1. 将弧度转换为舵机步数
                            targets = calibrator.robot.q_to_servo_targets(q_new)
                            # 2. 发送控制指令 (使用较慢的速度以确保安全和平滑)
                            calibrator.controller.fast_move_to_pose(targets, speed=200)
                            
                            cv2.putText(display, "Tracking...", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display, "IK Failed", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"❌ IK Failed completely. Reason: {ik_res.reason}")
            
            else:
                cv2.putText(display, "Target Lost", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 状态显示
            status = "FOLLOW ON" if following else "FOLLOW OFF (Press 'f')"
            color = (0, 255, 0) if following else (0, 255, 255)
            cv2.putText(display, status, (10, display.shape[0]-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Robot Tracking", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                following = not following
                print(f"跟随模式: {'开启' if following else '关闭'}")
            elif key == ord('h'):
                print("回中...")
                calibrator.controller.move_all_home()
                following = False

    except KeyboardInterrupt:
        pass
    finally:
        calibrator.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
