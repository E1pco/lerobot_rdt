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
    
    # 注意：handeye_calibration_eyeinhand.py 保存的是 cv2.calibrateHandEye 输出的 cam2gripper
    # 其物理含义为: ^G T_C (camera -> gripper)，即“相机坐标系下的点”变换到“末端坐标系”。
    # 本脚本链式计算会用到 ^B T_C = ^B T_G @ ^G T_C，因此这里不应取逆。
    T_cam_gripper = np.load(calib_file)
    print(f"✅ 已加载手眼标定参数 (^G T_C, cam2gripper / camera->gripper):\n{T_cam_gripper}")
    
    # 2. 初始化 (复用 HandEyeCalibrator 的初始化逻辑)
    # 注意：这里我们不需要保存数据，只是利用它的检测和机器人控制功能
    # 使用项目根目录下的 camera_intrinsics.yaml (与 right.py 保持一致)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    intrinsic_file = os.path.join(root_dir, './vision/camera_intrinsics.yaml')
    
    calibrator = HandEyeCalibrator(output_dir='/tmp', intrinsic_file=intrinsic_file)
    
    # 强制重置方格大小为 0.02073 (因为 root yaml 中可能是 0.025)
    target_square_size = 0.02073
    if abs(calibrator.square_size - target_square_size) > 0.0001:
        print(f"⚠️ 强制修正方格大小: {calibrator.square_size*1000:.2f}mm -> {target_square_size*1000:.2f}mm")
        calibrator.square_size = target_square_size
        calibrator.objp = np.zeros((calibrator.board_size[0] * calibrator.board_size[1], 3), np.float32)
        calibrator.objp[:, :2] = np.mgrid[0:calibrator.board_size[0], 0:calibrator.board_size[1]].T.reshape(-1, 2)
        calibrator.objp *= calibrator.square_size
    
    # 应用焦距修正系数
    # 1. 原始修正 (right.py): 600 / 647
    # 2. 现场修正 (2025-12-24): 测量值 250mm -> 实际值 230mm
    correction_factor = 1
    K_original_fx = calibrator.K[0, 0]
    K_original_fy = calibrator.K[1, 1]
    calibrator.K[0, 0] *= correction_factor  # fx
    calibrator.K[1, 1] *= correction_factor  # fy
    print(f"📷 焦距修正 (factor={correction_factor:.4f}):")
    print(f"   原始: fx={K_original_fx:.1f}, fy={K_original_fy:.1f}")
    print(f"   修正后: fx={calibrator.K[0,0]:.1f}, fy={calibrator.K[1,1]:.1f}")

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
    print("  --- 调试控制 ---")
    print("  '1' - 切换 相机X轴 控制 (左右)")
    print("  '2' - 切换 相机Y轴 控制 (上下)")
    print("  '3' - 切换 相机Z轴 控制 (前后)")
    print("  'x' - 反转 相机X轴 方向")
    print("  'y' - 反转 相机Y轴 方向")
    print("  'z' - 反转 相机Z轴 方向")
    print("  'm' - 切换 映射模式 (auto/manual)")
    print("  '7' - 切换 Cam X 映射 (Base X/Y/Z)")
    print("  '8' - 切换 Cam Y 映射 (Base X/Y/Z)")
    print("  '9' - 切换 Cam Z 映射 (Base X/Y/Z)")
    print("========================================")
    
    following = False
    target_distance = 0.30  # 目标距离 30cm

    # 低频诊断打印：用于快速确认 FK/手眼矩阵方向是否被用反
    diag_chain = True
    diag_every_n_frames = 30
    diag_frame_counter = 0

    # 映射模式:
    # - auto: 通过手眼 + 当前机械臂位姿计算 R_base_cam，实现动态映射
    # - base_direct: 直接在 Base 坐标系下控制（推荐，避免 Z 轴漂移）
    # - manual: 使用 axis_map/axis_sign 的静态映射（仅调试用）
    mapping_mode = "base_direct"
    
    # 轴控制掩码 (1:启用, 0:禁用)
    axis_mask = np.array([1.0, 1.0, 1.0]) 
    # 轴方向符号 (1:正向, -1:反向)
    # 分析:
    # Gripper Z ~ Base -Y. Cam Z ~ Gripper Z. => Cam Z ~ Base -Y.
    # Cam Z err > 0 (too far) => Move Cam Z+ => Move Base Y- => Sign -1.
    # Gripper X ~ Base X. Cam X ~ Gripper X. => Cam X ~ Base X. => Sign 1.
    # Gripper Y ~ Base Z. Cam Y ~ Gripper Y. => Cam Y ~ Base Z. => Sign 1.
    axis_sign = np.array([1.0, 1.0, -1.0])
    
    # 轴映射: Cam Axis Index -> Base Axis Index
    # 眼在手上典型配置:
    #   Cam Z (向前) -> Base X (向前)
    #   Cam X (向右) -> Base Y (向右) 
    #   Cam Y (向下) -> Base Z (向下)
    axis_map = np.array([1, 2, 0])  # Cam[0]->Base[1], Cam[1]->Base[2], Cam[2]->Base[0] 

    # 误差死区(单位: mm)。某一轴误差进入死区后，该轴不再继续驱动(避免“到位还在抖/越走越偏”)
    tol_cam_mm = np.array([20.0, 20.0, 10.0])
    # 全部轴都进入死区后，连续满足 N 帧才真正停止发指令(防抖)
    stable_required = 5
    stable_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            display = frame.copy()
            
            # 检测棋盘格
            success, T_target_cam, corners, err = calibrator.detect_chessboard(frame, refine_pose=True)
            
            if success:
                diag_frame_counter += 1
                # 绘制角点
                cv2.drawChessboardCorners(display, calibrator.board_size, corners, True)
                # 绘制坐标轴
                cv2.drawFrameAxes(display, calibrator.K, calibrator.dist, 
                                 T_target_cam[:3, :3], T_target_cam[:3, 3], 0.05)
                
                # --- 核心逻辑: 计算目标在基座标系下的位姿 ---
                # 1. 获取当前机械臂位姿 T_gripper_base
                T_gripper_base, q_curr = calibrator.read_robot_pose(verbose=False)

                if diag_chain and (diag_frame_counter % diag_every_n_frames == 0):
                    # 假设组合：FK为 B_T_G 或 G_T_B；手眼为 G_T_C 或 C_T_G
                    fk_candidates = {
                        "fk=B_T_G": T_gripper_base,
                        "fk=inv": np.linalg.inv(T_gripper_base),
                    }
                    he_candidates = {
                        "he=G_T_C": T_cam_gripper,
                        "he=inv": np.linalg.inv(T_cam_gripper),
                    }
                    parts = []
                    for fk_name, B_T_G in fk_candidates.items():
                        for he_name, G_T_C in he_candidates.items():
                            B_T_C = B_T_G @ G_T_C
                            B_T_T = B_T_C @ T_target_cam
                            p = (B_T_T[:3, 3] * 1000.0)
                            parts.append(f"{fk_name},{he_name}:[{p[0]:.0f},{p[1]:.0f},{p[2]:.0f}]")
                    print("[Diag Target(Base) mm] " + " | ".join(parts))
                
                # 2. 计算目标在基座标系下的位姿 T_target_base
                # 链式法则: ^B T_T = ^B T_G @ ^G T_C @ ^C T_T
                T_target_base = T_gripper_base @ T_cam_gripper @ T_target_cam
                
                # 显示目标坐标
                pos_target = T_target_base[:3, 3] * 1000
                cv2.putText(display, f"Target (Base): [{pos_target[0]:.0f}, {pos_target[1]:.0f}, {pos_target[2]:.0f}] mm", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示PnP结果 (相机坐标系)
                pos_cam = T_target_cam[:3, 3] * 1000
                cv2.putText(display, f"PnP (Cam): [{pos_cam[0]:.1f}, {pos_cam[1]:.1f}, {pos_cam[2]:.1f}] mm", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if following:
                    # --- 视觉伺服控制 ---
                    
                    # 1. 计算相机坐标系下的误差
                    target_pos_in_cam = T_target_cam[:3, 3]
                    desired_pos_in_cam = np.array([0, 0, target_distance])
                    
                    # 误差向量
                    error_in_cam = target_pos_in_cam - desired_pos_in_cam

                    # ---- 误差死区 + 停机判定(相机坐标系) ----
                    error_cam_mm = error_in_cam * 1000.0
                    in_tol_each = np.abs(error_cam_mm) < tol_cam_mm
                    if np.all(in_tol_each):
                        stable_count += 1
                    else:
                        stable_count = 0

                    # 在图像上显示误差与死区命中情况
                    cv2.putText(
                        display,
                        f"ErrCam(mm): [{error_cam_mm[0]:.0f}, {error_cam_mm[1]:.0f}, {error_cam_mm[2]:.0f}] tol:[{tol_cam_mm[0]:.0f},{tol_cam_mm[1]:.0f},{tol_cam_mm[2]:.0f}]",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display,
                        f"InTol(X,Y,Z): {int(in_tol_each[0])},{int(in_tol_each[1])},{int(in_tol_each[2])} stable:{stable_count}/{stable_required}",
                        (10, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    # 如果整体已经稳定到位，就不要继续发任何运动指令
                    if stable_count >= stable_required:
                        cv2.putText(display, "HOLD (in tolerance)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        continue

                    # 对单轴“锁定”：进入死区的轴不再驱动
                    error_in_cam_deadband = error_in_cam.copy()
                    error_in_cam_deadband[in_tol_each] = 0.0
                    
                    # 2. 将相机误差映射到基座(得到末端平移增量)
                    if mapping_mode == "base_direct":
                        # 直接在 Base 坐标系下控制：让夹爪移动到目标位置
                        # 目标：Gripper 移动到 Target 在 Base 下的 XY 位置，Z 保持不变
                        pos_target_base = T_target_base[:3, 3]
                        pos_gripper_base = T_gripper_base[:3, 3]
                        
                        # 误差直接在 Base 坐标系计算
                        error_base_direct = pos_target_base - pos_gripper_base
                        
                        # 只控制 X 和 Y，Z 轴保持稳定（设为 0 或很小的增益）
                        delta_base = np.array([
                            error_base_direct[0] * axis_mask[0],  # X
                            error_base_direct[1] * axis_mask[1],  # Y  
                            error_base_direct[2] * axis_mask[2] * 0.1  # Z 用小增益避免漂移
                        ])
                    elif mapping_mode == "auto":
                        # 动态映射：v_base = R_base_cam @ v_cam
                        # ^B T_C = ^B T_G @ ^G T_C
                        T_base_cam = T_gripper_base @ T_cam_gripper
                        R_base_cam = T_base_cam[:3, :3]
                        control_error_cam = error_in_cam_deadband * axis_mask
                        # 正反馈/负反馈分析：
                        # - err_cam_z > 0 表示"太远"，需要相机往前 (cam Z+)
                        # - R_base_cam 把 cam Z+ 映射到 base 某方向
                        # - 我们希望 gripper 往那个方向移动，所以是正号（同向）
                        delta_base = R_base_cam @ control_error_cam
                    else:
                        # 静态映射（调试）：axis_map + axis_sign
                        control_error_cam = error_in_cam_deadband * axis_mask * axis_sign
                        delta_base = np.zeros(3)
                        delta_base[axis_map[0]] += control_error_cam[0]
                        delta_base[axis_map[1]] += control_error_cam[1]
                        delta_base[axis_map[2]] += control_error_cam[2]
                    
                    # 3. 计算控制量
                    gain = 0.15  # 增益（加大以加速收敛）
                    step_limit = 0.03 # 限幅 (30mm，加大以加速收敛)
                    
                    delta_base = delta_base * gain
                    
                    # 限幅
                    norm_delta = np.linalg.norm(delta_base)
                    if norm_delta > step_limit:
                        delta_base = delta_base / norm_delta * step_limit
                    
                    # --- 计算关键输出 ---
                    pos_target_base_mm = (T_target_base[:3, 3] * 1000.0)
                    pos_gripper_curr = T_gripper_base[:3, 3]
                    pos_gripper_des = pos_gripper_curr + delta_base
                    pos_gripper_des_mm = pos_gripper_des * 1000.0
                    
                    # 计算夹爪与目标在Base下的误差
                    error_base = pos_target_base_mm - pos_gripper_des_mm
                    
                    # 简化日志输出
                    print(f"Target(Base): [{pos_target_base_mm[0]:7.1f}, {pos_target_base_mm[1]:7.1f}, {pos_target_base_mm[2]:7.1f}] mm | " +
                          f"PnP(Cam): [{pos_cam[0]:7.1f}, {pos_cam[1]:7.1f}, {pos_cam[2]:7.1f}] mm | " +
                          f"Gripper→: [{pos_gripper_des_mm[0]:7.1f}, {pos_gripper_des_mm[1]:7.1f}, {pos_gripper_des_mm[2]:7.1f}] mm | " +
                          f"Error: [{error_base[0]:7.1f}, {error_base[1]:7.1f}, {error_base[2]:7.1f}] mm")

                    # 显示映射模式
                    cv2.putText(display, f"Mapping: {mapping_mode}", (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # --- 简化控制: 直接移动末端 ---
                    # 既然 T_cam_base_des = T_cam_base_curr + delta
                    # 且 Camera 和 Gripper 刚性连接
                    # 那么 T_gripper_base_des = T_gripper_base_curr + delta
                    # 这样可以避免 T_cam_gripper 逆矩阵可能引入的误差
                    
                    # 保持末端姿态不变
                    R_gripper_des = T_gripper_base[:3, :3]
                    
                    T_gripper_base_des = np.eye(4)
                    T_gripper_base_des[:3, :3] = R_gripper_des
                    T_gripper_base_des[:3, 3] = pos_gripper_des
                    
                    # 显示目标机械臂位置
                    cv2.putText(display, f"Gripper Target: [{pos_gripper_des[0]*1000:.0f}, {pos_gripper_des[1]*1000:.0f}, {pos_gripper_des[2]*1000:.0f}] mm", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
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
                    
                    if ik_res.success:
                        q_new = ik_res.q
                        
                        # 安全检查: 防止剧烈运动
                        diff = np.linalg.norm(q_new - q_curr)
                        if diff > 1.5: # 弧度阈值 (放宽一点)
                            cv2.putText(display, f"Movement too large: {diff:.2f}", (10, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # 执行运动
                            # 1. 将弧度转换为舵机步数
                            targets = calibrator.robot.q_to_servo_targets(q_new)
                            # 2. 发送控制指令 (使用较慢的速度以确保安全和平滑)
                            calibrator.controller.fast_move_to_pose(targets, speed=200)
                            
                            if mapping_mode == "auto":
                                cv2.putText(display, "Tracking... AUTO", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                map_str = f"X->{axis_map[0]} Y->{axis_map[1]} Z->{axis_map[2]}"
                                cv2.putText(display, f"Tracking... Map:{map_str}", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(display, "IK Failed", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"❌ IK Failed completely. Reason: {ik_res.reason}")
            
            else:
                stable_count = 0
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
                print(f"Follow mode: {following}")
            elif key == ord('m'):
                mapping_mode = "manual" if mapping_mode == "auto" else "auto"
                print(f"Mapping mode: {mapping_mode}")
            elif key == ord('h'):
                print("Returning to home...")
                calibrator.controller.move_all_home()
                calibrator.controller.move_servo("gripper",3050)
                calibrator.controller.move_servo("wrist_roll",850)
                following = False
            # 调试按键
            elif key == ord('1'):
                axis_mask[0] = 1.0 - axis_mask[0]
                print(f"Toggle Cam X axis: {axis_mask[0]}")
            elif key == ord('2'):
                axis_mask[1] = 1.0 - axis_mask[1]
                print(f"Toggle Cam Y axis: {axis_mask[1]}")
            elif key == ord('3'):
                axis_mask[2] = 1.0 - axis_mask[2]
                print(f"Toggle Cam Z axis: {axis_mask[2]}")
            elif key == ord('x'):
                axis_sign[0] *= -1
                print(f"Invert Cam X sign: {axis_sign[0]}")
            elif key == ord('y'):
                axis_sign[1] *= -1
                print(f"Invert Cam Y sign: {axis_sign[1]}")
            elif key == ord('z'):
                axis_sign[2] *= -1
                print(f"Invert Cam Z sign: {axis_sign[2]}")
            elif key == ord('7'):
                axis_map[0] = (axis_map[0] + 1) % 3
                print(f"Cam X maps to Base: {axis_map[0]}")
            elif key == ord('8'):
                axis_map[1] = (axis_map[1] + 1) % 3
                print(f"Cam Y maps to Base: {axis_map[1]}")
            elif key == ord('9'):
                axis_map[2] = (axis_map[2] + 1) % 3
                print(f"Cam Z maps to Base: {axis_map[2]}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

