#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


# ============================================================
# 载入数据
# ============================================================

def load_poses(data_dir):
    pose_files = sorted(glob.glob(os.path.join(data_dir, "pose_*.npz")))
    T_target_cam, T_gripper_base = [], []
    for f in pose_files:
        data = np.load(f)
        T_target_cam.append(data["T_target_cam"])
        T_gripper_base.append(data["T_gripper_base"])
    print(f"✅ 加载 {len(T_target_cam)} 组数据。")
    return T_target_cam, T_gripper_base


def relative_motion(T_list):
    A = []
    for i in range(len(T_list) - 1):
        A_i = np.linalg.inv(T_list[i + 1]) @ T_list[i]
        A.append(A_i)
    return A


def validate_data(T_target_cam, T_gripper_base):
    """验证数据的有效性和多样性"""
    print("\n🔍 数据验证:")
    print(f"   采集的数据对数: {len(T_target_cam)}")
    
    if len(T_target_cam) < 3:
        print("   ❌ 错误：至少需要 3 组数据")
        return False
    
    # 检查平移多样性
    positions_cam = np.array([T[:3, 3] for T in T_target_cam])
    positions_gripper = np.array([T[:3, 3] for T in T_gripper_base])
    
    span_cam = np.max(positions_cam, axis=0) - np.min(positions_cam, axis=0)
    span_gripper = np.max(positions_gripper, axis=0) - np.min(positions_gripper, axis=0)
    
    print(f"\n   相机坐标系位置范围: {np.round(span_cam*1000, 2)} mm")
    print(f"   末端坐标系位置范围: {np.round(span_gripper*1000, 2)} mm")
    
    min_span = 0.001  # 最小 1mm 范围
    if np.any(span_cam < min_span) or np.any(span_gripper < min_span):
        print("   ⚠️  警告：某个方向的运动范围过小，标定精度可能较低")
    
    # 检查旋转多样性
    rotations_cam = [R.from_matrix(T[:3, :3]) for T in T_target_cam]
    rotvecs_cam = np.array([r.as_rotvec() for r in rotations_cam])
    rot_span_cam = np.linalg.norm(np.max(rotvecs_cam, axis=0) - np.min(rotvecs_cam, axis=0))
    
    print(f"\n   相机旋转多样性: {np.degrees(rot_span_cam):.2f} deg")
    
    if rot_span_cam < np.radians(5):
        print("   ⚠️  警告：旋转变化不足 5°，标定可能退化")
    
    return True


def rot_to_axis_angle(Rm):
    rot = R.from_matrix(Rm)
    angle = rot.magnitude()
    axis = rot.as_rotvec() / (angle + 1e-12)
    return axis, angle


# ============================================================
# Tsai-Lenz 求解部分
# ============================================================

def solve_rotation(A_list, B_list):
    """使用 Tsai-Lenz 方法求解旋转矩阵 R_X
    
    关键：对旋转角度加权，小角度旋转权重较低以提高稳健性
    """
    P, Q, weights = [], [], []
    for Ra, Rb in zip(A_list, B_list):
        axis_a, angle_a = rot_to_axis_angle(Ra)
        axis_b, angle_b = rot_to_axis_angle(Rb)
        
        # 如果旋转角度太小，降低权重以避免数值不稳定
        weight = max(np.sin(angle_a / 2), 0.1)
        
        P.append(axis_a * weight)
        Q.append(axis_b * weight)
        weights.append(weight)
    
    P, Q = np.array(P).T, np.array(Q).T
    H = P @ Q.T
    U, _, Vt = np.linalg.svd(H)
    
    # 确保正交矩阵行列式为 +1
    det = np.linalg.det(U @ Vt)
    R_X = U @ np.diag([1, 1, det]) @ Vt
    
    print(f"\n📊 旋转求解统计:")
    print(f"   用于求解的数据对数: {len(A_list)}")
    print(f"   使用的权重范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    print(f"   SVD 最小奇异值: {np.linalg.svd(H)[1][-1]:.6f}")
    
    return R_X


def solve_translation(A_list, B_list, R_X):
    """使用最小二乘法求解平移向量 t_X
    
    关键：添加条件数检查和正则化
    """
    M_list, b_list = [], []
    for A_i, B_i in zip(A_list, B_list):
        R_A, t_A = A_i[:3, :3], A_i[:3, 3].reshape(3, 1)
        R_B, t_B = B_i[:3, :3], B_i[:3, 3].reshape(3, 1)
        M_list.append(R_A - np.eye(3))
        b_list.append(R_X @ t_B - t_A)
    
    M = np.vstack(M_list)
    b = np.vstack(b_list)
    
    assert M.shape[0] == b.shape[0], f"维度不匹配: M={M.shape}, b={b.shape}"
    
    # 计算条件数以评估系统的稳定性
    cond = np.linalg.cond(M)
    print(f"\n📊 平移求解统计:")
    print(f"   矩阵 M 的形状: {M.shape}")
    print(f"   矩阵 M 的条件数: {cond:.2e}")
    if cond > 1e10:
        print("   ⚠️  警告：条件数过大，系统可能病态，精度可能有限")
    
    # 使用最小二乘法求解
    t_X, residuals, rank, _ = np.linalg.lstsq(M, b, rcond=None)
    
    if residuals.size > 0:
        residual_norm = np.sqrt(residuals[0])
        print(f"   残差范数: {residual_norm:.6f}")
        print(f"   矩阵秩: {rank}/{M.shape[1]}")
    
    return t_X.squeeze()


# ============================================================
# 精度验证
# ============================================================

def evaluate_handeye(T_target_cam, T_gripper_base, T_cam_gripper):
    """评估标定结果的精度
    
    假设: T_target_base = T_cam_gripper @ T_target_cam @ T_gripper_base^(-1)
    """
    T_target_base_all = []
    
    print("\n📐 评估标定结果...")
    print(f"   T_cam^gripper 的行列式 (应接近 1): {np.linalg.det(T_cam_gripper[:3,:3]):.6f}")
    
    for i, (T_tc, T_gb) in enumerate(zip(T_target_cam, T_gripper_base)):
        # 关键：理解坐标系变换的顺序
        T_tb = T_tc @ T_cam_gripper @ np.linalg.inv(T_gb)
        T_target_base_all.append(T_tb)

    T_target_base_all = np.array(T_target_base_all)

    # 平移分析
    positions = np.array([T[:3, 3] for T in T_target_base_all])
    pos_mean = np.mean(positions, axis=0)
    pos_error = np.linalg.norm(positions - pos_mean, axis=1)
    mean_pos_err = np.mean(pos_error)
    std_pos_err = np.std(pos_error)
    max_pos_err = np.max(pos_error)

    # 旋转分析
    rotations = [R.from_matrix(T[:3, :3]) for T in T_target_base_all]
    rotvecs = np.array([r.as_rotvec() for r in rotations])
    rot_mean = np.mean(rotvecs, axis=0)
    rot_err = np.linalg.norm(rotvecs - rot_mean, axis=1)
    mean_rot_err = np.degrees(np.mean(rot_err))
    std_rot_err = np.degrees(np.std(rot_err))
    max_rot_err = np.degrees(np.max(rot_err))

    print("\n" + "="*50)
    print("📊 标定精度评估")
    print("="*50)
    print(f"平移偏差:")
    print(f"  均值: {mean_pos_err*1000:.3f} mm")
    print(f"  标准差: {std_pos_err*1000:.3f} mm")
    print(f"  最大值: {max_pos_err*1000:.3f} mm")
    print(f"\n旋转偏差:")
    print(f"  均值: {mean_rot_err:.3f} deg")
    print(f"  标准差: {std_rot_err:.3f} deg")
    print(f"  最大值: {max_rot_err:.3f} deg")
    print("="*50)
    
    # 精度评分
    if mean_pos_err < 0.002 and mean_rot_err < 0.5:
        print("✅ 精度良好")
    elif mean_pos_err < 0.005 and mean_rot_err < 1.0:
        print("⚠️  精度一般，可能需要更多数据或更好的采集角度")
    else:
        print("❌ 精度不足，请检查:")
        print("   - 数据采集的多样性是否足够")
        print("   - 棋盘格检测是否准确")
        print("   - 机械臂关节角度读取是否正确")
        print("   - 相机内参是否准确")

    # 可视化平移分布
    fig = plt.figure(figsize=(12, 4))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', label='T_target^base')
    ax1.scatter(pos_mean[0], pos_mean[1], pos_mean[2], c='r', marker='*', s=200, label='Mean')
    ax1.set_title("Target Positions in Base Frame")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.legend()
    
    ax2 = fig.add_subplot(132)
    ax2.plot(pos_error*1000, 'o-')
    ax2.axhline(mean_pos_err*1000, color='r', linestyle='--', label=f'Mean: {mean_pos_err*1000:.2f}mm')
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Position Error [mm]")
    ax2.set_title("Position Error Distribution")
    ax2.legend()
    ax2.grid()
    
    ax3 = fig.add_subplot(133)
    ax3.plot(rot_err, 'o-')
    ax3.axhline(mean_rot_err, color='r', linestyle='--', label=f'Mean: {mean_rot_err:.3f}°')
    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("Rotation Error [deg]")
    ax3.set_title("Rotation Error Distribution")
    ax3.legend()
    ax3.grid()
    
    plt.tight_layout()
    plt.show()


# ============================================================
# 主流程
# ============================================================

def main(data_dir="dataset_eyeinhand", save_file="handeye_result.npy"):
    T_target_cam, T_gripper_base = load_poses(data_dir)
    
    # ✅ 数据验证
    if not validate_data(T_target_cam, T_gripper_base):
        print("\n❌ 数据验证失败，中止")
        return

    A_list = relative_motion(T_target_cam)
    B_list = relative_motion(T_gripper_base)
    A_rot = [A[:3, :3] for A in A_list]
    B_rot = [B[:3, :3] for B in B_list]

    R_X = solve_rotation(A_rot, B_rot)
    t_X = solve_translation(A_list, B_list, R_X)

    T_cam_gripper = np.eye(4)
    T_cam_gripper[:3, :3] = R_X
    T_cam_gripper[:3, 3] = t_X

    print("\n" + "="*50)
    print("✅ 手眼标定结果")
    print("="*50)
    np.set_printoptions(precision=6, suppress=True)
    print("T_cam^gripper =\n", T_cam_gripper)
    print("="*50)

    np.save(save_file, T_cam_gripper)
    print(f"\n✅ 已保存结果到 {save_file}")

    # 标定精度评估
    evaluate_handeye(T_target_cam, T_gripper_base, T_cam_gripper)


if __name__ == "__main__":
    main()
