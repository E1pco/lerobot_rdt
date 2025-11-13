#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手眼标定 (Eye-in-Hand) — Tsai-Lenz 实现
--------------------------------------
输入:
    dataset_eyeinhand/pose_*.npz
    每个文件包含:
        T_target_cam: 4x4
        T_gripper_base: 4x4

输出:
    handeye_result.npy
"""

import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


# ============================================================
# 数据载入与验证
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

    if np.any(span_cam < 0.001) or np.any(span_gripper < 0.001):
        print("   ⚠️  警告：某个方向的运动范围过小，标定精度可能较低")

    # 检查旋转多样性
    rotations_cam = [R.from_matrix(T[:3, :3]) for T in T_target_cam]
    rotvecs_cam = np.array([r.as_rotvec() for r in rotations_cam])
    rot_span_cam = np.linalg.norm(np.max(rotvecs_cam, axis=0) - np.min(rotvecs_cam, axis=0))

    print(f"\n   相机旋转多样性: {np.degrees(rot_span_cam):.2f} deg")

    if rot_span_cam < np.radians(5):
        print("   ⚠️  警告：旋转变化不足 5°，标定可能退化")

    return True


# ============================================================
# 相对运动构造 (A_i, B_i)
# ============================================================

def make_AB(T_target_cam, T_gripper_base):
    """构造 Tsai–Lenz 方程所需的相对运动 A_i, B_i"""
    A_list, B_list = [], []
    for i in range(len(T_target_cam) - 1):
        # A_i = (T_tc[i+1])^-1 @ T_tc[i]
        A_list.append(np.linalg.inv(T_target_cam[i + 1]) @ T_target_cam[i])

        # B_i = T_gb[i+1] @ inv(T_gb[i])
        B_list.append(T_gripper_base[i + 1] @ np.linalg.inv(T_gripper_base[i]))
    return A_list, B_list


# ============================================================
# Tsai-Lenz 方法
# ============================================================

def rot_to_axis_angle(Rm):
    rot = R.from_matrix(Rm)
    angle = rot.magnitude()
    axis = rot.as_rotvec() / (angle + 1e-12)
    return axis, angle


def solve_rotation(A_list, B_list):
    """求解旋转矩阵 R_X"""
    P, Q, weights = [], [], []
    for Ra, Rb in zip([A[:3, :3] for A in A_list], [B[:3, :3] for B in B_list]):
        axis_a, angle_a = rot_to_axis_angle(Ra)
        axis_b, angle_b = rot_to_axis_angle(Rb)

        # 小角度减小权重 (Huber 风格)
        weight = max(np.sin(angle_a / 2), 1e-3)

        P.append(axis_a * weight)
        Q.append(axis_b * weight)
        weights.append(weight)

    P, Q = np.array(P).T, np.array(Q).T
    H = P @ Q.T
    U, _, Vt = np.linalg.svd(H)
    det = np.linalg.det(U @ Vt)
    R_X = U @ np.diag([1, 1, det]) @ Vt

    print(f"\n📊 旋转求解统计:")
    print(f"   数据对数: {len(A_list)}")
    print(f"   权重范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    print(f"   SVD 最小奇异值: {np.linalg.svd(H)[1][-1]:.6f}")

    return R_X


def solve_translation(A_list, B_list, R_X):
    """最小二乘求解平移 t_X"""
    M_list, b_list = [], []
    for A_i, B_i in zip(A_list, B_list):
        R_A, t_A = A_i[:3, :3], A_i[:3, 3].reshape(3, 1)
        R_B, t_B = B_i[:3, :3], B_i[:3, 3].reshape(3, 1)
        M_list.append(R_A - np.eye(3))
        b_list.append(R_X @ t_B - t_A)

    M = np.vstack(M_list)
    b = np.vstack(b_list)

    cond = np.linalg.cond(M)
    print(f"\n📊 平移求解统计:")
    print(f"   M 形状: {M.shape}")
    print(f"   条件数: {cond:.2e}")
    if cond > 1e10:
        print("   ⚠️  警告：矩阵病态，精度可能有限")

    t_X, residuals, rank, _ = np.linalg.lstsq(M, b, rcond=None)
    if residuals.size > 0:
        print(f"   残差范数: {np.sqrt(residuals[0]):.6f}")
        print(f"   秩: {rank}/{M.shape[1]}")

    return t_X.squeeze()


# ============================================================
# 精度评估
# ============================================================

def evaluate_handeye(T_target_cam, T_gripper_base, T_cam_gripper):
    """验证结果 T_cam^gripper 的精度"""
    print("\n📐 评估标定结果...")
    print(f"   det(R_X): {np.linalg.det(T_cam_gripper[:3,:3]):.6f}")

    T_target_base_all = []
    for T_tc, T_gb in zip(T_target_cam, T_gripper_base):
        # 正确的链式关系:  T_target^base = T_target^cam * T_cam^gripper * T_gripper^base
        T_tb = T_tc @ T_cam_gripper @ T_gb
        T_target_base_all.append(T_tb)

    T_target_base_all = np.array(T_target_base_all)

    # 平移误差
    positions = np.array([T[:3, 3] for T in T_target_base_all])
    pos_mean = np.mean(positions, axis=0)
    pos_err = np.linalg.norm(positions - pos_mean, axis=1)
    mean_pos_err = np.mean(pos_err)
    std_pos_err = np.std(pos_err)
    max_pos_err = np.max(pos_err)

    # 旋转误差
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
    print(f"平移偏差: 均值={mean_pos_err*1000:.3f} mm, σ={std_pos_err*1000:.3f} mm, 最大={max_pos_err*1000:.3f} mm")
    print(f"旋转偏差: 均值={mean_rot_err:.3f}°, σ={std_rot_err:.3f}°, 最大={max_rot_err:.3f}°")
    print("="*50)

    if mean_pos_err < 0.002 and mean_rot_err < 0.5:
        print("✅ 精度良好")
    elif mean_pos_err < 0.005 and mean_rot_err < 1.0:
        print("⚠️ 精度一般，建议增加采集多样性")
    else:
        print("❌ 精度不足，请检查数据采集或标定输入")

    # 可视化
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', label='Samples')
    ax1.scatter(pos_mean[0], pos_mean[1], pos_mean[2], c='r', marker='*', s=200, label='Mean')
    ax1.set_title("Target Positions in Base Frame")
    ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]"); ax1.set_zlabel("Z [m]")
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.plot(pos_err*1000, 'o-')
    ax2.axhline(mean_pos_err*1000, color='r', linestyle='--', label=f'Mean {mean_pos_err*1000:.2f}mm')
    ax2.set_xlabel("Index"); ax2.set_ylabel("Position Error [mm]"); ax2.grid(); ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.plot(rot_err, 'o-')
    ax3.axhline(mean_rot_err, color='r', linestyle='--', label=f'Mean {mean_rot_err:.2f}°')
    ax3.set_xlabel("Index"); ax3.set_ylabel("Rotation Error [°]"); ax3.grid(); ax3.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 主流程
# ============================================================

def main(data_dir="dataset_eyeinhand", save_file="handeye_result.npy"):
    T_target_cam, T_gripper_base = load_poses(data_dir)

    if not validate_data(T_target_cam, T_gripper_base):
        print("\n❌ 数据验证失败，中止。")
        return

    # 构造 A, B
    A_list, B_list = make_AB(T_target_cam, T_gripper_base)

    # Tsai–Lenz 求解
    R_X = solve_rotation(A_list, B_list)
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
    print(f"✅ 已保存结果到 {save_file}")

    # 结果评估
    evaluate_handeye(T_target_cam, T_gripper_base, T_cam_gripper)


if __name__ == "__main__":
    main()
