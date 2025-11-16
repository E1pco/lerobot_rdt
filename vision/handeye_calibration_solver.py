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
import cv2
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
    
    # 诊断信息：打印前几个数据点的变换信息
    print("\n📊 前3个数据点的诊断信息:")
    for i in range(min(3, len(T_target_cam))):
        print(f"\n   数据点 {i}:")
        print(f"     T_target^cam 位置: {T_target_cam[i][:3, 3]}")
        print(f"     T_gripper^base 位置: {T_gripper_base[i][:3, 3]}")

    T_target_base_all = []
    for i, (T_tc, T_gb) in enumerate(zip(T_target_cam, T_gripper_base)):
        # Eye-in-Hand 验证关系:
        # solvePnP 返回的变换是: p_cam = R @ p_obj + t
        # 所以 T_tc 是从目标到相机的变换: T_target^cam
        # 
        # 验证链式: T_target^base = T_gripper^base @ T_cam^gripper @ T_target^cam
        T_tb = T_gb @ T_cam_gripper @ T_tc
        T_target_base_all.append(T_tb)
        
        if i < 3:
            print(f"     计算的 T_target^base 位置: {T_tb[:3, 3]}")

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
    # 保存图表而不显示（避免 GUI 阻塞）
    plt.savefig('handeye_evaluation.png', dpi=100)
    print(f"✅ 评估图表已保存到 handeye_evaluation.png")
    plt.close()


# ============================================================
# 主流程
# ============================================================

def main(data_dir="dataset_eyeinhand", save_file="handeye_result.npy"):
    T_target_cam, T_gripper_base = load_poses(data_dir)

    if not validate_data(T_target_cam, T_gripper_base):
        print("\n❌ 数据验证失败，中止。")
        return

    # 【使用 OpenCV 自带的手眼标定 - 参考 compute_in_hand.py】
    print("\n🔧 使用 OpenCV cv2.calibrateHandEye 进行手眼标定...")
    
    # 准备数据：从变换矩阵中提取旋转和平移
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    for T_gb, T_tc in zip(T_gripper_base, T_target_cam):
        # 末端相对于基座的旋转和平移
        R_gripper2base.append(T_gb[:3, :3])
        t_gripper2base.append(T_gb[:3, 3].reshape(3, 1))
        
        # 目标相对于相机的旋转和平移（从 solvePnP 得到）
        R_target2cam.append(T_tc[:3, :3])
        t_target2cam.append(T_tc[:3, 3].reshape(3, 1))
    
    print(f"   输入数据: {len(R_gripper2base)} 组位姿对")
    
    # 调用 OpenCV 手眼标定（Tsai 方法）
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    T_cam_gripper = np.eye(4)
    T_cam_gripper[:3, :3] = R_cam2gripper
    T_cam_gripper[:3, 3] = t_cam2gripper.squeeze()
    
    print("\n" + "="*50)
    print("✅ OpenCV 手眼标定结果 (T_cam^gripper)")
    print("="*50)
    print(T_cam_gripper)
    print("\n旋转矩阵 det:", np.linalg.det(R_cam2gripper))
    
    # 转换为四元数（便于理解姿态）
    from scipy.spatial.transform import Rotation as Rot
    quat = Rot.from_matrix(R_cam2gripper).as_quat()
    print(f"四元数 [x, y, z, w]: {quat}")
    print(f"平移向量 [x, y, z]: {t_cam2gripper.squeeze()}")
    print("="*50)

    # 构造 A, B
    A_list, B_list = make_AB(T_target_cam, T_gripper_base)

    # Tsai–Lenz 求解（自己实现）
    R_X = solve_rotation(A_list, B_list)
    t_X = solve_translation(A_list, B_list, R_X)

    T_cam_gripper = np.eye(4)
    T_cam_gripper[:3, :3] = R_X
    T_cam_gripper[:3, 3] = t_X

    print("\n" + "="*50)
    print("✅ Tsai-Lenz 手眼标定结果")
    print("="*50)
    np.set_printoptions(precision=6, suppress=True)
    print("T_cam^gripper =\n", T_cam_gripper)
    print("="*50)

    np.save(save_file, T_cam_gripper)
    print(f"✅ 已保存结果到 {save_file}")

    # ============================================================
    # 对比两种方法的精度
    # ============================================================
    print("\n" + "="*70)
    print("📊 两种方法精度对比")
    print("="*70)
    
    # 重新构造 OpenCV 结果的 T_cam_gripper
    T_cam_gripper_opencv = np.eye(4)
    T_cam_gripper_opencv[:3, :3] = R_cam2gripper
    T_cam_gripper_opencv[:3, 3] = t_cam2gripper.squeeze()
    
    print("\n【方法 1】OpenCV cv2.calibrateHandEye (Tsai 方法)")
    print("-" * 70)
    evaluate_handeye(T_target_cam, T_gripper_base, T_cam_gripper_opencv)
    
    print("\n【方法 2】自实现 Tsai-Lenz 方法")
    print("-" * 70)
    evaluate_handeye(T_target_cam, T_gripper_base, T_cam_gripper)
    
    # ============================================================
    # 两个结果的差异分析
    # ============================================================
    print("\n" + "="*70)
    print("📊 两个结果的差异分析")
    print("="*70)
    
    T_diff = np.linalg.inv(T_cam_gripper_opencv) @ T_cam_gripper
    R_diff = T_diff[:3, :3]
    t_diff = T_diff[:3, 3]
    
    # 旋转差异
    rot_diff = R.from_matrix(R_diff)
    angle_diff = np.degrees(rot_diff.magnitude())
    
    print(f"\n旋转矩阵差异:")
    print(f"   旋转角度差: {angle_diff:.2f}°")
    print(f"   旋转向量: {rot_diff.as_rotvec()}")
    
    print(f"\n平移向量差异:")
    print(f"   平移差 (mm): {t_diff * 1000}")
    print(f"   平移差范数 (mm): {np.linalg.norm(t_diff) * 1000:.2f}")
    
    # 诊断：检查数据是否有问题
    print("\n" + "="*70)
    print("🔍 数据质量诊断")
    print("="*70)
    
    # 检查每对数据的一致性
    print("\n检查每对数据的变换一致性:")
    print("(验证 Eye-in-Hand 关系是否满足)")
    
    consistency_errors = []
    for i in range(len(T_target_cam)):
        # 使用 OpenCV 结果进行验证
        T_target_base_i = T_gripper_base[i] @ T_cam_gripper_opencv @ T_target_cam[i]
        consistency_errors.append(T_target_base_i[:3, 3])
    
    consistency_errors = np.array(consistency_errors)
    pos_consistency = np.linalg.norm(consistency_errors - np.mean(consistency_errors, axis=0), axis=1)
    
    print(f"   计算的 T_target^base 位置一致性 (mm):")
    print(f"     均值偏差: {np.mean(pos_consistency)*1000:.2f} mm")
    print(f"     标准差: {np.std(pos_consistency)*1000:.2f} mm")
    print(f"     最大偏差: {np.max(pos_consistency)*1000:.2f} mm")
    
    if np.mean(pos_consistency) > 0.1:
        print("   ⚠️  警告: T_target 在基座坐标系中变化过大，说明:")
        print("      - 标定板位置在采集中变化（应该固定）")
        print("      - 或者手眼标定数据有较大噪声")
        print("      - 或者 solvePnP 精度不足")


# ============================================================
# 高级诊断工具
# ============================================================

def diagnose_data_quality(T_target_cam, T_gripper_base, T_cam_gripper):
    """深度诊断手眼标定数据质量"""
    print("\n" + "="*70)
    print("🔬 深度数据质量诊断")
    print("="*70)
    
    # 1. 检查标定板是否真的是固定的
    print("\n【诊断 1】标定板固定性检查")
    print("-" * 70)
    
    positions_base = []
    for i, (T_tc, T_gb) in enumerate(zip(T_target_cam, T_gripper_base)):
        T_target_base = T_gb @ T_cam_gripper @ T_tc
        positions_base.append(T_target_base[:3, 3])
    
    positions_base = np.array(positions_base)
    pos_std = np.std(positions_base, axis=0)
    pos_range = np.max(positions_base, axis=0) - np.min(positions_base, axis=0)
    
    print(f"T_target 在基座坐标系中的标准差 (mm): {pos_std * 1000}")
    print(f"T_target 在基座坐标系中的范围 (mm): {pos_range * 1000}")
    
    if np.any(pos_std > 0.1):
        print("❌ 严重问题: 标定板位置在采集中发生了显著变化")
        print("   可能原因:")
        print("   • 标定板在机械臂运动过程中移动")
        print("   • 标定板安装不够牢固")
        print("   • solvePnP 给出的位姿有较大误差")
    else:
        print("✅ 标定板位置稳定")
    
    # 2. 检查末端执行器的运动是否充分
    print("\n【诊断 2】末端执行器运动充分性检查")
    print("-" * 70)
    
    positions_gripper = np.array([T[:3, 3] for T in T_gripper_base])
    pos_gripper_std = np.std(positions_gripper, axis=0)
    pos_gripper_range = np.max(positions_gripper, axis=0) - np.min(positions_gripper, axis=0)
    
    print(f"末端位置标准差 (mm): {pos_gripper_std * 1000}")
    print(f"末端位置范围 (mm): {pos_gripper_range * 1000}")
    
    if np.any(pos_gripper_range < 0.05):
        print("⚠️  警告: 某个方向的运动范围太小 (< 50mm)")
        print("   可能导致手眼标定矩阵奇异")
    else:
        print("✅ 末端运动充分")
    
    # 3. 比较 solvePnP 结果和运动学的一致性
    print("\n【诊断 3】solvePnP 与运动学一致性检查")
    print("-" * 70)
    
    print("分析每对连续采集之间的相对运动...")
    
    for i in range(min(3, len(T_target_cam) - 1)):
        # 从 solvePnP 得到的相对运动
        T_rel_cam = np.linalg.inv(T_target_cam[i+1]) @ T_target_cam[i]
        
        # 从末端位姿得到的相对运动
        T_rel_gripper = np.linalg.inv(T_gripper_base[i+1]) @ T_gripper_base[i]
        
        # 如果手眼标定正确，这两个相对运动应该在手眼变换下相同
        # T_rel_gripper = inv(T_cg) @ T_rel_cam @ T_cg
        
        rel_cam_pos = T_rel_cam[:3, 3]
        rel_gripper_pos = T_rel_gripper[:3, 3]
        
        print(f"\n   数据对 [{i}, {i+1}]:")
        print(f"     相机观测的相对运动 (mm): {rel_cam_pos * 1000}")
        print(f"     末端位姿的相对运动 (mm): {rel_gripper_pos * 1000}")
        
        # 旋转部分
        R_rel_cam = R.from_matrix(T_rel_cam[:3, :3])
        R_rel_gripper = R.from_matrix(T_rel_gripper[:3, :3])
        
        angle_cam = np.degrees(R_rel_cam.magnitude())
        angle_gripper = np.degrees(R_rel_gripper.magnitude())
        
        print(f"     相机观测的旋转 (deg): {angle_cam:.2f}°")
        print(f"     末端位姿的旋转 (deg): {angle_gripper:.2f}°")
    
    # 4. 评估手眼标定矩阵的条件数
    print("\n【诊断 4】手眼标定系统的条件数")
    print("-" * 70)
    
    A_list, B_list = make_AB(T_target_cam, T_gripper_base)
    
    # 旋转部分条件数
    P, Q = [], []
    for Ra, Rb in zip([A[:3, :3] for A in A_list], [B[:3, :3] for B in B_list]):
        axis_a, angle_a = rot_to_axis_angle(Ra)
        axis_b, angle_b = rot_to_axis_angle(Rb)
        weight = max(np.sin(angle_a / 2), 1e-3)
        P.append(axis_a * weight)
        Q.append(axis_b * weight)
    
    P, Q = np.array(P).T, np.array(Q).T
    H = P @ Q.T
    
    _, singular_vals, _ = np.linalg.svd(H)
    cond_rot = singular_vals[0] / singular_vals[-1]
    
    print(f"旋转求解的奇异值: {singular_vals}")
    print(f"旋转求解的条件数: {cond_rot:.2e}")
    
    if cond_rot > 100:
        print("⚠️  警告: 旋转求解条件数过大，说明旋转变化不够多样化")
    
    # 平移部分条件数
    M_list = []
    for A_i in A_list:
        R_A = A_i[:3, :3]
        M_list.append(R_A - np.eye(3))
    
    M = np.vstack(M_list)
    cond_trans = np.linalg.cond(M)
    
    print(f"平移求解的条件数: {cond_trans:.2e}")
    
    if cond_trans > 1e4:
        print("⚠️  警告: 平移求解条件数过大，矩阵病态")


if __name__ == "__main__":
    T_target_cam, T_gripper_base = load_poses("dataset_eyeinhand")
    if validate_data(T_target_cam, T_gripper_base):
        main()
        
        # 加载结果进行深度诊断
        T_cam_gripper = np.load("handeye_result.npy")
        diagnose_data_quality(T_target_cam, T_gripper_base, T_cam_gripper)
