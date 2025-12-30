#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手眼标定公共工具模块
===================
提供 Eye-in-Hand 和 Eye-to-Hand 共用的一致性评估函数。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def evaluate_eye_in_hand_consistency(
    T_cam_gripper: np.ndarray,
    T_gripper_base_list: list,
    T_target_cam_list: list,
) -> dict:
    """
    评估 Eye-in-Hand 标定结果一致性 (AX = XB)

    Parameters
    ----------
    T_cam_gripper : 4x4 相机在末端坐标系下的位姿
    T_gripper_base_list : 各位姿下末端在基座坐标系的变换列表
    T_target_cam_list : 各位姿下标定板在相机坐标系的变换列表

    Returns
    -------
    dict: {
        'trans_errors': list[float],  # mm
        'rot_errors': list[float],    # deg
        'mean_trans': float,
        'max_trans': float,
        'mean_rot': float,
        'max_rot': float,
        'quality': str  # '优秀' / '一般' / '较差'
    }
    """
    if T_cam_gripper is None or len(T_gripper_base_list) < 2:
        return None

    trans_errors = []
    rot_errors = []

    for i in range(len(T_gripper_base_list) - 1):
        T_gb1 = T_gripper_base_list[i]
        T_gb2 = T_gripper_base_list[i + 1]
        T_tc1 = T_target_cam_list[i]
        T_tc2 = T_target_cam_list[i + 1]

        # A = T_gb2 @ inv(T_gb1)
        A = T_gb2 @ np.linalg.inv(T_gb1)
        # B = inv(T_tc2) @ T_tc1
        B = np.linalg.inv(T_tc2) @ T_tc1

        AX = A @ T_cam_gripper
        XB = T_cam_gripper @ B

        error_T = AX @ np.linalg.inv(XB)
        trans_errors.append(np.linalg.norm(error_T[:3, 3]) * 1000)
        rot_errors.append(
            np.linalg.norm(R.from_matrix(error_T[:3, :3]).as_rotvec()) * 180 / np.pi
        )

    mean_trans = float(np.mean(trans_errors))
    max_trans = float(np.max(trans_errors))
    mean_rot = float(np.mean(rot_errors))
    max_rot = float(np.max(rot_errors))

    if mean_trans < 5 and mean_rot < 2:
        quality = "优秀"
    elif mean_trans < 10 and mean_rot < 5:
        quality = "一般"
    else:
        quality = "较差"

    return {
        "trans_errors": trans_errors,
        "rot_errors": rot_errors,
        "mean_trans": mean_trans,
        "max_trans": max_trans,
        "mean_rot": mean_rot,
        "max_rot": max_rot,
        "quality": quality,
    }


def evaluate_eye_to_hand_consistency(
    T_cam_base: np.ndarray,
    T_gripper_base_list: list,
    T_target_cam_list: list,
) -> dict:
    """
    评估 Eye-to-Hand 标定结果一致性 (标定板相对于末端应恒定)

    对于 Eye-to-Hand，标定板固定在末端，因此:
        T_target_gripper = inv(T_gripper_base) @ T_cam_base @ T_target_cam
    各位姿下 T_target_gripper 应该一致。

    Parameters
    ----------
    T_cam_base : 4x4 相机在基座坐标系下的位姿
    T_gripper_base_list : 各位姿下末端在基座坐标系的变换列表
    T_target_cam_list : 各位姿下标定板在相机坐标系的变换列表

    Returns
    -------
    dict: 同 evaluate_eye_in_hand_consistency
    """
    if T_cam_base is None or len(T_gripper_base_list) < 2:
        return None

    n = min(len(T_gripper_base_list), len(T_target_cam_list))
    T_target_grippers = []
    for i in range(n):
        T_gb = T_gripper_base_list[i]
        T_tc = T_target_cam_list[i]
        T_tg = np.linalg.inv(T_gb) @ T_cam_base @ T_tc
        T_target_grippers.append(T_tg)

    trans_errors = []
    rot_errors = []
    for i in range(len(T_target_grippers)):
        for j in range(i + 1, len(T_target_grippers)):
            T_diff = T_target_grippers[i] @ np.linalg.inv(T_target_grippers[j])
            trans_errors.append(np.linalg.norm(T_diff[:3, 3]) * 1000)
            rot_errors.append(
                np.linalg.norm(R.from_matrix(T_diff[:3, :3]).as_rotvec()) * 180 / np.pi
            )

    if not trans_errors:
        return None

    mean_trans = float(np.mean(trans_errors))
    max_trans = float(np.max(trans_errors))
    mean_rot = float(np.mean(rot_errors))
    max_rot = float(np.max(rot_errors))

    if mean_trans < 10 and mean_rot < 2:
        quality = "优秀"
    elif mean_trans < 20 and mean_rot < 5:
        quality = "良好"
    else:
        quality = "一般/较差"

    return {
        "trans_errors": trans_errors,
        "rot_errors": rot_errors,
        "mean_trans": mean_trans,
        "max_trans": max_trans,
        "mean_rot": mean_rot,
        "max_rot": max_rot,
        "quality": quality,
    }


def print_consistency_report(result: dict, title: str = "一致性误差"):
    """打印一致性评估报告"""
    if result is None:
        print("   没有足够的数据进行评估")
        return

    print(f"\n{title}:")
    print(f"   平移误差: 平均={result['mean_trans']:.2f}mm, 最大={result['max_trans']:.2f}mm")
    print(f"   旋转误差: 平均={result['mean_rot']:.2f}°, 最大={result['max_rot']:.2f}°")

    q = result["quality"]
    if q == "优秀":
        print(f"\n   ✅ 标定质量: {q}")
    elif q in ("一般", "良好"):
        print(f"\n   ⚠️  标定质量: {q}")
    else:
        print(f"\n   ❌ 标定质量: {q}，建议重新采集数据")
