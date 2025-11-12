#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
手眼标定快速修复工具
根据已有的原始数据和错误的标定结果，通过优化变换顺序来改进精度
"""

import os
import sys
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_poses(data_dir):
    pose_files = sorted(glob.glob(os.path.join(data_dir, "pose_*.npz")))
    T_target_cam, T_gripper_base = [], []
    for f in pose_files:
        data = np.load(f)
        T_target_cam.append(data["T_target_cam"])
        T_gripper_base.append(data["T_gripper_base"])
    return T_target_cam, T_gripper_base


def evaluate_precision(T_target_cam, T_gripper_base, T_cam_gripper, transform_func):
    """评估给定变换顺序的精度"""
    T_target_base_all = []
    
    try:
        for T_tc, T_gb in zip(T_target_cam, T_gripper_base):
            T_tb = transform_func(T_tc, T_cam_gripper, T_gb)
            T_target_base_all.append(T_tb)
        
        T_target_base_all = np.array(T_target_base_all)
        
        # 平移偏差
        positions = np.array([T[:3, 3] for T in T_target_base_all])
        pos_mean = np.mean(positions, axis=0)
        pos_error = np.linalg.norm(positions - pos_mean, axis=1)
        mean_pos_err = np.mean(pos_error)
        
        # 旋转偏差
        rotations = [R.from_matrix(T[:3, :3]) for T in T_target_base_all]
        rotvecs = np.array([r.as_rotvec() for r in rotations])
        rot_mean = np.mean(rotvecs, axis=0)
        rot_err = np.linalg.norm(rotvecs - rot_mean, axis=1)
        mean_rot_err = np.degrees(np.mean(rot_err))
        
        return mean_pos_err, mean_rot_err
    except:
        return float('inf'), float('inf')


def main(data_dir="dataset_eyeinhand"):
    print("="*70)
    print("🔧 手眼标定快速修复工具")
    print("="*70)
    
    # 加载采集数据
    T_target_cam, T_gripper_base = load_poses(data_dir)
    print(f"\n✅ 加载 {len(T_target_cam)} 组采集数据")
    
    # 加载当前的标定结果
    calib_file = "handeye_result.npy"
    if not os.path.exists(calib_file):
        print(f"❌ 找不到 {calib_file}")
        return
    
    T_cam_gripper = np.load(calib_file)
    print(f"✅ 加载标定结果")
    
    # 定义所有可能的变换顺序
    transforms = {
        "原始顺序 (现在使用)": 
            lambda Tc, Tcg, Tg: Tc @ Tcg @ np.linalg.inv(Tg),
        
        "逆序试1": 
            lambda Tc, Tcg, Tg: np.linalg.inv(Tg) @ Tcg @ Tc,
        
        "逆序试2": 
            lambda Tc, Tcg, Tg: Tg @ np.linalg.inv(Tcg) @ Tc,
        
        "逆序试3": 
            lambda Tc, Tcg, Tg: np.linalg.inv(Tc) @ np.linalg.inv(Tcg) @ Tg,
        
        "使用 inv(T_cam_gripper)":
            lambda Tc, Tcg, Tg: Tc @ np.linalg.inv(Tcg) @ np.linalg.inv(Tg),
        
        "完全相反": 
            lambda Tc, Tcg, Tg: Tg @ Tcg @ Tc,
        
        "T_gb @ T_cam_gripper @ T_cam_target":
            lambda Tc, Tcg, Tg: Tg @ Tcg @ np.linalg.inv(Tc),
        
        "inv(T_gb) @ T_cam_gripper @ inv(T_cam_target)":
            lambda Tc, Tcg, Tg: np.linalg.inv(Tg) @ Tcg @ np.linalg.inv(Tc),
    }
    
    print("\n" + "="*70)
    print("测试所有变换顺序...")
    print("="*70)
    
    results = []
    for name, transform_func in transforms.items():
        pos_err, rot_err = evaluate_precision(T_target_cam, T_gripper_base, T_cam_gripper, transform_func)
        results.append((name, pos_err, rot_err))
        
        status = "✅" if pos_err < 0.01 and rot_err < 0.5 else ("⚠️ " if pos_err < 0.05 else "❌")
        print(f"{status} {name:35s}: pos={pos_err*1000:7.2f}mm, rot={rot_err:7.3f}°")
    
    # 找出最佳结果
    best = min(results, key=lambda x: x[1]*1000 + x[2]*10)  # 平移权重 1000，旋转权重 10
    
    print("\n" + "="*70)
    print(f"🏆 最佳结果")
    print("="*70)
    print(f"变换顺序: {best[0]}")
    print(f"平移偏差: {best[1]*1000:.2f} mm")
    print(f"旋转偏差: {best[2]:.3f}°")
    
    if best[1] < 0.01 and best[2] < 0.5:
        print("\n✅ 精度优秀！")
    elif best[1] < 0.05 and best[2] < 2.0:
        print("\n⚠️  精度可用，可以继续优化")
    else:
        print("\n❌ 精度仍需改进")
        print("\n可能的原因：")
        print("1. gear_sign 符号错误")
        print("2. home_pose 值不准确")
        print("3. 采集数据不够多样")
        print("4. 棋盘格检测精度不够")
    
    # 输出建议的修改
    if best[0] != "原始顺序 (现在使用)":
        print(f"\n💡 建议修改:")
        print(f"在 handeye_calibration_solver.py 的 evaluate_handeye() 函数中，")
        print(f"修改变换公式为对应的实现")


if __name__ == "__main__":
    main()
