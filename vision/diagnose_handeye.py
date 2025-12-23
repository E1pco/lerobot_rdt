import os
import sys
import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
import itertools

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ik.robot import create_so101_5dof_gripper

def load_data(data_dir='./handeye_data'):
    pose_files = sorted(glob.glob(os.path.join(data_dir, "pose_*.npz")))
    if not pose_files:
        print(f"❌ 未找到标定数据: {data_dir}")
        return None, None
    
    T_target_cam_list = []
    T_gripper_base_list = []
    q_list = []
    
    print(f"📂 加载标定数据: {len(pose_files)} 组")
    for f in pose_files:
        data = np.load(f)
        T_target_cam_list.append(data['T_target_cam'])
        T_gripper_base_list.append(data['T_gripper_base'])
        if 'q' in data:
            q_list.append(data['q'])
        
    return T_gripper_base_list, T_target_cam_list, q_list

def check_data_quality(T_gripper_base_list, T_target_cam_list):
    print("\n📊 数据质量检查 (旋转分布)")
    print("-" * 50)
    
    rotations = []
    for T in T_gripper_base_list:
        rotations.append(R.from_matrix(T[:3, :3]))
    
    # 计算相对于第一个位姿的旋转轴和角度
    base_rot = rotations[0]
    angles = []
    axes = []
    
    for i in range(1, len(rotations)):
        rel_rot = rotations[i] * base_rot.inv()
        angle = rel_rot.magnitude() * 180 / np.pi
        axis = rel_rot.as_rotvec() / (rel_rot.magnitude() + 1e-8)
        angles.append(angle)
        axes.append(axis)
        print(f"  Pose {i} vs Pose 0: 角度 = {angle:6.2f}°")
        
    print(f"\n  最大旋转角度: {max(angles):.2f}°")
    print(f"  平均旋转角度: {np.mean(angles):.2f}°")
    
    if max(angles) < 10:
        print("  ⚠️  警告: 旋转角度过小，可能导致标定不准确！建议至少包含 >15° 的旋转。")
    
    # 检查标定板位姿的连续性 (排查 PnP Flip)
    print("\n📊 标定板位姿连续性检查 (T_target_cam)")
    print("-" * 50)
    rotations_tc = [R.from_matrix(T[:3, :3]) for T in T_target_cam_list]
    base_rot_tc = rotations_tc[0]
    
    for i in range(1, len(rotations_tc)):
        rel_rot = rotations_tc[i] * rotations_tc[i-1].inv()
        angle = rel_rot.magnitude() * 180 / np.pi
        print(f"  Frame {i} vs {i-1}: 相对旋转 = {angle:6.2f}°")
        if angle > 90:
             print(f"  ⚠️  警告: Frame {i} 相对上一帧旋转过大 ({angle:.1f}°)，可能是棋盘格检测翻转 (Flip)！")

def evaluate_calibration_correct(T_gripper_base_list, T_target_cam_list, T_cam_gripper):
    print("\n📉 标定一致性评估 (AX = XB)")
    print("-" * 50)
    
    errors_trans = []
    errors_rot = []
    
    for i in range(len(T_gripper_base_list)):
        for j in range(i + 1, len(T_gripper_base_list)):
            T_gb1 = T_gripper_base_list[i]
            T_gb2 = T_gripper_base_list[j]
            T_tc1 = T_target_cam_list[i]
            T_tc2 = T_target_cam_list[j]
            
            # A = T_g2_g1 = inv(T_b_g2) * T_b_g1
            A = np.linalg.inv(T_gb2) @ T_gb1
            
            # B = T_c2_c1 = T_c2_t * T_t_c1 = T_c_t2 * inv(T_c_t1)
            # T_target_cam is T_c_t (Target in Camera)
            B = T_tc2 @ np.linalg.inv(T_tc1)
            
            # Check AX = XB
            # LHS = A * X
            LHS = A @ T_cam_gripper
            # RHS = X * B
            RHS = T_cam_gripper @ B
            
            # Error = LHS * inv(RHS)
            diff = LHS @ np.linalg.inv(RHS)
            
            trans_err = np.linalg.norm(diff[:3, 3]) * 1000
            rot_err = np.linalg.norm(R.from_matrix(diff[:3, :3]).as_rotvec()) * 180 / np.pi
            
            errors_trans.append(trans_err)
            errors_rot.append(rot_err)
            
    print(f"  平均平移误差: {np.mean(errors_trans):.4f} mm")
    print(f"  最大平移误差: {np.max(errors_trans):.4f} mm")
    print(f"  平均旋转误差: {np.mean(errors_rot):.4f} deg")
    print(f"  最大旋转误差: {np.max(errors_rot):.4f} deg")
    
    return np.mean(errors_trans), np.mean(errors_rot)

def run_calibration(T_gripper_base_list, T_target_cam_list):
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai-Lenz"),
        (cv2.CALIB_HAND_EYE_PARK, "Park"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis")
    ]
    
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    for T_gb, T_tc in zip(T_gripper_base_list, T_target_cam_list):
        R_gripper2base.append(T_gb[:3, :3])
        t_gripper2base.append(T_gb[:3, 3].reshape(3, 1))
        R_target2cam.append(T_tc[:3, :3])
        t_target2cam.append(T_tc[:3, 3].reshape(3, 1))
        
    best_error = float('inf')
    best_method = ""
    best_T = None
    
    print("\n🔄 尝试不同标定算法:")
    print("-" * 50)
    
    for method_enum, method_name in methods:
        try:
            R_cg, t_cg = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=method_enum
            )
            
            T_cg = np.eye(4)
            T_cg[:3, :3] = R_cg
            T_cg[:3, 3] = t_cg.squeeze()
            
            print(f"\n🔹 方法: {method_name}")
            t_err, r_err = evaluate_calibration_correct(T_gripper_base_list, T_target_cam_list, T_cg)
            
            score = t_err + r_err # 简单加权
            if score < best_error:
                best_error = score
                best_method = method_name
                best_T = T_cg
                
            print(f"  结果 T_cg:\n{T_cg}")
            
        except Exception as e:
            print(f"  {method_name} 失败: {e}")

    print("\n🏆 最佳方法:", best_method)
    return best_T

def test_inversions(T_gripper_base_list, T_target_cam_list):
    print("\n🔍 测试不同的输入数据组合 (排查坐标系定义问题)")
    print("-" * 50)
    
    combinations = [
        ("原始数据 (T_b_g, T_c_t)", False, False),
        ("机器人位姿取逆 (T_g_b, T_c_t)", True, False),
        ("标定板位姿取逆 (T_b_g, T_t_c)", False, True),
        ("两者都取逆 (T_g_b, T_t_c)", True, True)
    ]
    
    for name, inv_g, inv_t in combinations:
        print(f"\n👉 测试: {name}")
        
        # 准备数据
        T_gb_test = []
        T_tc_test = []
        
        for T_gb, T_tc in zip(T_gripper_base_list, T_target_cam_list):
            if inv_g:
                T_gb_test.append(np.linalg.inv(T_gb))
            else:
                T_gb_test.append(T_gb)
                
            if inv_t:
                T_tc_test.append(np.linalg.inv(T_tc))
            else:
                T_tc_test.append(T_tc)
        
        # 运行标定 (只用 Park 方法快速测试)
        try:
            R_gripper2base = [T[:3, :3] for T in T_gb_test]
            t_gripper2base = [T[:3, 3].reshape(3, 1) for T in T_gb_test]
            R_target2cam = [T[:3, :3] for T in T_tc_test]
            t_target2cam = [T[:3, 3].reshape(3, 1) for T in T_tc_test]
            
            R_cg, t_cg = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_PARK
            )
            
            T_cg = np.eye(4)
            T_cg[:3, :3] = R_cg
            T_cg[:3, 3] = t_cg.squeeze()
            
            evaluate_calibration_correct(T_gb_test, T_tc_test, T_cg)
            
        except Exception as e:
            print(f"  失败: {e}")

def optimize_kinematics(q_list, T_target_cam_list):
    print("\n🔧 尝试优化运动学参数 (Gear Signs)")
    print("-" * 50)
    
    if not q_list:
        print("❌ 没有关节角度数据，无法优化")
        return

    # 获取机器人模型
    robot = create_so101_5dof_gripper()
    joint_names = robot.joint_names
    
    # 生成所有可能的符号组合 (+1, -1)
    signs = [1, -1]
    combinations = list(itertools.product(signs, repeat=len(joint_names)))
    
    best_error = float('inf')
    best_signs = None
    
    print(f"  测试 {len(combinations)} 种符号组合...")
    
    for i, sign_combo in enumerate(combinations):
        # 更新 gear_sign
        current_gear_sign = {name: s for name, s in zip(joint_names, sign_combo)}
        robot.gear_sign = current_gear_sign
        
        # 重新计算 FK
        T_gb_new = []
        for q in q_list:
            # 注意：这里假设 q 已经是弧度，且已经应用了原始的 gear_sign
            # 我们需要反推原始 steps 或者假设 q 是 raw values?
            # read_joint_angles 返回的是 q = sign * (step - home) / scale
            # 如果我们想改变 sign，我们需要知道原始的 (step - home) / scale
            # 假设原始 sign 是正确的? 不，我们怀疑原始 sign 是错的。
            # 但是 q_list 保存的是已经计算好的 q。
            # 如果原始 sign 是 s_old，新 sign 是 s_new。
            # q_new = q_old * (s_new / s_old)
            
            # 获取原始 sign
            # create_so101_5dof_gripper 默认的 sign
            orig_signs = [-1, 1, 1, -1, 1] # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
            
            q_new = q.copy()
            for j, name in enumerate(joint_names):
                s_old = orig_signs[j]
                s_new = sign_combo[j]
                q_new[j] = q[j] * (s_new / s_old)
            
            T_gb_new.append(robot.fkine(q_new))
            
        # 运行标定评估
        try:
            R_gripper2base = [T[:3, :3] for T in T_gb_new]
            t_gripper2base = [T[:3, 3].reshape(3, 1) for T in T_gb_new]
            R_target2cam = [T[:3, :3] for T in T_target_cam_list]
            t_target2cam = [T[:3, 3].reshape(3, 1) for T in T_target_cam_list]
            
            R_cg, t_cg = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_PARK
            )
            
            T_cg = np.eye(4)
            T_cg[:3, :3] = R_cg
            T_cg[:3, 3] = t_cg.squeeze()
            
            # 计算误差
            t_err, r_err = evaluate_calibration_correct(T_gb_new, T_target_cam_list, T_cg)
            score = t_err + r_err
            
            if score < best_error:
                best_error = score
                best_signs = sign_combo
                print(f"  [{i}] 新最佳: Err={score:.2f} (T={t_err:.1f}mm, R={r_err:.1f}°) Signs={sign_combo}")
                
        except Exception:
            pass
            
    print("\n🏆 最佳符号组合:", best_signs)
    print("  原始组合: (-1, 1, 1, -1, 1)")

if __name__ == "__main__":
    T_gb, T_tc, qs = load_data()
    if T_gb:
        check_data_quality(T_gb, T_tc)
        test_inversions(T_gb, T_tc)
        if qs:
            optimize_kinematics(qs, T_tc)
        # run_calibration(T_gb, T_tc)
