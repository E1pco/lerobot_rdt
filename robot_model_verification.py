#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人建模测量验证脚本

功能：
1. 测量机械臂在不同关节角度下的实际末端位置
2. 对比正运动学计算结果与实际位置
3. 分析建模误差
4. 提供建模调整建议

使用方法：
python robot_model_verification.py --port /dev/left_arm --camera 0
"""

import numpy as np
import time
import cv2
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from driver.ftservo_controller import ServoController
from ik.robot import create_so101, create_so101_5dof ,create_so101_5dof_gripper


class RobotModelVerifier:
    """机器人建模验证器"""
    
    def __init__(self, port="/dev/left_arm", camera_id=0, use_camera=True):
        self.use_camera = use_camera
        self.controller = ServoController(
            port=port, 
            baudrate=1_000_000, 
            config_path="./driver/servo_config.json"
        )
        
        if use_camera:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"⚠️  无法打开相机 {camera_id}，禁用相机功能")
                self.use_camera = False
        
        # 创建不同的机器人模型进行对比
        self.robot_models = {
            "so101": create_so101(),              # 标准SO-101模型（5关节）
            "so101_5dof": create_so101_5dof()     # 5关节模型
        }
        
        self.test_results = []
        
        # 关节角度转换参数
        self.gear_sign = {
            "shoulder_pan": -1,
            "shoulder_lift": +1,
            "elbow_flex":   +1,
            "wrist_flex":   -1,
            "wrist_roll":   +1,
        }
        
        self.counts_per_rad = 4096 / (2 * np.pi)  # ≈ 651.8986
        
    def read_current_joint_angles(self, joint_names=None):
        """
        读取当前关节角度
        
        Parameters
        ----------
        joint_names : list of str, optional
            关节名称列表，默认使用5关节模型的关节
            
        Returns
        -------
        q : np.ndarray
            当前关节角度 (弧度)
        """
        if joint_names is None:
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        
        # 读取舵机位置
        positions = self.controller.read_servo_positions(joint_names=joint_names, verbose=False)
        
        # 获取home位置
        home_pose = {name: self.controller.get_home_position(name) for name in joint_names}
        
        # 转换为关节角度
        q = np.zeros(len(joint_names))
        for i, name in enumerate(joint_names):
            current = positions[name]
            delta = current - home_pose[name]
            q[i] = self.gear_sign[name] * delta / self.counts_per_rad
        
        return q
        
    def __del__(self):
        if hasattr(self, 'controller'):
            self.controller.close()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def capture_end_effector_position(self, manual=False):
        """
        获取末端执行器的实际位置
        
        Parameters
        ----------
        manual : bool
            是否手动测量（用户输入坐标）
            
        Returns
        -------
        position : np.ndarray or None
            实际末端位置 [x, y, z] (米)
        """
        if manual:
            print("\n📏 手动测量末端位置")
            print("请使用尺子等工具测量末端执行器相对于机器人基座的位置:")
            try:
                x = float(input("X坐标 (米): "))
                y = float(input("Y坐标 (米): "))
                z = float(input("Z坐标 (米): "))
                return np.array([x, y, z])
            except ValueError:
                print("❌ 输入格式错误")
                return None
        
        if self.use_camera:
            print("\n📷 使用相机检测末端位置...")
            # 这里可以集成视觉检测算法
            # 暂时返回None，提示用户手动测量
            print("💡 相机检测功能待实现，请使用手动测量")
            return self.capture_end_effector_position(manual=True)
        
        return self.capture_end_effector_position(manual=True)
    
    def generate_test_poses(self, num_poses=10):
        """
        生成测试关节角度
        
        Parameters
        ----------
        num_poses : int
            测试姿态数量
            
        Returns
        -------
        test_poses : list
            关节角度列表
        """
        # 获取关节限位
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        
        # 安全的关节角度范围（避免极限位置）
        safe_ranges = {
            "shoulder_pan": (-1.5, 1.5),
            "shoulder_lift": (-1.2, 1.2), 
            "elbow_flex": (-1.4, 1.4),
            "wrist_flex": (-1.4, 1.4),
            "wrist_roll": (-2.0, 2.0)
        }
        
        test_poses = []
        
        # 添加零位
        test_poses.append(np.zeros(5))
        
        # 生成随机测试姿态
        for i in range(num_poses - 1):
            pose = []
            for joint in joint_names:
                min_val, max_val = safe_ranges[joint]
                angle = np.random.uniform(min_val, max_val)
                pose.append(angle)
            test_poses.append(np.array(pose))
        
        return test_poses
    
    def move_to_pose(self, joint_angles):
        """
        移动到指定关节角度
        
        Parameters
        ----------
        joint_angles : np.ndarray
            目标关节角度 (弧度)
        """
        print(f"\n🤖 移动到关节角度: {np.round(np.degrees(joint_angles), 1)}°")
        
        # 转换为舵机目标
        robot = self.robot_models["so101_5dof"]  # 使用5关节模型进行控制
        home_pose = {name: self.controller.get_home_position(name) 
                     for name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]}
        
        targets = {}
        for i, name in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]):
            steps = int(round(
                home_pose[name] + 
                self.gear_sign[name] * joint_angles[i] * self.counts_per_rad
            ))
            targets[name] = self.controller.limit_position(name, steps)
        
        # 平滑移动
        self.controller.soft_move_to_pose(targets, step_count=8, interval=0.05)
        time.sleep(2)  # 等待稳定
    
    def verify_single_pose(self, joint_angles, pose_id):
        """
        验证单个姿态的建模精度
        
        Parameters
        ----------
        joint_angles : np.ndarray
            关节角度
        pose_id : int
            姿态编号
            
        Returns
        -------
        result : dict
            测试结果
        """
        print(f"\n{'='*60}")
        print(f"🧪 测试姿态 {pose_id}")
        print(f"{'='*60}")
        
        # 移动到目标姿态
        self.move_to_pose(joint_angles)
        
        # 读取实际达到的关节角度
        actual_joint_angles = self.read_current_joint_angles()
        print(f"\n📐 实际关节角度: {np.round(np.degrees(actual_joint_angles), 1)}°")
        print(f"   目标关节角度: {np.round(np.degrees(joint_angles), 1)}°")
        
        # 计算各模型的正运动学结果（使用实际关节角度）
        fk_results = {}
        for model_name, robot in self.robot_models.items():
            # 所有模型都是5关节，直接使用actual_joint_angles
            T = robot.fkine(actual_joint_angles)
            
            position = T[:3, 3]
            orientation = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
            
            fk_results[model_name] = {
                'position': position,
                'orientation': orientation,
                'transform': T
            }
        
        print("\n📊 正运动学计算结果:")
        for model_name, result in fk_results.items():
            pos = result['position']
            print(f"   {model_name:12s}: x={pos[0]:6.3f}, y={pos[1]:6.3f}, z={pos[2]:6.3f}")
        
        # 获取实际末端位置
        print("\n请测量实际末端执行器位置...")
        actual_position = self.capture_end_effector_position(manual=True)
        
        if actual_position is None:
            print("⚠️  跳过当前姿态测量")
            return None
        
        # 计算误差
        errors = {}
        for model_name, result in fk_results.items():
            predicted_pos = result['position']
            error = np.linalg.norm(actual_position - predicted_pos)
            errors[model_name] = error
            
            print(f"\n📏 {model_name} 误差分析:")
            print(f"   预测位置: x={predicted_pos[0]:6.3f}, y={predicted_pos[1]:6.3f}, z={predicted_pos[2]:6.3f}")
            print(f"   实际位置: x={actual_position[0]:6.3f}, y={actual_position[1]:6.3f}, z={actual_position[2]:6.3f}")
            print(f"   位置误差: {error*1000:6.1f} mm")
        
        # 保存结果
        result = {
            'pose_id': pose_id,
            'target_joint_angles': joint_angles.copy(),
            'actual_joint_angles': actual_joint_angles.copy(),
            'actual_position': actual_position.copy(),
            'fk_results': fk_results,
            'errors': errors
        }
        
        return result
    
    def run_verification(self, num_poses=5):
        """
        运行完整的验证流程
        
        Parameters
        ----------
        num_poses : int
            测试姿态数量
        """
        print("🚀 开始机器人建模验证")
        print(f"   测试姿态数量: {num_poses}")
        print(f"   机器人模型: {list(self.robot_models.keys())} (均为5关节模型)")
        
        # 回到初始位置
        print("\n🏠 移动到初始位置...")
        self.controller.move_all_home()
        time.sleep(2)
        
        # 生成测试姿态
        test_poses = self.generate_test_poses(num_poses)
        
        # 逐个测试
        for i, joint_angles in enumerate(test_poses, 1):
            result = self.verify_single_pose(joint_angles, i)
            if result is not None:
                self.test_results.append(result)
            
            # 询问是否继续
            if i < len(test_poses):
                response = input(f"\n继续下一个测试姿态 ({i+1}/{len(test_poses)})? [y/n]: ")
                if response.lower() != 'y':
                    break
        
        # 分析结果
        self.analyze_results()
    
    def analyze_results(self):
        """分析测试结果并生成报告"""
        if not self.test_results:
            print("❌ 没有测试结果可分析")
            return
        
        print(f"\n{'='*80}")
        print("📈 建模验证结果分析")
        print(f"{'='*80}")
        
        # 统计各模型的误差
        model_errors = {}
        for model_name in self.robot_models.keys():
            errors = [result['errors'][model_name] for result in self.test_results]
            model_errors[model_name] = {
                'mean': np.mean(errors) * 1000,  # 转换为mm
                'std': np.std(errors) * 1000,
                'max': np.max(errors) * 1000,
                'min': np.min(errors) * 1000,
                'errors': errors
            }
        
        # 打印统计结果
        print("\n📊 误差统计 (mm):")
        print(f"{'模型':12s} {'平均':>8s} {'标准差':>8s} {'最大':>8s} {'最小':>8s}")
        print("-" * 50)
        for model_name, stats in model_errors.items():
            print(f"{model_name:12s} {stats['mean']:8.1f} {stats['std']:8.1f} {stats['max']:8.1f} {stats['min']:8.1f}")
        
        # 推荐最佳模型
        best_model = min(model_errors.keys(), key=lambda k: model_errors[k]['mean'])
        print(f"\n🏆 推荐模型: {best_model}")
        print(f"   平均误差: {model_errors[best_model]['mean']:.1f} mm")
        
        # 模型差异说明
        print(f"\n📝 模型说明:")
        print(f"   so101: 标准SO-101模型（使用用户提供的DH参数）")
        print(f"   so101_5dof: 原始5关节模型")
        
        # 详细结果
        print(f"📋 详细测试结果:")
        print(f"{'测试':4s} {'目标角度(度)':35s} {'实际角度(度)':35s} {'实际位置':25s} {'最佳模型误差(mm)':15s}")
        print("-" * 120)
        for result in self.test_results:
            target_angles_deg = np.round(np.degrees(result['target_joint_angles']), 1)
            actual_angles_deg = np.round(np.degrees(result['actual_joint_angles']), 1)
            pos = result['actual_position']
            best_error = result['errors'][best_model] * 1000
            
            print(f"{result['pose_id']:4d} {str(target_angles_deg):35s} {str(actual_angles_deg):35s} "
                  f"[{pos[0]:6.3f},{pos[1]:6.3f},{pos[2]:6.3f}] {best_error:12.1f}")
        
        # 建议
        print(f"\n💡 建模改进建议:")
        if model_errors[best_model]['mean'] > 50:
            print("   - 误差较大（>50mm），建议重新测量机械尺寸或检查DH参数")
            print("   - 检查关节零位是否正确校准")
        elif model_errors[best_model]['mean'] > 20:
            print("   - 误差适中（20-50mm），可进一步精细调整DH参数")
            print("   - 考虑测量更多姿态以获得更全面的评估")
        else:
            print("   - 误差较小（<20mm），建模质量良好")
            print("   - 可用于实际应用")
            
        if model_errors[best_model]['std'] > 30:
            print("   - 误差不稳定（标准差>30mm），检查:")
            print("     * 机械间隙或关节精度")
            print("     * 测量方法的一致性")
            print("     * 关节编码器精度")
            
        # 保存结果
        self.save_results()
    
    def save_results(self):
        """保存测试结果到文件"""
        import json
        
        # 准备保存的数据
        save_data = {
            'test_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_tests': len(self.test_results),
                'models': list(self.robot_models.keys())
            },
            'results': []
        }
        
        for result in self.test_results:
            save_result = {
                'pose_id': result['pose_id'],
                'target_joint_angles_deg': np.degrees(result['target_joint_angles']).tolist(),
                'actual_joint_angles_deg': np.degrees(result['actual_joint_angles']).tolist(),
                'actual_position': result['actual_position'].tolist(),
                'errors_mm': {k: v*1000 for k, v in result['errors'].items()}
            }
            save_data['results'].append(save_result)
        
        filename = f"robot_verification_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {filename}")


def main():
    parser = argparse.ArgumentParser(description="机器人建模验证工具")
    parser.add_argument("--port", default="/dev/left_arm", help="串口设备路径")
    parser.add_argument("--camera", type=int, default=0, help="相机设备ID")
    parser.add_argument("--no-camera", action="store_true", help="禁用相机，使用手动测量")
    parser.add_argument("--poses", type=int, default=5, help="测试姿态数量")
    
    args = parser.parse_args()
    
    try:
        verifier = RobotModelVerifier(
            port=args.port,
            camera_id=args.camera,
            use_camera=not args.no_camera
        )
        
        verifier.run_verification(num_poses=args.poses)
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()