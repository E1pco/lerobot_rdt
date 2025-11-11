#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import JoyCon controller
sys.path.insert(0, '/home/elpco/code/lerobot/joycon-robotics')
from joyconrobotics import JoyconRobotics

# Import IK solver and servo controller
from ftservo_controller import ServoController
from lerobot_kinematics.ET import ET


# ============================================================================
# Robot Model Definition
# ============================================================================

def create_so101_5dof():
    """
    Create SO-101 5DOF robot kinematic model using Elementary Transform (ET)
    
    Joint sequence: base(Rz) → shoulder(Ry) → elbow(Ry) → wrist_pitch(Ry) → wrist_roll(Rx)
    
    Returns
    -------
    robot : ET
        Robot kinematic chain
    """
    E1 = ET.Rz()      # shoulder_pan
    E2 = ET.tx(0.0612)
    E3 = ET.tz(0.0598)
    E4 = ET.tx(0.02943)
    E5 = ET.tz(0.05504)
    E6 = ET.Ry()      # shoulder_lift
    E7 = ET.tz(0.1127)
    E8 = ET.tx(0.02798)
    E9 = ET.Ry()      # elbow_flex
    E10 = ET.tx(0.13504)
    E11 = ET.tz(0.00519)
    E12 = ET.Ry()     # wrist_flex
    E13 = ET.tx(0.0593)
    E14 = ET.tz(0.00996)
    E15 = ET.Rx()     # wrist_roll

    robot = E1 * E2 * E3 * E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15

    # Joint limits from URDF
    robot.qlim = np.array([
        [-1.91986, -1.74533, -1.69, -1.65806, -2.74385],
        [ 1.91986,  1.74533,  1.69,  1.65806,  2.84121]
    ])
    return robot


def build_target_pose(x, y, z, roll, pitch, yaw):
    """
    Build 4x4 homogeneous transformation matrix from position and orientation
    
    Parameters
    ----------
    x, y, z : float
        Position in meters
    roll, pitch, yaw : float
        Orientation in radians (XYZ Euler angles)
    
    Returns
    -------
    T : ndarray (4, 4)
        Homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


# ============================================================================
# Main Control Class
# ============================================================================

class JoyConIKController:
    """
    Main controller class that integrates Joy-Con input with IK solver
    """
    
    def __init__(self, device='right', port='/dev/ttyACM0', baudrate=1_000_000,
                 config_path='left_arm.json'):
        """
        Initialize JoyCon-IK controller
        
        Parameters
        ----------
        device : str
            'right' or 'left' Joy-Con
        port : str
            Serial port for servo controller
        baudrate : int
            Baudrate for serial communication
        config_path : str
            Path to servo configuration file
        """
        print("=" * 70)
        print("JoyCon-IK Controller Initialization")
        print("=" * 70)
        
        # Initialize servo controller first
        print(f"\n[1/5] Connecting to servo controller on {port}...")
        self.controller = ServoController(
            port=port,
            baudrate=baudrate,
            config_path=config_path
        )
        print(f"✓ Servo controller connected")
        
        # Home position map
        self.home_pose = {
            "shoulder_pan": 2096,
            "shoulder_lift": 1983,
            "elbow_flex": 2100,
            "wrist_flex": 1954,
            "wrist_roll": 2048,
            "gripper": 2037,
        }
        
        # Joint configuration
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                           "wrist_flex", "wrist_roll"]
        self.gear_sign = {k: +1 for k in self.joint_names}
        self.gear_ratio = {k: 1.0 for k in self.joint_names}
        
        # Initialize robot model
        print(f"\n[2/5] Building robot kinematic model...")
        self.robot = create_so101_5dof()
        print(f"✓ Robot model created (5 DOF)")
        
        # Move to home position BEFORE connecting JoyCon
        print(f"\n[3/5] Moving robot to home position...")
        # self.controller.move_all_home()
        self.controller.soft_move_to_pose(self.home_pose, step_count=10, interval=0.1)
        time.sleep(1.0)
        print(f"✓ Home position reached")
        
        # Read and sync current joint angles
        print(f"\n[4/5] Reading servo positions and calculating pose...")
        self._update_current_joints()
        
        # Now connect Joy-Con (after robot is ready)
        print(f"\n[5/5] Connecting to {device} Joy-Con...")
        self.joycon = JoyconRobotics(
            device=device,
            without_rest_init=False,  # Enable auto-calibration
            common_rad=True,
            lerobot=False
        )
        print(f"✓ Joy-Con connected and calibrated")
        
        # Control parameters
        self.speed = 800  # Default speed
        self.gripper_open = True
        self.running = True
        
        # 保存初始位姿作为基准（用于叠加JoyCon偏移）
        self.base_pos = self.current_pos.copy()
        self.base_rpy = self.current_rpy.copy()
        print(f"\n🎯 基准位姿已保存:")
        print(f"   基准位置: {np.round(self.base_pos, 3)} m")
        print(f"   基准姿态: {np.round(np.degrees(self.base_rpy), 1)} deg")
        
        print("\n" + "=" * 70)
        print("✓ Initialization Complete")
        print("=" * 70)
    
    def _update_current_joints(self):
        """Read current joint angles from servos and calculate pose"""
        # ✅ 从舵机读取当前位置
        ids = [self.controller.config[name]["id"] for name in self.joint_names]
        resp = self.controller.servo.sync_read(0x38, 2, ids)
        
        self.current_q = np.zeros(5)
        counts_per_rad = 4096 / (2 * np.pi)
        
        print("\n📡 读取当前舵机位置：")
        for i, name in enumerate(self.joint_names):
            sid = self.controller.config[name]["id"]
            cur_pos = resp.get(sid, [self.home_pose[name] & 0xFF, 
                                     self.home_pose[name] >> 8])
            current = cur_pos[0] + (cur_pos[1] << 8)
            delta = current - self.home_pose[name]
            self.current_q[i] = self.gear_sign[name] * delta / counts_per_rad
            print(f"  {name:15s}: {current:4d} (Δ={delta:+d}) → {self.current_q[i]:+.4f} rad")
        
        # ✅ 根据当前角度计算末端实际位姿
        T_current = self.robot.fkine(self.current_q).A
        self.current_pos = T_current[:3, 3]
        self.current_rpy = R.from_matrix(T_current[:3, :3]).as_euler('xyz')
        
        print(f"\n✅ 已同步当前机械臂姿态")
        print(f"   pos={np.round(self.current_pos, 3)}, rpy(deg)={np.round(np.degrees(self.current_rpy), 1)}")
    
    def _process_buttons(self):
        """Process Joy-Con button events"""
        # Check for exit button (X)
        if self.joycon.button.x == 1:
            print("\n🛑 X button pressed - Exiting...")
            self.running = False
            return
        
        # Home按钮：复位机械臂到初始位置
        if self.joycon.button.home == 1:
            print("\n🏠 Home按钮按下 - 机械臂复位中...")
            self.controller.fast_move_to_pose(self.home_pose)
            time.sleep(1.0)  # 等待复位完成
            print("✓ 机械臂已复位到初始位置")
            
            # 重新读取舵机位置并计算位姿
            print("� 读取复位后的舵机位置...")
            self._update_current_joints()
            
            # 重置JoyCon方向校准
            print("\n🔄 重置JoyCon方向校准...")
            self.joycon.reset_joycon()
            time.sleep(0.5)
        
        # Speed adjustment
        if self.joycon.button.plus == 1:
            self.speed = min(self.speed + 100, 2000)
            print(f"\n⚡ Speed increased: {self.speed}")
            time.sleep(0.2)
        
        if self.joycon.button.minus == 1:
            self.speed = max(self.speed - 100, 200)
            print(f"\n🐌 Speed decreased: {self.speed}")
            time.sleep(0.2)
        
        # Gripper control (ZR button)
        if self.joycon.button.zr == 1:
            self.gripper_open = not self.gripper_open
            gripper_pos = 2800 if self.gripper_open else 1200
            self.controller.move_servo("gripper", gripper_pos, self.speed)
            status = "OPEN" if self.gripper_open else "CLOSED"
            print(f"\n🤏 Gripper {status}")
            time.sleep(0.3)
    
    def run(self):
        """Main control loop"""
        print("\n" + "=" * 70)
        print("🎮 JoyCon 控制已启动")
        print("=" * 70)
        print("\n控制说明:")
        print("  移动 Joy-Con → 控制机械臂位置和姿态")
        print("  ZR → 切换夹爪开合")
        print("  Home → 机械臂复位到初始位置 + 重置JoyCon方向")
        print("  +/- → 调节速度")
        print("  X → 退出程序")
        print("\n" + "=" * 70 + "\n")
        
        try:
            while self.running:
                # 处理按键事件
                self._process_buttons()
                
                if not self.running:
                    break
                
                # 获取 Joy-Con 姿态数据（偏移量）
                pose, gripper_status, _ = self.joycon.get_control()
                joycon_offset_pos = np.array([pose[0], pose[1], pose[2]])
                joycon_offset_rpy = np.array([pose[3], pose[4], pose[5]])
                
                # 实时打印 JoyCon 原始数据
                print(f"JoyCon偏移: {[f'{x:.3f}' for x in pose]}, 夹爪状态={gripper_status}")
                
                # 叠加到基准位姿上
                pos = self.base_pos + joycon_offset_pos
                rpy = self.base_rpy + joycon_offset_rpy
                
                # 打印叠加后的目标位姿
                print(f"目标位姿: pos={pos.round(3)}, rpy(deg)={np.rad2deg(rpy).round(1)}")
                
                # 构建目标位姿矩阵
                T_goal = build_target_pose(*pos, *rpy)
                
                # 逆运动学求解
                sol = self.robot.ikine_LM(
                    Tep=T_goal,
                    q0=self.current_q,
                    ilimit=50,
                    tol=1e-3,
                    mask=[1, 1, 1, 0.8, 0.8,0], 
                    k=0.1,
                    method="sugihara"
                )
                
                if sol.success:
                    # 更新当前关节角度
                    self.current_q = sol.q
                    
                    # 使用 ServoController 的方法转换为舵机目标位置
                    servo_targets = self.controller.q_to_servo_targets(
                        self.current_q,
                        self.joint_names,
                        self.home_pose,
                        gear_ratio=self.gear_ratio,
                       
                        gear_sign = {
                                "shoulder_pan": -1,
                                "shoulder_lift": +1,
                                "elbow_flex":   +1,
                                "wrist_flex":   -1,
                                "wrist_roll":   +1,
                            }
                    )
                    
                    # 限位检查
                    for k in self.joint_names:
                        servo_targets[k] = self.controller.limit_position(k, servo_targets[k])
                    
                    # 一次性发送所有舵机指令
                    self.controller.fast_move_to_pose(servo_targets, speed=self.speed)
                    
                    # 打印状态
                    print(f"\r→ pos={pos.round(3)}, rpy(deg)={np.rad2deg(rpy).round(1)}, speed={self.speed}", end='')
                else:
                    print(f"\r❌ IK失败，跳过", end='')
                
                time.sleep(0.04)  # 与参考代码一致的更新频率
                
        except KeyboardInterrupt:
            print("\n\n⚠ Keyboard interrupt detected")
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup and shutdown"""
        print("\n\n" + "=" * 70)
        print("Shutting down...")
        print("=" * 70)
        
        try:
            print("\n[1/2] Disconnecting Joy-Con...")
            self.joycon.disconnnect()
            print("✓ Joy-Con disconnected")
        except Exception as e:
            print(f"⚠ Joy-Con disconnect error: {e}")
        
        try:
            print("\n[2/2] Stopping servos...")
            # Optionally return to home position
            # self.controller.move_all_home()
            print("✓ Control stopped")
        except Exception as e:
            print(f"⚠ Servo stop error: {e}")
        
        print("\n" + "=" * 70)
        print("✓ Shutdown complete")
        print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='JoyCon to IK Solver - Real-time robot control',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['right', 'left'],
        default='right',
        help='Select Joy-Con device (default: right)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=str,
        default='/dev/ttyACM0',
        help='Serial port for servo controller (default: /dev/ttyACM0)'
    )
    
    parser.add_argument(
        '--baudrate', '-b',
        type=int,
        default=1000000,
        help='Baudrate for serial communication (default: 1000000)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='servo_config.json',
        help='Path to servo configuration file (default: servo_config.json)'
    )
    
    args = parser.parse_args()
    
    try:
        controller = JoyConIKController(
            device=args.device,
            port=args.port,
            baudrate=args.baudrate,
            config_path=args.config
        )
        
        controller.run()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
