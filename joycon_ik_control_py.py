#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
# Import JoyCon controller
from joyconrobotics import JoyconRobotics

# Import IK solver and servo controller

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_6dof_gripper


class _ButtonHelper:
    def __init__(self) -> None:
        self._prev: dict[str, int] = {}
        self._cur: dict[str, int] = {}
        self._last_fire_s: dict[str, float] = {}

    def update(self, button_obj: object) -> None:
        self._prev = self._cur
        self._cur = {}
        for name in ("x", "home", "plus", "minus", "zr", "r", "b"):
            try:
                self._cur[name] = int(getattr(button_obj, name, 0) or 0)
            except Exception:
                self._cur[name] = 0

    def rising(self, name: str) -> bool:
        return self._cur.get(name, 0) == 1 and self._prev.get(name, 0) == 0

    def repeat(self, name: str, interval_s: float) -> bool:
        if self._cur.get(name, 0) != 1:
            return False
        now = time.time()
        if self.rising(name):
            self._last_fire_s[name] = now
            return True
        last = self._last_fire_s.get(name, 0.0)
        if now - last >= float(interval_s):
            self._last_fire_s[name] = now
            return True
        return False


def build_target_pose(x, y, z, roll, pitch, yaw):
    """
    Build 4x4 homogeneous transformation matrix from position and orientation
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
    
    def __init__(self, device='left', port='/dev/left_arm', baudrate=1_000_000,
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
        self.home_pose = self.controller.home_pose
        
        # Initialize robot model
        print(f"\n[2/5] Building robot kinematic model...")
        self.robot = create_so101_6dof_gripper()
        print(f"✓ Robot model created (5 DOF)")
        
        # 从 robot 对象获取关节配置
        self.joint_names = self.robot.joint_names
        self.gear_sign = self.robot.gear_sign
        self.gear_ratio = self.robot.gear_ratio
        
        # Move to home position BEFORE connecting JoyCon
        print(f"\n[3/5] Moving robot to home position...")
        self.controller.move_all_home()
        # self.controller.soft_move_to_pose(self.home_pose, step_count=10, interval=0.1)
        time.sleep(1.0)
        print(f"✓ Home position reached")
        
        # Read and sync current joint angles
        print(f"\n[4/5] Reading servo positions and calculating pose...")
        self._update_current_joints()
        
        # Now connect Joy-Con (after robot is ready)
        print(f"\n[5/5] Connecting to {device} Joy-Con...")
        self.joycon_device = device  # 保存设备类型以便后续重新连接
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

        self._btn = _ButtonHelper()
        
        # Gripper control parameters
        self.gripper_pos = 2037  # 初始夹爪位置
        self.gripper_min = 1200  # 最小值（完全闭合）
        self.gripper_max = 2800  # 最大值（完全打开）
        self.gripper_step = 50   # 每次调整步长
        
        # Z axis adjustment
        self.z_offset = 0.0  # Z 轴偏移
        self.z_step = 0.001  # Z 轴每次调整步长
        
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
        self.current_q = self.robot.read_joint_angles(
            joint_names=self.joint_names,
            home_pose=self.home_pose,
            gear_sign=self.gear_sign,
            verbose=True
        )
        
        pose = self.robot.fkine(self.current_q)
        self.current_pos = pose[:3, 3]  # [x, y, z]
        # 使用 scipy Rotation 从矩阵转换为 RPY 角度

        self.current_rpy = R.from_matrix(pose[:3, :3]).as_euler('xyz')  # [roll, pitch, yaw]

        print(f"\n✅ 已同步当前机械臂姿态")
        print(f"   pos={np.round(self.current_pos, 3)}, rpy(deg)={np.round(np.degrees(self.current_rpy), 1)}")
    
    def _reconnect_joycon(self):
        """Disconnect and reconnect Joy-Con"""
        try:
            print("\n🔄 断开 Joy-Con 连接...")
            self.joycon.disconnnect()
            time.sleep(0.5)
            print("✓ Joy-Con 已断开")
        except Exception as e:
            print(f"⚠ Joy-Con 断开错误: {e}")
        
        try:
            print("\n🔄 重新连接 Joy-Con...")
            # 重新创建 JoyCon 实例
            self.joycon = JoyconRobotics(
                device=self.joycon_device,
                without_rest_init=False,  # Enable auto-calibration
                common_rad=True,
                lerobot=False
            )
            print("✓ Joy-Con 已重新连接并校准")
            
            # 更新基准位姿
            print("\n📍 更新基准位姿...")
            self.base_pos = self.current_pos.copy()
            self.base_rpy = self.current_rpy.copy()
            print(f"   基准位置: {np.round(self.base_pos, 3)} m")
            print(f"   基准姿态: {np.round(np.degrees(self.base_rpy), 1)} deg")
        except Exception as e:
            print(f"❌ Joy-Con 重新连接失败: {e}")
            self.running = False
    
    def _process_buttons(self):
        """Process Joy-Con button events"""
        self._btn.update(self.joycon.button)

        # Check for exit button (X)
        if self._btn.rising("x"):
            print("\n🛑 X button pressed - Exiting...")
            self.running = False
            return
        
        # Home按钮：复位机械臂到初始位置
        if self._btn.rising("home"):
            print("\n🏠 Home按钮按下 - 机械臂复位中...")
            self.controller.fast_move_to_pose(self.home_pose)
            time.sleep(1.0)  # 等待复位完成
            print("✓ 机械臂已复位到初始位置")
            
            # 重新读取舵机位置并计算位姿
            print("📡 读取复位后的舵机位置...")
            self._update_current_joints()
            time.sleep(0.5)
            
            # 重新连接 Joy-Con
            self._reconnect_joycon()
            time.sleep(0.5)
        
        # Speed adjustment
        if self._btn.repeat("plus", 0.2):
            self.speed = min(self.speed + 100, 2000)
            print(f"\n⚡ Speed increased: {self.speed}")
        
        if self._btn.repeat("minus", 0.2):
            self.speed = max(self.speed - 100, 200)
            print(f"\n🐌 Speed decreased: {self.speed}")
        
        # Gripper control (ZR button to tighten, R button to loosen)
        if self._btn.repeat("zr", 0.1):
            # ZR 按下：夹爪收紧一点
            self.gripper_pos = max(self.gripper_pos - self.gripper_step, self.gripper_min)
            self.controller.move_servo("gripper", self.gripper_pos, self.speed)
            print(f"\n✊ Gripper tightened: {self.gripper_pos}")
        
        if self._btn.repeat("r", 0.1):
            # R 按下：夹爪松开一点
            self.gripper_pos = min(self.gripper_pos + self.gripper_step, self.gripper_max)
            self.controller.move_servo("gripper", self.gripper_pos, self.speed)
            print(f"\n✋ Gripper loosened: {self.gripper_pos}")
        
        # Z adjustment buttons
        if self._btn.repeat("b", 0.1):
            # B 按下：增大 z
            self.z_offset += self.z_step
            print(f"\n⬆️  Z increased: {self.z_offset:.4f}")
    
    def run(self):
        """Main control loop"""
        print("\n" + "=" * 70)
        print("🎮 JoyCon 控制已启动")
        print("=" * 70)
        print("\n控制说明:")
        print("  移动 Joy-Con → 控制机械臂位置和姿态")
        print("  ZR → 夹爪收紧一点")
        print("  R → 夹爪松开一点")
        print("  B → 增大 Z（向上移动）")
        print("  Home → 机械臂复位到初始位置 + 重新连接 Joy-Con")
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
                joycon_offset_rpy = np.array([-pose[3], -pose[4], -pose[5]])
                joycon_offset_pos[2] += self.z_offset
                print(f"JoyCon偏移: {[f'{x:.3f}' for x in joycon_offset_pos]}, Z_manual={self.z_offset:.4f}, 夹爪状态={gripper_status}")
                pos = self.base_pos + joycon_offset_pos
                rpy = self.base_rpy + joycon_offset_rpy
                
                print(f"目标位姿: pos={pos.round(3)}, rpy(deg)={np.rad2deg(rpy).round(1)}")
                
                # 构建目标位姿矩阵并使用 Robot 的 ikine_LM 求解
                T_goal = build_target_pose(*pos, *rpy)
                sol = self.robot.ikine_LM(
                    Tep=T_goal,
                    q0=self.current_q,
                    ilimit=50,
                    slimit=3,
                    tol=1e-3,
                    mask=[1, 1, 1, 0.8, 0.8, 0.8],
                    k=0.1,
                    method="chan"
                )

                if sol.success:
                    # 更新当前关节角度
                    self.current_q = sol.q
                    servo_targets = self.robot.q_to_servo_targets(
                        self.current_q,
                        self.joint_names,
                        self.home_pose,
                        gear_ratio=self.gear_ratio,
                        gear_sign=self.gear_sign
                    )
                    # 限位检查
                    for k in self.joint_names:
                        servo_targets[k] = self.controller.limit_position(k, servo_targets[k])
                    # 一次性发送所有舵机指令
                    self.controller.fast_move_to_pose(servo_targets, speed=self.speed)
                
                    
                    print(f"\r→ pos={pos.round(3)}, rpy(deg)={np.rad2deg(rpy).round(1)}, speed={self.speed}", end='')
                else:
                    print(f"\r❌ IK失败，跳过", end='')
                
                time.sleep(0.04) 
                
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
        default='/dev/right_arm',
        help='Serial port for servo controller (default: /dev/right_arm)'
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
        default='./driver/right_arm.json',
        help='Path to servo configuration file (default: right_arm.json)'
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
