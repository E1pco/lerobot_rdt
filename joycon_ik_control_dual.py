#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from joyconrobotics import JoyconRobotics
from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof_gripper


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


def build_target_pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


@dataclass
class ArmState:
    name: str
    controller: ServoController
    robot: object
    joint_names: Tuple[str, ...]
    gear_sign: dict
    gear_ratio: dict
    home_pose: dict
    joycon: JoyconRobotics
    btn: _ButtonHelper
    current_q: np.ndarray
    base_pos: np.ndarray
    base_rpy: np.ndarray
    gripper_pos: int
    gripper_min: int
    gripper_max: int
    gripper_step: int
    z_offset: float
    z_step: float
    speed: int


class DualJoyConIKController:
    def __init__(
        self,
        *,
        right_port: str,
        left_port: str,
        right_config: str,
        left_config: str,
        baudrate: int = 1_000_000,
        base_speed: int = 800,
    ) -> None:
        print("=" * 70)
        print("Dual JoyCon IK Controller Init")
        print("=" * 70)

        print(f"\n[1/5] Connecting servo controllers...")
        self.right_controller = ServoController(port=right_port, baudrate=baudrate, config_path=right_config)
        self.left_controller = ServoController(port=left_port, baudrate=baudrate, config_path=left_config)
        print("✓ Servo controllers connected")

        print("\n[2/5] Building robot models...")
        right_robot = create_so101_5dof_gripper()
        left_robot = create_so101_5dof_gripper()
        print("✓ Robot models ready (5 DOF each)")

        print("\n[3/5] Homing both arms...")
        self.right_controller.move_all_home()
        self.left_controller.move_all_home()
        time.sleep(1.0)
        print("✓ Both arms homed")

        print("\n[4/5] Connecting Joy-Cons (right->right arm, left->left arm)...")
        right_joycon = JoyconRobotics(device="right", without_rest_init=False, common_rad=True, lerobot=False)
        left_joycon = JoyconRobotics(device="left", without_rest_init=False, common_rad=True, lerobot=False)
        print("✓ Joy-Cons connected and calibrated")

        print("\n[5/5] Syncing joint states and saving baselines...")
        self.right_arm = self._init_arm(
            name="right",
            controller=self.right_controller,
            robot=right_robot,
            joycon=right_joycon,
            base_speed=base_speed,
        )
        self.left_arm = self._init_arm(
            name="left",
            controller=self.left_controller,
            robot=left_robot,
            joycon=left_joycon,
            base_speed=base_speed,
        )
        print("✓ Baselines captured")

        self.running = True

    def _init_arm(
        self,
        *,
        name: str,
        controller: ServoController,
        robot: object,
        joycon: JoyconRobotics,
        base_speed: int,
    ) -> ArmState:
        joint_names = robot.joint_names
        gear_sign = robot.gear_sign
        gear_ratio = robot.gear_ratio
        home_pose = controller.home_pose
        current_q = robot.read_joint_angles(joint_names=joint_names, home_pose=home_pose, gear_sign=gear_sign, verbose=False)
        pose = robot.fkine(current_q)
        current_pos = pose[:3, 3]
        current_rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz")

        gripper_home = int(home_pose.get("gripper", 2037)) if isinstance(home_pose, dict) else 2037

        print(f"   {name}: pos={np.round(current_pos, 3)}, rpy(deg)={np.round(np.degrees(current_rpy), 1)}")
        return ArmState(
            name=name,
            controller=controller,
            robot=robot,
            joint_names=joint_names,
            gear_sign=gear_sign,
            gear_ratio=gear_ratio,
            home_pose=home_pose,
            joycon=joycon,
            btn=_ButtonHelper(),
            current_q=current_q,
            base_pos=current_pos.copy(),
            base_rpy=current_rpy.copy(),
            gripper_pos=gripper_home,
            gripper_min=1200,
            gripper_max=2800,
            gripper_step=50,
            z_offset=0.0,
            z_step=0.001,
            speed=int(base_speed),
        )

    def _update_current_joints(self, arm: ArmState) -> None:
        arm.current_q = arm.robot.read_joint_angles(
            joint_names=arm.joint_names,
            home_pose=arm.home_pose,
            gear_sign=arm.gear_sign,
            verbose=False,
        )
        pose = arm.robot.fkine(arm.current_q)
        arm.base_pos = pose[:3, 3]
        arm.base_rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz")

    def _reconnect_joycon(self, arm: ArmState) -> None:
        try:
            arm.joycon.disconnnect()
            time.sleep(0.3)
        except Exception:
            pass
        arm.joycon = JoyconRobotics(device=arm.name, without_rest_init=False, common_rad=True, lerobot=False)
        time.sleep(0.3)
        self._update_current_joints(arm)

    def _process_buttons(self, arm: ArmState) -> None:
        arm.btn.update(arm.joycon.button)

        if arm.btn.rising("x"):
            print(f"\n🛑 {arm.name} Joy-Con X pressed -> exit")
            self.running = False
            return

        if arm.btn.rising("home"):
            print(f"\n🏠 {arm.name} home -> homing arm and recentering baseline")
            arm.controller.fast_move_to_pose(arm.home_pose)
            time.sleep(1.0)
            self._update_current_joints(arm)
            arm.base_pos = arm.base_pos.copy()
            arm.base_rpy = arm.base_rpy.copy()
            arm.z_offset = 0.0
            self._reconnect_joycon(arm)

        if arm.btn.repeat("plus", 0.2):
            arm.speed = min(arm.speed + 100, 2000)
            print(f"⚡ {arm.name} speed -> {arm.speed}")

        if arm.btn.repeat("minus", 0.2):
            arm.speed = max(arm.speed - 100, 200)
            print(f"🐌 {arm.name} speed -> {arm.speed}")

        if arm.btn.repeat("zr", 0.1):
            arm.gripper_pos = max(arm.gripper_pos - arm.gripper_step, arm.gripper_min)
            arm.controller.move_servo("gripper", arm.gripper_pos, arm.speed)
            print(f"✊ {arm.name} gripper -> {arm.gripper_pos}")

        if arm.btn.repeat("r", 0.1):
            arm.gripper_pos = min(arm.gripper_pos + arm.gripper_step, arm.gripper_max)
            arm.controller.move_servo("gripper", arm.gripper_pos, arm.speed)
            print(f"✋ {arm.name} gripper -> {arm.gripper_pos}")

        if arm.btn.repeat("b", 0.1):
            arm.z_offset += arm.z_step
            print(f"⬆️  {arm.name} z_offset -> {arm.z_offset:.4f}")

    def _step_arm(self, arm: ArmState) -> None:
        pose, _, _ = arm.joycon.get_control()
        offset_pos = np.array([pose[0], pose[1], pose[2]])
        offset_rpy = np.array([-pose[3], -pose[4], pose[5]])
        offset_pos[2] += arm.z_offset

        target_pos = arm.base_pos + offset_pos
        target_rpy = arm.base_rpy + offset_rpy

        T_goal = build_target_pose(*target_pos, *target_rpy)
        sol = arm.robot.ikine_LM(
            Tep=T_goal,
            q0=arm.current_q,
            ilimit=50,
            slimit=3,
            tol=1e-3,
            mask=[1, 1, 1, 0.8, 0.8, 0],
            k=0.1,
            method="chan",
        )

        if not sol.success:
            return

        arm.current_q = sol.q
        servo_targets = arm.robot.q_to_servo_targets(
            arm.current_q,
            arm.joint_names,
            arm.home_pose,
            gear_ratio=arm.gear_ratio,
            gear_sign=arm.gear_sign,
        )
        for joint in arm.joint_names:
            servo_targets[joint] = arm.controller.limit_position(joint, servo_targets[joint])
        arm.controller.fast_move_to_pose(servo_targets, speed=arm.speed)

    def run(self) -> None:
        print("\n" + "=" * 70)
        print("🎮 Dual JoyCon control ready")
        print("Controls (per Joy-Con): X=exit, Home=home+recenter, +/-=speed, ZR/R=gripper, B=raise Z")
        print("=" * 70 + "\n")

        try:
            while self.running:
                self._process_buttons(self.right_arm)
                self._process_buttons(self.left_arm)

                if not self.running:
                    break

                self._step_arm(self.right_arm)
                self._step_arm(self.left_arm)

                time.sleep(0.04)
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        print("\n" + "=" * 70)
        print("Shutting down...")
        print("=" * 70)
        for arm in (self.right_arm, self.left_arm):
            try:
                arm.joycon.disconnnect()
            except Exception:
                pass
        print("Joy-Cons disconnected")
        print("Done")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dual JoyCon IK control for both arms")
    parser.add_argument("--right-port", type=str, default="/dev/right_arm", help="Serial port for right arm")
    parser.add_argument("--left-port", type=str, default="/dev/left_arm", help="Serial port for left arm")
    parser.add_argument("--right-config", type=str, default="./driver/right_arm.json", help="Config for right arm")
    parser.add_argument("--left-config", type=str, default="./driver/left_arm.json", help="Config for left arm")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Servo baudrate")
    parser.add_argument("--base-speed", type=int, default=800, help="Initial servo speed for both arms")

    args = parser.parse_args()

    try:
        controller = DualJoyConIKController(
            right_port=args.right_port,
            left_port=args.left_port,
            right_config=args.right_config,
            left_config=args.left_config,
            baudrate=args.baudrate,
            base_speed=args.base_speed,
        )
        controller.run()
    except Exception as exc:  # pragma: no cover
        print(f"❌ Error: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
