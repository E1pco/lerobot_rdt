#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""JoyCon teleop + RDT (ICLR'25) fine-tuning dataset collector.

Paper alignment (2410.07864v2):
- o_t := (X_{t-1:t}, z_t, c)
- 3-view RGB order: exterior, right-wrist, left-wrist; missing views padded.
- Pad to square, resize 384x384.
- z_t and a_t embedded into 128-dim unified space (Table 4), with availability mask.
- Fine-tuning storage: HDF5.

Controls (default, right JoyCon):
- Y: toggle recording on/off
- A: start a new episode (closes previous if recording)
- X: exit
- Home: robot home + JoyCon reconnect (same as joycon_ik_control_py.py)

Robot motion controls keep the original mapping:
- Move Joy-Con: Cartesian pose offset (position + RPY)
- ZR/R: gripper tighten/loosen (servo steps)
- +/-: speed
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

# Ensure repo-root imports work when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from driver.ftservo_controller import ServoController
from ik.robot import create_so101_5dof
from joyconrobotics import JoyconRobotics

from rdt_hdf5 import (
    RDTHDF5EpisodeWriter,
    UnifiedVector,
    fill_slice,
    make_unified_vector,
    rotmat_to_rot6d,
    RIGHT_ARM_JOINT_POS,
    RIGHT_ARM_JOINT_VEL,
    RIGHT_EEF_POS,
    RIGHT_EEF_ROT6D,
    RIGHT_EEF_LIN_VEL,
    RIGHT_EEF_ANG_VEL,
    RIGHT_GRIPPER_POS,
    RIGHT_GRIPPER_VEL,
    LEFT_ARM_JOINT_POS,
    LEFT_ARM_JOINT_VEL,
    LEFT_EEF_POS,
    LEFT_EEF_ROT6D,
    LEFT_EEF_LIN_VEL,
    LEFT_EEF_ANG_VEL,
    LEFT_GRIPPER_POS,
    LEFT_GRIPPER_VEL,
)


@dataclass
class ArmRig:
    name: str  # "right" | "left"
    controller: Optional[ServoController]
    robot: object
    joint_names: Tuple[str, ...]
    gear_sign: dict
    gear_ratio: dict
    home_pose: Optional[dict]
    current_q: np.ndarray
    prev_q: np.ndarray
    prev_eef_pos: np.ndarray
    prev_eef_R: np.ndarray
    base_pos: np.ndarray
    base_rpy: np.ndarray
    gripper_pos_steps: int
    gripper_min: int
    gripper_max: int
    gripper_step: int


def build_target_pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def pad_to_square_and_resize_rgb(
    frame_bgr: np.ndarray,
    *,
    out_size: int,
    pad_bgr: Tuple[int, int, int],
) -> np.ndarray:
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")

    h, w = frame_bgr.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), pad_bgr, dtype=np.uint8)

    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = frame_bgr

    resized = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb


class MultiViewCamera:
    def __init__(
        self,
        *,
        exterior: Optional[str],
        right_wrist: Optional[str],
        left_wrist: Optional[str],
        width: int,
        height: int,
    ) -> None:
        self._caps: Dict[str, Optional[cv2.VideoCapture]] = {
            "exterior": self._open(exterior, width, height),
            "right_wrist": self._open(right_wrist, width, height),
            "left_wrist": self._open(left_wrist, width, height),
        }

    @staticmethod
    def _open(source: Optional[str], width: int, height: int) -> Optional[cv2.VideoCapture]:
        if source is None:
            return None
        try:
            idx = int(source)
            cap = cv2.VideoCapture(idx)
        except ValueError:
            cap = cv2.VideoCapture(source)

        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open camera/video source: {source}")
        return cap

    def read_bgr(self, view: str) -> Optional[np.ndarray]:
        cap = self._caps.get(view)
        if cap is None:
            return None
        ok, frame = cap.read()
        return frame if ok else None

    def close(self) -> None:
        for cap in self._caps.values():
            if cap is not None:
                cap.release()


def compute_ang_vel_rad_s(R_prev: np.ndarray, R_cur: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros((3,), dtype=np.float32)
    dR = R_prev.T @ R_cur
    rotvec = R.from_matrix(dR).as_rotvec()
    return (rotvec / dt).astype(np.float32)


def gripper_steps_to_rad(steps: int, home_steps: int) -> float:
    counts_per_rad = 4096.0 / (2.0 * np.pi)
    return float((steps - home_steps) / counts_per_rad)


class JoyConRDTCollector:
    def __init__(
        self,
        *,
        device: str,
        right_port: str,
        left_port: str,
        baudrate: int,
        right_config_path: str,
        left_config_path: str,
        control_arm: str,
        out_dir: Path,
        instruction: str,
        control_hz: float,
        ta: int,
        image_size: int,
        cam_exterior: Optional[str],
        cam_right_wrist: Optional[str],
        cam_left_wrist: Optional[str],
        cam_width: int,
        cam_height: int,
        pad_bgr: Tuple[int, int, int],
        no_robot: bool,
        no_camera: bool,
        no_home: bool,
    ) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.instruction = instruction
        self.control_hz = float(control_hz)
        self.dt = 1.0 / self.control_hz if self.control_hz > 0 else 0.04
        self.ta = int(ta)
        self.image_size = int(image_size)
        self.pad_bgr = pad_bgr

        self.no_robot = no_robot
        self.no_camera = no_camera

        self.speed = 800
        self.running = True
        self.recording = False
        self.episode_idx = 0

        self.control_arm = control_arm
        if self.control_arm not in ("right", "left"):
            raise ValueError("control_arm must be 'right' or 'left'")

        self.z_offset = 0.0
        self.z_step = 0.001

        self.right_arm = self._init_arm(
            name="right",
            port=right_port,
            baudrate=baudrate,
            config_path=right_config_path,
            no_home=no_home,
        )
        self.left_arm = self._init_arm(
            name="left",
            port=left_port,
            baudrate=baudrate,
            config_path=left_config_path,
            no_home=no_home,
        )

        if not self.no_robot and not no_home:
            # Home both arms together (reduces risk of collisions from staggered homing)
            if self.right_arm.controller is not None:
                self.right_arm.controller.move_all_home()
            if self.left_arm.controller is not None:
                self.left_arm.controller.move_all_home()
            time.sleep(1.0)

            # Re-sync after homing
            self.right_arm.current_q = self.right_arm.robot.read_joint_angles(
                joint_names=self.right_arm.joint_names,
                home_pose=self.right_arm.home_pose,
                gear_sign=self.right_arm.gear_sign,
                gear_ratio=self.right_arm.gear_ratio,
                verbose=False,
            )
            self.right_arm.prev_q = self.right_arm.current_q.copy()
            self.left_arm.current_q = self.left_arm.robot.read_joint_angles(
                joint_names=self.left_arm.joint_names,
                home_pose=self.left_arm.home_pose,
                gear_sign=self.left_arm.gear_sign,
                gear_ratio=self.left_arm.gear_ratio,
                verbose=False,
            )
            self.left_arm.prev_q = self.left_arm.current_q.copy()

            self._refresh_base_poses()

        self.joycon_device = device
        self.joycon = JoyconRobotics(device=device, without_rest_init=False, common_rad=True, lerobot=False)

        self.cams = None if self.no_camera else MultiViewCamera(
            exterior=cam_exterior,
            right_wrist=cam_right_wrist,
            left_wrist=cam_left_wrist,
            width=cam_width,
            height=cam_height,
        )

        # Per-view history buffers of processed RGB frames
        self.hist: Dict[str, Deque[np.ndarray]] = {
            "exterior": deque(maxlen=2),
            "right_wrist": deque(maxlen=2),
            "left_wrist": deque(maxlen=2),
        }

        if self.no_camera:
            dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            for k in self.hist:
                self.hist[k].append(dummy)
                self.hist[k].append(dummy)

        self.writer: Optional[RDTHDF5EpisodeWriter] = None

    def _init_arm(self, *, name: str, port: str, baudrate: int, config_path: str, no_home: bool) -> ArmRig:
        # In no-robot mode, create a dummy rig.
        if self.no_robot:
            robot = create_so101_5dof()
            q = np.zeros((5,), dtype=np.float32)
            return ArmRig(
                name=name,
                controller=None,
                robot=robot,
                joint_names=tuple(robot.joint_names),
                gear_sign=robot.gear_sign,
                gear_ratio=robot.gear_ratio,
                home_pose=None,
                current_q=q,
                prev_q=q.copy(),
                prev_eef_pos=np.zeros((3,), dtype=np.float32),
                prev_eef_R=np.eye(3, dtype=np.float32),
                base_pos=np.zeros((3,), dtype=np.float32),
                base_rpy=np.zeros((3,), dtype=np.float32),
                gripper_pos_steps=2037,
                gripper_min=1200,
                gripper_max=2800,
                gripper_step=50,
            )

        controller = ServoController(port=port, baudrate=baudrate, config_path=config_path)
        robot = create_so101_5dof()
        robot.set_servo_controller(controller)

        home_pose = controller.home_pose
        joint_names = tuple(robot.joint_names)

        # Sync current joint angles from hardware.
        q = robot.read_joint_angles(
            joint_names=joint_names,
            home_pose=home_pose,
            gear_sign=robot.gear_sign,
            gear_ratio=robot.gear_ratio,
            verbose=False,
        )

        # Try to read current gripper steps to avoid an unexpected jump.
        try:
            gripper_steps = int(controller.read_single_position("gripper"))
        except Exception:
            gripper_steps = int(home_pose.get("gripper", 2037))

        gr_cfg = controller.config.get("gripper", {})
        gmin = int(gr_cfg.get("range_min", 1200))
        gmax = int(gr_cfg.get("range_max", 2800))

        arm = ArmRig(
            name=name,
            controller=controller,
            robot=robot,
            joint_names=joint_names,
            gear_sign=robot.gear_sign,
            gear_ratio=robot.gear_ratio,
            home_pose=home_pose,
            current_q=np.asarray(q, dtype=np.float32),
            prev_q=np.asarray(q, dtype=np.float32).copy(),
            prev_eef_pos=np.zeros((3,), dtype=np.float32),
            prev_eef_R=np.eye(3, dtype=np.float32),
            base_pos=np.zeros((3,), dtype=np.float32),
            base_rpy=np.zeros((3,), dtype=np.float32),
            gripper_pos_steps=gripper_steps,
            gripper_min=gmin,
            gripper_max=gmax,
            gripper_step=50,
        )

        if not no_home:
            # Base pose will be refreshed after optional homing of both arms.
            self._refresh_arm_base_pose(arm)
        return arm

    def _refresh_arm_base_pose(self, arm: ArmRig) -> None:
        pose = arm.robot.fkine(arm.current_q)
        arm.base_pos = pose[:3, 3].astype(np.float32)
        arm.base_rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz").astype(np.float32)

    def _refresh_base_poses(self) -> None:
        self._refresh_arm_base_pose(self.right_arm)
        self._refresh_arm_base_pose(self.left_arm)

    def _controlled_arm(self) -> ArmRig:
        return self.right_arm if self.control_arm == "right" else self.left_arm

    def _reconnect_joycon(self) -> None:
        try:
            self.joycon.disconnnect()
            time.sleep(0.5)
        except Exception:
            pass
        self.joycon = JoyconRobotics(device=self.joycon_device, without_rest_init=False, common_rad=True, lerobot=False)

        if not self.no_robot:
            self._refresh_base_poses()

    def _start_new_episode(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.episode_idx += 1
        ep_path = self.out_dir / f"episode_{self.episode_idx:06d}.hdf5"
        self.writer = RDTHDF5EpisodeWriter(
            ep_path,
            timg=2,
            ncam=3,
            image_size=self.image_size,
            ta=self.ta,
            instruction=self.instruction,
            control_hz=self.control_hz,
        )

    def _toggle_recording(self) -> None:
        if not self.recording:
            self._start_new_episode()
            self.recording = True
            print(f"\n● Recording: ON  (episode {self.episode_idx:06d})")
        else:
            self.recording = False
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            print("\n● Recording: OFF")

    def _process_buttons(self) -> None:
        # Exit
        if getattr(self.joycon.button, "x", 0) == 1:
            self.running = False
            return

        # Toggle recording
        if getattr(self.joycon.button, "y", 0) == 1:
            self._toggle_recording()
            time.sleep(0.2)

        # New episode
        if getattr(self.joycon.button, "a", 0) == 1:
            self._start_new_episode()
            self.recording = True
            print(f"\n● New episode started: {self.episode_idx:06d}")
            time.sleep(0.2)

        # Home (both arms)
        if getattr(self.joycon.button, "home", 0) == 1 and not self.no_robot:
            print("\n🏠 Homing both arms + reconnect JoyCon...")
            if self.right_arm.controller is not None:
                self.right_arm.controller.move_all_home()
            if self.left_arm.controller is not None:
                self.left_arm.controller.move_all_home()
            time.sleep(1.0)

            self.right_arm.current_q = self.right_arm.robot.read_joint_angles(
                joint_names=self.right_arm.joint_names,
                home_pose=self.right_arm.home_pose,
                gear_sign=self.right_arm.gear_sign,
                gear_ratio=self.right_arm.gear_ratio,
                verbose=False,
            )
            self.right_arm.prev_q = self.right_arm.current_q.copy()
            self.left_arm.current_q = self.left_arm.robot.read_joint_angles(
                joint_names=self.left_arm.joint_names,
                home_pose=self.left_arm.home_pose,
                gear_sign=self.left_arm.gear_sign,
                gear_ratio=self.left_arm.gear_ratio,
                verbose=False,
            )
            self.left_arm.prev_q = self.left_arm.current_q.copy()

            self._reconnect_joycon()
            time.sleep(0.2)

        # Speed
        if getattr(self.joycon.button, "plus", 0) == 1:
            self.speed = min(self.speed + 100, 2000)
            time.sleep(0.15)
        if getattr(self.joycon.button, "minus", 0) == 1:
            self.speed = max(self.speed - 100, 200)
            time.sleep(0.15)

        # Gripper (controlled arm)
        arm = self._controlled_arm()
        if getattr(self.joycon.button, "zr", 0) == 1:
            arm.gripper_pos_steps = max(arm.gripper_pos_steps - arm.gripper_step, arm.gripper_min)
            if not self.no_robot and arm.controller is not None:
                arm.controller.move_servo("gripper", arm.gripper_pos_steps, self.speed)
            time.sleep(0.05)
        if getattr(self.joycon.button, "r", 0) == 1:
            arm.gripper_pos_steps = min(arm.gripper_pos_steps + arm.gripper_step, arm.gripper_max)
            if not self.no_robot and arm.controller is not None:
                arm.controller.move_servo("gripper", arm.gripper_pos_steps, self.speed)
            time.sleep(0.05)

        # Manual Z
        if getattr(self.joycon.button, "b", 0) == 1:
            self.z_offset += self.z_step
            time.sleep(0.05)

    def _update_image_history(self) -> np.ndarray:
        assert self.cams is not None

        # Missing views padded with background color
        fallback_bgr = np.full((self.image_size, self.image_size, 3), self.pad_bgr, dtype=np.uint8)

        for view in ("exterior", "right_wrist", "left_wrist"):
            frame_bgr = self.cams.read_bgr(view)
            if frame_bgr is None:
                rgb = cv2.cvtColor(fallback_bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb = pad_to_square_and_resize_rgb(frame_bgr, out_size=self.image_size, pad_bgr=self.pad_bgr)

            dq = self.hist[view]
            if len(dq) == 0:
                dq.append(rgb)
            dq.append(rgb)
            while len(dq) < 2:
                dq.append(dq[-1])

        # (Timg=2, Ncam=3, H, W, 3)
        imgs = np.stack(
            [
                np.stack(list(self.hist["exterior"]), axis=0),
                np.stack(list(self.hist["right_wrist"]), axis=0),
                np.stack(list(self.hist["left_wrist"]), axis=0),
            ],
            axis=1,
        )
        return imgs.astype(np.uint8)

    def _fill_arm_proprio(
        self,
        vec: UnifiedVector,
        *,
        arm: ArmRig,
        dt: float,
        joint_pos_sl: slice,
        joint_vel_sl: slice,
        eef_pos_sl: slice,
        eef_rot6d_sl: slice,
        eef_lin_vel_sl: slice,
        eef_ang_vel_sl: slice,
        gripper_pos_sl: slice,
    ) -> None:
        q = np.asarray(arm.current_q, dtype=np.float32).reshape(-1)
        qdot = ((q - arm.prev_q) / dt).astype(np.float32) if dt > 0 else np.zeros_like(q)

        pose = arm.robot.fkine(q)
        eef_pos = pose[:3, 3].astype(np.float32)
        eef_R = pose[:3, :3].astype(np.float32)

        eef_lin_vel = ((eef_pos - arm.prev_eef_pos) / dt).astype(np.float32) if dt > 0 else np.zeros((3,), dtype=np.float32)
        eef_ang_vel = compute_ang_vel_rad_s(arm.prev_eef_R, eef_R, dt)

        fill_slice(vec, joint_pos_sl, q)
        fill_slice(vec, joint_vel_sl, qdot)

        if arm.home_pose is not None and "gripper" in arm.home_pose:
            g_rad = gripper_steps_to_rad(arm.gripper_pos_steps, arm.home_pose["gripper"])
            fill_slice(vec, gripper_pos_sl, np.array([g_rad], dtype=np.float32))

        fill_slice(vec, eef_pos_sl, eef_pos)
        fill_slice(vec, eef_rot6d_sl, rotmat_to_rot6d(eef_R))
        fill_slice(vec, eef_lin_vel_sl, eef_lin_vel)
        fill_slice(vec, eef_ang_vel_sl, eef_ang_vel)

        arm.prev_q = q.copy()
        arm.prev_eef_pos = eef_pos.copy()
        arm.prev_eef_R = eef_R.copy()

    def _build_proprio(self, dt: float) -> UnifiedVector:
        vec = make_unified_vector()
        if self.no_robot:
            return vec

        self._fill_arm_proprio(
            vec,
            arm=self.right_arm,
            dt=dt,
            joint_pos_sl=RIGHT_ARM_JOINT_POS,
            joint_vel_sl=RIGHT_ARM_JOINT_VEL,
            eef_pos_sl=RIGHT_EEF_POS,
            eef_rot6d_sl=RIGHT_EEF_ROT6D,
            eef_lin_vel_sl=RIGHT_EEF_LIN_VEL,
            eef_ang_vel_sl=RIGHT_EEF_ANG_VEL,
            gripper_pos_sl=RIGHT_GRIPPER_POS,
        )
        self._fill_arm_proprio(
            vec,
            arm=self.left_arm,
            dt=dt,
            joint_pos_sl=LEFT_ARM_JOINT_POS,
            joint_vel_sl=LEFT_ARM_JOINT_VEL,
            eef_pos_sl=LEFT_EEF_POS,
            eef_rot6d_sl=LEFT_EEF_ROT6D,
            eef_lin_vel_sl=LEFT_EEF_LIN_VEL,
            eef_ang_vel_sl=LEFT_EEF_ANG_VEL,
            gripper_pos_sl=LEFT_GRIPPER_POS,
        )
        return vec

    def _build_action_joint_targets(self, q_target_right: np.ndarray, q_target_left: np.ndarray) -> UnifiedVector:
        vec = make_unified_vector()

        fill_slice(vec, RIGHT_ARM_JOINT_POS, np.asarray(q_target_right, dtype=np.float32))
        if self.right_arm.home_pose is not None and "gripper" in self.right_arm.home_pose:
            g_rad = gripper_steps_to_rad(self.right_arm.gripper_pos_steps, self.right_arm.home_pose["gripper"])
            fill_slice(vec, RIGHT_GRIPPER_POS, np.array([g_rad], dtype=np.float32))

        fill_slice(vec, LEFT_ARM_JOINT_POS, np.asarray(q_target_left, dtype=np.float32))
        if self.left_arm.home_pose is not None and "gripper" in self.left_arm.home_pose:
            g_rad = gripper_steps_to_rad(self.left_arm.gripper_pos_steps, self.left_arm.home_pose["gripper"])
            fill_slice(vec, LEFT_GRIPPER_POS, np.array([g_rad], dtype=np.float32))

        return vec

    def run(self) -> int:
        print("\n" + "=" * 70)
        print("RDT Teleop Dataset Collector")
        print("=" * 70)
        print("Controls: Y=record toggle, A=new episode, X=exit")
        print(f"Output: {self.out_dir}")

        try:
            last_t = time.time()
            while self.running:
                self._process_buttons()
                if not self.running:
                    break

                # JoyCon control
                pose, _gripper_status, _ = self.joycon.get_control()
                joycon_offset_pos = np.array([pose[0], pose[1], pose[2]], dtype=np.float32)
                # Keep consistent with `joycon_ik_control_py.py`:
                # roll/pitch are inverted for this robot's coordinate convention.
                joycon_offset_rpy = np.array([-pose[3], -pose[4], pose[5]], dtype=np.float32)
                joycon_offset_pos[2] += float(self.z_offset)

                controlled = self._controlled_arm()
                target_pos = controlled.base_pos + joycon_offset_pos
                target_rpy = controlled.base_rpy + joycon_offset_rpy
                T_goal = build_target_pose(*target_pos, *target_rpy)

                ik_success = True
                q_target_right = self.right_arm.current_q
                q_target_left = self.left_arm.current_q

                if not self.no_robot:
                    sol = controlled.robot.ikine_LM(
                        Tep=T_goal,
                        q0=controlled.current_q,
                        ilimit=50,
                        slimit=3,
                        tol=1e-3,
                        mask=[1, 1, 1, 0.8, 0.8, 0],
                        k=0.1,
                        method="sugihara",
                    )

                    if sol.success:
                        controlled.current_q = sol.q
                        if controlled.name == "right":
                            q_target_right = sol.q
                        else:
                            q_target_left = sol.q

                        if controlled.controller is not None:
                            servo_targets = controlled.robot.q_to_servo_targets(
                                controlled.current_q,
                                controlled.joint_names,
                                controlled.home_pose,
                                gear_ratio=controlled.gear_ratio,
                                gear_sign=controlled.gear_sign,
                            )
                            for k in controlled.joint_names:
                                servo_targets[k] = controlled.controller.limit_position(k, servo_targets[k])
                            servo_targets["gripper"] = controlled.gripper_pos_steps
                            controlled.controller.fast_move_to_pose(servo_targets, speed=self.speed)
                    else:
                        ik_success = False

                now = time.time()
                dt = max(1e-6, now - last_t)
                last_t = now

                if self.recording and self.writer is not None:
                    if self.no_camera:
                        images = np.zeros((2, 3, self.image_size, self.image_size, 3), dtype=np.uint8)
                    else:
                        images = self._update_image_history()

                    proprio = self._build_proprio(dt)
                    action = self._build_action_joint_targets(q_target_right, q_target_left)

                    self.writer.append_step(
                        images_timg_ncam=images,
                        proprio=proprio,
                        action=action,
                        timestamp_unix_s=now,
                        control_hz=self.control_hz,
                        ik_success=ik_success,
                    )

                time.sleep(self.dt)

        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

        return 0

    def _cleanup(self) -> None:
        try:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
        except Exception:
            pass

        try:
            self.joycon.disconnnect()
        except Exception:
            pass

        if self.cams is not None:
            self.cams.close()

        if not self.no_robot:
            for arm in (self.right_arm, self.left_arm):
                try:
                    if arm.controller is not None:
                        arm.controller.close()
                except Exception:
                    pass


def parse_bgr(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("pad_bgr must be like '0,0,0'")
    b, g, r = (int(x) for x in parts)
    for v in (b, g, r):
        if v < 0 or v > 255:
            raise argparse.ArgumentTypeError("pad_bgr values must be 0..255")
    return (b, g, r)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--out-dir", type=Path, default=Path("./rdt_finetune_hdf5"))
    ap.add_argument("--instruction", type=str, default="")
    ap.add_argument("--control-hz", type=float, default=25.0)
    ap.add_argument("--ta", type=int, default=64, help="Action chunk horizon Ta (saved on finalize)")
    ap.add_argument("--image-size", type=int, default=384)

    ap.add_argument("--device", choices=["right", "left"], default="right")
    ap.add_argument("--control-arm", choices=["right", "left"], default="right", help="Which arm the JoyCon controls")
    ap.add_argument("--right-port", type=str, default="/dev/right_arm")
    ap.add_argument("--left-port", type=str, default="/dev/left_arm")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--right-config", type=str, default="./driver/right_arm.json")
    ap.add_argument("--left-config", type=str, default="./driver/left_arm.json")

    # Backward-compatible aliases for single-arm scripts (map to right arm)
    ap.add_argument("--port", dest="deprecated_port", type=str, default=None, help="[deprecated] use --right-port")
    ap.add_argument("--config", dest="deprecated_config", type=str, default=None, help="[deprecated] use --right-config")
    ap.add_argument("--no-home", action="store_true")

    ap.add_argument("--cam-exterior", type=str, default=None, help="Camera index (e.g. '0') or video path")
    ap.add_argument("--cam-right-wrist", type=str, default=None)
    ap.add_argument("--cam-left-wrist", type=str, default=None)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--pad-bgr", type=parse_bgr, default=parse_bgr("0,0,0"))

    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--no-camera", action="store_true")

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="No hardware loop; writes a short dummy episode and exits.",
    )
    ap.add_argument("--dry-run-steps", type=int, default=10)

    args = ap.parse_args()

    right_port = args.right_port if args.deprecated_port is None else args.deprecated_port
    right_config = args.right_config if args.deprecated_config is None else args.deprecated_config

    if args.dry_run:
        out = args.out_dir
        out.mkdir(parents=True, exist_ok=True)
        ep_path = out / "episode_000001.hdf5"
        with RDTHDF5EpisodeWriter(
            ep_path,
            instruction=args.instruction,
            control_hz=args.control_hz,
            image_size=args.image_size,
            ta=args.ta,
        ) as w:
            images = np.zeros((2, 3, args.image_size, args.image_size, 3), dtype=np.uint8)
            for _ in range(int(args.dry_run_steps)):
                proprio = make_unified_vector()
                action = make_unified_vector()
                w.append_step(images_timg_ncam=images, proprio=proprio, action=action, control_hz=args.control_hz)
        print(f"Wrote dummy episode: {ep_path}")
        return 0

    collector = JoyConRDTCollector(
        device=args.device,
        control_arm=args.control_arm,
        right_port=right_port,
        left_port=args.left_port,
        baudrate=args.baudrate,
        right_config_path=right_config,
        left_config_path=args.left_config,
        out_dir=args.out_dir,
        instruction=args.instruction,
        control_hz=args.control_hz,
        ta=args.ta,
        image_size=args.image_size,
        cam_exterior=args.cam_exterior,
        cam_right_wrist=args.cam_right_wrist,
        cam_left_wrist=args.cam_left_wrist,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        pad_bgr=args.pad_bgr,
        no_robot=args.no_robot,
        no_camera=args.no_camera,
        no_home=args.no_home,
    )
    return collector.run()


if __name__ == "__main__":
    sys.exit(main())
