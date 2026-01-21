#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双主从臂遥操作数据采集脚本 (Leader-Follower Teleoperation)

用法示例:
    python3 RDT/collect_rdt_dataset_leader_follower.py \
        --left-leader-port /dev/left_leader --left-follower-port /dev/left_arm \
        --left-leader-config ./driver/left_arm_leader.json --left-follower-config ./driver/left_arm.json \
        --right-leader-port /dev/right_leader --right-follower-port /dev/right_arm \
        --right-leader-config ./driver/right_arm_leader.json --right-follower-config ./driver/right_arm.json \
        --cam-exterior 2 --cam-right-wrist 4 --cam-left-wrist 0 \
        --cam-width 640 --cam-height 480 --cam-backend v4l2 --cam-fourcc MJPG --cam-buffersize 1 \
        --save-format raw --out-dir ./rdt_raw \
        --preview --preview-timg 2

键盘控制:
    空格: 开始录制新 episode
    回车: 结束当前录制
    q/ESC: 退出
    h: 回中位
    +/-: 调整速度
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

# Ensure repo-root imports work when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

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


# ============================================================================
# Teleop Helper Functions (from dual_arm_teleop.py)
# ============================================================================

def arc_span(cfg: dict) -> int:
    """Return arc length on 12-bit circle (always from range_min to range_max)."""
    span = (cfg["range_max"] - cfg["range_min"]) % 4096
    return span or 4096


def clamp_on_arc(value: int, cfg: dict) -> int:
    """Clamp value to the configured arc (from range_min to range_max)."""
    rng_min = cfg["range_min"] % 4096
    span = arc_span(cfg)
    value = value % 4096
    offset = (value - rng_min) % 4096

    if offset <= span:
        return value

    rng_max = (rng_min + span) % 4096
    dist_to_min = (rng_min - value) % 4096
    dist_to_max = (value - rng_max) % 4096
    return rng_min if dist_to_min < dist_to_max else rng_max


def progress_from_position(value: int, cfg: dict) -> float:
    """将位置转换为 0-1 的进度值 (range_min=0, range_max=1)"""
    clamped = clamp_on_arc(value, cfg)
    rng_min = cfg["range_min"] % 4096
    span = arc_span(cfg)
    offset = (clamped - rng_min) % 4096
    return offset / span if span > 0 else 0.0


def position_from_progress(progress: float, cfg: dict) -> int:
    """将 0-1 的进度值转换为位置"""
    rng_min = cfg["range_min"] % 4096
    span = arc_span(cfg)
    progress = min(max(progress, 0.0), 1.0)
    return int(round((rng_min + progress * span) % 4096))


def shortest_delta(target: int, current: int, modulo: int = 4096) -> int:
    """计算环形空间上从 current 到 target 的最短有符号位移"""
    diff = (target - current) % modulo
    if diff > modulo // 2:
        diff -= modulo
    return diff


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


# ============================================================================
# Single Arm Teleop Class
# ============================================================================

class SingleArmTeleop:
    """单臂主从遥操作控制器（包含夹爪）"""

    def __init__(
        self,
        name: str,
        leader_controller,
        follower_controller,
        joint_names: List[str],
        alpha: float,
        speed: int,
    ):
        self.name = name
        self.leader = leader_controller
        self.follower = follower_controller
        # joint_names 包含所有关节（含夹爪）
        self.joint_names = joint_names
        self.alpha = alpha
        self.speed = speed

        self.filtered: Dict[str, float] = {}
        self.prev_leader_pos: Dict[str, int] = {}
        self.unwrapped_leader: Dict[str, int] = {}
        self.unwrapped_follower_current: Dict[str, int] = {}

        # 记录上一帧从臂位置（用于计算速度）
        self.prev_follower_pos: Dict[str, int] = {}
        
        # 夹爪当前步数（用于数据记录）
        self.gripper_pos_steps: int = follower_controller.home_pose.get("gripper", 2048)

    def step(self, debug: bool = False) -> Tuple[Dict[str, int], Dict[str, int], str]:
        """
        执行一步遥操作
        返回: (leader_positions, follower_targets, debug_string)
        """
        # 读取所有关节（包含夹爪）
        all_joint_names = self.joint_names + (["gripper"] if "gripper" not in self.joint_names else [])
        leader_pos = self.leader.read_servo_positions(all_joint_names)
        follower_pos_raw = self.follower.read_servo_positions(all_joint_names)
        targets = {}
        debug_info = []

        for name in self.joint_names:
            src_cfg = self.leader.config[name]
            dst_cfg = self.follower.config[name]

            # --- 1. 引导臂展开逻辑 ---
            raw_leader = leader_pos[name]
            if name not in self.prev_leader_pos:
                self.unwrapped_leader[name] = raw_leader
            else:
                delta = shortest_delta(raw_leader, self.prev_leader_pos[name])
                self.unwrapped_leader[name] += delta
            self.prev_leader_pos[name] = raw_leader

            # --- 2. 映射逻辑 ---
            progress = progress_from_position(self.unwrapped_leader[name], src_cfg)
            mapped_pos_0_4096 = position_from_progress(progress, dst_cfg)

            # --- 3. 从臂最短路径逻辑 ---
            current_raw = follower_pos_raw[name]
            if name not in self.unwrapped_follower_current:
                self.unwrapped_follower_current[name] = current_raw
            else:
                last_raw = self.unwrapped_follower_current[name] % 4096
                delta_feedback = shortest_delta(current_raw, last_raw)
                self.unwrapped_follower_current[name] += delta_feedback

            current_unwrapped = self.unwrapped_follower_current[name]
            diff_to_target = shortest_delta(mapped_pos_0_4096, current_unwrapped % 4096)
            target_unwrapped = current_unwrapped + diff_to_target

            # --- 4. 低通滤波 ---
            if name in self.filtered:
                smoothed = self.filtered[name] + self.alpha * (target_unwrapped - self.filtered[name])
            else:
                smoothed = float(target_unwrapped)

            self.filtered[name] = smoothed
            targets[name] = int(round(smoothed))

            if debug:
                debug_info.append(f"{name}:L{raw_leader}->T{targets[name] % 4096}")

        # 保存当前位置用于下一帧速度计算
        self.prev_follower_pos = follower_pos_raw.copy()

        # --- 夹爪映射 ---
        if "gripper" in leader_pos and "gripper" in self.leader.config and "gripper" in self.follower.config:
            gripper_name = "gripper"
            src_cfg = self.leader.config[gripper_name]
            dst_cfg = self.follower.config[gripper_name]
            
            raw_leader_gripper = leader_pos[gripper_name]
            if gripper_name not in self.prev_leader_pos:
                self.unwrapped_leader[gripper_name] = raw_leader_gripper
            else:
                delta = shortest_delta(raw_leader_gripper, self.prev_leader_pos[gripper_name])
                self.unwrapped_leader[gripper_name] += delta
            self.prev_leader_pos[gripper_name] = raw_leader_gripper
            
            progress = progress_from_position(self.unwrapped_leader[gripper_name], src_cfg)
            mapped_gripper = position_from_progress(progress, dst_cfg)
            
            current_raw_gripper = follower_pos_raw.get(gripper_name, mapped_gripper)
            if gripper_name not in self.unwrapped_follower_current:
                self.unwrapped_follower_current[gripper_name] = current_raw_gripper
            else:
                last_raw = self.unwrapped_follower_current[gripper_name] % 4096
                delta_feedback = shortest_delta(current_raw_gripper, last_raw)
                self.unwrapped_follower_current[gripper_name] += delta_feedback
            
            current_unwrapped_gripper = self.unwrapped_follower_current[gripper_name]
            diff_to_target = shortest_delta(mapped_gripper, current_unwrapped_gripper % 4096)
            target_unwrapped_gripper = current_unwrapped_gripper + diff_to_target
            
            if gripper_name in self.filtered:
                smoothed_gripper = self.filtered[gripper_name] + self.alpha * (target_unwrapped_gripper - self.filtered[gripper_name])
            else:
                smoothed_gripper = float(target_unwrapped_gripper)
            
            self.filtered[gripper_name] = smoothed_gripper
            targets[gripper_name] = int(round(smoothed_gripper))
            self.gripper_pos_steps = targets[gripper_name] % 4096
            
            if debug:
                debug_info.append(f"grip:L{raw_leader_gripper}->T{self.gripper_pos_steps}")

        # 发送指令
        self.follower.fast_move_to_pose(targets, speed=self.speed)

        debug_str = f"[{self.name}] " + " | ".join(debug_info) if debug else ""
        return leader_pos, targets, debug_str

    def get_follower_positions(self) -> Dict[str, int]:
        """获取当前从臂位置（0-4095），包含夹爪"""
        all_joint_names = self.joint_names + (["gripper"] if "gripper" not in self.joint_names else [])
        return self.follower.read_servo_positions(all_joint_names)


# ============================================================================
# Raw Episode Writer (from collect_rdt_dataset_teleop.py)
# ============================================================================

class RawEpisodeWriter:
    """原始数据写入器（CSV + JPG）"""

    def __init__(
        self,
        episode_dir: Path,
        *,
        instruction: str,
        control_hz: float,
        image_size: int,
        timg: int = 2,
        ncam: int = 3,
        ta: int = 64,
        jpg_quality: int = 95,
    ) -> None:
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.episode_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.timg = int(timg)
        self.ncam = int(ncam)
        self.image_size = int(image_size)
        self.ta = int(ta)
        self.instruction = str(instruction)
        self.control_hz = float(control_hz)
        self.jpg_quality = int(jpg_quality)
        self._t = 0

        meta = {
            "instruction": self.instruction,
            "timg": self.timg,
            "ncam": self.ncam,
            "image_size": self.image_size,
            "action_dim": 128,
            "proprio_dim": 128,
            "ta": self.ta,
            "control_hz": self.control_hz,
            "created_unix_s": float(time.time()),
        }
        (self.episode_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._fh_proprio = open(self.episode_dir / "proprio.csv", "w", newline="", encoding="utf-8")
        self._fh_proprio_mask = open(self.episode_dir / "proprio_mask.csv", "w", newline="", encoding="utf-8")
        self._fh_action = open(self.episode_dir / "action.csv", "w", newline="", encoding="utf-8")
        self._fh_action_mask = open(self.episode_dir / "action_mask.csv", "w", newline="", encoding="utf-8")
        self._fh_ts = open(self.episode_dir / "timestamps_unix_s.csv", "w", newline="", encoding="utf-8")
        self._fh_hz = open(self.episode_dir / "control_frequency_hz.csv", "w", newline="", encoding="utf-8")

        self._w_proprio = csv.writer(self._fh_proprio)
        self._w_proprio_mask = csv.writer(self._fh_proprio_mask)
        self._w_action = csv.writer(self._fh_action)
        self._w_action_mask = csv.writer(self._fh_action_mask)
        self._w_ts = csv.writer(self._fh_ts)
        self._w_hz = csv.writer(self._fh_hz)

        header_128 = [f"d{i}" for i in range(128)]
        self._w_proprio.writerow(header_128)
        self._w_proprio_mask.writerow(header_128)
        self._w_action.writerow(header_128)
        self._w_action_mask.writerow(header_128)
        self._w_ts.writerow(["timestamp_unix_s"])
        self._w_hz.writerow(["control_frequency_hz"])

        self._jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
        self._views = ["exterior", "right_wrist", "left_wrist"]

    @property
    def length(self) -> int:
        return int(self._t)

    def append_step(
        self,
        *,
        images_timg_ncam: np.ndarray,
        proprio: UnifiedVector,
        action: UnifiedVector,
        timestamp_unix_s: Optional[float] = None,
        control_hz: Optional[float] = None,
    ) -> None:
        images = np.asarray(images_timg_ncam, dtype=np.uint8)
        expected = (self.timg, self.ncam, self.image_size, self.image_size, 3)
        if images.shape != expected:
            raise ValueError(f"images must have shape {expected}, got {images.shape}")

        t = self._t
        for ti in range(self.timg):
            for ci in range(self.ncam):
                rgb = images[ti, ci]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                out = self.images_dir / f"step_{t:06d}_timg{ti}_{self._views[ci]}.jpg"
                cv2.imwrite(str(out), bgr, self._jpg_params)

        p = np.asarray(proprio.value, dtype=np.float32).reshape(128)
        pm = np.asarray(proprio.mask, dtype=np.uint8).reshape(128)
        a = np.asarray(action.value, dtype=np.float32).reshape(128)
        am = np.asarray(action.mask, dtype=np.uint8).reshape(128)

        self._w_proprio.writerow([f"{x:.9g}" for x in p.tolist()])
        self._w_proprio_mask.writerow([str(int(x)) for x in pm.tolist()])
        self._w_action.writerow([f"{x:.9g}" for x in a.tolist()])
        self._w_action_mask.writerow([str(int(x)) for x in am.tolist()])

        ts = float(time.time() if timestamp_unix_s is None else timestamp_unix_s)
        hz = float(self.control_hz if control_hz is None else control_hz)
        self._w_ts.writerow([f"{ts:.9f}"])
        self._w_hz.writerow([f"{hz:.6g}"])

        self._t += 1

    def close(self) -> None:
        for fh in (
            self._fh_proprio,
            self._fh_proprio_mask,
            self._fh_action,
            self._fh_action_mask,
            self._fh_ts,
            self._fh_hz,
        ):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass


# ============================================================================
# Camera Utilities
# ============================================================================

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
    """多视角相机管理器"""

    def __init__(
        self,
        *,
        exterior: Optional[str],
        right_wrist: Optional[str],
        left_wrist: Optional[str],
        width: int,
        height: int,
        backend: str = "auto",
        fourcc: Optional[str] = "MJPG",
        buffersize: int = 1,
    ) -> None:
        self._backend = str(backend)
        self._fourcc = None if fourcc is None else str(fourcc)
        self._buffersize = int(buffersize)

        self._caps: Dict[str, Optional[object]] = {
            "exterior": self._open(exterior, width, height),
            "right_wrist": self._open(right_wrist, width, height),
            "left_wrist": self._open(left_wrist, width, height),
        }

    def _open(self, source: Optional[str], width: int, height: int) -> Optional[object]:
        if source is None:
            return None

        try:
            idx = int(source)
            if self._backend == "v4l2":
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            elif self._backend == "gstreamer":
                cap = cv2.VideoCapture(idx, cv2.CAP_GSTREAMER)
            else:
                cap = cv2.VideoCapture(idx)
        except ValueError:
            cap = cv2.VideoCapture(source)

        try:
            if self._buffersize >= 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffersize)
        except Exception:
            pass

        if self._fourcc:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
            except Exception:
                pass

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


# ============================================================================
# Utilities
# ============================================================================

def _max_existing_episode_idx(out_dir: Path) -> int:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return 0

    pat = re.compile(r"^episode_(\d{6})(?:\.hdf5)?$")
    max_idx = 0
    for p in out_dir.iterdir():
        m = pat.match(p.name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx


def steps_to_radians(steps: int, home_steps: int) -> float:
    """将舵机步数转换为弧度（相对于 home 位置）"""
    counts_per_rad = 4096.0 / (2.0 * np.pi)
    return float((steps - home_steps) / counts_per_rad)


def compute_ang_vel_rad_s(R_prev: np.ndarray, R_cur: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros((3,), dtype=np.float32)
    dR = R_prev.T @ R_cur
    rotvec = R.from_matrix(dR).as_rotvec()
    return (rotvec / dt).astype(np.float32)


# ============================================================================
# Main Collector Class
# ============================================================================

@dataclass
class ArmState:
    """臂状态追踪"""
    name: str
    joint_names: List[str]
    home_pose: Dict[str, int]
    current_positions: Dict[str, int] = field(default_factory=dict)
    prev_positions: Dict[str, int] = field(default_factory=dict)
    prev_joint_rad: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))  # 6DOF
    prev_eef_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    prev_eef_R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))


class LeaderFollowerCollector:
    """主从遥操作数据采集器"""

    def __init__(
        self,
        *,
        # Left arm
        left_leader_port: str,
        left_follower_port: str,
        left_leader_config: str,
        left_follower_config: str,
        # Right arm
        right_leader_port: str,
        right_follower_port: str,
        right_leader_config: str,
        right_follower_config: str,
        # Common
        baudrate: int,
        alpha: float,
        speed: int,
        rate: float,
        # Output
        out_dir: Path,
        instruction: str,
        save_format: str,
        ta: int,
        image_size: int,
        # Camera
        cam_exterior: Optional[str],
        cam_right_wrist: Optional[str],
        cam_left_wrist: Optional[str],
        cam_width: int,
        cam_height: int,
        cam_backend: str,
        cam_fourcc: Optional[str],
        cam_buffersize: int,
        # Preview
        preview: bool,
        preview_scale: float,
        preview_timg: int,
        pad_bgr: Tuple[int, int, int],
        # Flags
        no_camera: bool,
        debug: bool,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.instruction = instruction
        self.save_format = save_format
        self.control_hz = float(rate)
        self.dt = 1.0 / self.control_hz if self.control_hz > 0 else 0.033
        self.ta = int(ta)
        self.image_size = int(image_size)
        self.pad_bgr = pad_bgr
        self.speed = speed
        self.alpha = alpha
        self.debug = debug

        self.no_camera = no_camera
        self.preview = bool(preview)
        self.preview_scale = float(preview_scale)
        self.preview_timg = int(preview_timg)

        self.running = True
        self.recording = False
        self.episode_idx = _max_existing_episode_idx(self.out_dir)

        # ---- Initialize Controllers ----
        from driver.ftservo_controller import ServoController

        print("Connecting to Left Arm...")
        self.left_leader = ServoController(
            port=left_leader_port, baudrate=baudrate, config_path=resolve_path(left_leader_config)
        )
        self.left_follower = ServoController(
            port=left_follower_port, baudrate=baudrate, config_path=resolve_path(left_follower_config)
        )

        print("Connecting to Right Arm...")
        self.right_leader = ServoController(
            port=right_leader_port, baudrate=baudrate, config_path=resolve_path(right_leader_config)
        )
        self.right_follower = ServoController(
            port=right_follower_port, baudrate=baudrate, config_path=resolve_path(right_follower_config)
        )

        # ---- Determine Joint Names ----
        self.left_joint_names = [
            n for n in self.left_leader.config.keys() if n in self.left_follower.config and n != "gripper"
        ]
        self.right_joint_names = [
            n for n in self.right_leader.config.keys() if n in self.right_follower.config and n != "gripper"
        ]

        # ---- Create Teleop Instances ----
        self.left_teleop = SingleArmTeleop(
            "Left", self.left_leader, self.left_follower, self.left_joint_names, alpha, speed
        )
        self.right_teleop = SingleArmTeleop(
            "Right", self.right_leader, self.right_follower, self.right_joint_names, alpha, speed
        )

        # ---- Arm State for Data Recording ----
        self.left_state = ArmState(
            name="left",
            joint_names=self.left_joint_names,
            home_pose=self.left_follower.home_pose,
        )
        self.right_state = ArmState(
            name="right",
            joint_names=self.right_joint_names,
            home_pose=self.right_follower.home_pose,
        )

        # ---- Camera ----
        self.cams = None if self.no_camera else MultiViewCamera(
            exterior=cam_exterior,
            right_wrist=cam_right_wrist,
            left_wrist=cam_left_wrist,
            width=cam_width,
            height=cam_height,
            backend=cam_backend,
            fourcc=cam_fourcc,
            buffersize=cam_buffersize,
        )

        # Image history buffers
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

        self.writer: Optional[object] = None
        self._preview_window = "Leader-Follower Teleop"

        print(f"\n=== Configuration ===")
        print(f"Left joints: {self.left_joint_names}")
        print(f"Right joints: {self.right_joint_names}")
        print(f"Control rate: {self.control_hz} Hz")
        print(f"Speed: {self.speed}, Alpha: {self.alpha}")
        print(f"Output: {self.out_dir}")
        print("=" * 30)

    def _start_new_episode(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        self.episode_idx += 1

        if self.save_format == "hdf5":
            ep_path = self.out_dir / f"episode_{self.episode_idx:06d}.hdf5"
            while ep_path.exists():
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
        else:
            ep_dir = self.out_dir / f"episode_{self.episode_idx:06d}"
            while ep_dir.exists():
                self.episode_idx += 1
                ep_dir = self.out_dir / f"episode_{self.episode_idx:06d}"
            self.writer = RawEpisodeWriter(
                ep_dir,
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

    def _process_keyboard(self, key: int) -> None:
        """处理键盘输入"""
        if key == ord("q") or key == 27:  # q or ESC
            self.running = False
        elif key == 32:  # 空格键: 开始录制新 episode
            if not self.recording:
                self._start_new_episode()
                self.recording = True
                print(f"\n● Recording started: episode_{self.episode_idx:06d}")
        elif key == 13 or key == 10:  # 回车键: 结束当前录制
            if self.recording:
                self.recording = False
                if self.writer is not None:
                    self.writer.close()
                    self.writer = None
                print(f"\n● Recording stopped: episode_{self.episode_idx:06d}")
        elif key == ord("h"):
            print("\nHoming all arms...")
            self.left_follower.move_all_home()
            self.right_follower.move_all_home()
            time.sleep(1.0)
        elif key == ord("+") or key == ord("="):
            self.speed = min(self.speed + 100, 2000)
            self.left_teleop.speed = self.speed
            self.right_teleop.speed = self.speed
            print(f"\nSpeed: {self.speed}")
        elif key == ord("-") or key == ord("_"):
            self.speed = max(self.speed - 100, 200)
            self.left_teleop.speed = self.speed
            self.right_teleop.speed = self.speed
            print(f"\nSpeed: {self.speed}")

    def _update_image_history(self) -> np.ndarray:
        assert self.cams is not None

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

        imgs = np.stack(
            [
                np.stack(list(self.hist["exterior"]), axis=0),
                np.stack(list(self.hist["right_wrist"]), axis=0),
                np.stack(list(self.hist["left_wrist"]), axis=0),
            ],
            axis=1,
        )
        return imgs.astype(np.uint8)

    def _positions_to_joint_rad(self, positions: Dict[str, int], home_pose: Dict[str, int]) -> np.ndarray:
        """将舵机步数转换为关节弧度（6DOF）"""
        # 6DOF 关节顺序：shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll
        joint_names_6dof = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
        joint_rad = []
        for name in joint_names_6dof:
            if name in positions and name in home_pose:
                rad = steps_to_radians(positions[name], home_pose[name])
                joint_rad.append(rad)
            else:
                joint_rad.append(0.0)
        return np.array(joint_rad, dtype=np.float32)

    def _build_proprio(self, dt: float) -> UnifiedVector:
        """构建本体感受向量"""
        vec = make_unified_vector()

        # Right arm
        right_pos = self.right_teleop.get_follower_positions()
        right_joint_rad = self._positions_to_joint_rad(right_pos, self.right_state.home_pose)
        right_joint_vel = (right_joint_rad - self.right_state.prev_joint_rad) / dt if dt > 0 else np.zeros(6)

        fill_slice(vec, RIGHT_ARM_JOINT_POS, right_joint_rad)
        fill_slice(vec, RIGHT_ARM_JOINT_VEL, right_joint_vel)

        # Gripper (使用 teleop 跟踪的夹爪位置)
        gripper_right = self.right_teleop.gripper_pos_steps
        gripper_right_rad = steps_to_radians(gripper_right, self.right_state.home_pose.get("gripper", 2048))
        fill_slice(vec, RIGHT_GRIPPER_POS, np.array([gripper_right_rad], dtype=np.float32))

        self.right_state.prev_joint_rad = right_joint_rad.copy()

        # Left arm
        left_pos = self.left_teleop.get_follower_positions()
        left_joint_rad = self._positions_to_joint_rad(left_pos, self.left_state.home_pose)
        left_joint_vel = (left_joint_rad - self.left_state.prev_joint_rad) / dt if dt > 0 else np.zeros(6)

        fill_slice(vec, LEFT_ARM_JOINT_POS, left_joint_rad)
        fill_slice(vec, LEFT_ARM_JOINT_VEL, left_joint_vel)

        # Gripper (使用 teleop 跟踪的夹爪位置)
        gripper_left = self.left_teleop.gripper_pos_steps
        gripper_left_rad = steps_to_radians(gripper_left, self.left_state.home_pose.get("gripper", 2048))
        fill_slice(vec, LEFT_GRIPPER_POS, np.array([gripper_left_rad], dtype=np.float32))

        self.left_state.prev_joint_rad = left_joint_rad.copy()

        return vec, right_joint_rad, left_joint_rad

    def _build_action(self, right_targets: Dict[str, int], left_targets: Dict[str, int]) -> UnifiedVector:
        """构建动作向量（目标关节位置）"""
        vec = make_unified_vector()

        # Right arm targets
        right_joint_rad = self._positions_to_joint_rad(right_targets, self.right_state.home_pose)
        fill_slice(vec, RIGHT_ARM_JOINT_POS, right_joint_rad)

        # Gripper (使用 teleop 跟踪的目标夹爪位置)
        gripper_right = self.right_teleop.gripper_pos_steps
        gripper_right_rad = steps_to_radians(gripper_right, self.right_state.home_pose.get("gripper", 2048))
        fill_slice(vec, RIGHT_GRIPPER_POS, np.array([gripper_right_rad], dtype=np.float32))

        # Left arm targets
        left_joint_rad = self._positions_to_joint_rad(left_targets, self.left_state.home_pose)
        fill_slice(vec, LEFT_ARM_JOINT_POS, left_joint_rad)

        # Gripper (使用 teleop 跟踪的目标夹爪位置)
        gripper_left = self.left_teleop.gripper_pos_steps
        gripper_left_rad = steps_to_radians(gripper_left, self.left_state.home_pose.get("gripper", 2048))
        fill_slice(vec, LEFT_GRIPPER_POS, np.array([gripper_left_rad], dtype=np.float32))

        return vec

    def _render_preview(
        self,
        images_rgb: np.ndarray,
        now_s: float,
        right_joint_rad: Optional[np.ndarray] = None,
        left_joint_rad: Optional[np.ndarray] = None,
        right_targets: Optional[Dict[str, int]] = None,
        left_targets: Optional[Dict[str, int]] = None,
    ) -> None:
        if not self.preview:
            return

        imgs = np.asarray(images_rgb, dtype=np.uint8)
        if imgs.ndim != 5 or imgs.shape[0] != 2 or imgs.shape[1] != 3:
            return

        timg, ncam, h, w, _ = imgs.shape

        if self.preview_timg == 1:
            row = []
            for ci in range(ncam):
                bgr = cv2.cvtColor(imgs[-1, ci], cv2.COLOR_RGB2BGR)
                row.append(bgr)
            canvas = np.concatenate(row, axis=1)
        else:
            tiles = []
            for ti in range(timg):
                row = []
                for ci in range(ncam):
                    bgr = cv2.cvtColor(imgs[ti, ci], cv2.COLOR_RGB2BGR)
                    row.append(bgr)
                tiles.append(np.concatenate(row, axis=1))
            canvas = np.concatenate(tiles, axis=0)

        if self.preview_scale != 1.0:
            canvas = cv2.resize(
                canvas,
                (int(canvas.shape[1] * self.preview_scale), int(canvas.shape[0] * self.preview_scale)),
                interpolation=cv2.INTER_AREA,
            )

        def fmt_vec(x: np.ndarray, prec: int = 3) -> str:
            arr = np.asarray(x).reshape(-1)
            return "[" + ",".join(f"{v:.{prec}f}" for v in arr.tolist()) + "]"

        def put(line: str, y: int, color=(255, 255, 255), scale: float = 0.55) -> None:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

        # Overlay global info
        step_idx = int(getattr(self.writer, "length", 0) or 0)
        put(f"rec={int(self.recording)} ep={self.episode_idx:06d} step={step_idx}", 22, (255, 255, 255), 0.6)
        put(f"speed={self.speed} save={self.save_format} hz={self.control_hz:.3g} time={now_s:.3f}", 44)

        controls = "SPACE:start ENTER:stop q:quit h:home +/-:speed"
        put(controls, 66, (0, 255, 255), 0.5)

        # 右臂数据
        y = 90
        try:
            if right_joint_rad is not None:
                right_grip = self.right_teleop.gripper_pos_steps
                put(
                    "R cur_q=" + fmt_vec(right_joint_rad, 3) + f" grip={right_grip}",
                    y,
                    (200, 255, 200),
                    0.5,
                )
                y += 18
            if right_targets is not None:
                right_tgt_rad = self._positions_to_joint_rad(right_targets, self.right_state.home_pose)
                put("R tgt_q=" + fmt_vec(right_tgt_rad, 3), y, (200, 255, 200), 0.5)
                y += 20
        except Exception:
            pass

        # 左臂数据
        try:
            if left_joint_rad is not None:
                left_grip = self.left_teleop.gripper_pos_steps
                put(
                    "L cur_q=" + fmt_vec(left_joint_rad, 3) + f" grip={left_grip}",
                    y,
                    (200, 200, 255),
                    0.5,
                )
                y += 18
            if left_targets is not None:
                left_tgt_rad = self._positions_to_joint_rad(left_targets, self.left_state.home_pose)
                put("L tgt_q=" + fmt_vec(left_tgt_rad, 3), y, (200, 200, 255), 0.5)
                y += 20
        except Exception:
            pass

        # Per-tile labels
        view_names = ["exterior", "right_wrist", "left_wrist"]
        tile_w = canvas.shape[1] // 3
        tile_h = canvas.shape[0] // (2 if self.preview_timg == 2 else 1)
        rows = 2 if self.preview_timg == 2 else 1
        for ti in range(rows):
            for ci in range(3):
                label = (f"timg{ti} " if rows == 2 else "") + view_names[ci]
                x0 = ci * tile_w + 10
                y0 = ti * tile_h + 20
                cv2.putText(canvas, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self._preview_window, canvas)

    def run(self) -> int:
        print("\n" + "=" * 70)
        print("Leader-Follower Teleop Dataset Collector")
        print("=" * 70)
        print("Controls: SPACE=start recording, ENTER=stop recording, q=quit, h=home, +/-=speed")
        print(f"Output: {self.out_dir}")

        last_t = time.time()
        debug_counter = 0

        try:
            while self.running:
                start_time = time.time()

                # ---- Step both arms ----
                _, right_targets, _ = self.right_teleop.step(debug=False)
                _, left_targets, _ = self.left_teleop.step(debug=False)

                now = time.time()
                dt = max(1e-6, now - last_t)
                last_t = now

                # ---- Update images ----
                if not self.no_camera:
                    images = self._update_image_history()
                else:
                    images = np.zeros((2, 3, self.image_size, self.image_size, 3), dtype=np.uint8)

                # ---- Build proprio (always, for preview) ----
                proprio, right_joint_rad, left_joint_rad = self._build_proprio(dt)
                action = self._build_action(right_targets, left_targets)

                # ---- Record data ----
                if self.recording and self.writer is not None:
                    self.writer.append_step(
                        images_timg_ncam=images,
                        proprio=proprio,
                        action=action,
                        timestamp_unix_s=now,
                        control_hz=self.control_hz,
                    )

                # ---- Preview ----
                if self.preview and not self.no_camera:
                    self._render_preview(
                        images, now,
                        right_joint_rad=right_joint_rad,
                        left_joint_rad=left_joint_rad,
                        right_targets=right_targets,
                        left_targets=left_targets,
                    )

                # ---- Keyboard handling ----
                key = cv2.waitKey(1) & 0xFF
                self._process_keyboard(key)

                # ---- Maintain loop rate ----
                elapsed = time.time() - start_time
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)

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

        if self.cams is not None:
            self.cams.close()

        if self.preview:
            try:
                cv2.destroyWindow(self._preview_window)
            except Exception:
                pass

        print("Closing connections...")
        for ctrl in (self.left_leader, self.left_follower, self.right_leader, self.right_follower):
            try:
                ctrl.close()
            except Exception:
                pass


# ============================================================================
# CLI
# ============================================================================

def parse_bgr(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("pad_bgr must be like '0,0,0'")
    b, g, r = (int(x) for x in parts)
    return (b, g, r)


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Dual arm leader-follower teleoperation dataset collector",
    )

    # Left Arm
    ap.add_argument("--left-leader-port", default="/dev/left_leader")
    ap.add_argument("--left-follower-port", default="/dev/left_arm")
    ap.add_argument("--left-leader-config", default="./driver/left_arm_leader.json")
    ap.add_argument("--left-follower-config", default="./driver/left_arm.json")

    # Right Arm
    ap.add_argument("--right-leader-port", default="/dev/right_leader")
    ap.add_argument("--right-follower-port", default="/dev/right_arm")
    ap.add_argument("--right-leader-config", default="./driver/right_arm_leader.json")
    ap.add_argument("--right-follower-config", default="./driver/right_arm.json")

    # Common
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--rate", type=float, default=30.0, help="Control loop rate (Hz)")
    ap.add_argument("--speed", type=int, default=1200, help="Servo speed for followers")
    ap.add_argument("--alpha", type=float, default=0.35, help="Low-pass smoothing factor 0-1")

    # Output
    ap.add_argument("--out-dir", type=Path, default=Path("./rdt_raw"))
    ap.add_argument("--instruction", type=str, default="")
    ap.add_argument("--save-format", choices=["raw", "hdf5"], default="raw")
    ap.add_argument("--ta", type=int, default=64, help="Action chunk horizon")
    ap.add_argument("--image-size", type=int, default=384)

    # Camera
    ap.add_argument("--cam-exterior", type=str, default=None)
    ap.add_argument("--cam-right-wrist", type=str, default=None)
    ap.add_argument("--cam-left-wrist", type=str, default=None)
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-backend", choices=["auto", "v4l2", "gstreamer"], default="v4l2")
    ap.add_argument("--cam-fourcc", type=str, default="MJPG")
    ap.add_argument("--cam-buffersize", type=int, default=1)

    # Preview
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-scale", type=float, default=1.0)
    ap.add_argument("--preview-timg", type=int, choices=[1, 2], default=1)
    ap.add_argument("--pad-bgr", type=parse_bgr, default=parse_bgr("0,0,0"))

    # Flags
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    collector = LeaderFollowerCollector(
        left_leader_port=args.left_leader_port,
        left_follower_port=args.left_follower_port,
        left_leader_config=args.left_leader_config,
        left_follower_config=args.left_follower_config,
        right_leader_port=args.right_leader_port,
        right_follower_port=args.right_follower_port,
        right_leader_config=args.right_leader_config,
        right_follower_config=args.right_follower_config,
        baudrate=args.baudrate,
        alpha=args.alpha,
        speed=args.speed,
        rate=args.rate,
        out_dir=args.out_dir,
        instruction=args.instruction,
        save_format=args.save_format,
        ta=args.ta,
        image_size=args.image_size,
        cam_exterior=args.cam_exterior,
        cam_right_wrist=args.cam_right_wrist,
        cam_left_wrist=args.cam_left_wrist,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        cam_backend=args.cam_backend,
        cam_fourcc=None if str(args.cam_fourcc).lower() == "none" else args.cam_fourcc,
        cam_buffersize=args.cam_buffersize,
        preview=args.preview,
        preview_scale=args.preview_scale,
        preview_timg=args.preview_timg,
        pad_bgr=args.pad_bgr,
        no_camera=args.no_camera,
        debug=args.debug,
    )
    return collector.run()


if __name__ == "__main__":
    sys.exit(main())
