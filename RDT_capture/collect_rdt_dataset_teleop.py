#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 python3 RDT/collect_rdt_dataset_teleop.py \
  --device-right right --device-left left \
  --right-port /dev/right_arm --left-port /dev/left_arm \
  --right-config ./driver/right_arm.json --left-config ./driver/left_arm.json \
  --cam-exterior 0 --cam-right-wrist 2 --cam-left-wrist 4 \
  --cam-width 640 --cam-height 480 --cam-backend v4l2 --cam-fourcc MJPG --cam-buffersize 1 \
  --save-format raw --out-dir ./rdt_raw \
  --preview --preview-timg 2

"""
from __future__ import annotations

import argparse
import csv
import json
import re
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
from ik.robot import create_so101_5dof

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


class RawEpisodeWriter:
    """

    Layout:
      episode_XXXXXX/
        meta.json
        proprio.csv
        proprio_mask.csv
        action.csv
        action_mask.csv
        timestamps_unix_s.csv
        control_frequency_hz.csv
        ik_success.csv
        images/step_000000_timg0_exterior.jpg
        ... (step × Timg=2 × view=3)

    Notes:
    - Images are saved as JPG in RGB->BGR conversion.
    - CSV files are append-only; header written once.
    """

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
        if cv2 is None:
            raise RuntimeError("RawEpisodeWriter requires OpenCV (cv2). Install opencv-python or use --save-format hdf5.")

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
        (self.episode_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        self._fh_proprio = open(self.episode_dir / "proprio.csv", "w", newline="", encoding="utf-8")
        self._fh_proprio_mask = open(self.episode_dir / "proprio_mask.csv", "w", newline="", encoding="utf-8")
        self._fh_action = open(self.episode_dir / "action.csv", "w", newline="", encoding="utf-8")
        self._fh_action_mask = open(self.episode_dir / "action_mask.csv", "w", newline="", encoding="utf-8")
        self._fh_ts = open(self.episode_dir / "timestamps_unix_s.csv", "w", newline="", encoding="utf-8")
        self._fh_hz = open(self.episode_dir / "control_frequency_hz.csv", "w", newline="", encoding="utf-8")
        self._fh_ik = open(self.episode_dir / "ik_success.csv", "w", newline="", encoding="utf-8")

        self._w_proprio = csv.writer(self._fh_proprio)
        self._w_proprio_mask = csv.writer(self._fh_proprio_mask)
        self._w_action = csv.writer(self._fh_action)
        self._w_action_mask = csv.writer(self._fh_action_mask)
        self._w_ts = csv.writer(self._fh_ts)
        self._w_hz = csv.writer(self._fh_hz)
        self._w_ik = csv.writer(self._fh_ik)

        header_128 = [f"d{i}" for i in range(128)]
        self._w_proprio.writerow(header_128)
        self._w_proprio_mask.writerow(header_128)
        self._w_action.writerow(header_128)
        self._w_action_mask.writerow(header_128)
        self._w_ts.writerow(["timestamp_unix_s"])
        self._w_hz.writerow(["control_frequency_hz"])
        self._w_ik.writerow(["ik_success"])

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
        ik_success: bool = True,
    ) -> None:
        images = np.asarray(images_timg_ncam, dtype=np.uint8)
        expected = (self.timg, self.ncam, self.image_size, self.image_size, 3)
        if images.shape != expected:
            raise ValueError(f"images must have shape {expected}, got {images.shape}")

        t = self._t
        # Save images as JPG
        for ti in range(self.timg):
            for ci in range(self.ncam):
                rgb = images[ti, ci]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                out = self.images_dir / f"step_{t:06d}_timg{ti}_{self._views[ci]}.jpg"
                ok = cv2.imwrite(str(out), bgr, self._jpg_params)
                if not ok:
                    raise RuntimeError(f"Failed to write image: {out}")

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
        self._w_ik.writerow(["1" if ik_success else "0"])

        self._t += 1

    def close(self) -> None:
        for fh in (
            self._fh_proprio,
            self._fh_proprio_mask,
            self._fh_action,
            self._fh_action_mask,
            self._fh_ts,
            self._fh_hz,
            self._fh_ik,
        ):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass


def _override_controller_home_pose_by_id(controller: object, *, home_by_id: Dict[int, int]) -> None:
    """Override ServoController.home_pose using servo IDs.

    The driver builds home_pose from config ranges by default. For calibrated robots,
    we sometimes want to force a measured home step per-servo.
    """

    # ServoController.config is a dict: {joint_name: {"id": int, ...}, ...}
    config = getattr(controller, "config", None)
    home_pose = getattr(controller, "home_pose", None)
    if not isinstance(config, dict) or not isinstance(home_pose, dict):
        raise TypeError("controller must have dict attributes: config and home_pose")

    for joint_name, cfg in config.items():
        try:
            sid = int(cfg.get("id"))
        except Exception:
            continue
        if sid in home_by_id:
            home_pose[joint_name] = int(home_by_id[sid])


@dataclass
class ArmRig:
    name: str  # "right" | "left"
    controller: Optional[object]
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
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for camera/image processing. Install opencv-python.")
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
        backend: str = "auto",
        fourcc: Optional[str] = "MJPG",
        buffersize: int = 1,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("MultiViewCamera requires OpenCV (cv2). Install opencv-python or run with --no-camera.")
        self._backend = str(backend)
        self._fourcc = None if fourcc is None else str(fourcc)
        self._buffersize = int(buffersize)

        if self._backend not in ("auto", "v4l2", "gstreamer"):
            raise ValueError("backend must be one of: auto|v4l2|gstreamer")

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

        # Reduce latency / memory where supported.
        try:
            if self._buffersize >= 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffersize)
        except Exception:
            pass

        # Prefer MJPG for USB cameras to reduce bandwidth/memory pressure.
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


def compute_ang_vel_rad_s(R_prev: np.ndarray, R_cur: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros((3,), dtype=np.float32)
    dR = R_prev.T @ R_cur
    rotvec = R.from_matrix(dR).as_rotvec()
    return (rotvec / dt).astype(np.float32)


def gripper_steps_to_rad(steps: int, home_steps: int) -> float:
    counts_per_rad = 4096.0 / (2.0 * np.pi)
    return float((steps - home_steps) / counts_per_rad)


def _max_existing_episode_idx(out_dir: Path) -> int:
    """Return max existing episode index in out_dir, or 0 if none.

    Matches both:
    - episode_000001/ (raw)
    - episode_000001.hdf5 (hdf5)
    """
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


class JoyConRDTCollector:
    def __init__(
        self,
        *,
        device_right: str,
        device_left: str,
        right_port: str,
        left_port: str,
        baudrate: int,
        right_config_path: str,
        left_config_path: str,
        out_dir: Path,
        instruction: str,
        save_format: str,
        control_hz: float,
        ta: int,
        image_size: int,
        cam_exterior: Optional[str],
        cam_right_wrist: Optional[str],
        cam_left_wrist: Optional[str],
        cam_width: int,
        cam_height: int,
        cam_backend: str,
        cam_fourcc: Optional[str],
        cam_buffersize: int,
        preview: bool,
        preview_scale: float,
        preview_timg: int,
        pad_bgr: Tuple[int, int, int],
        no_robot: bool,
        no_camera: bool,
        no_home: bool,
    ) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.instruction = instruction
        self.save_format = save_format
        if self.save_format not in ("raw", "hdf5"):
            raise ValueError("save_format must be 'raw' or 'hdf5'")
        self.control_hz = float(control_hz)
        self.dt = 1.0 / self.control_hz if self.control_hz > 0 else 0.04
        self.ta = int(ta)
        self.image_size = int(image_size)
        self.pad_bgr = pad_bgr

        self.no_robot = no_robot
        self.no_camera = no_camera

        self.preview = bool(preview)
        self.preview_scale = float(preview_scale)
        self.preview_timg = int(preview_timg)
        if self.preview_scale <= 0:
            raise ValueError("preview_scale must be > 0")
        if self.preview_timg not in (1, 2):
            raise ValueError("preview_timg must be 1 or 2")

        if self.preview and cv2 is None:
            raise RuntimeError("--preview requires OpenCV (cv2). Install opencv-python.")
        if self.preview and self.no_camera:
            raise RuntimeError("--preview requires cameras; remove --no-camera.")

        if (not self.no_camera) and cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for camera capture. Install opencv-python or pass --no-camera.")
        if self.save_format == "raw" and cv2 is None:
            raise RuntimeError("--save-format raw requires OpenCV (cv2). Install opencv-python or use --save-format hdf5.")

        self.speed = 800
        self.running = True
        self.recording = False
        # Continue numbering if out_dir already contains episodes.
        self.episode_idx = _max_existing_episode_idx(self.out_dir)

        self.z_offset = 0.0
        self.z_step = 0.005

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

        self.joycon_device_right = device_right
        self.joycon_device_left = device_left

        from joyconrobotics import JoyconRobotics
        self.joycon_right: Optional[object] = None
        self.joycon_left: Optional[object] = None
        self.joycon_right = JoyconRobotics(
            device=self.joycon_device_right, without_rest_init=False, common_rad=True, lerobot=False
        )
        self.joycon_left = JoyconRobotics(
            device=self.joycon_device_left, without_rest_init=False, common_rad=True, lerobot=False
        )

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

        self.writer: Optional[object] = None

        self._preview_window = "RDT Cameras"
        self._last_preview_bgr: Optional[np.ndarray] = None
        # IK telemetry for quick health checks without flooding stdout
        self._ik_stats = {
            "right": {"attempts": 0, "failures": 0},
            "left": {"attempts": 0, "failures": 0},
        }
        self._last_ik_log_ts = 0.0
        self._ik_log_interval_s = 2.0
        self._last_pose_log_ts = {"right": 0.0, "left": 0.0}
        self._pose_log_interval_s = 0.5

    def _render_preview(
        self,
        *,
        images_timg_ncam_rgb: np.ndarray,
        ik_success: bool,
        now_s: float,
        q_target_right: Optional[np.ndarray] = None,
        q_target_left: Optional[np.ndarray] = None,
    ) -> None:
        if not self.preview:
            return
        if cv2 is None:
            return

        imgs = np.asarray(images_timg_ncam_rgb, dtype=np.uint8)
        if imgs.shape[-1] != 3:
            return

        # Expect (Timg=2, Ncam=3, H, W, 3) RGB
        if imgs.ndim != 5 or imgs.shape[0] != 2 or imgs.shape[1] != 3:
            return

        timg, ncam, h, w, _ = imgs.shape

        # Convert to BGR and tile.
        # preview_timg=1: show latest frame only -> 1x3
        # preview_timg=2: show both frames -> 2x3
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

        # Optional scale for display performance.
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

        # Overlay global info.
        step_idx = int(getattr(self.writer, "length", 0) or 0)
        put(f"rec={int(self.recording)} ep={self.episode_idx:06d} step={step_idx}", 22, (255, 255, 255), 0.6)
        put(f"speed={self.speed} z_offset={self.z_offset:.4f} ik={int(bool(ik_success))}", 44)
        put(f"save={self.save_format} hz={self.control_hz:.3g} time={now_s:.3f}", 66)

        # Pose info (same semantics as dataset fields; readable subset).
        y = 90
        try:
            # Right arm
            pose_r = self.right_arm.robot.fkine(np.asarray(self.right_arm.current_q, dtype=np.float32).reshape(-1))
            eef_pos_r = pose_r[:3, 3].astype(np.float32)
            eef_rpy_r = R.from_matrix(pose_r[:3, :3]).as_euler("xyz").astype(np.float32)
            put(
                "R cur_q=" + fmt_vec(self.right_arm.current_q, 3)
                + f" grip={self.right_arm.gripper_pos_steps}",
                y,
                (200, 255, 200),
                0.5,
            )
            y += 18
            if q_target_right is not None:
                put("R tgt_q=" + fmt_vec(q_target_right, 3), y, (200, 255, 200), 0.5)
                y += 18
            put(
                "R eef_pos=" + fmt_vec(eef_pos_r, 3) + " eef_rpy=" + fmt_vec(eef_rpy_r, 3),
                y,
                (200, 255, 200),
                0.5,
            )
            y += 20

            # Left arm
            pose_l = self.left_arm.robot.fkine(np.asarray(self.left_arm.current_q, dtype=np.float32).reshape(-1))
            eef_pos_l = pose_l[:3, 3].astype(np.float32)
            eef_rpy_l = R.from_matrix(pose_l[:3, :3]).as_euler("xyz").astype(np.float32)
            put(
                "L cur_q=" + fmt_vec(self.left_arm.current_q, 3)
                + f" grip={self.left_arm.gripper_pos_steps}",
                y,
                (200, 200, 255),
                0.5,
            )
            y += 18
            if q_target_left is not None:
                put("L tgt_q=" + fmt_vec(q_target_left, 3), y, (200, 200, 255), 0.5)
                y += 18
            put(
                "L eef_pos=" + fmt_vec(eef_pos_l, 3) + " eef_rpy=" + fmt_vec(eef_rpy_l, 3),
                y,
                (200, 200, 255),
                0.5,
            )
        except Exception:
            # Don't crash preview due to pose formatting.
            pass

        # Per-tile labels.
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

        self._last_preview_bgr = canvas
        cv2.imshow(self._preview_window, canvas)
        cv2.waitKey(1)

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

        from driver.ftservo_controller import ServoController  # type: ignore
        controller = ServoController(port=port, baudrate=baudrate, config_path=config_path)

        # Force calibrated home steps (by servo ID) for consistent IK/teleop.
        if name == "left":
            _override_controller_home_pose_by_id(
                controller,
                home_by_id={1: 1988, 2: 1020, 3: 2951, 4: 1988, 5: 2017, 6: 2036},
            )
        elif name == "right":
            _override_controller_home_pose_by_id(
                controller,
                home_by_id={1: 2049, 2: 1061, 3: 3019, 4: 1983, 5: 2088, 6: 2036},
            )
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

    def _reconnect_joycons(self) -> None:
        from joyconrobotics import JoyconRobotics  # type: ignore

        for jc in (self.joycon_right, self.joycon_left):
            if jc is None:
                continue
            try:
                jc.disconnnect()
                time.sleep(0.2)
            except Exception:
                pass

        self.joycon_right = JoyconRobotics(
            device=self.joycon_device_right, without_rest_init=False, common_rad=True, lerobot=False
        )
        self.joycon_left = JoyconRobotics(
            device=self.joycon_device_left, without_rest_init=False, common_rad=True, lerobot=False
        )

        if not self.no_robot:
            self._refresh_base_poses()

    @staticmethod
    def _btn(jc: Optional[object], name: str) -> int:
        if jc is None:
            return 0
        try:
            return int(getattr(jc.button, name, 0) or 0)
        except Exception:
            return 0

    def _any_btn(self, name: str) -> bool:
        return bool(self._btn(self.joycon_right, name) or self._btn(self.joycon_left, name))

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

    def _process_buttons(self, button_control_right: Optional[int] = None) -> None:
        # Exit still allowed via X
        if self._any_btn("x"):
            self.running = False
            return

        # joycon-robotics mapped buttons (right JoyCon):
        # 1 -> start new episode + recording; -1 -> toggle recording
        if button_control_right == 1:
            self._start_new_episode()
            self.recording = True
            print(f"\n● New episode started: {self.episode_idx:06d}")
            time.sleep(0.2)
        elif button_control_right == -1:
            self._toggle_recording()
            time.sleep(0.2)

        # Home (both arms)
        if self._any_btn("home") and not self.no_robot:
            print("\nHoming both arms + reconnect JoyCon...")
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

            self._reconnect_joycons()
            time.sleep(0.2)

        # Speed
        if self._any_btn("plus"):
            self.speed = min(self.speed + 100, 2000)
            time.sleep(0.15)
        if self._any_btn("minus"):
            self.speed = max(self.speed - 100, 200)
            time.sleep(0.15)

        # Gripper (dual JoyCon)
        if self._btn(self.joycon_right, "zr") == 1:
            self.right_arm.gripper_pos_steps = max(
                self.right_arm.gripper_pos_steps - self.right_arm.gripper_step, self.right_arm.gripper_min
            )
            if not self.no_robot and self.right_arm.controller is not None:
                self.right_arm.controller.move_servo("gripper", self.right_arm.gripper_pos_steps, self.speed)
            time.sleep(0.05)
        if self._btn(self.joycon_right, "r") == 1:
            self.right_arm.gripper_pos_steps = min(
                self.right_arm.gripper_pos_steps + self.right_arm.gripper_step, self.right_arm.gripper_max
            )
            if not self.no_robot and self.right_arm.controller is not None:
                self.right_arm.controller.move_servo("gripper", self.right_arm.gripper_pos_steps, self.speed)
            time.sleep(0.05)

        if self._btn(self.joycon_left, "zl") == 1:
            self.left_arm.gripper_pos_steps = max(
                self.left_arm.gripper_pos_steps - self.left_arm.gripper_step, self.left_arm.gripper_min
            )
            if not self.no_robot and self.left_arm.controller is not None:
                self.left_arm.controller.move_servo("gripper", self.left_arm.gripper_pos_steps, self.speed)
            time.sleep(0.05)
        if self._btn(self.joycon_left, "l") == 1:
            self.left_arm.gripper_pos_steps = min(
                self.left_arm.gripper_pos_steps + self.left_arm.gripper_step, self.left_arm.gripper_max
            )
            if not self.no_robot and self.left_arm.controller is not None:
                self.left_arm.controller.move_servo("gripper", self.left_arm.gripper_pos_steps, self.speed)
            time.sleep(0.05)

        # Manual Z
        if self._any_btn("b"):
            self.z_offset += self.z_step
            print(f"Z+ {self.z_offset:.3f} m")
            time.sleep(0.05)
        # JoyconRobotics maps "down" on dpad; stick button is used for Z- when change_down_to_gripper=False.
        if self._any_btn("down") or self._any_btn("stick_r_btn") or self._any_btn("stick_l_btn"):
            self.z_offset -= self.z_step
            print(f"Z- {self.z_offset:.3f} m")
            time.sleep(0.05)

    def _log_ik_result(
        self,
        *,
        arm_name: str,
        success: bool,
        elapsed_s: float,
        goal_pos: np.ndarray,
        goal_rpy: np.ndarray,
        cur_pos: np.ndarray,
        cur_rpy: np.ndarray,
        residual: Optional[float],
        fallback_used: bool,
    ) -> None:
        stats = self._ik_stats[arm_name]
        stats["attempts"] += 1
        if not success:
            stats["failures"] += 1

        now = time.time()
        should_log = (not success) or (now - self._last_ik_log_ts >= self._ik_log_interval_s)
        if not should_log:
            return

        rate = 1.0 - (stats["failures"] / max(1, stats["attempts"]))
        res_str = "" if residual is None else f" res={residual:.3g}"
        fallback_str = " (fallback)" if fallback_used else ""
        print(
            f"[IK] {arm_name} {'OK' if success else 'FAIL'}{fallback_str} "
            f"rate={rate:.2%} took={elapsed_s*1e3:.1f}ms{res_str} "
            f"pos={np.asarray(goal_pos).round(4).tolist()} rpy={np.asarray(goal_rpy).round(4).tolist()}"
        )
        self._last_ik_log_ts = now

        # Throttled pose delta log for both arms to see responsiveness
        last_pose_ts = self._last_pose_log_ts.get(arm_name, 0.0)
        if now - last_pose_ts >= self._pose_log_interval_s:
            dpos = goal_pos - cur_pos
            drot = goal_rpy - cur_rpy
            print(
                f"[IKPOSE] {arm_name} cur_pos={np.asarray(cur_pos).round(4).tolist()} "
                f"tgt_pos={np.asarray(goal_pos).round(4).tolist()} dpos={np.asarray(dpos).round(4).tolist()} "
                f"cur_rpy={np.asarray(cur_rpy).round(4).tolist()} tgt_rpy={np.asarray(goal_rpy).round(4).tolist()} "
                f"drot={np.asarray(drot).round(4).tolist()}"
            )
            self._last_pose_log_ts[arm_name] = now

    def _solve_and_send(self, arm: ArmRig, T_goal: np.ndarray) -> bool:
        if self.no_robot:
            return True

        # Pre-filter: clamp workspace, limit per-step delta to help IK converge fast.
        goal_pos = T_goal[:3, 3].astype(np.float32)
        goal_rpy = R.from_matrix(T_goal[:3, :3]).as_euler("xyz").astype(np.float32)

        # Position clamp relative to base to stay inside reachable workspace.
        delta = goal_pos - arm.base_pos
        r_xy = float(np.linalg.norm(delta[:2]))
        r_xy_max = 0.82  # looser radial clamp to allow bigger sweeps
        if r_xy > r_xy_max and r_xy > 1e-6:
            delta[:2] *= r_xy_max / r_xy
        delta[2] = float(np.clip(delta[2], -0.08, 0.40))
        goal_pos = arm.base_pos + delta

        # Orientation clamp to avoid extreme angles that hurt convergence.
        goal_rpy = np.clip(goal_rpy, [-1.6, -1.6, -1.9], [1.6, 1.6, 1.9])

        # Limit per-step change vs current pose to reduce IK jumps.
        cur_pose = arm.robot.fkine(arm.current_q)
        cur_pos = cur_pose[:3, 3].astype(np.float32)
        cur_rpy = R.from_matrix(cur_pose[:3, :3]).as_euler("xyz").astype(np.float32)
        dpos = goal_pos - cur_pos
        drot = goal_rpy - cur_rpy
        max_step_pos = 0.2  # allow larger translational moves per cycle
        max_step_rot = 0.55  # allow larger rotational moves per cycle
        dpos_norm = float(np.linalg.norm(dpos))
        if dpos_norm > max_step_pos and dpos_norm > 1e-6:
            dpos *= max_step_pos / dpos_norm
        for i in range(3):
            if abs(drot[i]) > max_step_rot:
                drot[i] = np.sign(drot[i]) * max_step_rot
        goal_pos = cur_pos + dpos
        goal_rpy = cur_rpy + drot

        T_goal[:3, 3] = goal_pos
        T_goal[:3, :3] = R.from_euler("xyz", goal_rpy).as_matrix()
        start = time.time()
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

        success = bool(getattr(sol, "success", False))
        fallback_used = False
        if not success:
            # Second pass: fast position-priority solve (orientation dropped) to avoid long stalls
            sol = arm.robot.ikine_LM(
                Tep=T_goal,
                q0=arm.current_q,
                ilimit=50,
                slimit=3,
                tol=8e-3,
                mask=[1, 1, 1, 0, 0, 0],
                k=0.1,
                method="chan",
            )
            fallback_used = True
            success = bool(getattr(sol, "success", False))

        residual_raw = getattr(sol, "residual", None)
        residual = None
        try:
            if residual_raw is not None:
                residual = float(np.linalg.norm(residual_raw))
        except Exception:
            residual = None

        elapsed = time.time() - start
        self._log_ik_result(
            arm_name=arm.name,
            success=success,
            elapsed_s=elapsed,
            goal_pos=goal_pos,
            goal_rpy=goal_rpy,
            cur_pos=cur_pos,
            cur_rpy=cur_rpy,
            residual=residual,
            fallback_used=fallback_used,
        )

        if not success:
            return False

        arm.current_q = sol.q
        if arm.controller is not None:
            servo_targets = arm.robot.q_to_servo_targets(
                arm.current_q,
                arm.joint_names,
                arm.home_pose,
                gear_ratio=arm.gear_ratio,
                gear_sign=arm.gear_sign,
            )
            for k in arm.joint_names:
                servo_targets[k] = arm.controller.limit_position(k, servo_targets[k])
            servo_targets["gripper"] = arm.gripper_pos_steps
            arm.controller.fast_move_to_pose(servo_targets, speed=self.speed)

        return True

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
        print("Dual JoyCon: right->right arm, left->left arm")
        print(f"Output: {self.out_dir}")

        try:
            last_t = time.time()
            while self.running:
                # JoyCon control (dual-only)
                ik_success_right = True
                ik_success_left = True

                assert self.joycon_right is not None and self.joycon_left is not None

                pose_r, _gripper_status_r, button_control_r = self.joycon_right.get_control()
                pose_l, _gripper_status_l, _button_control_l = self.joycon_left.get_control()

                self._process_buttons(button_control_right=button_control_r)
                if not self.running:
                    break

                off_pos_r = np.array([pose_r[0], pose_r[1], pose_r[2]], dtype=np.float32)
                off_rpy_r = np.array([-pose_r[3], -pose_r[4], pose_r[5]], dtype=np.float32)
                off_pos_l = np.array([pose_l[0], pose_l[1], pose_l[2]], dtype=np.float32)
                off_rpy_l = np.array([-pose_l[3], -pose_l[4], pose_l[5]], dtype=np.float32)

                off_pos_r[2] += float(self.z_offset)
                off_pos_l[2] += float(self.z_offset)

                target_pos_r = self.right_arm.base_pos + off_pos_r
                target_rpy_r = self.right_arm.base_rpy + off_rpy_r
                T_goal_r = build_target_pose(*target_pos_r, *target_rpy_r)

                target_pos_l = self.left_arm.base_pos + off_pos_l
                target_rpy_l = self.left_arm.base_rpy + off_rpy_l
                T_goal_l = build_target_pose(*target_pos_l, *target_rpy_l)

                ik_success_right = self._solve_and_send(self.right_arm, T_goal_r)
                ik_success_left = self._solve_and_send(self.left_arm, T_goal_l)

                ik_success = bool(ik_success_right and ik_success_left)
                q_target_right = self.right_arm.current_q
                q_target_left = self.left_arm.current_q

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

                # Preview (show live even when not recording)
                if self.preview and not self.no_camera:
                    try:
                        images_prev = self._update_image_history()
                        self._render_preview(
                            images_timg_ncam_rgb=images_prev,
                            ik_success=ik_success,
                            now_s=now,
                            q_target_right=q_target_right,
                            q_target_left=q_target_left,
                        )
                    except Exception:
                        # Don't crash the control loop due to preview.
                        pass

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

        for jc in (self.joycon_right, self.joycon_left):
            if jc is None:
                continue
            try:
                jc.disconnnect()
            except Exception:
                pass

        if self.cams is not None:
            self.cams.close()

        if self.preview and cv2 is not None:
            try:
                cv2.destroyWindow(self._preview_window)
            except Exception:
                pass

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
    ap.add_argument("--control-hz", type=float, default=10.0)
    ap.add_argument("--ta", type=int, default=64, help="Action chunk horizon Ta (saved on finalize)")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--save-format", choices=["raw", "hdf5"], default="raw", help="Save raw CSV+JPG first, or directly save HDF5")

    ap.add_argument("--device-right", choices=["right", "left"], default="right", help="Right JoyCon side")
    ap.add_argument("--device-left", choices=["right", "left"], default="left", help="Left JoyCon side")
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
    ap.add_argument(
        "--cam-backend",
        choices=["auto", "v4l2", "gstreamer"],
        default="v4l2",
        help="Backend for numeric camera indices; v4l2 is usually more stable than gstreamer for /dev/video*",
    )
    ap.add_argument(
        "--cam-fourcc",
        type=str,
        default="MJPG",
        help="FOURCC for numeric camera indices (e.g. MJPG/YUYV). Use 'none' to disable.",
    )
    ap.add_argument(
        "--cam-buffersize",
        type=int,
        default=1,
        help="Capture buffer size (best-effort). Lower reduces latency/memory.",
    )

    ap.add_argument(
        "--preview",
        action="store_true",
        help="Show a live 2x3 (Timg=2 x Ncam=3) mosaic with overlay text.",
    )
    ap.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Scale factor for preview window (e.g. 0.75 for smaller window).",
    )
    ap.add_argument(
        "--preview-timg",
        type=int,
        choices=[1, 2],
        default=1,
        help="Preview temporal frames: 1 shows 3 tiles (latest per camera), 2 shows 6 tiles (Timg=2 x 3).",
    )
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
        images = np.zeros((2, 3, args.image_size, args.image_size, 3), dtype=np.uint8)
        ep_idx = _max_existing_episode_idx(out) + 1
        if args.save_format == "hdf5":
            ep_path = out / f"episode_{ep_idx:06d}.hdf5"
            while ep_path.exists():
                ep_idx += 1
                ep_path = out / f"episode_{ep_idx:06d}.hdf5"
            with RDTHDF5EpisodeWriter(
                ep_path,
                instruction=args.instruction,
                control_hz=args.control_hz,
                image_size=args.image_size,
                ta=args.ta,
            ) as w:
                for _ in range(int(args.dry_run_steps)):
                    proprio = make_unified_vector()
                    action = make_unified_vector()
                    w.append_step(images_timg_ncam=images, proprio=proprio, action=action, control_hz=args.control_hz)
            print(f"Wrote dummy episode: {ep_path}")
        else:
            ep_dir = out / f"episode_{ep_idx:06d}"
            while ep_dir.exists():
                ep_idx += 1
                ep_dir = out / f"episode_{ep_idx:06d}"
            w = RawEpisodeWriter(
                ep_dir,
                instruction=args.instruction,
                control_hz=args.control_hz,
                image_size=args.image_size,
                ta=args.ta,
                timg=2,
                ncam=3,
            )
            try:
                for _ in range(int(args.dry_run_steps)):
                    proprio = make_unified_vector()
                    action = make_unified_vector()
                    w.append_step(images_timg_ncam=images, proprio=proprio, action=action, control_hz=args.control_hz)
            finally:
                w.close()
            print(f"Wrote dummy raw episode: {ep_dir}")
        return 0

    collector = JoyConRDTCollector(
        device_right=args.device_right,
        device_left=args.device_left,
        right_port=right_port,
        left_port=args.left_port,
        baudrate=args.baudrate,
        right_config_path=right_config,
        left_config_path=args.left_config,
        out_dir=args.out_dir,
        instruction=args.instruction,
        save_format=args.save_format,
        control_hz=args.control_hz,
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
        no_robot=args.no_robot,
        no_camera=args.no_camera,
        no_home=args.no_home,
    )
    return collector.run()


if __name__ == "__main__":
    sys.exit(main())
