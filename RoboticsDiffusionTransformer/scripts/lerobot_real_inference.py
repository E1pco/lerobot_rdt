#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 实际机器人推理脚本（无 ROS）。

用法示例:
python3 RoboticsDiffusionTransformer/scripts/lerobot_real_inference.py \
    --checkpoint ./checkpoints/rdt-finetune-lerobot/checkpoint-1620 \
    --vision-encoder ./models/siglip-so400m-patch14-384 \
    --lang-embed ./data/datasets/lerobot/lerobot_task.pt \
    --left-port /dev/left_arm --left-config ./driver/left_arm.json \
    --right-port /dev/right_arm --right-config ./driver/right_arm.json \
    --cam-exterior 2 --cam-right-wrist 4 --cam-left-wrist 0 \
    --cam-width 640 --cam-height 480 --cam-backend v4l2 --cam-fourcc MJPG --cam-buffersize 1 \
    --rate 30 --speed 1200 --image-size 384 --preview
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Ensure repo-root imports work when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from driver.ftservo_controller import ServoController
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from RDT.rdt_hdf5 import (
    UnifiedVector,
    fill_slice,
    make_unified_vector,
    RIGHT_ARM_JOINT_POS,
    RIGHT_GRIPPER_POS,
    LEFT_ARM_JOINT_POS,
    LEFT_GRIPPER_POS,
)


# ===== Camera utils (from collector) =====

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


# ===== Kinematics helpers =====

def steps_to_radians(steps: int, home_steps: int) -> float:
    counts_per_rad = 4096.0 / (2.0 * np.pi)
    return float((steps - home_steps) / counts_per_rad)


def radians_to_steps(rad: float, home_steps: int) -> int:
    counts_per_rad = 4096.0 / (2.0 * np.pi)
    return int(round(rad * counts_per_rad + home_steps))


# ===== Main inference loop =====

class RealRobotInferencer:
    def __init__(
        self,
        *,
        checkpoint: str,
        vision_encoder: str,
        lang_embed: str,
        left_port: str,
        left_config: str,
        right_port: str,
        right_config: str,
        baudrate: int,
        rate: float,
        speed: int,
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
        pad_bgr: Tuple[int, int, int],
        device: str,
    ) -> None:
        self.rate = float(rate)
        self.dt = 1.0 / self.rate if self.rate > 0 else 0.033
        self.speed = int(speed)
        self.image_size = int(image_size)
        self.preview = bool(preview)
        self.pad_bgr = pad_bgr

        # Servo controllers
        self.left = ServoController(port=left_port, baudrate=baudrate, config_path=left_config)
        self.right = ServoController(port=right_port, baudrate=baudrate, config_path=right_config)

        # Joint names (6DOF order must match training)
        self.joint_names_6dof = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
        ]

        # Camera
        self.cams = MultiViewCamera(
            exterior=cam_exterior,
            right_wrist=cam_right_wrist,
            left_wrist=cam_left_wrist,
            width=cam_width,
            height=cam_height,
            backend=cam_backend,
            fourcc=cam_fourcc,
            buffersize=cam_buffersize,
        )
        self.hist: Dict[str, Deque[np.ndarray]] = {
            "exterior": deque(maxlen=2),
            "right_wrist": deque(maxlen=2),
            "left_wrist": deque(maxlen=2),
        }

        # Load model
        self.device = torch.device(device)
        self.rdt = RDTRunner.from_pretrained(checkpoint, dtype=torch.float32)
        self.rdt.to(self.device)
        self.rdt.eval()

        self.vision = SiglipVisionTower(vision_tower=vision_encoder, args=None)
        self.vision.vision_tower.to(self.device, dtype=torch.float32)
        self.image_processor = self.vision.image_processor

        # Load precomputed language embedding
        lang_obj = torch.load(lang_embed, weights_only=False)
        if isinstance(lang_obj, dict) and "embeddings" in lang_obj:
            self.lang_embed = lang_obj["embeddings"]
        else:
            self.lang_embed = lang_obj
        if self.lang_embed.ndim == 2:
            self.lang_embed = self.lang_embed.unsqueeze(0)  # (1, L, D)
        self.lang_attn_mask = torch.ones(self.lang_embed.shape[:2], dtype=torch.bool)

        # State tracking
        self.prev_left_rad = np.zeros(6, dtype=np.float32)
        self.prev_right_rad = np.zeros(6, dtype=np.float32)

    def _read_positions(self, ctrl: ServoController) -> Dict[str, int]:
        names = self.joint_names_6dof + ["gripper"]
        return ctrl.read_servo_positions(names)

    def _positions_to_joint_rad(self, positions: Dict[str, int], home_pose: Dict[str, int]) -> np.ndarray:
        joint_rad = []
        for name in self.joint_names_6dof:
            if name in positions and name in home_pose:
                joint_rad.append(steps_to_radians(positions[name], home_pose[name]))
            else:
                joint_rad.append(0.0)
        return np.asarray(joint_rad, dtype=np.float32)

    def _build_state_vec(self, dt: float) -> Tuple[UnifiedVector, np.ndarray, np.ndarray]:
        vec = make_unified_vector()

        right_pos = self._read_positions(self.right)
        right_joint_rad = self._positions_to_joint_rad(right_pos, self.right.home_pose)
        right_vel = (right_joint_rad - self.prev_right_rad) / dt if dt > 0 else np.zeros(6)
        self.prev_right_rad = right_joint_rad.copy()

        fill_slice(vec, RIGHT_ARM_JOINT_POS, right_joint_rad)
        fill_slice(vec, RIGHT_GRIPPER_POS, np.array([steps_to_radians(right_pos.get("gripper", 2048), self.right.home_pose.get("gripper", 2048))], dtype=np.float32))

        left_pos = self._read_positions(self.left)
        left_joint_rad = self._positions_to_joint_rad(left_pos, self.left.home_pose)
        left_vel = (left_joint_rad - self.prev_left_rad) / dt if dt > 0 else np.zeros(6)
        self.prev_left_rad = left_joint_rad.copy()

        fill_slice(vec, LEFT_ARM_JOINT_POS, left_joint_rad)
        fill_slice(vec, LEFT_GRIPPER_POS, np.array([steps_to_radians(left_pos.get("gripper", 2048), self.left.home_pose.get("gripper", 2048))], dtype=np.float32))

        # Velocities are currently unused in training data; keep zeroed to match collected format
        _ = right_vel, left_vel
        return vec, right_joint_rad, left_joint_rad

    def _update_image_history(self) -> np.ndarray:
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

    def _preprocess_images(self, imgs: np.ndarray) -> torch.Tensor:
        # imgs: (timg=2, ncam=3, H, W, 3)
        preprocessed = []
        for ti in range(imgs.shape[0]):
            for ci in range(imgs.shape[1]):
                image = Image.fromarray(imgs[ti, ci])
                image = transforms.Resize(self.image_size)(image)
                image = self.image_processor.preprocess(image, return_tensors='pt')["pixel_values"][0]
                preprocessed.append(image)
        return torch.stack(preprocessed, dim=0)

    def _apply_action(self, action_vec: np.ndarray) -> None:
        # action_vec: (128,)
        right_q = action_vec[RIGHT_ARM_JOINT_POS]
        right_grip = action_vec[RIGHT_GRIPPER_POS][0] if (RIGHT_GRIPPER_POS.stop - RIGHT_GRIPPER_POS.start) > 0 else 0.0
        left_q = action_vec[LEFT_ARM_JOINT_POS]
        left_grip = action_vec[LEFT_GRIPPER_POS][0] if (LEFT_GRIPPER_POS.stop - LEFT_GRIPPER_POS.start) > 0 else 0.0

        right_targets = {
            name: radians_to_steps(right_q[i], self.right.home_pose.get(name, 2048))
            for i, name in enumerate(self.joint_names_6dof)
        }
        right_targets["gripper"] = radians_to_steps(right_grip, self.right.home_pose.get("gripper", 2048))

        left_targets = {
            name: radians_to_steps(left_q[i], self.left.home_pose.get(name, 2048))
            for i, name in enumerate(self.joint_names_6dof)
        }
        left_targets["gripper"] = radians_to_steps(left_grip, self.left.home_pose.get("gripper", 2048))

        self.right.fast_move_to_pose(right_targets, speed=self.speed)
        self.left.fast_move_to_pose(left_targets, speed=self.speed)

    def run(self) -> int:
        print("\n=== Real Robot Inference ===")
        print("Press Ctrl+C to stop.")
        last_t = time.time()

        try:
            while True:
                start_time = time.time()
                now = time.time()
                dt = max(1e-6, now - last_t)
                last_t = now

                imgs = self._update_image_history()

                if self.preview:
                    # imgs shape: (Time=2, Cam=3, H, W, C=3) (RGB)
                    # Take latest timestep, concat cameras horizontally
                    latest = imgs[-1]
                    preview_canvas = np.concatenate([latest[0], latest[1], latest[2]], axis=1)
                    # RGB -> BGR for OpenCV
                    preview_canvas = cv2.cvtColor(preview_canvas, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Robot Vision", preview_canvas)
                    cv2.waitKey(1)

                image_tensor = self._preprocess_images(imgs).to(self.device, dtype=torch.float32)

                image_embeds = self.vision(image_tensor).detach()
                image_embeds = image_embeds.reshape(1, -1, self.vision.hidden_size)

                state_vec, _, _ = self._build_state_vec(dt)
                states = torch.from_numpy(state_vec.value).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                action_mask = torch.from_numpy(state_vec.mask).unsqueeze(0).to(self.device, dtype=torch.float32)
                ctrl_freqs = torch.tensor([self.rate], device=self.device)

                with torch.inference_mode():
                    pred = self.rdt.predict_action(
                        lang_tokens=self.lang_embed.to(self.device, dtype=torch.float32),
                        lang_attn_mask=self.lang_attn_mask.to(self.device),
                        img_tokens=image_embeds,
                        state_tokens=states,
                        action_mask=action_mask.unsqueeze(1),
                        ctrl_freqs=ctrl_freqs,
                    )
                action_vec = pred[0, 0].float().cpu().numpy()
                self._apply_action(action_vec)

                # Maintain control rate
                elapsed = time.time() - start_time
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
        except KeyboardInterrupt:
            return 0
        finally:
            self.cams.close()
            for ctrl in (self.left, self.right):
                try:
                    ctrl.close()
                except Exception:
                    pass


def parse_bgr(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("pad_bgr must be like '0,0,0'")
    b, g, r = (int(x) for x in parts)
    return (b, g, r)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--checkpoint", required=True, help="Checkpoint dir, e.g. ./checkpoints/rdt-finetune-lerobot/checkpoint-1620")
    ap.add_argument("--vision-encoder", required=True, help="SigLIP vision encoder path")
    ap.add_argument("--lang-embed", required=True, help="Precomputed language embedding .pt")

    ap.add_argument("--left-port", default="/dev/left_arm")
    ap.add_argument("--left-config", default="./driver/left_arm.json")
    ap.add_argument("--right-port", default="/dev/right_arm")
    ap.add_argument("--right-config", default="./driver/right_arm.json")
    ap.add_argument("--baudrate", type=int, default=1_000_000)

    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--speed", type=int, default=1200)
    ap.add_argument("--image-size", type=int, default=384)

    ap.add_argument("--cam-exterior", type=str, default=None)
    ap.add_argument("--cam-right-wrist", type=str, default=None)
    ap.add_argument("--cam-left-wrist", type=str, default=None)
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-backend", choices=["auto", "v4l2", "gstreamer"], default="v4l2")
    ap.add_argument("--cam-fourcc", type=str, default="MJPG")
    ap.add_argument("--cam-buffersize", type=int, default=1)
    ap.add_argument("--pad-bgr", type=parse_bgr, default=parse_bgr("0,0,0"))

    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    infer = RealRobotInferencer(
        checkpoint=args.checkpoint,
        vision_encoder=args.vision_encoder,
        lang_embed=args.lang_embed,
        left_port=args.left_port,
        left_config=args.left_config,
        right_port=args.right_port,
        right_config=args.right_config,
        baudrate=args.baudrate,
        rate=args.rate,
        speed=args.speed,
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
        pad_bgr=args.pad_bgr,
        device=args.device,
    )
    return infer.run()


if __name__ == "__main__":
    sys.exit(main())
