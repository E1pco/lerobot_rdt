"""RDT (arXiv:2410.07864v2) fine-tuning dataset utilities.

This module follows the paper's key format decisions:
- Observation: o_t := (X_{t-Timg+1:t+1}, z_t, c)
- 3-view RGB order: [exterior, right-wrist, left-wrist] (missing views padded)
- Image preprocessing: pad to square, resize to 384x384 (default)
- Unified action/proprio space: 128-dim (Table 4), with an availability mask.
- Fine-tuning storage: HDF5 (Appendix F).

It intentionally does NOT implement the paper's TFRecord pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


# Table 4 index mapping (half-open ranges)
RIGHT_ARM_JOINT_POS = slice(0, 10)
RIGHT_GRIPPER_POS = slice(10, 15)
RIGHT_ARM_JOINT_VEL = slice(15, 25)
RIGHT_GRIPPER_VEL = slice(25, 30)
RIGHT_EEF_POS = slice(30, 33)
RIGHT_EEF_ROT6D = slice(33, 39)
RIGHT_EEF_LIN_VEL = slice(39, 42)
RIGHT_EEF_ANG_VEL = slice(42, 45)

LEFT_ARM_JOINT_POS = slice(50, 60)
LEFT_GRIPPER_POS = slice(60, 65)
LEFT_ARM_JOINT_VEL = slice(65, 75)
LEFT_GRIPPER_VEL = slice(75, 80)
LEFT_EEF_POS = slice(80, 83)
LEFT_EEF_ROT6D = slice(83, 89)
LEFT_EEF_LIN_VEL = slice(89, 92)
LEFT_EEF_ANG_VEL = slice(92, 95)

BASE_LIN_VEL = slice(100, 102)
BASE_ANG_VEL = slice(102, 103)


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """6D rotation representation from a 3x3 rotation matrix.

    Uses the first two columns of R, flattened column-major to shape (6,).
    """
    if R.shape != (3, 3):
        raise ValueError(f"Expected (3,3) rotation matrix, got {R.shape}")
    return R[:, :2].reshape(6, order="F")


@dataclass
class UnifiedVector:
    value: np.ndarray  # (128,) float32
    mask: np.ndarray  # (128,) uint8 (0/1)


def make_unified_vector() -> UnifiedVector:
    return UnifiedVector(
        value=np.zeros((128,), dtype=np.float32),
        mask=np.zeros((128,), dtype=np.uint8),
    )


def fill_slice(vec: UnifiedVector, sl: slice, data: np.ndarray) -> None:
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    width = sl.stop - sl.start
    n = min(width, data.size)
    if n <= 0:
        return
    vec.value[sl.start : sl.start + n] = data[:n]
    vec.mask[sl.start : sl.start + n] = 1


class RDTHDF5EpisodeWriter:
    """Append-only HDF5 episode writer; finalizes action chunks on close."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        timg: int = 2,
        ncam: int = 3,
        image_size: int = 384,
        action_dim: int = 128,
        proprio_dim: int = 128,
        ta: int = 64,
        instruction: str = "",
        control_hz: float = 25.0,
        compression: Optional[str] = "gzip",
        compression_level: int = 4,
    ) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.timg = int(timg)
        self.ncam = int(ncam)
        self.image_size = int(image_size)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.ta = int(ta)
        self.instruction = instruction
        self.control_hz = float(control_hz)

        if self.timg <= 0:
            raise ValueError("timg must be > 0")
        if self.ncam != 3:
            raise ValueError("Paper config uses exactly 3 cameras (exterior, right-wrist, left-wrist)")
        if self.image_size <= 0:
            raise ValueError("image_size must be > 0")
        if self.action_dim != 128 or self.proprio_dim != 128:
            raise ValueError("Paper unified action/proprio space is 128-dim")
        if self.ta <= 0:
            raise ValueError("ta must be > 0")

        self._h5 = h5py.File(self.file_path, "w")

        meta = self._h5.require_group("meta")
        meta.attrs["created_unix_s"] = time.time()
        meta.attrs["instruction"] = instruction
        meta.attrs["timg"] = self.timg
        meta.attrs["ncam"] = self.ncam
        meta.attrs["image_size"] = self.image_size
        meta.attrs["action_dim"] = self.action_dim
        meta.attrs["proprio_dim"] = self.proprio_dim
        meta.attrs["ta"] = self.ta
        meta.attrs["control_hz"] = self.control_hz

        obs = self._h5.require_group("observations")
        act = self._h5.require_group("actions")

        comp = compression
        comp_opts = compression_level if compression == "gzip" else None

        self._images = obs.create_dataset(
            "images",
            shape=(0, self.timg, self.ncam, self.image_size, self.image_size, 3),
            maxshape=(None, self.timg, self.ncam, self.image_size, self.image_size, 3),
            dtype=np.uint8,
            chunks=(1, self.timg, self.ncam, self.image_size, self.image_size, 3),
            compression=comp,
            compression_opts=comp_opts,
        )
        self._proprio = obs.create_dataset(
            "proprio",
            shape=(0, self.proprio_dim),
            maxshape=(None, self.proprio_dim),
            dtype=np.float32,
            chunks=(1024, self.proprio_dim),
            compression=comp,
            compression_opts=comp_opts,
        )
        self._proprio_mask = obs.create_dataset(
            "proprio_mask",
            shape=(0, self.proprio_dim),
            maxshape=(None, self.proprio_dim),
            dtype=np.uint8,
            chunks=(1024, self.proprio_dim),
            compression=comp,
            compression_opts=comp_opts,
        )
        self._control_hz_ds = obs.create_dataset(
            "control_frequency_hz",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(4096,),
            compression=comp,
            compression_opts=comp_opts,
        )

        self._action = act.create_dataset(
            "action",
            shape=(0, self.action_dim),
            maxshape=(None, self.action_dim),
            dtype=np.float32,
            chunks=(1024, self.action_dim),
            compression=comp,
            compression_opts=comp_opts,
        )
        self._action_mask = act.create_dataset(
            "action_mask",
            shape=(0, self.action_dim),
            maxshape=(None, self.action_dim),
            dtype=np.uint8,
            chunks=(1024, self.action_dim),
            compression=comp,
            compression_opts=comp_opts,
        )

        self._timestamps = self._h5.create_dataset(
            "timestamps_unix_s",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(4096,),
            compression=comp,
            compression_opts=comp_opts,
        )
        self._ik_success = self._h5.create_dataset(
            "ik_success",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(4096,),
            compression=comp,
            compression_opts=comp_opts,
        )

    @property
    def length(self) -> int:
        return int(self._action.shape[0])

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

        t = self.length
        self._images.resize((t + 1, *self._images.shape[1:]))
        self._proprio.resize((t + 1, self.proprio_dim))
        self._proprio_mask.resize((t + 1, self.proprio_dim))
        self._control_hz_ds.resize((t + 1,))
        self._action.resize((t + 1, self.action_dim))
        self._action_mask.resize((t + 1, self.action_dim))
        self._timestamps.resize((t + 1,))
        self._ik_success.resize((t + 1,))

        self._images[t] = images
        self._proprio[t] = np.asarray(proprio.value, dtype=np.float32)
        self._proprio_mask[t] = np.asarray(proprio.mask, dtype=np.uint8)
        self._action[t] = np.asarray(action.value, dtype=np.float32)
        self._action_mask[t] = np.asarray(action.mask, dtype=np.uint8)
        self._timestamps[t] = float(time.time() if timestamp_unix_s is None else timestamp_unix_s)
        self._control_hz_ds[t] = float(self.control_hz if control_hz is None else control_hz)
        self._ik_success[t] = 1 if ik_success else 0

    def finalize_action_chunks(self) -> None:
        """Create action chunk datasets (T, Ta, 128) from per-step action."""
        T = self.length
        if T == 0:
            return

        act = self._h5["actions"]
        if "action_chunk" in act:
            return

        comp = self._action.compression
        comp_opts = self._action.compression_opts

        action_chunk = act.create_dataset(
            "action_chunk",
            shape=(T, self.ta, self.action_dim),
            dtype=np.float32,
            chunks=(1, self.ta, self.action_dim),
            compression=comp,
            compression_opts=comp_opts,
        )
        action_chunk_mask = act.create_dataset(
            "action_chunk_mask",
            shape=(T, self.ta, self.action_dim),
            dtype=np.uint8,
            chunks=(1, self.ta, self.action_dim),
            compression=comp,
            compression_opts=comp_opts,
        )

        action_all = self._action[...]
        mask_all = self._action_mask[...]

        for t in range(T):
            end = min(T, t + self.ta)
            chunk_len = end - t
            if chunk_len > 0:
                action_chunk[t, :chunk_len] = action_all[t:end]
                action_chunk_mask[t, :chunk_len] = mask_all[t:end]
            if chunk_len < self.ta:
                action_chunk[t, chunk_len:] = 0.0
                action_chunk_mask[t, chunk_len:] = 0

    def close(self) -> None:
        try:
            self.finalize_action_chunks()
        finally:
            self._h5.close()

    def __enter__(self) -> "RDTHDF5EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
