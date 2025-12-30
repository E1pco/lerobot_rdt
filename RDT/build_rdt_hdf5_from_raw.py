#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build an RDT fine-tuning HDF5 episode from a raw (CSV + JPG) episode folder.

Raw episode layout (produced by `collect_rdt_dataset_teleop.py --save-format raw`):
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
    ...

Outputs a single `episode_XXXXXX.hdf5` that matches `RDTHDF5EpisodeWriter` format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py  # noqa: F401  (ensures dependency exists)
import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f"OpenCV (cv2) is required to read JPG images: {e}")

# Ensure repo-root imports work when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdt_hdf5 import RDTHDF5EpisodeWriter, UnifiedVector


def _read_csv_matrix(path: Path, *, dtype: np.dtype, expected_cols: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return np.zeros((0, expected_cols), dtype=dtype)

    # Expect first row header.
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=dtype)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != expected_cols:
        raise ValueError(f"{path.name}: expected {expected_cols} cols, got {data.shape[1]}")
    return data


def _read_csv_vector(path: Path, *, dtype: np.dtype) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return np.zeros((0,), dtype=dtype)

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=dtype)
    if data.ndim == 0:
        data = data.reshape(1)
    return data


def _load_meta(ep_dir: Path) -> Dict:
    meta_path = ep_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.json: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _read_rgb_image(path: Path, *, image_size: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    if bgr.shape[0] != image_size or bgr.shape[1] != image_size or bgr.shape[2] != 3:
        raise ValueError(f"{path.name}: expected {image_size}x{image_size}x3, got {bgr.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def _build_images(ep_dir: Path, *, T: int, timg: int, ncam: int, image_size: int) -> np.ndarray:
    views = ["exterior", "right_wrist", "left_wrist"]
    if ncam != 3:
        raise ValueError("This builder assumes ncam=3")

    images = np.zeros((T, timg, ncam, image_size, image_size, 3), dtype=np.uint8)
    img_dir = ep_dir / "images"
    for t in range(T):
        for ti in range(timg):
            for ci, view in enumerate(views):
                p = img_dir / f"step_{t:06d}_timg{ti}_{view}.jpg"
                images[t, ti, ci] = _read_rgb_image(p, image_size=image_size)
    return images


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("raw_episode_dir", type=Path, help="episode_XXXXXX raw directory")
    ap.add_argument("--out", type=Path, default=None, help="output .hdf5 path (default: sibling episode_XXXXXX.hdf5)")
    args = ap.parse_args()

    ep_dir = args.raw_episode_dir
    if not ep_dir.exists() or not ep_dir.is_dir():
        print(f"Not a directory: {ep_dir}")
        return 2

    meta = _load_meta(ep_dir)

    timg = int(meta.get("timg", 2))
    ncam = int(meta.get("ncam", 3))
    image_size = int(meta.get("image_size", 384))
    ta = int(meta.get("ta", 64))
    instruction = str(meta.get("instruction", ""))
    control_hz = float(meta.get("control_hz", 25.0))

    proprio = _read_csv_matrix(ep_dir / "proprio.csv", dtype=np.float32, expected_cols=128)
    proprio_mask = _read_csv_matrix(ep_dir / "proprio_mask.csv", dtype=np.uint8, expected_cols=128)
    action = _read_csv_matrix(ep_dir / "action.csv", dtype=np.float32, expected_cols=128)
    action_mask = _read_csv_matrix(ep_dir / "action_mask.csv", dtype=np.uint8, expected_cols=128)

    ts = _read_csv_vector(ep_dir / "timestamps_unix_s.csv", dtype=np.float64)
    hz = _read_csv_vector(ep_dir / "control_frequency_hz.csv", dtype=np.float32)
    ik = _read_csv_vector(ep_dir / "ik_success.csv", dtype=np.uint8)

    T = int(proprio.shape[0])
    for name, arr in [
        ("proprio_mask", proprio_mask),
        ("action", action),
        ("action_mask", action_mask),
        ("timestamps", ts),
        ("control_frequency_hz", hz),
        ("ik_success", ik),
    ]:
        if int(arr.shape[0]) != T:
            raise ValueError(f"length mismatch: T={T} but {name} has {arr.shape[0]}")

    images = _build_images(ep_dir, T=T, timg=timg, ncam=ncam, image_size=image_size)

    if args.out is None:
        out_path = ep_dir.with_suffix(".hdf5")
    else:
        out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with RDTHDF5EpisodeWriter(
        out_path,
        timg=timg,
        ncam=ncam,
        image_size=image_size,
        ta=ta,
        instruction=instruction,
        control_hz=control_hz,
    ) as w:
        for t in range(T):
            p = UnifiedVector(value=np.asarray(proprio[t], dtype=np.float32), mask=np.asarray(proprio_mask[t], dtype=np.uint8))
            a = UnifiedVector(value=np.asarray(action[t], dtype=np.float32), mask=np.asarray(action_mask[t], dtype=np.uint8))
            w.append_step(
                images_timg_ncam=images[t],
                proprio=p,
                action=a,
                timestamp_unix_s=float(ts[t]),
                control_hz=float(hz[t]),
                ik_success=bool(int(ik[t]) != 0),
            )

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
