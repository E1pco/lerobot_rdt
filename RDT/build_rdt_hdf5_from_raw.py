#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build an RDT fine-tuning HDF5 episode from a raw (CSV + JPG) episode folder.

生成与 RDT 官方 hdf5_vla_dataset.py 兼容的 HDF5 格式：
  - observations/qpos: (T, 128) 本体感知状态
  - observations/images/cam_high: 压缩 JPEG 图像列表
  - observations/images/cam_right_wrist: 压缩 JPEG 图像列表  
  - observations/images/cam_left_wrist: 压缩 JPEG 图像列表
  - action: (T, 128) 动作

Raw episode layout (produced by `collect_rdt_dataset_teleop.py --save-format raw`):
  episode_XXXXXX/
    meta.json
    proprio.csv
    proprio_mask.csv
    action.csv
    action_mask.csv
    timestamps_unix_s.csv
    control_frequency_hz.csv
    images/step_000000_timg0_exterior.jpg
    ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

try:
    import cv2
except Exception as e:
    raise SystemExit(f"OpenCV (cv2) is required: {e}")

REPO_ROOT = Path(__file__).resolve().parents[1]

# ============== 配置区域 ==============
DATASET_NAME = "lerobot"
INSTRUCTION = "First, use your right hand to place the green cuboid into the black box, then use your left hand to take it out of the brown box."
# =====================================


def _read_csv_matrix(path: Path, *, dtype: np.dtype, expected_cols: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return np.zeros((0, expected_cols), dtype=dtype)
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


def _read_and_encode_image(path: Path) -> bytes:
    """读取图像并编码为 JPEG 字节"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return encoded.tobytes()


def _build_episode_hdf5(ep_dir: Path, out_path: Path) -> None:
    """构建与 RDT hdf5_vla_dataset.py 兼容的 HDF5 文件"""
    meta = _load_meta(ep_dir)

    timg = int(meta.get("timg", 2))

    # 读取 CSV 数据
    proprio = _read_csv_matrix(ep_dir / "proprio.csv", dtype=np.float32, expected_cols=128)
    action = _read_csv_matrix(ep_dir / "action.csv", dtype=np.float32, expected_cols=128)

    T = int(proprio.shape[0])
    
    # 相机视图名称映射
    views = ["exterior", "right_wrist", "left_wrist"]
    cam_keys = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, 'w') as f:
        # 创建 observations 组
        obs = f.create_group('observations')
        
        # qpos: 本体感知状态 (T, 128) - RDT 示例中用 qpos 作为字段名
        obs.create_dataset('qpos', data=proprio, dtype=np.float32)
        
        # 创建 images 组
        images_grp = obs.create_group('images')
        
        # 为每个相机创建数据集 (存储压缩 JPEG 字节)
        img_dir = ep_dir / "images"
        
        for cam_idx, (view, cam_key) in enumerate(zip(views, cam_keys)):
            # 使用可变长度字节数组存储压缩图像
            dt = h5py.special_dtype(vlen=np.uint8)
            cam_ds = images_grp.create_dataset(cam_key, shape=(T,), dtype=dt)
            
            for t in range(T):
                # 使用最新的时间图像 (timg-1)
                img_path = img_dir / f"step_{t:06d}_timg{timg-1}_{view}.jpg"
                if img_path.exists():
                    jpg_bytes = _read_and_encode_image(img_path)
                    cam_ds[t] = np.frombuffer(jpg_bytes, dtype=np.uint8)
                else:
                    # 如果图像不存在，尝试 timg0
                    img_path = img_dir / f"step_{t:06d}_timg0_{view}.jpg"
                    if img_path.exists():
                        jpg_bytes = _read_and_encode_image(img_path)
                        cam_ds[t] = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    else:
                        raise FileNotFoundError(f"Image not found: {img_path}")
        
        # action: 动作 (T, 128) - 直接放在根级别
        f.create_dataset('action', data=action, dtype=np.float32)
    
    # 在同目录下创建指令 JSON 文件
    instruction_json = {
        "instruction": INSTRUCTION,
        "simplified_instruction": INSTRUCTION,
        "expanded_instruction": [INSTRUCTION]
    }
    instruction_path = out_path.parent / "expanded_instruction_gpt-4-turbo.json"
    if not instruction_path.exists():
        instruction_path.write_text(json.dumps(instruction_json, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Created instruction file: {instruction_path}")


def _is_episode_dir(path: Path) -> bool:
    return (path / "meta.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "raw_path",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "rdt_raw",
        help="episode_XXXXXX raw directory OR a root containing episode_XXXXXX subfolders",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .hdf5 path (single episode only; default: dataset/episode_XXXXXX.hdf5)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "RoboticsDiffusionTransformer" / "data" / "datasets" / DATASET_NAME,
        help="output directory for batch conversion",
    )
    args = ap.parse_args()

    raw_path = args.raw_path
    if not raw_path.exists() or not raw_path.is_dir():
        print(f"Not a directory: {raw_path}")
        return 2

    if _is_episode_dir(raw_path):
        ep_dir = raw_path
        out_path = args.out if args.out is not None else (args.out_dir / f"{ep_dir.name}.hdf5")
        _build_episode_hdf5(ep_dir, out_path)
        print(f"Wrote: {out_path}")
        return 0

    episode_dirs = sorted(p for p in raw_path.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not episode_dirs:
        print(f"No episode_XXXXXX folders found in: {raw_path}")
        return 3

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for ep_dir in episode_dirs:
        if not _is_episode_dir(ep_dir):
            continue
        out_path = out_dir / f"{ep_dir.name}.hdf5"
        _build_episode_hdf5(ep_dir, out_path)
        print(f"Wrote: {out_path}")
        converted += 1

    if converted == 0:
        print(f"No valid episodes with meta.json found in: {raw_path}")
        return 4

    print(f"\n转换完成！共转换 {converted} 个 episodes")
    print(f"输出目录: {out_dir}")
    print(f"\n接下来需要修改 RDT 配置文件:")
    print(f"  1. configs/finetune_datasets.json -> [\"{DATASET_NAME}\"]")
    print(f"  2. configs/finetune_sample_weights.json -> {{\"{DATASET_NAME}\": 1.0}}")
    print(f"  3. configs/dataset_control_freq.json -> 添加 \"{DATASET_NAME}\": 30.0")
    print(f"  4. data/hdf5_vla_dataset.py -> 修改 HDF5_DIR 和 DATASET_NAME")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
