#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Inspect an RDT fine-tuning HDF5 episode.

Usage examples:
  python RDT/inspect_rdt_hdf5.py path/to/episode_000001.hdf5
  python RDT/inspect_rdt_hdf5.py path/to/episode.hdf5 --tree
  python RDT/inspect_rdt_hdf5.py path/to/episode.hdf5 --stats
  python RDT/inspect_rdt_hdf5.py path/to/episode.hdf5 --all

This script is intentionally read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np

# Optional dependency (already used elsewhere in this repo). Only needed for --export-images.
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# Allow running from repo root with relative imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from rdt_hdf5 import (
        LEFT_ARM_JOINT_POS,
        LEFT_ARM_JOINT_VEL,
        LEFT_EEF_ANG_VEL,
        LEFT_EEF_LIN_VEL,
        LEFT_EEF_POS,
        LEFT_EEF_ROT6D,
        LEFT_GRIPPER_POS,
        LEFT_GRIPPER_VEL,
        RIGHT_ARM_JOINT_POS,
        RIGHT_ARM_JOINT_VEL,
        RIGHT_EEF_ANG_VEL,
        RIGHT_EEF_LIN_VEL,
        RIGHT_EEF_POS,
        RIGHT_EEF_ROT6D,
        RIGHT_GRIPPER_POS,
        RIGHT_GRIPPER_VEL,
    )
except Exception:  # pragma: no cover
    # Fallback if import path is weird; keep the script usable.
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


def _fmt_shape(shape: Tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ")"


def _print_attrs(obj: Any, *, prefix: str = "") -> None:
    if not hasattr(obj, "attrs"):
        return
    keys = list(obj.attrs.keys())
    if not keys:
        return
    for k in sorted(keys):
        v = obj.attrs[k]
        # h5py returns numpy scalars/bytes sometimes
        if isinstance(v, (bytes, np.bytes_)):
            try:
                v = v.decode("utf-8")
            except Exception:
                pass
        print(f"{prefix}{k}: {v}")


def _walk_tree(h5: h5py.File) -> None:
    def visitor(name: str, obj: Any) -> None:
        depth = name.count("/")
        indent = "  " * depth
        if isinstance(obj, h5py.Group):
            print(f"{indent}- {name}/ (group)")
        elif isinstance(obj, h5py.Dataset):
            comp = obj.compression if obj.compression is not None else "none"
            print(
                f"{indent}- {name} (dataset) shape={_fmt_shape(obj.shape)} dtype={obj.dtype} comp={comp}"
            )

    print("\n**HDF5 树结构**")
    print(f"- / (file)\n")
    h5.visititems(visitor)


def _safe_stat(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    if arr.dtype.kind in ("U", "S", "O"):
        return f"dtype={arr.dtype}"
    a = np.asarray(arr)
    return f"min={np.min(a):.6g} max={np.max(a):.6g} mean={np.mean(a):.6g}"


def _try_get(h5: h5py.File, path: str) -> h5py.Dataset | None:
    obj = h5.get(path)
    return obj if isinstance(obj, h5py.Dataset) else None


def _finite_report(x: np.ndarray) -> str:
    x = np.asarray(x)
    if x.size == 0:
        return "empty"
    if x.dtype.kind not in ("f", "i", "u"):
        return f"dtype={x.dtype}"
    finite = np.isfinite(x)
    n_bad = int((~finite).sum())
    return "ok" if n_bad == 0 else f"bad={n_bad}/{x.size}"


def _clip_index(t: int, T: int) -> int:
    if T <= 0:
        return 0
    return int(max(0, min(T - 1, t)))


def _slice_table() -> Dict[str, slice]:
    return {
        "RIGHT_ARM_JOINT_POS": RIGHT_ARM_JOINT_POS,
        "RIGHT_GRIPPER_POS": RIGHT_GRIPPER_POS,
        "RIGHT_ARM_JOINT_VEL": RIGHT_ARM_JOINT_VEL,
        "RIGHT_GRIPPER_VEL": RIGHT_GRIPPER_VEL,
        "RIGHT_EEF_POS": RIGHT_EEF_POS,
        "RIGHT_EEF_ROT6D": RIGHT_EEF_ROT6D,
        "RIGHT_EEF_LIN_VEL": RIGHT_EEF_LIN_VEL,
        "RIGHT_EEF_ANG_VEL": RIGHT_EEF_ANG_VEL,
        "LEFT_ARM_JOINT_POS": LEFT_ARM_JOINT_POS,
        "LEFT_GRIPPER_POS": LEFT_GRIPPER_POS,
        "LEFT_ARM_JOINT_VEL": LEFT_ARM_JOINT_VEL,
        "LEFT_GRIPPER_VEL": LEFT_GRIPPER_VEL,
        "LEFT_EEF_POS": LEFT_EEF_POS,
        "LEFT_EEF_ROT6D": LEFT_EEF_ROT6D,
        "LEFT_EEF_LIN_VEL": LEFT_EEF_LIN_VEL,
        "LEFT_EEF_ANG_VEL": LEFT_EEF_ANG_VEL,
    }


def _print_slice_summaries(h5: h5py.File, *, t: int, max_values: int = 10) -> None:
    print("\n**Slice 检查（按 Table4 段位）**")

    proprio = _try_get(h5, "observations/proprio")
    proprio_mask = _try_get(h5, "observations/proprio_mask")
    action = _try_get(h5, "actions/action")
    action_mask = _try_get(h5, "actions/action_mask")

    if proprio is None or proprio_mask is None:
        print("- observations/proprio(_mask): missing")
        return
    if action is None or action_mask is None:
        print("- actions/action(_mask): missing")
        return

    T = int(proprio.shape[0])
    if T <= 0:
        print("- empty episode")
        return
    t = _clip_index(t, T)

    p = np.asarray(proprio[t], dtype=np.float32)
    pm = np.asarray(proprio_mask[t], dtype=np.uint8)
    a = np.asarray(action[t], dtype=np.float32)
    am = np.asarray(action_mask[t], dtype=np.uint8)

    print(f"- step t={t}/{T - 1}")
    print(f"- proprio finite: {_finite_report(p)} | mask unique={sorted(set(pm.tolist()))[:5]}")
    print(f"- action  finite: {_finite_report(a)} | mask unique={sorted(set(am.tolist()))[:5]}")

    table = _slice_table()

    def one(name: str, sl: slice) -> None:
        pv = p[sl]
        pms = pm[sl]
        av = a[sl]
        ams = am[sl]

        p_on = int(pms.sum())
        a_on = int(ams.sum())

        pv_show = np.array2string(pv[:max_values], precision=4, separator=", ", suppress_small=False)
        av_show = np.array2string(av[:max_values], precision=4, separator=", ", suppress_small=False)
        print(f"- {name} [{sl.start}:{sl.stop}]")
        print(f"  proprio: mask_on={p_on}/{sl.stop - sl.start} stat=({_safe_stat(pv)}) head={pv_show}")
        print(f"  action : mask_on={a_on}/{sl.stop - sl.start} stat=({_safe_stat(av)}) head={av_show}")

    # Order matters for readability.
    ordered = [
        "RIGHT_ARM_JOINT_POS",
        "RIGHT_GRIPPER_POS",
        "RIGHT_ARM_JOINT_VEL",
        "RIGHT_GRIPPER_VEL",
        "RIGHT_EEF_POS",
        "RIGHT_EEF_ROT6D",
        "RIGHT_EEF_LIN_VEL",
        "RIGHT_EEF_ANG_VEL",
        "LEFT_ARM_JOINT_POS",
        "LEFT_GRIPPER_POS",
        "LEFT_ARM_JOINT_VEL",
        "LEFT_GRIPPER_VEL",
        "LEFT_EEF_POS",
        "LEFT_EEF_ROT6D",
        "LEFT_EEF_LIN_VEL",
        "LEFT_EEF_ANG_VEL",
    ]
    for name in ordered:
        sl = table.get(name)
        if sl is not None:
            one(name, sl)


def _check_action_chunk_alignment(h5: h5py.File, *, max_steps: int = 5) -> int:
    """Return 0 if ok; else non-zero."""
    action = _try_get(h5, "actions/action")
    action_mask = _try_get(h5, "actions/action_mask")
    chunk = _try_get(h5, "actions/action_chunk")
    chunk_mask = _try_get(h5, "actions/action_chunk_mask")

    if action is None or action_mask is None or chunk is None or chunk_mask is None:
        return 0

    T = int(action.shape[0])
    if T <= 0:
        return 0

    Ta = int(chunk.shape[1])
    to_check = list({0, min(1, T - 1), min(2, T - 1), T - 1})
    to_check = [t for t in to_check if 0 <= t < T][:max_steps]

    errors: list[str] = []
    for t in to_check:
        end = min(T, t + Ta)
        ref = np.asarray(action[t:end], dtype=np.float32)
        refm = np.asarray(action_mask[t:end], dtype=np.uint8)
        got = np.asarray(chunk[t, : end - t], dtype=np.float32)
        gotm = np.asarray(chunk_mask[t, : end - t], dtype=np.uint8)
        if ref.shape != got.shape:
            errors.append(f"t={t}: chunk shape {got.shape} != {ref.shape}")
            continue
        if not np.allclose(ref, got, atol=0, rtol=0):
            errors.append(f"t={t}: action_chunk values mismatch")
        if not np.array_equal(refm, gotm):
            errors.append(f"t={t}: action_chunk_mask mismatch")

    print("\n**action_chunk 对齐检查**")
    if errors:
        for e in errors:
            print(f"- ERROR: {e}")
        return 2
    print(f"- OK (checked {len(to_check)} step(s))")
    return 0


def _export_images(h5: h5py.File, *, out_dir: Path, t: int) -> int:
    if cv2 is None:
        print("\n**导图**")
        print("- ERROR: 未找到 cv2（OpenCV），无法导出图片")
        return 2

    ds = _try_get(h5, "observations/images")
    if ds is None:
        print("\n**导图**")
        print("- ERROR: observations/images missing")
        return 2

    T = int(ds.shape[0])
    if T <= 0:
        print("\n**导图**")
        print("- ERROR: empty episode")
        return 2

    t = _clip_index(t, T)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = ["exterior", "right_wrist", "left_wrist"]
    # ds[t] shape: (Timg=2, Ncam=3, H, W, 3) in RGB
    imgs = np.asarray(ds[t], dtype=np.uint8)
    for ti in range(imgs.shape[0]):
        for ci in range(imgs.shape[1]):
            rgb = imgs[ti, ci]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out_path = out_dir / f"step_{t:06d}_timg{ti}_{views[ci]}.png"
            ok = cv2.imwrite(str(out_path), bgr)
            if not ok:
                print(f"- ERROR: failed to write {out_path}")
                return 2

    print("\n**导图**")
    print(f"- wrote images to: {out_dir}")
    return 0


def _print_key_datasets(h5: h5py.File) -> None:
    print("\n**关键 datasets**")

    paths = [
        "observations/images",
        "observations/proprio",
        "observations/proprio_mask",
        "observations/control_frequency_hz",
        "actions/action",
        "actions/action_mask",
        "actions/action_chunk",
        "actions/action_chunk_mask",
        "timestamps_unix_s",
        "ik_success",
    ]

    for p in paths:
        ds = _try_get(h5, p)
        if ds is None:
            print(f"- {p}: (missing)")
            continue
        comp = ds.compression if ds.compression is not None else "none"
        print(f"- {p}: shape={_fmt_shape(ds.shape)} dtype={ds.dtype} comp={comp}")


def _print_meta(h5: h5py.File) -> None:
    print("\n**meta attrs**")
    meta = h5.get("meta")
    if meta is None:
        print("- (missing group: meta)")
        return
    _print_attrs(meta, prefix="- ")


def _stats(h5: h5py.File, *, max_image_samples: int = 2) -> None:
    print("\n**统计摘要**")

    ts = _try_get(h5, "timestamps_unix_s")
    if ts is not None and ts.shape[0] > 0:
        t0 = float(ts[0])
        t1 = float(ts[-1])
        print(f"- steps: {ts.shape[0]}")
        print(f"- timestamps: start={t0:.3f} end={t1:.3f} span={t1 - t0:.3f}s")
    else:
        print("- steps: 0 (or missing timestamps)")

    ik = _try_get(h5, "ik_success")
    if ik is not None and ik.shape[0] > 0:
        ik_arr = np.asarray(ik[...], dtype=np.uint8)
        rate = float(np.mean(ik_arr))
        print(f"- ik_success: mean={rate:.3f} (1=success)")

    for name in [
        ("proprio_mask", "observations/proprio_mask"),
        ("action_mask", "actions/action_mask"),
    ]:
        ds = _try_get(h5, name[1])
        if ds is None:
            continue
        m = np.asarray(ds[...], dtype=np.uint8)
        print(f"- {name[0]}: sum={int(m.sum())} / elements={m.size}")

    a = _try_get(h5, "actions/action")
    if a is not None and a.shape[0] > 0:
        a0 = np.asarray(a[0], dtype=np.float32)
        print(f"- actions/action[0]: {_safe_stat(a0)}")

    p = _try_get(h5, "observations/proprio")
    if p is not None and p.shape[0] > 0:
        p0 = np.asarray(p[0], dtype=np.float32)
        print(f"- observations/proprio[0]: {_safe_stat(p0)}")

    img = _try_get(h5, "observations/images")
    if img is not None and img.shape[0] > 0:
        n = min(int(img.shape[0]), max_image_samples)
        # images are uint8; sample a few frames only
        sample = np.asarray(img[:n], dtype=np.uint8)
        print(f"- observations/images[:{n}] uint8: min={int(sample.min())} max={int(sample.max())}")


def _validate_shapes(h5: h5py.File) -> int:
    """Return 0 if ok, else non-zero."""
    errors: list[str] = []

    def expect(path: str, shape_suffix: Tuple[int, ...] | None = None, dtype: Any | None = None) -> None:
        ds = _try_get(h5, path)
        if ds is None:
            errors.append(f"missing dataset: {path}")
            return
        if dtype is not None and ds.dtype != dtype:
            errors.append(f"{path}: dtype {ds.dtype} != {dtype}")
        if shape_suffix is not None:
            if tuple(ds.shape[1:]) != tuple(shape_suffix):
                errors.append(f"{path}: shape {ds.shape} (suffix != {shape_suffix})")

    # Only validate fixed suffixes; T is variable.
    expect("observations/images", shape_suffix=(2, 3, 384, 384, 3), dtype=np.dtype("uint8"))
    expect("observations/proprio", shape_suffix=(128,), dtype=np.dtype("float32"))
    expect("observations/proprio_mask", shape_suffix=(128,), dtype=np.dtype("uint8"))
    expect("actions/action", shape_suffix=(128,), dtype=np.dtype("float32"))
    expect("actions/action_mask", shape_suffix=(128,), dtype=np.dtype("uint8"))

    if errors:
        print("\n**检查结果**")
        for e in errors:
            print(f"- ERROR: {e}")
        return 2

    print("\n**检查结果**")
    print("- OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("hdf5", type=Path, help="episode_*.hdf5 path")
    ap.add_argument("--tree", action="store_true", help="打印 HDF5 树结构")
    ap.add_argument("--meta", action="store_true", help="打印 meta attrs")
    ap.add_argument("--keys", action="store_true", help="打印关键 datasets 的 shape/dtype")
    ap.add_argument("--stats", action="store_true", help="打印统计摘要（mask sum / ik_success / images minmax 等）")
    ap.add_argument("--slices", action="store_true", help="按 Table4 slice 打印某个 step 的 proprio/action 数值与 mask")
    ap.add_argument("--t", type=int, default=0, help="用于 --slices/--export-images 的 step index")
    ap.add_argument("--check-chunk", action="store_true", help="校验 actions/action_chunk 与 actions/action 是否严格对齐")
    ap.add_argument("--export-images", type=Path, default=None, help="导出某个 step 的三相机×Timg=2 图片到目录")
    ap.add_argument("--check", action="store_true", help="做一个轻量 shape/dtype 校验")
    ap.add_argument(
        "--all",
        action="store_true",
        help="等同于 --tree --meta --keys --stats --slices --check-chunk --check",
    )

    args = ap.parse_args()

    if args.all:
        args.tree = args.meta = args.keys = args.stats = args.slices = args.check_chunk = args.check = True

    if not args.hdf5.exists():
        print(f"找不到文件: {args.hdf5}")
        return 2

    with h5py.File(args.hdf5, "r") as h5:
        print(f"File: {args.hdf5}")

        if args.meta:
            _print_meta(h5)
        if args.keys:
            _print_key_datasets(h5)
        if args.stats:
            _stats(h5)
        if args.slices:
            _print_slice_summaries(h5, t=args.t)
        if args.check_chunk:
            rc = _check_action_chunk_alignment(h5)
            if rc != 0:
                return rc
        if args.tree:
            _walk_tree(h5)
        if args.export_images is not None:
            rc = _export_images(h5, out_dir=args.export_images, t=args.t)
            if rc != 0:
                return rc
        if args.check:
            return _validate_shapes(h5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
