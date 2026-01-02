# 理论推导：RDT fine-tuning 数据格式（unified 128 + mask + action_chunk）

本章解释 2.3 数据集结构背后的“为什么”，并逐条对齐仓库实现：
- `RDT/rdt_hdf5.py`
- `RDT/collect_rdt_dataset_teleop.py`
- `RDT/build_rdt_hdf5_from_raw.py`

## 1. 观测定义与张量形状

实现遵循 `RDT/rdt_hdf5.py` 模块头部描述：

- 观测：

$$
o_t := (X_{t-T_{img}+1:t},\ z_t,\ c)
$$

其中：
- $X$：图像序列（本项目 `Timg=2`）
- $z_t$：proprio（统一 128 维）
- $c$：instruction / 条件（保存在 `meta/instruction`）

HDF5 的关键张量：
- `observations/images`：$(T,\ Timg=2,\ Ncam=3,\ H,\ W,\ 3)$
- `observations/proprio`：$(T,128)$
- `actions/action`：$(T,128)$

## 2. 为什么要 unified 128 + mask

真实机器人采集里，很多维度未必能“可靠获得”，例如：
- 末端速度/角速度可能只能由差分估计
- 某些关节/夹爪的物理量可能缺失

因此实现了：
- `UnifiedVector.value`：数值（缺失维度填 0）
- `UnifiedVector.mask`：可用性（0/1）

训练时模型可用 mask 忽略缺失维度，避免把“填 0”误当成真实观测。

对应实现：`RDT/rdt_hdf5.py::UnifiedVector/make_unified_vector/fill_slice`。

## 3. 图像预处理：pad-to-square + resize

`collect_rdt_dataset_teleop.py::pad_to_square_and_resize_rgb()` 做：
1) 按最大边补成正方形（padding 颜色可配）
2) resize 到 `image_size`（默认 384）
3) BGR→RGB

这样可保证不同相机分辨率下，输入张量形状一致。

## 4. action_chunk 的构造（Ta=64）

RDT fine-tuning 需要把“未来一段动作序列”打包为 chunk。

在 `RDTHDF5EpisodeWriter.finalize_action_chunks()` 中：
- 已有 per-step 动作：`action_all[t]`（shape `(T,128)`）
- 构造：

$$
\text{action\_chunk}[t,\tau] = \text{action}[t+\tau]
$$

其中 $\tau\in[0,Ta)$ 且 $t+\tau<T$。

越界部分做 0 填充，并把 mask 置 0：
- `action_chunk[t, chunk_len:] = 0`
- `action_chunk_mask[t, chunk_len:] = 0`

张量最终形状：
- `actions/action_chunk`：$(T,Ta,128)$

## 5. 从 raw（CSV+JPG）到 HDF5

- raw 的目的：可直接打开 `images/*.jpg` 目检、用 CSV 对齐定位问题。
- 合成的目的：得到标准 HDF5 episode，用于训练/评估。

raw writer 实现：`collect_rdt_dataset_teleop.py::RawEpisodeWriter`。
