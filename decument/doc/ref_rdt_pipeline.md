# 函数解析：RDT 数据采集与落盘（`RDT/`）

本章对应 3. 的“数据集采集源代码 + 格式说明”，关键文件：

- `RDT/collect_rdt_dataset_teleop.py`
- `RDT/rdt_hdf5.py`
- `RDT/build_rdt_hdf5_from_raw.py`
- `RDT/inspect_rdt_hdf5.py`


RDT 采集链路的理论目标是：把多模态时间序列（图像、proprio、action、时间戳）组织成**固定 shape 且可解释**的数据集，使下游 fine-tuning 可以直接消费。核心约束是 `unified 128 + mask + action_chunk`（详见 `theory_rdt_format.md`）。

## 程序设计结构

- `collect_rdt_dataset_teleop.py`：实时采集与“写 step”（raw 或直接 HDF5）
- `rdt_hdf5.py`：格式规范层（UnifiedVector + HDF5 writer）
- `build_rdt_hdf5_from_raw.py`：离线重建（raw→HDF5）
- `inspect_rdt_hdf5.py`：验收与排错（shape/抽样可视化）

## 脚本作用

- 采集：用 `collect_rdt_dataset_teleop.py` 录一段 raw/hdf5 episode。
- 转换：用 `build_rdt_hdf5_from_raw.py` 把 raw 整理成标准 HDF5。
- 检查：用 `inspect_rdt_hdf5.py` 确认张量 shape、mask、抽样帧正常。

## 方法作用

下文按“格式定义 → 采集写入 → 转换与检查”的顺序解释关键方法在管线中的职责。

## 1. `RDT/rdt_hdf5.py` - 数据格式定义与 HDF5 写入核心

**脚本功能**：
该模块定义了 RDT 数据集的核心数据结构，实现了统一的 128 维向量表示（Unified Vector）和 HDF5 文件的写入逻辑。它是数据采集流程中的格式规范层，确保生成的数据符合 RDT Fine-tuning 的输入要求。

### 1.1 `rotmat_to_rot6d(R)` - 旋转矩阵到6D表示的转换

**功能**：将 3×3 旋转矩阵压缩为 6 维连续表示，避免万向锁问题。

**实现原理**：

- **输入**：$3\times3$ 旋转矩阵 $R$
- **输出**：6D 向量（取 $R$ 的前两列，按列主序展平）
- **数学依据**：6D 表示法由 Zhou et al. (CVPR 2019) 提出，通过 Gram-Schmidt 正交化可恢复完整旋转矩阵，且在神经网络训练中比四元数更稳定。

**代码逻辑**：

```python
def rotmat_to_rot6d(R):
    # 取前两列
    return R[:, :2].T.flatten()  # shape: (6,)
```

### 1.2 `UnifiedVector` / `make_unified_vector()` / `fill_slice(vec, sl, data)`

**功能**：实现统一向量（Unified Vector）的构建与填充，这是 RDT 处理异构机器人数据的关键机制。

**设计动机**：
不同机器人的自由度、传感器配置差异很大。Unified Vector 通过 128 维固定长度 + Mask 机制，允许不同配置的机器人共享同一训练管道。

**核心数据结构**：

- `UnifiedVector.value`：`float32[128]` - 存储物理量的数值（未使用的维度填 0）
- `UnifiedVector.mask`：`uint8[128]` - 标记哪些维度有效（1=有效，0=无效）

**方法详解**：

- **`make_unified_vector()`**：创建一个空的 Unified Vector

  ```python
  def make_unified_vector():
      return UnifiedVector(
          value=np.zeros(128, dtype=np.float32),
          mask=np.zeros(128, dtype=np.uint8)
      )
  ```
- **`fill_slice(vec, sl, data)`**：向指定切片填充数据并自动更新 mask

  - **参数**：
    - `vec`：目标 UnifiedVector
    - `sl`：切片对象（如 `slice(0, 6)` 表示前 6 维）
    - `data`：要填充的数据（自动截断到切片长度）
  - **实现**：
    ```python
    def fill_slice(vec, sl, data):
        indices = range(*sl.indices(128))
        length = len(indices)
        vec.value[sl] = data[:length]
        vec.mask[sl] = 1  # 标记为有效
    ```

### 1.3 `RDTHDF5EpisodeWriter` - HDF5 文件写入器

**功能**：管理单个 Episode 的 HDF5 文件写入，自动处理张量形状、Action Chunk 生成等细节。

**初始化**：

```python
writer = RDTHDF5EpisodeWriter(
    filepath="episode_000001.hdf5",
    Timg=2,          # 历史图像帧数
    Ncam=3,          # 相机数量
    image_size=384,  # 图像分辨率
    Ta=64            # Action Chunk 长度
)
```

**核心方法**：

- **`append_step(...)`**：追加一条时间步数据

  - **校验**：自动检查 `images` 的 shape 是否为 `(Timg, Ncam, H, W, 3)`
  - **存储**：将 proprio/action 的 value 和 mask 分别追加到对应 Dataset
- **`finalize_action_chunks()`**：生成未来动作序列

  - **原理**：对于时刻 $t$，构造 `action_chunk[t, :, :] = [action[t], action[t+1], ..., action[t+Ta-1]]`
  - **边界处理**：当 $t+\tau \geq T$ 时，用零填充并将 mask 置 0
  - **自动调用**：在 `close()` 时自动执行
- **`close()`**：关闭文件并保存元数据

  - **流程**：finalize action_chunks → 写入 meta 信息 → 关闭 HDF5 文件句柄

## 2. `RDT/collect_rdt_dataset_teleop.py`

### 2.1 `RawEpisodeWriter`

- 作用：写 raw episode（CSV + JPG），便于肉眼检查
- `append_step(...)`：
  - 保存 `timg*ncam` 张图片到 `images/`
  - 追加 proprio/action/mask/timestamp 到 CSV

### 2.2 图像预处理：`pad_to_square_and_resize_rgb(frame_bgr, out_size, pad_bgr)`

- pad 成正方形 → resize 到 `out_size` → 转 RGB

### 2.3 相机抽象：`MultiViewCamera`

- `_open(source, width, height)`：支持 index 或路径；支持 v4l2/gstreamer backend
- `read_bgr(view)`：读取指定 view 的一帧
- `close()`：释放所有 `VideoCapture`

### 2.4 速度估计：`compute_ang_vel_rad_s(R_prev, R_cur, dt)`

- 通过 $dR=R\_{prev}^T R\_{cur}$ 的旋转向量除以 dt 估计角速度

### 2.5 episode 编号：`_max_existing_episode_idx(out_dir)`

- 扫描 `episode_XXXXXX/` 与 `episode_XXXXXX.hdf5`，返回最大编号
- 用于避免覆盖、实现跨运行递增

### 2.6 采集主类：`JoyConRDTCollector`

- 初始化阶段：
  - 创建双臂 `ArmRig`
  - 可选打开三相机 `MultiViewCamera`
  - 配置 preview 与保存格式
- `run()`（主循环，函数体较长）：
  - 读 JoyCon 控制
  - 生成目标位姿→IK→下发
  - 若录制：构建 unified proprio/action → writer.append_step

## 3. raw→HDF5 与检查

- `build_rdt_hdf5_from_raw.py`：读取 raw 的 CSV/JPG，重建 HDF5 episode
- `inspect_rdt_hdf5.py`：打印 meta/shape，并可抽样检查内容

理论解释（unified/mask/action_chunk）见：`theory_rdt_format.md`。
