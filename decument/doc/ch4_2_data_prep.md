# 4.2 数据集准备与转换

本节说明如何将采集的 raw/HDF5 数据集转换为 RDT 训练所需的格式。

---

## 1. 数据流概览

```
采集 raw 数据 → 转换为 HDF5 → 适配 RDT 格式 → 计算统计信息 → 开始训练
     ↓              ↓              ↓              ↓
  CSV + JPG     episode_*.hdf5   修改 Dataset    mean/std
```

---

## 2. 数据格式要求

### 2.1 RDT 期望的 HDF5 结构

每个 HDF5 文件存储一个 episode：

```
episode_000001.hdf5
├── observations/
│   ├── images/
│   │   ├── cam_exterior   [T, H, W, 3] uint8  # 外部相机
│   │   ├── cam_right_wrist [T, H, W, 3] uint8  # 右腕相机
│   │   └── cam_left_wrist  [T, H, W, 3] uint8  # 左腕相机
│   └── state              [T, 128] float32    # 统一状态向量
├── actions                [T, 128] float32    # 统一动作向量
├── action_mask            [T, 128] uint8      # 有效维度掩码
└── language_instruction   str                 # 语言指令
```

### 2.2 统一向量布局（128 维）

对于 SO-101 双臂 6DOF + 夹爪：

```python
# 右臂（索引 0-14）
right_arm_joints = [0, 1, 2, 3, 4, 5]      # 6 个关节位置
right_gripper = [10]                        # 夹爪开合

# 左臂（索引 50-64）
left_arm_joints = [50, 51, 52, 53, 54, 55]  # 6 个关节位置
left_gripper = [60]                         # 夹爪开合
```

---

## 3. 从 RDT raw 转换

### 3.1 使用现有工具

如果你使用 `RDT/collect_rdt_dataset_teleop.py` 采集数据：

```bash
# 已有 raw 数据在 ./rdt_raw/
cd /path/to/lerobot_rdt

# 转换为 HDF5
python RDT/build_rdt_hdf5_from_raw.py \
    --input-dir ./rdt_raw \
    --output-dir ./rdt_hdf5 \
    --instruction "pick up the object and place it"
```

### 3.2 验证转换结果

```bash
python RDT/inspect_rdt_hdf5.py ./rdt_hdf5/episode_000001.hdf5
```

输出应显示：

```
Episode: episode_000001.hdf5
  Steps: 150
  Images: cam_exterior (150, 224, 224, 3), cam_right_wrist (150, 224, 224, 3)
  State: (150, 128) float32
  Actions: (150, 128) float32
  Instruction: pick up the object and place it
```

---

## 4. 适配 RDT 数据加载器

### 4.1 修改 hdf5_vla_dataset.py

编辑 `RoboticsDiffusionTransformer/data/hdf5_vla_dataset.py`：

```python
import os
import fnmatch
import h5py
import numpy as np
from configs.state_vec import STATE_VEC_IDX_MAPPING

class HDF5VLADataset:
    def __init__(self) -> None:
        # [修改] 数据集路径
        HDF5_DIR = os.environ.get("DATASET_DIR", "data/datasets/lerobot/")
        self.DATASET_NAME = "lerobot"
        
        # 收集所有 HDF5 文件
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
        
        # ... 其余初始化代码 ...
    
    def parse_hdf5_file(self, file_path):
        """[修改] 解析单个 HDF5 文件"""
        try:
            with h5py.File(file_path, 'r') as f:
                # 读取图像
                images = {}
                if 'observations/images/cam_exterior' in f:
                    images['cam_exterior'] = f['observations/images/cam_exterior'][:]
                if 'observations/images/cam_right_wrist' in f:
                    images['cam_right_wrist'] = f['observations/images/cam_right_wrist'][:]
                if 'observations/images/cam_left_wrist' in f:
                    images['cam_left_wrist'] = f['observations/images/cam_left_wrist'][:]
                
                # 读取状态和动作
                state = f['observations/state'][:]  # [T, 128]
                actions = f['actions'][:]           # [T, 128]
                
                # 读取语言指令
                if 'language_instruction' in f.attrs:
                    instruction = f.attrs['language_instruction']
                else:
                    instruction = "manipulation task"
                
                # 随机采样时间步
                T = state.shape[0]
                t = np.random.randint(0, T)
                
                # 构建返回字典
                sample = {
                    "meta": {
                        "dataset_name": self.DATASET_NAME,
                        "#steps": T,
                        "instruction": instruction
                    },
                    "step_id": t,
                    "state": state[t:t+1],  # [1, 128]
                    "actions": self._get_action_chunk(actions, t),  # [64, 128]
                    # ... 图像处理 ...
                }
                
                return True, sample
                
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return False, None
    
    def _get_action_chunk(self, actions, t):
        """获取 action chunk，不足部分用 0 填充"""
        T = actions.shape[0]
        chunk = np.zeros((self.CHUNK_SIZE, self.STATE_DIM), dtype=np.float32)
        
        end_idx = min(t + self.CHUNK_SIZE, T)
        actual_len = end_idx - t
        chunk[:actual_len] = actions[t:end_idx]
        
        return chunk
```

### 4.2 状态向量映射函数

创建辅助函数将 SO-101 数据映射到统一向量：

```python
def so101_to_unified_vector(left_joints, left_gripper, right_joints, right_gripper):
    """
    将 SO-101 双臂数据映射到 RDT 统一向量
    
    Args:
        left_joints: [6] 左臂关节角度 (rad)
        left_gripper: float 左夹爪开合 (归一化 0-1)
        right_joints: [6] 右臂关节角度 (rad)
        right_gripper: float 右夹爪开合 (归一化 0-1)
    
    Returns:
        unified: [128] 统一向量
    """
    unified = np.zeros(128, dtype=np.float32)
    
    # 右臂关节位置 [0-5]
    unified[0:6] = right_joints
    
    # 右夹爪 [10]
    unified[10] = right_gripper
    
    # 左臂关节位置 [50-55]
    unified[50:56] = left_joints
    
    # 左夹爪 [60]
    unified[60] = left_gripper
    
    return unified

def unified_vector_to_so101(unified):
    """
    从统一向量提取 SO-101 双臂数据
    
    Args:
        unified: [128] 统一向量
    
    Returns:
        left_joints, left_gripper, right_joints, right_gripper
    """
    right_joints = unified[0:6]
    right_gripper = unified[10]
    left_joints = unified[50:56]
    left_gripper = unified[60]
    
    return left_joints, left_gripper, right_joints, right_gripper
```

---

## 5. 计算数据集统计信息

训练前需要计算状态/动作的均值和标准差：

```bash
cd RoboticsDiffusionTransformer

# 设置数据集路径
export DATASET_DIR=/path/to/your/hdf5_dataset

# 运行统计脚本
python -m data.compute_dataset_stat_hdf5
```

这将生成 `data/datasets/lerobot/dataset_stat.json`：

```json
{
    "state_mean": [0.1, 0.2, ...],
    "state_std": [0.5, 0.3, ...],
    "action_mean": [0.1, 0.2, ...],
    "action_std": [0.5, 0.3, ...]
}
```

---

## 6. 预计算语言嵌入

由于 T5-XXL 较大，建议预先计算语言嵌入：

### 6.1 单个指令

编辑 `scripts/encode_lang.py`：

```python
GPU = 0
MODEL_PATH = "./models/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
SAVE_DIR = "outs/lang_embeddings/"

TASK_NAME = "pick_and_place"
INSTRUCTION = "Pick up the red cube and place it in the blue box."

# 低显存设备启用 offload
OFFLOAD_DIR = "/tmp/t5_offload"  # 或 None
```

运行：

```bash
python -m scripts.encode_lang
```

输出 `outs/lang_embeddings/pick_and_place.pt`。

### 6.2 批量处理

编辑 `scripts/encode_lang_batch.py`，添加所有任务指令后运行。

---

## 7. 链接数据集到 RDT

```bash
cd RoboticsDiffusionTransformer/data
mkdir -p datasets

# 创建符号链接
ln -s /path/to/your/rdt_hdf5 datasets/lerobot

# 验证
ls datasets/lerobot/
# 应显示 episode_000001.hdf5, episode_000002.hdf5, ...
```

---

## 8. 数据增强

RDT 支持以下数据增强（在 `finetune.sh` 中启用）：

```bash
--image_aug \           # 启用图像增强
--state_noise_snr=40    # 状态噪声（信噪比 40dB）
```

图像增强包括：
- 随机裁剪
- 颜色抖动
- 随机翻转

---

## 9. 数据验证清单

在开始训练前，确保：

- [ ] HDF5 文件可以正常打开
- [ ] 图像形状正确（T, H, W, 3）
- [ ] 状态/动作向量形状为 (T, 128)
- [ ] 语言指令存在且有意义
- [ ] 数据集统计信息已生成
- [ ] 语言嵌入已预计算（如需要）
- [ ] 符号链接正确指向数据目录
