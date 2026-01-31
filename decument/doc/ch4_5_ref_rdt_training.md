# 函数解析：RDT 模型与训练

本文档解析 RDT 训练与推理中的核心函数和类。

---

## 1. 数据加载层

### 1.1 HDF5VLADataset (`data/hdf5_vla_dataset.py`)

```python
class HDF5VLADataset:
    """HDF5 数据集加载器"""
```

#### `__init__(self)`

**作用**：初始化数据集，扫描 HDF5 文件并计算采样权重。

**关键变量**：
- `HDF5_DIR`: HDF5 文件目录
- `DATASET_NAME`: 数据集名称
- `episode_sample_weights`: 按 episode 长度的采样权重

#### `get_item(self, index=None, state_only=False)`

**作用**：获取单个训练样本。

**参数**：
- `index`: episode 索引，None 则按权重随机采样
- `state_only`: 是否只返回状态序列

**返回**：
```python
{
    "meta": {"dataset_name", "#steps", "instruction"},
    "step_id": int,
    "state": np.ndarray,       # [1, 128]
    "actions": np.ndarray,     # [64, 128]
    "images": np.ndarray,      # [6, 3, H, W]
    ...
}
```

#### `parse_hdf5_file(self, file_path)`

**作用**：解析单个 HDF5 文件，生成训练样本。

**需要自定义**：根据你的数据格式实现此方法。

---

## 2. 模型层

### 2.1 RDTRunner (`models/rdt_runner.py`)

```python
class RDTRunner:
    """RDT 模型运行器，封装训练和推理逻辑"""
```

#### `from_pretrained(cls, pretrained_path, **kwargs)`

**作用**：从检查点加载预训练模型。

**参数**：
- `pretrained_path`: 检查点路径或 HuggingFace 模型 ID

**返回**：初始化好的 RDTRunner 实例

#### `predict_action(self, images, state, lang_embeddings=None)`

**作用**：推理，预测未来动作序列。

**参数**：
- `images`: [B, 6, 3, H, W] 图像张量（t-1 和 t 时刻各 3 视角）
- `state`: [B, 128] 当前状态向量
- `lang_embeddings`: 预计算的语言嵌入，或 None

**返回**：
- `actions`: [B, 64, 128] 预测的 64 步动作

#### `compute_loss(self, batch)`

**作用**：计算训练损失。

**参数**：
- `batch`: 训练批次数据

**返回**：
- `loss`: 扩散损失标量

---

## 3. 状态向量层

### 3.1 STATE_VEC_IDX_MAPPING (`configs/state_vec.py`)

**作用**：定义 128 维统一向量中每个物理量的索引。

**关键映射**：

| 物理量 | 索引 | 别名 |
|-------|------|-----|
| 右臂关节 0-9 位置 | 0-9 | `arm_joint_0_pos` ... |
| 右夹爪位置 | 10-14 | `gripper_open` |
| 右臂关节速度 | 15-24 | `arm_joint_0_vel` ... |
| 右末端位置 xyz | 30-32 | `eef_pos_x/y/z` |
| 右末端姿态 6D | 33-38 | `eef_angle_0` ... |
| 左臂关节位置 | 50-59 | `left_arm_joint_0_pos` ... |
| 左夹爪位置 | 60-64 | `left_gripper_open` |
| 左末端位置 xyz | 80-82 | `left_eef_pos_x/y/z` |
| 左末端姿态 6D | 83-88 | `left_eef_angle_0` ... |
| 底盘速度 | 100-102 | `base_vel_x/y`, `base_angular_vel` |

**使用示例**：

```python
from configs.state_vec import STATE_VEC_IDX_MAPPING

# 填充右臂第一个关节位置
idx = STATE_VEC_IDX_MAPPING['arm_joint_0_pos']  # = 0
state[idx] = joint_0_value

# 填充左夹爪
idx = STATE_VEC_IDX_MAPPING['left_gripper_open']  # = 60
state[idx] = gripper_value
```

---

## 4. 训练层

### 4.1 训练循环 (`train/train.py`)

#### `train_one_step(model, batch, optimizer)`

**作用**：执行一步训练。

**流程**：
1. 前向传播计算损失
2. 反向传播计算梯度
3. 梯度裁剪
4. 优化器更新
5. 记录指标

#### `validate(model, dataloader)`

**作用**：在验证集上评估模型。

**指标**：
- `overall_avg_sample_mse`: 采样 MSE（最重要）
- `loss`: 验证损失

---

## 5. 语言编码层

### 5.1 encode_lang.py (`scripts/encode_lang.py`)

#### `encode_instruction(model, tokenizer, instruction)`

**作用**：将文本指令编码为嵌入向量。

**参数**：
- `model`: T5 编码器
- `tokenizer`: T5 分词器
- `instruction`: 文本指令字符串

**返回**：
- `embeddings`: [1, seq_len, hidden_dim] 语言嵌入张量

**示例**：

```python
# 编码指令
embeddings = encode_instruction(
    model, tokenizer,
    "Pick up the red cube and place it in the blue box."
)

# 保存
torch.save(embeddings, "outs/lang_embeddings/pick_and_place.pt")
```

---

## 6. 图像处理层

### 6.1 图像预处理

```python
def preprocess_image(image, target_size=224):
    """
    标准图像预处理
    
    Args:
        image: np.ndarray (H, W, 3) RGB
        target_size: 目标尺寸
    
    Returns:
        tensor: [3, target_size, target_size]
    """
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image)
```

### 6.2 图像顺序

RDT 期望的图像顺序（6 视角）：

```
[ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1}, ext_{t}, right_wrist_{t}, left_wrist_{t}]
```

索引对应：
- 0: 外部相机 t-1
- 1: 右腕相机 t-1
- 2: 左腕相机 t-1
- 3: 外部相机 t
- 4: 右腕相机 t
- 5: 左腕相机 t

---

## 7. 6D 旋转表示

### 7.1 转换函数

```python
import numpy as np
from scipy.spatial.transform import Rotation

def rotation_matrix_to_6d(R):
    """3x3 旋转矩阵 → 6D 表示"""
    return R[:, :2].flatten()  # 取前两列

def euler_to_6d(roll, pitch, yaw):
    """欧拉角 (xyz) → 6D 表示"""
    R = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    return rotation_matrix_to_6d(R)

def quat_to_6d(quat):
    """四元数 (xyzw) → 6D 表示"""
    R = Rotation.from_quat(quat).as_matrix()
    return rotation_matrix_to_6d(R)

def rot6d_to_matrix(rot6d):
    """6D 表示 → 3x3 旋转矩阵（Gram-Schmidt 正交化）"""
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    
    return np.stack([b1, b2, b3], axis=1)
```

---

## 8. 数据统计计算

### 8.1 compute_dataset_stat_hdf5.py

```python
def compute_statistics(dataset):
    """
    计算数据集统计信息
    
    Returns:
        stat: {
            "state_mean": [128],
            "state_std": [128],
            "action_mean": [128],
            "action_std": [128]
        }
    """
    all_states = []
    all_actions = []
    
    for i in range(len(dataset)):
        sample = dataset.get_item(i, state_only=True)
        all_states.append(sample['state'])
        all_actions.append(sample['actions'])
    
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    return {
        "state_mean": all_states.mean(axis=0).tolist(),
        "state_std": all_states.std(axis=0).tolist(),
        "action_mean": all_actions.mean(axis=0).tolist(),
        "action_std": all_actions.std(axis=0).tolist(),
    }
```

---

## 9. DeepSpeed 配置

### 9.1 ZeRO-2 配置 (`configs/zero2.json`)

```json
{
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "fp16": {
        "enabled": "auto"
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8
    }
}
```

### 9.2 ZeRO-3 配置（更低显存）

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}
```

---

## 10. 关键参数汇总

| 参数 | 位置 | 默认值 | 说明 |
|-----|------|-------|------|
| `action_chunk_size` | `configs/base.yaml` | 64 | 动作序列长度 |
| `img_history_size` | `configs/base.yaml` | 2 | 图像历史帧数 |
| `state_dim` | `configs/base.yaml` | 128 | 状态向量维度 |
| `learning_rate` | CLI | 1e-4 | 学习率 |
| `train_batch_size` | CLI | 32 | 批次大小 |
| `max_train_steps` | CLI | 200000 | 最大训练步数 |
