# 4. RDT 训练与微调

快速跳转：

- 环境配置：[4.1 训练环境配置](stage1_4_1_env_setup.md)
- 数据准备：[4.2 数据集准备与转换](stage1_4_2_data_prep.md)
- 微调流程：[4.3 模型微调](stage1_4_3_finetune.md)
- 部署推理：[4.4 模型部署与推理](stage1_4_4_deploy.md)
- 参考：[函数解析：RDT 模型与训练](ref_rdt_training.md)

对应 `decument/task.txt` 的 4. 产出物：

- 微调脚本：`RoboticsDiffusionTransformer/finetune.sh`
- 推理脚本：`RoboticsDiffusionTransformer/inference.sh`
- 模型检查点：`RoboticsDiffusionTransformer/checkpoints/`

---

## 1. 理论背景

RDT（Robotics Diffusion Transformer）是一个基于扩散模型的机器人基础模型，专为双臂操作设计。其核心特点：

- **模型规模**：RDT-1B（10亿参数）或 RDT-170M（1.7亿参数，轻量版）
- **预训练数据**：100万+ 多机器人 episodes
- **输出格式**：预测未来 64 步动作（Action Chunking）
- **输入格式**：语言指令 + 最多三视角 RGB 图像

### 统一动作向量（Unified Action Vector）

RDT 使用 `float32[128]` 的统一向量表示所有机器人的状态和动作：

| 索引范围 | 含义 | 说明 |
|---------|------|------|
| [0, 10) | 右臂关节位置 | 最多 10 个关节 |
| [10, 15) | 右夹爪位置 | 夹爪开合 |
| [15, 25) | 右臂关节速度 | 可选 |
| [25, 30) | 右夹爪速度 | 可选 |
| [30, 33) | 右末端位置 | xyz |
| [33, 39) | 右末端姿态 | 6D 旋转表示 |
| [39, 45) | 右末端速度 | 可选 |
| [50, 60) | 左臂关节位置 | 最多 10 个关节 |
| [60, 65) | 左夹爪位置 | 夹爪开合 |
| [65, 80) | 左臂速度 | 可选 |
| [80, 83) | 左末端位置 | xyz |
| [83, 89) | 左末端姿态 | 6D 旋转表示 |
| [89, 95) | 左末端速度 | 可选 |
| [100, 103) | 底盘速度 | 移动底盘用 |

> **重要**：单臂机器人应将动作填入**右臂**部分（索引 0-50），以与预训练数据对齐。

### 6D 旋转表示

RDT 使用 [6D 旋转表示](https://arxiv.org/pdf/1812.07035) 而非欧拉角或四元数：

```python
import numpy as np

def rotation_matrix_to_6d(R):
    """将 3x3 旋转矩阵转换为 6D 表示"""
    return R[:, :2].flatten()  # 取前两列，展平为 6 维

def euler_to_6d(roll, pitch, yaw):
    """欧拉角转 6D 表示"""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    return rotation_matrix_to_6d(R)
```

---

## 2. 本章目标

1. **环境搭建**：配置 RDT 训练所需的 Python 环境、CUDA、DeepSpeed
2. **数据准备**：将采集的 HDF5 数据集转换为 RDT 训练格式
3. **模型微调**：基于预训练权重在自采数据上微调
4. **部署推理**：在真实机器人上部署微调后的模型

---

## 3. 程序设计结构

RDT 训练管线可分为以下模块：

```
RoboticsDiffusionTransformer/
├── configs/                    # 配置文件
│   ├── base.yaml              # 模型架构与数据处理配置
│   ├── state_vec.py           # 统一向量索引定义
│   ├── finetune_datasets.json # 微调数据集列表
│   └── zero2.json             # DeepSpeed ZeRO-2 配置
├── data/                       # 数据加载
│   ├── hdf5_vla_dataset.py    # HDF5 数据集加载器（需修改）
│   └── compute_dataset_stat_hdf5.py  # 数据集统计
├── models/                     # 模型实现
│   └── rdt_runner.py          # RDT 模型核心
├── train/                      # 训练逻辑
│   └── train.py               # 训练主循环
├── scripts/                    # 实用脚本
│   ├── encode_lang.py         # 语言编码预计算
│   └── agilex_inference.py    # 推理示例
├── finetune.sh                # 微调启动脚本
└── inference.sh               # 推理启动脚本
```

---

## 4. 快速开始

### 4.1 环境安装

```bash
# 创建 Conda 环境
conda create -n rdt python=3.10.0
conda activate rdt

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install packaging==24.0
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

### 4.2 下载预训练模型

```bash
cd RoboticsDiffusionTransformer

# 创建模型目录
mkdir -p models

# 下载 RDT-170M（轻量版，推荐 8GB 显存）
# 从 HuggingFace 下载：https://huggingface.co/robotics-diffusion-transformer/rdt-170m

# 下载编码器
# T5-XXL: https://huggingface.co/google/t5-v1_1-xxl
# SigLIP: https://huggingface.co/google/siglip-so400m-patch14-384

# 链接到 models 目录
ln -s /path/to/rdt-170m models/rdt-170m
ln -s /path/to/t5-v1_1-xxl models/t5-v1_1-xxl
ln -s /path/to/siglip-so400m-patch14-384 models/siglip-so400m-patch14-384
```

### 4.3 准备数据集

```bash
# 链接 HDF5 数据集
cd data
mkdir -p datasets
ln -s /path/to/your/hdf5_dataset datasets/lerobot

# 计算数据集统计信息
cd ..
python -m data.compute_dataset_stat_hdf5
```

### 4.4 预计算语言嵌入（低显存必需）

```bash
# 修改 scripts/encode_lang.py 中的参数
# 然后运行
python -m scripts.encode_lang
```

### 4.5 启动微调

```bash
# 编辑 finetune.sh 配置参数后运行
source finetune.sh
```

### 4.6 部署推理

```bash
# 编辑 inference.sh 配置检查点路径后运行
source inference.sh
```

---

## 5. 脚本作用

| 脚本 | 功能 | 输入 | 输出 |
|-----|------|-----|------|
| `finetune.sh` | 启动微调训练 | HDF5 数据集 | 模型检查点 |
| `inference.sh` | 启动推理 | 检查点 + 图像 | 动作序列 |
| `encode_lang.py` | 预计算语言嵌入 | 文本指令 | `.pt` 嵌入文件 |
| `compute_dataset_stat_hdf5.py` | 计算数据集统计 | HDF5 数据集 | 均值/方差 |

---

## 6. 常见问题

### Q1: 显存不足怎么办？

1. **使用 RDT-170M**：比 RDT-1B 小 6 倍
2. **预计算语言嵌入**：添加 `--precomp_lang_embed` 参数
3. **梯度累积**：设置 `--gradient_accumulation_steps=32`
4. **ZeRO-3**：使用 `configs/zero3.json` 配置
5. **混合精度**：使用 `--mixed_precision="bf16"`

### Q2: 需要训练多少步？

- 建议至少 **150K 步**以获得最佳效果
- 观察 `overall_avg_sample_mse` 指标收敛情况
- 如果训练振荡，增大 batch size

### Q3: 如何判断训练是否收敛？

监控以下指标（通过 TensorBoard 或 WandB）：

- `loss`：训练损失（移动平均）
- `overall_avg_sample_mse`：采样 MSE（越低越好）

---

## 7. 参考资料

- [RDT 官方仓库](https://github.com/thu-ml/RoboticsDiffusionTransformer)
- [RDT 论文](https://arxiv.org/pdf/2410.07864)
- [HuggingFace 模型](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)
- [6D 旋转表示论文](https://arxiv.org/pdf/1812.07035)
