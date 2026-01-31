# 4.1 RDT 训练环境配置

本节详细说明如何配置 RDT 微调所需的完整环境。

---

## 1. 硬件要求

### 最低配置（RDT-170M）

| 组件 | 要求 |
|-----|------|
| GPU | NVIDIA GPU，8GB+ 显存（如 RTX 4060/3070） |
| RAM | 32GB+ 系统内存 |
| 存储 | 100GB+ SSD（数据集 + 检查点） |
| CUDA | 11.8 或 12.1 |

### 推荐配置（RDT-1B）

| 组件 | 要求 |
|-----|------|
| GPU | 多卡 A100/H100，80GB+ 显存 |
| RAM | 128GB+ 系统内存 |
| 存储 | 500GB+ NVMe SSD |
| CUDA | 12.1+ |

---

## 2. 软件环境安装

### 2.1 创建 Conda 环境

```bash
# 创建新环境
conda create -n rdt python=3.10.0
conda activate rdt

# 验证 Python 版本
python --version  # 应显示 Python 3.10.0
```

### 2.2 安装 PyTorch

根据你的 CUDA 版本选择对应命令：

```bash
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1（推荐）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

### 2.3 安装 Flash Attention

```bash
# 必须先安装 packaging
pip install packaging==24.0

# 安装 flash-attn（需要编译，可能需要几分钟）
pip install flash-attn --no-build-isolation

# 验证安装
python -c "import flash_attn; print(flash_attn.__version__)"
```

> **注意**：如果编译失败，确保已安装 CUDA toolkit 和 gcc/g++。

### 2.4 安装其他依赖

```bash
cd RoboticsDiffusionTransformer

# 安装主要依赖
pip install -r requirements.txt

# 如果需要处理数据
pip install -r requirements_data.txt
```

---

## 3. 预训练模型下载

### 3.1 下载 RDT 模型

从 HuggingFace 下载预训练权重：

```bash
# 方式 1：使用 huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download robotics-diffusion-transformer/rdt-170m --local-dir ./models/rdt-170m

# 方式 2：使用 git lfs
git lfs install
git clone https://huggingface.co/robotics-diffusion-transformer/rdt-170m models/rdt-170m
```

### 3.2 下载编码器

```bash
# T5-v1.1-XXL（语言编码器）
huggingface-cli download google/t5-v1_1-xxl --local-dir ./models/t5-v1_1-xxl

# SigLIP（视觉编码器）
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir ./models/siglip-so400m-patch14-384
```

### 3.3 创建符号链接

```bash
# 在 RoboticsDiffusionTransformer 根目录下
mkdir -p google
ln -s $(pwd)/models/t5-v1_1-xxl google/t5-v1_1-xxl
ln -s $(pwd)/models/siglip-so400m-patch14-384 google/siglip-so400m-patch14-384
```

---

## 4. 配置文件修改

### 4.1 base.yaml

编辑 `configs/base.yaml`，设置缓冲区路径：

```yaml
dataset:
  # 设置数据缓冲区路径（需要至少 400GB 空间）
  buf_path: /path/to/your/buffer
```

### 4.2 数据集配置

编辑 `configs/finetune_datasets.json`：

```json
[
    "lerobot"
]
```

编辑 `configs/finetune_sample_weights.json`：

```json
{
    "lerobot": 1.0
}
```

编辑 `configs/dataset_control_freq.json`，添加你的数据集：

```json
{
    "lerobot": 30
}
```

---

## 5. 显存优化配置

### 5.1 低显存场景（<16GB）

修改 `finetune.sh`：

```bash
python main.py \
    --pretrained_model_name_or_path="./models/rdt-170m" \  # 使用小模型
    --train_batch_size=1 \                                  # 最小 batch
    --gradient_accumulation_steps=32 \                      # 梯度累积
    --mixed_precision="no" \                                # 关闭混合精度（如果 bf16 有问题）
    --precomp_lang_embed \                                  # 预计算语言嵌入
    --dataloader_num_workers=0                              # 减少内存占用
```

### 5.2 中等显存（24GB-48GB）

```bash
deepspeed --num_gpus=1 main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="./models/rdt-170m" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --mixed_precision="bf16" \
    --precomp_lang_embed
```

### 5.3 多卡训练（A100/H100）

创建 `hostfile.txt`：

```
localhost slots=4
```

启动训练：

```bash
deepspeed --hostfile=hostfile.txt main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --train_batch_size=32 \
    --mixed_precision="bf16"
```

---

## 6. 环境验证

运行以下脚本验证环境是否正确配置：

```python
#!/usr/bin/env python3
# test_env.py

import sys

def check_env():
    print("=" * 50)
    print("RDT 环境检查")
    print("=" * 50)
    
    # Python 版本
    print(f"\n[1] Python: {sys.version}")
    
    # PyTorch
    try:
        import torch
        print(f"[2] PyTorch: {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("[2] PyTorch: NOT INSTALLED")
    
    # Flash Attention
    try:
        import flash_attn
        print(f"[3] Flash Attention: {flash_attn.__version__}")
    except ImportError:
        print("[3] Flash Attention: NOT INSTALLED")
    
    # DeepSpeed
    try:
        import deepspeed
        print(f"[4] DeepSpeed: {deepspeed.__version__}")
    except ImportError:
        print("[4] DeepSpeed: NOT INSTALLED (optional)")
    
    # Transformers
    try:
        import transformers
        print(f"[5] Transformers: {transformers.__version__}")
    except ImportError:
        print("[5] Transformers: NOT INSTALLED")
    
    print("\n" + "=" * 50)
    print("检查完成！")

if __name__ == "__main__":
    check_env()
```

运行：

```bash
python test_env.py
```

---

## 7. 常见问题

### Q1: flash-attn 编译失败

```bash
# 确保安装了 CUDA toolkit
nvcc --version

# 安装编译工具
sudo apt install build-essential

# 尝试指定 CUDA 架构
TORCH_CUDA_ARCH_LIST="8.0;8.6" pip install flash-attn --no-build-isolation
```

### Q2: 找不到 CUDA

```bash
# 检查 CUDA 路径
echo $CUDA_HOME

# 如果为空，设置路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Q3: 内存不足（OOM）

1. 减小 `train_batch_size`
2. 增大 `gradient_accumulation_steps`
3. 使用 ZeRO-3 配置
4. 启用 CPU offload
