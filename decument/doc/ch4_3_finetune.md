# 4.3 RDT 模型微调

本节详细说明如何基于预训练权重在自采数据集上微调 RDT 模型。

---

## 1. 微调概览

```
预训练权重 (RDT-170M/RDT-1B)
         ↓
    + 自采数据集
         ↓
    微调训练 (150K+ steps)
         ↓
    微调后检查点
```

---

## 2. 微调脚本配置

### 2.1 基础配置（finetune.sh）

```bash
#!/bin/bash
# finetune.sh - SO-101 双臂微调配置

export TEXT_ENCODER_NAME="./models/t5-v1_1-xxl"
export VISION_ENCODER_NAME="./models/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-finetune-so101"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 启动训练
python main.py \
    --pretrained_model_name_or_path="./models/rdt-170m" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=1 \
    --gradient_accumulation_steps=32 \
    --sample_batch_size=1 \
    --max_train_steps=200000 \
    --checkpointing_period=2000 \
    --sample_period=1000 \
    --checkpoints_total_limit=10 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --report_to=tensorboard
```

### 2.2 参数说明

| 参数 | 说明 | 推荐值 |
|-----|------|-------|
| `pretrained_model_name_or_path` | 预训练权重路径 | `./models/rdt-170m` |
| `train_batch_size` | 每卡 batch size | 1-32 |
| `gradient_accumulation_steps` | 梯度累积步数 | 8-64 |
| `max_train_steps` | 最大训练步数 | 150000-200000 |
| `learning_rate` | 学习率 | 1e-4 ~ 2e-6 |
| `checkpointing_period` | 保存间隔 | 1000-5000 |
| `sample_period` | 验证间隔 | 500-2000 |
| `state_noise_snr` | 状态噪声信噪比 | 40 |
| `image_aug` | 图像增强 | 启用 |
| `precomp_lang_embed` | 预计算语言嵌入 | 低显存必需 |

---

## 3. 不同硬件配置

### 3.1 单卡低显存（8-16GB）

```bash
python main.py \
    --pretrained_model_name_or_path="./models/rdt-170m" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=32 \
    --mixed_precision="no" \
    --max_grad_norm=0 \
    --dataloader_num_workers=0 \
    --precomp_lang_embed \
    --learning_rate=2e-6
```

**等效 batch size**: 1 × 32 = 32

### 3.2 单卡中等显存（24-48GB）

```bash
deepspeed --num_gpus=1 main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="./models/rdt-170m" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --mixed_precision="bf16" \
    --learning_rate=1e-4
```

**等效 batch size**: 4 × 8 = 32

### 3.3 多卡大显存（A100×4）

创建 `hostfile.txt`：
```
localhost slots=4
```

```bash
deepspeed --hostfile=hostfile.txt main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --mixed_precision="bf16" \
    --learning_rate=1e-4
```

**等效 batch size**: 8 × 1 × 4 = 32

---

## 4. 启动训练

### 4.1 首次训练

```bash
cd RoboticsDiffusionTransformer
source finetune.sh
```

### 4.2 从检查点恢复

```bash
python main.py \
    ... \
    --resume_from_checkpoint="latest"  # 或具体路径
```

### 4.3 后台运行

```bash
nohup bash -c "source finetune.sh" > train.log 2>&1 &

# 查看日志
tail -f train.log
```

---

## 5. 监控训练

### 5.1 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir=./checkpoints/rdt-finetune-so101 --port=6006

# 浏览器访问 http://localhost:6006
```

### 5.2 关键指标

| 指标 | 含义 | 期望趋势 |
|-----|------|---------|
| `loss` | 训练损失 | 下降并稳定 |
| `overall_avg_sample_mse` | 采样 MSE | 下降并收敛 |
| `learning_rate` | 当前学习率 | 恒定或按计划变化 |

### 5.3 判断收敛

- `overall_avg_sample_mse` 不再显著下降
- 训练约 150K-200K 步后
- 可以尝试在验证集上评估

---

## 6. 检查点管理

### 6.1 检查点结构

```
checkpoints/rdt-finetune-so101/
├── checkpoint-10000/
│   ├── config.json
│   ├── pytorch_model/
│   │   └── mp_rank_00_model_states.pt
│   └── optimizer/
├── checkpoint-20000/
│   └── ...
└── runs/              # TensorBoard 日志
```

### 6.2 保留策略

`--checkpoints_total_limit=10` 表示最多保留 10 个检查点，旧的自动删除。

### 6.3 手动导出

```python
# 将 DeepSpeed 检查点转换为 PyTorch 格式
from models.rdt_runner import RDTRunner

model = RDTRunner.from_pretrained("checkpoints/rdt-finetune-so101/checkpoint-100000")
model.save_pretrained("exported_model/")
```

---

## 7. 常见问题排查

### Q1: CUDA OOM

```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 减小 `train_batch_size`
2. 增大 `gradient_accumulation_steps`
3. 启用 `--mixed_precision="bf16"`
4. 使用 ZeRO-3: `--deepspeed="./configs/zero3.json"`

### Q2: 训练震荡

**症状**：loss 上下剧烈波动

**解决方案**：
1. 增大等效 batch size
2. 降低学习率 (`--learning_rate=1e-5`)
3. 启用梯度裁剪 (`--max_grad_norm=1.0`)

### Q3: 训练很慢

**解决方案**：
1. 增加 `--dataloader_num_workers`
2. 使用更快的存储（NVMe SSD）
3. 启用混合精度 (`--mixed_precision="bf16"`)
4. 使用 DeepSpeed

### Q4: Loss 不下降

**可能原因**：
1. 数据集格式错误 → 检查 HDF5 结构
2. 学习率太小 → 尝试 `1e-4`
3. 数据量太少 → 至少需要几百个 episode

---

## 8. 超参数调优建议

### 8.1 学习率

| 数据量 | 推荐学习率 |
|-------|-----------|
| < 1K episodes | 1e-5 ~ 5e-5 |
| 1K-10K episodes | 5e-5 ~ 1e-4 |
| > 10K episodes | 1e-4 ~ 2e-4 |

### 8.2 训练步数

| 数据量 | 推荐步数 |
|-------|---------|
| < 1K episodes | 100K-150K |
| 1K-10K episodes | 150K-300K |
| > 10K episodes | 200K-500K |

### 8.3 Batch Size

- **等效 batch size** = `train_batch_size × gradient_accumulation_steps × num_gpus`
- 推荐范围：16-64
- 太小会导致训练不稳定，太大可能过拟合

---

## 9. 完整训练流程清单

- [ ] 数据集准备完成（HDF5 格式）
- [ ] 预训练权重下载完成
- [ ] 语言嵌入预计算完成
- [ ] `finetune.sh` 配置正确
- [ ] 启动训练并确认无错误
- [ ] 监控 TensorBoard 指标
- [ ] 训练收敛后保存最终检查点
- [ ] 在验证集/真机上测试
