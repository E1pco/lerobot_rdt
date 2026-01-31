#!/bin/bash
# RTX 5090 服务器训练配置
# 高显存(24GB+) & 大内存(90GB) 优化配置

export TEXT_ENCODER_NAME="./models/t5-v1_1-xxl"
export VISION_ENCODER_NAME="./models/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-finetune-server"
# 数据集目录 (相对于 RoboticsDiffusionTransformer 根目录)
export DATASET_DIR="data/datasets/lerobot/"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' not found!"
    echo "Please ensure the dataset is placed correctly or update DATASET_DIR in this script."
    exit 1
fi


# RTX 5090 强力训练配置
# 1. 开启 bf16 混合精度 (大幅提速且省显存)
# 2. 增加 batch_size (利用大显存)
# 3. 增加 num_workers (利用多核CPU和大内存)

python main.py \
    --pretrained_model_name_or_path="./models/rdt-170m" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --sample_batch_size=1 \
    --max_train_steps=10000 \
    --checkpointing_period=500 \
    --sample_period=1000 \
    --checkpoints_total_limit=20 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=16 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --report_to=tensorboard \
