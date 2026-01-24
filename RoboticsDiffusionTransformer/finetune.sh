#!/bin/bash
# 4060 Laptop (8GB) 单机训练配置
# 使用 RDT-170M 小模型 + 预计算语言嵌入以节省显存

export TEXT_ENCODER_NAME="./models/t5-v1_1-xxl"
export VISION_ENCODER_NAME="./models/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-finetune-lerobot"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# 4060 Laptop 单卡训练 (使用较小的 RDT-170M 模型)
python main.py \
    --pretrained_model_name_or_path="./models/rdt-170m" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=1 \
    --gradient_accumulation_steps=32 \
    --sample_batch_size=1 \
    --max_train_steps=2000 \
    --checkpointing_period=100 \
    --sample_period=1000 \
    --checkpoints_total_limit=10 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="no" \
    --max_grad_norm=0 \
    --dataloader_num_workers=0 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --report_to=tensorboard \
    --resume_from_checkpoint="checkpoint-30" \
