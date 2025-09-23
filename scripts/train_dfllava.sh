#!/bin/bash
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
deepspeed --include=localhost:1,2,3,4 --master_port=29501 llava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/data/szk/llava-v1.5-7b \
    --version v1 \
    --data_path /home/data/szk/Fakeclub/data_json/train_score.json \
    --image_folder /home/data/szk/Fakeclub/train \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/dfllava  \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 
