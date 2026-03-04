#!/bin/bash
# Training script for BIMBA + Motion on RSVidQA
# Usage: bash scripts/train_bimba_motion.sh

set -e

# ---- Paths ---------------------------------------------------------------
DS_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # ds/ directory
PYTHON=/anvil/projects/x-cis251076/.conda/envs/qwen/bin/python3

MODEL_ID="/anvil/projects/x-cis251076/likai/Downloads/huggingface/models/LLaVA-OneVision-1.5-8B-Instruct"

DATA_PATH=/anvil/projects/x-cis251076/likai/RSVidQA/train_llava.jsonl
IMAGE_FOLDER=/anvil/projects/x-cis251076/likai/RSVidQA   # root for relative video paths
OUTPUT_DIR=/anvil/projects/x-cis251076/checkpoints/bimba-motion-rsvidqa

# ---- Training config -----------------------------------------------------
GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=1
NUM_DEVICES=4
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH="${DS_DIR}:${DS_DIR}/src:$PYTHONPATH"

# ---- Launch --------------------------------------------------------------
deepspeed \
    --num_gpus $NUM_DEVICES \
    "${DS_DIR}/src/train/train_bimba_motion.py" \
    --deepspeed "${DS_DIR}/scripts/zero3.json" \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    \
    --bf16 True \
    --fp16 False \
    --tf32 True \
    --gradient_checkpointing True \
    \
    --fps 1.0 \
    --max_frames 64 \
    --video_min_pixels 100352 \
    --video_max_pixels 602112 \
    \
    --num_queries 256 \
    --n_bimba_layers 2 \
    --bimba_nhead 8 \
    --lq_t 4 \
    --lq_h 8 \
    --lq_w 8 \
    --num_object_queries 16 \
    --rec_n_layers 2 \
    --rec_chunk_size 8 \
    --rec_nhead 8 \
    \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --dataloader_num_workers 4
