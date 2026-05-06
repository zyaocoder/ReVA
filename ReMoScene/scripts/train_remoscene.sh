#!/usr/bin/env bash
# Training script for ReMoScene
# Usage: bash scripts/train_remoscene.sh

set -euo pipefail

# ---- Paths ---------------------------------------------------------------
PACKAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # ReMoScene/ directory
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
MODEL_ID="${MODEL_ID:-/path/to/base_model}"
DATA_PATH="${DATA_PATH:-/path/to/train.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/video_root}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/remoscene_run}"

# ---- Training config -----------------------------------------------------
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
NUM_DEVICES="${NUM_DEVICES:-4}"
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH="${PACKAGE_DIR}:${PACKAGE_DIR}/src:$PYTHONPATH"

require_config() {
    local name="$1"
    local value="$2"
    if [[ "$value" == /path/to/* ]]; then
        echo "Please set ${name} before running this script." >&2
        exit 1
    fi
}

require_config "MODEL_ID" "$MODEL_ID"
require_config "DATA_PATH" "$DATA_PATH"
require_config "IMAGE_FOLDER" "$IMAGE_FOLDER"

mkdir -p "$OUTPUT_DIR"

# ---- Launch --------------------------------------------------------------
"$DEEPSPEED_BIN" \
    --num_gpus "$NUM_DEVICES" \
    "${PACKAGE_DIR}/src/train/train_remoscene.py" \
    --deepspeed "${PACKAGE_DIR}/scripts/zero3.json" \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
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
    --n_remoscene_layers 2 \
    --remoscene_nhead 8 \
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
