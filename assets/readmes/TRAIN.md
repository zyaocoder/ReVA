# Training Instructions

## Repository Layout

- `ReMoScene/backbone/remoscene.py`: ReMoScene core implementation
- `ReMoScene/src/train/train_remoscene.py`: training entrypoint
- `ReMoScene/scripts/train_remoscene.sh`: simple non-Slurm training script
- `ReMoScene/convert_hf_video_to_reva.py`: dataset conversion helper
- `ReMoScene/test_remoscene_smoke.py`: CPU smoke test


## Smoke Test

```bash
python ReMoScene/test_remoscene_smoke.py
```

## Training

Set paths first:

```bash
export MODEL_ID=/path/to/base_model
export DATA_PATH=/path/to/train.jsonl
export IMAGE_FOLDER=/path/to/video_root
export OUTPUT_DIR=./checkpoints/remoscene_run
```

Run:

```bash
bash ReMoScene/scripts/train_remoscene.sh
```

Or launch manually:

```bash
deepspeed --num_gpus 4 ReMoScene/src/train/train_remoscene.py \
  --deepspeed ReMoScene/scripts/zero3.json \
  --model_id "$MODEL_ID" \
  --data_path "$DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --output_dir "$OUTPUT_DIR"
```
