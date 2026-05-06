# ReVA

ReVA is a lightweight repository for our video-centric multimodal model work.
This repo currently focuses on **ReMoScene**, a trainable video compression and
motion reasoning module built on top of a **LLaVA-ov-1.5** backbone.

## Repository Layout

- `ReMoScene/llavaonevision1_5/remoscene.py`: ReMoScene core implementation
- `ReMoScene/src/train/train_remoscene.py`: training entrypoint
- `ReMoScene/scripts/train_remoscene.sh`: simple non-Slurm training script
- `ReMoScene/convert_hf_video_to_reva.py`: dataset conversion helper
- `ReMoScene/test_remoscene_smoke.py`: CPU smoke test

## Installation

`requirements.txt` is the only official dependency specification for this repo.

```bash
conda create -n reva python=3.10 pip
conda activate reva
pip install -r requirements.txt
```

Optional:

```bash
pip install mamba-ssm causal-conv1d
```

The current official stack follows the pinned CUDA build in `requirements.txt`.

## Environment

```bash
export PYTHONPATH="$PWD/ReMoScene:$PWD/ReMoScene/src:$PYTHONPATH"
```

## Training Data Format

Training expects JSONL. A minimal sample:

```json
{
  "id": "sample_0001",
  "video": "videos/demo.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nDescribe the scene."},
    {"from": "gpt", "value": "A person is walking through a hallway."}
  ]
}
```

## Convert Local HF/Parquet Data

```bash
python ReMoScene/convert_hf_video_to_reva.py \
  --input_path /path/to/dataset_or_parquet \
  --output_path data/train.jsonl \
  --dataset_split train \
  --video_root /path/to/video_root
```

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

## Notes

- Project name: `ReVA`
- Model/module name: `ReMoScene`
- No Slurm dependency is required in this repo
