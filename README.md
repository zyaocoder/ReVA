# ReVA

ReVA is a lightweight repository for our video-centric multimodal model work.
This repo currently focuses on **ReMoScene**, a trainable video compression and
motion reasoning module built on top of a multimodal backbone adapted from
**LLaVA-ov-1.5**.

## Repository Layout

- `ReMoScene/backbone/remoscene.py`: ReMoScene core implementation
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

The raw ReVA split files are JSON files such as:

```json
{
  "metadata": {
    "total_videos": 646,
    "total_questions": 8628
  },
  "videos": {
    "DJI_0157_d4_01": {
      "file_path": "#dataset/ReVA_V2/videos/DJI_0157_d4_01.mp4",
      "subdir": "videos",
      "consolidated_caption": "...",
      "dataset_name": "ReVA_V2",
      "mcq": {
        "Temporal Understanding": {
          "Temporal Grounding": [
            {
              "question": "...",
              "options": {
                "A": "...",
                "B": "...",
                "C": "...",
                "D": "..."
              },
              "correct_answer": "B",
              "reasoning": "...",
              "example": "...",
              "qa_id": "..."
            }
          ]
        }
      }
    }
  }
}
```

`train_remoscene.py` consumes JSONL after conversion. A minimal converted record looks like:

```json
{
  "video": "videos/DJI_0157_d4_01.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nQuestion text\nA. ...\nB. ...\nC. ...\nD. ..."},
    {"from": "gpt", "value": "B"}
  ],
  "category": "Temporal Understanding",
  "subcategory": "Temporal Grounding",
  "video_id": "DJI_0157_d4_01"
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

For the current ReVA split JSON files, use:

```bash
python ReMoScene/convert_videoqa_to_reva.py \
  --input /home/liw324/code/VLM-Baselines/#dataset/ReVA_V2/train_set.json \
  --output data/train.jsonl \
  --video_root /home/liw324/code/VLM-Baselines/#dataset/ReVA_V2
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
