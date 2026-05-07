# ReVA Dataset Instruction

We introduce ReVA, a scene-centric remote sensing video question answering dataset comprising 2,798 UAV videos spanning 580K frames across 18 cities worldwide. ReVA contains 22K high-quality human-annotated question-answer pairs with 17K unique questions. To the best of our knowledge, ReVA is the first remote sensing dataset for video understanding.

## Directory Layout

This directory contains the ReVA video files and the three annotation splits of training, validation, and test sets. The dataset is organized under these top-level directories:

- `Hawk_UAV/`
- `VisDrone/`
- `UAVDT/`
- `ERA_Select/`

Our self-coolected Hawk_UAV videos are grouped by region:

- `Hawk_UAV/BE/`
- `Hawk_UAV/IL/`
- `Hawk_UAV/NJ/`
- `Hawk_UAV/PH/`

Current layout summary:

```text
ReVA/
├── Hawk_UAV/
│   ├── BE/
│   ├── IL/
│   ├── NJ/
│   └── PH/
├── VisDrone/
├── UAVDT/
├── ERA_Select/
├── train_set.json
├── valid_set.json
└── test_set.json

Each file contains:

- `metadata`: split-level statistics and source information
- `videos`: per-video annotations

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
      "file_path": "VisDrone/uav0000009_03358_v_01.mp4",
      "subdir": "videos",
      "consolidated_caption": "...",
      "dataset_name": "ReVA",
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
  --input ReVA/train_set.json \
  --output data/train.jsonl \
  --video_root /path/to/video_root
```