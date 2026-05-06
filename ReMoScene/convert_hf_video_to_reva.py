"""
Convert a local Hugging Face video dataset / parquet shard / manifest into
path-based ReVA JSONL records consumable by ReMoScene/src/train/train_remoscene.py.
"""

import argparse
import os
import sys

import ujson as json


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = _THIS_DIR
_SRC_DIR = os.path.join(_PACKAGE_DIR, "src")
for _path in (_PACKAGE_DIR, _SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.dataset.record_utils import load_records


def _extract_media_path(media_value, media_kind: str, extract_dir: str | None, sample_id: str, index: int):
    if isinstance(media_value, list):
        return [
            _extract_media_path(item, media_kind, extract_dir, sample_id, idx)
            for idx, item in enumerate(media_value)
        ]

    if isinstance(media_value, str):
        return media_value

    if not isinstance(media_value, dict):
        raise ValueError(f"Unsupported {media_kind} value: {media_value}")

    for key in ("path", "file_path", "video_path", "image_path", "fname", "name"):
        value = media_value.get(key)
        if isinstance(value, str) and value:
            return value

    media_bytes = media_value.get("bytes")
    if media_bytes is None:
        raise ValueError(f"Could not resolve a filesystem path for {media_kind}: {media_value}")
    if extract_dir is None:
        raise ValueError(
            f"{media_kind} contains embedded bytes. Pass --extract_dir to materialize them to disk."
        )

    extension = media_value.get("extension")
    if not extension:
        extension = "mp4" if media_kind == "video" else "jpg"
    extension = extension.lstrip(".")

    target_dir = os.path.join(extract_dir, media_kind)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"{sample_id}_{index}.{extension}")
    with open(target_path, "wb") as f:
        f.write(media_bytes)
    return target_path


def convert_dataset(input_path: str, output_path: str, dataset_split: str, video_root: str | None, image_root: str | None,
                    extract_dir: str | None, limit: int | None) -> int:
    records = load_records(input_path, dataset_split=dataset_split)
    if limit is not None:
        records = records[:limit]

    written = 0
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records):
            sample = dict(record)
            sample_id = str(sample.get("id") or sample.get("video_id") or f"sample_{idx}")

            if "video" in sample:
                sample["video"] = _extract_media_path(sample["video"], "video", extract_dir, sample_id, idx)
                if video_root and isinstance(sample["video"], str) and not os.path.isabs(sample["video"]):
                    sample["video_folder"] = video_root
            if "image" in sample:
                sample["image"] = _extract_media_path(sample["image"], "image", extract_dir, sample_id, idx)
                if image_root and isinstance(sample["image"], str) and not os.path.isabs(sample["image"]):
                    sample["image_folder"] = image_root

            sample.pop("_source_dir", None)

            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description="Convert a local HF/parquet video dataset to ReVA JSONL.")
    parser.add_argument("--input_path", required=True, help="Dataset directory, parquet file/dir, json/jsonl, or yaml manifest.")
    parser.add_argument("--output_path", required=True, help="Output JSONL path.")
    parser.add_argument("--dataset_split", default="train", help="Split to use for local HF datasets. Default: train.")
    parser.add_argument("--video_root", default=None, help="Root directory for relative video paths in the output JSONL.")
    parser.add_argument("--image_root", default=None, help="Root directory for relative image paths in the output JSONL.")
    parser.add_argument("--extract_dir", default=None, help="Directory to write embedded media bytes if the dataset stores videos/images inline.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke conversion.")
    args = parser.parse_args()

    written = convert_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        dataset_split=args.dataset_split,
        video_root=args.video_root,
        image_root=args.image_root,
        extract_dir=args.extract_dir,
        limit=args.limit,
    )
    print(f"Wrote {written} samples to {args.output_path}")


if __name__ == "__main__":
    main()
