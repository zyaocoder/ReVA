import copy
import math
import os
import random
from pathlib import Path
from typing import Any

import ujson as json


VIDEO_FIELD_CANDIDATES = (
    "video",
    "videos",
    "video_path",
    "video_paths",
    "video_file",
    "video_files",
)
IMAGE_FIELD_CANDIDATES = (
    "image",
    "images",
    "image_path",
    "image_paths",
    "image_file",
    "image_files",
)


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                chunks.append(str(item))
                continue

            item_type = item.get("type")
            if item_type in {"text", None}:
                text = item.get("text") or item.get("value") or item.get("content") or ""
                if text:
                    chunks.append(str(text))
            elif item_type == "image":
                chunks.append("<image>")
            elif item_type == "video":
                chunks.append("<video>")
            else:
                text = item.get("text") or item.get("value") or item.get("content")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunk for chunk in chunks if chunk)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "value" in content:
            return str(content["value"])
        if "content" in content:
            return _message_content_to_text(content["content"])
    return str(content)


def _normalize_conversations(conversations: Any) -> list[dict[str, str]]:
    if not isinstance(conversations, list):
        raise ValueError("Expected a list for conversations/messages.")

    normalized = []
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError(f"Conversation item must be a dict, got {type(message)!r}.")

        if "from" in message or "value" in message:
            role = message.get("from")
            value = _message_content_to_text(message.get("value"))
        else:
            role = message.get("role")
            value = _message_content_to_text(message.get("content"))

        if role in {"user", "human"}:
            role = "human"
        elif role in {"assistant", "gpt", "model"}:
            role = "gpt"
        elif role is None:
            raise ValueError(f"Conversation item missing role/from: {message}")

        normalized.append({"from": role, "value": value})

    return normalized


def _extract_media_value(media_value: Any) -> Any:
    if isinstance(media_value, list):
        return [_extract_media_value(item) for item in media_value]
    if isinstance(media_value, dict):
        for key in ("path", "file_path", "video_path", "image_path", "fname", "name"):
            value = media_value.get(key)
            if isinstance(value, str) and value:
                return value
        return media_value
    return media_value


def normalize_sample(record: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(record)

    conversations = (
        normalized.get("conversations")
        or normalized.get("messages")
        or normalized.get("conversation")
    )
    if conversations is None:
        raise ValueError(f"Sample is missing conversations/messages keys: {record.keys()}")
    normalized["conversations"] = _normalize_conversations(conversations)

    for field_candidates, canonical_key in (
        (VIDEO_FIELD_CANDIDATES, "video"),
        (IMAGE_FIELD_CANDIDATES, "image"),
    ):
        for key in field_candidates:
            if key in normalized and normalized[key] is not None:
                normalized[canonical_key] = _extract_media_value(normalized[key])
                break

    return normalized


def _load_jsonl_records(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json_records(path: str, dataset_split: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    if isinstance(data, dict) and isinstance(data.get("datasets"), list):
        return _load_manifest_records(data, os.path.dirname(path), dataset_split=dataset_split)

    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_hf_records(path: str, dataset_split: str, parquet_files: list[str] | None = None) -> list[dict[str, Any]]:
    from datasets import DatasetDict, load_dataset, load_from_disk

    if parquet_files:
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    elif os.path.isdir(path):
        dataset = load_from_disk(path)
    else:
        dataset = load_dataset("parquet", data_files=[path], split="train")

    if isinstance(dataset, DatasetDict):
        if dataset_split not in dataset:
            raise ValueError(f"Split '{dataset_split}' not found in {path}. Available: {list(dataset.keys())}")
        dataset = dataset[dataset_split]

    return [dict(record) for record in dataset]


def _find_recursive_files(base_dir: str, patterns: tuple[str, ...]) -> list[str]:
    matches = []
    for pattern in patterns:
        matches.extend(str(path) for path in Path(base_dir).rglob(pattern))
    return sorted(set(matches))


def _apply_sampling_strategy(records: list[dict[str, Any]], sampling_strategy: str) -> list[dict[str, Any]]:
    sampling_number = None
    strategy = sampling_strategy or "all"

    if ":" in strategy:
        strategy, sampling_number = strategy.split(":", 1)
        if "%" in sampling_number:
            sampling_number = math.ceil(int(sampling_number.split("%", 1)[0]) * len(records) / 100)
        else:
            sampling_number = int(sampling_number)

    if sampling_number is None or strategy == "all":
        return records
    if strategy == "first":
        return records[:sampling_number]
    if strategy == "end":
        return records[-sampling_number:]
    if strategy == "random":
        sampled = list(records)
        random.shuffle(sampled)
        return sampled[:sampling_number]

    raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")


def _annotate_records(
    records: list[dict[str, Any]],
    source_dir: str | None = None,
    video_folder: str | None = None,
    image_folder: str | None = None,
) -> list[dict[str, Any]]:
    annotated = []
    for record in records:
        item = normalize_sample(record)
        if source_dir:
            item.setdefault("_source_dir", source_dir)
        if video_folder:
            item.setdefault("video_folder", video_folder)
        if image_folder:
            item.setdefault("image_folder", image_folder)
        annotated.append(item)
    return annotated


def _load_manifest_records(manifest_data: dict[str, Any], base_dir: str, dataset_split: str) -> list[dict[str, Any]]:
    datasets = manifest_data.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Manifest must contain a top-level 'datasets' list.")

    all_records = []
    for dataset_cfg in datasets:
        if not isinstance(dataset_cfg, dict):
            raise ValueError("Each manifest entry must be a mapping.")

        data_path = (
            dataset_cfg.get("json_path")
            or dataset_cfg.get("data_path")
            or dataset_cfg.get("path")
            or dataset_cfg.get("parquet_path")
        )
        if not data_path:
            raise ValueError(f"Manifest entry missing data path: {dataset_cfg}")

        if not os.path.isabs(data_path):
            data_path = os.path.join(base_dir, data_path)

        records = _load_records_from_path(data_path, dataset_split=dataset_split)
        records = _apply_sampling_strategy(records, dataset_cfg.get("sampling_strategy", "all"))
        records = _annotate_records(
            records,
            source_dir=os.path.dirname(data_path),
            video_folder=dataset_cfg.get("video_folder"),
            image_folder=dataset_cfg.get("image_folder"),
        )
        all_records.extend(records)

    return all_records


def _load_records_from_path(path: str, dataset_split: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        return _annotate_records(_load_jsonl_records(path), source_dir=os.path.dirname(path))
    if path.endswith(".json"):
        return _annotate_records(_load_json_records(path, dataset_split=dataset_split), source_dir=os.path.dirname(path))
    if path.endswith((".yaml", ".yml")):
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            manifest_data = yaml.safe_load(f)
        return _load_manifest_records(manifest_data, os.path.dirname(path), dataset_split=dataset_split)
    if path.endswith(".parquet"):
        return _annotate_records(_load_hf_records(path, dataset_split=dataset_split), source_dir=os.path.dirname(path))

    if os.path.isdir(path):
        dataset_markers = ("dataset_info.json", "dataset_dict.json", "state.json")
        if any(os.path.exists(os.path.join(path, marker)) for marker in dataset_markers):
            return _annotate_records(_load_hf_records(path, dataset_split=dataset_split), source_dir=path)

        processed_jsons = _find_recursive_files(path, ("*_processed.json",))
        if processed_jsons:
            records = []
            for file_path in processed_jsons:
                records.extend(_load_records_from_path(file_path, dataset_split=dataset_split))
            return records

        jsonls = _find_recursive_files(path, ("*.jsonl",))
        if jsonls:
            records = []
            for file_path in jsonls:
                records.extend(_load_records_from_path(file_path, dataset_split=dataset_split))
            return records

        parquets = _find_recursive_files(path, ("*.parquet",))
        if parquets:
            return _annotate_records(
                _load_hf_records(path, dataset_split=dataset_split, parquet_files=parquets),
                source_dir=path,
            )

        jsons = _find_recursive_files(path, ("*.json",))
        if len(jsons) == 1:
            return _load_records_from_path(jsons[0], dataset_split=dataset_split)

        raise ValueError(f"Could not infer dataset files under directory: {path}")

    raise ValueError(f"Unsupported data path: {path}")


def load_records(data_path: str | list[dict[str, Any]], dataset_split: str = "train") -> list[dict[str, Any]]:
    if isinstance(data_path, list):
        return [normalize_sample(record) for record in data_path]
    return _load_records_from_path(data_path, dataset_split=dataset_split)
