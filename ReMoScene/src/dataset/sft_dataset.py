import copy
import os
import tarfile
import tempfile
from typing import Dict
import torch
import transformers
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, conversations_to_openai, pad_sequence
from .record_utils import load_records
import random

_TAR_INDEX_CACHE = {}


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        self.data_path = data_path

        if isinstance(data_path, str):
            list_data_dict = load_records(data_path, dataset_split=data_args.dataset_split)
        else:
            list_data_dict = load_records(data_path, dataset_split=data_args.dataset_split)

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.max_frames = data_args.max_frames

    def __len__(self):
        return len(self.list_data_dict)

    def _resolve_from_tar_archives(self, media_path: str, source_dir: str | None):
        if not source_dir or not os.path.isdir(source_dir):
            return None

        tar_files = sorted(
            os.path.join(source_dir, name)
            for name in os.listdir(source_dir)
            if name.endswith(".tar.gz")
        )
        if not tar_files:
            return None

        member_name = media_path.lstrip("./")
        tar_index = _TAR_INDEX_CACHE.get(source_dir)
        if tar_index is None:
            tar_index = {}
            for tar_path in tar_files:
                try:
                    with tarfile.open(tar_path, "r:gz") as tar:
                        for member in tar.getmembers():
                            if member.isfile():
                                tar_index[member.name.lstrip("./")] = tar_path
                except tarfile.TarError:
                    continue
            _TAR_INDEX_CACHE[source_dir] = tar_index

        tar_path = tar_index.get(member_name)
        if tar_path is None:
            return None

        cache_root = os.path.join(
            tempfile.gettempdir(),
            "reva_video_extracted",
            os.path.basename(source_dir),
        )
        extracted_path = os.path.join(cache_root, member_name)
        if os.path.exists(extracted_path):
            return extracted_path

        os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            member = tar.getmember(member_name)
            src = tar.extractfile(member)
            if src is None:
                raise FileNotFoundError(f"Failed to extract {member_name} from {tar_path}")
            with open(extracted_path, "wb") as dst:
                dst.write(src.read())

        return extracted_path

    def _resolve_media_path(self, media_path, sample_folder=None, global_folder=None, source_dir=None, media_kind="media"):
        if isinstance(media_path, dict):
            for key in ("path", "file_path", "video_path", "image_path", "fname", "name"):
                value = media_path.get(key)
                if isinstance(value, str) and value:
                    media_path = value
                    break
            else:
                if media_path.get("bytes") is not None:
                    raise ValueError(
                        f"{media_kind} is stored as embedded bytes. Convert it to path-based JSONL first."
                    )
                raise ValueError(f"Unsupported {media_kind} descriptor: {media_path}")

        if not isinstance(media_path, str) or not media_path:
            raise ValueError(f"Invalid {media_kind} path: {media_path}")

        if media_path.startswith("http") or os.path.isabs(media_path) or os.path.exists(media_path):
            return media_path

        candidates = []
        if sample_folder:
            candidates.append(os.path.join(sample_folder, media_path))
        if global_folder:
            candidates.append(os.path.join(global_folder, media_path))
        if source_dir:
            candidates.append(os.path.join(source_dir, media_path))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        if media_kind == "video":
            tar_resolved = self._resolve_from_tar_archives(media_path, source_dir)
            if tar_resolved is not None:
                return tar_resolved

        return candidates[0] if candidates else media_path

    def _get_item(self, i):
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = sources.get("image_folder")
            global_image_folder = self.data_args.image_folder
            source_dir = sources.get("_source_dir")

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            for image_file in image_files:
                image_file = self._resolve_media_path(
                    image_file,
                    sample_folder=image_folder,
                    global_folder=global_image_folder,
                    source_dir=source_dir,
                    media_kind="image",
                )
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = sources.get("video_folder")
            global_video_folder = self.data_args.image_folder
            source_dir = sources.get("_source_dir")

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                video_file = self._resolve_media_path(
                    video_file,
                    sample_folder=video_folder,
                    global_folder=global_video_folder,
                    source_dir=source_dir,
                    media_kind="video",
                )
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.data_args.fps, self.max_frames)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(conversations_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        
        # FIXME
        # if input_ids.size(0) > 4096:
        #     print(input_ids.size(0), "input_ids is too long, using random sample")
        #     return self._get_item(random.randint(0, len(self.list_data_dict) - 1))

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            return self._get_item(i)
        except Exception as e:
            print(e)
            print("error in :")
            print(self.list_data_dict[i])
            return self._get_item(0)


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    
def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
