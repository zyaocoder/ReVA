"""
Convert RSVidQA JSON format to LLaVA conversations JSONL format.

Input format (5_train_set.json / 5_val_set.json / 5_test_set.json):
{
  "metadata": {"total_videos": 646, "total_questions": 8628},
  "videos": {
    "video_id": {
      "file_path": "path/to/video.mp4",
      "mcq": {
        "category_name": {
          "subcategory_name": [
            {
              "question": "...",
              "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
              "correct_answer": "B",
              "reasoning": "..."
            }
          ]
        }
      }
    }
  }
}

Output format (LLaVA JSONL, one JSON object per line):
{"video": "path.mp4", "conversations": [
  {"from": "human", "value": "<video>\nQ?\nA. ...\nB. ...\nC. ...\nD. ..."},
  {"from": "gpt",   "value": "B"}
]}
"""

import argparse
import json
import os


LLAVA_VIDEO_TOKEN = "<video>"
OPTION_KEYS = ["A", "B", "C", "D"]


def build_question_text(question: str, options: dict) -> str:
    """Format MCQ question with options into a human turn."""
    lines = [LLAVA_VIDEO_TOKEN, question]
    for k in OPTION_KEYS:
        if k in options:
            lines.append(f"{k}. {options[k]}")
    return "\n".join(lines)


RSVID_PATH_PREFIX = "#dataset/RSVidQA/"


def convert_split(input_path: str, output_path: str, video_root: str | None = None) -> int:
    """
    Convert one split JSON file to JSONL.

    Args:
        input_path:  path to the RSVidQA JSON file
        output_path: path to write the output JSONL
        video_root:  root directory that replaces the '#dataset/RSVidQA/' placeholder
                     in file_path values. If None, file_path is used as-is.

    Returns:
        number of QA pairs written
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", {})
    count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for vid_id, vid_info in videos.items():
            file_path = vid_info.get("file_path", "")
            if video_root:
                # Replace '#dataset/RSVidQA/' placeholder with the actual root
                if file_path.startswith(RSVID_PATH_PREFIX):
                    file_path = os.path.join(video_root, file_path[len(RSVID_PATH_PREFIX):])
                elif not os.path.isabs(file_path):
                    file_path = os.path.join(video_root, file_path)

            mcq = vid_info.get("mcq", {})
            for category, subcategories in mcq.items():
                for subcategory, qa_list in subcategories.items():
                    for qa in qa_list:
                        question = qa.get("question", "")
                        options = qa.get("options", {})
                        correct = qa.get("correct_answer", "")

                        # Skip malformed entries
                        if not question or not options or not correct:
                            continue

                        human_text = build_question_text(question, options)

                        entry = {
                            "video": file_path,
                            "conversations": [
                                {"from": "human", "value": human_text},
                                {"from": "gpt",   "value": correct},
                            ],
                            # preserve metadata for analysis / filtering
                            "category": category,
                            "subcategory": subcategory,
                            "video_id": vid_id,
                        }

                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert RSVidQA JSON splits to LLaVA JSONL format."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to RSVidQA JSON file (e.g. 5_train_set.json)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path for output JSONL file"
    )
    parser.add_argument(
        "--video_root", default=None,
        help="Root directory that replaces the '#dataset/RSVidQA/' placeholder "
             "in file_path values (e.g. /anvil/projects/x-cis251076/likai/RSVidQA)"
    )
    args = parser.parse_args()

    n = convert_split(args.input, args.output, args.video_root)
    print(f"Written {n} QA pairs to {args.output}")


if __name__ == "__main__":
    main()
