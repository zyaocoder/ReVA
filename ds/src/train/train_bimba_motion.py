"""
train_bimba_motion.py
=====================
Training script for the BIMBA + Motion extension of LLaVA-OneVision-1.5.

Training strategy
-----------------
  Frozen  : RICE vision encoder  +  Qwen2 LLM (all base-model weights)
  Trained : BimbaSelector  ·  MotionBranch  ·  BimbaMotionProjector  ·  gate

Usage example
-------------
torchrun --nproc_per_node=4 train_bimba_motion.py \\
    --model_id /path/to/LLaVA-OneVision-1.5 \\
    --data_path /path/to/train.jsonl \\
    --image_folder /path/to/videos \\
    --output_dir ./checkpoints/bimba_motion \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 8 \\
    --learning_rate 2e-4 \\
    --bf16 True \\
    --fps 1.0 \\
    --num_queries 256
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    TrainerCallback,
    Trainer,
    set_seed,
)

# Use the custom TrainingArguments from params.py which extends HF's with
# LoRA fields (lora_enable, lora_rank, lora_alpha, lora_dropout, lora_bias).
from src.params import TrainingArguments

# ---- Path setup -------------------------------------------------------
# Ensure the ds/ subdirectories are importable regardless of cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DS_DIR   = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # ds/
_SRC_DIR  = os.path.join(_DS_DIR, "src")
for _p in (_DS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llavaonevision1_5.bimba_motion import (
    BimbaMotionConfig,
    BimbaMotionWrapper,
    build_bimba_motion_model,
)
from src.dataset import make_supervised_data_module   # reuse existing loader
from src.params import DataArguments, ModelArguments  # reuse existing params

logger = logging.getLogger(__name__)


def _series_points_to_svg(points, title="Training Series", color="#2563eb", value_label="value") -> str:
    width = 960
    height = 540
    left = 70
    right = 30
    top = 45
    bottom = 55
    plot_w = width - left - right
    plot_h = height - top - bottom

    if not points:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="white"/>'
            f'<text x="{width/2}" y="{height/2}" text-anchor="middle" fill="#444" '
            f'font-family="sans-serif" font-size="24">No points yet</text></svg>'
        )

    steps = [p[0] for p in points]
    values = [p[1] for p in points]
    min_step = min(steps)
    max_step = max(steps)
    min_value = min(values)
    max_value = max(values)

    if max_step == min_step:
        max_step = min_step + 1
    if max_value == min_value:
        pad = 0.5 if max_value == 0 else abs(max_value) * 0.05
        min_value -= pad
        max_value += pad

    def x_map(step):
        return left + (step - min_step) / (max_step - min_step) * plot_w

    def y_map(value):
        return top + (max_value - value) / (max_value - min_value) * plot_h

    path = " ".join(
        f"{'M' if idx == 0 else 'L'} {x_map(step):.2f} {y_map(value):.2f}"
        for idx, (step, value) in enumerate(points)
    )

    tick_lines = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + frac * plot_h
        value = max_value - frac * (max_value - min_value)
        tick_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" fill="#374151" font-family="sans-serif" font-size="12">{value:.4f}</text>'
        )

    x_ticks = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + frac * plot_w
        step = min_step + frac * (max_step - min_step)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#f3f4f6" stroke-width="1"/>'
            f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" fill="#374151" font-family="sans-serif" font-size="12">{step:.0f}</text>'
        )

    latest_step, latest_value = points[-1]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" fill="#111827" font-family="sans-serif" font-size="22">{title}</text>
<text x="{width-right}" y="28" text-anchor="end" fill="#4b5563" font-family="sans-serif" font-size="13">latest step={latest_step}, {value_label}={latest_value:.4f}</text>
{''.join(tick_lines)}
{''.join(x_ticks)}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>
<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="{x_map(latest_step):.2f}" cy="{y_map(latest_value):.2f}" r="4" fill="#dc2626"/>
<text x="{width/2}" y="{height-12}" text-anchor="middle" fill="#374151" font-family="sans-serif" font-size="13">global step</text>
</svg>'''


def _loss_points_to_svg(points, title="Training Loss") -> str:
    return _series_points_to_svg(points, title=title, color="#2563eb", value_label="loss")


def _maybe_write_png(points, png_path, title="Training Loss", color="#2563eb", value_label="value"):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    width, height = 960, 540
    left, right, top, bottom = 70, 30, 45, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    if not points:
        draw.text((width // 2 - 60, height // 2), "No loss points yet", fill="black")
        image.save(png_path)
        return

    steps = [p[0] for p in points]
    values = [p[1] for p in points]
    min_step, max_step = min(steps), max(steps)
    min_value, max_value = min(values), max(values)
    if max_step == min_step:
        max_step += 1
    if max_value == min_value:
        pad = 0.5 if max_value == 0 else abs(max_value) * 0.05
        min_value -= pad
        max_value += pad

    def x_map(step):
        return left + (step - min_step) / (max_step - min_step) * plot_w

    def y_map(value):
        return top + (max_value - value) / (max_value - min_value) * plot_h

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + frac * plot_h
        x = left + frac * plot_w
        draw.line((left, y, width - right, y), fill="#e5e7eb", width=1)
        draw.line((x, top, x, height - bottom), fill="#f3f4f6", width=1)

    draw.line((left, height - bottom, width - right, height - bottom), fill="#111827", width=2)
    draw.line((left, top, left, height - bottom), fill="#111827", width=2)

    coords = [(x_map(step), y_map(value)) for step, value in points]
    if len(coords) >= 2:
        draw.line(coords, fill=color, width=3)
    last_x, last_y = coords[-1]
    draw.ellipse((last_x - 4, last_y - 4, last_x + 4, last_y + 4), fill="#dc2626")
    draw.text((width // 2 - 45, 10), title, fill="#111827")
    draw.text((width - 280, 10), f"step={points[-1][0]} {value_label}={points[-1][1]:.4f}", fill="#4b5563")
    image.save(png_path)


class LiveLossPlotCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.svg_path = os.path.join(output_dir, "train_loss.svg")
        self.png_path = os.path.join(output_dir, "train_loss.png")
        self.json_path = os.path.join(output_dir, "train_loss_points.json")
        self.metrics_json_path = os.path.join(output_dir, "train_metrics_points.json")
        self.mem_svg_path = os.path.join(output_dir, "gpu_memory_reserved.svg")
        self.mem_png_path = os.path.join(output_dir, "gpu_memory_reserved.png")
        self.points = []
        self.metrics = []
        os.makedirs(output_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        loss = logs.get("loss")
        if loss is None:
            return

        point = (int(state.global_step), float(loss))
        self.points.append(point)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump([{"step": step, "loss": loss_val} for step, loss_val in self.points], f, ensure_ascii=False, indent=2)
        with open(self.svg_path, "w", encoding="utf-8") as f:
            f.write(_loss_points_to_svg(self.points))
        _maybe_write_png(self.points, self.png_path, title="Training Loss", color="#2563eb", value_label="loss")

        metrics_row = {
            "step": int(state.global_step),
            "loss": float(loss),
        }
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            metrics_row.update({
                "gpu_index": int(device_idx),
                "allocated_gb": float(torch.cuda.memory_allocated(device_idx) / (1024 ** 3)),
                "reserved_gb": float(torch.cuda.memory_reserved(device_idx) / (1024 ** 3)),
                "max_allocated_gb": float(torch.cuda.max_memory_allocated(device_idx) / (1024 ** 3)),
                "max_reserved_gb": float(torch.cuda.max_memory_reserved(device_idx) / (1024 ** 3)),
            })
        self.metrics.append(metrics_row)
        with open(self.metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

        if self.metrics and "reserved_gb" in self.metrics[-1]:
            mem_points = [(row["step"], row["reserved_gb"]) for row in self.metrics if "reserved_gb" in row]
            with open(self.mem_svg_path, "w", encoding="utf-8") as f:
                f.write(_series_points_to_svg(mem_points, title="GPU Memory Reserved (GB)", color="#059669", value_label="GB"))
            _maybe_write_png(mem_points, self.mem_png_path, title="GPU Memory Reserved (GB)", color="#059669", value_label="GB")
            latest = self.metrics[-1]
            print(
                "[gpu-mem] "
                f"step={latest['step']} "
                f"gpu={latest['gpu_index']} "
                f"allocated={latest['allocated_gb']:.2f}GB "
                f"reserved={latest['reserved_gb']:.2f}GB "
                f"max_allocated={latest['max_allocated_gb']:.2f}GB "
                f"max_reserved={latest['max_reserved_gb']:.2f}GB"
            )

# ============================================================================
# Extra argument dataclasses
# ============================================================================

@dataclass
class BimbaArguments:
    """Hyper-parameters for the BIMBA + Motion modules."""

    num_queries        : int   = field(default=256,  metadata={"help": "Number of compressed video tokens (Lq) for BimbaSelector."})
    n_bimba_layers     : int   = field(default=2,    metadata={"help": "BIMBA selector depth."})
    bimba_nhead        : int   = field(default=8,    metadata={"help": "Attention heads (fallback block)."})
    bimba_expand       : int   = field(default=2,    metadata={"help": "Inner-dim expansion factor."})
    bimba_d_state      : int   = field(default=16,   metadata={"help": "Mamba SSM state dim."})
    bimba_d_conv       : int   = field(default=4,    metadata={"help": "Mamba conv kernel size."})
    lq_t               : int   = field(default=4,    metadata={"help": "Temporal pool targets for query init (lq_t*lq_h*lq_w should = num_queries)."})
    lq_h               : int   = field(default=8,    metadata={"help": "Height pool target for query init."})
    lq_w               : int   = field(default=8,    metadata={"help": "Width pool target for query init."})
    num_object_queries : int   = field(default=16,   metadata={"help": "Learnable object queries per frame in RecurrentObjectSelector (N)."})
    rec_n_layers       : int   = field(default=2,    metadata={"help": "Per-step Transformer depth in RecurrentObjectSelector."})
    rec_nhead          : int   = field(default=8,    metadata={"help": "Attention heads for RecurrentObjectSelector."})
    trainable_ckpt        : Optional[str] = field(default=None, metadata={"help": "Path to a previously saved trainable-module checkpoint to resume from."})
    use_bimba_selector    : bool          = field(default=True,  metadata={"help": "Whether to use BimbaSelector (learned compression). False = 3D-pool only."})
    use_motion_branch     : bool          = field(default=True,  metadata={"help": "Whether to use MotionBranch (global motion token + GRU)."})
    use_recurrent_selector: bool          = field(default=True,  metadata={"help": "Whether to use RecurrentObjectSelector (object query tokens)."})


# ============================================================================
# Helpers
# ============================================================================

local_rank = None


def rank0_print(*args):
    if local_rank in (0, "0", None):
        print(*args)


def count_parameters(model: torch.nn.Module) -> dict:
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ============================================================================
# Custom Trainer subclass
# ============================================================================

class BimbaMotionTrainer(Trainer):
    """
    Minimal Trainer extension that:
      • Saves only the trainable (new) modules at checkpoint time.
      • Restores them on resume.
    """

    # Prefixes that identify our trainable new modules (not the frozen base model)
    _TRAINABLE_PREFIXES = (
        "bimba_selector.",
        "motion_branch.",
        "recurrent_selector.",
        "projector.",
        "motion_gate_logit",
    )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override to save trainable-module state dict (+ LoRA weights if enabled).

        With DeepSpeed ZeRO-3 the model parameters are sharded across GPUs, so
        calling model.state_dict() / trainable_state_dict() directly yields
        empty tensors.  We instead extract the trainable keys from the
        model.safetensors that super()._save() writes (which is ZeRO-3 safe),
        or fall back to direct state_dict() when DeepSpeed is not in use.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Always save the full HF-style trainer state (ZeRO-3 safe)
        super()._save(output_dir, state_dict=state_dict)

        # Only rank 0 writes the compact checkpoint
        if self.args.process_index != 0:
            return

        # Build trainable-only state dict.
        # Prefer reading back from the just-saved model.safetensors so that
        # ZeRO-3 gathered tensors are used; fall back to direct state_dict().
        sf_path = os.path.join(output_dir, "model.safetensors")
        if os.path.exists(sf_path):
            try:
                from safetensors import safe_open
                trainable_sd = {}
                with safe_open(sf_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if any(k.startswith(p) for p in self._TRAINABLE_PREFIXES):
                            trainable_sd[k] = f.get_tensor(k)
            except Exception as e:
                rank0_print(f"[BimbaMotionTrainer] safetensors read failed ({e}), falling back to state_dict()")
                trainable_sd = self.model.trainable_state_dict()
        else:
            trainable_sd = self.model.trainable_state_dict()

        ckpt_path = os.path.join(output_dir, "bimba_motion_modules.pt")
        torch.save(trainable_sd, ckpt_path)
        rank0_print(f"[BimbaMotionTrainer] Saved trainable modules to {ckpt_path}")

        # Save LoRA weights separately when LoRA is enabled
        if getattr(self.args, "lora_enable", False):
            if os.path.exists(sf_path):
                try:
                    from safetensors import safe_open
                    lora_sd = {}
                    with safe_open(sf_path, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            if "lora_" in k:
                                lora_sd[k] = f.get_tensor(k)
                except Exception:
                    lora_sd = {k: v for k, v in self.model.state_dict().items() if "lora_" in k}
            else:
                lora_sd = {k: v for k, v in self.model.state_dict().items() if "lora_" in k}
            lora_path = os.path.join(output_dir, "lora_weights.pt")
            torch.save(lora_sd, lora_path)
            rank0_print(f"[BimbaMotionTrainer] Saved LoRA weights to {lora_path}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss    = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Main
# ============================================================================

def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, BimbaArguments)
    )
    model_args, data_args, training_args, bimba_args = \
        parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    set_seed(training_args.seed)

    # ---- Compute dtype ------------------------------------------------
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    rank0_print("=" * 70)
    rank0_print("BIMBA + Motion  ·  LLaVA-OneVision-1.5")
    rank0_print(f"  base model      : {model_args.model_id}")
    rank0_print(f"  num_queries (Lq): {bimba_args.num_queries}")
    rank0_print(f"  mamba_ssm       : {'available' if _mamba_available() else 'fallback (attention)'}")
    rank0_print("=" * 70)

    # ---- Build BIMBA config -------------------------------------------
    bimba_cfg = BimbaMotionConfig(
        num_queries        = bimba_args.num_queries,
        n_bimba_layers     = bimba_args.n_bimba_layers,
        bimba_nhead        = bimba_args.bimba_nhead,
        bimba_expand       = bimba_args.bimba_expand,
        bimba_d_state      = bimba_args.bimba_d_state,
        bimba_d_conv       = bimba_args.bimba_d_conv,
        lq_t               = bimba_args.lq_t,
        lq_h               = bimba_args.lq_h,
        lq_w               = bimba_args.lq_w,
        num_object_queries = bimba_args.num_object_queries,
        rec_n_layers       = bimba_args.rec_n_layers,
        rec_nhead          = bimba_args.rec_nhead,
        use_bimba_selector     = bimba_args.use_bimba_selector,
        use_motion_branch      = bimba_args.use_motion_branch,
        use_recurrent_selector = bimba_args.use_recurrent_selector,
    )

    # ---- Load model ---------------------------------------------------
    rank0_print("Loading frozen base model …")
    model = build_bimba_motion_model(
        base_model_path = model_args.model_id,
        bimba_cfg       = bimba_cfg,
        torch_dtype     = compute_dtype,
        device_map      = "auto" if not training_args.deepspeed else None,
    )
    model.set_trainable()

    # ---- Optional: LoRA on the LLM ------------------------------------
    # When lora_enable=True, we inject trainable LoRA adapters into the
    # Qwen2 language model while keeping our compression modules trainable.
    # Both sets of parameters are optimised simultaneously.
    if training_args.lora_enable:
        # bitsandbytes may be partially installed (CUDA setup failed) and lack
        # the .nn sub-module that PEFT checks at import time.  We only use bf16
        # so we never need quantized layers; patch the missing attribute so PEFT
        # can load without error.
        try:
            import bitsandbytes as _bnb
            if not hasattr(_bnb, "nn"):
                import types as _types
                _bnb.nn = _types.SimpleNamespace(Linear4bit=None, Linear8bitLt=None)
        except ImportError:
            pass

        from peft import LoraConfig, get_peft_model, TaskType
        rank0_print("Applying LoRA to Qwen2 language model …")
        lora_config = LoraConfig(
            r              = training_args.lora_rank,
            lora_alpha     = training_args.lora_alpha,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_dropout   = training_args.lora_dropout,
            bias           = training_args.lora_bias,
            # language_model is an internal LLaVA sub-module, not a standalone
            # CausalLM, so it lacks prepare_inputs_for_generation.
            # FEATURE_EXTRACTION uses the base PeftModel which has no such
            # requirement; LoRA adapters are injected identically.
            task_type      = TaskType.FEATURE_EXTRACTION,
        )
        llm = model.base_model.model.language_model
        model.base_model.model.language_model = get_peft_model(llm, lora_config)
        # set_trainable() froze all base_model params; re-enable LoRA adapters
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
        lora_n = sum(p.numel() for n, p in model.named_parameters()
                     if "lora_" in n and p.requires_grad)
        rank0_print(f"  LoRA trainable parameters: {lora_n:,}")

    # ---- Optionally resume trainable modules -------------------------
    if bimba_args.trainable_ckpt is not None:
        rank0_print(f"Resuming trainable modules from {bimba_args.trainable_ckpt}")
        sd = torch.load(bimba_args.trainable_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            rank0_print(f"  [WARN] missing keys: {missing}")
        if unexpected:
            rank0_print(f"  [WARN] unexpected keys: {unexpected}")

    # ---- Parameter counts --------------------------------------------
    counts = count_parameters(model)
    rank0_print(
        f"Parameters: total={counts['total']:,}  "
        f"trainable={counts['trainable']:,}  "
        f"frozen={counts['frozen']:,}"
    )

    # ---- Processor ---------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        model_args.model_id, trust_remote_code=True
    )

    # ---- Dataset -----------------------------------------------------
    data_module = make_supervised_data_module(
        model_id    = model_args.model_id,
        processor   = processor,
        data_args   = data_args,
    )

    # ---- Optimizer config: only trainable params ---------------------
    # We need to tell the Trainer to optimise only trainable parameters.
    # The cleanest way: set requires_grad=False on frozen params (already
    # done by set_trainable()), and pass a custom optimizer.
    # Alternatively rely on the Trainer's default behaviour of skipping
    # requires_grad=False parameters in the AdamW constructor.

    rank0_print("Trainable module names:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            rank0_print(f"  {name}  ({p.numel():,})")

    # ---- Trainer -----------------------------------------------------
    callbacks = []
    live_loss_plot_dir = os.environ.get("MOTION_OV15_LIVE_PLOT_DIR")
    if live_loss_plot_dir:
        rank0_print(f"Writing live loss plots to {live_loss_plot_dir}")
        callbacks.append(LiveLossPlotCallback(live_loss_plot_dir))

    trainer = BimbaMotionTrainer(
        model           = model,
        args            = training_args,
        callbacks       = callbacks,
        **data_module,
    )

    # ---- Train -------------------------------------------------------
    if list(training_args.resume_from_checkpoint or []):
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # ---- Save final trainable-only checkpoint -----------------------
    trainer.save_model(training_args.output_dir)
    rank0_print(f"Training complete.  Model saved to {training_args.output_dir}")


def _mamba_available() -> bool:
    try:
        import mamba_ssm  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================

if __name__ == "__main__":
    train()
