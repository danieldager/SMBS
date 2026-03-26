"""Utility functions for training."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel

from smbs.train.models import LSTM, LSTMConfig
from smbs.config import WEIGHTS_DIR


def get_model_timestamp():
    """Get timestamp for model naming (format: mmmDD, e.g., feb03)."""
    return datetime.now().strftime("%b%d").lower()


def find_latest_model_dir(base_pattern: str, encoder: str) -> str | None:
    """Find most recent model directory matching a pattern."""
    encoder_path = WEIGHTS_DIR / encoder
    if not encoder_path.exists():
        return None

    matching_dirs = [
        d for d in encoder_path.iterdir()
        if d.is_dir() and d.name.startswith(base_pattern)
    ]

    if not matching_dirs:
        return None

    matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(matching_dirs[0])


def find_latest_checkpoint(model_dir: Path) -> Path:
    """Return the checkpoint subdirectory with the highest step number."""
    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return checkpoints[-1]


def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple[Any, Any]:
    """Load a trained model from a HuggingFace-style checkpoint directory.

    Supports LSTM (custom) and GPT2 architectures. Detects model type from
    the saved config.json.

    Returns:
        (model, config) — model is in eval mode on the requested device.
    """
    checkpoint_dir = Path(checkpoint_path)

    with open(checkpoint_dir / "config.json", "r") as f:
        config_dict = json.load(f)

    model_type = config_dict.get("model_type", "gpt2")

    if model_type == "lstm":
        config = LSTMConfig(**config_dict)
        model = LSTM(config)
    else:
        config = GPT2Config(**config_dict)
        model = GPT2LMHeadModel(config)

    # Load weights
    state_dict = load_file(str(checkpoint_dir / "model.safetensors"))

    # GPT2 weight tying: lm_head.weight is tied to transformer.wte.weight
    if model_type == "gpt2" and "lm_head.weight" not in state_dict:
        if "transformer.wte.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    model.load_state_dict(state_dict)
    model.to(device)  # type: ignore
    model.eval()

    print(f"Loaded {model_type} model from {checkpoint_path}")
    return model, config


def print_training_summary(config, args, model, max_tokens):
    """Print training configuration summary."""
    total_params = sum(p.numel() for p in model.parameters())
    param_size_mb = (total_params * (2 if args.bf16 else 4)) / (1024**2)

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    tokens_per_step = effective_batch * (config.n_positions if hasattr(config, "n_positions") else max_tokens)

    model_type = "GPT2" if hasattr(config, "n_layer") else "LSTM"

    print(f"""
{'='*70}
{model_type} TRAINING SUMMARY
{'='*70}

MODEL:
  Parameters:      {total_params/1e6:.2f}M
  Weight Size:     {param_size_mb:.2f} MB ({'BF16' if args.bf16 else 'FP32'})

BATCH:
  GPUs:            {world_size}
  Per Device:      {args.per_device_train_batch_size}
  Accumulation:    {args.gradient_accumulation_steps}
  Effective:       {effective_batch}
  Tokens/Step:     {tokens_per_step/1e6:.2f}M

OPTIMIZATION:
  Optimizer:       {args.optim}
  Learning Rate:   {args.learning_rate}
  Betas:           ({args.adam_beta1}, {args.adam_beta2})
  Weight Decay:    {args.weight_decay}
  Grad Clip:       {args.max_grad_norm}

SCHEDULE:
  Type:            {args.lr_scheduler_type}
  Warmup Steps:    {args.warmup_steps}
  Max Steps:       {args.max_steps}

CHECKPOINTING:
  Save Every:      {args.save_steps} steps
  Eval Every:      {args.eval_steps} steps
  Keep Last:       {args.save_total_limit}
  Output:          {args.output_dir}

{'='*70}
""")
