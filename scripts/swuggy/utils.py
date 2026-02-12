"""Model loading utilities for evaluation scripts."""

import json
from pathlib import Path
from typing import Any, Tuple

import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.train.models import LSTM, LSTMConfig


def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple[Any, Any]:
    """Load a trained model from a HuggingFace-style checkpoint directory.

    Supports LSTM (custom) and GPT2 architectures. Detects model type from
    the saved config.json.

    Args:
        checkpoint_path: Path to directory containing config.json + model.safetensors.
        device: "cuda" or "cpu".

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
