"""Audio loading and preprocessing utilities."""

import numpy as np
import torch
import torchaudio  # type: ignore

from smbs.config import SAMPLE_RATE


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert to mono by taking first channel."""
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform[0:1, :]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def resample(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample to 16 kHz if needed."""
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )
    return waveform


def load_audio(path: str, mono: bool = True) -> tuple[torch.Tensor, int]:
    """Load an audio file, optionally convert to mono and resample to 16 kHz.

    Returns:
        (waveform, sample_rate) — waveform is [1, T] at 16 kHz if mono=True.
    """
    waveform, sr = torchaudio.load(path)
    if mono:
        waveform = to_mono(waveform)
        waveform = resample(waveform, sr)
        sr = SAMPLE_RATE
    return waveform, sr
