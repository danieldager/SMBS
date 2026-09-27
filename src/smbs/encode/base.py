"""Base classes for audio encoders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from smbs.utils.audio import to_mono, resample


@dataclass(frozen=True)
class EncoderConfig:
    """Vocabulary configuration derived from an encoder's raw codebook size.

    BOS and EOS are appended after the encoder's token range so that
    token IDs 0..n_tokens-1 are encoder outputs, and n_tokens / n_tokens+1
    are the special tokens used by the language model.
    """

    name: str
    n_tokens: int  # raw codebook size (e.g. 256, 500, 2000)

    @property
    def bos_token_id(self) -> int:
        return self.n_tokens

    @property
    def eos_token_id(self) -> int:
        return self.n_tokens + 1

    @property
    def vocab_size(self) -> int:
        return self.n_tokens + 2  # codebook + BOS + EOS


class AudioEncoder(ABC):
    """Base class for audio encoders."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Preprocess waveform and encode to deduplicated int16 tokens.

        Args:
            waveform: Raw audio tensor, any shape/channel/sample rate.
            sample_rate: Sample rate of the input waveform.

        Returns:
            1D numpy array of int16 token IDs (deduplicated).
        """
        ...

    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        return to_mono(waveform)

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return resample(waveform, sample_rate)
