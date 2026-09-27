"""SPIDR encoder — layer 6 codebook with manual deduplication."""

import numpy as np
import torch

from smbs.encode.base import AudioEncoder


class SpidrEncoder(AudioEncoder):
    """SPIDR encoder: layer 6 codebook with manual deduplication."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        from spidr.models import spidr_base

        self.model = spidr_base().to(device)
        self.model.eval()
        print(f"Loaded spidr_base on {device}")

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sample_rate)

        # SPIDR expects layer-normed input on GPU
        waveform = waveform.to(self.device)
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)

        with torch.no_grad():
            codebooks = self.model.get_codebooks(waveform, onehot=True)
            tokens = codebooks[5].argmax(dim=-1)
            tokens = torch.unique_consecutive(tokens)

        return tokens.cpu().numpy().astype(np.int16)
