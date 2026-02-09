"""Audio encoders that convert waveforms to discrete token sequences.

Each encoder handles its own preprocessing (mono, resample, normalize)
and returns deduplicated int16 numpy token arrays.

Supported encoders:
    - spidr_base: SPIDR model, layer 6 codebook, manual dedup
    - mhubert, hubert-500, etc.: Legacy textless encoders via conda env

Usage:
    encoder = load_encoder("spidr_base", device="cuda")
    tokens = encoder.encode(waveform, sample_rate)

    # Get vocab config without loading the model:
    config = get_encoder_config("spidr_base")
    print(config.vocab_size, config.bos_token_id, config.eos_token_id)
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

SAMPLE_RATE = 16000


# =============================================================================
# Encoder Config (available without loading a model)
# =============================================================================


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
        """Convert to mono by taking first channel."""
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample to 16kHz if needed."""
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE
            )
        return waveform


# =============================================================================
# SPIDR Encoder
# =============================================================================


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


# =============================================================================
# Legacy Textless Encoders (mHuBERT, HuBERT, etc.)
# =============================================================================


def _apply_textless_patches():
    """Monkey-patches for old fairseq checkpoints with modern PyTorch/OmegaConf.

    Must be called before loading any textless SpeechEncoder.
    Safe to call multiple times (patches are idempotent).
    """
    import argparse
    import omegaconf  # type: ignore
    import fairseq.checkpoint_utils  # type: ignore
    import fairseq.data.dictionary  # type: ignore

    # --- OmegaConf: old checkpoints store ints as floats (50.0 → 50) ---
    if not getattr(omegaconf.OmegaConf, "_patched_for_textless", False):
        _original_merge = omegaconf.OmegaConf.merge

        def _patched_merge(*configs):
            def fix_floats(obj):
                if isinstance(obj, dict):
                    return {k: fix_floats(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [fix_floats(v) for v in obj]
                elif isinstance(obj, float) and obj.is_integer():
                    return int(obj)
                return obj

            fixed = []
            for cfg in configs:
                try:
                    if hasattr(cfg, "_metadata"):
                        container = omegaconf.OmegaConf.to_container(cfg)
                        fixed.append(omegaconf.OmegaConf.create(fix_floats(container)))
                    elif isinstance(cfg, dict):
                        fixed.append(fix_floats(cfg))
                    else:
                        fixed.append(cfg)
                except Exception:
                    fixed.append(cfg)
            return _original_merge(*fixed)

        omegaconf.OmegaConf.merge = _patched_merge
        omegaconf.OmegaConf._patched_for_textless = True  # type: ignore

    # --- PyTorch 2.6+: allowlist fairseq classes for safe deserialization ---
    torch.serialization.add_safe_globals([
        argparse.Namespace,
        fairseq.data.dictionary.Dictionary,
    ])

    # --- Fairseq: remove incompatible pos_conv weight_norm keys ---
    if not getattr(fairseq.checkpoint_utils, "_patched_for_textless", False):
        _original_load = fairseq.checkpoint_utils.load_checkpoint_to_cpu

        def _patched_load(path, *args, **kwargs):
            state = _original_load(path, *args, **kwargs)
            if "model" in state:
                model_state = state["model"]
                keys_to_remove = []
                if "encoder.pos_conv.0.weight_g" in model_state:
                    weight_g = model_state["encoder.pos_conv.0.weight_g"]
                    if weight_g.dim() == 3 and weight_g.shape[0] != 1:
                        keys_to_remove.extend([
                            "encoder.pos_conv.0.weight_g",
                            "encoder.pos_conv.0.weight_v",
                        ])
                if "encoder.pos_conv.0.weight" in model_state:
                    keys_to_remove.extend([
                        "encoder.pos_conv.0.weight",
                        "encoder.pos_conv.0.running_mean",
                        "encoder.pos_conv.0.running_var",
                        "encoder.pos_conv.0.num_batches_tracked",
                        "encoder.pos_conv.1.weight",
                        "encoder.pos_conv.1.bias",
                    ])
                for key in keys_to_remove:
                    model_state.pop(key, None)
            return state

        fairseq.checkpoint_utils.load_checkpoint_to_cpu = _patched_load
        fairseq.checkpoint_utils._patched_for_textless = True  # type: ignore

    # --- Hydra: register pkg:// source to avoid errors ---
    try:
        from hydra.core.plugins import Plugins  # type: ignore
        from hydra.core.config_search_path import ConfigSearchPath  # type: ignore
        from hydra.plugins.search_path_plugin import SearchPathPlugin  # type: ignore

        class PkgSearchPathPlugin(SearchPathPlugin):
            def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
                pass

        try:
            Plugins.instance().register(PkgSearchPathPlugin)  # type: ignore
        except Exception:
            pass
    except Exception:
        pass


class TextlessEncoder(AudioEncoder):
    """Legacy textless encoder (mHuBERT, HuBERT, etc.).

    Requires the 'textless' conda environment.

    Args:
        model_str: Slash-separated model spec, e.g.
            "mhubert-base-vp_mls_cv_8lang/kmeans/2000" or
            "hubert-base-ls960/kmeans/500"
        device: "cuda" or "cpu"
    """

    def __init__(self, model_str: str, device: str = "cuda"):
        super().__init__(device)
        _apply_textless_patches()

        from textless.data.speech_encoder import SpeechEncoder  # type: ignore

        dense_model, quantizer, vocab_size = model_str.split("/")
        self.model = SpeechEncoder.by_name(
            dense_model_name=dense_model,
            quantizer_model_name=quantizer,
            vocab_size=int(vocab_size),
            deduplicate=True,
            need_f0=False,
        )
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f"Loaded {dense_model} ({quantizer}/{vocab_size}) on {device}")

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sample_rate)

        if self.device == "cuda":
            waveform = waveform.cuda()

        with torch.no_grad():
            tokens = self.model(waveform)["units"].cpu().numpy().astype(np.int16)

        return tokens


# =============================================================================
# Encoder Registry
# =============================================================================

# Maps encoder name → config + loader info
ENCODER_REGISTRY: dict[str, dict] = {
    "spidr_base": {
        "class": SpidrEncoder,
        "n_tokens": 256,
    },
    "mhubert": {
        "class": TextlessEncoder,
        "model_str": "mhubert-base-vp_mls_cv_8lang/kmeans/2000",
        "n_tokens": 2000,
    },
    "hubert-500": {
        "class": TextlessEncoder,
        "model_str": "hubert-base-ls960/kmeans/500",
        "n_tokens": 500,
    },
}


def get_encoder_config(name: str) -> EncoderConfig:
    """Get encoder vocab config without loading the model.

    Use this in train/eval scripts to get vocab_size, bos_token_id, eos_token_id.
    """
    if name not in ENCODER_REGISTRY:
        available = ", ".join(ENCODER_REGISTRY.keys())
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}")
    return EncoderConfig(name=name, n_tokens=ENCODER_REGISTRY[name]["n_tokens"])


def load_encoder(name: str, device: str = "cuda") -> AudioEncoder:
    """Load an encoder by name.

    Args:
        name: Encoder name (e.g. "spidr_base", "mhubert", "hubert-500").
        device: "cuda" or "cpu".

    Returns:
        An AudioEncoder instance ready to encode waveforms.
    """
    if name not in ENCODER_REGISTRY:
        available = ", ".join(ENCODER_REGISTRY.keys())
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}")

    entry = ENCODER_REGISTRY[name]
    cls = entry["class"]
    kwargs = {k: v for k, v in entry.items() if k not in ("class", "n_tokens")}
    return cls(device=device, **kwargs)


def is_legacy_encoder(name: str) -> bool:
    """Check if an encoder requires the legacy conda environment."""
    if name not in ENCODER_REGISTRY:
        return False
    return ENCODER_REGISTRY[name]["class"] is TextlessEncoder
