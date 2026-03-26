"""Legacy textless encoders (mHuBERT, HuBERT, etc.).

Requires the separate textless uv environment (envs/textless/).
All monkey-patching for modern PyTorch / OmegaConf compatibility is
self-contained here — callers don't need to worry about it.
"""

import warnings

import numpy as np
import torch

from smbs.encode.base import AudioEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


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

    Requires the textless uv environment (envs/textless/, Python 3.9).

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
