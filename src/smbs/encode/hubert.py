"""HuBERT / mHuBERT encoders using HuggingFace transformers + k-means.

Replaces the legacy textlesslib+fairseq pipeline. Loads HuBERT models from
HuggingFace Hub (hubert-base-ls960) or converts fairseq checkpoints on first
use (mHuBERT), then applies k-means quantization for discrete tokenization.

Requires: transformers, scikit-learn, joblib (all in main uv env).
"""

import sys
import types
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch

from smbs.encode.base import AudioEncoder

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*InconsistentVersionWarning.*"
)

TEXTLESS_CACHE = Path.home() / ".textless"

# Model specs: how to load each dense HuBERT model
MODELS = {
    "hubert-base-ls960": {
        "hf_model_id": "facebook/hubert-base-ls960",
        "fairseq_ckpt": None,
    },
    "mhubert-base-vp_mls_cv_8lang": {
        "hf_model_id": None,  # Not on HF Hub — convert from fairseq
        "fairseq_ckpt": "mhubert_base_vp_mls_cv_8lang_it3.pt",
    },
}

# K-means quantizer files (in ~/.textless/)
KMEANS_FILES = {
    (
        "mhubert-base-vp_mls_cv_8lang",
        2000,
    ): "mhubert_base_vp_mls_cv_8lang_it3_L12_km2000.pt",
    ("hubert-base-ls960", 500): "hubert_base_ls960_km500.pt",
}


# ── Fairseq → HuggingFace checkpoint conversion ─────────────────────────


def _stub_fairseq_modules() -> None:
    """Register minimal stub modules so torch.load can unpickle fairseq checkpoints."""
    if "fairseq" in sys.modules:
        return
    fairseq_mod = types.ModuleType("fairseq")
    data_mod = types.ModuleType("fairseq.data")
    dict_mod = types.ModuleType("fairseq.data.dictionary")

    class _StubDict:
        pass

    dict_mod.Dictionary = _StubDict  # type: ignore[attr-defined]
    fairseq_mod.data = data_mod  # type: ignore[attr-defined]
    data_mod.dictionary = dict_mod  # type: ignore[attr-defined]
    sys.modules["fairseq"] = fairseq_mod
    sys.modules["fairseq.data"] = data_mod
    sys.modules["fairseq.data.dictionary"] = dict_mod


def _map_key(key: str) -> str | None:
    """Map one fairseq state-dict key to HuggingFace HubertModel format.

    Returns None for keys that should be dropped (pre-training heads, etc.).
    """
    # Pre-training-only parameters
    if key in ("final_proj.weight", "final_proj.bias", "label_embs_concat"):
        return None

    # Feature projection
    if key.startswith("post_extract_proj."):
        return key.replace("post_extract_proj.", "feature_projection.projection.")
    if key.startswith("layer_norm."):
        return key.replace("layer_norm.", "feature_projection.layer_norm.", 1)

    # Masked spec embedding
    if key == "mask_emb":
        return "masked_spec_embed"

    # Feature extractor CNN layers
    for i in range(7):
        old = f"feature_extractor.conv_layers.{i}.0."
        new = f"feature_extractor.conv_layers.{i}.conv."
        if key.startswith(old):
            return key.replace(old, new)
    if key.startswith("feature_extractor.conv_layers.0.2."):
        return key.replace(
            "feature_extractor.conv_layers.0.2.",
            "feature_extractor.conv_layers.0.layer_norm.",
        )

    # Positional convolution (weight_norm renaming)
    if key.startswith("encoder.pos_conv.0."):
        key = key.replace("encoder.pos_conv.0.", "encoder.pos_conv_embed.conv.")
        key = key.replace("weight_g", "parametrizations.weight.original0")
        key = key.replace("weight_v", "parametrizations.weight.original1")
        return key

    # Transformer encoder layers
    if key.startswith("encoder.layers."):
        key = key.replace(".self_attn.", ".attention.")
        key = key.replace(".self_attn_layer_norm.", ".layer_norm.")
        key = key.replace(".fc1.", ".feed_forward.intermediate_dense.")
        key = key.replace(".fc2.", ".feed_forward.output_dense.")
        return key

    # encoder.layer_norm stays the same
    return key


def _convert_fairseq_to_hf(fairseq_path: Path, save_dir: Path) -> None:
    """One-time conversion of a fairseq HuBERT checkpoint to HF format."""
    from transformers import HubertConfig, HubertModel

    _stub_fairseq_modules()

    print(f"Converting {fairseq_path.name} → HuggingFace format ...")
    ckpt = torch.load(str(fairseq_path), map_location="cpu", weights_only=False)
    fairseq_sd = ckpt["model"]

    # Map keys
    hf_sd: dict[str, torch.Tensor] = {}
    for fs_key, tensor in fairseq_sd.items():
        hf_key = _map_key(fs_key)
        if hf_key is not None:
            hf_sd[hf_key] = tensor

    # Build model from default HuBERT-base config and load weights
    config = HubertConfig()
    model = HubertModel(config)
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)

    if unexpected:
        print(f"  ⚠ {len(unexpected)} unexpected keys (ignored)")

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)
    print(f"  Saved to {save_dir}")


# ── Encoder ──────────────────────────────────────────────────────────────


class HuBERTEncoder(AudioEncoder):
    """HuBERT/mHuBERT encoder — HuggingFace transformers + k-means.

    Extracts layer-12 features from a HuBERT model, quantises with k-means,
    and deduplicates consecutive tokens (same as textlesslib pipeline).

    Args:
        model_name: Dense model key, e.g. "hubert-base-ls960".
        vocab_size: Number of k-means clusters (determines which quantiser to load).
        device: "cuda" or "cpu".
    """

    def __init__(self, model_name: str, vocab_size: int, device: str = "cpu"):
        super().__init__(device)
        from transformers import HubertModel

        info = MODELS[model_name]

        # Load HuBERT model
        if info["hf_model_id"]:
            self.model = HubertModel.from_pretrained(info["hf_model_id"])
        else:
            cache_dir = TEXTLESS_CACHE / f"{model_name}_hf"
            if not (cache_dir / "config.json").exists():
                fairseq_path = TEXTLESS_CACHE / info["fairseq_ckpt"]
                if not fairseq_path.exists():
                    raise FileNotFoundError(
                        f"Fairseq checkpoint not found: {fairseq_path}\n"
                        f"Download from https://dl.fbaipublicfiles.com/hubert/ first."
                    )
                _convert_fairseq_to_hf(fairseq_path, cache_dir)
            self.model = HubertModel.from_pretrained(str(cache_dir))

        self.model.eval()
        self.model.to(device)  # type: ignore

        # Load k-means quantizer
        km_key = (model_name, vocab_size)
        if km_key not in KMEANS_FILES:
            raise ValueError(
                f"No k-means file for ({model_name}, {vocab_size}). "
                f"Available: {list(KMEANS_FILES.keys())}"
            )
        km_path = TEXTLESS_CACHE / KMEANS_FILES[km_key]
        if not km_path.exists():
            raise FileNotFoundError(f"K-means file not found: {km_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.kmeans = joblib.load(str(km_path))

        print(f"Loaded {model_name} (layer 12, {vocab_size} clusters) on {device}")

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sample_rate)
        waveform = waveform.squeeze(0).to(self.device)  # (T,)

        # Extract layer-12 features (last hidden state for HuBERT base)
        outputs = self.model(waveform.unsqueeze(0))
        features = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (T', 768)

        # K-means quantization → deduplicate
        tokens = torch.from_numpy(self.kmeans.predict(features))
        tokens = torch.unique_consecutive(tokens)
        return tokens.numpy().astype(np.int16)
