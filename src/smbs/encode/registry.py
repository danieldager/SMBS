"""Encoder registry — maps names to encoder classes and configs."""

from smbs.encode.base import AudioEncoder, EncoderConfig
from smbs.encode.spidr import SpidrEncoder
from smbs.encode.hubert import HuBERTEncoder

# Maps encoder name → config + loader info
ENCODER_REGISTRY: dict[str, dict] = {
    "spidr_base": {
        "class": SpidrEncoder,
        "n_tokens": 256,
    },
    "mhubert": {
        "class": HuBERTEncoder,
        "model_name": "mhubert-base-vp_mls_cv_8lang",
        "vocab_size": 2000,
        "n_tokens": 2000,
    },
    "hubert-500": {
        "class": HuBERTEncoder,
        "model_name": "hubert-base-ls960",
        "vocab_size": 500,
        "n_tokens": 500,
    },
}


def list_encoders() -> list[str]:
    """Return all registered encoder names."""
    return list(ENCODER_REGISTRY.keys())


def get_encoder_config(name: str) -> EncoderConfig:
    """Get encoder vocab config without loading the model."""
    if name not in ENCODER_REGISTRY:
        available = ", ".join(ENCODER_REGISTRY.keys())
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}")
    return EncoderConfig(name=name, n_tokens=ENCODER_REGISTRY[name]["n_tokens"])


def load_encoder(name: str, device: str = "cuda") -> AudioEncoder:
    """Load an encoder by name."""
    if name not in ENCODER_REGISTRY:
        available = ", ".join(ENCODER_REGISTRY.keys())
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}")

    entry = ENCODER_REGISTRY[name]
    cls = entry["class"]
    kwargs = {k: v for k, v in entry.items() if k not in ("class", "n_tokens")}
    return cls(device=device, **kwargs)



