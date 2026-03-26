"""Encoder registry — maps names to encoder classes and configs."""

from smbs.encode.base import AudioEncoder, EncoderConfig
from smbs.encode.spidr import SpidrEncoder
from smbs.encode.textless import TextlessEncoder

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


def is_legacy_encoder(name: str) -> bool:
    """Check if an encoder requires the legacy textless environment."""
    if name not in ENCODER_REGISTRY:
        return False
    return ENCODER_REGISTRY[name]["class"] is TextlessEncoder
