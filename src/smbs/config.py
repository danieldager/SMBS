"""Shared constants and project-level configuration."""

from pathlib import Path

# Audio
SAMPLE_RATE = 16_000

# Token sequences
MAX_TOKENS = 2048

# WebDataset shards
MAX_SHARD_SIZE = 1 * 1024**3  # 1 GB
MAX_SHARD_COUNT = 10_000

# Encoding
MIN_AUDIO_DURATION = 3.0  # seconds — skip segments shorter than this during bulk encoding
MIN_SWUGGY_DURATION = 0.5  # seconds — skip swuggy samples shorter than this

# Training
SEED = 101
SHUFFLE_BUFFER = 1_000

# Project root — resolved once at import time
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
TOKENS_DIR = PROJECT_ROOT / "tokens"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
METADATA_DIR = PROJECT_ROOT / "metadata"
FIGURES_DIR = PROJECT_ROOT / "figures"
