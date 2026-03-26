"""GPU test suite for the SMBS pipeline.

Run via SLURM: sbatch slurm/test.slurm
Tests all three encoders, audio loading, and basic output sanity.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torchaudio

# ── Helpers ──────────────────────────────────────────────────────────────

AUDIO_PATH = "/store/projects/lexical-benchmark/audio/symlinks/50h/02/3662_LibriVox_en_seq_00.wav"

passed = 0
failed = 0
errors: list[str] = []


def run_test(name: str, fn):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"  PASSED ({dt:.2f}s)")
        passed += 1
    except Exception as e:
        dt = time.time() - t0
        msg = f"  FAILED ({dt:.2f}s): {e}"
        print(msg)
        traceback.print_exc()
        failed += 1
        errors.append(f"{name}: {e}")


# ── Tests ────────────────────────────────────────────────────────────────


def test_imports():
    """All main modules import without error."""
    from smbs.config import SAMPLE_RATE, PROJECT_ROOT
    from smbs.encode.registry import ENCODER_REGISTRY, load_encoder, get_encoder_config
    from smbs.encode.base import AudioEncoder, EncoderConfig
    from smbs.encode.hubert import HuBERTEncoder
    from smbs.encode.spidr import SpidrEncoder
    from smbs.utils.audio import load_audio, to_mono, resample
    print("  All imports OK")


def test_config():
    """Config constants are sane."""
    from smbs.config import SAMPLE_RATE, TOKENS_DIR, WEIGHTS_DIR
    assert SAMPLE_RATE == 16_000
    assert TOKENS_DIR.exists(), f"TOKENS_DIR missing: {TOKENS_DIR}"
    assert WEIGHTS_DIR.exists(), f"WEIGHTS_DIR missing: {WEIGHTS_DIR}"
    print(f"  SAMPLE_RATE={SAMPLE_RATE}, dirs exist")


def test_registry():
    """Encoder registry has all three encoders with correct configs."""
    from smbs.encode.registry import ENCODER_REGISTRY, get_encoder_config
    names = list(ENCODER_REGISTRY.keys())
    assert set(names) == {"spidr_base", "mhubert", "hubert-500"}, f"Got: {names}"
    for name in names:
        cfg = get_encoder_config(name)
        assert cfg.bos_token_id == cfg.n_tokens
        assert cfg.eos_token_id == cfg.n_tokens + 1
        assert cfg.vocab_size == cfg.n_tokens + 2
        print(f"  {name}: n_tokens={cfg.n_tokens}, vocab={cfg.vocab_size}")


def test_load_audio():
    """Audio loading, mono conversion, and resampling work."""
    from smbs.utils.audio import load_audio
    waveform, sr = load_audio(AUDIO_PATH)
    assert sr == 16_000, f"Expected 16kHz, got {sr}"
    assert waveform.dim() == 2 and waveform.shape[0] == 1, f"Shape: {waveform.shape}"
    assert waveform.shape[1] > 16_000, "Audio too short (< 1s)"
    duration = waveform.shape[1] / sr
    print(f"  Loaded: shape={waveform.shape}, duration={duration:.2f}s")


def test_cuda_available():
    """CUDA is available and we can allocate on GPU."""
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    x = torch.randn(10, device=device)
    assert x.device.type == "cuda"
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    print(f"  Memory: {mem_gb:.1f} GB")


def test_hubert500_encoder():
    """HuBERT-500 encoder loads and produces valid tokens."""
    from smbs.encode.registry import load_encoder, get_encoder_config
    from smbs.utils.audio import load_audio

    encoder = load_encoder("hubert-500", device="cuda")
    waveform, sr = load_audio(AUDIO_PATH)
    tokens = encoder.encode(waveform, sr)

    cfg = get_encoder_config("hubert-500")
    assert isinstance(tokens, np.ndarray), f"Expected ndarray, got {type(tokens)}"
    assert tokens.dtype == np.int16, f"Expected int16, got {tokens.dtype}"
    assert len(tokens) > 0, "No tokens produced"
    assert tokens.min() >= 0, f"Negative token: {tokens.min()}"
    assert tokens.max() < cfg.n_tokens, f"Token {tokens.max()} >= {cfg.n_tokens}"
    # Deduplicated: no consecutive duplicates
    if len(tokens) > 1:
        assert not np.all(tokens[1:] == tokens[:-1]), "Tokens not deduplicated"
    print(f"  Tokens: {len(tokens)}, range=[{tokens.min()}, {tokens.max()}], sample={tokens[:10]}")


def test_mhubert_encoder():
    """mHuBERT encoder loads (converting from fairseq if needed) and produces valid tokens."""
    from smbs.encode.registry import load_encoder, get_encoder_config
    from smbs.utils.audio import load_audio

    encoder = load_encoder("mhubert", device="cuda")
    waveform, sr = load_audio(AUDIO_PATH)
    tokens = encoder.encode(waveform, sr)

    cfg = get_encoder_config("mhubert")
    assert isinstance(tokens, np.ndarray)
    assert tokens.dtype == np.int16
    assert len(tokens) > 0
    assert tokens.min() >= 0
    assert tokens.max() < cfg.n_tokens, f"Token {tokens.max()} >= {cfg.n_tokens}"
    if len(tokens) > 1:
        assert not np.all(tokens[1:] == tokens[:-1])
    print(f"  Tokens: {len(tokens)}, range=[{tokens.min()}, {tokens.max()}], sample={tokens[:10]}")


def test_spidr_encoder():
    """SPIDR encoder loads and produces valid tokens."""
    from smbs.encode.registry import load_encoder, get_encoder_config
    from smbs.utils.audio import load_audio

    encoder = load_encoder("spidr_base", device="cuda")
    waveform, sr = load_audio(AUDIO_PATH)
    tokens = encoder.encode(waveform, sr)

    cfg = get_encoder_config("spidr_base")
    assert isinstance(tokens, np.ndarray)
    assert tokens.dtype == np.int16
    assert len(tokens) > 0
    assert tokens.min() >= 0
    assert tokens.max() < cfg.n_tokens, f"Token {tokens.max()} >= {cfg.n_tokens}"
    if len(tokens) > 1:
        assert not np.all(tokens[1:] == tokens[:-1])
    print(f"  Tokens: {len(tokens)}, range=[{tokens.min()}, {tokens.max()}], sample={tokens[:10]}")


def test_encoder_determinism():
    """Encoders produce the same output for the same input."""
    from smbs.encode.registry import load_encoder
    from smbs.utils.audio import load_audio

    waveform, sr = load_audio(AUDIO_PATH)

    for name in ["hubert-500", "mhubert", "spidr_base"]:
        encoder = load_encoder(name, device="cuda")
        t1 = encoder.encode(waveform, sr)
        t2 = encoder.encode(waveform, sr)
        assert np.array_equal(t1, t2), f"{name}: non-deterministic output"
        print(f"  {name}: deterministic ({len(t1)} tokens)")


def test_tenvad_import():
    """TenVAD can be imported."""
    import ten_vad  # noqa: F401
    print(f"  ten_vad module imported OK")


# ── Main ─────────────────────────────────────────────────────────────────


TESTS = [
    ("imports", test_imports),
    ("config", test_config),
    ("registry", test_registry),
    ("load_audio", test_load_audio),
    ("cuda_available", test_cuda_available),
    ("hubert500_encoder", test_hubert500_encoder),
    ("mhubert_encoder", test_mhubert_encoder),
    ("spidr_encoder", test_spidr_encoder),
    ("encoder_determinism", test_encoder_determinism),
    ("tenvad_import", test_tenvad_import),
]


if __name__ == "__main__":
    print(f"SMBS Test Suite — {len(TESTS)} tests")
    print(f"Audio: {AUDIO_PATH}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    t_start = time.time()

    for name, fn in TESTS:
        run_test(name, fn)

    dt = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed ({dt:.1f}s total)")
    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(f"  ✗ {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
