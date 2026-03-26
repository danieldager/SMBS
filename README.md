# SMBS — Speech Model Benchmarking Suite

Encode audio → discrete tokens, train language models, evaluate on lexical benchmarks.

## Quick start

```bash
# Install everything in one environment (Python ≥3.12)
uv sync

# Verify the CLI works
uv run smbs --help
```

All commands submit SLURM jobs by default. Add `--local` to run directly
(e.g. in an interactive GPU session). Override SLURM defaults with
`-p`/`--partition` and `--time`.

## Pipeline

### 1. Scan — discover audio files

```bash
smbs scan /path/to/audio/dataset
```

Writes `manifests/<dataset>.csv` with `file_id` and `audio_filepath` columns, sorted for NFS locality.

### 2. VAD — voice activity detection (optional)

```bash
smbs vad --manifest chunks5
smbs vad --manifest chunks5 --workers 32
```

Runs TenVAD on CPU. Outputs `metadata/<manifest>/ten/metadata.parquet` and `segments.parquet`.

> **Note:** The VAD SLURM script sets `LD_LIBRARY_PATH` to load system LLVM
> (`libc++.so.1`) required by TenVAD's native library. See `slurm/vad_ten.slurm`.

### 3. Encode — audio → discrete tokens

```bash
smbs encode --encoder spidr_base --dataset chunks30
smbs encode --encoder mhubert    --dataset chunks30
smbs encode --encoder hubert-500 --dataset chunks30

# Custom array size for more parallelism:
smbs encode --encoder spidr_base --dataset chunks30 --array 0-5
```

Tokens are written as WebDataset `.tar` shards to `tokens/<dataset>_<encoder>/`.

### 4. Train — language models on token sequences

```bash
smbs train --encoder spidr_base                    # GPT-2 (default)
smbs train --encoder spidr_base --arch lstm        # LSTM

# Grid search over LSTM hyperparameters:
smbs grid --encoder spidr_base
```

Weights are saved to `weights/<encoder>/<model_name>/`.

### 5. Evaluate — sWuggy lexical discrimination

**Prepare** — The raw sWuggy dataset stores word pairs (real word + pseudo-word)
with embedded audio. `prepare-swuggy` flattens these pairs into individual
samples, encodes each audio clip into tokens, and writes the result as
WebDataset shards + a metadata parquet. This only needs to run once per encoder.

```bash
# Prepare: encode sWuggy benchmark audio (one-time per encoder)
smbs prepare-swuggy --encoder spidr_base --parquet-pattern '/path/to/swuggy/*.parquet'

# Evaluate a trained model
smbs evaluate --encoder spidr_base --model gpt2_e768_l12_h12_feb12

# Plot all results (always runs locally)
smbs plots
```

#### Evaluation dataset schema

To bring your own lexical discrimination dataset, the metadata parquet
(`metadata/<dataset>.parquet`) must follow this schema:

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | int/str | Shared ID linking a positive sample to its negative counterpart(s) |
| `file_id` | str | Unique sample key, must match the `__key__` in the token `.tar` shards |
| `positive` | bool | `true` for real words, `false` for pseudo-words |
| `voice` | str | *(optional)* Speaker/voice label — enables per-voice accuracy breakdown |
| `word` | str | *(optional)* Orthographic or phonemic form, for inspection |

The token shards (`tokens/<dataset>_<encoder>/shard-*.tar`) contain one
entry per sample where `__key__` matches `file_id` and `tokens.npy` is the
encoded int16 token array.

Discrimination accuracy is computed per group: for each `group_id`, the model
must assign higher log-probability to the positive sample(s) than the
negative(s). The final score is the macro-average across all groups.

**Example** — a minimal 2-pair dataset:

```
group_id  file_id       positive  voice   word
1         1_en_pos      true      en      brick
1         1_en_neg      false     en      blick
2         2_en_pos      true      en      house
2         2_en_neg      false     en      hause
```

## Encoders

| Name | Tokens | Backend |
|------|--------|---------|
| `spidr_base` | 256 | [spidr](https://github.com/archinetai/spidr) — layer 6 codebook |
| `mhubert` | 2000 | HuggingFace `HubertModel` + k-means (converted from fairseq on first use) |
| `hubert-500` | 500 | HuggingFace `HubertModel` (from Hub) + k-means |

All three encoders run in the same `uv` environment. The legacy
`textlesslib`/`fairseq` dependency has been replaced — mHuBERT's fairseq
checkpoint is automatically converted to HuggingFace format on first use and
cached in `~/.textless/`.

## Project structure

```
src/smbs/
├── cli.py              SLURM job launcher (sbatch wrapper + --local mode)
├── config.py           Constants: sample rate, paths, limits
├── scan.py             Audio file discovery
├── encode/
│   ├── base.py         AudioEncoder ABC + EncoderConfig dataclass
│   ├── registry.py     Encoder registry: load_encoder(), get_encoder_config()
│   ├── hubert.py       HuBERT/mHuBERT encoder (HF transformers + k-means)
│   ├── spidr.py        SPIDR encoder (layer 6 codebook)
│   └── run.py          Encoding pipeline + WebDataset shard writer
├── train/
│   ├── models.py       LSTM model class + GPT-2/LSTM configs
│   ├── dataset.py      TokenDataset (streaming WebDataset) + EvalDataset
│   ├── run.py          Training entry point (HF Trainer, torchrun)
│   ├── grid.py         LSTM hyperparameter grid search
│   └── utils.py        Checkpoint loading, model discovery
├── evaluate/
│   ├── swuggy.py       sWuggy preparation + evaluation
│   ├── metrics.py      discrimination_accuracy, per_voice_accuracy
│   └── plots.py        Result visualization (Plotly + Matplotlib)
├── vad/
│   └── tenvad.py       TenVAD CPU pipeline (multiprocessing)
└── utils/
    ├── audio.py        Audio loading, resampling, mono conversion
    └── manifest.py     Manifest resolution + task sharding

slurm/                  SLURM job scripts (one per pipeline stage)
tests/test_gpu.py       GPU test suite (run via slurm/test.slurm)
```

## Testing

```bash
sbatch slurm/test.slurm
```

Runs 10 tests on a GPU node: imports, config, registry, audio loading,
CUDA availability, all three encoders, determinism, and TenVAD.

## Dependencies

Key packages (all installable via `uv sync`, no system dependencies required):

- **torch** ≥2.8, **torchaudio** ≥2.8 — model inference and audio I/O
- **soundfile** — torchaudio audio backend (pure Python, no system ffmpeg needed)
- **transformers** — HuBERT model loading (HuggingFace Hub)
- **scikit-learn**, **joblib** — k-means quantization for HuBERT encoders
- **spidr** — SPIDR encoder
- **ten-vad** — voice activity detection
- **webdataset** — streaming token storage
- **polars** — data manipulation
- **plotly**, **matplotlib**, **seaborn** — visualization

## Adding new components

**Encoder:** Subclass `AudioEncoder` in `src/smbs/encode/`, register in `registry.py`.

**Model architecture:** Add config + model class in `train/models.py`, handle in `train/run.py`.

**Evaluation benchmark:** Add to `evaluate/`, wire into `cli.py`.
