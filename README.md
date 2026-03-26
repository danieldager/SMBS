# SMBS — Speech Model Benchmarking Suite

Encode audio → discrete tokens, train language models, evaluate on lexical benchmarks.

## Setup

```bash
uv sync
```

For legacy encoders (mHuBERT, HuBERT-500) which require Python 3.9:
```bash
uv sync --project envs/textless
```

## Workflow

The `smbs` CLI submits SLURM jobs by default. Add `--local` to run directly
on the current node (e.g. an interactive GPU session). Override SLURM defaults
with `-p`/`--partition` and `--time`.

### 1. Scan — create a manifest of audio files

```bash
smbs scan /path/to/audio/dataset
# Override partition:
smbs scan /path/to/audio/dataset -p cpu
# Run locally (no SLURM):
smbs scan /path/to/audio/dataset --local
```

Writes `manifests/<dataset>.txt` (one absolute path per line, sorted for NFS locality).

### 2. (Optional) VAD — voice activity detection

**PyAnnote** (GPU, per-file segments):
```bash
smbs vad-pyannote --manifest chunks5
smbs vad-pyannote --manifest chunks5 --array 0-4   # 5 parallel tasks
```

**TenVAD** (CPU, multiprocessing):
```bash
smbs vad-ten --manifest chunks5
```

### 3. Encode — audio → discrete tokens

```bash
smbs encode --encoder spidr_base --dataset chunks30
smbs encode --encoder mhubert --dataset chunks30     # uses textless env
smbs encode --encoder hubert-500 --dataset chunks30   # uses textless env

# Custom array size and partition:
smbs encode --encoder spidr_base --dataset chunks30 --array 0-5 -p gpu-p1
```

Tokens are written as WebDataset `.tar` shards to `tokens/<dataset>_<encoder>/`.

### 4. Train — language models on token sequences

```bash
smbs train --encoder spidr_base --arch gpt2
smbs train --encoder spidr_base --arch lstm --gpus 4
```

Weights are saved to `weights/<encoder>/<model_name>/`.

**Grid search** (LSTM hyperparameters):
```bash
smbs grid --encoder spidr_base
```

### 5. Evaluate — lexical discrimination (sWuggy)

**Prepare** sWuggy data (encode benchmark audio):
```bash
smbs prepare-swuggy --encoder spidr_base --parquet-pattern '/path/to/swuggy/*.parquet'
```

**Evaluate** a trained model:
```bash
smbs evaluate --encoder spidr_base --model gpt2_e768_l12_h12_feb12
```

**Plot** all results (always runs locally):
```bash
smbs plots
smbs plots --raw
```

## Project structure

```
src/smbs/
  cli.py            # SLURM job launcher (sbatch wrapper)
  config.py         # Shared constants and paths
  scan.py           # Audio file discovery
  utils/
    audio.py        # Audio loading, resampling, mono conversion
    manifest.py     # Manifest loading, --dataset auto-resolution, task sharding
  encode/
    base.py         # AudioEncoder ABC + EncoderConfig
    spidr.py        # SPIDR encoder
    textless.py     # Legacy mHuBERT/HuBERT-500 (self-contained monkey-patches)
    registry.py     # Encoder registry and loader
    run.py          # Tokenization pipeline + WebDataset writer
  train/
    models.py       # LSTM model + config defaults
    dataset.py      # TokenDataset (streaming) + EvalDataset
    run.py          # Training entry point (HF Trainer)
    grid.py         # Hyperparameter grid search
    utils.py        # Checkpoint loading, model discovery
  evaluate/
    metrics.py      # discrimination_accuracy, per_voice_accuracy
    swuggy.py       # sWuggy prepare + evaluate + analysis
    plots.py        # Result visualization
  vad/
    pyannote.py     # PyAnnote GPU pipeline
    tenvad.py       # TenVAD CPU pipeline
slurm/              # SLURM job scripts
envs/textless/      # Python 3.9 environment for legacy encoders
```

## Encoders

| Name | Library | Environment |
|------|---------|-------------|
| `spidr_base` | [spidr](https://github.com/archinetai/spidr) | Main (`uv run`) |
| `mhubert` | textlesslib (fairseq) | Legacy (`uv run --project envs/textless`) |
| `hubert-500` | textlesslib (fairseq) | Legacy (`uv run --project envs/textless`) |

## Adding new components

**Encoder**: Subclass `AudioEncoder` in `src/smbs/encode/`, register in `registry.py`.
**Model architecture**: Add config + model class in `train/models.py`, handle in `train/run.py`.
**Evaluation benchmark**: Add to `evaluate/`, wire into `cli.py`.
