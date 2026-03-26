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

### 1. Scan — create a manifest of audio files

```bash
smbs scan /path/to/audio/dataset
# or on SLURM:
sbatch slurm/scan.slurm /path/to/audio/dataset
```

Writes `manifests/<dataset>.txt` (one absolute path per line, sorted for NFS locality).

### 2. (Optional) VAD — voice activity detection

**PyAnnote** (GPU, per-file segments):
```bash
smbs vad-pyannote --manifest chunks5
# SLURM: sbatch slurm/vad_pyannote.slurm chunks5
```

**TenVAD** (CPU, multiprocessing):
```bash
smbs vad-ten --manifest chunks5
# SLURM: sbatch slurm/vad_ten.slurm chunks5
```

### 3. Encode — audio → discrete tokens

```bash
smbs encode --encoder spidr_base --dataset chunks30
# SLURM (3 parallel GPU tasks):
sbatch slurm/encode.slurm spidr_base chunks30
sbatch slurm/encode.slurm mhubert chunks30      # uses textless env
sbatch slurm/encode.slurm hubert-500 chunks30    # uses textless env
```

Tokens are written as WebDataset `.tar` shards to `tokens/<dataset>_<encoder>/`.

### 4. Train — language models on token sequences

```bash
smbs train --encoder spidr_base --arch gpt2
smbs train --encoder spidr_base --arch lstm
# SLURM (multi-GPU with torchrun):
sbatch slurm/train.slurm spidr_base gpt2
```

Weights are saved to `weights/<encoder>/<model_name>/`.

**Grid search** (LSTM hyperparameters):
```bash
smbs grid --encoder spidr_base
# SLURM: sbatch slurm/grid.slurm spidr_base
```

### 5. Evaluate — lexical discrimination (sWuggy)

**Prepare** sWuggy data (encode benchmark audio):
```bash
smbs prepare-swuggy --encoder spidr_base --parquet-pattern '/path/to/swuggy/*.parquet'
# SLURM: sbatch slurm/evaluate.slurm prepare spidr_base '/path/*.parquet'
```

**Evaluate** a trained model:
```bash
smbs evaluate --encoder spidr_base --model gpt2_e768_l12_h12_feb12
# SLURM: sbatch slurm/evaluate.slurm evaluate spidr_base gpt2_e768_l12_h12_feb12
```

**Plot** all results:
```bash
smbs plots
# SLURM: sbatch slurm/evaluate.slurm plots
```

## Project structure

```
src/smbs/
  cli.py            # Unified CLI entry point
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
