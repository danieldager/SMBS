# sWuggy Lexical Discrimination Evaluation

Evaluate language models on phonological plausibility using the sWuggy word-nonword classification task.

## Overview

**sWuggy** is a lexical discrimination benchmark that measures whether a language model assigns higher probabilities to real words than to phonetically matched nonwords. Each test item consists of a real word (positive) paired with a phonologically similar pseudoword (negative) in the same voice.

**Example pair:**
- Positive: "castle" /ˈkæsəl/
- Negative: "casple" /ˈkæspəl/

A good model should assign higher sequence probability to the real word.

## Quick Start

```bash
# 1. Encode swuggy audio with an encoder (e.g., hubert-500)
sbatch scripts/swuggy/run.slurm prepare_swuggy hubert-500

# 2. Evaluate a trained model (auto-computes accuracy)
sbatch scripts/swuggy/run.slurm evaluate hubert-500 \
  --dataset swuggy \
  --model lstm_h256_l2_d0.0_09feb13
```

That's it! The evaluation script automatically:
- Finds the latest checkpoint in `weights/hubert-500_lstm_h256_l2_d0.0_09feb13/`
- Loads tokens from `tokens/swuggy_hubert-500/`
- Scores all samples with log-probabilities
- Computes discrimination accuracy
- Saves results to `metadata/swuggy_hubert-500_lstm_h256_l2_d0.0_09feb13.parquet`

## Architecture

### Path Conventions

Everything is derived from three names: **dataset**, **encoder**, **model**.

| Path | Convention | Example |
|------|------------|---------|
| Raw metadata | `metadata/{dataset}.parquet` | `metadata/swuggy.parquet` |
| Tokens | `tokens/{dataset}_{encoder}/` | `tokens/swuggy_hubert-500/` |
| Model weights | `weights/{encoder}/{model}/` | `weights/hubert-500/lstm_h256_l2_d0.0_09feb13/` |
| Checkpoint | Latest `checkpoint-*` in model dir | `checkpoint-10000` |
| Output | `metadata/{dataset}_{encoder}_{model}.parquet` | `metadata/swuggy_hubert-500_lstm_h256_l2_d0.0_09feb13.parquet` |

**No paths need to be specified manually** — just provide the three names.

### File Structure

```
scripts/swuggy/
├── prepare_swuggy.py    # Dataset-specific: ingest raw parquet, encode audio
├── evaluate.py          # Generic: score samples + compute accuracy
├── utils.py             # Model loading utilities
└── run.slurm           # SLURM launcher (auto-switches uv/conda by encoder)
```

**Key insight:** Only `prepare_swuggy.py` knows about sWuggy's schema. Everything else is generic and works with any dataset that follows the standard format:

```
Metadata: parquet with columns [group_id, file_id, positive, ...]
Tokens:   WebDataset .tar files with {__key__: file_id, tokens.npy: int16[]}
```

## Usage

### Step 1: Encode Audio

```bash
sbatch scripts/swuggy/run.slurm prepare_swuggy <encoder>
```

**Encoders:**
- `spidr_base` (256 tokens, uses `uv run`)
- `hubert-500` (500 tokens, uses `conda activate textless`)
- `mhubert` (2000 tokens, uses `conda activate textless`)

**What it does:**
1. Reads raw sWuggy parquet from `/store/projects/lexical-benchmark/swuggy/data/*.parquet`
2. Unpivots positive/negative audio columns into individual samples
3. Encodes each audio clip → discrete tokens
4. Writes WebDataset shards to `tokens/swuggy_{encoder}/`
5. Saves metadata (without audio bytes) to `metadata/swuggy.parquet`

**Note:** The metadata file is shared across all encoders (it only has `file_id` keys, not the tokens themselves).

### Step 2: Evaluate Model

```bash
sbatch scripts/swuggy/run.slurm evaluate <encoder> \
  --dataset swuggy \
  --model <model_dir_name>
```

**Examples:**

```bash
# LSTM on hubert-500
sbatch scripts/swuggy/run.slurm evaluate hubert-500 \
  --dataset swuggy \
  --model lstm_h256_l2_d0.0_09feb13

# GPT2 on mhubert
sbatch scripts/swuggy/run.slurm evaluate mhubert \
  --dataset swuggy \
  --model gpt2_e768_l12_h12_09feb13
```

**What it does:**

1. **Check for existing results:** If `metadata/{dataset}_{encoder}_{model}.parquet` exists, skips scoring and jumps straight to printing the accuracy. Use `--force` to re-score.

2. **Load model:** Finds the latest `checkpoint-*` in `weights/{encoder}_{model}/` and loads it.

3. **Score samples:** For each audio sample, computes:
   - `log_prob`: sum of log P(token_i | context) over the sequence
   - `log_prob_norm`: length-normalized variant (divide by number of tokens)
   - `num_tokens`: sequence length

4. **Save scored data:** Writes parquet with original metadata + new columns.

5. **Compute accuracy:** Calculates discrimination accuracy (see Metrics below) and prints results broken down by voice if available.

**Output:**

```
============================================================
DISCRIMINATION ACCURACY
============================================================
  Source:  metadata/swuggy_hubert-500_lstm_h256_l2_d0.0_09feb13.parquet
  Samples: 2400
  Groups:  600 (column: group_id)

  Raw: 0.7850

    AM_F         0.7900
    AM_M         0.7800
    ES_F         0.7833
    ES_M         0.7867

  Normalized: 0.7917

    AM_F         0.7967
    AM_M         0.7867
    ES_F         0.7883
    ES_M         0.7950

============================================================
```

### Re-running Analysis Only

If you've already scored a dataset and just want to see the accuracy again:

```bash
# Same command — it detects the output exists and skips to analysis
sbatch scripts/swuggy/run.slurm evaluate hubert-500 --dataset swuggy --model lstm_h256_l2_d0.0_09feb13
```

To force re-scoring (e.g., after fixing a bug):

```bash
sbatch scripts/swuggy/run.slurm evaluate hubert-500 --dataset swuggy --model lstm_h256_l2_d0.0_09feb13 --force
```

## Metrics

### Discrimination Accuracy

For each `group_id` (one positive + its matching negative(s)):

1. Compare the log-probability of each positive to each negative
2. For each positive, compute: **fraction of negatives it beats**
3. Average across positives within the group
4. Macro-average across all groups

**Works with N×M configurations:**
- sWuggy: 1 positive, 1 negative per group
- Future datasets: 1 positive, multiple negatives (or even N×M)

**Two variants:**
- **Raw (`log_prob`)**: Sum of log-probabilities over the full sequence
- **Normalized (`log_prob_norm`)**: Average log-probability per token (controls for length)

**Chance level:** 0.5 (coin flip)

### Voice Breakdown

If the metadata has a `voice` column (sWuggy does), accuracy is also reported per voice. This helps detect if the model has voice-specific biases.

## How It Works

### Sequence Scoring

For each token sequence `[BOS, t1, t2, ..., tN, EOS]`:

1. Forward pass through the model → logits `[seq_len, vocab_size]`
2. Shift for autoregressive prediction: predict `tokens[1:]` from `tokens[:-1]`
3. Apply log-softmax to get log P(token | context)
4. Gather the log-probabilities of the actual next tokens
5. Sum them to get sequence log-probability

```python
log P(sequence) = Σ log P(t_i | t_1...t_{i-1})
```

Higher log-probability = model thinks the sequence is more likely.

### Token Wrapping

Tokens are automatically wrapped with BOS/EOS:
- `BOS = vocab_size - 2` (e.g., 500 for hubert-500)
- `EOS = vocab_size - 1` (e.g., 501 for hubert-500)

This is handled transparently by `load_tokens_from_tar()` in [evaluate.py](evaluate.py).

## Adding a New Dataset

To evaluate on a different lexical discrimination dataset:

1. **Write a `prepare_<dataset>.py`** script (copy `prepare_swuggy.py` as a template):
   - Read your raw data (whatever format it's in)
   - Encode audio to tokens using `load_encoder()`
   - Write output with the standard schema:
     - **Metadata:** parquet with columns `group_id`, `file_id`, `positive`, and any extras
     - **Tokens:** WebDataset tars with `__key__=file_id`, `tokens.npy=int16 array`

2. **Run encoding:**
   ```bash
   sbatch scripts/swuggy/run.slurm prepare_<dataset> <encoder>
   ```

3. **Evaluate:**
   ```bash
   sbatch scripts/swuggy/run.slurm evaluate <encoder> --dataset <dataset> --model <model>
   ```

The evaluation pipeline is fully generic — it just needs `group_id`, `file_id`, and `positive` columns.

## Technical Notes

### Environment Switching

The SLURM script automatically selects the correct Python environment based on encoder:

- **`spidr_base`**: Uses `uv run --frozen` (main venv)
- **`hubert-500`, `mhubert`**: Uses `conda activate textless` (legacy environment)

This is transparent to the user — just specify the encoder name.

### Skip-if-Exists Logic

To avoid wasting GPU time, `evaluate.py` checks if the output parquet already exists before loading the model. If found, it loads the cached results and prints the accuracy. This makes it fast to re-check scores without re-running inference.

Override with `--force` if you need to re-score (e.g., after a model update).

### Model Loading

Models are loaded via `load_checkpoint()` in [utils.py](utils.py), which:
- Reads `config.json` to detect model type (LSTM or GPT2)
- Loads weights from `model.safetensors`
- Handles GPT2 weight tying (`lm_head.weight` tied to `transformer.wte.weight`)
- Returns model in eval mode on the requested device

All vocab parameters (BOS/EOS/vocab_size) come from the checkpoint's saved config, not hardcoded defaults.

## Troubleshooting

**"No checkpoints found in {model_dir}"**
- Check that the model directory name is correct
- Ensure there's at least one `checkpoint-*` subdirectory inside it

**"No tar files found in {tokens_dir}"**
- Run `prepare_swuggy` first to encode the audio
- Check that the encoder name matches (e.g., `hubert-500` not `hubert500`)

**Missing tokens during scoring**
- Some samples may be skipped during encoding (e.g., too short, encoding error)
- The script reports how many samples are missing tokens
- Scores are only computed for samples with valid tokens

**CUDA OOM**
- GPT2 with large vocab (mhubert) may need more memory
- Check [train.py](../train/train.py) — it reduces batch size to 16 for encoders with `vocab_size > 1000`
- For evaluation, you can add `device="cpu"` if needed (edit the script)

## Reference

**sWuggy nonword generator:**
Keuleers, E., & Brysbaert, M. (2010). Wuggy: A multilingual pseudoword generator. *Behavior Research Methods*, 42(3), 627-633.

**Lexical discrimination in speech models:**
This evaluation follows the tradition of word-nonword classification tasks used to probe phonological representations in neural models.
