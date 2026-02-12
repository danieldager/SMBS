"""LSTM Hyperparameter Grid Search

Tests key hyperparameter combinations for LSTM language model on spidr_base tokens.
Runs 3 seeds × 8 configs × 1000 steps each (24 total runs).

Configs tested:
  1. paper_default   — Large model (200/1024/3) + paper training (lr=1e-4, β2=0.98, wd=0.01, gnorm=0.0)
  2. optimal         — Small model (256/256/2) + optimal training (lr=1e-2, β2=0.99, wd=0.0, gnorm=5.0)
  3. opt-low_lr      — Ablation: optimal but lr=1e-4
  4. opt-no_clip     — Ablation: optimal but grad_norm=0.0
  5. opt-with_wd     — Ablation: optimal but weight_decay=0.01
  6. opt-beta2_98    — Ablation: optimal but β2=0.98
  7. large-opt_train — Large model + optimal training
  8. small-paper_tr  — Small model + paper training

Usage:
    python scripts/train/grid.py                          # local
    sbatch scripts/train/grid.slurm                       # cluster
"""

import os
import gc
import json
from time import time
from collections import defaultdict

import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_callback import PrinterCallback

from scripts.train.datasets import TokenDataset, EvalDataset, collate_fn, MAX_TOKENS
from scripts.train.models import LSTM, LSTMConfig
from scripts.encode.encoders import get_encoder_config


# ============================================================================
# GRID CONFIGURATION
# ============================================================================

ENCODER = "spidr_base"
MAX_STEPS = 1000
SEEDS = [42, 123, 456]
LOGGING_STEPS = 50
EVAL_STEPS = 500
WARMUP_STEPS = 100  # Proportional: 100/1000 ≈ 1000/100000

# Two model profiles (architecture is not an independent grid variable)
MODEL_PROFILES = {
    "small": {"embedding_dim": 256, "hidden_size": 256, "num_layers": 2, "dropout": 0.0},
    "large": {"embedding_dim": 200, "hidden_size": 1024, "num_layers": 3, "dropout": 0.1},
}

# Configs to test — 2 endpoints + 4 ablations from optimal + 2 cross-tests
CONFIGS = [
    # --- Endpoints ---
    {
        "name": "paper_default",
        "model": "large",
        "lr": 1e-4,
        "beta2": 0.98,
        "wd": 0.01,
        "grad_norm": 0.0,
    },
    {
        "name": "optimal",
        "model": "small",
        "lr": 1e-2,
        "beta2": 0.99,
        "wd": 0.0,
        "grad_norm": 5.0,
    },
    # --- Ablations from optimal (change ONE factor each) ---
    {
        "name": "opt-low_lr",
        "model": "small",
        "lr": 1e-4,
        "beta2": 0.99,
        "wd": 0.0,
        "grad_norm": 5.0,
    },
    {
        "name": "opt-no_clip",
        "model": "small",
        "lr": 1e-2,
        "beta2": 0.99,
        "wd": 0.0,
        "grad_norm": 0.0,
    },
    {
        "name": "opt-with_wd",
        "model": "small",
        "lr": 1e-2,
        "beta2": 0.99,
        "wd": 0.01,
        "grad_norm": 5.0,
    },
    {
        "name": "opt-beta2_98",
        "model": "small",
        "lr": 1e-2,
        "beta2": 0.98,
        "wd": 0.0,
        "grad_norm": 5.0,
    },
    # --- Cross-tests: model size × training recipe ---
    {
        "name": "large-opt_train",
        "model": "large",
        "lr": 1e-2,
        "beta2": 0.99,
        "wd": 0.0,
        "grad_norm": 5.0,
    },
    {
        "name": "small-paper_tr",
        "model": "small",
        "lr": 1e-4,
        "beta2": 0.98,
        "wd": 0.01,
        "grad_norm": 0.0,
    },
]


# ============================================================================
# TRAINING CALLBACK
# ============================================================================


class GridCallback(TrainerCallback):
    """Lightweight callback that collects loss values and prints progress."""

    def __init__(self):
        self.train_losses = {}
        self.eval_losses = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Only print from rank 0 in distributed training
        if os.environ.get("RANK", "0") != "0":
            return
        
        step = state.global_step
        if "loss" in logs:
            # Normalize by gradient accumulation steps for display
            normalized_loss = logs["loss"] / args.gradient_accumulation_steps
            self.train_losses[step] = normalized_loss
            print(f"    Step {step:4d} | Loss: {normalized_loss:.4f} | LR: {logs.get('learning_rate', 0):.2e}")
        if "eval_loss" in logs:
            self.eval_losses[step] = logs["eval_loss"]
            print(f"    EVAL {step:4d} | Eval Loss: {logs['eval_loss']:.4f}")


# ============================================================================
# SINGLE RUN
# ============================================================================


def run_single(config, seed, train_dataset, eval_dataset, enc_config, run_idx, total_runs):
    """Train one config+seed combination and return results dict."""

    profile = MODEL_PROFILES[config["model"]]

    model_config = LSTMConfig(
        vocab_size=enc_config.vocab_size,
        bos_token_id=enc_config.bos_token_id,
        eos_token_id=enc_config.eos_token_id,
        **profile,
    )

    name = f"{config['name']}_seed{seed}"
    output_dir = f"./weights/grid/{name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        disable_tqdm=True,
        seed=seed,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        learning_rate=config["lr"],
        adam_beta1=0.9,
        adam_beta2=config["beta2"],
        weight_decay=config["wd"],
        max_grad_norm=config["grad_norm"],
        lr_scheduler_type="inverse_sqrt",
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        bf16=True,
        dataloader_num_workers=4,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="no",
        report_to="none",
    )

    callback = GridCallback()
    set_seed(seed)
    model = LSTM(model_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        optimizers=(None, None),  # Trainer creates AdamW from training_args
        callbacks=[callback],
    )
    trainer.pop_callback(PrinterCallback)

    n_params = sum(p.numel() for p in model.parameters())
    h, l = profile["hidden_size"], profile["num_layers"]
    print(
        f"\n[{run_idx}/{total_runs}] {config['name']} | seed={seed} | "
        f"h={h} l={l} | {n_params / 1e6:.1f}M params"
    )
    print(
        f"  lr={config['lr']:.0e}  beta2={config['beta2']}  "
        f"wd={config['wd']}  gnorm={config['grad_norm']}"
    )

    start = time()
    trainer.train()
    elapsed = time() - start

    # Extract milestone losses
    def _last(d):
        if not d:
            return None
        return d[max(d.keys())]

    loss_100 = callback.train_losses.get(100)
    loss_500 = callback.train_losses.get(500)
    final_train = _last(callback.train_losses)
    final_eval = _last(callback.eval_losses)

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print(
        f"  Done in {elapsed:.0f}s | "
        f"loss@100={_fmt(loss_100)}  loss@500={_fmt(loss_500)}  "
        f"loss@1000={_fmt(final_train)}  eval={_fmt(final_eval)}"
    )

    # Cleanup GPU memory between runs
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "config": config["name"],
        "seed": seed,
        "train_losses": callback.train_losses,
        "eval_losses": callback.eval_losses,
        "final_train_loss": final_train,
        "final_eval_loss": final_eval,
        "loss_100": loss_100,
        "loss_500": loss_500,
        "elapsed": elapsed,
        "n_params": n_params,
    }


# ============================================================================
# SUMMARY
# ============================================================================


def print_summary(all_results):
    """Print aggregated results table."""

    by_config = defaultdict(list)
    for r in all_results:
        by_config[r["config"]].append(r)

    sep = "=" * 110
    print(f"\n{sep}")
    print("GRID SEARCH RESULTS SUMMARY")
    print(f"{sep}\n")

    header = (
        f"{'Config':<20} {'Model':>6} {'Params':>7} "
        f"{'Loss@100':>14} {'Loss@500':>14} {'Loss@1000':>14} {'Eval':>14} {'Time':>7}"
    )
    print(header)
    print("-" * 110)

    # Compute per-config statistics
    rows = []
    for config_name, runs in by_config.items():
        n_params = runs[0]["n_params"]
        # Find model profile name from first run
        model_name = next(
            c["model"] for c in CONFIGS if c["name"] == config_name
        )

        def _stat(key):
            vals = [r[key] for r in runs if r[key] is not None]
            if not vals:
                return None, None
            return float(np.mean(vals)), float(np.std(vals))

        m100, s100 = _stat("loss_100")
        m500, s500 = _stat("loss_500")
        m1000, s1000 = _stat("final_train_loss")
        meval, seval = _stat("final_eval_loss")
        mean_time = np.mean([r["elapsed"] for r in runs])

        rows.append((config_name, model_name, n_params, m100, s100, m500, s500,
                      m1000, s1000, meval, seval, mean_time))

    # Sort by eval loss (best first)
    rows.sort(key=lambda r: r[9] if r[9] is not None else float("inf"))

    def _fmt(mean, std):
        if mean is None:
            return "N/A".center(14)
        return f"{mean:.4f}±{std:.4f}"

    for row in rows:
        (name, model, n_params, m100, s100, m500, s500,
         m1000, s1000, meval, seval, mean_time) = row
        print(
            f"{name:<20} {model:>6} {n_params / 1e6:>6.1f}M "
            f"{_fmt(m100, s100):>14} {_fmt(m500, s500):>14} "
            f"{_fmt(m1000, s1000):>14} {_fmt(meval, seval):>14} "
            f"{mean_time:>5.0f}s"
        )

    # Ranking
    print(f"\nRanking (by mean eval loss):")
    print("-" * 45)
    for i, row in enumerate(rows, 1):
        name, _, _, _, _, _, _, _, _, meval, seval, _ = row
        eval_str = f"{meval:.4f}" if meval is not None else "N/A"
        print(f"  {i}. {name:<20} eval={eval_str}")

    # Detailed per-seed table
    print(f"\n{'=' * 80}")
    print("PER-SEED DETAIL")
    print(f"{'=' * 80}\n")
    print(f"{'Config':<20} {'Seed':>6} {'Loss@100':>10} {'Loss@500':>10} {'Loss@1000':>10} {'Eval':>10}")
    print("-" * 80)

    ordered_names = [r[0] for r in rows]
    for config_name in ordered_names:
        runs = sorted(by_config[config_name], key=lambda x: x["seed"])
        for r in runs:
            def _v(key):
                v = r[key]
                return f"{v:.4f}" if v is not None else "N/A"
            print(
                f"{config_name:<20} {r['seed']:>6} "
                f"{_v('loss_100'):>10} {_v('loss_500'):>10} "
                f"{_v('final_train_loss'):>10} {_v('final_eval_loss'):>10}"
            )
        print()

    print(sep)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    train_dir = os.path.join(project_root, f"tokens/chunks30_{ENCODER}")
    eval_dir = os.path.join(project_root, f"tokens/chunks-eval_{ENCODER}")

    enc_config = get_encoder_config(ENCODER)

    # Load datasets ONCE — reused across every config and seed
    print("Loading datasets...")
    t0 = time()
    train_dataset = TokenDataset(train_dir, enc_config.bos_token_id, enc_config.eos_token_id)
    eval_dataset = EvalDataset(eval_dir, enc_config.bos_token_id, enc_config.eos_token_id)
    print(f"Datasets loaded in {time() - t0:.1f}s | Eval: {len(eval_dataset)} blocks")

    total_runs = len(CONFIGS) * len(SEEDS)
    print(f"\nEncoder: {ENCODER} (vocab={enc_config.vocab_size})")
    print(f"Configs: {len(CONFIGS)} | Seeds: {SEEDS} | Steps/run: {MAX_STEPS}")
    print(f"Total runs: {total_runs}")

    # Print config table
    print(f"\n{'=' * 75}")
    print("CONFIGURATIONS")
    print(f"{'=' * 75}")
    for i, c in enumerate(CONFIGS, 1):
        p = MODEL_PROFILES[c["model"]]
        print(
            f"  {i}. {c['name']:<20} "
            f"h={p['hidden_size']:<5} l={p['num_layers']}  "
            f"lr={c['lr']:.0e}  β2={c['beta2']}  "
            f"wd={c['wd']}  gnorm={c['grad_norm']}"
        )
    print()

    # --- Run grid: iterate seeds in outer loop so data stays cached ---
    all_results = []
    run_idx = 0

    for seed in SEEDS:
        print(f"\n{'=' * 75}")
        print(f"SEED {seed}")
        print(f"{'=' * 75}")

        for config in CONFIGS:
            run_idx += 1
            result = run_single(
                config, seed, train_dataset, eval_dataset,
                enc_config, run_idx, total_runs,
            )
            all_results.append(result)

    # --- Summary ---
    print_summary(all_results)

    # --- Save raw results to JSON ---
    results_path = os.path.join(project_root, "logs/train/grid_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    json_results = []
    for r in all_results:
        jr = dict(r)
        jr["train_losses"] = {str(k): v for k, v in r["train_losses"].items()}
        jr["eval_losses"] = {str(k): v for k, v in r["eval_losses"].items()}
        json_results.append(jr)

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nRaw results saved to {results_path}")
