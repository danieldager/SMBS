"""LSTM Hyperparameter Grid Search.

Tests key hyperparameter combinations for LSTM language model.
Runs seeds × configs × limited steps each.

Usage:
    smbs grid --encoder spidr_base
    smbs grid --encoder spidr_base --max-steps 2000 --seeds 42 123
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

from smbs.config import MAX_TOKENS, TOKENS_DIR, WEIGHTS_DIR
from smbs.encode import get_encoder_config
from smbs.train.dataset import TokenDataset, EvalDataset, collate_fn
from smbs.train.models import LSTM, LSTMConfig


# ============================================================================
# GRID CONFIGURATION
# ============================================================================

DEFAULT_MAX_STEPS = 1000
DEFAULT_SEEDS = [42, 123, 456]
LOGGING_STEPS = 50
EVAL_STEPS = 500
WARMUP_STEPS = 100

MODEL_PROFILES = {
    "small": {"embedding_dim": 256, "hidden_size": 256, "num_layers": 2, "dropout": 0.0},
    "large": {"embedding_dim": 200, "hidden_size": 1024, "num_layers": 3, "dropout": 0.1},
}

CONFIGS = [
    # --- Endpoints ---
    {"name": "paper_default", "model": "large", "lr": 1e-4, "beta2": 0.98, "wd": 0.01, "grad_norm": 0.0},
    {"name": "optimal", "model": "small", "lr": 1e-2, "beta2": 0.99, "wd": 0.0, "grad_norm": 5.0},
    # --- Ablations from optimal (change ONE factor) ---
    {"name": "opt-low_lr", "model": "small", "lr": 1e-4, "beta2": 0.99, "wd": 0.0, "grad_norm": 5.0},
    {"name": "opt-no_clip", "model": "small", "lr": 1e-2, "beta2": 0.99, "wd": 0.0, "grad_norm": 0.0},
    {"name": "opt-with_wd", "model": "small", "lr": 1e-2, "beta2": 0.99, "wd": 0.01, "grad_norm": 5.0},
    {"name": "opt-beta2_98", "model": "small", "lr": 1e-2, "beta2": 0.98, "wd": 0.0, "grad_norm": 5.0},
    # --- Cross-tests: model size × training recipe ---
    {"name": "large-opt_train", "model": "large", "lr": 1e-2, "beta2": 0.99, "wd": 0.0, "grad_norm": 5.0},
    {"name": "small-paper_tr", "model": "small", "lr": 1e-4, "beta2": 0.98, "wd": 0.01, "grad_norm": 0.0},
]


# ============================================================================
# CALLBACK
# ============================================================================


class GridCallback(TrainerCallback):
    """Lightweight callback that collects loss values and prints progress."""

    def __init__(self):
        self.train_losses = {}
        self.eval_losses = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or os.environ.get("RANK", "0") != "0":
            return

        step = state.global_step
        if "loss" in logs:
            normalized_loss = logs["loss"] / args.gradient_accumulation_steps
            self.train_losses[step] = normalized_loss
            print(f"    Step {step:4d} | Loss: {normalized_loss:.4f} | LR: {logs.get('learning_rate', 0):.2e}")
        if "eval_loss" in logs:
            self.eval_losses[step] = logs["eval_loss"]
            print(f"    EVAL {step:4d} | Eval Loss: {logs['eval_loss']:.4f}")


# ============================================================================
# SINGLE RUN
# ============================================================================


def run_single(config, seed, train_dataset, eval_dataset, enc_config, max_steps, run_idx, total_runs):
    """Train one config+seed combination and return results dict."""
    profile = MODEL_PROFILES[config["model"]]

    model_config = LSTMConfig(
        vocab_size=enc_config.vocab_size,
        bos_token_id=enc_config.bos_token_id,
        eos_token_id=enc_config.eos_token_id,
        **profile,
    )

    name = f"{config['name']}_seed{seed}"
    output_dir = str(WEIGHTS_DIR / "grid" / name)

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
        max_steps=max_steps,
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
        optimizers=(None, None),
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

    def _last(d):
        return d[max(d.keys())] if d else None

    loss_100 = callback.train_losses.get(100)
    loss_500 = callback.train_losses.get(500)
    final_train = _last(callback.train_losses)
    final_eval = _last(callback.eval_losses)

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print(
        f"  Done in {elapsed:.0f}s | "
        f"loss@100={_fmt(loss_100)}  loss@500={_fmt(loss_500)}  "
        f"loss@{max_steps}={_fmt(final_train)}  eval={_fmt(final_eval)}"
    )

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

    rows = []
    for config_name, runs in by_config.items():
        n_params = runs[0]["n_params"]
        model_name = next(c["model"] for c in CONFIGS if c["name"] == config_name)

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

    print(f"\nRanking (by mean eval loss):")
    print("-" * 45)
    for i, row in enumerate(rows, 1):
        name, _, _, _, _, _, _, _, _, meval, seval, _ = row
        eval_str = f"{meval:.4f}" if meval is not None else "N/A"
        print(f"  {i}. {name:<20} eval={eval_str}")

    # Per-seed detail
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
# MAIN ENTRY POINT
# ============================================================================


def run_grid(
    encoder: str = "spidr_base",
    max_steps: int = DEFAULT_MAX_STEPS,
    seeds: list[int] | None = None,
    train_dataset: str = "chunks30",
    eval_dataset: str = "chunks-eval",
) -> None:
    """Run LSTM hyperparameter grid search."""
    seeds = seeds or DEFAULT_SEEDS

    train_dir = str(TOKENS_DIR / f"{train_dataset}_{encoder}")
    eval_dir = str(TOKENS_DIR / f"{eval_dataset}_{encoder}")

    enc_config = get_encoder_config(encoder)

    print("Loading datasets...")
    t0 = time()
    train_ds = TokenDataset(train_dir, enc_config.bos_token_id, enc_config.eos_token_id)
    eval_ds = EvalDataset(eval_dir, enc_config.bos_token_id, enc_config.eos_token_id)
    print(f"Datasets loaded in {time() - t0:.1f}s | Eval: {len(eval_ds)} blocks")

    total_runs = len(CONFIGS) * len(seeds)
    print(f"\nEncoder: {encoder} (vocab={enc_config.vocab_size})")
    print(f"Configs: {len(CONFIGS)} | Seeds: {seeds} | Steps/run: {max_steps}")
    print(f"Total runs: {total_runs}")

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

    all_results = []
    run_idx = 0

    for seed in seeds:
        print(f"\n{'=' * 75}")
        print(f"SEED {seed}")
        print(f"{'=' * 75}")

        for config in CONFIGS:
            run_idx += 1
            result = run_single(
                config, seed, train_ds, eval_ds,
                enc_config, max_steps, run_idx, total_runs,
            )
            all_results.append(result)

    print_summary(all_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="spidr_base")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--train-dataset", default="chunks30")
    parser.add_argument("--eval-dataset", default="chunks-eval")
    args = parser.parse_args()

    run_grid(
        encoder=args.encoder,
        max_steps=args.max_steps,
        seeds=args.seeds,
        train_dataset=args.train_dataset,
        eval_dataset=args.eval_dataset,
    )
