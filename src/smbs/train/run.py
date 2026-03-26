"""Language model training — supports LSTM and GPT-2 architectures."""

import os
from time import time

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import PrinterCallback

from smbs.config import MAX_TOKENS, TOKENS_DIR, WEIGHTS_DIR
from smbs.encode import get_encoder_config
from smbs.train.dataset import TokenDataset, EvalDataset, collate_fn
from smbs.train.models import (
    LSTM, LSTMConfig,
    LSTM_MODEL_CONFIG, LSTM_TRAINING_CONFIG,
    GPT2_MODEL_CONFIG, GPT2_TRAINING_CONFIG,
)
from smbs.train.utils import print_training_summary, get_model_timestamp


class CustomCallback(TrainerCallback):
    """Logs loss, tokens/sec, and learning rate during training."""

    def __init__(self, use_lstm=False):
        self.start_time = None
        self.use_lstm = use_lstm

    def on_step_begin(self, args, state, control, **kwargs):
        if self.start_time is None:
            self.start_time = time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if os.environ.get("RANK", "0") != "0":
            return
        if logs is None:
            return
        if "loss" not in logs and "eval_loss" not in logs:
            return

        steps_done = state.global_step

        if "loss" in logs:
            elapsed = time() - self.start_time if self.start_time else 0
            current_loss = logs["loss"]
            if self.use_lstm and args.gradient_accumulation_steps > 1:
                current_loss /= args.gradient_accumulation_steps

            effective_batch_size = (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * args.world_size
            )
            tokens_per_step = effective_batch_size * MAX_TOKENS
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            tokens_per_sec = steps_per_sec * tokens_per_step

            print(
                f"Step: {steps_done:5d} | Loss: {current_loss:.4f} | "
                f"TPS: {tokens_per_sec/1000:.1f}k tokens/s | "
                f"LR: {logs['learning_rate']:.2e}"
            )

        elif "eval_loss" in logs:
            print(
                f"EVAL Step: {steps_done:5d} | Eval Loss: {logs['eval_loss']:.4f}"
            )


def run_train(
    encoder: str,
    arch: str,
    train_dataset: str | None = None,
    eval_dataset: str | None = None,
) -> None:
    """Train a language model on encoded token data.

    Args:
        encoder: Encoder name (e.g. spidr_base, mhubert, hubert-500).
        arch: Architecture — "lstm" or "gpt2".
        train_dataset: Name of training dataset (default: chunks30).
        eval_dataset: Name of eval dataset (default: chunks-eval).
    """
    train_name = train_dataset or "chunks30"
    eval_name = eval_dataset or "chunks-eval"

    train_dir = str(TOKENS_DIR / f"{train_name}_{encoder}")
    eval_dir = str(TOKENS_DIR / f"{eval_name}_{encoder}")

    enc_config = get_encoder_config(encoder)
    vocab_kwargs = {
        "vocab_size": enc_config.vocab_size,
        "bos_token_id": enc_config.bos_token_id,
        "eos_token_id": enc_config.eos_token_id,
    }

    train_ds = TokenDataset(train_dir, enc_config.bos_token_id, enc_config.eos_token_id)
    eval_ds = EvalDataset(eval_dir, enc_config.bos_token_id, enc_config.eos_token_id)

    if os.environ.get("RANK", "0") == "0":
        print(f"Architecture: {arch.upper()}")
        print(f"Eval samples: {len(eval_ds)}")

    timestamp = get_model_timestamp()

    if arch == "lstm":
        config = LSTMConfig(**LSTM_MODEL_CONFIG, **vocab_kwargs)
        model = LSTM(config)
        base_name = f"lstm_h{config.hidden_size}_l{config.num_layers}_d{config.dropout}_{timestamp}"
        training_config = LSTM_TRAINING_CONFIG
    elif arch == "gpt2":
        config = GPT2Config(**GPT2_MODEL_CONFIG, **vocab_kwargs)  # type: ignore
        model = GPT2LMHeadModel(config)
        base_name = f"gpt2_e{config.n_embd}_l{config.n_layer}_h{config.n_head}_{timestamp}"
        training_config = GPT2_TRAINING_CONFIG.copy()
        if enc_config.vocab_size > 1000:
            training_config["per_device_train_batch_size"] = 16
            training_config["gradient_accumulation_steps"] = 8
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    training_args = TrainingArguments(
        output_dir=str(WEIGHTS_DIR / encoder / base_name),
        **training_config,
    )

    optimizers = (None, None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        optimizers=optimizers,
        callbacks=[
            CustomCallback(use_lstm=(arch == "lstm")),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )
    trainer.pop_callback(PrinterCallback)

    if os.environ.get("RANK", "0") == "0":
        print_training_summary(config, training_args, model, MAX_TOKENS)

    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--arch", default="gpt2", choices=["gpt2", "lstm"])
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--eval-dataset", default=None)
    args = parser.parse_args()

    run_train(
        encoder=args.encoder,
        arch=args.arch,
        train_dataset=args.train_dataset,
        eval_dataset=args.eval_dataset,
    )
