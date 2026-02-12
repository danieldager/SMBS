from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from scripts.train.datasets import MAX_TOKENS

# ============================================================================
# MODEL & TRAINING CONFIGURATIONS
# ============================================================================

LSTM_MODEL_CONFIG = {
    "embedding_dim": 200,
    "hidden_size": 1024,
    "num_layers": 3,
    "dropout": 0.1,
}

LSTM_TRAINING_CONFIG = {
    "overwrite_output_dir": True,
    "disable_tqdm": True,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optim": "adamw_torch",
    "learning_rate": 1e-4,
    "max_grad_norm": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "weight_decay": 0.01,
    "lr_scheduler_type": "inverse_sqrt",
    "warmup_steps": 1000,
    "max_steps": 100000,
    "bf16": True,
    "dataloader_num_workers": 4,
    "logging_steps": 100,
    "save_strategy": "steps",
    "save_steps": 1000,
    "save_total_limit": 3,
    "eval_strategy": "steps",
    "eval_steps": 1000,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}

GPT2_MODEL_CONFIG = {
    "n_positions": MAX_TOKENS,
    "n_ctx": MAX_TOKENS,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,
}

GPT2_TRAINING_CONFIG = {
    "overwrite_output_dir": True,
    "disable_tqdm": True,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    # "gradient_checkpointing": True,
    "optim": "adamw_torch",
    "max_grad_norm": 0.0,
    "learning_rate": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "weight_decay": 0.01,
    "lr_scheduler_type": "inverse_sqrt",
    "warmup_steps": 1000,
    "max_steps": 100000,
    "bf16": True,
    "dataloader_num_workers": 4,
    "logging_steps": 100,
    "save_strategy": "steps",
    "save_steps": 1000,
    "save_total_limit": 3,
    "eval_strategy": "steps",
    "eval_steps": 1000,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}



class LSTMConfig(PretrainedConfig):
    """Config for LSTM language model.

    All parameters have defaults to allow transformers to instantiate
    the config for checkpoint saving. Actual values are provided at
    training time via EncoderConfig and model config dictionaries.
    When loading from a checkpoint, from_pretrained() reads them
    from the saved config.json automatically.
    """

    model_type = "lstm"

    def __init__(
        self,
        vocab_size: int = 258,
        embedding_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        bos_token_id: int = 256,
        eos_token_id: int = 257,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


class LSTM(PreTrainedModel):
    config_class = LSTMConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, config.vocab_size)


    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        embeddings = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embeddings)
        logits = self.output(lstm_output)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), 
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits} if loss is not None else logits
