"""Language model training pipeline."""

__all__ = ["LSTM", "LSTMConfig"]


def __getattr__(name):
    if name in __all__:
        from smbs.train.models import LSTM, LSTMConfig
        return {"LSTM": LSTM, "LSTMConfig": LSTMConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
