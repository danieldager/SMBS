"""Evaluation pipeline — benchmark trained models on lexical tasks."""

from smbs.evaluate.metrics import discrimination_accuracy, per_voice_accuracy

__all__ = ["discrimination_accuracy", "per_voice_accuracy"]
