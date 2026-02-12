"""Deep pipelines for MEG speech decoding."""

from .train_megformer import run_megformer_training
from .train_word_decoder import run_word_decoder_training

__all__ = ["run_megformer_training", "run_word_decoder_training"]
