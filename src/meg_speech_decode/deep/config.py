from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class MegformerConfig:
    manifest_csv: Path
    output_dir: Path
    meg_sfreq: float = 100.0
    l_freq: float = 1.0
    h_freq: float = 40.0
    segment_s: float = 2.0
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    depth: int = 4
    n_heads: int = 8
    proj_dim: int = 256
    temperature: float = 0.07
    val_size: float = 0.2
    random_seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MegformerConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw["manifest_csv"] = Path(raw["manifest_csv"]).expanduser().resolve()
        raw["output_dir"] = Path(raw["output_dir"]).expanduser().resolve()
        return cls(**raw)


@dataclass(slots=True)
class WordDecoderConfig:
    manifest_csv: Path
    output_dir: Path
    meg_sfreq: float = 100.0
    l_freq: float = 0.1
    h_freq: float = 60.0
    window_s: float = 3.0
    baseline_s: float = 0.5
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.1
    cnn_hidden_dim: int = 256
    proj_dim: int = 1024
    sentence_layers: int = 8
    sentence_heads: int = 8
    sentence_dropout: float = 0.1
    temperature_init: float = 10.0
    bias_init: float = -10.0
    vocab_eval_size: int = 250
    max_words_per_sentence: int = 64
    random_seed: int = 42
    device: str = "cpu"
    t5_model_name: str = "t5-large"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WordDecoderConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw["manifest_csv"] = Path(raw["manifest_csv"]).expanduser().resolve()
        raw["output_dir"] = Path(raw["output_dir"]).expanduser().resolve()
        return cls(**raw)
