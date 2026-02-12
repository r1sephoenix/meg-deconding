from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class PipelineConfig:
    manifest_csv: Path
    output_dir: Path
    meg_sfreq: float = 100.0
    l_freq: float = 1.0
    h_freq: float = 40.0
    window_size_s: float = 0.05
    hop_size_s: float = 0.02
    n_mels: int = 40
    ridge_alpha: float = 10.0
    test_size: float = 0.2
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw["manifest_csv"] = Path(raw["manifest_csv"]).expanduser().resolve()
        raw["output_dir"] = Path(raw["output_dir"]).expanduser().resolve()
        return cls(**raw)
