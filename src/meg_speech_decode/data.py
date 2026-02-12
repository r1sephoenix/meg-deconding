from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import mne
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"meg_path", "audio_path", "start_s", "end_s"}


@dataclass(slots=True)
class Segment:
    meg_path: Path
    audio_path: Path
    start_s: float
    end_s: float


def load_manifest(path: Path) -> list[Segment]:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    segments: list[Segment] = []
    for row in df.to_dict(orient="records"):
        segments.append(
            Segment(
                meg_path=Path(row["meg_path"]).expanduser().resolve(),
                audio_path=Path(row["audio_path"]).expanduser().resolve(),
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
            )
        )
    return segments


def load_meg_segment(
    path: Path,
    start_s: float,
    end_s: float,
    target_sfreq: float,
    l_freq: float,
    h_freq: float,
) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
    raw.pick("meg")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    raw.resample(target_sfreq, verbose="ERROR")
    seg = raw.copy().crop(tmin=start_s, tmax=end_s, include_tmax=False)
    data = seg.get_data()
    return data.astype(np.float32), seg.info["sfreq"]


def load_audio_segment(path: Path, start_s: float, end_s: float, target_sfreq: int = 16000) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sfreq, mono=True, offset=start_s, duration=end_s - start_s)
    return y.astype(np.float32), sr
