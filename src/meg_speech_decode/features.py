from __future__ import annotations

import librosa
import numpy as np


def sliding_windows(data: np.ndarray, sfreq: float, window_s: float, hop_s: float) -> np.ndarray:
    n_channels, n_samples = data.shape
    win = int(round(window_s * sfreq))
    hop = int(round(hop_s * sfreq))
    if win <= 0 or hop <= 0:
        raise ValueError("window and hop sizes must be positive")
    if n_samples < win:
        return np.empty((0, n_channels * 2), dtype=np.float32)

    features = []
    for start in range(0, n_samples - win + 1, hop):
        w = data[:, start : start + win]
        mean = w.mean(axis=1)
        std = w.std(axis=1)
        features.append(np.concatenate([mean, std]))
    return np.asarray(features, dtype=np.float32)


def mel_frames(audio: np.ndarray, sr: int, hop_s: float, n_mels: int) -> np.ndarray:
    hop_length = max(1, int(round(hop_s * sr)))
    mels = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        power=2.0,
    )
    mels_db = librosa.power_to_db(mels, ref=np.max)
    return mels_db.T.astype(np.float32)


def align_feature_frames(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    if n == 0:
        return x[:0], y[:0]
    return x[:n], y[:n]
