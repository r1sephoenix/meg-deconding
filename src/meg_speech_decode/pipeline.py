from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .config import PipelineConfig
from .data import load_audio_segment, load_manifest, load_meg_segment
from .features import align_feature_frames, mel_frames, sliding_windows
from .model import fit_and_eval


def run_baseline(config: PipelineConfig) -> dict[str, float]:
    segments = load_manifest(config.manifest_csv)
    if not segments:
        raise ValueError("Manifest has no rows")

    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    groups: list[int] = []

    for seg_idx, seg in enumerate(tqdm(segments, desc="Building features")):
        meg, meg_sfreq = load_meg_segment(
            seg.meg_path,
            start_s=seg.start_s,
            end_s=seg.end_s,
            target_sfreq=config.meg_sfreq,
            l_freq=config.l_freq,
            h_freq=config.h_freq,
        )
        audio, audio_sfreq = load_audio_segment(seg.audio_path, seg.start_s, seg.end_s)

        x = sliding_windows(meg, sfreq=meg_sfreq, window_s=config.window_size_s, hop_s=config.hop_size_s)
        y = mel_frames(audio, sr=audio_sfreq, hop_s=config.hop_size_s, n_mels=config.n_mels)
        x, y = align_feature_frames(x, y)

        if len(x) == 0:
            continue

        x_all.append(x)
        y_all.append(y)
        groups.extend([seg_idx] * len(x))

    if not x_all:
        raise RuntimeError("No usable frames were built from manifest")

    x_arr = np.concatenate(x_all, axis=0)
    y_arr = np.concatenate(y_all, axis=0)
    group_arr = np.asarray(groups)

    splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_seed)
    train_idx, test_idx = next(splitter.split(x_arr, y_arr, groups=group_arr))

    trained = fit_and_eval(
        x_train=x_arr[train_idx],
        y_train=y_arr[train_idx],
        x_test=x_arr[test_idx],
        y_test=y_arr[test_idx],
        alpha=config.ridge_alpha,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / "ridge_mel.joblib"
    metrics_path = config.output_dir / "metrics.json"

    joblib.dump(trained.pipeline, model_path)
    metrics = {"mse": trained.metrics.mse, "r2": trained.metrics.r2}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
