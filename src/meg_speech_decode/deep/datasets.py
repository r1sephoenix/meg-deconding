from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..data import load_meg_segment


@dataclass(slots=True)
class PairSegment:
    meg_path: Path
    audio_path: Path
    start_s: float
    end_s: float


@dataclass(slots=True)
class WordEvent:
    word: str
    onset_s: float
    offset_s: float
    order: int


@dataclass(slots=True)
class SentenceSample:
    sentence_id: str
    sentence_key: str
    subject_id: str
    meg_path: Path
    events: list[WordEvent]


def _fix_length_2d(arr: np.ndarray, target: int) -> np.ndarray:
    if arr.shape[1] >= target:
        return arr[:, :target]
    pad = np.zeros((arr.shape[0], target - arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def _fix_length_1d(arr: np.ndarray, target: int) -> np.ndarray:
    if len(arr) >= target:
        return arr[:target]
    pad = np.zeros((target - len(arr),), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _baseline_correct(meg: np.ndarray, sfreq: float, baseline_s: float) -> np.ndarray:
    n_base = max(1, int(round(sfreq * baseline_s)))
    n_base = min(n_base, meg.shape[1])
    baseline = meg[:, :n_base].mean(axis=1, keepdims=True)
    return meg - baseline


class MegAudioPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_csv: Path,
        meg_sfreq: float,
        l_freq: float,
        h_freq: float,
        segment_s: float,
        audio_sfreq: int = 16000,
    ) -> None:
        df = pd.read_csv(manifest_csv)
        required = {"meg_path", "audio_path", "start_s", "end_s"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        self.samples: list[PairSegment] = [
            PairSegment(
                meg_path=Path(r["meg_path"]).expanduser().resolve(),
                audio_path=Path(r["audio_path"]).expanduser().resolve(),
                start_s=float(r["start_s"]),
                end_s=float(r["end_s"]),
            )
            for r in df.to_dict(orient="records")
        ]
        self.meg_sfreq = meg_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.segment_s = segment_s
        self.audio_sfreq = audio_sfreq

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        end_s = min(sample.end_s, sample.start_s + self.segment_s)

        meg, _ = load_meg_segment(
            sample.meg_path,
            start_s=sample.start_s,
            end_s=end_s,
            target_sfreq=self.meg_sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
        )
        y, _ = librosa.load(
            sample.audio_path,
            sr=self.audio_sfreq,
            mono=True,
            offset=sample.start_s,
            duration=end_s - sample.start_s,
        )

        target_meg_len = int(round(self.meg_sfreq * self.segment_s))
        target_audio_len = int(round(self.audio_sfreq * self.segment_s))

        meg = _fix_length_2d(meg.astype(np.float32), target_meg_len)
        y = _fix_length_1d(y.astype(np.float32), target_audio_len)

        return torch.from_numpy(meg), torch.from_numpy(y)


class SentenceWordDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest_csv: Path,
        meg_sfreq: float,
        l_freq: float,
        h_freq: float,
        window_s: float,
        baseline_s: float,
        max_words_per_sentence: int,
    ) -> None:
        df = pd.read_csv(manifest_csv)
        required = {"meg_path", "word", "start_s", "end_s", "sentence_id", "subject_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        if "word_index" not in df.columns:
            df["word_index"] = df.groupby("sentence_id").cumcount()
        if "sentence_text" not in df.columns:
            df["sentence_text"] = df["sentence_id"].astype(str)

        grouped: list[SentenceSample] = []
        for sentence_id, g in df.groupby("sentence_id", sort=False):
            g = g.sort_values("word_index")
            events = [
                WordEvent(
                    word=str(r["word"]).strip(),
                    onset_s=float(r["start_s"]),
                    offset_s=float(r["end_s"]),
                    order=int(r["word_index"]),
                )
                for r in g.to_dict(orient="records")
                if str(r["word"]).strip() != ""
            ]
            if not events:
                continue
            grouped.append(
                SentenceSample(
                    sentence_id=str(sentence_id),
                    sentence_key=str(g.iloc[0]["sentence_text"]),
                    subject_id=str(g.iloc[0]["subject_id"]),
                    meg_path=Path(str(g.iloc[0]["meg_path"])).expanduser().resolve(),
                    events=events[:max_words_per_sentence],
                )
            )

        if not grouped:
            raise RuntimeError("No valid sentence samples in manifest")

        self.samples = grouped
        self.meg_sfreq = meg_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.window_s = window_s
        self.baseline_s = baseline_s
        self.target_meg_len = int(round(self.meg_sfreq * self.window_s))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        words: list[str] = []
        meg_windows: list[torch.Tensor] = []
        for ev in sample.events:
            end_s = ev.onset_s + self.window_s
            meg, sfreq = load_meg_segment(
                sample.meg_path,
                start_s=ev.onset_s,
                end_s=end_s,
                target_sfreq=self.meg_sfreq,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
            )
            meg = _fix_length_2d(meg.astype(np.float32), self.target_meg_len)
            meg = _baseline_correct(meg, sfreq=sfreq, baseline_s=self.baseline_s)
            meg_windows.append(torch.from_numpy(meg))
            words.append(ev.word)

        return {
            "meg": torch.stack(meg_windows, dim=0),
            "words": words,
            "subject_id": sample.subject_id,
            "sentence_id": sample.sentence_id,
            "sentence_key": sample.sentence_key,
        }


def collate_sentences(batch: list[dict[str, object]]) -> dict[str, object]:
    batch_size = len(batch)
    max_words = max(int(item["meg"].shape[0]) for item in batch)
    channels = int(batch[0]["meg"].shape[1])
    time = int(batch[0]["meg"].shape[2])

    meg_pad = torch.zeros((batch_size, max_words, channels, time), dtype=torch.float32)
    mask = torch.ones((batch_size, max_words), dtype=torch.bool)
    subject_ids: list[str] = []
    sentence_ids: list[str] = []
    sentence_keys: list[str] = []
    words_nested: list[list[str]] = []

    for i, item in enumerate(batch):
        meg = item["meg"]
        n_words = int(meg.shape[0])
        meg_pad[i, :n_words] = meg
        mask[i, :n_words] = False
        subject_ids.append(str(item["subject_id"]))
        sentence_ids.append(str(item["sentence_id"]))
        sentence_keys.append(str(item["sentence_key"]))
        words_nested.append(list(item["words"]))

    return {
        "meg": meg_pad,
        "padding_mask": mask,
        "subject_ids": subject_ids,
        "sentence_ids": sentence_ids,
        "sentence_keys": sentence_keys,
        "words_nested": words_nested,
    }
