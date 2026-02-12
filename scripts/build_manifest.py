#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import pandas as pd


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}
MEG_FILE_EXTS = {".fif", ".con", ".sqd", ".pdf"}
MEG_DIR_EXTS = {".ds"}


def index_audio(audio_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in audio_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out[p.stem.lower()] = p
    return out


def find_meg_files(meg_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in meg_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in MEG_FILE_EXTS:
            files.append(p)
        if p.is_dir() and p.suffix.lower() in MEG_DIR_EXTS:
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a rough MEG+audio manifest by stem matching")
    parser.add_argument("--meg-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    meg_dir = Path(args.meg_dir).expanduser().resolve()
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    out_csv = Path(args.output).expanduser().resolve()

    audio_by_stem = index_audio(audio_dir)
    rows = []

    for meg in find_meg_files(meg_dir):
        key = meg.stem.lower()
        audio = audio_by_stem.get(key)
        if audio is None:
            continue

        duration = float(librosa.get_duration(path=str(audio)))
        rows.append(
            {
                "meg_path": str(meg),
                "audio_path": str(audio),
                "start_s": 0.0,
                "end_s": duration,
            }
        )

    if not rows:
        raise RuntimeError("No MEG/audio matches found by stem")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
