#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"meg_path", "word", "start_s", "end_s", "sentence_id", "subject_id"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize lexical manifest for sentence-level word decoding")
    parser.add_argument("--input", required=True, help="CSV/TSV with sentence-word events")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    sep = "\t" if input_path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(input_path, sep=sep)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    out = df.loc[:, ["meg_path", "word", "start_s", "end_s", "sentence_id", "subject_id"]].copy()
    if "sentence_text" in df.columns:
        out["sentence_text"] = df["sentence_text"].astype(str)

    out["meg_path"] = out["meg_path"].map(lambda p: str(Path(p).expanduser().resolve()))
    out["word"] = out["word"].astype(str).str.strip()
    out["sentence_id"] = out["sentence_id"].astype(str)
    out["subject_id"] = out["subject_id"].astype(str)
    out = out[(out["word"] != "") & (out["end_s"] > out["start_s"])]

    out = out.sort_values(["subject_id", "sentence_id", "start_s"]).reset_index(drop=True)
    out["word_index"] = out.groupby(["subject_id", "sentence_id"]).cumcount()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out)} rows to {output_path}")


if __name__ == "__main__":
    main()
