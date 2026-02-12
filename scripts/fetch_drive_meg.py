#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1dPaFP9IrlA_GUoEc-O4okDV6HhquwCU9"
REQUIRED_SUFFIXES = (
    "_wombat_raw_tsss_mc_trans.fif",
    "_hedgehog_raw_tsss_mc_trans.fif",
    "_birch_raw_tsss_mc_trans.fif",
    "_2mothers_raw_tsss_mc_trans.fif",
)


def run_gdown_folder(url: str, output_dir: Path) -> None:
    cmd = [
        "gdown",
        "--folder",
        url,
        "--output",
        str(output_dir),
        "--remaining-ok",
    ]
    subprocess.run(cmd, check=True)


def collect_matching_files(root: Path) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*.fif"):
        name = path.name
        if any(name.endswith(suffix) for suffix in REQUIRED_SUFFIXES):
            matches.append(path)
    return matches


def group_subject_files(paths: list[Path]) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in paths:
        name = path.name
        for suffix in REQUIRED_SUFFIXES:
            if not name.endswith(suffix):
                continue
            subject_id = name[: -len(suffix)]
            if subject_id not in grouped:
                grouped[subject_id] = {}
            grouped[subject_id][suffix] = path
            break
    return grouped


def copy_complete_sets(grouped: dict[str, dict[str, Path]], output_dir: Path, strict: bool) -> tuple[int, int]:
    copied_files = 0
    copied_subjects = 0

    for subject_id, files_by_suffix in sorted(grouped.items()):
        missing = [suffix for suffix in REQUIRED_SUFFIXES if suffix not in files_by_suffix]
        if missing:
            if strict:
                missing_str = ", ".join(missing)
                raise RuntimeError(f"Incomplete file set for subject '{subject_id}'. Missing: {missing_str}")
            continue

        copied_subjects += 1
        for suffix in REQUIRED_SUFFIXES:
            src = files_by_suffix[suffix]
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            copied_files += 1

    return copied_subjects, copied_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download MEG FIF files from Google Drive and copy complete 4-file sets "
            "(_wombat/_hedgehog/_birch/_2mothers) for each subject."
        )
    )
    parser.add_argument("--drive-url", default=DRIVE_FOLDER_URL, help="Google Drive folder URL")
    parser.add_argument(
        "--output-dir",
        default="artifacts/raw_meg",
        help="Where to copy the selected FIF files (default: artifacts/raw_meg)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary downloaded folder for debugging",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any subject has an incomplete 4-file set",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_ctx = tempfile.TemporaryDirectory(prefix="drive_meg_")
    tmp_dir = Path(tmp_ctx.name).resolve()

    try:
        print(f"Downloading Google Drive folder to: {tmp_dir}")
        run_gdown_folder(args.drive_url, tmp_dir)

        found = collect_matching_files(tmp_dir)
        if not found:
            raise RuntimeError("No matching .fif files found in downloaded Google Drive folder")

        grouped = group_subject_files(found)
        subjects, files = copy_complete_sets(grouped, out_dir, strict=args.strict)

        if subjects == 0:
            raise RuntimeError("No complete 4-file subject sets were found")

        print(f"Copied {files} files for {subjects} subject(s) to: {out_dir}")
    finally:
        if args.keep_temp:
            print(f"Temporary files kept at: {tmp_dir}")
        else:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
