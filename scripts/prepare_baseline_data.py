#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DRIVE_URL = "https://drive.google.com/drive/folders/1dPaFP9IrlA_GUoEc-O4okDV6HhquwCU9"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare baseline data in one command: download MEG FIF from Drive and build manifest.csv."
    )
    parser.add_argument("--drive-url", default=DEFAULT_DRIVE_URL, help="Google Drive folder URL")
    parser.add_argument(
        "--meg-dir",
        default="artifacts/raw_meg",
        help="Directory where MEG FIF files will be copied (default: artifacts/raw_meg)",
    )
    parser.add_argument("--audio-dir", required=True, help="Directory with audio files")
    parser.add_argument(
        "--manifest-output",
        default="artifacts/manifest.csv",
        help="Output CSV path for baseline manifest (default: artifacts/manifest.csv)",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require complete 4-file set for each subject (default: true)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Pass through to fetch step to keep temporary downloaded folder",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    fetch_script = project_root / "scripts" / "fetch_drive_meg.py"
    manifest_script = project_root / "scripts" / "build_manifest.py"
    meg_dir = Path(args.meg_dir).expanduser().resolve()
    manifest_output = Path(args.manifest_output).expanduser().resolve()

    fetch_cmd = [
        sys.executable,
        str(fetch_script),
        "--drive-url",
        args.drive_url,
        "--output-dir",
        str(meg_dir),
    ]
    if args.strict:
        fetch_cmd.append("--strict")
    if args.keep_temp:
        fetch_cmd.append("--keep-temp")

    print("Step 1/2: downloading and selecting MEG files from Google Drive...")
    subprocess.run(fetch_cmd, check=True)

    manifest_cmd = [
        sys.executable,
        str(manifest_script),
        "--meg-dir",
        str(meg_dir),
        "--audio-dir",
        str(Path(args.audio_dir).expanduser().resolve()),
        "--output",
        str(manifest_output),
    ]
    print("Step 2/2: building manifest CSV...")
    subprocess.run(manifest_cmd, check=True)

    print(f"Done. Manifest saved to: {manifest_output}")


if __name__ == "__main__":
    main()
