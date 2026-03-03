#!/usr/bin/env python3
"""Run speaker diarization on a .wav file with pyannote.audio.

Example:
    python dyarization/run_pyannote_diarization.py \
      --audio data/conv-250507.wav \
      --token "$HF_TOKEN" \
      --output-dir data/diarization
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diarize a WAV file with pyannote.audio")
    parser.add_argument("--audio", type=Path, required=True, help="Path to input .wav file")
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="Hugging Face model id for diarization pipeline",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face access token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diarization"),
        help="Directory to write outputs (.rttm and .csv)",
    )
    return parser.parse_args()


def write_csv(diarization, csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_s", "end_s", "speaker"])
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            writer.writerow([f"{turn.start:.3f}", f"{turn.end:.3f}", speaker])


def main() -> int:
    args = parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    if not args.token:
        raise RuntimeError(
            "No Hugging Face token provided. Pass --token or set HF_TOKEN. "
            "You must also accept model terms on Hugging Face for the selected model."
        )

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "pyannote.audio is not installed. Install it with `pip install pyannote.audio`."
        ) from exc

    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=args.token)
    diarization = pipeline(str(args.audio))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.audio.stem
    rttm_path = args.output_dir / f"{stem}.rttm"
    csv_path = args.output_dir / f"{stem}.csv"

    with rttm_path.open("w", encoding="utf-8") as f:
        diarization.write_rttm(f)

    write_csv(diarization, csv_path)

    print(f"Wrote RTTM: {rttm_path}")
    print(f"Wrote CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
