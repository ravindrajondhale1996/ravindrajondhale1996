#!/usr/bin/env python3
"""
Download a Minimax 2.7 Q4 GGUF model for macOS.

By default this script queries a Hugging Face repository, selects the best
matching Q4 GGUF file, and downloads it to ~/Downloads.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

HF_API_MODEL = "https://huggingface.co/api/models/{repo}"
HF_RESOLVE_FILE = "https://huggingface.co/{repo}/resolve/main/{filename}?download=true"

# You can override this at runtime with --repo
DEFAULT_REPO = "bartowski/MiniMax-M2-7B-GGUF"
PREFERRED_Q4_PATTERNS = ("q4_k_m", "q4_0", "q4")


def fetch_model_files(repo: str) -> list[str]:
    url = HF_API_MODEL.format(repo=quote(repo, safe="/"))
    request = Request(url, headers={"User-Agent": "minimax-downloader/1.0"})
    with urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    siblings = payload.get("siblings", [])
    return [entry.get("rfilename", "") for entry in siblings if entry.get("rfilename")]


def select_q4_file(files: list[str]) -> str:
    gguf_files = [f for f in files if f.lower().endswith(".gguf")]
    q4_files = [f for f in gguf_files if "q4" in f.lower()]
    if not q4_files:
        raise RuntimeError("No Q4 GGUF file found in repository.")

    # Prefer common quant variants.
    sorted_candidates = sorted(
        q4_files,
        key=lambda name: next(
            (idx for idx, token in enumerate(PREFERRED_Q4_PATTERNS) if token in name.lower()),
            len(PREFERRED_Q4_PATTERNS),
        ),
    )
    return sorted_candidates[0]


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "minimax-downloader/1.0"})
    with urlopen(request) as response, destination.open("wb") as output:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else None
        downloaded = 0
        chunk_size = 1024 * 1024

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded / total_size * 100
                print(f"\rDownloading... {percent:5.1f}% ({downloaded // (1024 * 1024)} MB)", end="")
            else:
                print(f"\rDownloading... {downloaded // (1024 * 1024)} MB", end="")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Minimax 2.7 Q4 GGUF model for macOS."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Hugging Face model repository (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "Downloads"),
        help="Destination directory for the downloaded GGUF file (default: ~/Downloads)",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional explicit filename in the repo. If omitted, the script auto-selects a Q4 file.",
    )
    return parser


def main() -> int:
    if platform.system() != "Darwin":
        print("Warning: this script is intended for macOS (Darwin). Continuing anyway.", file=sys.stderr)

    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.filename:
            selected = args.filename
        else:
            files = fetch_model_files(args.repo)
            selected = select_q4_file(files)

        url = HF_RESOLVE_FILE.format(repo=args.repo, filename=quote(selected))
        destination = output_dir / Path(selected).name

        print(f"Repository: {args.repo}")
        print(f"File:       {selected}")
        print(f"URL:        {url}")
        print(f"Save to:    {destination}")

        download_file(url, destination)
        print("Download complete.")
        return 0
    except Exception as exc:  # pragma: no cover - simple CLI script
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
