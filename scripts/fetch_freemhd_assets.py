#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path


FREE_MHD_REPO = "https://github.com/PlasmaControl/FreeMHD.git"
ZENODO_STARTING = "https://zenodo.org/records/13964055/files/StartingFiles.zip"
ZENODO_FIGURES = "https://zenodo.org/records/13964055/files/FreeMHDPaperAllFigures.zip"


def _is_valid_zip(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            archive.testzip()
    except zipfile.BadZipFile:
        return False
    return True


def _download(url: str, path: Path, *, validate_zip: bool = False) -> None:
    if path.exists() and (not validate_zip or _is_valid_zip(path)):
        return
    subprocess.run(["curl", "-L", "-C", "-", url, "-o", str(path)], check=True)
    if validate_zip and not _is_valid_zip(path):
        raise RuntimeError(f"Downloaded archive is still invalid: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=Path("./external"))
    parser.add_argument(
        "--include-starting-files",
        action="store_true",
        help="Also download the large StartingFiles archive (~8.9 GB).",
    )
    args = parser.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    repo_dir = dest / "FreeMHD"
    if not repo_dir.exists():
        subprocess.run(["git", "clone", FREE_MHD_REPO, str(repo_dir)], check=True)

    _download(ZENODO_FIGURES, dest / Path(ZENODO_FIGURES).name, validate_zip=True)
    if args.include_starting_files:
        _download(ZENODO_STARTING, dest / Path(ZENODO_STARTING).name, validate_zip=True)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
