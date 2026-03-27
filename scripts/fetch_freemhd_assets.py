#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.request import urlretrieve


FREE_MHD_REPO = "https://github.com/PlasmaControl/FreeMHD.git"
ZENODO_STARTING = "https://zenodo.org/records/13964055/files/StartingFiles.zip"
ZENODO_FIGURES = "https://zenodo.org/records/13964055/files/FreeMHDPaperAllFigures.zip"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=Path("./external"))
    args = parser.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    repo_dir = dest / "FreeMHD"
    if not repo_dir.exists():
        subprocess.run(["git", "clone", FREE_MHD_REPO, str(repo_dir)], check=True)

    for url in [ZENODO_STARTING, ZENODO_FIGURES]:
        filename = dest / Path(url).name
        if not filename.exists():
            urlretrieve(url, filename)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
