#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


DOCKERFILE = """\
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \\
    build-essential git curl ca-certificates openmpi-bin libopenmpi-dev python3 && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/PlasmaControl/FreeMHD.git

WORKDIR /opt/FreeMHD
CMD ["/bin/bash"]
"""


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "docker"
    root.mkdir(parents=True, exist_ok=True)
    (root / "Dockerfile").write_text(DOCKERFILE)
    print(root / "Dockerfile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
