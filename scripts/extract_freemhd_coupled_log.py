from __future__ import annotations

import argparse
import json
from pathlib import Path


def _coerce_value(raw: str):
    if raw in {"true", "false"}:
        return raw == "true"
    try:
        if "." not in raw and "e" not in raw.lower():
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def parse_lmx_diag_line(line: str) -> dict[str, object] | None:
    if not line.startswith("LMX_DIAG "):
        return None
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    payload: dict[str, object] = {"kind": parts[1]}
    for token in parts[2:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        payload[key] = _coerce_value(value)
    return payload


def extract_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        record = parse_lmx_diag_line(line)
        if record is not None:
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract structured LMX_DIAG records from a FreeMHD solver log.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    records = extract_records(args.log_path)
    payload = {"records": records}
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
