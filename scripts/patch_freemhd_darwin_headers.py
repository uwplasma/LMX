#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


INCLUDE_PATCH_BLOCK = """\
DARWIN_LIB_HEADER_DIRS := $(subst -I$(WM_PROJECT_DIR)/src/OpenFOAM/lnInclude,-idirafter $(WM_PROJECT_DIR)/src/OpenFOAM/lnInclude,$(subst -I$(WM_PROJECT_DIR)/src/OSspecific/POSIX/lnInclude,-idirafter $(WM_PROJECT_DIR)/src/OSspecific/POSIX/lnInclude,$(LIB_HEADER_DIRS)))
"""


def patch_rule_file(path: Path) -> bool:
    text = path.read_text()
    if "DARWIN_LIB_HEADER_DIRS :=" in text:
        return False
    marker = "include $(DEFAULT_RULES)/"
    marker_index = text.find(marker)
    if marker_index == -1:
        raise RuntimeError(f"Could not find DEFAULT_RULES include marker in {path}")
    line_end = text.find("\n", marker_index)
    if line_end == -1:
        raise RuntimeError(f"Could not find end of include line in {path}")
    updated = text[: line_end + 1] + "\n" + INCLUDE_PATCH_BLOCK + text[line_end + 1 :]
    updated = updated.replace("$(FOAM_EXTRA_CXXFLAGS) $(LIB_HEADER_DIRS) -fPIC", "$(FOAM_EXTRA_CXXFLAGS) $(DARWIN_LIB_HEADER_DIRS) -fPIC")
    updated = updated.replace("$(FOAM_EXTRA_CFLAGS) $(LIB_HEADER_DIRS) -fPIC", "$(FOAM_EXTRA_CFLAGS) $(DARWIN_LIB_HEADER_DIRS) -fPIC")
    path.write_text(updated)
    return True


def patch_darwin_header_rules(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    targets = [
        root / "OpenFOAM-v2206" / "wmake" / "rules" / "darwin64Clang" / "c++",
        root / "OpenFOAM-v2206" / "wmake" / "rules" / "darwin64Clang" / "c",
    ]
    results = []
    changed_any = False
    for path in targets:
        exists = path.exists()
        changed = False
        if exists:
            changed = patch_rule_file(path)
            changed_any = changed_any or changed
        results.append({"path": str(path), "exists": exists, "changed": changed})
    return {"repo_root": str(root), "changed": changed_any, "files": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patch local FreeMHD/OpenFOAM Darwin wmake rules to avoid libc++ header shadowing.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external" / "FreeMHD",
        help="FreeMHD checkout root containing OpenFOAM-v2206.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = patch_darwin_header_rules(args.repo_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
