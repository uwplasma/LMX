#!/usr/bin/env python3
"""Inventory, bundle, and verify generated LMX release assets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import mimetypes
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.audit_architecture import ROOT, _release_asset_candidates
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from audit_architecture import ROOT, _release_asset_candidates


MANIFEST_PATH = ROOT / "docs" / "release-assets.json"
RELEASE_TAG = "lmx-research-assets-v1"
ARCHIVE_NAME = f"{RELEASE_TAG}.tar.gz"
DOWNLOAD_URL = (
    f"https://github.com/uwplasma/lmx/releases/download/{RELEASE_TAG}/{ARCHIVE_NAME}"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_showcase(root: Path = ROOT) -> dict[str, Any]:
    """Inventory the small media derivatives kept in the repository."""

    paths = sorted(path for path in (root / "docs" / "_static").glob("*") if path.is_file())
    files = [
        {
            "bytes": path.stat().st_size,
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    return {"bytes": sum(item["bytes"] for item in files), "files": files}


def build_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Build the canonical source-asset inventory, grouping duplicate content."""

    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    for item in _release_asset_candidates(root):
        grouped[(str(item["sha256"]), int(item["bytes"]))].append(str(item["path"]))
    assets = []
    for (digest, size), paths in sorted(grouped.items()):
        media_types = sorted(
            {
                mimetypes.guess_type(path)[0] or "application/octet-stream"
                for path in paths
            }
        )
        assets.append(
            {
                "bytes": size,
                "media_types": media_types,
                "paths": sorted(paths),
                "sha256": digest,
            }
        )
    return {
        "schema_version": 2,
        "generated_by": "scripts/manage_release_assets.py",
        "release": {
            "archive_name": ARCHIVE_NAME,
            "download_url": DOWNLOAD_URL,
            "repository": "uwplasma/lmx",
            "status": "planned",
            "tag": RELEASE_TAG,
        },
        "showcase": _tracked_showcase(root),
        "summary": {
            "logical_bytes": sum(
                asset["bytes"] * len(asset["paths"]) for asset in assets
            ),
            "logical_file_count": sum(len(asset["paths"]) for asset in assets),
            "unique_bytes": sum(asset["bytes"] for asset in assets),
            "unique_content_count": len(assets),
        },
        "assets": assets,
    }


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_manifest(path: Path = MANIFEST_PATH, root: Path = ROOT) -> dict[str, Any]:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("release", {}).get("status") == "uploaded":
            payload = existing | {"showcase": _tracked_showcase(root)}
        else:
            payload = build_manifest(root)
    else:
        payload = build_manifest(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload), encoding="utf-8")
    return payload


def check_manifest(path: Path = MANIFEST_PATH, root: Path = ROOT) -> dict[str, Any]:
    tracked = json.loads(path.read_text(encoding="utf-8"))
    if tracked.get("schema_version") != 2:
        raise ValueError("Release-asset manifest schema_version must be 2")
    release = tracked.get("release", {})
    if release.get("status") not in {"planned", "uploaded"}:
        raise ValueError("Release status must be 'planned' or 'uploaded'")
    if release.get("status") == "uploaded" and not release.get("archive_sha256"):
        raise ValueError("Uploaded release assets require archive_sha256")
    if tracked.get("showcase") != _tracked_showcase(root):
        raise ValueError("Tracked showcase media differ from the manifest")
    expected = {
        relative: (int(asset["bytes"]), str(asset["sha256"]))
        for asset in tracked.get("assets", [])
        for relative in asset["paths"]
    }
    current = {str(item["path"]): item for item in _release_asset_candidates(root)}
    unexpected = sorted(set(current) - set(expected))
    if unexpected:
        raise ValueError(f"Untracked generated release assets: {unexpected}")
    missing = []
    for relative, (size, digest) in expected.items():
        source = root / relative
        if not source.is_file():
            missing.append(relative)
            continue
        if source.stat().st_size != size or _sha256(source) != digest:
            raise ValueError(f"Release-asset source drift: {relative}")
    if missing and release.get("status") != "uploaded":
        raise ValueError(f"Planned release assets are missing: {missing}")
    return tracked


def build_archive(output: Path, root: Path = ROOT) -> str:
    """Create a deterministic archive containing every logical source path."""

    manifest = build_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for asset in manifest["assets"]:
                    for relative in asset["paths"]:
                        path = root / relative
                        info = archive.gettarinfo(str(path), arcname=relative)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
    verify_archive(output, manifest)
    return _sha256(output)


def verify_archive(path: Path, manifest: dict[str, Any] | None = None) -> None:
    """Verify archive membership, sizes, and hashes against the manifest."""

    expected = manifest or json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_paths = {
        relative: (int(asset["bytes"]), str(asset["sha256"]))
        for asset in expected["assets"]
        for relative in asset["paths"]
    }
    with tarfile.open(path, mode="r:gz") as archive:
        members = {
            member.name: member for member in archive.getmembers() if member.isfile()
        }
        if set(members) != set(expected_paths):
            raise ValueError("Release archive membership differs from the manifest")
        for relative, (size, digest) in expected_paths.items():
            member = members[relative]
            if member.size != size:
                raise ValueError(f"Release archive size mismatch for {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read release archive member {relative}")
            actual = hashlib.sha256(stream.read()).hexdigest()
            if actual != digest:
                raise ValueError(f"Release archive checksum mismatch for {relative}")


def write_benchmark_a_validation_plot(
    acceptance: dict[str, object],
    output: Path,
    *,
    flow_error_target: float,
    mesh_change_target: float,
    order_target: float = 0.5,
) -> Path:
    """Render the accepted Samper ladder from frozen JSON without a solve."""

    import matplotlib.pyplot as plt
    import numpy as np

    from lmx.plotting import _set_plot_style

    literature = acceptance.get("literature_table_i")
    rows = literature.get("rows") if isinstance(literature, dict) else None
    if not isinstance(rows, list) or len(rows) != 8 or not literature.get("pass"):
        raise ValueError("Benchmark A plot requires eight accepted literature rows")
    grouped = {
        case: sorted(
            (row for row in rows if row.get("case") == case),
            key=lambda row: int(row["hartmann_number"]),
        )
        for case in ("shercliff", "hunt")
    }
    if any(len(case_rows) != 4 for case_rows in grouped.values()):
        raise ValueError("Benchmark A plot requires four Shercliff and four Hunt rows")

    def balance(row: dict[str, object], key: str) -> tuple[float, float]:
        values = row[key]
        if not isinstance(values, dict):
            raise ValueError(f"Benchmark A row lacks {key}")
        residuals = [
            float(value)
            for name, value in values.items()
            if name != "acceptance_target"
            and (name.endswith("_normalized") or name.endswith("_relative_error"))
        ]
        return max(residuals), float(values["acceptance_target"])

    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.5), constrained_layout=True)
    fig.suptitle("Benchmark A: frozen high-Hartmann validation", fontsize=16)
    for case, case_rows in grouped.items():
        ha = np.asarray([row["hartmann_number"] for row in case_rows], dtype=float)
        color, marker = {"shercliff": ("#0f766e", "o"), "hunt": ("#b45309", "s")}[case]
        label = case.capitalize()
        metrics = (
            100 * np.asarray([row["analytical_flow_relative_error"] for row in case_rows]),
            100 * np.asarray([row["finest_mesh_change_relative"] for row in case_rows]),
            np.asarray([row["observed_order"] for row in case_rows]),
        )
        for ax, values in zip((axes[0, 0], axes[0, 1], axes[1, 0]), metrics, strict=True):
            ax.semilogx(ha, values, marker=marker, color=color, label=label)
        for key, fill, linestyle in (
            ("current_balance", "white", "-"),
            ("power_balance", color, "--"),
        ):
            axes[1, 1].loglog(
                ha,
                100 * np.asarray([balance(row, key)[0] for row in case_rows]),
                marker=marker,
                markerfacecolor=fill,
                linestyle=linestyle,
                color=color,
                label=f"{label}: {key.removesuffix('_balance')}",
            )

    balance_target = min(
        balance(row, key)[1]
        for case_rows in grouped.values()
        for row in case_rows
        for key in ("current_balance", "power_balance")
    )
    panels = (
        (axes[0, 0], "Flow rate vs analytical solution", "Relative error [%]", 100 * flow_error_target, "gate 1%"),
        (axes[0, 1], "Finest-grid change", "Relative change [%]", 100 * mesh_change_target, "gate 0.25%"),
        (axes[1, 0], "Observed order (dashed guide: p = 2)", "Observed order", order_target, "gate p ≥ 0.5"),
        (axes[1, 1], "Current and power closure", "Maximum normalized residual [%]", 100 * balance_target, "gate 0.1%"),
    )
    for ax, title, ylabel, target, gate_label in panels:
        ax.axhline(target, color="#475569", linestyle=":", linewidth=1.4)
        ax.set(title=title, xlabel="Hartmann number", ylabel=ylabel)
        y = 0.06 if ax is axes[1, 0] else 0.94
        ax.text(0.98, y, gate_label, transform=ax.transAxes, ha="right", va="top")
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].text(
        0.02, 0.94, "ACCEPTED | frozen JSON", transform=axes[0, 0].transAxes,
        color="#166534", weight="bold", fontsize=9, va="top",
    )
    axes[1, 0].axhline(2.0, color="#94a3b8", linestyle="--", linewidth=1.0)
    axes[1, 1].legend(fontsize=8.5, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight", pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)
    return output


def write_animated_webp(
    source: Path,
    output: Path,
    *,
    width: int = 520,
    seconds: float = 7.0,
    fps: int = 8,
    quality: int = 50,
    sampling_power: float = 1.0,
    last_frame: int | None = None,
    right_source: Path | None = None,
) -> Path:
    """Downsample one animation, or a synchronized pair, into a bounded WebP."""

    from PIL import Image

    if sampling_power <= 0.0:
        raise ValueError("sampling_power must be positive")
    if not 0.0 < seconds <= 7.0:
        raise ValueError("animated WebP duration must be in (0, 7] seconds")
    if fps <= 0:
        raise ValueError("fps must be positive")
    with Image.open(source) as image:
        right = Image.open(right_source) if right_source is not None else None
        frame_count = max(2, round(seconds * fps))
        terminal = image.n_frames - 1 if last_frame is None else int(last_frame)
        if not 1 <= terminal < image.n_frames:
            raise ValueError("last_frame must select at least two available source frames")
        if right is not None and right.n_frames < 2:
            raise ValueError("right_source must contain at least two frames")
        try:
            frames = []
            for index in range(frame_count):
                fraction = (index / (frame_count - 1)) ** sampling_power
                sources = ((image, terminal),) if right is None else (
                    (image, terminal),
                    (right, right.n_frames - 1),
                )
                parts = []
                for side, (animation, end) in enumerate(sources):
                    animation.seek(round(fraction * end))
                    part_width = (
                        width if right is None else width // 2 + side * (width % 2)
                    )
                    height = round(animation.height * part_width / animation.width)
                    parts.append(
                        animation.convert("RGB").resize(
                            (part_width, height), Image.Resampling.LANCZOS
                        )
                    )
                if right is None:
                    frames.append(parts[0])
                else:
                    canvas = Image.new(
                        "RGB", (width, max(part.height for part in parts)), "white"
                    )
                    x = 0
                    for part in parts:
                        canvas.paste(part, (x, (canvas.height - part.height) // 2))
                        x += part.width
                    frames.append(canvas)
        finally:
            if right is not None:
                right.close()
    total_duration_ms = min(round(1000 * seconds), 7000)
    duration_ms, longer_frames = divmod(total_duration_ms, frame_count)
    durations = [
        duration_ms + (index < longer_frames) for index in range(frame_count)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        quality=quality,
        method=6,
        minimize_size=True,
    )
    return output


def write_static_webp(
    source: Path, output: Path, *, width: int = 1200, quality: int = 82
) -> Path:
    """Compress one source figure to a bounded directly embeddable WebP."""

    from PIL import Image

    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.width > width:
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="WEBP", quality=quality, method=6)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--build-archive", type=Path)
    action.add_argument("--verify-archive", type=Path)
    action.add_argument("--require-uploaded", action="store_true")
    action.add_argument("--write-benchmark-a-plot", type=Path)
    action.add_argument(
        "--write-animated-webp",
        type=Path,
        nargs=2,
        metavar=("SOURCE", "OUTPUT"),
    )
    action.add_argument(
        "--write-static-webp", type=Path, nargs=2, metavar=("SOURCE", "OUTPUT")
    )
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--quality", type=int, default=50)
    parser.add_argument("--sampling-power", type=float, default=1.0)
    parser.add_argument(
        "--last-frame",
        type=int,
        default=None,
        help="inclusive terminal source frame for a converged animation",
    )
    parser.add_argument(
        "--right-source",
        type=Path,
        default=None,
        help="optional second animation to display beside SOURCE",
    )
    args = parser.parse_args(argv)
    if args.write_static_webp:
        source, output = args.write_static_webp
        write_static_webp(source, output, width=args.width, quality=args.quality)
        print(f"Static WebP: {output}")
    elif args.write_animated_webp:
        source, output = args.write_animated_webp
        write_animated_webp(
            source,
            output,
            width=args.width,
            seconds=args.seconds,
            fps=args.fps,
            quality=args.quality,
            sampling_power=args.sampling_power,
            last_frame=args.last_frame,
            right_source=args.right_source,
        )
        print(f"Animated WebP: {output}")
    elif args.write_benchmark_a_plot:
        results = ROOT / "benchmarks" / "results"
        acceptance = json.loads(
            (results / "benchmark-a-acceptance.json").read_text(encoding="utf-8")
        )
        table = json.loads(
            (results / "samper-table-i-accepted.json").read_text(encoding="utf-8")
        )
        write_benchmark_a_validation_plot(
            acceptance,
            args.write_benchmark_a_plot,
            flow_error_target=float(table["flow_error_target"]),
            mesh_change_target=float(table["finest_mesh_change_target"]),
        )
        print(f"Benchmark A plot: {args.write_benchmark_a_plot}")
    elif args.write:
        payload = write_manifest()
        print(
            f"release assets: {payload['summary']['logical_file_count']} files inventoried"
        )
    elif args.check:
        check_manifest()
        print("release-asset manifest verified")
    elif args.build_archive:
        digest = build_archive(args.build_archive)
        print(f"release archive sha256={digest}")
    elif args.verify_archive:
        verify_archive(args.verify_archive)
        print("release archive verified")
    else:
        payload = check_manifest()
        if payload["release"]["status"] != "uploaded":
            raise SystemExit("release assets have not been uploaded")
        print("release assets are marked uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
