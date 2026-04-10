#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, colors
import numpy as np


def _portable_path(path: str | Path, *, relative_to: str | Path | None = None) -> str:
    candidate = Path(path)
    base = Path(relative_to) if relative_to is not None else Path.cwd()
    try:
        return str(candidate.relative_to(base))
    except ValueError:
        try:
            return str(candidate.resolve().relative_to(base.resolve()))
        except ValueError:
            return candidate.name if candidate.name else str(candidate)


def _style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _metadata(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata_json" not in data:
        return {}
    return json.loads(str(data["metadata_json"]))


def _plot_field(ax: plt.Axes, z_faces: np.ndarray, y_faces: np.ndarray, field: np.ndarray, title: str) -> None:
    field_min = float(np.min(field))
    field_max = float(np.max(field))
    if field_min >= 0.0:
        cmap = "magma"
        norm = colors.Normalize(vmin=field_min, vmax=max(field_max, field_min + 1e-12))
    elif field_max <= 0.0:
        cmap = "magma_r"
        norm = colors.Normalize(vmin=min(field_min, field_max - 1e-12), vmax=field_max)
    else:
        vmax = max(float(np.max(np.abs(field))), 1e-12)
        cmap = "RdBu_r"
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax.pcolormesh(z_faces, y_faces, field, shading="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_solution_npz(npz_path: str | Path, output_dir: str | Path | None = None, *, title: str | None = None) -> list[Path]:
    _style()
    npz_path = Path(npz_path)
    output_dir = Path(output_dir) if output_dir else npz_path.with_suffix("").parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=False) as data:
        meta = _metadata(data)
        title = title or str(meta.get("case", npz_path.stem))
        y_centers = data["y_centers"]
        z_centers = data["z_centers"]
        y_faces = data["y_faces"]
        z_faces = data["z_faces"]
        u = data["u"]
        phi = data["phi"]
        mid_y = u[:, len(z_centers) // 2]
        mid_z = u[len(y_centers) // 2, :]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        fig.suptitle(title, fontsize=16)
        _plot_field(axes[0, 0], z_faces, y_faces, u, "Velocity u")
        _plot_field(axes[0, 1], z_faces, y_faces, phi, "Electric potential phi")
        axes[1, 0].plot(y_centers / max(np.max(np.abs(y_centers)), 1e-12), mid_y / max(np.max(np.abs(mid_y)), 1e-12), color="#0f766e")
        axes[1, 0].set_title("Midplane y profile")
        axes[1, 0].set_xlabel("Normalized y")
        axes[1, 0].set_ylabel("Normalized u")
        axes[1, 1].plot(z_centers / max(np.max(np.abs(z_centers)), 1e-12), mid_z / max(np.max(np.abs(mid_z)), 1e-12), color="#0f766e")
        axes[1, 1].set_title("Midplane z profile")
        axes[1, 1].set_xlabel("Normalized z")
        axes[1, 1].set_ylabel("Normalized u")

        overview_png = output_dir / "overview_from_npz.png"
        overview_pdf = output_dir / "overview_from_npz.pdf"
        fig.savefig(overview_png, bbox_inches="tight")
        fig.savefig(overview_pdf, bbox_inches="tight")
        plt.close(fig)

        outputs = [overview_png, overview_pdf]
        if "time_history" in data and data["time_history"].size:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            time = data["time_history"]
            axes[0].plot(time, data["u_max_history"], label="max |u|", color="#1d4ed8")
            axes[0].plot(time, data["current_max_history"], label="max |J|", color="#b91c1c")
            axes[0].plot(time, data["lorentz_max_history"], label="max |JxB|", color="#6d28d9")
            axes[0].set_title("Trace magnitudes")
            axes[0].set_xlabel("time")
            axes[0].legend()
            axes[1].plot(time, data["residual_history"], label="velocity residual", color="#0f766e")
            axes[1].plot(time, data["potential_residual_history"], label="potential residual", color="#b45309")
            axes[1].set_title("Solver residuals")
            axes[1].set_xlabel("time")
            axes[1].set_yscale("log")
            axes[1].legend()
            diagnostics_png = output_dir / "diagnostics_from_npz.png"
            diagnostics_pdf = output_dir / "diagnostics_from_npz.pdf"
            fig.savefig(diagnostics_png, bbox_inches="tight")
            fig.savefig(diagnostics_pdf, bbox_inches="tight")
            plt.close(fig)
            outputs.extend([diagnostics_png, diagnostics_pdf])
    return outputs


def plot_movie_npz(npz_path: str | Path, output_dir: str | Path | None = None, *, stem: str | None = None, fps: int = 6) -> list[Path]:
    _style()
    npz_path = Path(npz_path)
    output_dir = Path(output_dir) if output_dir else npz_path.with_suffix("").parent / "movie"
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=False) as data:
        meta = _metadata(data)
        stem = stem or str(meta.get("case", npz_path.stem))
        title = str(meta.get("title", stem))
        y_centers = data["y_centers"]
        z_centers = data["z_centers"]
        y_faces = data["y_faces"]
        z_faces = data["z_faces"]
        times = data["time"]
        u_stack = data["u_stack"]

    frame_peaks = np.maximum(np.max(np.abs(u_stack), axis=(1, 2)), 1e-12)
    display_stack = u_stack / frame_peaks[:, None, None]
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    fig2d, ax2d = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax2d.pcolormesh(z_faces, y_faces, display_stack[0], shading="auto", cmap="magma", norm=norm)
    ax2d.set_xlabel("z")
    ax2d.set_ylabel("y")
    ax2d.set_aspect("equal")
    ax2d.set_title(f"{title}\n2D normalized velocity")
    label = ax2d.text(0.02, 0.98, "", transform=ax2d.transAxes, ha="left", va="top", bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    plt.colorbar(image, ax=ax2d, fraction=0.046, pad=0.04, label="u / max|u(t)|")

    def update_2d(index: int):
        image.set_array(display_stack[index].ravel())
        label.set_text(f"t = {times[index]:.2e}\nmax|u| = {frame_peaks[index]:.2e}")
        return image, label

    anim2d = animation.FuncAnimation(fig2d, update_2d, frames=len(times), interval=1000 / fps, blit=False)
    update_2d(len(times) - 1)
    poster_2d = output_dir / f"{stem}_2d_poster.png"
    movie_2d = output_dir / f"{stem}_2d.gif"
    fig2d.savefig(poster_2d, bbox_inches="tight")
    anim2d.save(movie_2d, writer=animation.writers["pillow"](fps=fps), dpi=140)
    plt.close(fig2d)

    zz, yy = np.meshgrid(z_centers, y_centers)
    fig3d = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax3d = fig3d.add_subplot(111, projection="3d")

    def update_3d(index: int):
        ax3d.cla()
        surface = ax3d.plot_surface(zz, yy, display_stack[index], cmap="magma", norm=norm, linewidth=0, antialiased=True)
        ax3d.set_xlabel("z")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("u / max|u(t)|")
        ax3d.set_zlim(0.0, 1.05)
        ax3d.set_title(f"{title}\n3D normalized velocity | t = {times[index]:.2e}")
        ax3d.view_init(elev=26, azim=38 + 8 * index)
        return (surface,)

    anim3d = animation.FuncAnimation(fig3d, update_3d, frames=len(times), interval=1000 / fps, blit=False)
    update_3d(len(times) - 1)
    poster_3d = output_dir / f"{stem}_3d_poster.png"
    movie_3d = output_dir / f"{stem}_3d.gif"
    fig3d.savefig(poster_3d, bbox_inches="tight")
    anim3d.save(movie_3d, writer=animation.writers["pillow"](fps=fps), dpi=140)
    plt.close(fig3d)

    return [poster_2d, movie_2d, poster_3d, movie_3d]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot LMX .npz result files with Matplotlib.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--movies", action="store_true")
    parser.add_argument("--stem", default=None)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args(argv)

    if args.movies:
        outputs = plot_movie_npz(args.npz, args.output, stem=args.stem, fps=args.fps)
    else:
        outputs = plot_solution_npz(args.npz, args.output, title=args.title)
    print(json.dumps({"npz": _portable_path(args.npz), "outputs": [_portable_path(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
