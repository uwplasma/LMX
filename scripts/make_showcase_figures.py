"""Regenerate the README gallery assets from the public LMX API.

Run ``python scripts/make_showcase_figures.py`` (CPU, about three minutes).
Outputs go to ``docs/_static``: a 256^2 quasi-2D turbulence animation and
poster, a Hunt-duct Hartmann-number sweep, and the validation ladder of the
staggered core against the spectral reference. Sizes are kept under the
documented Git media budget; production-resolution movies are release assets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import lmx  # noqa: E402
from lmx.cases import solve_steady  # noqa: E402
from lmx.validation import extract_midplane_profile  # noqa: E402

STATIC = Path(__file__).resolve().parents[1] / "docs" / "_static"


def q2d_turbulence(n: int = 256, steps: int = 3000, stride: int = 50, seed: int = 3) -> None:
    """Decaying Q2D turbulence: band-limited random vorticity, weak friction."""
    length = 2.0 * np.pi
    kx = np.fft.fftfreq(n, d=1.0 / n)[:, None]
    ky = np.fft.fftfreq(n, d=1.0 / n)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k0 = 12.0
    amplitude = np.where(k > 0, (k / k0) ** 4 * np.exp(-2.0 * (k / k0) ** 2), 0.0)
    phase = np.exp(2j * np.pi * np.asarray(jax.random.uniform(jax.random.PRNGKey(seed), (n, n))))
    vorticity = np.fft.ifftn(amplitude * phase).real
    vorticity *= 4.0 / np.sqrt(np.mean(vorticity**2))
    problem = lmx.Q2DProblem(
        jnp.asarray(vorticity, dtype=jnp.float32),
        length=(length, length),
        viscosity=2.0e-4,
        hartmann_friction=2.0e-2,
        dt=2.0e-3,
        steps=steps,
        history_stride=stride,
        energy_budget_tolerance=5.0e-2,
    )
    result = lmx.solve(problem)
    frames = np.asarray(result.vorticity_history)
    limit = float(np.percentile(np.abs(frames), 99.5))
    print(
        f"q2d: {result.status}, {frames.shape[0]} frames, energy defect "
        f"{float(result.diagnostics.energy_budget_residual):.2e}"
    )

    cmap = plt.get_cmap("RdBu_r")
    images = []
    for frame in frames[::2]:
        rgba = cmap(0.5 + 0.5 * np.clip(frame.T[::-1] / limit, -1.0, 1.0))
        images.append(
            Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8)).resize((320, 320), Image.LANCZOS)
        )
    images[0].save(
        STATIC / "q2d_turbulence_256.webp",
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        quality=52,
        method=6,
    )

    def spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k2 = kx**2 + ky**2
        psi_hat = np.divide(np.fft.fftn(field), k2, out=np.zeros((n, n), complex), where=k2 > 0)
        energy_k = 0.5 * np.abs(psi_hat) ** 2 * k2 / n**4
        bins = np.arange(1, n // 2)
        return bins, np.array([energy_k[(k >= b - 0.5) & (k < b + 0.5)].sum() for b in bins])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    snapshots = (0, len(frames) // 3, len(frames) - 1)
    for axis, index in zip(axes[:3], snapshots, strict=True):
        axis.imshow(
            frames[index].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(0, length, 0, length),
        )
        axis.set_title(f"t = {index * stride * problem.dt:.1f}")
        axis.set_xticks([])
        axis.set_yticks([])
    for index, style in zip(snapshots, (":", "--", "-"), strict=True):
        bins, energy = spectrum(frames[index])
        axes[3].loglog(bins, energy, style, color="k", label=f"t = {index * stride * problem.dt:.1f}")
    guide = np.arange(8, 60)
    axes[3].loglog(guide, 3.0e-1 * guide**-3.0, color="tab:red", lw=1, label="$k^{-3}$")
    axes[3].set_xlabel("k")
    axes[3].set_ylabel("E(k)")
    axes[3].set_title("Energy spectrum")
    axes[3].legend(frameon=False)
    _save_webp(fig, STATIC / "q2d_turbulence_poster.webp")


def hunt_sweep(hartmann_numbers: tuple[float, ...] = (20.0, 100.0, 500.0, 1000.0), cells: int = 72) -> None:
    """Hunt duct profiles versus Ha: side-layer jets and their Ha^-1/2 thickness."""
    profiles, maps, peaks = {}, {}, []
    for ha in hartmann_numbers:
        solution = solve_steady(lmx.make_hunt_case(ha=ha, ny=cells, nz=cells))
        if not solution.converged:
            raise RuntimeError(f"Hunt Ha={ha:g} ended with {solution.status}")
        profile = extract_midplane_profile(solution, axis="z")
        z, u = np.asarray(profile["z"]), np.asarray(profile["u"])
        profiles[ha] = (z, u / (np.trapezoid(u, z) / (z[-1] - z[0])))
        maps[ha] = (
            np.asarray(solution.mesh.y_centers),
            np.asarray(solution.mesh.z_centers),
            np.asarray(solution.state.u),
        )
        half = len(u) // 2
        peaks.append((ha, 1.0 + z[np.argmax(u[:half])]))
        print(f"hunt Ha={ha:g}: {solution.steps} steps, jet at wall distance {peaks[-1][1]:.4f}")

    fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.9])
    ax_profile, ax_map, ax_scaling = (fig.add_subplot(grid[i]) for i in range(3))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(hartmann_numbers)))
    for (ha, (z, u)), color in zip(profiles.items(), colors, strict=True):
        ax_profile.plot(z, u, color=color, lw=1.8, label=f"Ha = {ha:g}")
    ax_profile.set_xlabel("z / a")
    ax_profile.set_ylabel("u / ⟨u⟩")
    ax_profile.set_title("Hunt duct midplane profiles: side-layer jets grow with Ha")
    ax_profile.set_xlim(-1, 1)
    ax_profile.legend(frameon=False)
    ha_max = hartmann_numbers[-1]
    y, z, u = maps[ha_max]
    image = ax_map.pcolormesh(z, y, u / np.max(u), cmap="magma", shading="auto", vmin=0, vmax=1)
    ax_map.set_title(f"u / max u at Ha = {ha_max:g}, c = 0.05, {cells}² fluid cells")
    ax_map.set_xlabel("z / a")
    ax_map.set_ylabel("y / a")
    ax_map.set_aspect("equal")
    fig.colorbar(image, ax=ax_map, shrink=0.85)
    ha_values, distances = (np.array(v) for v in zip(*peaks, strict=True))
    ax_scaling.loglog(ha_values, distances, "o", color="k", label="jet maximum, LMX")
    guide = np.array([15.0, 1500.0])
    ax_scaling.loglog(
        guide,
        distances[1] * (guide / ha_values[1]) ** -0.5,
        "--",
        color="tab:red",
        label=r"$\propto Ha^{-1/2}$",
    )
    ax_scaling.set_xlabel("Ha")
    ax_scaling.set_ylabel("distance of jet maximum from side wall / a")
    ax_scaling.set_title("Side-layer thickness")
    ax_scaling.legend(frameon=False)
    _save_webp(fig, STATIC / "hunt_side_layers.webp")


def validation_ladder(
    hartmann_numbers: tuple[float, ...] = (20.0, 100.0, 300.0),
    conductances: tuple[float, ...] = (0.0, 0.027, 0.1),
    cells: int = 48,
) -> None:
    """Rows 2-4 of the validation ladder, solved by the staggered core.

    Every curve here is the steady Newton-Krylov solve of `lmx.steady` on a
    wall-resolving mesh, and every reference is `validation.shercliff`, which
    shares no operator, mesh or solver with the package. The point of the middle
    panel is that the error does not grow with the field: the Hartmann layer is
    resolved rather than tolerated.
    """
    from validation.shercliff import duct_flow

    # The gates here are parts in a thousand; float32 cannot express them.
    lmx.enable_x64()
    profiles, errors, refinement = {}, {}, {}
    for hartmann in hartmann_numbers:
        for conductance in conductances:
            problem = _duct_problem(cells, hartmann, conductance)
            rate, velocity = _steady_duct(problem)
            exact = _reference_rate(hartmann, conductance)
            errors[(hartmann, conductance)] = abs(rate - exact) / exact
            if conductance == 0.0:
                profiles[hartmann] = (
                    np.asarray(problem.grid.centers[1]),
                    np.asarray(velocity)[0][:, cells // 2],
                )
            print(f"ladder Ha={hartmann:g} c={conductance:g}: {rate:.6g} against {exact:.6g}")
    for count in (24, 32, 40, 48):
        problem = _duct_problem(count, 20.0, 0.0)
        rate, _ = _steady_duct(problem)
        refinement[count] = abs(rate - _reference_rate(20.0, 0.0)) / _reference_rate(20.0, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(hartmann_numbers)))
    for (hartmann, (y, u)), color in zip(profiles.items(), colors, strict=True):
        half = y < 0.0
        axes[0].semilogx(
            (1.0 + y[half]) * hartmann, u[half] / np.max(u), color=color, lw=1.8, label=f"Ha = {hartmann:g}"
        )
        nodes, reference, _ = duct_flow(hartmann, 48)
        middle = reference[:, reference.shape[1] // 2]
        inside = nodes < 0.0
        axes[0].semilogx(
            (1.0 + nodes[inside]) * hartmann,
            middle[inside] / np.max(middle),
            "o",
            ms=2.5,
            color=color,
            alpha=0.5,
        )
    wall = np.logspace(-1.5, 1.2, 60)
    axes[0].semilogx(wall, 1.0 - np.exp(-wall), "--", color="tab:red", lw=1.2, label=r"$1-e^{-\xi}$")
    axes[0].set_xlabel(r"$\xi = (1 + y/a)\,Ha$   (wall distance in layer widths)")
    axes[0].set_ylabel("u / max u   (lines LMX, points spectral)")
    axes[0].set_title("Hartmann layers collapse", fontsize=11)
    axes[0].set_xlim(3.0e-2, 20.0)
    axes[0].legend(frameon=False, loc="lower right")

    markers = {0.0: "o", 0.027: "s", 0.1: "^"}
    for conductance in conductances:
        values = [errors[(hartmann, conductance)] for hartmann in hartmann_numbers]
        axes[1].loglog(
            hartmann_numbers,
            values,
            markers[conductance] + "-",
            lw=1.4,
            label=f"c = {conductance:g}" if conductance else "insulating",
        )
    axes[1].axhline(0.02, color="tab:red", ls="--", lw=1, label="2 % gate")
    axes[1].set_xlabel("Ha")
    axes[1].set_ylabel("relative error in Q / A")
    axes[1].set_title(f"Flow rate vs the spectral reference, {cells}² cells", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8)

    counts = np.array(sorted(refinement))
    values = np.array([refinement[count] for count in counts])
    axes[2].loglog(counts, values, "o-", color="k", label="LMX, Ha = 20")
    axes[2].loglog(
        counts, values[0] * (counts / counts[0]) ** -2.0, "--", color="tab:red", label="second order"
    )
    axes[2].set_xlabel("cells per transverse direction")
    axes[2].set_ylabel("relative error in Q / A")
    axes[2].set_title("Mesh convergence, insulating Ha = 20", fontsize=11)
    axes[2].legend(frameon=False)
    _save_webp(fig, STATIC / "validation_ladder.webp")


def _duct_problem(cells: int, hartmann: float, conductance: float):
    from lmx.core3d import duct_problem

    return duct_problem(hartmann=hartmann, cells=cells, wall_conductance=conductance)


def _steady_duct(problem) -> tuple[float, np.ndarray]:
    from lmx.steady import solve_steady_state

    velocity = solve_steady_state(problem, pseudo_step=1.0e3, linear_restart=600).velocity[0].data
    volumes = np.asarray(problem.grid.cell_volumes())[0]
    rate = float((np.asarray(velocity)[0] * volumes).sum() / volumes.sum())
    return rate, np.asarray(velocity)


def _reference_rate(hartmann: float, conductance: float) -> float:
    from validation.shercliff import flow_rate

    return flow_rate(hartmann, 48, hartmann_wall=conductance)


def _save_webp(fig: plt.Figure, path: Path, dpi: int = 120) -> None:
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    Image.open(png).convert("RGB").save(path, quality=82, method=6)
    png.unlink()
    plt.close(fig)
    print(f"wrote {path.name}: {path.stat().st_size / 1024:.0f} KiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=("q2d", "hunt", "ladder"), help="Regenerate one asset group.")
    args = parser.parse_args()
    STATIC.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "q2d"):
        q2d_turbulence()
    if args.only in (None, "hunt"):
        hunt_sweep()
    if args.only in (None, "ladder"):
        validation_ladder()


if __name__ == "__main__":
    main()
