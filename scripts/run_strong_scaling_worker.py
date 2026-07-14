from __future__ import annotations

# ruff: noqa: E402 -- repository-root bootstrap must precede project imports.

import argparse
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lmx
from lmx.fringing import (
    _axial_field_sharding,
    _explicit_deviatoric_stress_duct,
    _face_flux_pressure_projection_duct,
    _flow_rate_inlet_profile,
    _initialize_duct_mass_flux,
    _solvax_implicit_momentum_duct,
    _unpack_duct_mass_flux,
)
from lmx.scaling import (
    benchmark_extruded_inductionless_solve,
    benchmark_sharded_extruded_operator,
)

if ROOT not in Path(lmx.__file__).resolve().parents:
    raise RuntimeError(
        f"Scaling worker imported LMX outside its source tree: {lmx.__file__}"
    )


def _source_fingerprint() -> str:
    """Hash the source and frozen specifications used by this worker."""

    paths = [*sorted((ROOT / "lmx").glob("*.py")), Path(__file__).resolve()]
    paths.extend(
        sorted(
            path
            for path in (ROOT / "benchmarks" / "specs").rglob("*")
            if path.is_file()
        )
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _placement(array) -> dict[str, int | bool]:
    return {"addressable_shards": len(array.addressable_shards),
        "global_shards": len(array.global_shards),
        "replicated": bool(array.sharding.is_fully_replicated)}


def _duct_step_gate(*, nx: int, ny: int, nz: int, iterations: int, num_devices: int):
    """Run one tiny production-faithful compact-flux projection/momentum step."""

    if (nx, ny, nz) != (8, 4, 3):
        raise ValueError("duct_step_gate requires the fixed 8x4x3 mesh")
    if nx % num_devices:
        raise ValueError("duct_step_gate axial cells must divide the device count")
    dx, dt = 0.25, 0.02
    dy, dz = jnp.asarray([0.18, 0.22, 0.27, 0.23]), jnp.asarray([0.21, 0.31, 0.28])
    x = np.linspace(-0.9, 0.8, nx)[:, None, None]
    y = np.linspace(-0.8, 0.7, ny)[None, :, None]
    z = np.linspace(-0.6, 0.9, nz)[None, None, :]
    shape = (nx, ny, nz)
    velocity = np.stack((
        np.broadcast_to(0.24 + 0.035 * x + 0.012 * y - 0.009 * z, shape),
        np.broadcast_to(0.018 * np.sin(np.pi * x) * (1.0 - y**2) + 0.003 * z, shape),
        np.broadcast_to(-0.014 * np.cos(np.pi * x) * (1.0 - z**2) + 0.002 * y, shape)), -1)
    force = np.stack((np.broadcast_to(0.08 + 0.01 * y, shape),
        np.broadcast_to(-0.025 + 0.006 * z, shape),
        np.broadcast_to(0.018 - 0.004 * y, shape)), -1)
    density, viscosity = np.ones(shape), np.full(shape, 0.035)
    pressure, mask, target_flow = np.zeros(shape), np.ones(shape, dtype=bool), 0.16
    mesh = _axial_field_sharding(num_devices).mesh
    field_sharding = NamedSharding(mesh, P("x", None, None))
    vector_sharding = NamedSharding(mesh, P("x", None, None, None))
    flux_sharding = NamedSharding(mesh, P(None, "x", None, None))
    axial_sharding, replicated = NamedSharding(mesh, P("x")), NamedSharding(mesh, P())
    velocity, force = (jax.device_put(value, vector_sharding) for value in (velocity, force))
    density, viscosity, pressure, mask = (jax.device_put(value, field_sharding)
        for value in (density, viscosity, pressure, mask))
    area = dy[:, None] * dz[None, :]

    def initialize(velocity0, density0):
        inlet = velocity0[0].at[..., 0].set(
            _flow_rate_inlet_profile(velocity0[0, ..., 0], area, target_flow))
        return _initialize_duct_mass_flux(
            velocity0, density0, inlet, dx=dx, dy=dy, dz=dz)

    initialize = jax.jit(initialize,
        in_shardings=(vector_sharding, field_sharding),
        out_shardings=(flux_sharding, replicated))
    initial_plus, initial_inlet = initialize(velocity, density)

    def project(velocity0, pressure0, density0, mask0):
        return _face_flux_pressure_projection_duct(
            velocity0[..., 0], velocity0[..., 1], velocity0[..., 2], density0, mask0,
            inlet_flow_rate=target_flow, dt=dt, dx=dx, dy=dy, dz=dz,
            iterations=iterations, tolerance=1.0e-10, fluid_bounds=(0, ny, 0, nz),
            initial_pressure=pressure0, single_reduction=True, include_axial_line=False,
            field_sharding=field_sharding)

    project = jax.jit(project,
        in_shardings=(vector_sharding, field_sharding, field_sharding, field_sharding),
        out_shardings=(field_sharding,) * 4
        + (axial_sharding, replicated, replicated, flux_sharding, replicated))
    projected = project(velocity, pressure, density, mask)
    projected_velocity = jnp.stack(projected[:3], axis=-1)
    rho_phi_plus, rho_phi_inlet = projected[-2:]

    def momentum(velocity0, force0, density0, viscosity0, plus0, inlet0):
        inlet_patch = velocity0[0].at[..., 0].set(inlet0 / (density0[0] * area))
        zero_y, zero_z = jnp.zeros_like(velocity0[:, 0]), jnp.zeros_like(velocity0[:, :, 0])
        boundaries = (inlet_patch, velocity0[-1], zero_y, zero_y, zero_z, zero_z)
        corrected_force = force0 + _explicit_deviatoric_stress_duct(
            velocity0, density0 * viscosity0, boundaries,
            (jnp.full((nx,), dx), dy, dz))
        return _solvax_implicit_momentum_duct(
            velocity0, corrected_force, density0, viscosity0,
            _unpack_duct_mass_flux(plus0, inlet0), boundaries,
            dt=dt, dx=dx, dy=dy, dz=dz, iterations=iterations,
            tolerance=1.0e-10, include_axial_line=False)

    momentum = jax.jit(momentum,
        in_shardings=(vector_sharding,) * 2 + (field_sharding,) * 2
        + (flux_sharding, replicated),
        out_shardings=(vector_sharding, replicated, replicated))
    solved, momentum_residual, momentum_converged = momentum(
        projected_velocity, force, density, viscosity, rho_phi_plus, rho_phi_inlet)
    jax.block_until_ready((projected, solved, momentum_residual))
    full_flux = _unpack_duct_mass_flux(rho_phi_plus, rho_phi_inlet)
    signature = np.concatenate([np.asarray(value).reshape(-1) for value in (
        initial_plus, initial_inlet, *projected[:5], rho_phi_plus, rho_phi_inlet, solved)])
    cut = np.asarray(rho_phi_plus[0, nx // 2 - 1])
    return {"benchmark_kind": "duct_step_gate", "num_devices": num_devices,
        "signature": signature.tolist(), "divergence": float(projected[5]),
        "flow_error": float(projected[6]), "momentum_residual": float(momentum_residual),
        "momentum_converged": bool(momentum_converged),
        "convection_flux_l2": float(sum(jnp.linalg.norm(value) for value in full_flux)),
        "lower_wall_flux": float(max(jnp.max(jnp.abs(full_flux[1][:, 0])),
            jnp.max(jnp.abs(full_flux[2][:, :, 0])))),
        "cut_boundary_separation": float(min(np.linalg.norm(cut - np.asarray(rho_phi_inlet)),
            np.linalg.norm(cut - np.asarray(rho_phi_plus[0, -1])))),
        "placement": {name: _placement(value) for name, value in (
            ("initial_flux", initial_plus), ("velocity", projected[0]),
            ("pressure", projected[3]), ("corrected_flux", rho_phi_plus),
            ("inlet_flux", rho_phi_inlet), ("momentum", solved))}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single strong-scaling benchmark worker."
    )
    parser.add_argument(
        "--benchmark-kind",
        choices=("extruded3d", "extruded_solve", "duct_step_gate"),
        default="extruded3d",
    )
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument("--nz", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-devices", type=int, required=True)
    parser.add_argument("--platform", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument(
        "--restart",
        type=Path,
        default=None,
        help="Verified extruded restart used to initialize solver-faithful timing.",
    )
    args = parser.parse_args(argv)

    if args.benchmark_kind == "duct_step_gate":
        payload = _duct_step_gate(nx=args.nx, ny=args.ny, nz=args.nz,
            iterations=args.iterations, num_devices=args.num_devices)
    elif args.benchmark_kind == "extruded_solve":
        record = benchmark_extruded_inductionless_solve(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            max_steps=args.iterations,
            repeats=args.repeats,
            num_devices=args.num_devices,
            profile_dir=args.profile_dir,
            restart_path=args.restart,
        )
    elif args.benchmark_kind == "extruded3d":
        record = benchmark_sharded_extruded_operator(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            iterations=args.iterations,
            repeats=args.repeats,
            num_devices=args.num_devices,
        )
    if args.benchmark_kind != "duct_step_gate":
        payload = {**record.__dict__}
    payload.update(platform=args.platform, source_fingerprint=_source_fingerprint())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    printed = payload if args.benchmark_kind != "duct_step_gate" else {
        key: value for key, value in payload.items() if key != "signature"}
    print(json.dumps(printed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
