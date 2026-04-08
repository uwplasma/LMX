from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import tomllib

from .specs import (
    BoundaryCondition,
    CaseSpec,
    GeometrySpec,
    MagneticFieldSpec,
    OutputSpec,
    RegionSpec,
    TimeStepperConfig,
)


SolveMode = Literal["steady", "transient"]


@dataclass(frozen=True)
class LoggingSpec:
    enabled: bool = True
    banner: bool = True
    print_regions: bool = True
    print_boundaries: bool = True
    print_footer: bool = True
    flush: bool = True
    step_stride: int = 1


@dataclass(frozen=True)
class RunConfig:
    case: CaseSpec
    solve_mode: SolveMode = "steady"
    logging: LoggingSpec = field(default_factory=LoggingSpec)
    input_path: Path | None = None


def _load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required TOML key '{key}'")
    return mapping[key]


def _optional_tuple(mapping: dict[str, Any], key: str, *, length: int | None = None, cast=float) -> tuple[Any, ...] | None:
    if key not in mapping:
        return None
    values = tuple(cast(value) for value in mapping[key])
    if length is not None and len(values) != length:
        raise ValueError(f"TOML key '{key}' must have length {length}")
    return values


def _parse_regions(entries: list[dict[str, Any]]) -> tuple[RegionSpec, ...]:
    return tuple(
        RegionSpec(
            name=str(_require(entry, "name")),
            kind=str(_require(entry, "kind")),
            conductivity=float(_require(entry, "conductivity")),
            density=None if entry.get("density") is None else float(entry["density"]),
            viscosity=None if entry.get("viscosity") is None else float(entry["viscosity"]),
            wall_thickness=None if entry.get("wall_thickness") is None else float(entry["wall_thickness"]),
        )
        for entry in entries
    )


def _parse_boundary_value(value: Any) -> float | tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        values = tuple(float(component) for component in value)
        if len(values) != 3:
            raise ValueError("Boundary-condition vector values must have length 3")
        return values
    raise ValueError(f"Unsupported boundary-condition value {value!r}")


def _parse_boundaries(entries: list[dict[str, Any]]) -> tuple[BoundaryCondition, ...]:
    return tuple(
        BoundaryCondition(
            name=str(_require(entry, "name")),
            kind=str(_require(entry, "kind")),
            value=_parse_boundary_value(entry.get("value")),
            region=None if entry.get("region") is None else str(entry["region"]),
            axis=None if entry.get("axis") is None else str(entry["axis"]),
            side=None if entry.get("side") is None else str(entry["side"]),
        )
        for entry in entries
    )


def load_run_config(path: str | Path) -> RunConfig:
    input_path = Path(path).resolve()
    root = _load_toml(input_path)

    case_table = root.get("case", {})
    geometry_table = root.get("geometry", {})
    field_table = root.get("magnetic_field", {})
    time_table = root.get("time_stepper", {})
    output_table = root.get("output", {})
    logging_table = root.get("logging", {})
    regions_table = root.get("regions", [])
    boundaries_table = root.get("boundary_conditions", [])

    if field_table.get("kind") == "analytic":
        raise ValueError("TOML input does not support analytic magnetic-field callables; use the Python API for that case")

    geometry = GeometrySpec(
        kind=str(_require(geometry_table, "kind")),
        width=float(_require(geometry_table, "width")),
        height=float(_require(geometry_table, "height")),
        length=float(geometry_table.get("length", 1.0)),
        nx=int(geometry_table.get("nx", 1)),
        ny=int(_require(geometry_table, "ny")),
        nz=int(_require(geometry_table, "nz")),
        radius=None if geometry_table.get("radius") is None else float(geometry_table["radius"]),
        nr=None if geometry_table.get("nr") is None else int(geometry_table["nr"]),
        ntheta=None if geometry_table.get("ntheta") is None else int(geometry_table["ntheta"]),
        wall_thickness=_optional_tuple(geometry_table, "wall_thickness", length=4, cast=float) or (0.0, 0.0, 0.0, 0.0),
        wall_cells=_optional_tuple(geometry_table, "wall_cells", length=4, cast=int) or (0, 0, 0, 0),
        target_ha=None if geometry_table.get("target_ha") is None else float(geometry_table["target_ha"]),
        target_side_layer=None if geometry_table.get("target_side_layer") is None else float(geometry_table["target_side_layer"]),
    )

    magnetic_field = MagneticFieldSpec(
        kind=str(_require(field_table, "kind")),
        value=_optional_tuple(field_table, "value", length=3, cast=float),
        fn=None,
        table_path=None
        if field_table.get("table_path") is None
        else str((input_path.parent / str(field_table["table_path"])).resolve()),
        ramp_start=float(field_table.get("ramp_start", 0.0)),
        ramp_duration=float(field_table.get("ramp_duration", 0.0)),
    )

    time_stepper = TimeStepperConfig(
        dt=float(_require(time_table, "dt")),
        t_final=float(_require(time_table, "t_final")),
        max_steps=int(_require(time_table, "max_steps")),
        outer_iterations=int(time_table.get("outer_iterations", 2)),
        potential_iterations=int(time_table.get("potential_iterations", 400)),
        potential_tolerance=None if time_table.get("potential_tolerance") is None else float(time_table["potential_tolerance"]),
        potential_relaxation=float(time_table.get("potential_relaxation", 1.0)),
        potential_solver=str(time_table.get("potential_solver", "auto")),
        current_reconstruction=str(time_table.get("current_reconstruction", "cell_centered")),
        steady_tolerance=float(time_table.get("steady_tolerance", 1e-8)),
        steady_potential_tolerance=None
        if time_table.get("steady_potential_tolerance") is None
        else float(time_table["steady_potential_tolerance"]),
        relaxation=float(time_table.get("relaxation", 0.35)),
        velocity_update_limit=float(time_table.get("velocity_update_limit", 1e-3)),
        velocity_update_limiter=str(time_table.get("velocity_update_limiter", "global_scale")),
        checkpoint_stride=int(time_table.get("checkpoint_stride", 1)),
    )

    output_dir = output_table.get("directory")
    if output_dir is not None:
        output_dir = str((input_path.parent / str(output_dir)).resolve())

    output = OutputSpec(
        directory=output_dir,
        write_paraview=bool(output_table.get("write_paraview", True)),
        write_csv_profiles=bool(output_table.get("write_csv_profiles", True)),
        write_npz=bool(output_table.get("write_npz", True)),
        write_json_summary=bool(output_table.get("write_json_summary", True)),
        write_plots=bool(output_table.get("write_plots", False)),
        copy_input_file=bool(output_table.get("copy_input_file", True)),
        write_stride=int(output_table.get("write_stride", 1)),
    )

    case = CaseSpec(
        name=str(_require(case_table, "name")),
        geometry=geometry,
        regions=_parse_regions(regions_table),
        magnetic_field=magnetic_field,
        boundary_conditions=_parse_boundaries(boundaries_table),
        time_stepper=time_stepper,
        output=output,
        forcing=float(case_table.get("forcing", 1.0)),
        initial_velocity=float(case_table.get("initial_velocity", 0.0)),
        reference_pressure_gradient=float(case_table.get("reference_pressure_gradient", -1.0)),
        reference_phi_cell=_optional_tuple(case_table, "reference_phi_cell", length=2, cast=int) or (0, 0),
        notes=str(case_table.get("notes", "")),
    )

    logging = LoggingSpec(
        enabled=bool(logging_table.get("enabled", True)),
        banner=bool(logging_table.get("banner", True)),
        print_regions=bool(logging_table.get("print_regions", True)),
        print_boundaries=bool(logging_table.get("print_boundaries", True)),
        print_footer=bool(logging_table.get("print_footer", True)),
        flush=bool(logging_table.get("flush", True)),
        step_stride=int(logging_table.get("step_stride", 1)),
    )

    solve_mode = str(case_table.get("solve_mode", "steady"))
    if solve_mode not in {"steady", "transient"}:
        raise ValueError(f"Unsupported solve_mode {solve_mode!r}")

    return RunConfig(case=case, solve_mode=solve_mode, logging=logging, input_path=input_path)
