"""Reduced wall-stack electrical models used by validation and design studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .units import wall_conductance_ratio


@dataclass(frozen=True)
class WallLayer:
    """One solid layer in a fluid-facing wall stack.

    ``conductivity`` is electrical conductivity in ``S/m`` and ``thickness`` is
    layer thickness in meters.  ``cells`` is optional mesh metadata used by
    nested wall-layer QA; it does not alter the electrical reduction.
    """

    name: str
    conductivity: float
    thickness: float
    cells: int = 1


def tangential_stack_conductance_ratio(
    layers: Sequence[WallLayer],
    *,
    fluid_conductivity: float,
    length_scale: float,
) -> float:
    """Return the thin-wall tangential ratio for layers conducting in parallel."""

    _validate_layers(layers)
    if fluid_conductivity <= 0.0:
        raise ValueError("fluid_conductivity must be positive")
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    equivalent_surface_conductance = sum(
        float(layer.conductivity) * float(layer.thickness) for layer in layers
    )
    return equivalent_surface_conductance / (float(fluid_conductivity) * float(length_scale))


def normal_stack_leakage_ratio(
    layers: Sequence[WallLayer],
    *,
    fluid_conductivity: float,
    length_scale: float,
) -> float:
    """Return normal leakage ratio for layers in electrical series."""

    _validate_layers(layers)
    if fluid_conductivity <= 0.0:
        raise ValueError("fluid_conductivity must be positive")
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    resistance = 0.0
    for layer in layers:
        if layer.conductivity <= 0.0:
            return 0.0
        resistance += float(layer.thickness) / float(layer.conductivity)
    normal_conductance = 1.0 / resistance
    return normal_conductance * float(length_scale) / float(fluid_conductivity)


def effective_pinhole_conductance_ratio(
    *,
    intact_conductance_ratio: float,
    metal_conductance_ratio: float,
    pinhole_fraction: float,
) -> float:
    """Return ``(1-f_p)c_intact + f_p c_metal`` for smooth degradation sweeps."""

    if not 0.0 <= pinhole_fraction <= 1.0:
        raise ValueError("pinhole_fraction must be between 0 and 1")
    if intact_conductance_ratio < 0.0:
        raise ValueError("intact_conductance_ratio must be non-negative")
    if metal_conductance_ratio < 0.0:
        raise ValueError("metal_conductance_ratio must be non-negative")
    return (1.0 - float(pinhole_fraction)) * float(intact_conductance_ratio) + float(
        pinhole_fraction
    ) * float(metal_conductance_ratio)


def equivalent_single_layer(
    layers: Sequence[WallLayer],
    *,
    name: str = "equivalent_wall",
) -> WallLayer:
    """Return a single layer with the same tangential surface conductance."""

    _validate_layers(layers)
    total_thickness = sum(float(layer.thickness) for layer in layers)
    equivalent_conductivity = (
        sum(float(layer.conductivity) * float(layer.thickness) for layer in layers) / total_thickness
    )
    return WallLayer(
        name=name,
        conductivity=equivalent_conductivity,
        thickness=total_thickness,
        cells=sum(max(int(layer.cells), 0) for layer in layers),
    )


def nested_wall_layer_resolution_summary(
    layers: Sequence[WallLayer],
    *,
    minimum_cells_per_layer: int = 3,
) -> dict[str, object]:
    """Return lightweight QA metrics for nested wall-layer meshes."""

    _validate_layers(layers)
    if minimum_cells_per_layer < 1:
        raise ValueError("minimum_cells_per_layer must be positive")
    rows = []
    for layer in layers:
        cells = int(layer.cells)
        cell_width = float(layer.thickness) / max(cells, 1)
        rows.append(
            {
                "name": layer.name,
                "thickness": float(layer.thickness),
                "conductivity": float(layer.conductivity),
                "cells": cells,
                "cell_width": cell_width,
                "cell_count_pass": cells >= minimum_cells_per_layer,
            }
        )
    total_cells = sum(int(layer.cells) for layer in layers)
    min_cells = min(int(layer.cells) for layer in layers)
    return {
        "layer_count": len(layers),
        "total_thickness": sum(float(layer.thickness) for layer in layers),
        "total_cells": total_cells,
        "minimum_cells_per_layer": min_cells,
        "minimum_required_cells_per_layer": int(minimum_cells_per_layer),
        "resolution_pass": bool(min_cells >= minimum_cells_per_layer),
        "layers": rows,
    }


def wall_layer_from_conductance_ratio(
    *,
    name: str,
    conductance_ratio: float,
    thickness: float,
    fluid_conductivity: float,
    length_scale: float,
    cells: int = 1,
) -> WallLayer:
    """Construct a wall layer from target thin-wall conductance ratio ``c``."""

    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if conductance_ratio < 0.0:
        raise ValueError("conductance_ratio must be non-negative")
    conductivity = (
        float(conductance_ratio) * float(fluid_conductivity) * float(length_scale) / float(thickness)
    )
    layer = WallLayer(name=name, conductivity=conductivity, thickness=thickness, cells=cells)
    recovered = wall_conductance_ratio(
        wall_conductivity=layer.conductivity,
        wall_thickness=layer.thickness,
        fluid_conductivity=fluid_conductivity,
        length_scale=length_scale,
    )
    if not math.isclose(recovered, conductance_ratio, rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError("failed to construct requested wall conductance ratio")
    return layer


def _validate_layers(layers: Sequence[WallLayer]) -> None:
    if not layers:
        raise ValueError("at least one wall layer is required")
    for layer in layers:
        if layer.thickness <= 0.0:
            raise ValueError(f"wall layer {layer.name!r} has non-positive thickness")
        if layer.conductivity < 0.0:
            raise ValueError(f"wall layer {layer.name!r} has negative conductivity")
        if layer.cells < 0:
            raise ValueError(f"wall layer {layer.name!r} has negative cells")
