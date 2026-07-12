import math

import pytest

from lmx import (
    WallLayer,
    dynamic_to_kinematic_viscosity,
    effective_pinhole_conductance_ratio,
    equivalent_single_layer,
    hartmann_number,
    interaction_parameter,
    kinematic_to_dynamic_viscosity,
    magnetic_field_from_hartmann,
    magnetic_reynolds_number,
    nested_wall_layer_resolution_summary,
    normal_stack_leakage_ratio,
    reynolds_number,
    tangential_stack_conductance_ratio,
    wall_conductance_ratio,
)
from lmx.units import normal_leakage_ratio
from lmx.wall_models import wall_layer_from_conductance_ratio


pytestmark = pytest.mark.unit


def test_viscosity_conversion_round_trip_and_hartmann_convention():
    rho = 500.0
    mu = 2.5e-3
    nu = dynamic_to_kinematic_viscosity(mu, rho)

    assert nu == pytest.approx(5.0e-6)
    assert kinematic_to_dynamic_viscosity(nu, rho) == pytest.approx(mu)

    b = magnetic_field_from_hartmann(
        hartmann=20.0,
        length_scale=0.1,
        conductivity=2.0e6,
        density=rho,
        kinematic_viscosity=nu,
    )
    recovered = hartmann_number(
        magnetic_field=b,
        length_scale=0.1,
        conductivity=2.0e6,
        density=rho,
        kinematic_viscosity=nu,
    )
    assert recovered == pytest.approx(20.0)


def test_nondimensional_numbers_are_hand_calculable():
    assert reynolds_number(
        velocity=0.2, length_scale=0.05, kinematic_viscosity=1.0e-6
    ) == pytest.approx(10000.0)
    assert interaction_parameter(
        magnetic_field=2.0,
        length_scale=0.1,
        conductivity=1.0e6,
        density=1000.0,
        velocity=0.5,
    ) == pytest.approx(800.0)
    assert magnetic_reynolds_number(
        velocity=0.5, length_scale=0.1, conductivity=1.0e6
    ) == pytest.approx(4.0e-7 * math.pi * 5.0e4)


def test_thin_wall_and_normal_leakage_ratios_are_distinct():
    c_parallel = wall_conductance_ratio(
        wall_conductivity=5.0,
        wall_thickness=1.0e-3,
        fluid_conductivity=1.0,
        length_scale=0.1,
    )
    g_perp = normal_leakage_ratio(
        coating_conductivity=5.0,
        coating_thickness=1.0e-3,
        fluid_conductivity=1.0,
        length_scale=0.1,
    )

    assert c_parallel == pytest.approx(0.05)
    assert g_perp == pytest.approx(500.0)
    assert g_perp != c_parallel


def test_nested_wall_stack_reduces_to_equivalent_single_layer_for_tangential_conduction():
    layers = (
        WallLayer("aln", conductivity=1.0e-8, thickness=2.0e-4, cells=4),
        WallLayer("metal", conductivity=1.0e6, thickness=1.0e-3, cells=8),
    )
    stack_c = tangential_stack_conductance_ratio(
        layers, fluid_conductivity=1.0e6, length_scale=0.01
    )
    equivalent = equivalent_single_layer(layers)
    equivalent_c = wall_conductance_ratio(
        wall_conductivity=equivalent.conductivity,
        wall_thickness=equivalent.thickness,
        fluid_conductivity=1.0e6,
        length_scale=0.01,
    )

    assert stack_c == pytest.approx(equivalent_c)
    assert equivalent.cells == 12


def test_normal_stack_leakage_is_limited_by_insulating_layer():
    layers = (
        WallLayer("aln", conductivity=1.0e-8, thickness=2.0e-4, cells=4),
        WallLayer("metal", conductivity=1.0e6, thickness=1.0e-3, cells=8),
    )
    leakage = normal_stack_leakage_ratio(
        layers, fluid_conductivity=1.0e6, length_scale=0.01
    )
    single_aln = normal_leakage_ratio(
        coating_conductivity=1.0e-8,
        coating_thickness=2.0e-4,
        fluid_conductivity=1.0e6,
        length_scale=0.01,
    )

    assert leakage == pytest.approx(single_aln, rel=1.0e-8)


def test_effective_pinhole_model_interpolates_between_intact_and_metal_limits():
    assert effective_pinhole_conductance_ratio(
        intact_conductance_ratio=1.0e-8,
        metal_conductance_ratio=10.0,
        pinhole_fraction=0.0,
    ) == pytest.approx(1.0e-8)
    assert effective_pinhole_conductance_ratio(
        intact_conductance_ratio=1.0e-8,
        metal_conductance_ratio=10.0,
        pinhole_fraction=1.0,
    ) == pytest.approx(10.0)
    assert effective_pinhole_conductance_ratio(
        intact_conductance_ratio=1.0e-8,
        metal_conductance_ratio=10.0,
        pinhole_fraction=0.25,
    ) == pytest.approx(2.5000000075)


def test_nested_wall_layer_resolution_summary_marks_underresolved_layers():
    summary = nested_wall_layer_resolution_summary(
        (
            WallLayer("aln", conductivity=1.0e-8, thickness=2.0e-4, cells=2),
            WallLayer("metal", conductivity=1.0e6, thickness=1.0e-3, cells=8),
        ),
        minimum_cells_per_layer=3,
    )

    assert summary["resolution_pass"] is False
    assert summary["minimum_cells_per_layer"] == 2


def test_wall_layer_from_conductance_ratio_recovers_requested_ratio():
    layer = wall_layer_from_conductance_ratio(
        name="hunt_wall",
        conductance_ratio=0.05,
        thickness=0.001,
        fluid_conductivity=1.0,
        length_scale=0.1,
        cells=6,
    )

    assert layer.conductivity == pytest.approx(5.0)
    assert layer.cells == 6


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (
            dynamic_to_kinematic_viscosity,
            {"dynamic_viscosity": 1.0, "density": 0.0},
            "density",
        ),
        (
            dynamic_to_kinematic_viscosity,
            {"dynamic_viscosity": -1.0, "density": 1.0},
            "dynamic_viscosity",
        ),
        (
            kinematic_to_dynamic_viscosity,
            {"kinematic_viscosity": 1.0, "density": 0.0},
            "density",
        ),
        (
            kinematic_to_dynamic_viscosity,
            {"kinematic_viscosity": -1.0, "density": 1.0},
            "kinematic_viscosity",
        ),
        (
            reynolds_number,
            {"velocity": 1.0, "length_scale": 0.0, "kinematic_viscosity": 1.0},
            "length_scale",
        ),
        (
            reynolds_number,
            {"velocity": 1.0, "length_scale": 1.0, "kinematic_viscosity": 0.0},
            "kinematic_viscosity",
        ),
        (
            interaction_parameter,
            {
                "magnetic_field": 1.0,
                "length_scale": 1.0,
                "conductivity": 1.0,
                "density": 1.0,
                "velocity": 0.0,
            },
            "velocity",
        ),
        (
            magnetic_reynolds_number,
            {
                "velocity": 1.0,
                "length_scale": 1.0,
                "conductivity": 1.0,
                "magnetic_permeability": 0.0,
            },
            "magnetic_permeability",
        ),
        (
            wall_conductance_ratio,
            {
                "wall_conductivity": -1.0,
                "wall_thickness": 1.0,
                "fluid_conductivity": 1.0,
                "length_scale": 1.0,
            },
            "wall_conductivity",
        ),
        (
            normal_leakage_ratio,
            {
                "coating_conductivity": -1.0,
                "coating_thickness": 1.0,
                "fluid_conductivity": 1.0,
                "length_scale": 1.0,
            },
            "coating_conductivity",
        ),
    ],
)
def test_unit_helpers_reject_invalid_physical_inputs(function, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(**kwargs)


@pytest.mark.parametrize(
    "bad_name", ["length_scale", "conductivity", "density", "kinematic_viscosity"]
)
def test_hartmann_helpers_reject_nonpositive_scales(bad_name):
    kwargs = dict(
        length_scale=1.0, conductivity=1.0, density=1.0, kinematic_viscosity=1.0
    )
    kwargs[bad_name] = 0.0
    with pytest.raises(ValueError, match=bad_name):
        hartmann_number(magnetic_field=1.0, **kwargs)
    with pytest.raises(ValueError, match=bad_name):
        magnetic_field_from_hartmann(hartmann=1.0, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "intact_conductance_ratio": 0.0,
            "metal_conductance_ratio": 1.0,
            "pinhole_fraction": -0.1,
        },
        {
            "intact_conductance_ratio": -1.0,
            "metal_conductance_ratio": 1.0,
            "pinhole_fraction": 0.1,
        },
        {
            "intact_conductance_ratio": 0.0,
            "metal_conductance_ratio": -1.0,
            "pinhole_fraction": 0.1,
        },
    ],
)
def test_pinhole_model_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        effective_pinhole_conductance_ratio(**kwargs)


@pytest.mark.parametrize(
    "layers",
    [
        (),
        (WallLayer("bad", 1.0, 0.0),),
        (WallLayer("bad", -1.0, 1.0),),
        (WallLayer("bad", 1.0, 1.0, cells=-1),),
    ],
)
def test_wall_stack_rejects_invalid_layers(layers):
    with pytest.raises(ValueError):
        equivalent_single_layer(layers)


def test_wall_stack_rejects_invalid_scales_and_handles_perfect_insulator():
    layers = (WallLayer("wall", 1.0, 1.0),)
    for function in (tangential_stack_conductance_ratio, normal_stack_leakage_ratio):
        with pytest.raises(ValueError, match="fluid_conductivity"):
            function(layers, fluid_conductivity=0.0, length_scale=1.0)
        with pytest.raises(ValueError, match="length_scale"):
            function(layers, fluid_conductivity=1.0, length_scale=0.0)
    assert (
        normal_stack_leakage_ratio(
            (WallLayer("insulator", 0.0, 1.0),),
            fluid_conductivity=1.0,
            length_scale=1.0,
        )
        == 0.0
    )


def test_wall_resolution_and_conductance_constructor_validate_controls(monkeypatch):
    layers = (WallLayer("wall", 1.0, 1.0),)
    with pytest.raises(ValueError, match="minimum_cells"):
        nested_wall_layer_resolution_summary(layers, minimum_cells_per_layer=0)
    with pytest.raises(ValueError, match="thickness"):
        wall_layer_from_conductance_ratio(
            name="wall",
            conductance_ratio=1.0,
            thickness=0.0,
            fluid_conductivity=1.0,
            length_scale=1.0,
        )
    with pytest.raises(ValueError, match="conductance_ratio"):
        wall_layer_from_conductance_ratio(
            name="wall",
            conductance_ratio=-1.0,
            thickness=1.0,
            fluid_conductivity=1.0,
            length_scale=1.0,
        )

    monkeypatch.setattr("lmx.wall_models.wall_conductance_ratio", lambda **kwargs: -1.0)
    with pytest.raises(RuntimeError, match="failed to construct"):
        wall_layer_from_conductance_ratio(
            name="wall",
            conductance_ratio=1.0,
            thickness=1.0,
            fluid_conductivity=1.0,
            length_scale=1.0,
        )
