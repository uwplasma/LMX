import pytest

from lmx.cases import _wall_conductivity_from_conductance_ratio


pytestmark = pytest.mark.unit


def test_wall_conductivity_from_conductance_ratio_rejects_nonpositive_thickness():
    with pytest.raises(ValueError, match="wall_thickness"):
        _wall_conductivity_from_conductance_ratio(
            wall_conductance_ratio=1.0,
            fluid_conductivity=1.0,
            wall_thickness=0.0,
            hartmann_half_spacing=1.0,
        )


def test_wall_conductivity_from_conductance_ratio_rejects_nonpositive_half_spacing():
    with pytest.raises(ValueError, match="hartmann_half_spacing"):
        _wall_conductivity_from_conductance_ratio(
            wall_conductance_ratio=1.0,
            fluid_conductivity=1.0,
            wall_thickness=1.0,
            hartmann_half_spacing=0.0,
        )
