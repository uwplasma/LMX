import jax.numpy as jnp
import pytest

from lmx.mesh import (
    centerline_pipe_mesh_quality_metrics,
    generate_bent_pipe_mesh,
    generate_centerline_pipe_mesh,
    generate_layered_duct_mesh,
    generate_layered_duct_mesh_from_fluid_faces,
    generate_multilayer_duct_mesh,
    generate_pipe_ogrid_mesh,
    generate_rect_duct_mesh,
    generate_rect_duct_mesh_from_faces,
)
from lmx.wall_models import WallLayer


pytestmark = pytest.mark.unit


def test_rect_duct_mesh_shape():
    mesh = generate_rect_duct_mesh(width=2.0, height=1.0, nx=2, ny=8, nz=6)
    assert mesh.nx == 2
    assert mesh.ny == 8
    assert mesh.nz == 6


def test_layered_duct_mesh_has_solid_cells():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=1.0,
        ny=8,
        nz=8,
        wall_thickness=(0.0, 0.0, 0.1, 0.1),
        wall_cells=(0, 0, 2, 2),
        target_ha=100.0,
    )
    assert mesh.fluid_mask is not None
    assert (~mesh.fluid_mask).sum() > 0


def test_pipe_ogrid_points_exist():
    mesh = generate_pipe_ogrid_mesh(radius=1.0, nx=2, nr=4, ntheta=8)
    assert mesh.point_coordinates is not None
    assert mesh.point_coordinates.shape[-1] == 3


def test_bent_pipe_points_follow_curved_centerline():
    mesh = generate_bent_pipe_mesh(tube_radius=0.2, bend_radius=1.0, bend_angle=1.0, nx=4, nr=4, ntheta=8)
    assert mesh.point_coordinates is not None
    start = mesh.point_coordinates[0, 0, 0]
    end = mesh.point_coordinates[-1, 0, 0]
    assert float(start[0]) == pytest.approx(0.0)
    assert float(start[1]) == pytest.approx(0.0)
    assert float(end[0]) > 0.0
    assert float(end[1]) > 0.0


def test_centerline_pipe_mesh_follows_arbitrary_route():
    centerline = {
        "x": jnp.asarray([0.0, 0.6, 0.8, 0.8]),
        "y": jnp.asarray([0.0, 0.0, 0.4, 0.9]),
        "z": jnp.asarray([0.0, 0.0, 0.1, 0.1]),
    }

    mesh = generate_centerline_pipe_mesh(centerline, tube_radius=0.08, nx=8, nr=5, ntheta=12)
    metrics = centerline_pipe_mesh_quality_metrics(mesh)

    assert mesh.geometry == "centerline_pipe"
    assert mesh.point_coordinates is not None
    assert mesh.point_coordinates.shape == (9, 6, 13, 3)
    assert metrics["validation_pass"] is True
    assert metrics["cell_count"] == 8 * 5 * 12
    assert metrics["max_radius_error"] < 1.0e-6
    assert metrics["max_theta_closure_error"] < 1.0e-6


def test_centerline_pipe_mesh_rejects_invalid_centerline():
    with pytest.raises(ValueError, match="at least three"):
        generate_centerline_pipe_mesh(
            {"x": jnp.asarray([0.0, 1.0]), "y": jnp.asarray([0.0, 0.0]), "z": jnp.asarray([0.0, 0.0])},
            tube_radius=0.1,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_centerline_pipe_mesh(
            {
                "x": jnp.asarray([0.0, 1.0, 2.0]),
                "y": jnp.asarray([0.0, 0.0, 0.0]),
                "z": jnp.asarray([0.0, 0.0, 0.0]),
                "station": jnp.asarray([0.0, 0.0, 1.0]),
            },
            tube_radius=0.1,
        )


def test_moderate_ha_rect_mesh_clusters_boundary_layers():
    mesh = generate_rect_duct_mesh(width=0.2, height=0.2, ny=32, nz=32, target_ha=20.0, magnetic_axis="z")
    dy = mesh.dy
    dz = mesh.dz
    uniform_spacing = 0.2 / 32.0
    assert float(dy.min()) < float(dy.max())
    assert float(dz.min()) < float(dz.max())
    assert float(dy.min()) < 0.4 * uniform_spacing
    assert float(dz.min()) < 0.4 * uniform_spacing


def test_generate_rect_duct_mesh_from_faces_preserves_explicit_faces():
    mesh = generate_rect_duct_mesh_from_faces(
        y_faces=jnp.asarray([-0.1, -0.05, 0.0, 0.1]),
        z_faces=jnp.asarray([-0.2, 0.0, 0.2]),
        length=3.0,
        nx=3,
    )

    assert mesh.geometry == "rect_duct"
    assert mesh.nx == 3
    assert mesh.ny == 3
    assert mesh.nz == 2
    assert mesh.y_centers.tolist() == pytest.approx([-0.075, -0.025, 0.05])
    assert mesh.z_centers.tolist() == pytest.approx([-0.1, 0.1])


def test_generate_rect_duct_mesh_from_faces_rejects_invalid_faces():
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_rect_duct_mesh_from_faces(y_faces=jnp.asarray([0.0, 0.0]), z_faces=jnp.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        generate_rect_duct_mesh_from_faces(y_faces=jnp.ones((2, 2)), z_faces=jnp.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="at least two"):
        generate_rect_duct_mesh_from_faces(y_faces=jnp.asarray([0.0]), z_faces=jnp.asarray([0.0, 1.0]))


def test_generate_layered_duct_mesh_from_fluid_faces_adds_wall_regions():
    mesh = generate_layered_duct_mesh_from_fluid_faces(
        fluid_y_faces=jnp.asarray([-0.1, 0.0, 0.1]),
        fluid_z_faces=jnp.asarray([-0.2, 0.0, 0.2]),
        width=0.2,
        height=0.4,
        wall_thickness=(0.02, 0.04, 0.03, 0.05),
        wall_cells=(1, 2, 1, 1),
    )

    assert mesh.geometry == "layered_duct"
    assert mesh.y_faces.tolist() == pytest.approx([-0.12, -0.1, 0.0, 0.1, 0.12, 0.14])
    assert mesh.z_faces.tolist() == pytest.approx([-0.23, -0.2, 0.0, 0.2, 0.25])
    assert mesh.fluid_mask.shape == mesh.yz_shape
    assert bool(mesh.fluid_mask[1, 1])
    assert not bool(mesh.fluid_mask[0, 1])
    assert not bool(mesh.fluid_mask[1, 0])


def test_generate_multilayer_duct_mesh_aligns_interfaces_and_sigma():
    wall_layers = {
        side: (
            WallLayer("aln", conductivity=1.0e-8, thickness=0.01, cells=2),
            WallLayer("metal", conductivity=1.0e6, thickness=0.02, cells=2),
        )
        for side in ("left", "right", "bottom", "top")
    }

    mesh = generate_multilayer_duct_mesh(
        width=1.0,
        height=1.0,
        ny=8,
        nz=8,
        wall_layers=wall_layers,
        fluid_conductivity=2.0,
    )

    assert mesh.fluid_mask is not None
    assert mesh.sigma is not None
    assert mesh.region_ids is not None
    assert mesh.region_names[0] == "fluid"
    assert "left:aln" in mesh.region_names
    assert float(mesh.sigma[mesh.region_ids == 0][0]) == pytest.approx(2.0)
    assert float(mesh.sigma[mesh.region_ids == mesh.region_names.index("left:aln")][0]) == pytest.approx(1.0e-8)
    assert float(mesh.sigma[mesh.region_ids == mesh.region_names.index("left:metal")][0]) == pytest.approx(1.0e6)
    y_faces = [float(value) for value in mesh.y_faces]
    z_faces = [float(value) for value in mesh.z_faces]
    assert any(abs(value + 0.5) < 1.0e-12 for value in y_faces)
    assert any(abs(value + 0.51) < 1.0e-12 for value in y_faces)
    assert any(abs(value - 0.51) < 1.0e-12 for value in y_faces)
    assert any(abs(value + 0.51) < 1.0e-12 for value in z_faces)
    assert any(abs(value - 0.51) < 1.0e-12 for value in z_faces)
