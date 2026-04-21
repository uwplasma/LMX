import pytest

from lmx.mesh import generate_bent_pipe_mesh, generate_layered_duct_mesh, generate_pipe_ogrid_mesh, generate_rect_duct_mesh


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


def test_moderate_ha_rect_mesh_clusters_boundary_layers():
    mesh = generate_rect_duct_mesh(width=0.2, height=0.2, ny=32, nz=32, target_ha=20.0, magnetic_axis="z")
    dy = mesh.dy
    dz = mesh.dz
    assert float(dy.min()) < float(dy.max())
    assert float(dz.min()) < float(dz.max())
    assert float(dy.min()) < 0.002
    assert float(dz.min()) < 0.002
