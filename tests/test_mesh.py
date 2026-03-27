from lmx.mesh import generate_layered_duct_mesh, generate_pipe_ogrid_mesh, generate_rect_duct_mesh


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
