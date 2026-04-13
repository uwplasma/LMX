# Geometry and Mesh Workflows

LMX `1.0` ships three user-facing geometry paths:

- `rect_duct`
  - uniform rectilinear duct meshes for Hartmann and Shercliff studies
- `layered_duct`
  - rectilinear duct meshes with explicit conducting or insulating wall layers
- `pipe_ogrid`
  - mapped O-grid meshes for geometry inspection and post-1.0 fringing-field scaffolds

## Build geometries from Python

```python
from lmx.cases import make_hartmann_case, make_hunt_case
from lmx.mesh import generate_pipe_ogrid_mesh
from lmx.solvers import _build_mesh

hartmann_case = make_hartmann_case(ha=20.0, ny=48, nz=48)
hunt_case = make_hunt_case(ha=40.0, ny=40, nz=32, wall_cells=2)

hartmann_mesh = _build_mesh(hartmann_case)
hunt_mesh = _build_mesh(hunt_case)
pipe_mesh = generate_pipe_ogrid_mesh(radius=0.5, length=2.0, nx=16, nr=18, ntheta=48)
```

## Preview geometries before running

LMX now ships a dedicated preview utility in `lmx/plotting.py` and a complete
example in `examples/geometry_preview_demo.py`.

```bash
python examples/geometry_preview_demo.py --output artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output artifacts/examples/geometry_preview_full
```

That example writes:

- `hartmann_geometry/geometry_preview.png`
- `hunt_geometry/geometry_preview.png`
- `pipe_geometry/geometry_preview.png`
- a matching `PDF` for each figure
- optional post-run overview plots in the same output tree when `--with-post-run` is enabled

The preview figure always contains:

- a cross-sectional material map, so layered walls are visible before the run
- a 3D wireframe preview of the extruded duct or mapped pipe geometry

The default path is intentionally preview-only so it remains a fast preprocessing
step. Use `--with-post-run --post-case hartmann` or `--with-post-run --post-case hunt`
when you want a matching short benchmark solve in the same folder.

## Mesh-resolution guidance

- increase `ny` and `nz` for better Hartmann and Shercliff profile resolution
- use `wall_cells > 0` and nonzero `wall_thickness` for `layered_duct`
- treat `pipe_ogrid` as a geometry/preprocessing and post-1.0 scaffold path until
  `extruded_inductionless` lands

The benchmark-oriented layer-thickness metrics reported by
`lmx.validation.duct_layer_resolution_metrics(...)` are the current way to
quantify whether Hartmann and side layers are resolved on a duct mesh.
