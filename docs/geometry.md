# Geometry and Mesh Workflows

LMX currently exposes three geometry descriptions through the public case and
mesh APIs:

- `rect_duct`
  - a uniform structured cross-section extruded in the streamwise direction
  - used for Hartmann- and Shercliff-type fully developed studies
- `layered_duct`
  - the same rectilinear fluid core with explicit wall layers around the fluid
  - used for conducting-wall and mixed-wall benchmark studies
- `pipe_ogrid`
  - a mapped structured O-grid around a circular core
  - currently used for geometry preview, postprocessing, and the first
    fringing-field `extruded_inductionless` pipe slice
- `bent_pipe`
  - a centerline-following mapped pipe around a constant-radius bend
  - currently used for preprocessing, geometry preview, and the low-De
    curved-pipe inductionless baseline

## Geometry objects in the source tree

The geometry configuration enters the code through:

- `lmx/specs.py`
  - dataclass definitions for `GeometrySpec`
- `lmx/cases.py`
  - benchmark-oriented case constructors
- `lmx/mesh.py`
  - structured mesh generation and mapped pipe O-grid generation
- `lmx/solvers.py`
  - `_build_mesh(...)` dispatch from case specifications to concrete mesh arrays
- `lmx/fringing.py`
  - explicit 3D fringing problems for rectangular ducts, layered ducts, and
    mapped pipes

## Build geometries from Python

```python
from dataclasses import replace

from lmx.cases import make_hartmann_case, make_hunt_case
from lmx.mesh import generate_bent_pipe_mesh, generate_pipe_ogrid_mesh
from lmx.solvers import _build_mesh

hartmann_case = make_hartmann_case(ha=20.0, ny=48, nz=48)
hunt_case = make_hunt_case(ha=40.0, ny=40, nz=32, wall_cells=2)

custom_hartmann = replace(
    hartmann_case,
    geometry=replace(hartmann_case.geometry, width=2.5, height=1.5, ny=64, nz=48),
)

hartmann_mesh = _build_mesh(custom_hartmann)
hunt_mesh = _build_mesh(hunt_case)
pipe_mesh = generate_pipe_ogrid_mesh(radius=0.5, length=2.0, nx=16, nr=18, ntheta=48)
bent_pipe_mesh = generate_bent_pipe_mesh(tube_radius=0.25, bend_radius=1.0, bend_angle=1.2, nx=18, nr=16, ntheta=48)
```

That is the intended route for variable-geometry studies. The benchmark helpers
provide a stable starting point, and `dataclasses.replace(...)` makes the
resulting research driver explicit.

## Variable magnetic fields from Python

Analytic spatially varying magnetic fields are currently configured from Python,
not from TOML. The public examples for that workflow are
`examples/variable_field_geometry_demo.py` and
`examples/variable_field_validation.py`.

```python
from dataclasses import replace

from lmx.cases import make_hartmann_case
from lmx.specs import MagneticFieldSpec


def analytic_field(y, z):
    return (0.0, 0.0, 18.0 * (1.0 + 0.15 * y - 0.05 * z))


case = make_hartmann_case(ha=18.0, ny=48, nz=40)
case = replace(
    case,
    magnetic_field=MagneticFieldSpec(kind="analytic", fn=analytic_field, ramp_start=0.0, ramp_duration=0.0),
)
```

This route is the current way to explore:

- fringing profiles approximated by smooth streamwise stations
- parameterized field maps in inverse-design studies
- custom benchmark variants that are easier to define as Python callables than
  as table files
- divergence checks on analytic fields before they are used in a solve

## Preview geometries before running

LMX ships a dedicated preview utility in `lmx/plotting.py` and complete
drivers in:

- `examples/geometry_preview_demo.py`
- `examples/variable_field_geometry_demo.py`
- `examples/geometry_panel_demo.py`
- `examples/bent_pipe_preview.py`
- `examples/bent_pipe_inductionless_demo.py`
- `examples/readme_showcase_demo.py`

```bash
python examples/geometry_preview_demo.py --output artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output artifacts/examples/geometry_preview_full
python examples/variable_field_geometry_demo.py --output artifacts/examples/variable_field_geometry
python examples/geometry_panel_demo.py --output artifacts/examples/geometry_panel
python examples/bent_pipe_preview.py
python examples/bent_pipe_inductionless_demo.py
python examples/readme_showcase_demo.py --output docs/_static/generated
# optional Hartmann alternative for wall-layer startup media
python examples/readme_showcase_demo.py --output docs/_static/generated --movie-case-kind hartmann
python examples/readme_showcase_demo.py --output docs/_static/generated --skip-geometry --movie-view 2d
python examples/readme_showcase_demo.py --output docs/_static/generated --skip-geometry --movie-view 3d
```

Those drivers write:

- cross-sectional material maps
- 3D wireframe previews of the extruded mesh
- centerline-following previews for bent pipes
- `PNG` and `PDF` outputs
- optional short benchmark solves and post-run overview plots in the same output tree
  - README-ready media, including bounded 2D/3D startup GIFs
  - bent-pipe geometry-plus-solution panels for the low-De curved-pipe lane

The preview path is intentionally cheap so it can be used as a preprocessing
step before longer studies.

Current geometry panel:

![LMX geometry panel](_static/generated/geometry_gallery.png)

## Mesh-resolution guidance

- increase `ny` and `nz` to resolve Hartmann and Shercliff boundary layers
- use explicit `wall_cells` and nonzero `wall_thickness` for `layered_duct`
- use `pipe_ogrid` for preprocessing, visualization, and the current first
  fringing `extruded_inductionless` pipe slice
- use `bent_pipe` for curved-centerline geometry QA and for the current low-De
  inductionless baseline before the full Dean-vortex solver lane is added
- use the metrics from `lmx/validation.py`, especially
  `duct_layer_resolution_metrics(...)`, to quantify whether the side and
  Hartmann layers are resolved enough for a benchmark-quality run

Current bent-pipe baseline:

![LMX bent-pipe inductionless baseline](_static/generated/bent_pipe_overview.png)
