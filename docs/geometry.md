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
- `examples/wham_blanket_geometry_preview.py`
- `examples/wham_blanket_mesh_demo.py`
- `examples/wham_blanket_field_on_mesh_demo.py`
- `examples/wham_blanket_current_closure_demo.py`
- `examples/readme_showcase_demo.py`

```bash
python examples/geometry_preview_demo.py --output artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output artifacts/examples/geometry_preview_full
python examples/variable_field_geometry_demo.py --output artifacts/examples/variable_field_geometry
python examples/geometry_panel_demo.py --output artifacts/examples/geometry_panel
python examples/bent_pipe_preview.py
python examples/bent_pipe_inductionless_demo.py
python examples/wham_blanket_geometry_preview.py
python examples/wham_blanket_mesh_demo.py
python examples/wham_blanket_field_on_mesh_demo.py
python examples/wham_blanket_current_closure_demo.py
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
- a circular WHAM blanket pipe route around the central-cell clearance envelope
- a mapped WHAM blanket pipe mesh and local magnetic-field projection QA panel
- `PNG` and `PDF` outputs
- optional short benchmark solves and post-run overview plots in the same output tree
  - README-ready media, including bounded 2D/3D startup GIFs
  - bent-pipe geometry-plus-solution panels for the low-De curved-pipe lane

The preview path is intentionally cheap so it can be used as a preprocessing
step before longer studies.

Current geometry panel:

![LMX geometry panel](_static/generated/geometry_gallery.png)

## Reference-Slice Meshes

External-code parity work sometimes needs LMX to use the exact cross-section
coordinates exported by a processed slice rather than a generated clustered
mesh. `generate_rect_duct_mesh_from_faces(...)` builds a rectangular mesh from
explicit `y` and `z` faces, and
`generate_layered_duct_mesh_from_fluid_faces(...)` wraps explicit fluid-region
faces with conducting or insulating wall cells. For processed slice CSV files,
`processed_slice_point_mesh(...)` converts the unique `Points:1` and
`Points:2` coordinates into a rectangular reference mesh for A/B diagnostics.
This path is intended for validation and mesh-sensitivity studies; production
examples should still use the case builders unless exact external-grid
reproduction is the goal. The standalone
`examples/reference_slice_mesh_diagnostic.py` driver writes a side-by-side
mesh panel and layer-count JSON for this workflow.

Solver calls can also take a validated mesh override:

```python
from lmx import generate_layered_duct_mesh_from_fluid_faces, solve_steady

mesh = generate_layered_duct_mesh_from_fluid_faces(...)
solution = solve_steady(case, mesh=mesh)
```

Use this path for A/B validation against a known external grid. It bypasses the
case's generated `ny`/`nz` spacing but keeps the case physics, material fields,
boundary-condition interpretation, and solver controls. Highly clustered
external slice coordinates can make the linear systems much more expensive, so
promote a reference-grid run only after checking both accuracy and runtime.
`duct_mesh_quality_metrics(...)` records min/max spacing, aspect-ratio proxies,
and a diffusion-conditioning proxy so validation summaries can distinguish
well-resolved meshes from over-clustered meshes that are expensive for the
linear solves.

## Mesh-resolution guidance

- increase `ny` and `nz` to resolve Hartmann and Shercliff boundary layers
- use explicit `wall_cells` and nonzero `wall_thickness` for `layered_duct`
- use `pipe_ogrid` for preprocessing, visualization, and the current first
  fringing `extruded_inductionless` pipe slice
- use `bent_pipe` for curved-centerline geometry QA and for the current low-De
  inductionless baseline before the full Dean-vortex solver lane is added
- use the metrics from `lmx/validation.py`, especially
  `duct_layer_resolution_metrics(...)`, to quantify side/Hartmann layer
  thicknesses and cell counts
- pair layer-count gates with `duct_mesh_quality_metrics(...)` before copying
  external slice spacing into production validation runs
- use `duct_layer_resolution_gate(...)` for publication-facing runs; the
  default gate requires at least eight cells across the Hartmann layer and six
  cells across the side layer before a straight-duct result should be promoted
  as benchmark-quality
- inspect `hartmann_layer_cell_ratio`, `side_layer_cell_ratio`, and
  `minimum_mesh_refinement_factor` when a high-`Ha` comparison fails; ratios
  below one indicate an under-resolved layer rather than a solver-physics
  parity claim

Current bent-pipe baseline:

![LMX bent-pipe inductionless baseline](_static/generated/bent_pipe_overview.png)

The bent-pipe summary reports low-De straight-pipe equivalence plus
Dean/curvature observables (`secondary_flow_rms_ratio`,
`secondary_flow_peak_ratio`, signed-radius velocity-centroid shift, and
inner/outer velocity ratio). These observables are present so the same example
can become the higher-inertia Dean-vortex validation once a curved-duct
reference dataset is digitized or generated. The current local and global
current closure are clean (`max_charge_balance_residual ≈ 2.16e-12`,
`max_wall_current_leakage = 0`, and `net_boundary_current_residual = 0`) after
the conservative mapped-pipe potential solve was corrected to cancel
`div(sigma u×B)`.

## WHAM blanket pipe route preview

`examples/wham_blanket_geometry_preview.py` is the geometry-review entry point
for a liquid-metal blanket concept around WHAM. It uses the public
`WhamBlanketLoop`, `build_wham_blanket_centerline(...)`,
`tube_surface_from_centerline(...)`, and
`write_wham_blanket_geometry_preview(...)` helpers. The first route is a
circular pipe in the mirror midplane: it enters from negative `x`, bends around
the central-cell clearance envelope, and returns on the opposite side. This is
not yet a solver mesh or validation claim; it records the route, path length,
pipe radius, coil separation, and tube-to-cell clearance before the mapped-pipe
simulation is built.

![LMX WHAM blanket pipe geometry preview](_static/generated/wham_blanket_geometry_preview.png)

## WHAM blanket reduced-flow preview

`examples/wham_blanket_flow_demo.py` is the first flow-facing artifact for the
approved route. It uses the same centerline and a fixed-flow-rate engineering
model with PbLi-like properties. The sampled WHAM field enters through the
local transverse field `B_\perp`, and the pressure budget is

```text
Δp = ∫ [ f_D ρ U²/(2D) + C_m σ U B_\perp² + K_b ρ U² κ/(2∫κ ds) ] ds .
```

The current reference run uses `U = 0.20 m/s`, `R = 0.12 m`, and an explicit
high-field design multiplier on the parsed WHAM coil field. It reports
`Δp ≈ 26.5 kPa`, `Re ≈ 2.48e5`, peak `Ha ≈ 904`, and a pressure budget
dominated by the MHD term. The cross-sections are local Hartmann-layer
approximations at fixed flow rate.

The same example now also runs a centerline pressure-velocity transient with a
turbulent Darcy-friction closure, local `σ U B_\perp^2` MHD drag, bend losses,
and an incompressibility projection along the pipe. The retained movie and
diagnostic panel run to `t = 15 s`, beyond filling/startup, and settle to
`U_mean ≈ 0.200 m/s` with `Δp ≈ 26.4 kPa`. This is the first executable
curved-route pressure/velocity gate. It does not yet resolve 3D secondary
flows, cross-section turbulence, heat transfer, or induced magnetic field.
For manuscript planning, the same example also writes a pressure-drop sweep at
fixed flow rate; the current field multipliers `4`, `6`, `8`, and `10` produce
terminal pressure drops of `≈ 6.7`, `15.0`, `26.5`, and `41.4 kPa`, with the
MHD term following the expected `B_\perp^2` trend.

![LMX WHAM blanket reduced-flow pressure and steady sections](_static/generated/wham_blanket_flow.png)

![LMX WHAM blanket transient pressure-velocity solve](_static/generated/wham_blanket_transient_flow.png)

![LMX WHAM blanket pressure sweep](_static/generated/wham_blanket_pressure_sweep.png)

![LMX WHAM blanket reduced-flow movie](_static/generated/wham_blanket_flow.gif)

## WHAM blanket mapped-pipe mesh

`examples/wham_blanket_mesh_demo.py` promotes the approved geometry from a
surface route into a mapped circular-pipe O-grid. The low-level reusable API is
`generate_centerline_pipe_mesh(...)`, with QA from
`centerline_pipe_mesh_quality_metrics(...)` and a mesh panel from
`write_centerline_pipe_mesh_preview(...)`. The example also writes
`wham_blanket_centerline_pipe_mesh.vtu` in the artifact directory for ParaView
inspection.

The retained preview mesh has `65` stations, `18 × 48` cross-section cells,
`55,296` volume cells, nearly uniform `Δs ≈ 0.114 m`, and roundoff-level radius
and periodic-closure errors. This closes the geometry handoff needed before
the generalized curved-pipe MHD operator can be promoted into a real blanket
solve.

![LMX WHAM blanket mapped pipe mesh](_static/generated/wham_blanket_mesh_preview.png)

## WHAM field sampling on the mapped blanket mesh

`examples/wham_blanket_field_on_mesh_demo.py` is the solver-facing field
handoff after the mapped pipe mesh. The reusable API is
`sample_wham_field_on_centerline_pipe_mesh(...)` for WHAM-like fields,
`sample_field_on_centerline_pipe_mesh(...)` for arbitrary vector fields,
`centerline_pipe_frames(...)` for local frame recovery, and
`write_centerline_field_preview(...)` for QA artifacts. It samples the global
field on every mapped mesh point and projects it into local streamwise
`B_s` and transverse `B_\perp` components before any conservative electric
potential/current assembly.

The retained WHAM blanket mesh-field run uses the same `65` station,
`18 × 48` cross-section mesh as the mesh preview. It passes finite-value
checks, reports peak centerline `B_\perp ≈ 3.60e-1 T`, negligible streamwise
field on the centerline (`max |B_s|/|B| ≈ 3.4e-15`), and
`max_cross_section_relative_b_span ≈ 2.28`. This does not claim a full MHD
blanket solve; it closes the coordinate-frame and field-data handoff needed
before the generalized curved-pipe `φ`, `J`, and pressure solver is promoted.

![LMX WHAM field sampled on the mapped blanket pipe mesh](_static/generated/wham_blanket_field_on_mesh.png)

## WHAM conservative current closure on the mapped blanket mesh

`examples/wham_blanket_current_closure_demo.py` is the first conservative
`phi/J` gate on the approved mapped blanket route. It uses the sampled
WHAM-like field, prescribes a streamwise pipe velocity profile, solves the
inductionless potential equation using the same conservative pipe current
operators as the extruded solver, and reconstructs `J_s`, `J_r`, and
`J_theta`. This is still not a full curved-pipe momentum or turbulence solve;
it gates electric-potential/current assembly before the pressure-velocity
operator is promoted to the generalized centerline mesh.

The retained run uses a bounded `48 × 14 × 36` mapped-pipe mesh with
PbLi-like electrical conductivity. It records dimensional
`max |div J| ≈ 1.08e-2`, but the physically meaningful solver gate is the
residual relative to the EMF-divergence source: `relative |div J| ≈ 3.21e-9`
and `charge_balance_to_current_scale ≈ 2.05e-8`, with zero wall-current
leakage and zero net boundary-current residual. The same reconstructed
current field gives a streamwise `J×B` pressure-drop proxy of `≈ 6.97 kPa`,
which is now available separately from the reduced `σ U B_\perp^2` estimate.

![LMX WHAM conservative current closure on the mapped blanket pipe](_static/generated/wham_blanket_current_closure.png)

## WHAM blanket differentiable pressure-drop study

`examples/wham_blanket_autodiff_research_demo.py` reuses the approved blanket
route, WHAM-like field sampling, PbLi-like properties, and reduced
fixed-flow-rate pressure budget, but evaluates the budget with JAX arrays. The
same reduced model is then differentiated with respect to coil separation,
field multiplier, and mean velocity. This answers bounded design questions
before the full curved-pipe pressure-velocity solver is promoted.

The retained study reports `Delta p ≈ 26.5 kPa` at `U = 0.20 m/s`,
`R = 0.12 m`, field multiplier `8.0`, and coil separation `1.96 m`. The local
autodiff sensitivity is `d(Delta p)/ds ≈ 13.1 kPa/m`. A simple Newton update
using `d(Delta p)/d(field_scale)` reduces the field multiplier to `≈ 6.94` to
hit a `20 kPa` pressure-drop target at fixed flow rate. The result is a
research-design gate and publication-ready pressure/sensitivity figure; it is
not a claim of turbulent curved-pipe validation.

![LMX WHAM blanket differentiable pressure-drop study](_static/generated/wham_blanket_autodiff_research.png)
