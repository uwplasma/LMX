# Walls and imposed fields

This tutorial builds explicit wall stacks and imposed magnetic fields, then
uses the same case/result model as the named duct and fringing workflows.

## Wall layers

Start with `WallLayer` to state thickness, resolution, and conductivity explicitly,
then pass the layers to the layered mesh builders. LMX distinguishes fluid,
conducting solid, and insulating regions; interface conductance uses the
adjacent material values and face distances.

```python
from lmx import WallLayer, generate_multilayer_duct_mesh

layers = (
    WallLayer(name="steel", thickness=2e-3, cells=3, conductivity=8e5),
    WallLayer(name="insulator", thickness=5e-4, cells=2, conductivity=1e-8),
)
mesh = generate_multilayer_duct_mesh(
    width=0.2,
    height=0.1,
    ny=48,
    nz=32,
    wall_layers={"left": layers, "right": layers},
)
```

Inspect the resulting regime with `wall_conductance_ratio`,
`tangential_stack_conductance_ratio`, and
`normal_stack_leakage_ratio` to document the physical regime. The mesh and
validation helpers report the resolved layer thicknesses and interface-current
residuals.

## Add an analytic field

`MagneticFieldSpec(kind="analytic", fn=...)` accepts a callable that returns a
three-component field. `make_divergence_free_cross_section_field` provides a
compact analytic example. Sample the function with
`sample_cross_section_field` before a production run and verify its divergence
and intended extrema.

## Use measured field data

Store Cartesian coordinates (metres) and components (tesla) in an NPZ table.
This synthetic affine example is divergence-free; replace its arrays with your
measured/exported data. Include the conducting walls in the table's domain.

```python
from pathlib import Path
from dataclasses import replace
import numpy as np
from lmx import make_hartmann_case
from lmx.mesh import write_tabulated_field_npz, sample_tabulated_cross_section_field
from lmx.specs import MagneticFieldSpec

y = z = np.linspace(-1.2, 1.2, 17)
yy, zz = np.meshgrid(y, z, indexing="ij")
path = write_tabulated_field_npz(
    Path("artifacts/imposed_field.npz"), y=y, z=z,
    bx=np.zeros_like(yy), by=0.1 * yy, bz=5.0 - 0.1 * zz,
)
sampled = sample_tabulated_cross_section_field(path, y=yy, z=zz)
assert np.allclose(sampled[..., 2], 5.0 - 0.1 * zz)
case = replace(make_hartmann_case(), magnetic_field=MagneticFieldSpec(
    kind="tabulated", table_path=str(path),
))
```

For 3-D data also provide `x` and component arrays shaped `(nx, ny, nz)`;
extruded sampling uses the supplied physical axial stations, including their
origin and spacing. Axes must be finite, strictly increasing vectors with at
least two points; component shapes must match. Out-of-domain/nonfinite queries
raise `ValueError`, never silently extrapolate. The file-loading interface is
host-side setup, not a differentiable live coil/geometry interface. Keep source
provenance, interpolation error and independent Maxwell checks with each run.

Run `python examples/li_aln_wall_stack_example.py` for explicit conducting and
insulating layers. The differentiated field/wall design workflow is executable
as `python examples/variable_field_extruded_demo.py`; it writes a compact JSON
record and optimization figure under `artifacts/examples/`.
