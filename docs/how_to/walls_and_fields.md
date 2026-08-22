# Configure walls and imposed fields

## Wall layers

Use `WallLayer` to state thickness, resolution, and conductivity explicitly,
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

Use `wall_conductance_ratio`, `tangential_stack_conductance_ratio`, and
`normal_stack_leakage_ratio` to document the physical regime. The mesh and
validation helpers report the resolved layer thicknesses and interface-current
residuals.

## Analytic fields

`MagneticFieldSpec(kind="analytic", fn=...)` accepts a callable that returns a
three-component field. `make_divergence_free_cross_section_field` provides a
compact analytic example. Sample the function with
`sample_cross_section_field` before a production run and verify its divergence
and intended extrema.

## Tabulated fields

Store coordinates and vector components in an NPZ table, load it with
`load_tabulated_rect_field`, and pass the returned callable through
`MagneticFieldSpec`. Interpolation is bounded by the tabulated domain. Keep the
source table, coordinate units, field units, interpolation rule, and Maxwell
checks with the run record.

`examples/variable_field_extruded_demo.py` shows the same solve interface for a
custom vector field.
