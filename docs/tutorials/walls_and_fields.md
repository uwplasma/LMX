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

Store coordinates and vector components in an NPZ table, load it with
`load_tabulated_rect_field`, and pass the returned callable through
`MagneticFieldSpec`. Interpolation is bounded by the tabulated domain. Keep the
source table, coordinate units, field units, interpolation rule, and Maxwell
checks with the run record.

Run `python examples/li_aln_wall_stack_example.py` for explicit conducting and
insulating layers, then `python examples/variable_field_extruded_demo.py` for
an analytic divergence-free field and a complete 3-D solve. Both write a
compact JSON summary under `artifacts/examples/` so wall, charge-closure, and
field-response diagnostics can be inspected together.
