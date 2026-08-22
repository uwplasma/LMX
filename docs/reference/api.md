# Python API

The package root is the small, stable convenience surface. Advanced workflows
live in the module that owns their concepts.

## Root API

| Area | Names |
|---|---|
| Cases and solves | `make_hartmann_case`, `make_shercliff_case`, `make_hunt_case`, `solve` |
| Meshes | `generate_rect_duct_mesh`, `generate_rect_duct_mesh_from_faces`, `generate_layered_duct_mesh`, `generate_layered_duct_mesh_from_fluid_faces`, `generate_multilayer_duct_mesh` |
| Wall models | `WallLayer`, `wall_conductance_ratio`, `effective_pinhole_conductance_ratio`, `tangential_stack_conductance_ratio`, `normal_stack_leakage_ratio`, `equivalent_single_layer`, `nested_wall_layer_resolution_summary` |
| Units | `dynamic_to_kinematic_viscosity`, `kinematic_to_dynamic_viscosity`, `hartmann_number`, `reynolds_number`, `interaction_parameter`, `magnetic_reynolds_number`, `magnetic_field_from_hartmann` |
| Evidence | Power balance in `lmx.solvers` and the analytical, conservation, and packaged benchmark tools in `lmx.validation` |
| Runtime | `enable_compilation_cache` |

`solve(model)` accepts either a `CaseSpec` or an
`ExtrudedInductionlessProblem`. Both result types expose `converged`, `status`,
`steps`, `residual`, `fields`, and `diagnostics`; specialized solve functions
provide restart, progress, logging, and timing hooks in their owning modules.

## Case schema

```{eval-rst}
.. automodule:: lmx.specs
   :members:
   :show-inheritance:
```

## Three-dimensional fringing

```{eval-rst}
.. automodule:: lmx.fringing
   :members:
```

## Imposed fields

```{eval-rst}
.. automodule:: lmx.mesh
   :members:
```

## Differentiation

```{eval-rst}
.. automodule:: lmx.cases
   :members:
```

## Output and restart

```{eval-rst}
.. automodule:: lmx.io
   :members:
```

## Validation

```{eval-rst}
.. automodule:: lmx.validation
   :members:
```

## Units and walls

```{eval-rst}
.. automodule:: lmx.physics
   :members:
```
