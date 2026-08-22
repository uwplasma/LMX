# Python API

The package root is the small, stable convenience surface. Advanced workflows
live in the module that owns their concepts.

## Root API

| Area | Names |
|---|---|
| Cases and solves | `make_hartmann_case`, `make_shercliff_case`, `make_hunt_case`, `solve_steady`, `solve_transient` |
| Meshes | `generate_rect_duct_mesh`, `generate_rect_duct_mesh_from_faces`, `generate_layered_duct_mesh`, `generate_layered_duct_mesh_from_fluid_faces`, `generate_multilayer_duct_mesh` |
| Wall models | `WallLayer`, `wall_conductance_ratio`, `effective_pinhole_conductance_ratio`, `tangential_stack_conductance_ratio`, `normal_stack_leakage_ratio`, `equivalent_single_layer`, `nested_wall_layer_resolution_summary` |
| Units | `dynamic_to_kinematic_viscosity`, `kinematic_to_dynamic_viscosity`, `hartmann_number`, `reynolds_number`, `interaction_parameter`, `magnetic_reynolds_number`, `magnetic_field_from_hartmann` |
| Evidence | `fully_developed_power_balance`, `load_shercliff_analytical`, `load_hunt_analytical`, `load_closed_channel_analytical`, `load_processed_slice` |
| Runtime | `enable_compilation_cache` |

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
.. automodule:: lmx.field_models
   :members:
```

## Differentiation

```{eval-rst}
.. automodule:: lmx.autodiff
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
