# Python API

`enable_x64` explicitly enables JAX float64 arrays process-wide. Call it before
constructing meshes or tracing functions; importing LMX does not change precision.
Case factories and TOML `[case]` accept `dtype="float32"` or `dtype="float64"`
(default). During the transition, constructing a float64 case with x64 disabled
enables it with a `DeprecationWarning`; explicit activation avoids that warning.
The case dtype controls fully developed mesh, material, initial/restart and output
fields, including compiled derivatives. Float32 is qualified only for the tested
low-Hartmann cases, not high-Hartmann validation. Supplied meshes are cast to the
case dtype; casting cannot recover precision lost during their construction.
See JAX's [dtype and x64 contract](https://docs.jax.dev/en/latest/101/default_dtypes.html).

```python
import lmx

lmx.enable_x64()
case = lmx.make_hartmann_case(ha=2, ny=32, nz=32, dtype="float32")
u, phi, jy, jz, lorentz = lmx.solve_fully_developed_fields(case)
```

The package root is the small, stable convenience surface. Advanced workflows
live in the module that owns their concepts.

## Root API

| Area | Names |
|---|---|
| Cases and solves | `make_hartmann_case`, `make_shercliff_case`, `make_hunt_case`, `make_q2d_case`, `solve_fully_developed_fields`, `evolve_q2d`, `Q2DProblem`, `solve` |
| Meshes | `generate_rect_duct_mesh`, `generate_rect_duct_mesh_from_faces`, `generate_layered_duct_mesh`, `generate_layered_duct_mesh_from_fluid_faces`, `generate_multilayer_duct_mesh` |
| Wall models | `WallLayer`, `wall_conductance_ratio`, `effective_pinhole_conductance_ratio`, `tangential_stack_conductance_ratio`, `normal_stack_leakage_ratio`, `equivalent_single_layer`, `nested_wall_layer_resolution_summary` |
| Units | `dynamic_to_kinematic_viscosity`, `kinematic_to_dynamic_viscosity`, `hartmann_number`, `reynolds_number`, `interaction_parameter`, `magnetic_reynolds_number`, `magnetic_field_from_hartmann` |
| Evidence | Power balance in `lmx.solvers` and the analytical, conservation, and packaged benchmark tools in `lmx.validation` |
| Runtime | `enable_compilation_cache` |

`solve(model)` accepts a `CaseSpec`, `ExtrudedInductionlessProblem`, or
`Q2DProblem`. All result types expose `converged`, `status`,
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

## Quasi-two-dimensional flow

```{eval-rst}
.. automodule:: lmx.q2d
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
