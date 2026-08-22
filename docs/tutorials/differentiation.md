# Differentiate physical solves

LMX separates traced numerical fields from host-side validation, status, I/O,
and plotting. An accepted field core must carry continuous physical parameters
through the production discretization; mesh counts, iteration limits, and
output policies remain static controls. The accepted field cores are the
steady duct, finite 3-D rectangular/layered-duct, and transient Q2D models.

SOLVAX-backed linear systems use implicit differentiation: a VJP solves one
transposed system instead of recording PCG or GMRES iterations. A coupled
steady-flow interface is documented as end-to-end differentiable only when its
production field equations use that contract and pass independent gradient,
residual, runtime, and memory gates.

## Steady duct response

```python
import jax
import jax.numpy as jnp
import lmx

case = lmx.make_shercliff_case(ha=20, ny=48, nz=48)


def objective(forcing, field_scale):
    velocity, potential, jy, jz, lorentz = lmx.solve_fully_developed_fields(
        case,
        forcing=forcing,
        magnetic_field_scale=field_scale,
    )
    return jnp.mean(velocity) + 1e-3 * jnp.mean(jy**2 + jz**2)


value, gradient = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(1.0, 1.0)
```

This path assembles the same mesh, material coefficients, potential equation,
momentum equation, no-slip interpolation, and current/Lorentz fields as
`solve`. The coupled affine state and its nested linear systems use implicit
SOLVAX derivatives, so iteration histories are absent from the reverse tape.
The current continuous inputs are pressure forcing and a scalar multiplier on
the imposed magnetic field. Rectangular Hartmann/Shercliff and layered Hunt
cases pass production-field and independent adjoint gates. Fixed-flow and 3-D
cases remain outside this API until their own gates pass.

## Three-dimensional fringe response

`evolve_extruded_fields` returns the production generic-duct velocity,
pressure, potential, current, and Lorentz-force fields after a static number of
steps. Pressure forcing and a scalar imposed-field multiplier are continuous:

```python
import jax
import jax.numpy as jnp
from lmx.fringing import build_square_duct_extruded_problem, evolve_extruded_fields

problem = build_square_duct_extruded_problem(nx_stations=12, ny=16, nz=16)


def objective(parameters):
    u, _, _, _, potential, *_ = evolve_extruded_fields(
        problem,
        forcing=parameters[0],
        magnetic_field_scale=parameters[1],
        steps=40,
    )
    return jnp.mean(u**2) + 1e-3 * jnp.mean(potential**2)


value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.ones(2))
```

Electric closure uses an implicit SOLVAX VJP. The finite collocated projection
and outer recurrence use exact two-level checkpoint schedules with
`O(N/C + C)` retained states and square-root defaults. Tests require parity
with the ordinary production solve, independent finite differences, JVP/VJP
duality, and lower compiled reverse temporary memory than a full tape.
Geometry, material layout, step count, and checkpoint width are static.
Specialized ALEX B2 and pipe paths fail closed until their coupled operators
have independent derivative gates.

## Transient Q2D response

For a time-dependent field objective, call the field-only core:

```python
import jax.numpy as jnp
import lmx

case = lmx.make_q2d_case(shape=(32, 32), steps=80)


def objective(parameters):
    viscosity, friction = parameters
    vorticity, _, _ = lmx.evolve_q2d(
        case.initial_vorticity,
        viscosity=viscosity,
        hartmann_friction=friction,
        dt=case.dt,
        steps=case.steps,
    )
    return jnp.mean(vorticity**2)


value, gradient = jax.value_and_grad(objective)(
    jnp.asarray([case.viscosity, case.hartmann_friction])
)
```

This derivative is exact for the finite dealiased IFRK4 evolution. Its default
SOLVAX checkpoint schedule stores `O(sqrt(steps))` trajectory states instead of
the full tape. The analytical decay, JVP/VJP identity, and compiled reverse
memory tests in `tests/test_physics.py` are the executable acceptance contract.

The explicit field-level optimization surfaces are
`solve_fully_developed_fields`, `evolve_extruded_fields`, and `evolve_q2d`.
Other result objects are host orchestration unless their API reference
explicitly identifies a traced field core and derivative evidence.
