# Differentiate physical solves

LMX separates traced numerical fields from host-side validation, status, I/O,
and plotting. An accepted field core must carry continuous physical parameters
through the production discretization; mesh counts, iteration limits, and
output policies remain static controls. The accepted field cores are the
steady rectangular/layered-duct and transient Q2D models below.

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
`solve_fully_developed_fields` and `evolve_q2d`. Other result objects are host
orchestration unless their API reference explicitly identifies a traced field
core and derivative evidence.
