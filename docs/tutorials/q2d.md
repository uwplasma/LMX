# Evolve a Q2D vortex field

The Sommeria--Moreau (SM82) model describes a depth-averaged conducting flow
under a strong transverse magnetic field. LMX evolves periodic vorticity with
viscous diffusion and linear Hartmann-layer friction.

```python
import jax.numpy as jnp
import lmx

case = lmx.make_q2d_case(
    shape=(64, 64),
    viscosity=2.0e-3,
    hartmann_friction=4.0e-2,
    dt=1.5e-2,
    steps=160,
    history_stride=4,
)
result = lmx.solve(case)

assert result.converged
print(result.diagnostics.kinetic_energy_final)
```

`Q2DProblem` also accepts an arbitrary two-dimensional initial-vorticity array
and a vorticity source of the same shape. Lengths, viscosity, friction, time,
and forcing must use one consistent unit system. The zero Fourier mode is
projected out, so the periodic problem has zero net circulation.

```{image} ../_static/q2d_vortex_decay.webp
:alt: Initial and final Q2D vorticity beside the kinetic-energy decay
:align: center
```

Run `python examples/q2d_turbulence_demo.py` to reproduce this poster, a JSON
diagnostic record, and an MP4 when FFmpeg is installed. Sparse field frames are
opt-in through `history_stride`; the default keeps only the final state.

## Differentiate the finite evolution

`evolve_q2d` is the field-only numerical core for optimization. It omits host
status and output assembly so continuous array, forcing, length, viscosity,
Hartmann-friction, and timestep parameters remain traced:

```python
import jax


def objective(friction):
    vorticity, velocity_x, velocity_y = lmx.evolve_q2d(
        case.initial_vorticity,
        viscosity=case.viscosity,
        hartmann_friction=friction,
        dt=case.dt,
        steps=case.steps,
    )
    return jnp.mean(vorticity**2) + 0.1 * jnp.mean(velocity_x**2 + velocity_y**2)


value, gradient = jax.jit(jax.value_and_grad(objective))(case.hartmann_friction)
```

This is the exact reverse derivative of the dealiased IFRK4 discretization.
SOLVAX divides a long recurrence into segments and rematerializes one segment
during the backward pass. With $N$ steps and checkpoint width $C$, retained
trajectory state is $O(N/C+C)$; the default $C=\lceil\sqrt N\rceil$ is
`O(sqrt(N))`. Set `adjoint_checkpoint_size` only after measuring a different
memory/recomputation trade-off.

`steps`, grid shape, and checkpoint width are discrete static controls. Use
`Q2DProblem` plus `solve` when validated host diagnostics or saved frames are
needed; use `evolve_q2d` inside `grad`, `jvp`, `vjp`, or `vmap`.

## CPU and GPU execution

The same public API selects JAX's active backend. A controlled office-host run
used JAX 0.6.2, float32 fields, a $256\times256$ grid, 80 steps, one compilation
run, and five warm repetitions. The warm median was 2.233 s on CPU and 0.08663 s
on one NVIDIA RTX A4000, a 25.78x speedup. The final CPU and GPU vorticity fields
agreed to relative $L_2=2.38\times10^{-6}$ and
$L_\infty=3.67\times10^{-6}$; warm-time coefficients of variation were 0.21%
and 1.39%, respectively.

These figures characterize this workload and hardware, not every grid or
device. Timings exclude compilation, use identical precision and inputs, and
synchronize the final field before stopping the clock. Multi-GPU performance
is reported separately because the present periodic Q2D state is not sharded.

## Interpretation

The solver reports kinetic energy, enstrophy, the integrated energy-budget
residual, spectral velocity divergence, and maximum Courant number. A result is
complete only when every field is finite and the Courant number does not exceed
one.

SM82 is a basic quasi-two-dimensional closure, not a general replacement for
the three-dimensional inductionless solver. Use the 3-D model for developing
Hartmann layers, spatially varying transverse structure, fringing fields, and
geometries whose depth-average assumptions are not satisfied. Higher-order
inertial corrections described by Pothérat, Sommeria, and Moreau are not part of
this Q2D path.
