# Numerics and Implementation

This page describes how the current LMX solver families are assembled in code.
The emphasis is on the equations actually implemented today, the discrete
operators used by the code, and where each step lives in the source tree.

## Solver families

LMX currently exposes three solver-family names:

- `fully_developed_inductionless`
  - the default research solver for duct benchmarks
- `reduced_inductionless`
  - a reduced-model alternative retained for comparative studies
- `extruded_inductionless`
  - reserved for the next 3D/fringing-field solver family

The active implementation today is the fully developed cross-sectional solver
in `lmx/solvers.py`.

## Discrete operators

The structured finite-volume-style operators live in `lmx/operators.py`. They
are written in vectorized `jax.numpy` form and provide:

- cell-center coordinate construction
- scalar gradients on the `y-z` cross-section
- Laplacians on the `y-z` cross-section

These operators are used by both the standard solver lane and the
differentiable lane.

## Material and magnetic-field assembly

`lmx/physics.py` assembles:

- conductivity
- density
- viscosity
- fluid/solid masks
- constant, analytic, and ramped magnetic-field components

The current magnetic-field entry points are:

- `kind="constant"`
- `kind="analytic"` with a Python callback `fn(y, z) -> (..., ..., ...)`

The magnetic-field ramp law is:

$$
\alpha_B(t) = \mathrm{clip}\left(\frac{t - t_{\mathrm{start}}}{t_{\mathrm{duration}} + 10^{-6}}, 0, 1\right)
$$

and is implemented in `lmx/physics.py`.

## Fully developed formulation

For the fully developed duct solver, the unknowns are the streamwise velocity
$u(y,z,t)$ and electric potential $\phi(y,z,t)$. The governing equations used
by the implementation are described in detail on the theory page; the discrete
solver pipeline is:

1. build the mesh and material fields
2. evaluate the prescribed magnetic field
3. assemble the potential equation from conservative face-current fluxes
4. solve for $\phi$
5. assemble the streamwise momentum equation with Lorentz coupling
6. solve for $u$
7. update diagnostics and convergence checks

The conservative face-conductance and face-EMF helpers are implemented in
`lmx/solvers.py`:

- `_interface_conductance_y`
- `_interface_conductance_z`
- `_face_emf_y`
- `_face_emf_z`
- `_potential_coefficients`
- `_velocity_system_coefficients`

## Charge conservation treatment

LMX explicitly tracks three charge/current consistency signals:

- `div_current_max_history`
- `charge_balance_residual_history`
- `interface_current_residual_history`

The fully developed solver applies a compatibility projection to the potential
right-hand side before solving, so that the discrete Poisson problem satisfies
the zero-net-source condition required for a bounded potential solve.

This logic lives in `lmx/solvers.py`, while the diagnostics are stored in
`lmx/core.py` and persisted through `lmx/io.py`.

## Linear solver backends

Low-level iterative linear solver helpers are implemented in `lmx/linear.py`.
The current public linear-solver options are:

- `auto`
- `cg`
- `gmres`
- `bicgstab`
- `jacobi` / `cg_volume` / `lineax_cg` for selected potential solves

The actual use of those controls depends on solver family and mode, and is
parsed through `lmx/config.py`.

## Time stepping and bounded end time

The bounded-step logic lives in `lmx/solvers.py` through
`_bounded_time_step_count(...)`. Its purpose is to prevent overshooting
`t_final` when `dt` does not divide the time horizon exactly. In practice:

- `t_final` is the hard stop horizon
- `max_steps` is the hard upper bound
- the solver uses the minimum of both limits

## JAX usage

LMX uses JAX in three distinct ways:

1. `jax.numpy` vectorization for the main PDE operators
2. the differentiable lane in `lmx/autodiff.py`
3. sharded-kernel benchmarking in `lmx/scaling.py`

The production CLI, plotting, JSON writing, and documentation build path are
allowed to use pragmatic non-differentiable utilities where that improves
robustness or runtime.

## Source-code map

- `lmx/specs.py`
  - dataclasses for geometry, regions, solver controls, outputs, and cases
- `lmx/mesh.py`
  - structured duct and mapped pipe mesh generators
- `lmx/physics.py`
  - magnetic fields, ramps, and material-field assembly
- `lmx/operators.py`
  - structured finite-volume operators
- `lmx/linear.py`
  - iterative linear solves and residual norms
- `lmx/solvers.py`
  - steady/transient solver families and diagnostics
- `lmx/runtime_logging.py`
  - detailed runtime logging, including initial/final residuals
- `lmx/io.py`
  - `.npz`, restart, and VTK output
- `lmx/autodiff.py`
  - differentiable Hartmann studies and inverse-design workflows
- `lmx/fringing.py`
  - axial fringing-field research slice and stacked station bundles
