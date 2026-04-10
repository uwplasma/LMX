# Theory

LMX targets low magnetic Reynolds number liquid-metal flows in the
inductionless limit. The current public solver scope is laminar single-phase
flow on structured cross-sections, with explicit conductivity regions and a JAX
implementation that keeps the core operators vectorized, JIT-safe, and suitable
for differentiation.

## Governing equations

The fully developed duct solver advances a streamwise velocity field
`u(y, z, t)` and an electric potential field `phi(y, z, t)` over the
cross-section:

- `div(J) = 0`
- `J = sigma (-grad(phi) + u x B)`
- `rho * du/dt = -dp/dx + mu * Lap(u) + (J x B)_x`

For the current duct solvers, the imposed magnetic field is prescribed and is
not flow-evolved.

## Assumptions

- low magnetic Reynolds number
- incompressible single-phase liquid metal
- piecewise-constant material properties by region
- no temperature coupling
- no turbulence model
- no free surface

## Nondimensional benchmark families

LMX uses the classical duct benchmark families as primary validation anchors:

- Hartmann flow: conducting walls normal to the magnetic field
- Shercliff flow: insulating walls
- Hunt flow: conducting Hartmann walls with insulating side walls

These are the mandatory Benchmark A cases in the verification and validation
ladder summarized by [Samper et al.](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf).

## Numerical formulation

### Default solver family

The default research path is `solver.kind = "fully_developed_inductionless"`.

That path:

- solves the duct problem in terms of `u` and `phi`
- uses conservative face-current fluxes
- handles layered solid walls explicitly through region conductivity fields
- exposes steady and transient modes
- avoids the legacy velocity-limiter-driven pseudo-transient closure

### Legacy solver family

`solver.kind = "legacy_reduced"` is retained only for regression and historical
comparison. It is not the recommended path for new research results.

## Physical diagnostics

LMX exposes both solution fields and integral diagnostics. The research-grade
diagnostics currently include:

- volumetric flow rate
- mean current magnitude
- Lorentz power
- electric-current divergence residual
- gauge residual
- interface current continuity residual
- linear-solve residual and iteration history

## File map

The main physics and numerics live in:

- `lmx/physics.py`
  - magnetic field specification and material-field assembly
- `lmx/operators.py`
  - finite-volume style gradient, divergence, and Laplacian operators
- `lmx/linear.py`
  - linear solver helpers and iterative backends
- `lmx/solvers.py`
  - steady/transient solver families and time-integration control
- `lmx/validation.py`
  - analytical benchmark comparison and validation summaries

## Differentiable lane

The core JAX solver path is intended to remain differentiable end to end for
steady and short transient studies. The current research references for that
direction are:

- [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
- [Lineax matrix-free linear solvers](https://docs.kidger.site/lineax/api/solvers/)
- [Diffrax adjoints](https://docs.kidger.site/diffrax/api/adjoints/)
- [Φ-Flow and differentiable PDE workflows](https://proceedings.mlr.press/v235/holl24a.html)

The CLI, plotting, reporting, and postprocessing layers are allowed to use
pragmatic non-differentiable utilities where that improves usability and
performance.
