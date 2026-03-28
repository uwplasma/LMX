# Theory

LMX targets low magnetic Reynolds number liquid-metal flows in the inductionless
limit. The current solver scope is laminar duct-style flow with an electric-potential
closure, explicit conductivity regions, and a JAX execution model that keeps the core
operators vectorized and differentiable.

## Governing model

The present solver advances a streamwise velocity field `U(y, z, t)` together with a
cross-sectional electric potential `phi(y, z, t)`:

- `div(J) = 0`
- `J = sigma (-grad(phi) + U x B)`
- `rho dU/dt = -dp/dx + mu Lap(U) + (J x B)_x`

This is the inductionless electric-potential formulation typically used for laminar
Hartmann, Shercliff, and Hunt-type liquid-metal flows.

## Physical assumptions

- Magnetic Reynolds number is small, so the imposed field is not flow-evolved.
- Material properties are piecewise constant by region in the current implementation.
- The current milestone excludes temperature coupling, free surfaces, and turbulence.

## Discretization principles

- Structured finite-volume style operators over logically rectangular control volumes.
- Conservative electric-current closure across fluid-solid conductivity jumps.
- Explicit wall regions for conducting liners rather than boundary-only shortcuts.
- Semi-implicit treatment of linear Lorentz damping where it improves stability.

## Numerical design rules

- Derive controls from geometry, materials, and nondimensional scales where possible.
- Keep native runs and validation runs on the same solver path.
- Avoid embedding backend-specific assumptions in the governing solver.
- Prefer mesh-derived interior sampling and geometry-derived scales over fixed offsets.

## Validation context

LMX is validated first against analytical liquid-metal benchmarks. Optional external
case archives are used as secondary comparison targets and regression anchors, but
they do not define the solver interface or solver design.
