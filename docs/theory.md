# Theory

LMX targets the inductionless electric-potential formulation used in FreeMHD for low magnetic Reynolds number flows.

## Governing model

- Ohm's law: `J = sigma (-grad(phi) + U x B)`.
- Charge conservation: `div(J) = 0`.
- Streamwise momentum for the current laminar parity implementation:
  `du/dt = nu * Lap(u) + (dpdx + (J x B)_x) / rho`.

The current implementation focuses on fully developed laminar duct flow on structured cross-sections, with explicit solid conductivity regions for conducting-wall cases.

## Discretization

- Cell-centered tensor-product mesh in the `y-z` cross-section.
- Conservative face-flux balance for electric potential.
- Pseudo-transient implicit-Euler-style relaxation for streamwise velocity.
- Explicit conductivity maps for fluid and solid cells.
- JAX arrays and `jax.lax.scan` for fixed-shape time marching.

## Mapping to FreeMHD

- The electric-potential solve mirrors the `epotMultiRegionFoam` formulation rather than the full OpenFOAM runtime stack.
- Conducting walls are represented as explicit solid regions, aligned with the plan for Hunt-style cases.
- Free-surface, temperature, and turbulence are deferred.
