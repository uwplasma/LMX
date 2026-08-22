# Numerical methods

LMX uses cell-centered structured finite-volume/finite-difference operators.
Face conductivities and viscosities use distance-aware harmonic interpolation;
the same face fluxes feed conservation diagnostics. Nonuniform meshes retain
their local metric widths.

The fully developed solve alternates electric potential, current/Lorentz
reconstruction, and the axial momentum update until the configured physical
gate passes. The 3-D solve advances momentum, applies a face-flux pressure
projection, enforces the prescribed flow constraint when present, and resolves
electric current closure. Gauge constraints remove the constant nullspace of
pressure and electric potential.

## LMX and SOLVAX

LMX owns:

- geometry metrics and material coefficients;
- boundary and interface equations;
- MHD coupling and dimensional scaling;
- charge, mass, momentum, and power residuals;
- case-level convergence and validation.

SOLVAX owns:

- PCG, GMRES/FGMRES, and fixed-point iteration;
- Jacobi, line, additive, deflation, and Schur preconditioning primitives;
- tridiagonal and sparse direct solves;
- solver state, termination metadata, and implicit linear differentiation.

LMX calls these algorithms with MHD-specific operator actions and then certifies
the returned state in physical units. This keeps solver policy reusable without
moving geometry or physics into SOLVAX.

## Accuracy and performance

Analytical and manufactured tests check observed order on refined meshes.
Production claims additionally require stable physical observables, stricter
solver tolerances, and conservation gates. JAX compilation and warm execution
are measured separately. Repeated cases benefit from the compilation cache and
reused structured factors; optional output and histories should remain disabled
when memory is the limiting resource.
