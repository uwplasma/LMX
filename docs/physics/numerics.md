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

The production B2 fixed-point residual uses the prescribed mean flow velocity
and induced-potential scale to balance its velocity and electric blocks before
Anderson mixing. Numerical safety limits are guardrails, not state scales, and
therefore do not enter the residual norm.

B2 couples momentum and pressure with a conservative SIMPLE-style correction.
For each frozen Lorentz field, two pressure--momentum correctors apply

$$
A^n u^* = b_L^n - Gp^n, \qquad
D r_{AU} Gp' = D u^*, \qquad
u^{n+1} = u^* - r_{AU}Gp', \qquad
p^{n+1} = p^n + 0.4p'.
$$

Here $r_{AU}$ is the inverse diagonal already assembled for the implicit
momentum predictor. The pressure operator and reconstructed face flux use the
same distance-weighted harmonic interpolation on nonuniform cells. The fixed
pressure relaxation stabilizes the segregated correction without adding a
second field history, and two correctors give the selected production balance
between physical defect reduction and runtime. Electric closure and Lorentz
reconstruction then use the corrected conservative velocity. This follows the
pressure-correction structure of SIMPLE and its consistent refinements while
retaining LMX's MHD-specific residual and boundary contracts.

The primal-only B2 projection stops at a $10^{-10}$ linear tolerance while
also enforcing the volume-scaled local mass-balance target. This keeps the
linear error below five percent of the independent $10^{-3}$ balance gate on
the production mesh without spending iterations on roundoff-level pressure
corrections. Traced 3-D paths retain roundoff-level primal solves where their
implicit derivatives require them.

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
- solver state, termination metadata, implicit linear differentiation, and
  checkpointed exact reverse mode for long recurrences.

LMX calls these algorithms with MHD-specific operator actions and then certifies
the returned state in physical units. This keeps solver policy reusable without
moving geometry or physics into SOLVAX.

## Q2D spectral evolution

The periodic Q2D path uses full complex Fourier transforms, the two-thirds
dealiasing rule for the vorticity-advection product, and fourth-order
integrating-factor Runge--Kutta time stepping. Viscous and Hartmann-friction
terms are integrated exactly within each step. SOLVAX supplies the reusable
periodic Poisson symbol and zero-mean spectral inversion for the streamfunction;
LMX owns vorticity dynamics, velocity reconstruction, the energy identity, and
physical acceptance.

The largest stable integration segment is JIT compiled. A positive
`history_stride` divides a run into compiled segments and transfers only the
requested vorticity frames to the host; zero retains no field history. This
keeps primal result storage independent of the number of time steps.

## Derivative policy

The derivative algorithm is part of each numerical method. Converged linear or
steady nonlinear equations use an implicit tangent/adjoint system, so reverse
cost is one additional transposed solve and does not depend on the number of
primal iterations. Finite transient models differentiate the discrete update.
Generic 3-D ducts use an implicit electric VJP and exact checkpointed
projection and outer recurrences. Q2D uses the same two-level schedule;
retained trajectory state is $O(N/C+C)$ for $N$ steps and width $C$, with a
square-root default.

Field arrays and continuous physical coefficients are traced. Mesh topology,
array shapes, iteration limits, checkpoint widths, convergence strings,
logging, and file output are static or host-side. Gradient acceptance combines
an independent analytical/finite-difference/transpose check with primal and
adjoint residuals, compiled memory scaling, warm runtime, and CPU/GPU parity.

## Accuracy and performance

Analytical and manufactured tests check observed order on refined meshes.
Production claims additionally require stable physical observables, stricter
solver tolerances, and conservation gates. JAX compilation and warm execution
are measured separately. A fully developed solve assembles invariant potential
and velocity coefficients and its preconditioner once, then reuses them at each
coupling step. Repeated cases also benefit from the compilation cache; optional
output and histories should remain disabled when memory is the limiting resource.
