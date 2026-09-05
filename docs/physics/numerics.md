# Numerical methods

LMX uses cell-centered structured finite-volume/finite-difference operators.
The conservative face operators use distance-aware harmonic interpolation and
local metric widths. Generic collocated momentum and its reconstructed field
diagnostics have distinct operators; they do not inherit the face scheme's
conservation guarantees.

The fully developed solve alternates electric potential, current/Lorentz
reconstruction, and the axial momentum update until the configured physical
gate passes. Extruded solvers advance momentum, correct pressure/flow, and
resolve electric current closure. Specialized finite-volume paths retain face
fluxes; the generic duct uses a collocated correction and stationwise flow
adjustments. Gauge constraints remove constant nullspaces where the boundary
conditions leave them unconstrained.

## Pressure-operator contract

On the orthogonal duct mesh, let $B$ contain oriented differences between
neighboring cell pressures, $W$ the diagonal cell volumes, and $T$ the positive
face transmissibilities. The conservative face-pressure block satisfies

$$
WL=B^T T B,
\qquad T_f=\frac{A_f}{\delta_L/m_L+\delta_R/m_R}.
$$

Here $m$ is frozen pressure mobility and $\delta$ the half-cell distance to
the face. An outlet pressure fixed to zero adds $A_fm/\delta$ to the outlet
diagonal. This construction follows discrete adjoint/Green-identity principles
described by [Hyman and Shashkov](https://doi.org/10.1137/S0036142996314044).
The test suite independently assembles this matrix on a nonuniform 18-cell
mesh with variable mobility. It checks the implemented face corrections and
divergence, volume-weighted symmetry, nonnegative energy, reverse derivatives,
rank 17 with all-Neumann boundaries and rank 18 with fixed outlet pressure.

These are properties of the frozen **face-pressure block**, not certification
of the coupled velocity/pressure/current residual or its cell reconstruction.
The generic collocated correction instead relaxes a compact Poisson stencil;
composing its centered divergence and gradient produces a wider stencil.
For an axial alternating pressure mode on unit cells, that composition is
zero in the interior while the compact Laplacian has magnitude four. Thus a
small compact Poisson residual does not certify projected cell divergence.
Coupled residual, boundary-work and continuum-refinement checks remain needed.

## B2 pressure and momentum coupling

The mass-flux initializer and pressure predictor share distance-weighted linear
interpolation of normal cell components to faces. For adjacent widths $h_L,h_R$,
$u_f=(h_Ru_L+h_Lu_R)/(h_L+h_R)$; internal affine fields and their geometry
derivatives are exact. Transverse wall fluxes are zero and the axial outlet
extrapolates the terminal cell. Face-flux divergence is conservative, but this
does not establish a weighted adjoint relation for the reconstructed cell
pressure force on nonuniform grids. Energy compatibility requires that separate
operator contract; see [Santos et al., §2–3](https://www.scipedia.com/wd/images/2/22/Draft_Sanchez_Pinedo_5754368871854_paper.pdf).

The Newtonian viscous stress uses unlimited least-squares velocity gradients:
at fixed viscosity its discrete action is linear in velocity and boundary
data. Advective reconstruction still limits its speed-squared gradient. The
packed setup selects these components separately without duplicating the
gradient implementation. Limiting the viscous gradient would introduce
direction-dependent derivatives at a resting state, despite a linear
constitutive law.

`_duct_momentum_residual` retains signed field residuals in force-density
units; acceptance diagnostics reduce that same residual to normalized maxima.
A 3×3×3 resting, constant-viscosity, zero-advection mechanical test checks its
108×108 automatic Jacobian, full rank with fixed outlet pressure, the pressure/
continuity adjoint relation, and directional differences down to $10^{-7}$.
This Stokes-limit check does not certify finite-advection limiter transitions,
the electromagnetic coupled residual, or a converged B2 steady state.

The production B2 accelerator acts only on the three mechanical velocity
components and their conservative compact face fluxes. One electric solve then
closes the accepted velocity before current, Lorentz force, charge balance, and
momentum defect are evaluated. This keeps every diagnostic and restart field on
the same accepted state, avoids a second electric solve, and removes electric
potential from the acceleration history.

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
between physical defect reduction and runtime. Mechanical acceleration is
applied after those correctors; electric closure and Lorentz reconstruction
then use the accepted conservative velocity. This follows the
pressure-correction structure of SIMPLE and its consistent refinements while
retaining LMX's MHD-specific residual and boundary contracts.

The B2 predictor also uses the positive local electromagnetic pseudo-mass
$R_B=\sigma|B|^2I$:

$$
(A^n+\Delta tR_B)u^* = b_L^n-Gp^n+\Delta tR_Bu^n.
$$

At a fixed point $u^*=u^n$, the added terms cancel exactly, so this changes
neither the discrete steady equations nor their physical residual. It damps
the stiff magnetic pseudo-time mode, and the same augmented diagonal defines
$r_{AU}$ in the pressure correction. A dense-operator test verifies both the
linear system and this fixed-point identity.

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
