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

`SolverConfig.extruded_formulation` selects the numerical formulation explicitly.
The benchmark builders select `b1_finite_volume` or `b2_finite_volume`; generic
builders default to `stokes_projection`. Renaming a case does not alter its
equations. B1 requires pipe geometry and B2 a layered duct; the formulation
does not relax their documented boundary, derivative or sharding restrictions.

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
continuity adjoint relation, finite eager/compiled reverse derivatives, transpose
duality, and directional differences down to $10^{-7}$. The limited-linear
reconstruction guards the inactive quotient before division, following the
[JAX guidance on reverse-mode NaNs](https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where).
Its switching rule and primal weights are unchanged; the manufactured nonzero
flow also checks reverse derivatives against directional differences.
The viscous-stress output has an identity optimization barrier so compiled
constant-cotangent transposes agree with eager evaluation across supported JAX
versions; it does not change the stress or introduce a custom derivative.
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

## Fully developed design: eliminating the drive

At fixed field, materials and geometry the fully developed inductionless problem
is linear in the drive, so the volumetric flow rate is $Q=Gf$ for a single
response $G$ that one solve measures. `lmx.design` uses that directly: the drive
delivering a requested throughput is $f=Q_{\rm target}/G$ exactly, and its
derivative is $df/dQ=1/G$.

The point is not economy of solves alone. An optimizer asked to find the drive
would return a scalar good only to its own tolerance, and would spend its budget
rediscovering a linear relation instead of exploring the inputs that genuinely
change the flow: wall conductance, aspect ratio and field strength. The tests
check the analytic derivative against automatic differentiation of the solve, so
the elimination is verified rather than assumed.

For a duct driven by a uniform pressure gradient the drop over a length $L$ is
$fL$ and the hydraulic power is $fLQ$. That is the isothermal power of a fully
developed segment. It excludes entry and exit losses, manifolds and all thermal
effects, so it is not a blanket pumping budget.

## Staggered grid and wall-resolving coordinates

`lmx.grid` supplies the geometry the plan's conservative 3-D core is being built
on. A `Grid` stores strictly increasing face coordinates per axis as host-side
metadata: it is hashable and never traced, so stencil bookkeeping and any
eigendecomposition happen once at trace time. A `Field` pairs a traced array with
its staggered offset, where `CENTER` places the value at the cell centre along an
axis and `FACE` places it on the lower face; a face field therefore carries one
extra entry on that axis. This is the marker-and-cell layout, for which the
normal velocity already lives where a conservative face flux needs it.

High-Hartmann ducts require the mesh to resolve two very different layers: the
Hartmann layer scales as $a/Ha$ and the side layer as $a/\sqrt{Ha}$. Three
coordinate families are available. `uniform_faces` is the unstretched control,
`geometric_faces` grows successive cells by a fixed ratio, and `tanh_faces`
clusters symmetrically at both walls. `wall_resolving_faces` inverts the
requirement directly: given a layer thickness it places a requested number of
cells inside the layer while bounding the growth ratio, and raises when the cell
count cannot meet the request rather than returning an unresolved mesh.

`lmx.bc` expresses a wall condition once, as the ghost value that reproduces it,
and `lmx.ops` differences every face with the same expression. A cell-centred
value sits half a cell from the wall, so a prescribed value $g$ needs
$p_{\rm ghost}=2g-p_0$ and a prescribed normal derivative $q$ needs
$p_{\rm ghost}=p_0\mp q\,\Delta x_0$.

`face_gradient` maps a cell field to the faces normal to one axis and
`divergence` maps three face fields back to cells as the net flux per unit
volume. Under the cell volumes and the face weights $A_f d_f$ these are exact
discrete adjoints,

$$
\langle p,\nabla\!\cdot\mathbf u\rangle_V=-\langle\mathbf u,\nabla p\rangle_{Ad},
$$

for a wall-impermeable flux; the test suite checks this to a relative 1e-14 on a
stretched mesh. That identity is what makes a projection idempotent and stops the
Lorentz force doing spurious work in the core of a high-Hartmann duct.

Two accuracy properties are deliberate and pinned by test rather than left
implicit. The wall flux is the two-point difference $(p_0-g)/(\Delta x_0/2)$,
first order at the wall, because that stencil is what keeps the assembled
Laplacian symmetric and the fluxes conservative. On a stretched mesh the interior
two-point gradient is centred between cell centres rather than on the face, so
its truncation error is first order in the spacing change; solution order there
is a manufactured-solution question and is verified in the step that owns it.

`lmx.poisson` inverts that Laplacian directly. On a tensor-product grid the
operator is the Kronecker sum of three one-dimensional operators, each symmetric
once the cell widths are folded in, so diagonalizing them on the host reduces a
solve to three tensor contractions and one elementwise divide. The one-dimensional
operators are read out of `lmx.ops` by applying the assembled Laplacian to unit
vectors, so the factorization cannot drift away from the stencil the rest of the
code uses.

The consequences matter for this code in particular: the cost does not grow with
the Hartmann number the way an iteration count does, the answer is exact to
round-off instead of to a tolerance, and the solve is a linear map, so it
differentiates without taping any iteration. A pure Neumann or fully periodic
problem is singular; the constant is removed from the right-hand side and the
returned field has zero volume-weighted mean. Inhomogeneous boundary data is
affine rather than linear and belongs in the right-hand side, so a condition
carrying a value is refused instead of silently linearized. A guard rejects any
axis operator that is not symmetric under the cell widths, which is the tripwire
that a future three-point wall stencil would trip.

`lmx.em` builds the electric coupling on those operators, following Ni et al.
Its rule is that one face-normal current

$$
J_{n,f}=\sigma_f\left[-\frac{\phi_N-\phi_P}{d_{PN}}
+(\mathbf u_f\times\mathbf B_f)\cdot\mathbf n_f\right]
$$

is the single source of truth: the potential equation is the divergence of
exactly that flux and the Lorentz force is rebuilt from the same numbers. The
reason is quantitative. In the core the momentum balance is
$-\nabla p+\mathbf J\times\mathbf B=0$ to $O(Ha^{-2})$, so an $O(\Delta)$
inconsistency between the two parts of $\mathbf J$ is amplified by $Ha^2$ and
appears as a spurious core current.

The force uses Ni's face form,

$$
(\mathbf J\times\mathbf B)_c=\frac{1}{\Omega_c}\sum_f J_{n,f}\,s_f\,
(\mathbf r_f-\mathbf r_c)\times\mathbf B_f,
$$

which never forms a cell-centred current vector and samples the magnetic field on
the faces, so it remains correct where the field varies along the duct. Face
conductivity is the distance-weighted harmonic mean, the series resistance of the
two half-cells and therefore the right average across a fluid-wall jump; the
arithmetic mean would let a poorly conducting wall draw too much current.

These modules supply geometry, operators, the scalar solve and the electric
coupling. The momentum discretization and the time loop follow in their own plan
steps, and the existing fully developed and extruded solvers continue to use
`lmx.mesh` until those land.

## The projection step and its two stiffnesses

`lmx.core3d` assembles one fractional step: the potential is solved, the face
currents of `lmx.em` give the Lorentz force, momentum advances, and a pressure
Poisson solve returns the velocity to the discretely divergence-free space.

The two stiffnesses are handled differently, on purpose. The velocity part of the
Lorentz force is a damping of rate $\sigma B^2/\rho$, which would force
$\Delta t\propto Ha^{-2}$ if left explicit. It is applied implicitly as a
correction to the conservative face force,

$$
\mathbf u^{*}=\mathbf u+\frac{\Delta t\,\mathbf r}{1+\Delta t\,\lambda},
$$

so the correction vanishes with the right-hand side and changes the path to a
steady state but not the steady state itself. Viscosity is left explicit and its
limit is reported by `ChannelProblem.diffusive_step_limit` rather than enforced,
so a caller sweeping a parameter sees the constraint instead of a silently
clipped step. The magnetic stiffness grows as $Ha^2$ and must go; the viscous one
depends only on the mesh and can stay until an implicit viscous solve lands.

Two constraints are imposed before any divergence is taken: a wall-normal
velocity is zero on its wall faces, and on a periodic axis the duplicated first
and last face are made equal. Without either the discrete divergence carries a
net boundary flux, and with every axis periodic or Neumann the pressure cannot
remove that constant, so the projection would return a field that is still not
divergence free.

Diffusion may be taken implicitly instead. `ChannelProblem.viscous_factorizations`
factorizes $(1+\Delta t\,\lambda)I-\Delta t\,\nu\nabla^2$ at each velocity
position, folding the magnetic damping into the shift, so one solve removes both
stiff terms and the step is bounded by accuracy rather than by the mesh. That
operator separates exactly as the pressure Laplacian does, on the free faces of
each axis: a walled face axis has its two boundary faces prescribed and a
periodic one carries a duplicate, and solving on anything else would either
invent a value for a prescribed face or treat one face as two unknowns.

The gain grows with refinement, because the explicit limit falls as the square of
the cell size. Integrating the plane channel to the same accuracy takes 29.0 s
explicitly and 1.9 s implicitly at 16 cells, and 95.1 s against 2.6 s at 32, a
**16-fold and then 36-fold** reduction for errors that agree to three digits.

Steady Stokes flow between plates has the exact profile $f(1-y^2)/(2\nu)$. The
step reproduces it and **converges at second order**, measured as 2.01 and 2.04
over 8, 16 and 32 cells across the channel. Convective transport is omitted;
that is the Stokes limit, appropriate at blanket interaction parameters and
stated rather than implied.

## Running the step as one compiled trajectory

A Python loop around the projection step dispatches every operation from the
host. On an accelerator that is the difference between a queue the device can run
ahead on and a round trip per step, and it is why the audit that opened this plan
found one GPU slower than a laptop CPU on a small duct. `lmx.timeloop` compiles
the whole run with `jax.lax.scan` instead. Measured on this laptop's CPU, 200
steps of a 4x16x16 duct take 3.58 s through the host loop and 0.44 s through the
scan, an **8x speedup before any accelerator is involved**; the 32-cell case gives
the same ratio, because what is removed is per-step dispatch rather than
arithmetic.

The step body is wrapped in `jax.checkpoint`, so reverse mode keeps one state per
step and recomputes each step's interior rather than storing every intermediate.
Diagnostics leave as scan outputs, so a run reports its divergence residual and
kinetic-energy history without synchronising mid-trajectory. A test greps the
step and loop sources for `float(`, `bool(`, `device_get` and `.item()`: one of
those inside the loop would serialise the queue and undo the change.

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
