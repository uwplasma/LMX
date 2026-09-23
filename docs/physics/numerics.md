# Numerical methods

LMhdX uses cell-centered structured finite-volume/finite-difference operators.
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
retaining LMhdX's MHD-specific residual and boundary contracts.

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
response $G$ that one solve measures. `lmhdx.design` uses that directly: the drive
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

`lmhdx.grid` supplies the geometry the plan's conservative 3-D core is being built
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

`lmhdx.bc` expresses a wall condition once, as the ghost value that reproduces it,
and `lmhdx.ops` differences every face with the same expression. A cell-centred
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

`lmhdx.poisson` inverts that Laplacian directly. On a tensor-product grid the
operator is the Kronecker sum of three one-dimensional operators, each symmetric
once the cell widths are folded in, so diagonalizing them on the host reduces a
solve to three tensor contractions and one elementwise divide. The one-dimensional
operators are read out of `lmhdx.ops` by applying the assembled Laplacian to unit
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

`lmhdx.em` builds the electric coupling on those operators, following Ni et al.
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

The electromotive force and the Lorentz force travel the same interpolation path
in opposite directions. A velocity component is averaged from its faces to the
cell centres and carried to the current faces; the force is averaged from the
current faces to the cell centres, which is Ni's face form above, and carried to
the velocity faces. Every cell-to-face step is `lmhdx.ops.face_average`, the
average of the piecewise-constant cell field over the control volume straddling
the face, $(h_L c_L+h_R c_R)/(h_L+h_R)$, and every face-to-cell step is its
transpose `lmhdx.ops.face_average_adjoint` under the cell volumes and the face
weights $A_f d_f$, including the polar metric. The force map is then exactly
minus the adjoint of the electromotive map, the discrete form of
$\int\mathbf u\cdot(\mathbf J\times\mathbf B)=-\int\mathbf J\cdot(\mathbf u\times\mathbf B)$,
so the Lorentz force does exactly minus the Joule dissipation and the steady
Stokes operator is symmetric in the face-volume inner product on a stretched
mesh, to round-off. The distance-weighted interpolation is exact for a linear
field where the face average is not, but its transpose is not an average on a
stretched mesh: with it the steady operator was asymmetric by 1e-2 on a Ha 100
layer mesh and the ohmic identity was off by up to 3e-3. The two interpolations
coincide on uniform cells. On the layer meshes of the validation ladder the face
average moves each insulating duct flow rate towards the spectral reference, by
at most 0.3 % of it. The pipe solver of `lmhdx.pipe` uses the same pair, with the
polar rotation of the field taken at the cell centres.

A thin conducting wall of conductance ratio $c=\sigma_w t_w/(\sigma a)$ is a
sheet with a potential $\varphi_w$ of its own (Walker's condition). The fluid
reaches it across the half cell against the wall,
$J_n=\sigma(\varphi_P-\varphi_w)/(h_P/2)$, and the sheet carries that current
along itself, $J_n=-c\,\sigma\nabla_\tau^2\varphi_w$, with the five-point surface
Laplacian in the metric of the wall. On the wall-normal axis the sheet is one
more node, weighted by $c$ times the wall area where a cell is weighted by its
volume, so the tangential operators act on it as on a cell and the potential
operator stays a Kronecker sum. The solve is therefore the same three
contractions as for an insulating wall (`lmhdx.poisson.fast_diagonal_thin_wall_poisson`
and the `wall_conductance` option of the polar factorization), with no inner
Krylov iteration, and it is symmetric in the cell volumes. Taking the adjacent
cell value as the wall potential, as before, was first order: a manufactured
wall potential now converges at second order, the ohmic identity closes to
round-off through the wall, and the conducting pipe reaches second order
under joint refinement. Where two conducting walls meet, the corner node
joins the sheets in series, so one delivers what the other receives, and a
rank-four Woodbury correction per mode of the third axis removes the conduction
the Kronecker sum would otherwise give the corner edge. The half-cell current
into a sheet carries no electromotive force, so it exerts no Lorentz force
either: `lmhdx.core3d.electric_state` closes the wall faces before forming the
force. Its product with a field tangential to the wall would be work the Joule
dissipation never sees; with conducting side walls in a uniform field that left
the steady operator asymmetric by 1.1e-3 at Ha 20, and closing the faces moves
the flow rate by at most 2.2e-4 of itself on 24 and 48 cells.

The imposed field may vary in space. `ChannelProblem.magnetic_field` takes
three numbers, or an `lmhdx.core3d.ImposedField` of cell-centred arrays (three
arrays become one); both paths carry each component to the current face with the
same interpolation and multiply it there, so the adjoint pairing above holds for
any field, and a constant field given as arrays reproduces the three numbers bit
for bit. `lmhdx.core3d.fringe_field` builds the fringe of the ANL benchmark (square
duct, $x_0=3$): with $s=x-x_c$ and $k=\pi/(2x_0)$, the midplane profile
$B_y=\tfrac12 B_0[1-\sin(ks)]$ over $|s|\le x_0$, and, as Votyakov et al. (2009)
ask of a fringe model, the companion that makes it divergence and curl free,
$B_y+iB_x=\tfrac12 B_0[1-\sin k(s+iy)]$:

$$
B_x=-\tfrac12 B_0\cos(ks)\sinh(ky),\qquad
B_y=\tfrac12 B_0\left[1-\sin(ks)\cosh(ky)\right].
$$

The profile is not analytic at $s=\pm x_0$, so no harmonic field keeps it on the
whole midplane: $B_x$ vanishes there, which keeps the divergence continuous, and
$B_y$ jumps by $\tfrac12 B_0(\cosh ky-1)$, 7.0 % of $B_0$ at $|y|=1$. Both
components come from one flux function, $B_x=\partial_y A$ and $B_y=-\partial_x A$;
each face-normal value is the mean of the field over its face, a difference of
$A$, so the discrete divergence of the faces cancels to round-off (0 on uniform
cells, 2.6e-16 on a tanh mesh), and the cells take the average of their two
faces.

These modules supply geometry, operators, the scalar solve and the electric
coupling. The momentum discretization and the time loop follow in their own plan
steps, and the existing fully developed and extruded solvers continue to use
`lmhdx.mesh` until those land.

## The projection step and its two stiffnesses

`lmhdx.core3d` assembles one fractional step: the potential is solved, the face
currents of `lmhdx.em` give the Lorentz force, momentum advances, and a pressure
Poisson solve returns the velocity to the discretely divergence-free space.

The two stiffnesses are handled differently, on purpose. The velocity part of the
Lorentz force is a damping of rate $\sigma B^2/\rho$, which would force
$\Delta t\propto Ha^{-2}$ if left explicit. It is applied implicitly as a
correction to the conservative face force,

$$
\mathbf u^{*}=\mathbf u+\frac{\Delta t\,\mathbf r}{1+\Delta t\,\lambda},
$$

so the correction vanishes with the right-hand side and changes the path to a
steady state but not the steady state itself. The fast solve needs one shift per
component, so a varying field takes the largest rate over the cells; in the
diagonal model a cell of rate $r$ then updates by $1-\Delta t\,r/(1+\Delta t\,\lambda)$,
inside $(0,1]$ wherever $r\le\lambda$, where a smaller shift such as the volume
mean would overshoot past $-1$ once $\Delta t\,r>2(1+\Delta t\,\lambda)$. The
steady preconditioner of `lmhdx.steady` takes the peak $|\mathbf B|^2$ for the same
reason. Viscosity is left explicit and its
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

The steady residual of `lmhdx.steady` projects twice. One projection's pressure
comes from the fast diagonal solve, exact only to round-off, and on a stretched
mesh that round-off leaves about $10^{-13}$ of the removed gradient outside the
divergence-free fields. The conjugate-gradient preconditioner ends in a
projection, so it cannot see that part of a residual, and CG stalls on it. A
uniform duct's force is nearly solenoidal and the leak does not matter; a varying
field's force has a large gradient part, and on the four-cell varying duct of the
tests the floor was $5.0\times10^{-10}$ of the right-hand side at Ha 20 and
$6.4\times10^{-7}$ at Ha 100. The second projection is one defect correction of
the pressure solve: the floors fall to $1.2\times10^{-13}$ and $3\times10^{-12}$,
and the certificate reads the same field CG converged. Every varying-field duct
measured, the ANL fringe included, certifies at the default tolerance $10^{-9}$
through Ha 300, with iteration counts on uniform ducts unchanged.

Above Ha 300 a floor remains. On the fringe mesh of the tests (24 cells, 16 axial)
CG stalls at $1.5\times10^{-10}$ of its right-hand side at Ha 300,
$7.5\times10^{-10}$ at Ha 600 and $1.5\times10^{-8}$ at Ha 1000, where round-off
in the potential solve leaves the operator asymmetric by $6\times10^{-7}$. Steps
1.9b–1.9d therefore take a tolerance of $10^{-9}$ through Ha 300, $10^{-8}$ to
Ha 600 and $10^{-7}$ to Ha 1000, with `linear_max_restarts=600` (36,000 CG
iterations) above Ha 300: the three solves take 5,581, 12,056 and 19,299.
`test_the_fringe_certifies_at_the_tolerance_rule` holds the fringe to that rule.

## Running the step as one compiled trajectory

A Python loop around the projection step dispatches every operation from the
host. On an accelerator that is the difference between a queue the device can run
ahead on and a round trip per step, and it is why the audit that opened this plan
found one GPU slower than a laptop CPU on a small duct. `lmhdx.timeloop` compiles
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

## LMhdX and SOLVAX

LMhdX owns:

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

LMhdX calls these algorithms with MHD-specific operator actions and then certifies
the returned state in physical units. This keeps solver policy reusable without
moving geometry or physics into SOLVAX.

## Q2D spectral evolution

The periodic Q2D path uses full complex Fourier transforms, the two-thirds
dealiasing rule for the vorticity-advection product, and fourth-order
integrating-factor Runge--Kutta time stepping. Viscous and Hartmann-friction
terms are integrated exactly within each step. SOLVAX supplies the reusable
periodic Poisson symbol and zero-mean spectral inversion for the streamfunction;
LMhdX owns vorticity dynamics, velocity reconstruction, the energy identity, and
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
