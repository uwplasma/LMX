# Numerics and Implementation

This page describes how the current LMX solver families are assembled in code.
The emphasis is on the equations actually implemented today, the discrete
operators used by the code, and where each step lives in the source tree.

## Solver families

LMX currently exposes two solver-family names:

- `fully_developed_inductionless`
  - the default research solver for duct benchmarks
- `extruded_inductionless`
  - first low-Re 3D pressure-velocity-potential slices for rectangular ducts,
    layered ducts, and mapped pipes now exist in `lmx/fringing.py`
  - full production 3D family hardening remains post-`1.0`

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
- kinematic viscosity `nu`
- fluid/solid masks
- constant, analytic, and ramped magnetic-field components

The public `RegionSpec.viscosity` field is retained for compatibility, but its
meaning is kinematic viscosity in `m^2/s`. Dynamic viscosity from a material
table must be converted with `lmx.dynamic_to_kinematic_viscosity(mu, rho)`
before building a case. The fully developed and extruded operators use this
field directly as the diffusion coefficient and divide pressure, body forcing,
and Lorentz terms by density.

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

The default coupling remains unaccelerated for backward compatibility. Set
`solver.coupling_acceleration = "aitken"` or `"anderson"` to enable a released
SOLVAX accelerator. The compatible SOLVAX scalar Aitken update is useful for a
single dominant slow mode; bounded-memory Anderson mixing is the production
choice for the multimode high-Hartmann-number campaign. LMX checks the
**unaccelerated** residual `max(abs(G(u)-u))` and the change across outer steady
states, so neither relaxation nor a converged inner subsystem can manufacture
a false steady pass. Tolerance-controlled potential CG reuses the previous
potential; fixed-iteration legacy solves retain their zero start. Accepted
gradients still require implicit root differentiation, not differentiation
through the iteration count.

On confirmation-sized stretched cross-sections (at least 110 cells in each
direction), volume-scaled potential PCG uses a symmetric additive y+z line
preconditioner built from SOLVAX batched tridiagonal solves and its released
additive combinator. LMX retains gauge projection and line geometry. When the
scaled diagonal range exceeds `3e4`, it adds a balanced Galerkin coarse correction:
linear prolongation and its exact transpose define the coarse operator, a
small Cholesky solve removes the global modes, and SOLVAX
`galerkin_deflation` applies the balanced correction. The line
solve follows the dominant coupling direction (or uses both directions when
neither dominates), and the balanced composition remains suitable for PCG.
The preconditioner is built once per coupled step and reused by every
potential solve. Smaller, less stretched meshes retain cheap Jacobi. The
line-only crossover is backed by the Samper `Ha=10000` audit; the coarse
correction addresses the separately recorded `Ha=15000`, 99 x 99 conditioning
boundary and must pass that row before it becomes accepted benchmark evidence.

Uniform-conductivity tensor-product ducts use a stronger specialization based
on the classic [tensor-product direct method of Lynch, Rice, and
Thomas](https://eudml.org/doc/131604). After
volume scaling, the potential operator is the generalized Kronecker sum
`Ty (x) Mz + My (x) Tz`. LMX diagonalizes only the mildly stretched transverse
problem and solves all strongly stretched Hartmann-direction modes with
SOLVAX's batched tridiagonal solver. Before applying this inverse it restores
the Poisson equation omitted by the single-cell gauge so that the modal
right-hand side has exactly zero sum; the result is regauged afterward. This
compatibility projection is essential—without it, a small residual can hide a
large null-mode solution error. SOLVAX flexible GMRES wraps the modal inverse
because the anchored gauge is not strictly symmetric. Dense-solve regression
tests require relative solution agreement, not residual reduction alone.

The high-Ha literature runner also scales its absolute steady-update target by
the analytical flow rate, with a `1e-9` ceiling. A fixed `1e-9` update is a
large relative velocity error when `Q` is `O(1/Ha)` and was shown to admit a
false mechanical-balance pass/fail boundary at `Ha=15000`.

For `Ha >= 100`, the rectangular fluid mesh uses a symmetric geometric
progression from each wall to the center. It allocates the requested cells
inside the physical Hartmann/side layer while keeping a single bounded
adjacent-cell ratio. The earlier piecewise wall/expansion/core mesh was removed
because its spacing jump immediately outside the Hartmann layer polluted the
high-Ha integral-flow benchmark.

The implicit momentum solve treats the local magnetic damping term
$(\sigma/\rho)(B_y^2 + B_z^2)u$ as a reaction coefficient on the velocity
matrix. The right-hand side therefore adds back only the corresponding
linearized contribution from the explicit Lorentz assembly. This split keeps
the steady Hartmann/Shercliff/Hunt operator from double-counting the
streamwise Lorentz damping while still retaining the potential-coupled current
terms in the conservative face-current reconstruction.

The transient mass term dominates the canonical B2 momentum matrix. Its
SOLVAX GMRES solve therefore uses diagonal scaling: dense-reference and
implicit-gradient gates preserve the solution, while avoiding repeated GPU
line solves. Pressure and electric operators retain their stronger structured
preconditioners because profiling still shows them to be conditioning-limited.

For `inlet_flow_rate` / constant-`Q` fully developed runs, the solver treats
the requested mean velocity as a hard area-weighted constraint after the
bounded velocity update and wall interpolation. This keeps direct-wall
reconstruction intact while preventing the limiter from silently shifting the
reported volumetric flow rate away from the requested value. The pressure /
forcing proxy is still reported separately and remains part of the external
parity hardening lane.

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

The interface residual is the maximum finite-volume current imbalance in cells
adjacent to a conductivity or fluid/solid interface, multiplied by the local
characteristic cell length. It therefore has the same current-density units as
a face flux. It is not a comparison between a shared conservative face flux
and a smoothed cell-centered current; that older definition measured legitimate
interface gradients and could not serve as a continuity error.

The fully developed solver applies a compatibility projection to the potential
right-hand side before solving, so that the discrete Poisson problem satisfies
the zero-net-source condition required for a bounded potential solve.

This logic lives in `lmx/solvers.py`, while the diagnostics are stored in
`lmx/core.py` and persisted through `lmx/io.py`.

`fully_developed_power_balance(case, solution)` adds a final dimensional audit
in W/m. It reports pressure work, Lorentz work, viscous dissipation, and Joule
dissipation. Joule power is assembled from the same conservative face-current
network as the potential equation, including explicit wall regions. The output
separates three residuals:

- network Joule minus `J·(u×B)`, which tests the electric solve;
- Lorentz work plus electromagnetic transfer, which tests face-to-cell force
  reconstruction;
- pressure plus Lorentz minus viscous power, which tests the steady momentum
  solve.

Keeping these residuals separate prevents a cancellation between two numerical
errors from masquerading as a closed total power budget.

The `extruded_inductionless` research slice in `lmx/fringing.py` extends this
to explicit boundary-flux audits. In addition to `div J`, it now records:

- stationwise integrated axial current
- stationwise wall-current leakage on the external `y`/`z` boundaries
- a net boundary-current residual over the full 3D control volume

Those are the right diagnostics for hardening inlet/outlet behavior and for
checking that the 3D pressure-velocity-potential loop is not quietly creating
or destroying charge through the axial boundaries.

The manual validation lane now treats those quantities as hard gates. The main
driver `scripts/run_manual_solver_family_validation.py` can reject a run when:

- `charge_balance_residual > --max-charge-balance`
- `interface_current_residual > --max-interface-current`
- `max_wall_current_leakage > --max-fringing-wall-current-leakage`
- `net_boundary_current_residual > --max-fringing-boundary-current`

That moves the conservation checks from “reported” to “enforced” for the heavy
validation path.

For rectangular and layered extruded runs, the conservation audit is
face-conservative rather than cell-gradient-based. The electric source term
`∇·(σ u×B)`, the post-solve `div J` diagnostic, and the boundary-current
integrals are all assembled from the same face conductances. This matters at
conductivity jumps: a cell-centered gradient check can report large spurious
`div J` across fluid/solid interfaces even when the face fluxes are locally
balanced, so the heavier validation gate uses the conservative face form
instead. For layered 3D cases, the electric potential subproblem is also solved
with a sparse direct variable-coefficient solve, which is what brings the
layered fringing datasets inside the same conservation gate as the rectangular
and mapped-pipe cases.

## Extruded 3D rectangular/pipe numerics

The implemented `extruded_inductionless` path in `lmx/fringing.py` uses:

1. a structured axial mesh in `x`
2. either a rectangular `y-z` cross-section or a mapped `r-\theta` pipe grid
3. a low-Re projection update for `u`, `v`, `w`, and `p`
4. a Poisson-like electric solve for `\phi`
5. conservative current reconstruction
6. post-step conservation audits

For mapped pipes, the same discrete logic is expressed in local cylindrical
coordinates. The local magnetic field and current components are projected
between Cartesian and pipe-local frames only at the assembly boundaries; the
rest of the control-volume bookkeeping remains metric-aware inside
`lmx/fringing.py`.

The ALEX B1 pressure and electric PCG preconditioners augment the axial and
radial lines with SOLVAX's exact cyclic-tridiagonal solve in periodic `theta`.
Unlike the former circulant FFT shortcut, this remains correct when line
coefficients vary azimuthally. It is an additive preconditioner only; it does
not change the cylindrical finite-volume operator or its converged solution.
Generic mapped-pipe calls retain the axial/radial preconditioner unless the
periodic line is explicitly enabled.

The compatible B1 projection uses a bounded physical-convergence pilot: one
GMRES restart cycle is tested against the mean-free face-flux divergence and
normalized fixed-flow residual. A passing state is accepted at that physical
tolerance; otherwise the original tight algebraic solve continues from the
pilot state. This avoids oversolving easy projections without weakening the
fallback or conservation criteria.

The ALEX B2 square-duct path uses the actual nonuniform `dy` and `dz` arrays for
masked viscous diffusion, conservative electric current, and a compatible
face-flux pressure projection. Its Neumann pressure and electric-potential
systems are multiplied by cell volume to form Euclidean-symmetric operators;
a rank-one term fixes the constant gauge, and released SOLVAX implicit PCG
solves the resulting systems. This construction supports JIT execution and
implicit derivatives with respect to the right-hand side and variable
coefficient. The generic low-`Ha` rectangular path retains its characterized
collocated update until equivalence evidence justifies changing it.

### Mapped-pipe electric-potential sign convention

The mapped-pipe sparse electric operator represents

$$
-\nabla\cdot\left(\sigma\nabla\phi\right),
$$

while the EMF source helper assembles

$$
\nabla\cdot\left(\sigma(\mathbf{u}\times\mathbf{B})\right).
$$

Because the inductionless current is

$$
\mathbf{J} = \sigma\left(-\nabla\phi + \mathbf{u}\times\mathbf{B}\right),
$$

local charge closure requires

$$
-\nabla\cdot\left(\sigma\nabla\phi\right)
= -\nabla\cdot\left(\sigma(\mathbf{u}\times\mathbf{B})\right)
$$

for this discrete operator sign convention. The bent-pipe low-De charge
closure now uses that sign and verifies it with a direct conservative-flux
regression test. This changed the public bent-pipe local charge residual from
`O(1e-2)` to `2.16e-12`; see the [validation report](validation_report.md).

## Linear solver backends

Low-level iterative linear solver helpers are implemented in `lmx/linear.py`.
The current public linear-solver options are:

- `auto` (released SOLVAX PCG)
- `solvax_pcg`
- `cg` (compatibility alias for SOLVAX PCG)
- finite-iteration `jacobi` or volume-scaled SOLVAX PCG (`cg_volume`) for
  selected potential solves

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
