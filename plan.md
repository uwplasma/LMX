# LMX research and engineering plan

Status: active · Revised: 2026-09-05
Audited source: `d85fb5b73931688deb5161b0c049346c868935b9`

This is the single forward roadmap, acceptance register and compact work log.
It supersedes the old six-gate finish queue. Trustworthy physics remains the
critical path, but field integration, profiling and documentation need not
wait for B2 convergence. Detailed past investigations remain in the
[immutable prior plan](https://github.com/uwplasma/LMX/blob/d85fb5b73931688deb5161b0c049346c868935b9/plan.md)
and linked PRs, not another tracked archive. User-facing documentation and
source comments describe current capabilities and limits; history belongs
here or in Git.

## 1. Destination and scope

LMX should be a small, accessible, differentiable liquid-metal MHD research
package for hydraulic and thermal design of blanket channels in stellarators,
tokamaks and mirrors. Fields come from VMEX equilibria and ESSOS coils.
CPU/GPU execution, low memory, useful parallel scaling and independently
verified derivatives are requirements, not optional performance decorations.

Three achievements must not be conflated:

1. **Hydraulic screening:** verified inductionless isothermal pressure/flow,
   electrical closure, wall-current and magnetic-drag calculations.
2. **Thermomagnetic channel optimization:** verified heat transport, conjugate
   walls, buoyancy where relevant, physical pump work, realistic fields,
   admissible geometry and checked end-to-end sensitivities.
3. **Integrated blanket design:** coupling channels to independently supplied
   heat/neutron loads, engineering/material limits, shielding, breeding and
   structural assessments. LMX does not become a neutronics, equilibrium,
   coil-design, CAD or structural solver.

A channel demonstration is not an optimized complete blanket. Field-amplitude
derivatives are not coil/equilibrium/shape derivatives. A finite trajectory is
not a converged steady solution. Sharded arrays do not establish strong scaling.

Retain 3D/fringing, conducting walls, pipes and rectangular geometries, Q2D,
restart/output and external validation. Remove duplicated mechanisms and
unproved alternatives—not capabilities needed for these objectives. Add
physics through verified equations and a small composable API, not parallel
configuration frameworks.

## 2. Review findings and established evidence

The September review inspected package modules, test/validation and example
inventory, workflows, documentation, commits/PRs through #53, relevant SOLVAX
algorithms, local FreeMHD source and VMEX/ESSOS field interfaces. Primary
literature, accessible manuscript sections, publisher abstracts/book contents
and official software documentation informed the priorities. This is a focused
research review, not a claim to have read every publication or every page of
paywalled books.

### Fresh evidence at the audited source

| Item | Measured result | What it establishes |
|---|---|---|
| Portable tests | 501 passed; pytest 179.40 s, approximately 181.5 s end-to-end | Existing regression contracts pass |
| Combined coverage | 95.22% | Aggregate line/branch gate, not physical completeness |
| Architecture/import audit | Passed; 15 modules, 14,793 raw package lines, 6,347 core lines, 28 root exports | Existing enforced budgets pass |
| Inventory | 87 tracked files; 13 test files / 11,950 test lines; six Python examples and one TOML example | Consolidation baseline |
| Normal network clone | 3,648 KiB including checkout and Git; Git alone 1,644 KiB | Below 10 MB without shallow cloning |
| Documentation | Warning-as-error HTML build passed; 18 source documents | Build validity, not snippet/scientific correctness |
| Open PRs | None at review | Work through #53 merged |
| Hosted numerical CI | Latest run failed before any steps: account billing/spending limit | Infrastructure blocker, not an observed numerical failure |
| Office hardware | Accessible; Xeon W-2295 and two 16 GiB RTX A4000s | Accelerator access available |

Local: Apple M3 Max, Python 3.11.14, JAX 0.9.2, SOLVAX 0.20.0.
Office: Python 3.12, JAX 0.11.1, SOLVAX 0.20.0, driver 580.173.02.
SOLVAX was supplied through an isolated audit installation, not by changing
the shared GPU environment. Its linear/root, fixed-point, Schur and nonlinear
least-squares contracts were inspected; its complete suite/coverage was **not**
recertified by this LMX audit.

Accumulated developer `.git` objects/worktrees are not the ordinary-clone
metric. No history rewrite is currently necessary.

### Findings that change the priorities

| ID | Finding and source | Required response |
|---|---|---|
| F1 | `_generic_duct_step` and `_generic_pipe_step` omit convective momentum transport; the equations page shows full Navier–Stokes | Label Stokes-like recurrence correctly; verify inertia before general 3D claims |
| F2 | `extruded_engineering_objectives` uses end-pressure difference times flow without prescribed body-drive work | Separate pressure taps from physical pump work and verify the mechanical/electrical energy identity |
| F3 | `_solve_extruded_projection` chooses physical algorithms by case-name prefix | Explicit model/BC selection; names must be metadata |
| F4 | Generic layered momentum uses uniform spacing while electric closure uses actual widths; pipe diffusion applies scalar cylindrical Laplacians componentwise | Audit nonuniform and vector-metric/cross terms using continuous manufactured solutions |
| F5 | Traced generic evolution freezes imposed-field samples; geometry scaling does not resample live global fields | Differentiate field location, basis, geometry and coil/equilibrium parameters together |
| F6 | Tabulation converts through NumPy, silently extrapolates and reconstructs uniform axial stations | Traceable in-domain SI field protocol; use actual mesh coordinates |
| F7 | Generic recurrence clips velocity, adjusts stationwise flow, and uses a compact Poisson stencil inconsistent with its centered divergence/gradient composition | Audit physical consistency and energy; require one compatible coupled residual |
| F8 | No accepted terminal B2 steady primal/adjoint | Certify the physical residual, not a small iteration update |
| F9 | Q2D energy defect is reported, not the advertised terminal gate; mixed weak/strong dtypes can break checkpointed evolution | Explicit completion/acceptance and dtype policies, with workflow tests |
| F10 | Specialized B1 sharding is unsupported; Q2D has no dedicated spatial-shard path; generic two-GPU runs slow down | Close documented restrictions and measure useful scaling |
| F11 | README gradient uses 41 coefficients with a differently sized problem; CONTRIBUTING benchmark link is missing | Repair and execute public entry points |
| F12 | Several 400–800-line solver functions mix equations, iteration, diagnostics and dispatch | Consolidate operator/state contracts; transfer independent algebra to SOLVAX |
| F13 | File-based CI shards concentrate compilation cost; `.github/` changes escape numerical scope selection | Cost-aware grouping and independent workflow/selector checks |

F2 reproducer: unmagnetized generic square duct, four axial stations, 4×4
cross-section, length 2, four steps. The reducer reports flow 0.003952255488,
pressure drop 0 and pumping power 0; prescribed body-drive work is
0.007904510976 in the case's units. This finite-state definition check is
**not** steady validation. Account separately for energy storage, boundary
and body-force work; do not double-count equivalent imposed pressure forcing.

F9 reproducer: a default Q2D initial state combined with explicitly float64
viscosity/friction changes a complex64 loop carry to complex128. Casting the
initial state to float64 allowed the audit profile; this is not a released fix.
Audit import-time global x64 configuration as part of the dtype contract.

Regression parity and discrete manufactured forcing can reproduce the same
incorrect equation twice. Coverage does not establish continuum consistency,
mesh independence, external validity or useful sensitivities. Require
independent mathematics, continuous MMS, physical balances, published
observables with uncertainty and accurate time-to-solution measurements.

## 3. Performance baseline

The review captured cold lowering/compilation, synchronized warm runs,
compiler memory, cProfile, HLO and selected CPU/GPU traces. These are
**diagnostic profiles**, not completed production qualification.
The generic duct uses Ha=2, four steps, 12 electric iterations, two coupling
iterations and a mean-squared-axial-velocity objective. Q2D uses 256², 32 steps
and mean squared vorticity. Five warm samples follow first execution; cold
profiles disable persistent compilation caching.

| Workload/backend | Warm primal, ms | Warm value-and-gradient, ms |
|---|---:|---:|
| Generic duct 16×12×12, local CPU | 11.73 | 22.61 |
| Same, two logical local CPU devices | 64.19 | 114.01 |
| Same, one A4000 | 46.44 | 82.86 |
| Same, two A4000s | 75.16 | 152.14 |
| Generic duct 64×32×32, one A4000 | 282.19 | 535.95 |
| Same, two A4000s | 439.61 | 817.61 |
| Q2D 256², local CPU | 194.54 | 617.46 |
| Q2D 256², office CPU | 295.82 | 1,256.86 |
| Q2D 256², one A4000 | 49.94 | 146.37 |
| Fully developed Hartmann 32², local CPU | 6.81 | 14.71 |

Office Q2D uses matched JAX/SOLVAX versions: approximately 5.9× forward and
8.6× gradient GPU speedup for this workload. Cross-machine local-CPU/GPU
numbers also differ in JAX version and are not controlled hardware comparisons.

The larger duct achieves only 0.64× primal / 0.66× gradient one-to-two-GPU
speedup, approximately 32–33% efficiency. Full fields are actually partitioned.
One-/two-GPU sampled fields agree below 2e-16 maximum absolute difference;
the larger case's near-zero gradient differs by approximately 1.2e-8 relative
and 3e-20 absolute. Use mixed tolerances. Office Q2D CPU/GPU fields agree below
7e-16. Parity is promising; efficient strong scaling is not established.

Compiler temporary estimates for the larger duct gradient are 147.0 MB on
one GPU and 81.2 MB per compiled shard on two. Q2D GPU gradient/primal
estimates are 164.2/11.5 MB. These are **not measured peak device memory**.

Trace findings identify concrete next actions:

- Small one-GPU duct: hundreds of tridiagonal kernels and thousands of
  copies/reductions; measure batching, factor reuse and iteration economy.
- Small two-GPU primal: 1,796 all-reduce, 1,898 send/receive and 398 all-gather
  kernel events across both GPUs. Gradient: 3,488, 3,802 and 920.
  These dynamic counts are not static HLO counts; overlapping durations must
  not be summed into wall time.
- Small duct gradient compilation: 9.75 s one GPU / 14.50 s two GPUs, plus
  approximately 6 s lowering. Specialization/tracing affects development speed.
- Instrumented two-step B2 remains `step_limit`: momentum defect 0.320154,
  mass approximately 1.4e-10, charge approximately 7.1e-9. Last warm runtimes:
  22.4 ms local CPU / 166.6 ms one GPU / 456.2 ms two GPUs. Much time is outside
  labelled numerical phases; investigate orchestration, diagnostics, launches
  and synchronization without attributing all untimed work to one cause.
  Instrumented timings are not throughput benchmarks.

Reports, arrays, traces, harnesses, coverage and JUnit are retained under
ignored `artifacts/review-20260904/`. M4 must fold useful harness logic into
the existing benchmark command and publish checksum-addressed evidence.
Local audit files are not a durable public reproducibility solution.

## 4. Milestones and immediate work order

Use these IDs in PRs. “Implemented” alone never closes a milestone.

| ID | Deliverable | Status/dependency | Exit gate |
|---|---|---|---|
| M0 | Truthful model contracts and physical objectives | In progress | F1–F3/F9/F11 scoped; executable entry points; no unsupported pump/steady claims |
| M1 | Conservative consistent 3D residual and B2 steady primal | Open; M0 definitions | Rank/MMS/physical convergence and frozen B2 refinement gates |
| M2 | Efficient implicit 3D derivatives | Open; accepted residual for B2 | Tangent/adjoint/Taylor, failure semantics and bounded-memory evidence |
| M3 | Independent B1/B2 and regime validation | B1 setup can start now | Matched inputs, independent runs, refinement and uncertainty |
| M4 | CPU/GPU kernels, profiling and sharding | Audit baseline done; optimization open | Correctness, runtime peak memory, time-to-accuracy and scaling |
| M5 | Differentiable VMEX/ESSOS field/geometry coupling | Contract work can start now | Live coil/equilibrium/shape derivatives and independent field tests |
| M6 | Thermomagnetic blanket-channel physics | Specification now; production after M1 | Heat/conjugate/buoyancy and geometry validation in declared regimes |
| M7 | Constrained device-design examples | Screening after M0/M2/M5; thermal after M6 | Held-out finer-grid/external checks and reproducible feasible designs |
| M8 | Student/research docs, examples and slim API | Continuous | Executable learning paths and complete equation/API/evidence links |
| M9 | Papers and reproducible releases | Staged | Every claim maps to a versioned evidence artifact and command |

1. **PR A — truthful contracts/objectives:** test F2, define pressure/work
   conventions, fix public claims/snippet shapes and Q2D acceptance/dtype
   contracts. Split small documentation repairs from equation changes as needed.
2. **PR B — one conservative residual:** independent unknowns/constraints,
   explicit model selection, reduced-rank and continuous MMS checks, then a
   safeguarded steady solve. No production B2 campaign before tiny-system proof.
3. **PR C — performance/evidence tooling:** extend the existing benchmark CLI
   for primal/gradient/compile/memory/sharding; fix CI workflow scope and
   balance by measured cost. This does not depend on B2 nonlinear convergence.

B1 comparator preparation, exterior-field contracts and documentation can
advance between numerical gates. Do not let one stalled solver experiment
block all useful work.

## 5. M0–M2: physical residual and efficient derivatives

For inductionless isothermal flow, define one residual for momentum, mass,
electric potential, wall interfaces and the selected drive. State explicitly
when inertia is omitted:

$$
\nabla\cdot u=0,\qquad
\rho(\partial_tu+u\cdot\nabla u)
=-\nabla p+\nabla\cdot(2\mu D(u))+J\times B+f,
\quad J=\sigma(-\nabla\phi+u\times B),\quad\nabla\cdot J=0.
$$

- Specify unknown locations, units, quadrature, orientations, boundary terms
  and block scaling. Verify divergence/gradient adjoint compatibility, gauges,
  conductor connectivity and global compatibility.
- Use fluid velocity/pressure and fluid/solid electric potential as minimal
  state. Retain face fluxes or a fixed-flow multiplier only if independently
  required. Eliminate solid velocities. Gauges/BCs replace redundant equations;
  they are not extra least-squares rows concealing a rank-deficient PDE.
- Distinguish prescribed pressure gradient, body force and fixed flow.
  Enforce flow without post-hoc stationwise correction. Gauge changes must
  not alter observables. Separate physical pump work from pressure-tap drop.
- Derive nonuniform Cartesian and cylindrical/mapped stress, pressure and
  electric operators, including vector cross terms and the axis. Test wall
  thickness/interface placement, positive mapping Jacobians and face measures.
- Add conservative advection with stable, documented discretization. Verify
  B=0 limits, spatial/temporal order and energy behavior. Retain Stokes reduction
  where nondimensional assumptions justify it.
- Use compatible current/Lorentz discretization. Expose boundary, viscous,
  Joule, storage and imposed-drive work. Temperature Joule sources must use
  the same current/conductivity.
- Treat clipping, masks and limiters as mathematics requiring tests. Reject
  invalid materials, folded meshes and failed solves explicitly.

B2's saved step-152 state has momentum defect 0.1378074003 despite update
0.00144850045 and smaller mass/charge defects. The momentum target remains
1e-3; use all other normalizations/tolerances in the frozen specification.
Update tolerance 5e-5 is not a substitute for physical convergence.
Local restart SHA-256:
`3aa1c4626889b056ef601867320d80c6b5a2eab5d3fe1ae1c3d4198a43dc23f5`.
It is a debugging checkpoint, not the sole reproducibility input.

Compare tiny scaled Jacobians against independent dense operators and inspect
rank/nullspaces. Production remains matrix-free. Use safeguarded Newton/Krylov
or another justified root method, physical admissibility and inexact-solve
controls. Judge progress by physical defect per second/memory at fixed accuracy.

Preserve lessons from [#48](https://github.com/uwplasma/LMX/pull/48),
[#49](https://github.com/uwplasma/LMX/pull/49),
[#51](https://github.com/uwplasma/LMX/pull/51) and
[#52](https://github.com/uwplasma/LMX/pull/52): no blind transient retuning,
predictors that shrink updates while worsening momentum, or costly full
momentum inversion inside every pressure iteration. Remove rejected prototypes
after recording evidence; do not keep experimental public lanes.

For accepted steady state $R(z,q)=0$ and scalar objective $F(z,q)$:

$$
R_z z_q=-R_q,\qquad R_z^\mathsf{T}\lambda=F_z^\mathsf{T},\qquad
dF/dq=F_q-\lambda^\mathsf{T}R_q.
$$

SOLVAX owns reusable linear/root solvers, globalization, preconditioners,
tangent/transpose wrappers and checkpoint schedules. LMX owns PDE assembly,
physical blocks, BCs, units and objectives. Reuse existing root/Schur contracts.
Least-squares stationarity does not automatically imply zero PDE residual.

Require JIT/JVP/VJP for each advertised algorithm; label reverse-only
exceptions. PCG requires proven symmetry/positivity. Nonsymmetric coupled
systems need appropriate Krylov and tested transpose/preconditioning.
Never mark an operator symmetric merely to reuse a faster solver.

Test every advertised continuous input: drive, viscosity, density, fluid/wall
conductivity, field, geometry and initial state where relevant. Discrete mesh
topology/counts and BC kinds remain static; absent derivatives are explicit.

- Steady/linear solves: no nonlinear/Krylov reverse tape. Test adjoint
  residual, tolerance sensitivity, failed-solve behavior and root independence
  from initial guess when the same root is reached.
- Finite trajectories: differentiate the declared discrete recurrence with
  selective checkpointing; never attach steady-root AD to an unconverged run.
- Use finite-difference step sweeps, Taylor remainder curves, JVP/VJP duality,
  dense tiny checks and analytic responses. Freeze case-specific tolerances;
  float64 duality near 1e-8 and second-order Taylor remainder are starting
  targets, not universal thresholds for ill-conditioned high-Ha systems.
- Measure runtime peak host/device memory against mesh/parameter count,
  nonlinear work and horizons 8, 32, 128, 512. Compiler estimates are not proof.
- SOLVAX algorithm merges require its complete passing tests, **above 95%**
  measured coverage, clear API/reference documentation and README usage.


## 6. M3: verification and validation matrix

Use one versioned benchmark schema and the existing validation runner.
Each case records equations, SI/nondimensional mapping, geometry, input-field
provenance, BCs, grid/time/solver controls, reference source, observables,
uncertainty, acceptance and artifact hashes. No new runner per paper.

| Layer | Required cases/observables | Evidence |
|---|---|---|
| Mathematics | Constants/linear fields, operator adjoints, gauges, rank, material jumps, interfaces, mapped metrics/axis | Independent identities, tiny dense systems and failure tests |
| Continuous verification | Velocity/pressure/potential MMS with convection, nonuniform grid and variable conductivity | Observed space/time order, not forcing generated solely with the tested discrete operator |
| Analytical MHD | B=0 Poiseuille, Hartmann, Shercliff, Hunt, independently specified conducting pipe | Full profiles, flow/pressure/current/work identities across Ha, aspect ratio and conductance |
| High-Ha resolution | Core, Hartmann/side layers, wall thickness and inlet/outlet extent | At least three meaningful refinements; axial/layer/wall errors separated |
| ALEX B1 | Pipe: Ha=6600, N=10700, wall conductance 0.027 | Frozen geometry/field/drive, pressure taps/flow, independent solver and experiment uncertainty |
| ALEX B2 | Square duct: Ha=2900, N=540, wall conductance 0.07 | Same provenance; momentum, mass, charge and work gates before pressure comparison |
| Q2D | Exact decay, nontrivial nonlinear advection/forcing, energy/enstrophy, dealiasing, timestep refinement | Consistency/convergence; experimental comparison only within implemented closure/BC regime |
| Thermal | Conduction/advection, conjugate interfaces, buoyant B=0 limit, heated nonuniform-field duct | Heat balance, temperature/heat-transfer profiles and published cross-code cases |
| Global fields | Circular-loop/straight-wire limits, coordinate rotations, analytic gradients, source-free divergence/curl, domain checks | Independent formulas, interpolation refinement, VMEX/ESSOS and parameter JVP/VJP checks |
| Mapped channels | Straight identity, hydrodynamic curved duct, then MHD curvature | Metric identities, mesh studies and independent comparison |
| Optimization | Analytic sensitivities, active bounds, infeasible trials, selected feasible designs | Taylor tests and held-out finer-grid/external forward evaluations |

Resolve relevant Hartmann $a/Ha$ and side $a/\sqrt{Ha}$ scales or validate
a declared wall model. Tiny uniform meshes cannot resolve arbitrary
fusion-scale layers. Distinguish laminar, Q2D and turbulent regimes with Re,
Ha, N and appropriate boundary-layer criteria.

Apply Richardson/GCI estimates only where refinement/asymptotic behavior
supports them. Document anisotropy and nonmonotone convergence. Separate
numerical, digitization/reference, measurement and model-form uncertainty.
Agreement of two codes alone is not validation against nature.

FreeMHD/OpenFOAM are external comparators, not LMX runtime dependencies.
Pin source, container digest, dictionaries, mesh and data. Match the physical
problem—not necessarily the numerical method. The existing build pins FreeMHD
`14b54a3e8e1a05b6ee4c98331995abaaae96e7a5` and OpenFOAM v2206;
see the [external validation guide](docs/validation/freemhd.md).
The pinned FreeMHD repository
lacks a ready ALEX B1 case, but public case/result archives exist.
The available `S3_Buhler_Ha616` archive is a different pipe experiment:
reproduce it under its own name or construct/review a matched B1 case.
Never relabel it ALEX.

The existing two-step Docker B2 smoke establishes execution/parser behavior,
not production validation. B1/B2 refinement and independent acceptance remain
open. Run Docker once per relevant candidate, not for unchanged docs.
Keep bulky inputs outside Git with licenses and checksums.

## 7. M4: performance and parallelism

Extend the current benchmark CLI, covering fully developed, generic 3D,
specialized B1/B2 and Q2D, field gradients, then thermal/device objectives.
Record source/dependency versions, hardware/backend, dtype, cache state,
threads/affinity, grid, tolerances, work count, residuals and correctness.

Separate setup/field sampling; cold lowering/compile/first run; synchronized
warm primal/gradient; transfers/I/O; actual peak resident host/device memory;
and compiler estimates. Production timings need at least ten samples and
median/spread. Distinguish allocated pools from live buffers and solver work
from physical time or terminal accuracy.

Use equal problems/accuracy, time-to-tolerance curves over grid and Ha,
matched JAX versions and isolated hardware. Finite-step timings cannot support
converged-solver speed claims. Work in this order:

1. Remove redundant operator/geometry construction, closure retracing and
   inner-loop host synchronization. Move diagnostics outside hot kernels.
   Reuse factors only while their coefficients are valid.
2. Measure tridiagonal/RHS batching, factorization, launches and Krylov work.
   Transfer generic improvements to SOLVAX with derivative benchmarks.
3. Keep full fields local; exchange halos, combine reductions and amortize
   small coarse operations. Audit primal **and transpose** all-gathers.
   Measure dynamic calls, bytes and overlap—not just HLO text.
4. Sweep CPU threads 1/2/4/available cores and controlled logical layouts.
   Logical devices on one machine are decomposition/correctness evidence,
   not physical-core or multi-node strong scaling.
5. Run one-/two-A4000 strong scaling at fixed global problem and weak scaling
   at fixed work/device. Cover primal/adjoint, at least three useful sizes,
   high-Ha workloads and iteration growth.
6. Measure whether Q2D distributed FFTs help on two GPUs. Independent-design
   batching may be more useful. Distinguish ensemble throughput from spatial
   scaling; implement/document only the justified path.
7. Close specialized B1's sharding restriction after one-device physics/AD
   gates. Explicitly validate unsupported layouts and mesh divisibility.

Targets: at least 2× one-GPU CPU speedup and 60% two-GPU efficiency on selected
meaningful production cases, not all tiny problems. Report failures and retain
the best single-device path. Add custom kernels only after profiling and a
portability, maintenance and derivative-cost comparison.

## 8. M5–M7: VMEX/ESSOS to blanket design

### Field and geometry contract

Pin integration evidence to VMEX
`09f18464e936a8c9bf0abba62bcdc919bdc7c55b` and ESSOS
`1b3210ca34efaceec09272aa29599c9788c4ec35`, then test supported minimum/current
versions at release. Keep both optional; importing LMX must not import them.

Use one small JAX field-provider contract: Cartesian positions in metres,
Cartesian B in tesla, explicit parameter PyTrees, coordinate conventions and
a validity domain. Support analytic functions, traceable interpolation,
ESSOS Biot–Savart and appropriate VMEX exterior fields.
ESSOS `BiotSavart.B` evaluates one Cartesian point; batch with `jax.vmap`.
Do not assume another array API or mutate coil objects inside JIT.

Blankets are outside the plasma. Do not extrapolate VMEX interior equilibrium
samples into channels. Toroidal finite-beta examples need a verified total
exterior field: external coils plus plasma contribution with correct signs
and no double counting. Inspect VMEX parameterized-surface exterior interfaces
for live equilibrium sensitivities; frozen WOUT sampling is not that derivative.
Vacuum mirrors can start with ESSOS. Open-ended finite-beta mirror boundary
treatment needs separate verification, not unexamined toroidal virtual casing.

For channel map $X(\xi;q_g)$ and basis $Q(\xi;q_g)$, use
$B_{\rm local}=Q^\mathsf{T}B_{\rm global}(X;q_{\rm coil},q_{\rm eq})$.
Differentiate position, basis, metric, field parameters and PDE together.
Test identity maps/rigid motions, then curvature. Rotating B alone does not
implement curved-duct physics. Reject out-of-domain interpolation by default;
report proximity to coils/singularities and verify divergence preservation
or quantify interpolation error.

### Thermal and engineering scope

Implement and independently verify energy transport/solid conduction:
$$
\rho c_p(\partial_tT+u\cdot\nabla T)
=\nabla\cdot(k\nabla T)+Q_{\rm vol}+J^2/\sigma,
$$
with declared heat-flux/interface conventions and justified viscous heating.
Begin with constant properties. Add temperature-dependent laws only within
documented ranges and Boussinesq buoyancy only where valid. Report Pr, Pe,
Gr/Ri, energy storage/balance and MHD groups.

A straight segment is local screening, not a nonplanar blanket circuit.
Choose one defensible mapped-channel extension; explicitly model relevant
bends, connections, conducting paths and inserts before manifold claims.
Do not simultaneously pursue arbitrary unstructured geometry.
Turbulence, free surfaces, corrosion, tritium transport, stress and disruption
transients remain outside initial qualification unless a study supplies its
own model/validation gate. Low Rm alone does not justify omitting externally
induced electric fields during rapidly changing B.

### One parameterized application example, three configurations

Use stellarator/tokamak/mirror configurations of the same reusable example:

1. Verified prescribed field and isothermal hydraulic screening.
2. Live ESSOS field and VMEX exterior contribution where appropriate.
3. Shape/wall/material/flow design on admissible fixed topology.
4. Thermal loading/constraints after M6.
5. Held-out finer-grid and independent-solver checks, then a reproducible
   feasible Pareto/frontier study.

Variables may include channel dimensions/placement, insert/wall properties,
flow allocation and selected coil/equilibrium DOFs. Constrain clearance,
thickness, mapping Jacobian, temperature, pressure, flow, material validity and
relevant plasma/coil engineering properties. Do not improve channel performance
by destroying confinement or moving coils without their constraints.

Use physical work/heat/nonuniformity objectives with documented scaling.
Prevent trivial zero-flow optima through heat/flow requirements.
Reject failed forward/adjoint results; use external optimizers and JAX
composition rather than an LMX optimizer framework. Compare AD cost/accuracy
with finite differences, not merely optimizer iteration counts.
Evaluate uncertain loads/fields/materials with a bounded ensemble.
External ParaStell/OpenMC or equivalent tools can supply geometry, neutron
loads and breeding/shielding constraints with independent provenance.
LMX alone cannot certify breeding ratio or whole-blanket viability.

## 9. M8: slim code, API, docs and examples

### Simplification without losing science

Reduce duplication and parameter complexity, not merely lines.
`_solve_duct_projection` and `_solve_pipe_projection` span roughly 783 and 638
lines. Their shared setup, run state and diagnostics are better targets than
deleting every `_fringing*` file: those modules contain real physics.

- One case/result model, output path, validation schema and benchmark runner.
  No wrapper-only files, catch-all switches, proxies or parallel pipelines.
- Keep PDE stencils/BCs in LMX; transfer independent numerical machinery only
  with a general contract, tests and documented SOLVAX ownership.
- Pure array kernels separate from host validation/I/O; explicit units,
  immutable records, shape/dtype contracts, typed public signatures and errors.
  `solve(case)` is the normal entry; traced field functions support composition.
- Preserve Ruff/formatting and cohesive functions. No compressed one-liners,
  wildcard exports, hidden global state or line-count tricks.
- Parameterize tests/fixtures where failures remain identifiable. Preserve
  independent oracles and regimes; never shrink coverage by excluding code.

Retain enforced ceilings until deliberately revised with evidence.
Planning targets: normal clone <10,000,000 bytes, root exports around 28–30,
package modules at/below the current 15 where practical. Aim to reduce the
existing 14,793-line implementation toward 12,000 by deduplication, accounting
separately for justified thermal/mapped physics. Do not hide complexity in
SOLVAX to meet LMX counts. Track net files/lines, capabilities, runtime/memory
in source PRs. A purposeful module can beat an unreadable merged file.

Keep essential posters within the existing 500 KiB media budget, using crisp
WebP or compact SVG. Movies, arrays/checkpoints and print-resolution figures
are release assets. Recheck a normal clone at release. Rewrite history only
for an evidenced size problem, with backups/scope and repaired references.

### Reader-centered documentation

Use the existing Sphinx theme and MHX/VMEX's clarity as references, not a new
theme project before correcting content. Follow Diátaxis:

- **Tutorials:** install; complete small solve; interpret acceptance; change
  geometry/material/field; sweep a research parameter; then differentiate.
- **How-to:** own field/data, walls/forcing, restart/export, mesh/tolerances,
  CPU/GPU profiling, sharding and external validation.
- **Explanation:** model hierarchy/validity, equations mapped to operators,
  BCs/conservation, derivatives and performance.
- **Reference/evidence:** complete API with units/shapes/defaults/errors,
  TOML/CLI, capability/benchmark matrix, provenance and bibliography.

Each capability links equation → implementation → executable example →
verification/validation. Examples declare hardware, approximate runtime,
limits, outputs and failure meaning. An API dump is not a usable guide.

README order: purpose/evidence envelope; installation; executable
case → solve → check → plot/export; changing one's physical inputs; a real
parameter curve; advanced fringing/3D, Q2D, AD and device-study links.
Avoid bullet-only workflows and ambiguously qualified optimization plots.
Use tested snippets/literal includes; keep history out of user-facing material.

| Audience | Reusable example outcome | Required checks |
|---|---|---|
| Student | Hartmann profile/current, changing Ha and mesh | Analytic curve, units, conservation, mesh error |
| Walls researcher | Shercliff/Hunt/layered conductance sweep | Interface closure, limits, three useful meshes |
| Dynamics researcher | Q2D decay then nonlinear forcing/advection | Energy/enstrophy/time error; decay ≠ validated turbulence |
| Advanced 3D | Fringe/pipe, restart and VTK | Model regime, physical residuals, pressure/current profiles |
| AD user | Field/material/geometry response | Taylor/duality and forward/adjoint acceptance |
| Application scientist | Configurable three-device channel study | Field provenance, feasibility, fine-grid/external checks |

Reuse six existing Python examples where possible; no copied device scripts.
Add public-workflow coverage for Q2D. Replace the full multi-dozen-iteration
optimization in every portable run with a deterministic small contract plus
a scheduled full study, preserving gradient and independent physics checks.

Plots need resolved profiles/meaningful sweeps, normally ≥20 points for smooth
curves with justified exceptions. Never smooth two points into evidence.
Show measured samples/uncertainty as applicable. Convergence uses at least
three useful refinements; optimization marks infeasible/failed iterates.
Generate field/current/temperature, error, time/memory/scaling plots and a
3D/Q2D movie when dynamics warrant it.


## 10. Fast CI without weaker evidence

The fresh local full gate is approximately three minutes. Slow individual
tests: pipe update/derivative 67.0 s; B1 bounded gradient 66.6 s; pipe steady
projection 45.8 s; full design example 43.5 s; layered parity 35.0 s; reverse
memory 32.0 s; reduced B1 26.8 s; forced two-CPU parity 24.9 s. They overlap;
their sum is not suite wall time.

Balance shards by measured durations and compilation affinity, not filenames.
Reuse safe immutable fixtures/JIT executables within workers, control
oversubscription and preserve independent state. Do not repeat full examples
for every plot/export assertion. Smaller test grids must retain their
mathematical/physics contracts; move production sweeps to named jobs, not out
of the evidence matrix.

| Boundary | Checks | Target |
|---|---|---:|
| Edit loop | Changed lint and exact affected tests | <60 s where practical |
| Local source candidate | Conservative impacted set, no subset coverage claim | <3 min |
| Source PR, once | Full balanced suite, combined coverage ≥95%, architecture/import | <5 min compute; <10 min end-to-end |
| Docs/plan only | Relevant snippets/local links/Sphinx | <2 min; no unchanged full numerical suite |
| Workflow/dependencies | Workflow/selector tests and affected compatibility | Unknown executable changes fail closed |
| External numerical candidate | Pinned affected smoke/comparator once | Small cached smoke <2 min |
| Scheduled/release | Supported versions, production refinement, package/links, GPU and optimization artifacts | Separately budgeted campaigns |

Fix `.github/` scope so workflows cannot skip their own validation. Move
external link checks from every main push to scheduled/release boundaries.
Cache by OS/Python/JAX/SOLVAX/config with correct invalidation; cold profiles
must bypass compilation cache. Combine disjoint full-shard coverage and fail
on missing evidence. Keep inexpensive literature/physics tests in PRs.

Hosted execution is available: PR #54 passed on 2026-09-04 after the audit's
billing-blocked runs. Never hide failed checks. A maintainer-authorized
local-evidence exception applies only when annotations prove jobs never
started and equivalent gates pass on the exact source. It does not waive real
failures, SOLVAX coverage or final reproducibility. Do not merge new algorithm
PRs on the strength of this audit's older successful suite.

## 11. M9: papers and release acceptance

Stage publications rather than waiting for every blanket capability:

1. **Methods/software:** conservative differentiable inductionless MHD,
   analytical/continuous verification, credible B1/B2, implicit adjoints,
   CPU/GPU time-to-accuracy, actual peak memory and measured strong scaling.
   State limitations; speedup need not be universal.
2. **Physics/model assessment:** a new justified result on nonuniform fields,
   walls, curvature or thermomagnetic behavior, with resolution/uncertainty
   studies—not just solver galleries.
3. **Integrated design:** constrained channel optimization using live fields,
   thermal loads and independent held-out checks. Publish only the device
   configurations actually qualified.

Map every abstract/README/paper claim to case ID, versions, command, figure,
raw data, uncertainty and acceptance. Supply clean-install reproduction,
citation metadata, licenses and DOI-addressable artifact bundles. Small useful
licensed reference tables may live in Git; bulky inputs must be immutable
external assets.

Final acceptance requires all advertised matrix entries passed or explicitly
excluded; clean-install student examples and separate production configs;
reproducible CPU/GPU parity, timing, peak memory and claimed scaling; independent
physics evidence matching each paper's scope; LMX ≥95% coverage and SOLVAX
>95% for changed algorithm releases; complete docs/package checks; clone <10 MB.
No abandoned lane, hidden equation dispatch, unqualified optimality/speed
claim, or historical narrative remains in user-facing material.

## 12. Primary reading and implementation references

These sources inform requirements, not transferred validation certificates.
If only an abstract/catalogue was accessible, inspect the relevant complete
method/data before freezing its benchmark.

| Source | Role and review access |
|---|---|
| [Smolentsev et al., V&V, 2014](https://doi.org/10.1016/j.fusengdes.2014.04.049), [manuscript](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf) | Benchmark hierarchy/ALEX; manuscript definitions/tables inspected |
| [Ni et al., 2007](https://doi.org/10.1016/j.jcp.2007.07.025), [author-hosted text](https://bpb-us-w2.wpmucdn.com/research.seas.ucla.edu/dist/d/39/files/2019/08/JCP-v227-NiCurrentPart1.pdf) | Compatible current/Lorentz scheme; introductory method text inspected |
| [Mistrangelo et al., Nuclear Fusion 65 116006, 2025](https://doi.org/10.1088/1741-4326/ae0800), [full text](https://publikationen.bibliothek.kit.edu/1000185454/167701235) | Heated nonuniform duct/conjugate wall benchmark; equations/tables/results inspected |
| [Smolentsev, pressure-drop review, 2021](https://www.mdpi.com/2311-5521/6/3/110) | Blanket literature map; metadata/search abstract, full-page access limited |
| [Müller and Bühler, textbook, 2001](https://link.springer.com/book/10.1007/978-3-662-04405-6) | Model/asymptotic reference; publisher contents/synopsis, not entire book |
| [Shercliff, 1953](https://doi.org/10.1017/S0305004100028139), [Hunt, 1965](https://doi.org/10.1017/S0022112065000344) | Canonical profiles; reproduce original normalization/BCs |
| [Sommeria–Moreau, 1982](https://doi.org/10.1017/S0022112082001177), [Pothérat et al., 2000](https://www.cambridge.org/core/product/identifier/S0022112000001944/type/journal_article) | Q2D validity/higher-order effects; latter abstract inspected, not an implemented LMX closure |
| [FreeMHD](https://github.com/PlasmaControl/FreeMHD), [Wynne et al., 2025](https://doi.org/10.1063/5.0230242), [data](https://zenodo.org/records/13964055) | Independent FV/cases; source/project material inspected |
| [Vertex-CFD full-induction preprint, 2025](https://arxiv.org/abs/2511.15549) | Model boundary/implicit multiphysics inspiration; abstract, not ready oracle |
| [JAX-Fluids 2.0](https://arxiv.org/abs/2402.05193), [source](https://github.com/tumaer/JAXFLUIDS) | Parallel differentiable CFD comparison; no transfer of its scaling claims |
| [JAX-CFD](https://github.com/google/jax-cfd), [NekRS](https://github.com/Nek5000/nekRS) | Operator/solver organization; inspect specific implementation and license before reuse |
| [PETSc SNES](https://petsc.org/main/manual/snes/), [KSP](https://petsc.org/release/manual/ksp/) | Globalization/block preconditioning, not a new LMX dependency |
| [Yashchuk, 2023](https://arxiv.org/abs/2309.07137), [Optimistix adjoints](https://docs.kidger.site/optimistix/api/adjoints/), [Revolve](https://doi.org/10.1145/347837.347846) | Root vs trajectory AD and checkpointing; abstract/docs and canonical reference |
| [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html), [profiling](https://docs.jax.dev/en/latest/profiling.html), [memory](https://docs.jax.dev/en/latest/device_memory_profiling.html), [shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html) | Synchronization, compile/memory accounting and distributed arrays |
| [NASA V&V](https://www.grc.nasa.gov/www/wind/valid/tutorial/tutorial.html), [grid convergence](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html), [Celik et al., 2008](https://doi.org/10.1115/1.2960953) | Defensible discretization uncertainty; NASA guidance inspected |
| [VMEX](https://github.com/uwplasma/vmex), [ESSOS](https://github.com/uwplasma/ESSOS) | Exterior-field/Biot–Savart contracts checked at pinned revisions |
| [ParaStell, 2024](https://www.frontiersin.org/journals/nuclear-engineering/articles/10.3389/fnuen.2024.1384788/full) | External parametric geometry/neutronics and whole-device scope |
| [Diátaxis](https://diataxis.fr/), [MHX](https://github.com/uwplasma/MHX), [VMEX README](https://github.com/uwplasma/vmex#readme) | Reader-centered organization and presentation references |

## 13. Compact work log

### Resume checkpoint

- Main is #57 merge `d18fefc`; #54–#57 passed all applicable exact-head gates.
  Q2D acceptance is explicit, pressure-tap flux work is available; F2 remains open.
- Integration PR #63 contains #58 `ede81ec`, #59 `a3ddfad`, #60 `75b0ef4`,
  #61 `f25ec90`, and #62 `d53da86`, all preserved as ancestor commits.
  Target main directly; require exact-head full hosted coverage/docs/FreeMHD.
  After #63 merges, close superseded #58–#62 with its merge link. This avoids
  serial rebase/qualification of the same combined tree; keep them open until then.
  Jobs have been queued for runners; some scope/impact jobs now progress.
  Never restart solely for queue delay or waive gates using prior-head results.
- #58: body-drive work and stored kinetic energy use the tap-center slab;
  three geometries and force/velocity/geometry derivatives tested. Original
  head passed numerical shards and FreeMHD; rebased head must qualify.
- #59: independent 18-cell pressure oracle verifies weighted symmetry,
  positive energy, rank 17 Neumann / 18 mixed and AD. Signed B2 momentum
  residual supplies diagnostic norms. Unlimited Newtonian velocity gradients
  fix a 7.8858% resting Jacobian mismatch; advective gradients stay limited.
  Resting 108-unknown uniform Stokes rank/adjoint/FD checks pass, not full B2.
- #60: shared distance-weighted face interpolation fixes affine consistency;
  23 fewer package lines, no files added; geometry JVP and B2 restart pass.
  Nonuniform energy compatibility is separate and remains open.
- #61: explicit SolverConfig/TOML formulation replaces name-based dispatch;
  renamed production B1/B2 cases pass and restart identity is checked.
  Full combined-stack local gate: 520 passed / 95.24% / 144.4 s; docs pass.
  Reproduce: `scripts/run_full_test_suite.py --coverage-xml artifacts/explicit-formulation-coverage.xml --junit-xml artifacts/explicit-formulation-junit.xml`.
- #62 `codex/bounded-field-tables` is based on #61; formatting also checked.
  F6: honor supplied axial coordinates, reject extrapolation/nonfinite queries,
  validate table axes/components, load once, interpolate vector components
  together. Five distinct affected tests pass; fringing/table group 16.55 s.
  Both wall/field tutorial snippets execute; clean Sphinx/Ruff/audit pass.
  Pipe caller now supplies mesh cell centres rather than profile stations;
  shifted-origin sampling and production/AD parity tests pass (47.72 s).
  No new files; test allowance 12,330 covers these additional checks.
  Warm local float64 interpolation (17³ table, 25³ queries, 10 samples):
  separate component calls 24.57 ms median vs vector call 7.63 ms, matching
  outputs at 1e-13. Excludes I/O/compilation; not a solver speedup claim.
  Host-side table sampling is not live coil/geometry AD; M5 remains open.
- Active `codex/ci-superseded-work` is based on #62 `d53da86`.
  Add per-PR cancellation to docs/FreeMHD, keeping workflows independent and
  non-PR runs distinct. Workflow/action changes trigger numerical qualification;
  docs/FreeMHD selectors cover their own setup and physical-input dependencies.
  Eleven real-Git-diff cases exercise all three workflow selectors (33 shell
  executions) in 1.24 s, including Q2D/external exclusions and B2 markers.
  No new files; test allowance 12,400 matches the combined inventory.
  YAML parses for all changed workflows. One inventory comparison failed while
  docs/log were being edited; stable-tree rerun passed all 35 config tests (5.73 s).
  Six obsolete docs/FreeMHD runs on #59/#61/#62 were confirmed cancelled;
  current-head and main runs were preserved. No numerical gates relaxed.
  Full combined local qualification: 532 passed, 95.28% coverage, 129.7 s.
  Reproduce: `scripts/run_full_test_suite.py --coverage-xml artifacts/ci-stack-coverage.xml --junit-xml artifacts/ci-stack-junit.xml`.
  #59 numerical shards and FreeMHD passed; integration #63 still needs its gates.
- Next M1: nonuniform pressure/continuity compatibility, advection limiter
  transitions and electromagnetic closure before a production B2 campaign.
  Uniform checkerboard diagnostic: interior |DG|=0 vs compact |L|=4.
  Pre-#60 nonuniform resting test (dy=dz=[0.4,0.8,1.3], dx=1):
  weighted pressure/continuity defect 0.3714285714 despite full rank/FD parity.
  Affine consistency alone does not close that energy identity.
- F2 still needs consistent viscous/Joule/electrical flux and storage-rate
  balance. M1–M9 remain open; no accepted B2 steady primal/adjoint is claimed.
  SOLVAX minimum/CI pins 0.19; no algorithm changes in these LMX tranches.
  Python 3.11.14 / JAX 0.10.2 locally. CI shard budget stays 540 s.
- Raw profiles remain local ignored artifacts; public packaging remains open.
  Resume by fetching origin, inspecting PR/working-tree state, preserving
  unrelated edits, and recording/pushing evidence and next steps here.

Keep at most ten substantive entries. Older evidence becomes immutable
commit/PR/artifact links, not another tracked log. Each entry records work,
evidence, unresolved gate and next action. Keep this active plan below about
700 lines; move detailed scientific specifications into their owning docs/tests.

| Date | Work/evidence | Remaining / next |
|---|---|---|
| 2026-09-04 | Source/docs/tests/history/literature review; fresh 501-test/95.22% gate; architecture/docs passed; normal clone 3.56 MiB; CPU and real one-/two-GPU traces/profiles/parity | F1–F13 recorded; production B1/B2, objective correctness, actual peak-memory campaign and full device-design gradients remain open |
| 2026-09-04 | Replaced 4,263-line historical plan with this evidence-led roadmap; repaired README gradient shape, standalone Q2D snippet and contributor link; qualified generic momentum, pump-work and Q2D acceptance claims | M0 documentation partially addressed; begin physical-objective/dtype regression work in PR A. No new PDE/SOLVAX algorithm or blanket result certified |
| 2026-09-04 | #54 merged; #55 Q2D precision fix: 507 local tests / 95.22%, Q2D 100%, 259.9 s on SOLVAX 0.19. Six additional cases; test budget 11,950→12,010 (57 lines), no new files/API. Minimum/CI SOLVAX corrected to 0.19 | Initial 0.17 run failed (454 pass, 3 budget failures, 1 collection error); successful complete rerun supersedes it. Hosted #55 gates govern merge. F2 and Q2D energy acceptance remain open; no SOLVAX algorithm change |
| 2026-09-04 | Hosted #55 found a single-device B2 assumption and cold compile timeouts. Fixed the type guard, reused reduced sharding AD controls/thread limits and redistributed B1. 28 affected local tests pass (39.9 s); all 508 CI cases partition exactly once | New hosted coverage required before #55 merges. No numerical assertions/tolerances removed; cached local timings are not a cold CI speed claim |
| 2026-09-05 | #55 merged after all exact-head hosted gates and ≥95% coverage passed. Pressure-tap flux-work tranche rebased onto main; 33 affected local tests and clean docs pass | F2 still requires drive/storage/dissipation work on one control volume. Qualify the pressure-work PR before merge; M1–M9 remain open |
