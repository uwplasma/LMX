# LMX: evidence-led channel-design plan

Revised 2026-09-06. This PR is planning/documentation only: no solver change,
deletion, merge or release. This is the forward plan and resume log; earlier
investigations remain in the [prior plan][prior] and PRs, not another archive.
User documentation describes current capabilities; history belongs here.

## 1. Decision: useful channel science before a blanket platform

The destination remains accurate, differentiable, low-memory liquid-metal
blanket-channel design for stellarators, tokamaks and mirrors. The next useful
deliverable is narrower: **verified hydraulic design examples and one credible
3-D channel formulation**, followed by **laminar thermal-channel design**.
A device-shaped rendering or decreasing loss does not establish blanket
performance. LMX supplies channel thermofluid calculations, not equilibrium,
coil design, CAD, neutronics, breeding or structural analysis.

Material changes to the previous proposals:

- Do not delete specialized B1/B2 operators before their useful conservation,
  wall, pipe and validation contracts have a verified replacement. Freeze
  unsuccessful iteration variants; remove duplication by mathematical ownership.
- Do not make high-Ha B2 convergence a prerequisite for every useful result.
  Fully developed design, field/geometry preparation and a prescribed-velocity
  thermal specification can advance without it.
- Do not add convection to an inconsistent projection and call it general 3-D
  MHD. Resolve coupled operators, constraints and derivatives first.
- Thermal physics is essential to the blanket goal, not indefinitely postponed.
  Start with a bounded laminar problem, not turbulent whole-device optimization.
- Retain Q2D; forcing already exists. Qualify a forced example before adding
  another turbulence solver or closure.
- No promised version number, paper count, universal GPU speedup or arbitrary
  source-line target. Release later, after PR reconciliation and important goals.

### Work order and stop rules

Only one numerical formulation change should be active at a time. Documentation
and independent benchmark preparation need not start another solver.

| Order | Deliverable | Exit / decision |
|---|---|---|
| A — next | Reconcile PRs; qualify reverse residual; concise README | Resolve reproduced reverse NaNs, then exact-head tests/coverage/docs; no merge in this planning PR |
| B — next useful result | Fully developed pressure/flow/profile design | Analytic forcing oracle, checked implicit gradients, feasible optimum, finer-grid confirmation |
| C — numerical critical path | One conservative inertial rectangular-channel residual | Nonuniform mathematical contracts, continuous MMS, physical energy and steady residual gates; then implicit adjoint |
| D — application bridge | Laminar heated channel and device-derived representative channel | Thermal balance/refinement; valid exterior field and admissible geometry; no whole-blanket claim |
| E — conditional research | Forced Q2D statistics; pipe turbulent heat transfer if affordable | Statistics, resolution and external evidence, declared compute budget; otherwise retain external CFD |
| Continuous | Profiling, consolidation, docs and evidence | Time-to-accuracy, measured memory, exact commands; no lost physical tests |

Before each costly campaign, log its hypothesis, expected diagnostic change,
work budget and exit criterion. A failed bounded attempt produces an operator
or model decision, not a longer continuation. Do not revive August B2 predictor,
relaxation or acceleration experiments without new residual evidence. No dense
production Jacobian, full momentum inversion inside every pressure iteration,
or new public experimental API.

## 2. Audit: code, PRs and evidence

Reviewed main `d18fefce2eb241b4c8d8bc733e914e530e8ce4d4`; integration candidate
`1cef392a55d5266e4a0bb16c907d30c1f699a4a4` (#63); planning proposal
`58e5b974b195d478d7fb1f77bf6d6fa6c54d7ba7` (#64). Inspected generic/specialized
momentum, electric and pressure operators, traced/host loops, Q2D, field sampling,
objectives, tests, examples, docs, recent commits and CI selectors. This is a
focused review, not a claim to have read every publication or every line of
every upstream project.

| State | Evidence and implication |
|---|---|
| Merged #54–57 | Model-contract roadmap, Q2D dtype/acceptance fixes and pressure-tap work; useful correctness work, not blanket-goal completion |
| #58–62 | Tap-slab work, pressure/viscous contracts, shared interpolation, explicit formulation and bounded field tables; all included in #63 |
| #63 | 24 files, +698/−206 against main; hosted numerical shards, combined coverage, docs and pinned FreeMHD check green at reviewed head |
| #64 | Proposes immediate specialized-path retirement and a 2.0 work order; this plan is a replacement proposal, not an instruction to merge both roadmaps |
| `64be572d38255aeb3e8ca5c50a9e81b89e52716b` | Pushed face-projection energy follow-up outside #63; inverse-mobility face norm is not full physical kinetic energy |
| Local | Clean main-derived checkout at audit; numerical edits recorded in an older unavailable workspace were not recovered or imported |

After approval, reconcile README/plan changes with #63 deliberately: both
branches edit these files. Resolve the reverse-mode finding before declaring
integration ready. Merge one qualified integration tree, then close redundant
#58–62 with its link. Resolve #64 explicitly, not two incompatible work orders.
Do not close or merge PRs in this audit.

### Fresh local evidence

Apple M3 Max, Python 3.11.14, JAX 0.10.2, SOLVAX 0.20.0. Concurrent unrelated
CPU work means these test durations are not controlled speed benchmarks.

| Check | Result | Qualification |
|---|---|---|
| Main portable suite | 508 passed; 198.79 s pytest / 199.3 s runner | Full runner excludes curated marker |
| Curated first-run test | 1 passed, 24.32 s | Executes Hartmann, Hunt and TOML workflows |
| Architecture | 15 modules, 6,385 core / 14,835 raw source lines, 28 root exports, seven examples | 87 tracked files; package size already bounded |
| Ordinary network clone | 3,468 KiB including checkout and Git; Git alone 1,692 KiB | Fresh non-shallow clone at main; below 10 MB without rewriting history |
| Local coverage | Full run exited 1: no data collected despite passing tests | Existing root `lmx/__pycache__` makes `--cov=lmx` select that directory instead of `src/lmx`; explicit `--cov=src/lmx` collects data on five diagnostic tests. No fresh aggregate percentage claimed; cache untouched |
| #63 resting residual | Forward 108×108 Jacobian finite; VJP has 81 non-finite entries | Reproduced in detached #63 worktree without source changes |
| Revised README | 317→183 lines; all four Python blocks execute (9.57, 4.86, 22.74, 2.26 s) | Advanced duct returns `step_limit`, as disclosed; Q2D completes with energy defect 6.79e-6 |
| Documentation / links | Sphinx warning-as-error build, Markdown parsing, local link existence and diff whitespace checks pass | Source and test files unchanged; no invented thermal or blanket outputs |

VJP reproduction: use the residual and zero state in
`test_b2_stokes_residual_has_consistent_resting_jacobian`, then evaluate
`jax.vjp(residual, state)[1](cos(arange(108) + 0.3))[0]`. The existing
forward test passes. Inspect zero-gradient division inside
`_limited_linear_vector_face_weights_duct`: masking an undefined quotient
afterward does not make its reverse derivative safe ([JAX FAQ][jax-where]).
Repair must preserve primal values and pass resting/nonzero JVP/VJP, JIT,
transpose-duality and finite differences. This plan does not implement it.

Retained earlier evidence, **not remeasured here**:
#63 local qualification 532 tests / 95.28% coverage / 129.7 s; B2 saved state
momentum defect 0.1378074 against 1e-3; no accepted steady B2 primal/adjoint.
A small update (0.0014485) did not imply momentum balance. Keep failed-state
provenance without treating a local checkpoint as a research result.

### Findings that determine priority

1. Generic duct/pipe steps omit momentum advection, clip fields and alter
   stationwise flow. Centered gradient/divergence do not compose to the compact
   pressure Poisson operator; alternating pressure can evade the former.
2. Generic layered momentum/electric spacing contracts differ. Pipe transverse
   diffusion applies scalar Laplacians componentwise. Electric closure alone
   certifies neither vector momentum nor mechanical energy.
3. #63 improves face operators and signed residuals, but a uniform resting
   rank/forward-AD check is not nonuniform coupled-energy or reverse-AD proof.
4. Finite 3-D derivatives describe a checkpointed finite recurrence. Geometry
   scaling does not resample live global fields. Fully developed field-core AD
   exposes forcing and scalar magnetic scaling, not general shape AD.
5. No LMX temperature equation, conjugate heat transfer, buoyancy or qualified
   3-D turbulence. Taylor–Green decay is not turbulent heat-transfer evidence.
   Q2D already accepts a vorticity-forcing array.
6. Compact ALEX tables need raw/digitized-data provenance and uncertainty review.
   Rounded points alone establish neither fabrication nor reliable digitization.
   A reduced FreeMHD comparison is not production B1/B2 validation.
7. Host solves synchronize for diagnostics, but traced finite recurrences
   already exist. Absence of `lax.scan` does not establish a bottleneck.
   Profile the compiled path and communication before prescribing a rewrite.

## 3. B: first useful optimization result

Extend the existing fully developed tutorial/example, not a new optimizer API.
State SI units, field direction, wall conductance and no-slip conditions.
Use Hartmann as analytic control, then Shercliff and conducting-wall Hunt.

1. **Drive for requested throughput.** At fixed field/material/geometry, the
   fully developed inductionless problem is linear in force density f. Compute
   G = Q(f=1); eliminate drive with f = Q_target/G. For a uniform imposed
   pressure-gradient drive, pressure drop = f L. Check two direct solves and
   df/dQ = 1/G. Do not optimize a scalar that can be eliminated exactly.
2. **Fit a velocity profile.** Minimize area-weighted squared profile error
   over positive drive and bounded field multiplier using
   `solve_fully_developed_fields`. Recover known synthetic parameters first;
   check identifiability. Drive alone changes amplitude, not arbitrary shape.
   Incompatible targets must report irreducible error, not a fictitious optimum.
3. **Minimum work at fixed flow.** Eliminate drive; compare admissible wall
   conductance, aspect ratio or field orientation only as their derivatives
   become supported. Fix throughput, field requirements, area/volume and wall
   bounds. Otherwise zero flow, zero B or infinite area is a trivial answer.
   Use documented sweeps for material/shape controls until steady-core AD
   supports them; do not substitute finite-step 3-D gradients silently.

Acceptance: full analytic profiles over declared Ha/conductance ranges, three
resolved meshes and tighter tolerances, implicit/direct agreement, Taylor and
finite-difference step sweeps, feasibility and projected-gradient/KKT checks,
then re-solve the optimum on a held-out finer mesh. Keep finite-trajectory and
accepted steady design distinct. Plot profiles before/after, actual objective
and constraint iterates, gradient error and work versus flow. Show discrete
samples as samples, not an interpolated claim of experimental resolution.

## 4. C: one credible 3-D formulation and deliberate derivatives

First target a rectangular conducting-wall channel. Retain the best existing
conservative face machinery, not automatically the shortest generic recurrence.
Define one residual R(u,p,phi;theta) for momentum, incompressibility and electric
closure. Fluid velocity/pressure and fluid/solid potential are the state;
eliminate solid velocities. Apply gauges according to BCs and conductor
connectivity. A fixed-flow multiplier belongs in the equations, not a later
stationwise rescaling.

- Define cell/face locations, volumes, orientation, boundary work and weighted
  inner products. Require compatible pressure/divergence and current/Lorentz
  coupling on nonuniform meshes and across conducting interfaces.
- Compare face-primary staggered/MAC placement with repairing the retained
  collocated placement on the **same tiny oracle**. Choose once on nullspaces,
  energy, implementation size and adjoint cost; keep one production formulation.
  [Ni et al.][ni] motivates consistent electromagnetic coupling.
- Prove Stokes/electric limits before adding conservative advection and full
  vector stress. Stop on this same physical residual. Smooth limiter
  regularization changes the model and requires an error study.
- Derive continuum MMS independently of the discrete implementation: three
  meshes, velocity/pressure/current, integral flow and wall fluxes. Require
  formal order for smooth cases; interfaces need their own analysis.
- Close energy on one volume: pressure/body work, kinetic storage/transport,
  viscous and Joule dissipation, electrical boundary work. Do not count an
  imposed pressure-gradient drive twice. Joule heating later also enters the
  thermal accounting.

Dense matrices are tiny rank/transpose oracles only. Choose one safeguarded
steady solve; least-squares minimization cannot replace a correctly constrained
square PDE. Preserve B2's normalized momentum gate 1e-3 plus its frozen
mass/current and independence gates. Specify new tolerances from scales,
refinement and uncertainty before running, not from an attractive result.

For accepted steady states use R_y^T lambda = J_y^T and
dJ/dtheta = J_theta − lambda^T R_theta. Use matrix-free JVP/VJP with SOLVAX
linear/root algorithms ([Blondel et al.][implicit]). The inspected SOLVAX
`root_solve` default tangent constructs a dense Jacobian: supply the
large-system tangent/transpose solve explicitly. Reuse factors/preconditioners
where valid; test nonsymmetric transposes and parameter-dependent coefficients.
Reject failed primals/adjoints rather than returning plausible gradients.

Finite trajectories retain checkpointed recurrences; implicit steady AD does
not replace their derivative. Test rest, zero field, nonzero/nonuniform states,
conducting interfaces and limiter transitions; include duality, Taylor,
finite differences, tolerance sweeps and memory scaling. Keep topology/mesh
counts static unless explicitly supported. Any SOLVAX change needs its own
>95% combined coverage, docs/API and measured memory; this LMX review does
not recertify the SOLVAX suite.

### B1/B2 without another tuning campaign

Preserve compact specs, observations/provenance, wall contracts, restart
semantics and the independent FreeMHD harness. Docker tooling stays outside
the runtime package. Freeze benchmark-specific acceleration pending the residual gate.

Verify B=0, Hartmann/Shercliff/Hunt and modest-Ha 3-D cases first; B2 next;
B1 only if layer resolution is affordable. Match units, geometry, field map,
wall thickness/conductance, drive, inlet/outlet, pressure reference and stations.
Resolve Hartmann (~L/Ha) and side (~L/sqrt(Ha)) layers, not only axial cells.
Thin-wall approximations need their own thickness/validity check.

Prefer original measurements. Digitization needs figure/axis calibration,
repeat-digitization error, source and redistribution permission. Separate
same-discretization regression from continuum code-to-code comparison.
Use independently converged FreeMHD, not initial-profile installation demos;
matching unfinished trajectories cannot close acceptance.
[Smolentsev et al.][vv] and [Celik et al.][gci] anchor benchmark/refinement
reporting. Separate numerical, measurement and input uncertainty; disclose
non-asymptotic refinement. Unaffordable high Ha keeps that claim open.

## 5. Coordinates: what helps, what does not

The plasma magnetic axis and a pipe's coordinate axis are different. A blanket
outside the plasma need not contain either. VMEX/Boozer flux coordinates cannot
fix pressure compatibility or missing inertia and generally do not describe
the blanket exterior.

| Choice | Benefit | Cost / decision |
|---|---|---|
| Cartesian rectangular channels | Regular geometry, conservative metrics, efficient arrays | Default verified 3-D/thermal target; cannot silently represent bends |
| Cylindrical pipe with proper pole treatment | Exact circular wall and radial layer clustering | Full vector operators/axis regularity; near-axis cells can restrict timestep |
| Nonsingular multi-block mapped channel | Curved ducts or a circular core without a pole | Metric/interface/AD/halo complexity; introduce only when a selected application needs it |
| Cartesian embedded/cut cells | Flexible shape without cylindrical coordinates | High-Ha wall resolution, small-cell stiffness, moving-topology derivatives; not first replacement |

Cylindrical transverse vector diffusion includes
`Delta(u_r)-u_r/r**2-2*d_theta(u_theta)/r**2` and
`Delta(u_theta)-u_theta/r**2+2*d_theta(u_r)/r**2`.
Replacing r by epsilon or using scalar Laplacians independently does not
establish the smooth Cartesian limit. A uniform transverse translation is
nonzero at the axis although its cylindrical components depend on theta.

Before choosing a pipe rewrite, test smooth Cartesian scalar modes, transverse
translation, rigid rotation and an axis-crossing manufactured vector field.
Check divergence, vector Laplacian, energy and radial refinement; derive parity/
Fourier regularity from Cartesian smoothness. Radial flux r*u_r on a staggered
mesh is a documented alternative ([Verzicco–Orlandi][axis]); [Oud et al.][oud]
demonstrate conservative nonuniform cylindrical treatment. These motivate one
bounded comparison, not a new permanent implementation branch.

A future map X(xi;q) needs positive Jacobian, transformed conservative fluxes,
free-stream preservation, metric derivatives, rigid-motion invariance and
nonorthogonal pressure/stress/current tests. Rotating B alone is not curved
fluid physics. Promote a map only after the rectangular limit and one
independently verified curved case pass through the same public interface.

## 6. D: heat transfer for an engineering question

LMX cannot currently answer heat-removal questions. Add the minimum thermal
model after the velocity/flux contract it consumes is qualified:

    rho*cp*(dT/dt + u·grad(T)) = div(k*grad(T)) + q_vol + q_Joule

q_vol is supplied heating, not a neutron calculation. Solid walls have u=0.
Conjugate interfaces preserve T and normal heat flux unless a specified contact
resistance changes that law. Thermal k and electrical sigma are different inputs.

1. Constant-property passive transport on verified velocity: conduction slab,
   analytic advection–diffusion MMS, thermally developing duct. Include axial
   conduction when Peclet requires it; high Re alone does not justify high Pe.
2. After pipe qualification, check constant-property fully developed laminar
   limits Nu_D=48/11 for uniform wall flux and approximately 3.66 for uniform
   wall temperature ([Lienhard & Lienhard, §7.2][heat-textbook]). Compute bulk
   temperature by mass/enthalpy-flux weighting.
   Use rectangular analytic/MMS cases first if the pipe gate is still open.
3. Add conjugate walls and magnetically modified laminar velocity. Close inlet/
   outlet enthalpy, heat input, storage and losses. Include or quantify neglect
   of Joule/viscous heating at the selected scales. Do not extrapolate a
   hydrodynamic heat-transfer correlation into MHD.
4. Add Boussinesq buoyancy when Gr/Re² or equivalent scale analysis requires it;
   verify small density variation. Temperature-dependent properties follow when
   constant-property error matters. [Mistrangelo et al.][thermal-benchmark]
   provide a specific heated conducting-duct target, not a multiphysics framework.

Gate independent source/flux integrals, spatial/temporal order, interface
temperature/flux, bulk/wall temperature and Nu, gradient Taylor tests and
time-to-accuracy. Extend an existing example once supported. First thermal
design: minimize maximum wall temperature or hydraulic power subject to heat
removal, material-temperature, throughput and geometry constraints; confirm
feasible candidates on a finer mesh and externally. Do not wait for full B2
acceptance to verify passive heating on an already qualified laminar velocity.

## 7. D: VMEX / ESSOS to a representative blanket channel

**An equilibrium can seed an envelope. LMX cannot currently fill an arbitrary
shell with a validated liquid-metal calculation.** Envelope, channel topology,
exterior field and CFD mesh are separate inputs.

Local VMEX reviewed at `fedde0a9aea4162c945ea357880a98df87b34c68`;
ESSOS field source at `083f7cacc513810c312d56a220468cb05231a106` (unrelated
dirty files untouched). Inputs are in VMEX `examples/data`, not root `data`:

| Device | Concrete candidate | Missing choices / qualification |
|---|---|---|
| Tokamak-like | `examples/data/input.solovev` | Axisymmetric geometry seed; compatible external field, first-wall gap, channel routing |
| Stellarator | `examples/data/input.LandremanPaul2021_QH_reactorScale_lowres` | Generate/qualify equilibrium output and match coils; low-resolution input is not a convergence certificate |
| Mirror | `examples/mirror/mirror_fixed_boundary_nonaxisymmetric.py`; Pleiades two-coil reference for independent comparison | Open ends, lateral envelope and manifolds; no toroidal periodicity |

Pin units, input/upstream hashes, convergence and licenses before selecting a
fixture. Start with the simplest qualified input, not all devices in the first
PR. Large equilibria/coils should be optional checksum-verified downloads.

1. Reconstruct the boundary; specify first-wall gap and layer thicknesses;
   check orientation, clearance, self-intersection and positive fluid/wall
   thickness. [ParaStell §2.1][parastell] uses poloidal profile offsets and
   lofted layers for conceptual stellarator CAD/neutronics. That is a geometry
   precedent, not channel validation; geometric s>1 extrapolation does not
   supply an exterior magnetic field.
2. Choose one physical channel inside the build: inlet/outlet, solids, fluid,
   electrical contacts, length, heat load and temperature limits. Homogenized
   breeding material is not automatically coolant volume. Begin with a straight
   local channel; quantify the curvature/field-variation scales supporting that
   approximation. If bends/manifolds dominate, return to the mapped-geometry gate.
3. Supply B_global(X) in tesla at Cartesian metres throughout fluid **and walls**.
   Vacuum cases can use ESSOS Biot–Savart. Finite-beta tokamak/stellarator cases
   need plasma field contributions without double counting. Interior WOUT/VMEX
   samples are not exterior fields. Mirror end/exterior conditions need their
   own validation rather than an assumed toroidal virtual-casing construction.
4. Qualify offline bounded sampling first: actual coordinates, units/basis,
   coil clearance, interpolation error and divergence/curl in the appropriate
   source-free region. #63 fixes table issues; trilinear interpolation is not
   exactly divergence-preserving.
5. Then one traceable provider `B(X, parameters)` with explicit parameter
   PyTrees and validity domain; VMEX/ESSOS remain optional. Inspected ESSOS
   `BiotSavart.B` takes one point and captures self statically: vmap batches
   positions, not automatically live coil parameters. For basis Q use
   B_local=Q^T B_global(X); differentiate X, Q, metrics, field parameters and
   accepted PDE together. Remeshing, topology and Boolean CAD are not claimed AD.

Output plasma/first-wall/channel overlay, global/local field, admissibility,
pressure/current/flow and later temperature, optimization constraints and
independent gradient/refinement checks. First claim: **device-informed channel
screening**, not optimized blanket. Full blanket studies require externally
provided heat/neutron loads, materials and engineering limits as well.

## 8. E: turbulence and heat transfer

Separate “better at fixed flow” from “better at fixed pump power.” With imposed
total heating, steady heat removal is already constrained by energy balance:
the benefit may be lower wall temperature, not additional watts.

Transverse fields suppressed turbulent heat transfer in an insulated mercury
pipe ([Gardner–Lykoudis][pipe-heat]); a different heating/wall arrangement showed
enhancement ([Sukoriansky et al.][enhancement]). Low-Pr DNS offers no-field
thermal statistics ([Pirozzoli][low-pr]). These motivate controlled comparisons,
not claims that either effect has been reproduced in LMX.

Near term: adapt `q2d_turbulence_demo.py` with existing `Q2DProblem.forcing`
and resolved multi-mode initialization. Keep analytic decay as a test. Report
forcing/viscous/Hartmann energy terms, enstrophy, spectra, averaging-window
uncertainty, at least two seeds and grid/timestep sensitivity. State the Q2D
validity range from [Sommeria–Moreau][sm82]. Periodic SM82 has neither pipe
walls nor a thermal equation: it cannot answer the pipe question.

After inertial momentum, pipe and thermal gates, estimate DNS cells, timesteps
and memory for modest Re before implementation. If unaffordable, use qualified
external CFD and retain LMX for laminar screening. Do not add RANS/LES simply
to draw turbulence. An OpenFOAM turbulence switch is not low-Pr MHD validation.

The bounded pipe comparison fixes fluid/Pr, diameter, wall conductance, heating
pattern and inlet temperature; varies Re, Ha and field orientation with Rm
checked; separates forced convection and buoyancy. Record local/mean Nu,
maximum/RMS wall temperature, turbulent heat flux, velocity/Reynolds stresses,
total drive work and pumping cost. Compare fixed flow and fixed power; check
domain length, grid/time resolution and statistical error. Use no-field low-Pr
data and matched magnetic experiment or validated external CFD. Movies accompany
measurements, never replace them.

Ordinary AD of a short chaotic trajectory is not a reliable sensitivity of a
long-time average ([Blonigan et al.][chaos]). Start optimization with steady/
laminar or checked finite-horizon objectives. No shadowing algorithm is added
to SOLVAX by this plan; that requires a separate demonstrated need and budget.

## 9. Performance, consolidation and CI

Earlier matched-office Q2D (256², 32 steps) gave 5.9× primal / 8.6× gradient
GPU/CPU speedups. Generic duct 64×32×32 one-to-two A4000 timings were 282→440 ms
primal, 536→818 ms gradient: sharding worked but slowed down. Compiler temporary
estimates are not runtime peak memory. These are prior diagnostic baselines,
not new publication measurements.

1. Extend existing benchmark command: pin source/dependencies/hardware, dtype,
   mesh, parameters, tolerance, accepted status and error. Separate setup,
   lowering/compile, synchronized warm primal and value+gradient, transfers/I/O,
   runtime peak memory and compiler estimates ([JAX profiling][jax-perf]).
2. Profile host synchronization and kernels separately. One pure update should
   serve host and traced solves; bounded compiled chunks can retain acceptance,
   restart and logging without per-step synchronization. Choose fori_loop/scan/
   while_loop by history and AD semantics, not fashion; avoid storing all frames.
3. Reduce repeated setup, batch small solves, reuse factors/preconditioners.
   Compare stiffness treatment by time-to-accuracy. SOLVAX owns reusable algebra;
   LMX owns coefficients, boundary physics and physical acceptance.
4. Make precision explicit at application setup; audit import-time x64. Do not
   promise float32 for high-Ha cancellation/ill-conditioning. Mixed precision
   needs float64 physics/gradient checks, not bitwise identity after model changes.
5. For design throughput, prefer independent batching before spatial sharding.
   For large single problems measure fixed-global-size one/two-device scaling,
   gradients and communication. Logical CPU devices test partition parity,
   not physical strong scaling. Q2D distributed FFTs may lose on two GPUs.

Seek useful GPU speedup but report crossover and failure. Do not require tiny
cases to beat CPU or keep a slower multi-device default. No expensive GPU
campaign in this planning PR; CPU work can continue when office is unavailable.

Size: ordinary clone <10 MB; current module/export/example budgets remain.
The clone goal is already met. Remove duplicate assembly, dead helpers, redundant
test setup and failed orchestration, not independent physical oracles. No test/
source line-ratio cap; no giant functions solely to reduce files. Private
`_fringing_*` modules are legitimate for distinct equation ownership.
Docker, campaigns, large data and movies stay outside the runtime wheel.
No history rewrite needed for measured clone size.

CI: changed-surface tests during iteration, one full combined coverage run per
integration head, separate docs and pinned external checks. Correct coverage
path ambiguity without shrinking measured surface. Keep >95% combined line/
branch coverage and physics thresholds. Balance measured compilation/runtime
(the slow tests are B1 implicit AD, pipe parity/projection), cache safely and
cancel superseded heads. Separate runner queue from compute; target <10 minutes
compute for ordinary qualification. Schedule large refinement/statistical/GPU
campaigns without pretending unit CI establishes those results.

## 10. Documentation, figures and publication

README: concise purpose/limits, install, complete solve→check→save, steady
derivative, one example table, advanced 3-D/Q2D blocks and docs/citation.
Keep scientific posters; no history, prose roadmap or duplicate capability
tables. Examples distinguish demo and research settings.

Organize existing pages: start → fully developed/walls → gradients → advanced
3-D/Q2D; how-to for field data/output/restart; equations/assumptions; API;
validation mapping claim→test→command→artifact→limitation. Thermal/device
tutorials appear when executable. Public parameters need units, shapes,
defaults, validity, failure semantics and static/derivative status. Use VMEX/MHX
as organizational references, not sources of borrowed claims.

Figures show resolved profiles, refinement, constraints and synchronized
performance; discrete controls remain points. Movies use bounded frame storage,
fixed scales, physical time and model parameters. Compressed readable posters
stay in Git; checksum-addressed raw data/movies outside it. No new scientific
result is claimed in this planning task.

First plausible paper: verified differentiable channel methods/software with
analytic/physical/numerical validation, independent gradients, memory,
CPU/GPU time-to-accuracy and one constrained design. State B1/B2 honestly if
open. Thermal/device publications follow their physics/field/engineering gates;
turbulent heat transfer needs statistics and external evidence.

**Release hold:** no tag, version bump, PyPI upload or publication-ready claim.
Later reconcile all reviewed PRs, complete A–C and the explicitly selected
important D/E deliverables, qualify docs/examples and the exact source/wheel,
and review the evidence with the user before release. Do not silently drop a
release goal because it is difficult.

## 11. Sources and their decision impact

New targeted sources were read at the level stated; inaccessible full texts
are not represented as fully reviewed. Existing references/manifests remain in
`docs/reference/bibliography.md` and `benchmarks/provenance.json`.

- [Verzicco–Orlandi (1996)][axis], institutional abstract; [Oud et al. (2016)][oud],
  institutional abstract and publisher equation excerpt: axis/metric verification
  instead of epsilon regularization or an automatic coordinate rewrite.
- [Gardner–Lykoudis (1971)][pipe-heat], publisher abstract; [Sukoriansky et al.
  (1989)][enhancement], publisher abstract: thermal effects depend on arrangement.
- [Pirozzoli (2023)][low-pr], abstract/dataset description: low-Pr pipe DNS
  reference, not MHD validation.
- [Lienhard & Lienhard, sixth edition §7.2][heat-textbook], pipe heat-transfer
  equations and limits inspected: independent no-field laminar thermal oracles.
- [Blonigan, Gomez & Wang (2014)][chaos], preprint abstract: distinguish finite
  trajectories from turbulent long-time sensitivities; defer new algorithms.
- [ParaStell (2024) §2.1][parastell], methods: offset/lofted envelopes differ from
  CFD channels and exterior fields.
- [Mistrangelo et al. (2025)][thermal-benchmark], accessible manuscript model,
  interface and benchmark tables: bounded heated conducting-duct target.
  Local FreeMHD momentum/temperature headers were inspected, not newly validated.
- [Blondel et al. (2022)][implicit], preprint abstract plus local SOLVAX implicit
  contracts; [Optimistix adjoints][adjoints] and [JAX FAQ][jax-where]: accepted
  residuals, safe elementary derivatives and explicit matrix-free tangents.
  SOLVAX source inspection: `5a49926992fe1a3aebac4b8b8cb098798e977c14`;
  no source changes or full upstream suite rerun.
- [JAX profiling][jax-perf], official guidance: synchronized timing/memory
  provenance; a trace, not a loop keyword, determines the optimization.

[prior]: https://github.com/uwplasma/LMX/blob/d18fefce2eb241b4c8d8bc733e914e530e8ce4d4/plan.md
[axis]: https://art.torvergata.it/handle/2108/53061
[oud]: https://repository.tudelft.nl/record/uuid:b9c6eb37-7f80-4f6e-83c2-d1a0ae810d2b
[pipe-heat]: https://doi.org/10.1017/S0022112071001502
[enhancement]: https://doi.org/10.1016/S0920-3796(89)80118-8
[low-pr]: https://doi.org/10.1017/jfm.2023.387
[heat-textbook]: https://ahtt.mit.edu/wp-content/uploads/2024/04/AHTTv600.pdf
[chaos]: https://arxiv.org/abs/1401.4163
[parastell]: https://www.frontiersin.org/journals/nuclear-engineering/articles/10.3389/fnuen.2024.1384788/full
[thermal-benchmark]: https://doi.org/10.1088/1741-4326/ae0800
[implicit]: https://arxiv.org/abs/2105.15183
[adjoints]: https://docs.kidger.site/optimistix/api/adjoints/
[jax-where]: https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where
[jax-perf]: https://docs.jax.dev/en/latest/201/profiling.html
[ni]: https://doi.org/10.1016/j.jcp.2007.07.025
[vv]: https://doi.org/10.1016/j.fusengdes.2014.04.049
[gci]: https://doi.org/10.1115/1.2960953
[sm82]: https://doi.org/10.1017/S0022112082001177

## 12. Resume log

Keep at most ten compact entries: source, command/result, limitations and next
action. Detailed logs belong in the PR.

| Date | Work / evidence | Next |
|---|---|---|
| 2026-09-06 | Main review; 508 tests pass but local coverage source ambiguity; curated test passes; #63 finite forward/reverse-NaN diagnostic; primary literature/upstream inspection; fresh clone 3,468 KiB; README 317→183 with four executed blocks; docs/architecture/local links pass | Review this proposal, reconcile #64, then A: reverse residual and #63 qualification; B: pressure/flow inverse tutorial |

Resume from this branch, not an assumed workspace path. Inspect git status and
current PR heads/checks; preserve unrelated edits. Local artifacts:
`artifacts/plan-review-junit.xml`, `artifacts/plan-review-curated.xml`;
failed aggregate coverage produced no usable XML. Detached #63 diagnostic:
`/tmp/lmx-plan-pr63-audit` (temporary, not durable evidence).
Commands used:

```console
python scripts/run_full_test_suite.py --coverage-xml artifacts/plan-review-coverage.xml --junit-xml artifacts/plan-review-junit.xml
python -m pytest tests/test_example_runner.py -m curated --no-cov
python scripts/audit_architecture.py
python -m sphinx -W -b html docs docs/_build/html
```

Before repeating coverage, confirm its source is src/lmx, not a stale root
directory; do not delete another checkout's data. Do not rerun numerical tests
for prose-only edits; validate changed snippets/links/docs. Qualify exact-head
coverage at integration. No numerical implementation, thermal result, GPU rerun,
external production validation or release occurred in this planning task.
