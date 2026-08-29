# LMX product plan

**Status:** active

**Last updated:** 2026-08-29

**Purpose:** product goal, executable roadmap, decision register, and work log

This file is the authoritative product plan and engineering log for LMX.
README, documentation, examples, names, and code comments describe only the
supported current state.

## Execution reset: shortest path to research-grade LMX

Decision D-049 replaces serial release qualification after every edit with a
staged, change-aware evidence process. Repository compactness, the public API,
README, and documentation structure are already within their product budgets;
they are no longer the critical path. Dedicated line-count-only tranches stop
until the physics and accelerator blockers below are closed. Code encountered
while closing those blockers is still simplified in place, and generic
numerical algebra still moves to SOLVAX when it has an independent contract.

Work proceeds in this order:

1. **P0 — terminal B2 primal and adjoint.** Establish a conservative B2 map
   that reaches the declared steady gate on the production mesh, then implement
   its implicit tangent/transpose solve in SOLVAX and verify Taylor, JVP/VJP,
   residual, runtime, and memory contracts. This is the methods-paper blocker.
2. **P0 — matched B1 external validation.** Run an independently matched pipe
   case with the same equations, annulus, drive, controls, and observable. Do
   not expand the solver family while this validation contract is incomplete.
3. **P1 — fusion-design demonstration.** Use the verified 3-D adjoint in one
   bounded field/wall/geometry optimization and validate selected gradients and
   Pareto points independently.
4. **P1 — deferred accelerator evidence.** After the CPU physics and derivative
   contracts pass, restore the office execution path and measure cold compile,
   warm primal, warm gradient, transfer, peak memory, and one-to-two-GPU strong
   scaling on the same accepted workload. Emulated CPU devices remain
   correctness tests only; GPU evidence is not a blocker for the current CPU
   tranche and no GPU-performance claim is made before these measurements.
5. **P2 — publication evidence and final release.** Generate the final plots,
   movie, performance tables, validation matrix, and concise current-state
   documentation; then perform the one final history rewrite and public release.

### Finalization ledger

No new solver family, example, public API, documentation architecture, or
standalone trimming tranche enters the finish queue. A change must close one
of the gates below or remove work from its critical path.

| Order | Deliverable | Current state | Completion evidence | Stop condition |
|---:|---|---|---|---|
| 1 | Terminal CPU B2 primal | **active**: exact step-152 restart is conservative but momentum defect is `0.13780740` | declared coarse-mesh momentum, mass, charge, pressure, restart, runtime, and peak-memory gates | promote a bounded Schur/block method only if it improves physical defect per second; otherwise delete it |
| 2 | B2 implicit derivative | blocked only by gate 1 | tangent/transpose residual, Taylor, JVP/VJP, finite difference, warm runtime, and peak memory on the accepted equation | no differentiation through nonlinear or Krylov histories |
| 3 | Matched B1 external validation | case construction required: the pinned FreeMHD source has no ready ALEX B1 pipe case | independently executed identical equations, annulus, drive, controls, mesh study, and pressure/flow observable | do not relabel the existing internal B1 benchmark as external evidence |
| 4 | Fusion-design demonstration | implementation exists; production evidence waits for gate 2 | one bounded field/wall/geometry study with independently checked gradients and selected Pareto points | use LMX objectives and an external optimizer; add no optimizer framework |
| 5 | GPU parity and scaling | explicitly deferred until gates 1--4; office runner unavailable | synchronized one-/two-A4000 primal and gradient measurements with memory and collective counts | make no GPU speed or scaling claim from emulated devices |
| 6 | Release qualification | blocked by gates 1--5 and GitHub billing | one commit passes complete validation, clean install/clone, docs, package, citation, and release artifact checks | no repeated release matrix before the candidate is feature-complete |

Everything else in this document is specification, completed evidence, or
historical engineering log. The next numerical experiment is exactly one
fixed-work Schur defect-correction candidate using the production frozen
momentum response and retained pressure solve on the reduced fixture. It earns
one exact step-152 comparison only after beating the retained reduced map in
both physical defect and warm time. Failure deletes the experiment and closes
that algorithmic branch; it does not create another user option.

The numerical policy is fixed for these priorities: converged linear and steady
systems use implicit derivatives; finite trajectories use selective
checkpointing rather than storing every iteration; explicit axial sharding
must preserve local arrays and make collectives visible; benchmarks synchronize
device work and separate compilation from warm execution. FreeMHD remains an
independent inductionless finite-volume comparator, not a source of inherited
acceptance or a runtime dependency.

### Evidence cadence

| Boundary | Required work | Target |
|---|---|---:|
| Edit loop | Ruff on touched Python plus exact unit/numerical tests named by the edit | under 60 s |
| Local candidate | `run_full_test_suite.py --changed-from <base>` for the conservative impacted set; no global coverage | under 3 min |
| Source PR, once | complete parallel suite with combined branch coverage; architecture/import audit | under 5 min |
| Documentation-only PR | Sphinx HTML only; no Python, wheel, or Docker rerun | under 2 min |
| B2/validation candidate | pinned FreeMHD smoke once after the source candidate is clean | under 2 min with the cached image |
| Release/scheduled | all Python versions, links, package/Twine/clean-wheel, production FreeMHD, GPU/scaling, artifact checksums | under 10 min excluding declared production campaigns |

The full suite is not rerun after an evidence-only `plan.md` commit. Package
builds are required only when package metadata/distribution changes or at the
release boundary. External link checks run scheduled/release, not on numerical
source edits. JAX's persistent compilation cache is enabled outside the
repository for local test processes; cold performance evidence explicitly
disables it. Unknown executable paths fail closed to the complete suite.

## Active physics and performance tranche

The live root is the enforced repository baseline. Commit `d4dcc9b` contains
87 tracked files, 1,847,518 bytes of tracked data, 15 package modules, 14,793
package lines, 11,950 test lines, and 28 root exports. Its latest complete
qualification passes 501 tests with 95.24% combined line/branch coverage; the
latest B2-affected gate selected 211 tests and completed in 84.0 seconds end to
end. Every change below must preserve the
normal-clone limit of 9,766 KiB and the file, API, five-minute test-time, and
coverage budgets. The capability-adjusted ceiling remains 15,370 package lines
across at most 16 modules.

The tranche has six coupled outcomes:

1. **Three-dimensional imposed fields.** Rectangular, layered, straight-pipe,
   and mapped-pipe cases use one extruded inductionless formulation for
   constant, analytic, tabulated, fringe, and localized divergence-free
   fields. Manufactured operators, charge and mass closure, projection,
   restart, mesh convergence, and B1/B2 observables are required.
2. **Q2D flow.** Add one deliberately small depth-averaged incompressible MHD
   path for strong transverse fields. It must reuse the common case/result,
   field, diagnostics, output, JAX runtime, and SOLVAX interfaces; it must not
   restore a parallel configuration, plotting, campaign, or solver framework.
   Acceptance requires decaying-flow identities, divergence and energy gates,
   mesh convergence, CPU/GPU parity, and a named analytical, published, or
   external-code comparison.
3. **Accelerated execution.** CPU and GPU results must agree at declared
   float64 tolerances. On a production B1, B2, or Q2D workload that amortizes
   compilation, one RTX A4000 must be at least 2x faster than the audit CPU in
   warm wall time. Compilation, transfer, warm solve, peak host/device memory,
   iterations, and terminal physics residuals are reported separately.
4. **Parallel scaling.** Decompose the largest physical axis with JAX sharding
   while keeping one-device execution unchanged. On the two office A4000s,
   the selected production case targets at least 60% strong-scaling efficiency
   from one to two GPUs and must reproduce the one-device observables within
   numerical tolerance. Report results honestly if the workload is too small
   or communication-bound; never label emulated devices as scaling evidence.
5. **Engaging evidence.** Publish reproducible field, current, velocity,
   pressure, convergence, parity, and scaling plots plus at least one 3-D or
   Q2D movie. The repository keeps only essential WebP/SVG posters within the
   500 KiB media budget; MP4, full-resolution images, checkpoints, and field
   arrays are checksum-addressed release assets generated by a supported
   example or validation command.
6. **Differentiable design.** Continuous geometry, field, material, forcing,
   and state parameters must reach retained Q2D, 2-D, and 3-D field outputs and
   scalar blanket/stellarator objectives through the production discretization.
   Steady solves use implicit tangent/adjoint solves; long trajectories use an
   exact discrete adjoint with bounded checkpoint storage. Derivative accuracy,
   warm runtime, peak memory, CPU/GPU parity, and sharded collective behavior
   are release evidence, not inferred from JAX compatibility.

External validation is fail-closed. B2 retains the pinned, independently
observed FreeMHD Docker comparison and adds production-mesh evidence. B1 is not
called externally validated until LMX and an independently executed solver use
matched equations, geometry, wall conductivity, forcing, controls, and
observables. FreeMHD and OpenFOAM source may inform discretization and case
construction, but copied algorithms enter LMX only when their license,
mathematical contract, numerical benefit, and maintenance cost are explicit.

## Research position and publication program

LMX will lead in a deliberately narrower space than a general CFD code:

> **LMX is the compact, accelerator-native, differentiable inductionless
> liquid-metal MHD solver for duct, access-channel, and fringing-field design
> loops in fusion blankets and stellarators.**

The distinguishing deliverable is not merely that JAX can trace the code. It
is a verified map from continuous field, forcing, material, wall, and supported
geometry parameters to conservative production fields, engineering observables,
and gradients, with explicit primal/adjoint residuals, bounded reverse memory,
and independent physics validation. This supports thousands of design points
and gradient-based optimization while retaining a path to selected 3-D
high-fidelity comparisons.

### Landscape boundary

| Code or ecosystem | Established strength | Boundary for LMX | Useful relationship |
|---|---|---|---|
| ParaStell | Parametric stellarator in-vessel CAD, VMEC/first-wall input, radial builds, DAGMC/OpenMC neutronics | LMX does not own CAD, meshing arbitrary blanket solids, or neutronics | Consume versioned field/centerline/cross-section samples and return hydraulic, electromagnetic, and sensitivity observables to a device-design workflow |
| DESC | Differentiable pseudo-spectral 3-D equilibrium, coil, and stellarator optimization with verified force-balance objectives | LMX does not solve plasma equilibrium or duplicate DESC's optimization framework | Consume equilibrium magnetic fields and smooth geometry controls; return differentiable liquid-metal hydraulic/electromagnetic constraints so a coupled design loop differentiates each model at its natural boundary |
| NekRS and SAM | NekRS provides massively scalable high-order GPU CFD; SAM provides fast system models and MHD closures | LMX does not claim exascale arbitrary-geometry CFD or plant transients | Use matched duct/fringe cases for code comparison; publish when LMX gradients and design throughput add information not supplied by a primal CFD solve |
| FreeMHD / FreeMHD2 | OpenFOAM finite-volume free-surface, multi-region, and now full-induction liquid-MHD physics | LMX remains inductionless and internal-flow focused; it does not chase free surfaces or full induction without a fusion design requirement | Maintain a pinned inductionless oracle, add a separately pinned current-source comparison, and exchange common dimensional observables |
| JAX-Fluids, XLB, and JAX-CFD | Differentiable accelerator CFD, compressible/multiphase or lattice-Boltzmann methods, and scientific-ML workflows | LMX does not become a general differentiable CFD framework | Adopt evidence standards for multi-device execution, end-to-end gradients, profiling, and reproducible examples while keeping liquid-MHD equations and validation unique |
| OpenFOAM MHD solvers and legacy HIMAG/MTC-H/ALEX codes | Broad finite-volume precedent and historical high-Hartmann validation | LMX does not copy solver families or inherit claims through similar equations | Use manufactured, analytical, experimental, and independently executed case contracts as external oracles |

The public literature already covers scalable primal CFD, differentiable
general CFD, free-surface liquid MHD, and parametric stellarator neutronics.
LMX therefore must not claim novelty from Python, GPUs, automatic
differentiation, or inductionless equations alone. A defensible contribution
requires the combination below and quantitative evidence against the relevant
alternative.

### Research-grade claims and gates

| Research claim | Evidence required before publication |
|---|---|
| A production liquid-MHD discrete adjoint is accurate and memory efficient | Component and end-to-end Taylor tests; centered differences; JVP/VJP duality; primal and transpose residuals; iteration-independent implicit memory; checkpoint scaling over at least four trajectory lengths; float64 CPU/GPU parity |
| High-Hartmann wall and fringing physics are resolved deliberately | Hartmann/Shercliff/Hunt layer-aware refinement; observed order away from singular corners; charge, mass, momentum, and energy closure; monotone convergence of pressure loss and jets; declared limits for under-resolved layers |
| Gradients improve a fusion-relevant design rather than a toy loss | At least one bounded pressure-loss/flow-uniformity/wall-current optimization over an imposed-field profile, wall conductance, or smooth duct geometry; independent finite-difference checkpoints; baseline/Pareto comparison; physical constraints and uncertainty bands |
| Accelerator execution enables useful design throughput | Cold compilation, warm primal, warm gradient, peak device/host memory, and transfer time reported separately; one-GPU speedup on an amortized production problem; two-device strong scaling and gradient communication counts; batched-design throughput |
| LMX predicts a measured or independent 3-D liquid-MHD response | Matched equations and dimensional inputs; mesh and solver convergence; native-output observation; B2 production comparison and a genuinely matched B1 or modern equivalent; discrepancies carried into the conclusion rather than tuned away |

Every derivative claim includes the derivative of the accepted discretization,
not merely the continuous equations or a shadow approximation. Every
performance claim fixes hardware, precision, software versions, compilation
policy, repetitions, and physical residuals. Every validation figure is
generated from a manifest containing source commits, inputs, checksums, and the
command that produced it.

### Publication ladder

1. **JCP methods paper — differentiable high-Hartmann liquid MHD.** Complete
   rectangular/layered/pipe discrete adjoints, profile/material/geometry design
   variables, primal and adjoint verification, checkpoint-memory theory,
   convergence, and CPU/GPU performance. Compare algorithms and observables
   with FreeMHD and, where executable, NekRS/SAM. The paper contribution is a
   validated differentiation and performance methodology for constrained
   inductionless MHD, not a broad multiphysics claim.
2. **JPP or Physics of Plasmas physics paper — fringing-field sensitivity and
   topology of recirculation/current paths.** Use verified gradients and
   continuation across $Ha$, $N$, wall conductance, and field-gradient length
   to explain pressure loss, jets, stagnant zones, and current closure. Anchor
   the regime map to ALEX B1/B2 or another matched experiment/code.
3. **Nuclear Fusion or Fusion Engineering and Design application paper —
   optimization-ready blanket access ducts.** Couple versioned magnetic-field
   and geometry samples from VMEC/ParaStell-style workflows to LMX; optimize a
   manufacturable duct/field/wall parameterization for pressure drop, flow
   distribution, wall current, and pumping power; validate selected Pareto
   points with an independent high-fidelity solver.

These papers are sequential evidence products, not simultaneous feature
promises. The methods paper comes first because later physical and design
conclusions depend on its gradient and discretization credibility.

### Academic and industry adoption gates

- one dimensional SI-unit case schema with explicit material-temperature
  provenance and no silent nondimensional defaults;
- stable Python field/objective API plus TOML/CLI workflow and semantic
  deprecation policy after the public release;
- deterministic restart, machine-readable convergence/failure reasons, and
  compact VTK-compatible or XDMF-compatible field export without adding a
  visualization framework to the runtime;
- versioned validation matrix linking every capability to analytical,
  manufactured, experimental, or external-code evidence;
- ensemble and batched-design interface that composes with JAX optimization
  libraries without LMX owning an optimizer;
- uncertainty and robustness examples for field/material tolerances;
- archived release evidence sufficient to reproduce every quantitative claim
  from a clean environment.

Near-term exclusions are equally important: free surfaces, full induction,
turbulence/LES, conjugate heat transfer, tritium transport, corrosion, CAD,
neutronics, and plant systems remain external until a validated coupling study
shows that adding one is necessary and can preserve the compact product.

## Product definition

LMX will be a compact JAX library for inductionless liquid-metal MHD in ducts,
including fully developed cross-section models and actively developed 3-D
fringing-field models. It will make the common case obvious, keep advanced
configuration composable, and delegate reusable numerical algebra to SOLVAX.

The supported product will include:

- fully developed laminar Hartmann, Shercliff, and Hunt duct flow;
- rectangular and layered cross-sections on structured meshes;
- insulating and explicitly conducting wall regions;
- prescribed constant, analytic, and tabulated transverse magnetic fields;
- pressure-driven and fixed-flow steady or transient solves;
- extruded 3-D rectangular and straight-pipe domains with spatially varying
  imposed magnetic fields, including fringe entry and exit regions;
- 3-D electric-potential, current-continuity, Lorentz-force, momentum, and
  pressure/projection coupling needed by the fringing-field model;
- charge, current, power, convergence, and analytical validation diagnostics;
- reproducible 3-D validation against analytical/manufactured cases and a
  pinned FreeMHD Docker reference workflow;
- restartable solves and compact NPZ/JSON output;
- end-to-end differentiable production fields and design objectives for
  continuous state, material, field, forcing, and supported geometry inputs;
- CPU and accelerator execution through JAX and SOLVAX.

Generality will come from a small set of orthogonal objects—geometry, mesh,
materials, magnetic field, walls, drive, dimensionality, and solve options—not
from separate solver families for every experiment. The 2-D and 3-D paths will
share models, field definitions, coefficients, diagnostics, result semantics,
and SOLVAX primitives wherever their mathematics is identical.

LMX will trim by evidence, not by module size. The following are protected
capabilities during the refactor:

- the 3-D extruded/fringing-field formulation and its rectangular and
  straight-pipe validation cases;
- the pressure/projection and variable-coefficient operations required by
  that formulation;
- the minimal FreeMHD input, execution-contract, output-observation, and
  comparison code required for reproducible parity tests;
- field and geometry machinery used by a retained 3-D validation case.

No protected implementation is deleted until its replacement passes the same
physics, convergence, restart, performance, and external-parity gates. Generic
linear algebra inside it may move to SOLVAX, and duplicated campaign/report
machinery may be collapsed without removing the capability.

Q2D flow, magnetic obstacles, and additional 3-D imposed-field applications
are active development surfaces. A lane stays only if it has an
owner, a clear physical contract, an executable numerical test, and a
documented validation path. Compatibility shims, alternative solvers already
supplied by SOLVAX, dashboards, paper pipelines, one-off media generators, and
duplicated campaign infrastructure are not protected.

FreeMHD remains an external validation dependency, not an LMX runtime
dependency. Its pinned Docker workflow, minimal cases, scalar/field observers,
and tolerances live in a dedicated non-wheel validation area. Generated
OpenFOAM trees, images, logs, and field dumps remain untracked.

## Capability matrix

This matrix is the deletion authority for Phase 0. A capability marked
**protected** cannot be removed or reduced to a proxy. An **audit** capability
remains intact until its acceptance decision is recorded here.

| Capability | Mathematical/user contract | Current executable evidence | Current cost | Decision and Phase 0 gate |
|---|---|---|---:|---|
| Fully developed 2-D duct MHD | Inductionless Hartmann, Shercliff, and Hunt flow; rectangular/layered walls; pressure or flow drive; steady/transient | analytical profiles, conservation/power gates, high-Ha rows, closed-channel observations in `test_solver.py`, `test_physics.py`, and `test_reference_data.py` | shared core | **protected, production**; preserve canonical outputs and convergence metadata |
| Rectangular/layered 3-D fringing | Variable imposed field, electric-current closure, Lorentz force, momentum and face-flux pressure projection on extruded grids | manufactured operators, projection/divergence, restart, variable-field and B2 square-duct tests in `test_fringing.py`; B2 spec/reference | shared 3-D core | **protected, active development**; pass reduced 3-D gates and executed B2 FreeMHD smoke after every structural tranche |
| Straight-pipe 3-D fringing | O-grid cylindrical metrics, conducting annulus, fixed-flow projection, variable field | manufactured pipe Poisson/diffusion/projection tests, B1 spec/reference, retained-modal benchmark | shared 3-D core | **protected, active development**; preserve B1 internal/manufactured gates and establish a matched external parity claim before promotion |
| FreeMHD parity | Matched equations, geometry, material, forcing, controls, native-output observation, and declared tolerances | B2 deterministic contract, two-update Docker harness, independent observers, and parser/forgery tests in `test_freemhd.py` | `freemhd.py` plus one external runner | **protected validation boundary**; retain the pinned B2 Docker path and add B1 only when its equations and observables can be matched end to end |
| Bent-pipe flow | Curvilinear momentum/current equations with curvature-driven secondary flow | the removed lane generated curved display points but solved straight cylindrical equations and labeled MHD transverse motion as Dean observables | none in live tree | **removed as a proxy**; restore only with curvature metrics in every production operator, manufactured tests, mesh convergence, derivatives, and a named external target |
| Magnetic obstacle | Localized imposed field in a 3-D rectangular channel with velocity, pressure, current, and wake observables | internal field-response and baseline tests through the common 3-D solver | shared 3-D core | **retained as a 3-D field application**; add executable external data before a quantitative validation claim |
| Q2D MHD | Depth-averaged incompressible inductionless flow with transverse-field damping and optional forcing | analytical decay/energy identity, spectral incompressibility, nonlinear three-grid refinement, CPU/GPU parity, reproducible poster/movie example | one 284-line module plus existing shared API/docs/tests | **protected, active development**; keep the compact SM82 path and do not restore a separate configuration, campaign, plotting, or solver framework |
| Branded mirror-pipe adapter | Product-specific proxy around generic tabulated fields and pipe fringing | archived source and tests | none in live tree | **removed**; generic tabulated/vector-field and straight-pipe capabilities remain available for a future matched case |
| Blanket reduced flow | Separate 1-D pressure-budget and filling model | archived source, tests, and media | none in live tree | **removed**; the standalone reduced model had no external validation owner and did not share the retained 3-D solver |
| Differentiable design | Gradients from continuous design inputs through the retained Q2D, 2-D, and 3-D discretizations to fields and blanket/stellarator objectives | Hartmann finite-difference/JVP/VJP gates; Q2D analytical JVP/VJP and reverse-memory gates; 3-D production-path audit active | traceable physics core plus released SOLVAX adjoints | **protected product invariant**; no surrogate or primal-only production lane, and no end-to-end claim for a path until its derivative cost and accuracy gates pass |
| Reusable solver algebra | Krylov, fixed point, structured direct solves, preconditioners, projected/nullspace algebra | LMX unit/manufactured tests and overlapping SOLVAX APIs | portions of `solvers.py`/`fringing.py` | **move to SOLVAX when general**; retain LMX coefficient assembly and physics gates |

Current costs overlap where capabilities share modules; they are navigation
estimates, not additive budgets. Each audit decision must name the accountable
owner, retained public workflow, numerical gate, external or analytical
reference, and maintenance cost.

The accountable owner for retained 3-D geometry/application rows is the LMX
3-D/fringing maintainer. Their common workflow is one `FringingCase` and one
`solve` result; they do not retain separate solver families. Magnetic-obstacle
gates are imposed-field Maxwell checks, current/flow conservation, mesh
convergence, and an executable
Votyakov/Andreev comparison before any quantitative claim. Until those final
external gates exist, the capabilities may remain documented as development
applications but not as validated benchmarks.

## Reusable-algebra ownership map

LMX owns physical coefficients, boundary/interface equations, dimensional
scaling, convergence gates, and MHD observables. SOLVAX owns reusable solver
algorithms and algebraic diagnostics. This table is the Phase 1 deletion map.

| Current LMX surface | Owner | Action | Replacement/gate |
|---|---|---|---|
| Rectangular and pipe 3-D Poisson paths | SOLVAX algorithm, LMX operator | **replaced** | matrix-free `solvax.pcg_linear_solve`; the validated collocated projection composes `solvax.fixed_point_iteration` with the LMX stencil and physical residual |
| `solve_five_point_solvax_pcg_state` | SOLVAX algorithm, LMX residual contract | **deleted** | `solvax.pcg_linear_solve` is composed inside one measured JIT boundary with the MHD coefficient action and physical max-norm certification |
| `solve_poisson_cg_state` | SOLVAX algorithm, LMX gauge/scaling | merge into MHD solve | direct `pcg_linear_solve`; retain only anchor projection, volume scaling, and physical residual in LMX |
| `solve_poisson_jacobi_state` and 3-D Jacobi loops | SOLVAX | **replaced** | released `solvax.fixed_point_iteration` owns relaxation, stopping, and iteration state; LMX retains stencil maps, gauges, and physical residuals |
| five-point/Poisson coefficient application | LMX | keep and merge into operators | it is the action of LMX-assembled boundary-aware coefficients, not a general solver |
| five-point/Poisson physical residual norms | LMX | keep and merge into validation/solve | acceptance normalization is part of the MHD convergence contract |
| 2-D line, additive, and deflated preconditioner builders | shared boundary | reduce to composition | LMX forms coefficient lines/coarse restriction; SOLVAX supplies tridiagonal solves, additive composition, and deflation |
| 2-D fast-diagonalization/generalized modal inverse | candidate SOLVAX | upstream only after benchmark | reusable separable structured inverse API with factor reuse, adjoint/JIT tests, and performance evidence |
| 3-D axial-mean/transverse-modal and pipe retained-modal factors | candidate SOLVAX | upstream common algebra | generic projected/nullspace and separable factor APIs; LMX retains geometry metrics, gauge, and boundary assembly |
| Aitken, Anderson, GMRES/PCG, block Thomas, tridiagonal solves | SOLVAX | remove local wrappers/switches | released SOLVAX APIs already used; retain MHD-specific stopping and result translation only |
| small fixed-flow response matrices | LMX | keep | geometry/constraint closure of at most a few degrees of freedom, not a reusable solver implementation |
| shadow 3-D/WHAM/blanket autodiff solvers | neither | delete | proxies do not differentiate the retained production path and must not be upstreamed |

## Non-negotiable outcomes

### Size and structure

All measurements are from a clean checkout with environments, caches, build
outputs, and generated artifacts excluded.

| Measure | Baseline | Completion target |
|---|---:|---:|
| default full-clone disk size | 43,372 KiB | **< 9,766 KiB** (< 10 MB) |
| `.git` in default full clone | 38,672 KiB | **< 6,500 KiB** |
| tracked checkout | 4,428,404 bytes / 182 files | **<= 3,500,000 bytes / <= 120 files** |
| package source | 30,852 lines / 32 modules | **<= 15,370 lines / <= 16 implementation modules** |
| largest source module | 7,889 lines | **<= 1,800 lines; target <= 1,200** |
| tests | 19,669 lines / 28 files | **<= 12,000 lines / <= 14 files** |
| maintenance scripts | 8,297 lines / 13 files | **<= 4 scripts; target <= 2** |
| runnable examples | 23 tracked files | **<= 8 files** |
| authored documentation | 40 tracked files | **<= 26 pages** |
| tracked documentation media | 1.23 MiB | **<= 500 KiB total; no file > 250 KiB** |
| built wheel | about 290 KiB | **<= 400 KiB, target <= 300 KiB** |
| public top-level API | 30 exported names | **<= 30 deliberate names** |
| portable PR test gate | 150.9 s | **<= 90 s on the audit host** |

Line count is a design signal, not permission to compress readable code. A
change only counts as simplification when it deletes an ownership boundary,
duplicate path, parameter, state, abstraction, or unsupported behavior.

The default-clone target refers to a normal `git clone`, not `--depth 1`, Git
LFS pointers, or partial-clone filters. The current depth-one clone is 6,832
KiB, which shows that the target is achievable after the live tree is slimmed
and history is archived/squashed.

### Correctness

- Every returned steady result reports `converged`, `status`, terminal gate
  values, and iteration counts.
- A steady CLI run exits nonzero when convergence is not reached.
- Nonfinite linear or nonlinear state fails closed; numerical state is never
  repaired with `nan_to_num`.
- Hartmann, Shercliff, Hunt, charge closure, and power balance retain explicit
  numerical gates.
- Retained 3-D cases gate divergence, charge closure, boundary conditions,
  stationwise flow, pressure behavior, momentum balance, and mesh convergence.
- The FreeMHD Docker comparison pins source/image identity, records matched
  equations and controls, and compares declared observables at stated
  tolerances; a parser-only fixture cannot substitute for an executed run.
- The regenerated high-Hartmann campaign passes from the released source and
  records the exact LMX and SOLVAX versions.
- All accepted gradients pass independent finite-difference or transpose
  checks.
- Deleted features leave no public import, option, documentation claim, frozen
  acceptance claim, or compatibility proxy.

### Differentiability

- End-to-end means that continuous initial state, forcing, material properties,
  imposed-field parameters, supported geometry parameters, and objective
  weights can be traced through the retained production equations to final
  fields or a scalar objective. Mesh sizes, topology, iteration limits,
  checkpoint widths, status strings, logging, I/O, and branch-changing events
  are explicit static or host-side controls, not falsely advertised as smooth.
- Each public physical path has one array-valued numerical core that is safe
  under `jit`, `jvp`, `vjp`, `grad`, `vmap`, and applicable sharding. Convenience
  case/result assembly may validate inputs and materialize host diagnostics but
  must call that same numerical core.
- Converged linear and steady nonlinear systems use the implicit-function
  theorem: reverse mode costs one additional transposed solve with explicit
  adjoint tolerances and no tape of Krylov, fixed-point, or Newton iterations.
- Transient Q2D and 3-D paths differentiate the finite timestepping scheme.
  Their default reverse pass uses checkpointing/rematerialization with a stated
  `O(N/C + C)` state bound; a full trajectory tape is never the production
  default. Any custom continuous adjoint must be separately discretized and
  validated before replacing the exact discrete derivative.
- Every accepted derivative is checked against an analytical derivative,
  central finite difference, complex step, dense oracle, or transpose identity
  independent of the rule under test. Float32 and float64 tolerances, primal
  and adjoint residuals, and nonsmooth parameter regions are stated per case.
- Release evidence records primal, JVP, and value-plus-VJP compile/warm time,
  compiled temporary/device memory, recomputation factor, and CPU/GPU agreement
  across increasing mesh and step counts. Reverse trajectory storage must grow
  sublinearly with step count under the default policy; a canonical adjoint may
  not regress by more than 5% without a written accuracy or memory benefit.
- Blanket and stellarator workflows consume general objective callbacks and
  field outputs, not product-specific proxy solvers. Batched design points use
  `vmap`; distributed paths compare primal and adjoint collective counts and
  preserve one-device values and directional derivatives.

### Performance and memory

- Preserve or improve numerical work per solve; deletion alone does not count
  as a speedup.
- No accepted canonical case may regress by more than 5% in warm runtime or
  peak memory without a written accuracy justification.
- The refactor aims for at least a 20% reduction in warm runtime and peak
  memory on the canonical 2-D and reduced 3-D matrices through fewer
  intermediates, factor reuse, larger JIT boundaries, and optional histories.
- Cold compile time, warm runtime, host RSS, device memory, iteration count,
  and terminal residual are measured separately.
- Default results store the final state and compact diagnostics; full field and
  iteration histories are opt-in.
- Derivative benchmarks report the marginal cost over the primal. A default
  adjoint that saves memory by recomputation is accepted only when the measured
  trade-off is documented and materially below the full-tape peak.

### Usability

- A first accurate 2-D duct or 3-D fringing solve requires at most one case
  object and one `solve` call.
- Common defaults are physical and convergent; expert controls are grouped in
  one options object.
- Python and CLI use the same case schema and result semantics.
- Public types, arguments, return values, units, defaults, errors, and array
  shapes are documented and type annotated.
- The full supported API fits on one documentation page and its 2-D and 3-D
  workflows can be learned from four tutorials.

## Ownership boundary: LMX and SOLVAX

The boundary is the same clear model used by MHX:

- **LMX owns physics:** MHD equations, material and wall models, geometry,
  coefficient assembly, physical boundary conditions, coupling, diagnostics,
  validation cases, and user workflows.
- **SOLVAX owns differentiable algebra:** reusable operators, direct and iterative linear
  solves, Krylov methods, fixed-point algorithms, preconditioners, structured
  factorization, checkpointed recurrences, implicit/custom derivatives, and
  solver termination metadata.

SOLVAX must not become a storage location for discarded LMX experiments. Code
moves upstream only when all of these are true:

1. the API is independent of LMX types, MHD terminology, geometry, and units;
2. it represents a reusable numerical method rather than coefficient assembly;
3. it has a second plausible consumer or a clear general structured-algebra
   role;
4. SOLVAX owns focused correctness, transpose/gradient, convergence, derivative
   memory scaling, and primal/adjoint performance tests;
5. the public SOLVAX documentation explains when to use it and its failure
   modes;
6. LMX can delete its implementation and depend on a released SOLVAX version.

### SOLVAX derivative contract

Every public SOLVAX numerical family is classified before LMX may consume it:

| Family | Production derivative | Memory/work contract | Required evidence |
|---|---|---|---|
| Linear/Krylov solve | implicit JVP/VJP through `custom_linear_solve` | one tangent/transposed solve; no iteration tape; explicit adjoint tolerance/preconditioner | dense or analytical gradient, transpose identity, primal/adjoint residuals, collective counts |
| Nonlinear root/fixed point | implicit function theorem through `custom_root` | one Jacobian tangent/transpose solve; no nonlinear history | finite difference on the converged branch, tangent-solve residual, branch/failure documentation |
| Structured direct solve | analytical transpose solve or custom VJP with factor reuse | no differentiated factorization; checkpointed/generated variants state exact storage bound | dense oracle, JVP/VJP where supported, factorization/reverse memory and warm-time scaling |
| Truncated/localized solve | exact retained-state derivative with an explicitly bounded tail approximation | reverse workspace depends on retained window, not full chain | full-window equality, window certificate/convergence, higher-order limitation documented |
| Mixed-precision refinement | implicit adjoint reusing transposed factors | no refinement tape; gradient inherits certified refined accuracy | float64 oracle, conditioning sweep, factorization count, reverse memory/runtime |
| Eigenpair | implicit simple-eigenpair VJP with left/right modes | no eigensolver tape; one bordered/reduced solve | residual/conditioning gate, dense eigenvalue/eigenvector gradient, exceptional-point rejection |
| Long recurrence | exact discrete JVP/VJP with segmented checkpoint replay | `O(N/C+C)` retained state; square-root default | equality to plain finite recurrence, analytical/FD gradient, memory/runtime scaling |
| Dense Jacobian | chunked JVP/VJP basis | peak derivative batch proportional to declared chunk | equality to `jax.jacfwd`/`jacrev`, chunk memory/runtime curve |
| Operators, transfers, preconditioners | traced algebra with an explicit transpose/adjoint when nontrivial | no hidden host conversion or materialization | bilinear adjointness, dtype/JIT/vmap/sharding, directional derivative |
| Host-native bridge/status/I/O | explicitly nondifferentiable orchestration | never inside an advertised traced core | fail-loud transform behavior and a documented differentiable external composition |

Raw iterative routines remain useful primals, but LMX differentiates their
implicit wrapper, not the stopping algorithm. A new SOLVAX algorithm is not
complete when its primal tests pass; its derivative classification, accuracy,
failure behavior, reverse memory, and marginal runtime are part of acceptance.

This policy follows established differentiable-solver practice rather than a
blanket choice of “differentiate every iteration.” Optimistix defaults
converged nonlinear solves to an implicit adjoint and reserves recursive
checkpoint adjoints for derivatives of the finite iteration. PETSc TSAdjoint
uses discrete adjoints plus Revolve checkpoint scheduling for long transient
integrations. JAX-CFD demonstrates the complementary architecture of keeping
FFT/finite-volume field operations in traceable array programs. LMX and SOLVAX
adopt those principles without inheriting another framework or adding a second
physics implementation.

### Migration inventory

| LMX numerical surface | Decision | Destination or replacement |
|---|---|---|
| PCG/GMRES/FGMRES invocation and status | delete wrappers | call released SOLVAX result APIs directly |
| Jacobi, line, and additive preconditioners | delete duplicated builders | `solvax.jacobi` and `solvax.additive_tridiagonal_line_preconditioner` |
| Aitken and Anderson algebra/history weights | delete LMX algebra | SOLVAX fixed-point APIs; LMX supplies the physical map and gate |
| tridiagonal solves and reusable factors | delete local solve code | SOLVAX direct structured solves |
| bordered/Schur projection algebra | delete local generic algebra | `BorderedOperator` and `schur_projected_precond` |
| generic five-point matvec and residual | initially keep one private compact kernel | propose a SOLVAX structured stencil operator only after a second consumer is demonstrated |
| pinned/gauge linear operator composition | upstream candidate | general SOLVAX nullspace/projected-operator API, with no MHD defaults |
| separable Kronecker-sum/fast-diagonalization inverse | upstream candidate | general SOLVAX elliptic/direct API with reusable factors |
| physical residual scaling and gauge-cell choice | keep | LMX solve/validation code |
| conductivity/viscosity face coefficients | keep | LMX operators/physics code |
| charge/current/Lorentz and power identities | keep | LMX physics/validation code |
| time stepping and MHD outer coupling | keep orchestration only | LMX calls SOLVAX fixed-point and linear primitives |
| differentiable root/linear solve machinery | delete local machinery | released SOLVAX implicit APIs |
| 3-D Jacobi, sparse, modal, and projection algebra | split by ownership | reusable solves/preconditioners move to SOLVAX; LMX keeps 3-D MHD assembly and coupling |
| 3-D fringing-field cases, boundary conditions, and diagnostics | keep and refactor | compact LMX physics/API path with no parallel implementations |
| FreeMHD case contract and observable comparison | keep and consolidate | dedicated external-validation area, excluded from the runtime wheel |
| FreeMHD/OpenFOAM execution | keep reproducible, not importable | pinned Docker runner used locally, on schedule, and before release |
| Q2D flow | keep one compact physical path | LMX owns the SM82 vorticity model and diagnostics; released SOLVAX owns periodic Poisson inversion and the exact checkpointed recurrence adjoint |
| Magnetic-obstacle and related field helpers | use shared 3-D path | retain only coherent cases with an executable test and named validation target |
| Blanket and product-specific proxy helpers | remove | archive outside the live package; do not restore parallel solver or campaign frameworks |

### Upstream workflow

For each upstream candidate:

1. write a small SOLVAX issue/RFC with the mathematical contract, shapes,
   symmetry, differentiation semantics, and at least two use cases;
2. build it in SOLVAX with unit, dense-reference, transpose/gradient,
   primal/adjoint runtime, and peak-memory-scaling coverage;
3. merge and release SOLVAX;
4. raise LMX's minimum SOLVAX version to that release;
5. replace LMX code with the direct SOLVAX call and delete its tests/docs;
6. rerun LMX physics, convergence, runtime, and memory gates;
7. never maintain a long-lived copied implementation in both repositories.

LMX will depend on a real released lower bound. It will not claim support for a
version that was not published or lacks a required API.

## Target package

Adopt the PyPA `src` layout so tests exercise the installed package rather than
an accidental repository-root import.

```text
src/lmx/
├── __init__.py       # explicit public API only
├── __main__.py       # CLI entry
├── model.py          # 2-D/3-D cases, options, results, errors, units
├── mesh.py           # shared structured mesh and geometry definitions
├── fields.py         # constant, analytic, tabulated, and fringing fields
├── physics.py        # materials, currents, Lorentz force, power
├── operators.py      # 2-D finite-volume coefficient assembly
├── operators3d.py    # 3-D MHD assembly and boundary operations
├── solve.py          # shared MHD orchestration around SOLVAX
├── fringing.py       # compact 3-D fringing workflow
├── validation.py     # analytics, conservation, convergence, comparisons
├── io.py             # TOML, NPZ, JSON, restart
├── cli.py            # run, validate, inspect
├── autodiff.py       # supported objectives only
└── plotting.py       # small optional plotting surface
```

This is a target, not a requirement to create empty modules. Merge modules
when ownership remains clear and the merged file stays below the source-size
ceiling. Do not create `utils.py`, `common.py`, `helpers.py`, compatibility
packages, or one-file namespaces.

### Capability audit and trimming

Classify functions and data, not entire large modules, using this table:

| Surface | Initial disposition | Required evidence |
|---|---|---|
| `fringing.py` and `_fringing_types.py` | protect, decompose, and reduce | 3-D manufactured/analytical gates plus FreeMHD parity |
| relevant `field_models.py`, geometry, mesh, and I/O | protect and share across 2-D/3-D | retained fringing cases exercise each path |
| minimal `freemhd.py` observation/contract logic | protect and consolidate outside the runtime API where possible | executed Docker result is independently observed and compared |
| broad `external_validation.py` report/plot helpers | retain only comparison kernels used by a validation gate | deterministic scalar/field comparison tests |
| Q2D, Dean, bent-pipe, obstacle, blanket, and WHAM surfaces | decide in Phase 0 | owner, equations, executable test, validation target, roadmap |
| `scaling.py`, dashboards, report/media generators, campaign adapters | collapse into benchmarks, validation CLI, or delete | one stable user/developer workflow justifies retention |

The protected fringing and FreeMHD-related modules currently account for more
than 11,000 source lines and 4,800 focused test lines. Their size is a reason to
separate physics from algebra and consolidate tests, not evidence that the
capability should disappear. No protected file is deleted wholesale; each
function receives a keep, merge, upstream, or remove decision with a test or
call-site reference.

### Consolidation rules

- One implementation per physical operation and numerical path.
- One case schema family for Python, TOML, CLI, restart, and examples, with
  shared fields and explicit `DuctCase` and `FringingCase` variants.
- One `solve(case, *, options=None)` entry point.
- One typed result model for steady and transient runs.
- One convergence policy used by API, CLI, examples, and validation.
- One output writer; plotting consumes the returned result.
- One source of units and nondimensional definitions.
- No boolean matrix of feature flags; use small typed variants only when
  behavior is genuinely different.
- No parameter that is unused, derivable, duplicated, or relevant only to a
  deleted implementation.
- No nested wrapper that only renames a SOLVAX function or forwards arguments.
- No public name without a documented user workflow.
- No simplified 2-D assumption hidden inside a shared 3-D path; dimensional
  differences are explicit and tested.

## Target public API

The exact names are finalized after the capability audit. The intended 2-D
shape is:

```python
import lmx

case = lmx.DuctCase(
    hartmann=20.0,
    mesh=(64, 64),
    walls="insulating",
)
result = lmx.solve(case)

assert result.converged
result.save("hartmann.npz")
result.plot("hartmann.png")
```

Advanced configuration composes explicit objects without changing the solve
entry point:

```python
case = lmx.DuctCase(
    geometry=lmx.RectDuct(width=0.2, height=0.2),
    mesh=lmx.Mesh(shape=(96, 96), wall_cells=8),
    material=lmx.Material(rho=1_000.0, nu=1.0e-3, sigma=1.0e6),
    field=lmx.MagneticField(by=0.2),
    walls=lmx.ConductingWalls(thickness=1.0e-3, conductivity=5.0e6),
    drive=lmx.FlowRate(0.0388),
)
result = lmx.solve(case, options=lmx.SolveOptions(tolerance=1.0e-9))
```

The same solve entry point handles the retained 3-D fringing workflow:

```python
case = lmx.FringingCase(
    geometry=lmx.ExtrudedDuct(length=1.0, width=0.2, height=0.2),
    mesh=lmx.Mesh(shape=(96, 48, 48), wall_cells=6),
    material=lmx.Material(rho=1_000.0, nu=1.0e-3, sigma=1.0e6),
    field=lmx.FringeField.from_profile(x, by),
    walls=lmx.ConductingWalls(thickness=1.0e-3, conductivity=5.0e6),
    drive=lmx.FlowRate(0.0388),
)
result = lmx.solve(case, options=lmx.SolveOptions(tolerance=1.0e-8))
```

API rules:

- `lmx.__all__` is the complete supported interface.
- Public objects use inline types and NumPy-style docstrings.
- Input types accept appropriate protocols/array-like values; returned types
  are concrete and stable.
- Results expose `converged`, `status`, `steps`, `residual`, compact
  diagnostics, final fields, and timings.
- Non-convergence and numerical failure have distinct typed exceptions/status.
- Defaults choose solver and preconditioner automatically; expert SOLVAX
  controls live under `SolveOptions`, not the common constructor.
- A major release removes unsupported names cleanly rather than shipping a
  compatibility layer that preserves complexity.
- Configuration keys map directly to public model fields; aliases are avoided.

## Code standards

Use standards that improve correctness and reduce tooling:

- PyPA `src` layout and `pyproject.toml` as the single configuration source.
- Ruff as the only formatter, import sorter, and linter. Start with `E4`, `E7`,
  `E9`, `F`, `I`, `UP`, `B`, `SIM`, `RUF`, and `PERF`; exceptions must be
  narrow and explained inline.
- Python 3.10+ syntax, `from __future__ import annotations`, built-in generics,
  and `collections.abc` protocols.
- Complete inline annotations for the public API and a packaged `py.typed`
  marker; private numerical kernels are typed when the annotation clarifies
  shape or ownership.
- NumPy-style public docstrings: summary, parameters, returns, raises, units,
  shapes, notes, and minimal examples where useful.
- Immutable dataclasses or named result records for configuration and outputs;
  dictionaries are not public result schemas.
- Explicit `__all__`; modules and names beginning with `_` are private.
- Pure, vectorized numerical kernels separated from I/O and orchestration.
- Array shapes and units stated at boundaries; validate once at the boundary,
  not repeatedly inside hot kernels.
- JIT the largest stable numerical boundary, avoid redundant nested JITs,
  preserve static shapes, reuse factors/preconditioners, and keep host
  conversions outside compiled code.
- Prefer SOLVAX/JAX primitives over local loops and assembled dense matrices.
- Comments explain mathematical invariants or non-obvious constraints, not
  how the implementation changed.
- No function exists only to make a test possible. Tests exercise public
  behavior or a mathematically meaningful private kernel.
- Avoid abstractions until two live call sites need them.
- Avoid files over 1,800 lines, target 1,200, and functions over roughly 80
  lines; exceptions need a decomposition note in this plan.

Dependency policy:

- core runtime target: JAX, NumPy, SOLVAX, plus `tomli` only on Python 3.10;
- remove direct SciPy use if no supported core path needs it;
- plotting remains an optional extra;
- docs and development tools remain extras;
- follow Scientific Python SPEC 0 for minimum Python/core-dependency support;
- test the declared minimum and current dependency endpoints in CI.

## Test design

Target test tree:

```text
tests/
├── test_api_model.py
├── test_mesh_operators.py
├── test_solver.py
├── test_physics.py
├── test_fringing_model.py
├── test_fringing_solver.py
├── test_validation.py
├── test_freemhd_contract.py
├── test_io_cli.py
├── test_autodiff.py
├── test_package_docs.py
└── integration/
    └── test_freemhd_docker.py
```

Rules:

- Consolidate by user/correctness surface, not one file per source module.
- Parameterize Hartmann/Shercliff/Hunt, fringing profiles, wall, mesh, solver,
  dimensionality, and dtype variants.
- Preserve analytical, manufactured, conservation, convergence, restart,
  packaging, CLI, gradient, and external-parity assertions; delete only tests
  for capabilities rejected by the Phase 0 audit.
- Prefer small real numerical problems over mocks, proxies, and test-only
  implementations.
- Keep the default suite below 90 seconds and 95% combined line/branch
  coverage. Coverage does not replace numerical acceptance.
- Put the complete high-Hartmann campaign behind `lmx validate --full`; run a
  bounded representative in PR CI and the full campaign in release CI.
- Maintain three FreeMHD validation layers:
  1. fast contract/parser tests with small immutable fixtures in every PR;
  2. a local `lmx validate freemhd --docker` smoke case that builds or pulls a
     pinned image, runs FreeMHD, independently observes its native output, and
     compares matched LMX observables;
  3. production-mesh Docker parity and refinement gates in scheduled and
     release CI.
- Mark Docker and production 3-D tests explicitly; they do not silently skip
  when requested, and missing Docker produces a clear prerequisite failure.
- Store small reference scalars/specifications in package data. Put raw fields,
  movies, external solver trees, and large reports in release assets.
- Test that every public symbol is documented and every documentation code
  example executes.
- Test installed wheel/sdist contents rather than only the checkout.

Maintenance scripts will become CLI subcommands or tests where they are part
of the product. Delete one-off audit/report/media scripts. A retained script
must have one stable external workflow that cannot reasonably live in `lmx`
and must replace, not add to, an existing script.

## Performance review

Before changing each hot path, record a reproducible baseline for:

- Hartmann `Ha=20` at the standard example mesh;
- Shercliff and Hunt at representative low and high Hartmann numbers;
- insulating and conducting wall cases;
- fixed pressure and fixed flow;
- a reduced rectangular 3-D fringe-entry/exit case for PR benchmarks;
- the matched FreeMHD square-duct and straight-pipe cases at smoke and
  production meshes;
- one production-path gradient for every retained solver family;
- one cold run and at least five warm runs;
- compile time, warm median/CV, host RSS, device memory, Krylov/fixed-point
  iterations, terminal residual, and physical errors.

Review checklist for every supported kernel:

- Is the operation physics or reusable algebra?
- Is the same array/operator computed more than once?
- Can SOLVAX return the diagnostics already being recomputed?
- Can coefficients, factors, or preconditioners be reused?
- Can an allocation be replaced with a view, fused expression, scan, or
  matrix-free action?
- Is a history or plot-only quantity created when the user did not request it?
- Does a Python loop or callback interrupt JIT execution?
- Is device-to-host transfer happening inside a solve?
- Is a dense matrix materialized where structure exists?
- Does a general option force branches into the common hot path?
- Is reverse mode taping iterations or trajectory state that can instead use
  an implicit adjoint, factor reuse, checkpointing, or a matrix-free transpose?

Optimization is accepted only when numerical and gradient gates pass and the
measured gain survives an independent rerun. Remove rejected implementations
instead of retaining switches.

## Documentation architecture

Use the MHX information hierarchy and visual clarity, with a smaller Diátaxis
tree inspired by VMEX. Keep MyST Markdown, Sphinx, Furo, autodoc, MathJax,
copybutton, and lightweight design cards. Do not use executed notebooks or
commit their outputs; tutorials are Markdown files whose code is tested.

```text
docs/
├── index.md
├── install.md
├── tutorials/
│   ├── first-duct.md
│   ├── conducting-walls.md
│   ├── fringing-field-3d.md
│   └── gradients.md
├── how-to/
│   ├── configure.md
│   ├── restart.md
│   ├── run-on-gpu.md
│   ├── prescribe-a-field.md
│   ├── validate.md
│   └── validate-with-freemhd.md
├── explanation/
│   ├── equations-2d.md
│   ├── equations-3d.md
│   ├── discretization.md
│   ├── solvers-and-solvax.md
│   └── differentiation.md
├── reference/
│   ├── api.md
│   ├── configuration.md
│   ├── cli.md
│   ├── outputs.md
│   └── capabilities.md
├── validation.md
├── papers.md
├── references.bib
└── conf.py
```

Documentation rules:

- `index.md` answers what LMX does, shows one validated visual, runs one small
  example, and routes users to tutorials, how-to, equations, validation, API,
  and papers.
- Equations cover the fully developed and 3-D inductionless systems,
  assumptions, nondimensional groups, boundary/interface conditions, and a
  direct equation-to-code map.
- Discretization explains control volumes, face conductance, charge
  compatibility, gauge, momentum coupling, time integration, and convergence.
- `solvers-and-solvax.md` states the ownership boundary and links each LMX
  algebraic step to its SOLVAX primitive.
- API reference is generated from the deliberate public surface; signatures
  and parameter descriptions are not copied manually.
- Tutorials teach; how-to pages solve one task; reference pages enumerate;
  explanation pages derive and justify.
- Validation states cases, meshes, observables, thresholds, versions, and
  limitations without turning into a project diary.
- FreeMHD validation documents the pinned source/image, matched equations,
  field and material mapping, execution command, observed native outputs,
  tolerances, and reproducibility limits.
- `papers.md` lists the LMX paper/citation when available and the primary
  physics/numerics literature grouped by topic.
- Examples in README, docs, and `examples/` use the same API and are executed
  in CI.
- Tracked visuals are compressed still images derived from accepted results.
  Movies and large data live in release assets and are linked.
- Sphinx warnings, missing references, API coverage, and internal links fail
  CI; external linkcheck runs on a schedule.

## README design

Use VMEX's direct product-first structure and MHX's approachable first run:

1. project name and only live badges;
2. two-sentence purpose and explicit scope boundary;
3. one validated, attractive hero image;
4. `pip install lmx` and accelerator note;
5. a complete Python solve in about 15–20 lines;
6. matching three-command CLI workflow;
7. compact capability/maturity table that includes 3-D fringing and FreeMHD
   validation without overstating their evidence;
8. two or three quantitative validation results linked to full evidence;
9. documentation routing table;
10. citation, license, and concise development commands.

The README will not contain project chronology, rejected approaches, stale
test counts, implementation fingerprints, internal campaign status, or a long
competitor matrix. It presents the supported product by showing a real
successful workflow and being precise about boundaries.

## Current-state prose rule

README, docs, examples, package files, comments, filenames, and user-visible
messages state what exists and how to use it. They do not narrate evolution.

Remove or rewrite prose and names containing historical framing such as:

- `legacy`, `old`, `former`, `previously`, `now`, `migration`, `deprecated`;
- “was removed,” “was rejected,” “remains after,” or commit-by-commit notes;
- version-specific development diaries and stale status snapshots;
- compatibility aliases whose only purpose is an earlier internal API.

Exceptions are this plan's work log, Git history, and GitHub release notes.
Scientific literature history is allowed where it explains the field, not the
repository's evolution.

A repository test will scan user-facing tracked text for banned project-history
language, with narrow allowlists for citations and this file.

## Execution roadmap

Each phase lands as a reviewed, green tranche. Do not combine a semantic API
change, a numerical algorithm change, and a history rewrite in one commit.

### Phase 0 — lock scope, baselines, and safety

- [x] Create a feature branch; do not refactor directly on `main`.
- [x] Restore required GitHub CI. Bounded Python 3.10 shards, exact combined
  line/branch coverage, documentation, external links, and the pinned FreeMHD
  Docker comparison pass on the canonical `uwplasma/LMX` repository.
- [ ] Enable branch protection when the repository is public or its GitHub plan
  supports protection for private repositories. The GitHub API currently
  rejects both branch-protection and ruleset configuration with that explicit
  account-plan requirement.
- [x] Record clean-clone, tracked-tree, wheel, module/file/line, import,
  runtime, memory, and test baselines using reproducible commands. The full
  canonical runtime/memory matrix is stored outside the checkout with its
  environment, commit, repetition policy, stopping state, and physical gates.
- [x] Create a capability matrix for 2-D, 3-D fringing, straight pipe, Q2D,
  Dean/bent-pipe, obstacle, blanket, and WHAM lanes; record owner, equations,
  call sites, tests, evidence, cost, and intended user workflow.
- [x] Freeze canonical 2-D and protected 3-D case outputs and physical
  tolerances before moving or deleting their implementations.
- [x] Reproduce the FreeMHD smoke case from a clean pinned Docker environment;
  record build/run commands, digest or source commit, controls, runtime,
  observables, and tolerances.
- [x] Add fail-closed steady-result/CLI semantics before large deletions.
- [x] Replace nonfinite state sanitation with typed numerical failure. Both
  2-D and protected 3-D solution fields now fail closed; diagnostic matrices
  retain explicit not-available sentinels paired with solver status codes.
- [x] Decide each unprotected research lane from the capability matrix and
  record the rationale in the decision register before deletion.
- [x] Export a checksummed bundle and verify it by independent clone and
  `git fsck`; retain the verified release-asset manifest before deletion.

Exit: baseline artifacts reproduce, CI is green, the archive is restorable,
and supported behavior has explicit gates.

### Phase 1 — establish the SOLVAX boundary

- [x] Map every LMX algebraic function to keep, replace, upstream, or delete.
- [x] Replace wrappers with existing released SOLVAX APIs first.
- [x] Propose only the projected/nullspace and separable structured APIs that
  pass the upstream criteria.
- [x] Implement, benchmark, document, and release accepted SOLVAX additions.
- [x] Bump LMX's SOLVAX lower bound and remove copied implementations/tests.
- [x] Delete `lmx/linear.py` when no unique owner remains. Boundary-aware
  stencil actions and physical residuals live with LMX operators; the remaining
  adapters compose released SOLVAX APIs inside the physical solver module.

Exit: no LMX module implements a general matrix solver, Krylov iteration,
fixed-point algebra, direct structured solve, or generic preconditioner.

### Phase 2 — refactor retained capabilities and trim audited lanes

- [x] Decompose the 3-D/fringing monolith by mathematical ownership while
  preserving one end-to-end path and the frozen gates at every tranche.
- [x] Consolidate FreeMHD code into a minimal case contract, Docker runner,
  native-output observer, and comparison layer outside the runtime wheel.
- [x] Remove duplicated reports, plots, fingerprints, frozen-output trees, and
  campaign adapters that do not contribute to a numerical gate.
- [x] For each research lane rejected in Phase 0, remove its implementation,
  exports, configuration, dependencies, tests, examples, scripts, docs, data,
  and claims together; preserve selected material in the verified archive.
- [x] For each retained research lane, give it the same compact API/result
  semantics and an explicit validation roadmap.
- [x] Verify installed-package discovery and wheel contents against the
  capability matrix.

Exit: 3-D fringing and FreeMHD parity still pass, every remaining lane has a
user and validation contract, and package source is below 18,000 lines before
consolidation.

### Phase 3 — redesign the API and package

- [x] Move to `src/lmx` layout.
- [x] Introduce the single case/options/result/convergence model.
- [x] Collapse duplicate config, cases, core, logging, units, output, and
  validation representations.
- [x] Reduce `__all__` to the documented API.
- [x] Remove aliases, pass-through wrappers, redundant parameters, and boolean
  feature matrices.
- [x] Merge files according to the target ownership map without creating a
  mega-module.
- [x] Add `py.typed` and public type-completeness verification.

Exit: the 2-D first-run, 3-D fringing, and advanced API examples work; package
is <= 16 implementation modules and every public symbol has one documented
purpose.

### Phase 4 — simplify and optimize the supported solver

- [x] Finish the function-level ownership and necessity audit. Repeated
  reachability and production-call audits removed the solver façade, rejected
  lanes, private testbeds, duplicate recurrences, and generic algebra now owned
  by SOLVAX. The retained 6,000-line fringing implementation has one public
  orchestration/API owner plus common, duct, and pipe numerical owners; each is
  reached by protected production, restart, differentiation, B1/B2, or
  validation paths. Fusing them would enlarge the public module or mix
  coordinate systems without removing work. No further line-count-only tranche
  is authorized; code encountered while closing a finalization gate is still
  simplified in place.
- [x] Reuse coefficients, factors, preconditioners, and initial guesses.
- [x] Make full histories opt-in and remove plot-only work from solves.
- [x] Consolidate JIT boundaries and eliminate hot-path host transfers.
- [x] Remove dense/intermediate allocations where matrix-free structure exists.
- [x] Benchmark 2-D and 3-D cold, warm, memory, iterations, and physical errors
  after each change.
- [x] Delete every rejected alternative immediately.
- [x] Expose the Q2D finite evolution as a traced field core and use a released
  SOLVAX exact checkpointed reverse pass. Gate analytical continuous-parameter
  gradients, forcing finite differences, JVP/VJP consistency, and compiled
  memory against a full trajectory tape.
- [x] Replace the fixed-iteration Hartmann-only differentiation wrapper with a
  field core shared by the retained fully developed solve. Its converged
  coupled equations must use a SOLVAX implicit linear/root derivative, with no
  Jacobi or coupling-iteration tape, and pass finite-difference, JVP/VJP,
  residual, memory, and runtime checks. The accepted rectangular Hartmann,
  Shercliff, and layered Hunt path uses `solve_fully_developed_fields`.
- [ ] Complete field-level traced cores for every retained 3-D geometry. The
  rectangular, layered, and straight-pipe production recurrences accept
  continuous forcing, station-wise field, material, and fixed-topology geometry
  controls with implicit electric derivatives and bounded reverse storage.
  Specialized ALEX acceptance paths remain deliberately unavailable to
  differentiation until their production equations pass independent primal,
  derivative, runtime, and memory gates.
- [ ] Complete blanket/stellarator design evidence without adding an optimizer
  framework. Pressure-loss, flow-uniformity, pumping-power, wall-current, and
  recirculation objectives, bounded field/material controls, and chunked batched
  design points are accepted. Smooth coordinate controls, CPU/GPU derivative
  parity, production adjoint memory and warm runtime, uncertainty/Pareto gates,
  and one-/two-device collective counts remain open.

Exit: primal and derivative performance gates pass, no canonical case regresses
>5%, every retained solver family has an independently checked field/objective
derivative, and remaining nonsmooth/static controls are explicit in the API.

### Phase 5 — consolidate tests, examples, and tools

- [x] Build the target <= 14-file behavior-oriented test tree.
- [x] Parameterize duplicated case setup and retain physical assertions.
- [x] Replace maintenance scripts with `lmx validate`, tests, or deletion.
- [ ] Implement the three-layer FreeMHD contract, local Docker smoke, and
  scheduled/release production validation workflow. Contract fixtures and the
  independently observed pinned Docker smoke pass; production refinement is
  blocked on a suitable registered high-memory/GPU runner.
- [x] Keep no more than seven Python examples plus one TOML case, including one
  3-D fringing example and one external-validation example.
- [x] Make all examples fast, self-contained, editable, and CI-executed.
- [x] Enforce source/file/size/media/API budgets in package tests.

Exit: the complete six-worker suite is <= 300 seconds, coverage >=95%, Docker
smoke passes when requested, scheduled/release parity is reproducible, all
examples run, and source/tests/scripts meet their budgets.

### Phase 6 — rebuild docs and README

- [x] Delete the existing user-facing documentation tree and recreate the
  target information architecture from the stable API.
- [x] Write equations and equation-to-code mapping from primary sources.
- [x] Generate complete API reference from docstrings.
- [x] Write and execute four tutorials, including 3-D fringing, and focused
  how-to guides, including local FreeMHD Docker validation.
- [x] Publish validation with regenerated current-source evidence.
- [x] Create the product-first README and one compact validated hero visual.
- [x] Run the current-state prose scan, Sphinx `-W`, API coverage, and links.

Exit: a new user can install, solve, interpret, validate, and extend LMX from
the docs without reading source or this plan.

### Phase 7 — make the repository itself small

Deleting live files will not shrink the current 38-MiB history pack, which is
dominated by generated images, movies, repeated lockfiles, and deleted research
artifacts. Meeting the normal-clone target requires a deliberate history
operation.

- [x] Create a private read-only archival mirror containing every branch, tag,
  release reference, and a checksummed `git bundle`.
- [x] Clone the archive independently and run `git fsck` before rewriting.
- [x] Export release metadata and move large reusable artifacts to releases.
- [x] Create a reviewed squashed root for the slim product.
- [x] Delete old refs from the live repository and force-push only after
  separate explicit approval for this destructive operation.
- [x] Confirm that unreachable objects do not enter an ordinary remote clone;
  no additional GitHub garbage-collection wait is required for the target.
- [x] Measure a new ordinary authenticated clone, not a local/shared clone.
- [x] Record clone disk size, pack transfer size, checkout bytes, and file
  count in this plan.

Exit: `du -sk` of a fresh normal clone is below 9,766 KiB and the archive has
been independently verified. No history rewrite occurs without explicit
approval at execution time.

### Phase 8 — release the standalone product

- [x] Run minimum/current Python and dependency matrices.
- [ ] Run complete 2-D duct, 3-D fringing, FreeMHD parity, conservation,
  convergence, gradient, packaging, documentation, and full validation gates.
- [x] Build and inspect wheel/sdist contents and sizes.
- [ ] Publish one coherent major release from the same reviewed commit.
- [ ] Verify clean install, first-run Python example, CLI, docs, citation, and
  fresh clone.

Exit: release artifacts, documentation, examples, numerical evidence, and
source all describe the same standalone API and commit.

## CI gates

Until GitHub Actions billing is restored, the same gates run locally and are
the merge authority. A local merge candidate must record the commit, Python,
JAX/JAXLIB, SOLVAX, platform, commands, elapsed time, combined branch coverage,
and generated evidence checksums in this log. At minimum it runs Ruff check and
format, the architecture/import audit, the complete non-curated test suite with
branch coverage and per-test timeouts, all curated examples, Sphinx with
warnings as errors, package build/Twine inspection, a fresh non-editable wheel
smoke, external links, and the pinned FreeMHD Docker comparison when Docker is
available. A local failure is handled exactly like a required hosted failure;
no check is waived because the hosted runner is unavailable.

PyPI and the public GitHub release are deliberately deferred until the final
public-repository phase. Local version tags before then identify immutable
release candidates, not published distributions.

Every pull request must pass:

```console
ruff check lmx tests examples scripts docs/conf.py
ruff format --check lmx tests examples scripts docs/conf.py
pytest -q -m "not docker and not production"
sphinx-build -W -b html docs docs/_build/html
python -m build
twine check dist/*
```

Add installed-wheel smoke, minimum/current dependencies, API/docs coverage,
source budgets, and current-state prose checks to the same required workflow.
The normal PR gate runs reduced 3-D physics cases and immutable FreeMHD
contract fixtures without Docker. A separately labeled integration workflow
runs `pytest -m docker` against the pinned FreeMHD image. Scheduled and release
CI run production 3-D refinement, FreeMHD parity, and the full high-Hartmann
campaign; their generated meshes, native outputs, logs, and plots are ephemeral
artifacts or release assets, not committed files.

## Decision register

| ID | Decision | Reason |
|---|---|---|
| D-001 | Center LMX on fully developed and 3-D fringing-field inductionless duct MHD | These form the product core and share a coherent physics and API model. |
| D-002 | Decide non-core research lanes with a capability-and-evidence audit | Size alone cannot distinguish valuable developing physics from disposable scaffolding. |
| D-003 | SOLVAX owns reusable numerical algebra; LMX owns MHD assembly and validation | This prevents duplicate solver maintenance and follows the MHX/SOLVAX boundary. |
| D-004 | Use a single case, solve, result, and convergence model | Python, CLI, examples, restart, and docs need identical semantics. |
| D-005 | Use PyPA `src` layout, Ruff, inline typing, pytest, and Sphinx/MyST | These give a small standard toolchain with clear installed-package behavior. |
| D-006 | Use a compact Diátaxis documentation tree modeled on MHX/VMEX | It separates learning, tasks, theory, and reference without project diaries. |
| D-007 | Keep project history out of user-facing surfaces | LMX should read as a standalone product; this file, Git, and releases retain the record. |
| D-008 | Archive and squash history only after the live product is complete | A default clone cannot reach <10 MB while the existing 38-MiB pack remains reachable. |
| D-009 | Protect 3-D fringing and straight-pipe capabilities throughout the refactor | They are active development goals and must improve rather than disappear during trimming. |
| D-010 | Keep FreeMHD as a pinned Docker validation oracle outside the runtime wheel | Executed independent parity is valuable; OpenFOAM runtime and generated data are not Python package responsibilities. |
| D-011 | Do not restore the archived Q2D or blanket frameworks | The broad solver/configuration/plotting/campaign families duplicate common infrastructure. Q2D physics returns only as one compact shared-API model; the unvalidated blanket model remains outside the live product. |
| D-012 | Retain magnetic-obstacle cases only as applications of the common 3-D fringing solver | A localized divergence-free field broadens the retained field model without a separate algorithm; independent validation remains a promotion gate. |
| D-013 | Remove branded WHAM proxy builders while retaining generic tabulated fields and straight-pipe fringing | The general capability can express a future WHAM case once matched evidence exists, without carrying product-specific unvalidated code now. |
| D-014 | Differentiate only retained production equations, and extend the canonical Hartmann contract to every retained Q2D/2-D/3-D solver family | Shadow 3-D, nonrectangular surrogate, WHAM, and blanket objectives can report gradients of a different model; deleting them does not justify leaving the production paths primal-only. |
| D-015 | Use released SOLVAX native sparse solves instead of calling SciPy solvers from LMX | LMX owns MHD matrix assembly; reusable host factorization and solve behavior belongs to SOLVAX. |
| D-016 | Use one explicit current restart-schema family for state, compact flux, Aitken, and Anderson checkpoints | A single fail-closed contract is easier to reason about, test, and document than internal format-version branches. |
| D-017 | Keep the executed matched B2 Docker path and remove the private straight-pipe archive smoke | The private lane explicitly could not establish B1 equation/observable parity and duplicated the accepted B2 execution boundary; B1 remains protected by internal/manufactured gates until a genuinely matched external case exists. |
| D-018 | Use Ruff with a 110-column limit for all maintained Python | The numerical expressions remain readable, every file has one formatter, and the format reduces line count without hand-compressed layouts. |
| D-019 | Expose one SOLVAX PCG velocity path instead of naming identical `auto`, `cg`, and `solvax_pcg` choices | LMX assembles and certifies the physical system; a user-facing switch between aliases of the same released algorithm adds no flexibility. |
| D-020 | Consolidate by stable concept: units and wall models into physics, and state/result schemas into specs | These types are small parts of the physical and public data contracts; separate modules added navigation and import boundaries without independent ownership. |
| D-021 | Consolidate spatial construction, run configuration, output, and validation by user-facing ownership | Meshes own spatial operators and imposed fields; configuration owns run logging; IO owns lazy plotting; validation owns references and benchmark contracts. These boundaries minimize navigation while remaining acyclic and independently testable. |
| D-022 | Separate reusable fully developed assembly from case-level solve workflows | `solvers.py` owns the physical systems delegated to SOLVAX; `cases.py` owns factories, orchestration, and differentiable case objectives. This keeps both files below 1,500 lines and removes a one-workflow autodiff module. |
| D-023 | Keep one public fringing module over three private numerical owners, and consolidate run configuration with result schemas | Users retain one `lmx.fringing` surface; it owns solve orchestration while shared, duct, and pipe kernels retain explicit private ownership. Configuration and result schemas share one typed input/output contract in `specs.py`. |
| D-024 | Keep FreeMHD execution, observation, and comparison in one repository-only validation module | Docker/reference tooling remains fully tested and executable from a checkout but is not installed with the runtime wheel. Shipped analytical and benchmark-data loaders remain in `lmx.validation`. |
| D-025 | Remove the unbundled ClosedChannel/processed-slice adapter and example | A default clone and installed wheel cannot run this private-directory workflow; the packaged analytical and ALEX data plus the pinned, independently observed B2 Docker comparison provide standalone validation without silent optional-data skips. |
| D-026 | Remove LMX's 3-D sparse-direct and duplicate Jacobi implementations | Reusable iteration belongs to SOLVAX. Matrix-free SOLVAX PCG serves the duct and pipe elliptic systems; the established collocated projection retains its LMX stencil and physical residual while SOLVAX owns fixed-point state and stopping. |
| D-027 | Implement Q2D flow as one dealiased periodic SM82 vorticity path | This preserves an active strong-field capability with analytical identities, JAX acceleration, and common result semantics while released SOLVAX owns the reusable periodic Poisson inversion. |
| D-028 | Treat FreeMHD's public `S3_Buhler_Ha616` archive as a distinct candidate validation target, not ALEX B1 | The executed archive uses a 48.59 mm-radius pipe, a mirrored 0.21 T MEKKA-style fringe, and a thick copper wall; ALEX B1 uses a 54.1 mm radius, 2.1 T field, and different nondimensional groups and wall conductance. Similar geometry does not satisfy the matched-contract gate. |
| D-029 | Make differentiability a release invariant and choose the derivative algorithm with the primal algorithm | JAX traceability alone does not control gradient meaning, accuracy, runtime, or memory. Linear/steady solves use implicit transpose solves; long finite trajectories use exact checkpointed discrete reverse mode; host diagnostics and discrete controls stay outside the traced core. |
| D-030 | Differentiate the mathematical result, not incidental solver work | Implicit adjoints are the default for converged steady equations; checkpointed discrete adjoints are the default for finite trajectories. This matches established Optimistix, PETSc TSAdjoint/Revolve, and JAX scientific-computing practice while preserving one LMX production equation path. |
| D-031 | Fail closed when a production adjoint does not converge | A finite primal is insufficient evidence: unsupported geometry remains unavailable until its transpose solve passes an independent derivative gate. |
| D-032 | Eliminate inactive degrees of freedom from SPD off-diagonals | A fluid boundary contribution belongs on the diagonal, not as a coupling to an identity-constrained solid cell. Keeping the volume-scaled momentum operator symmetric makes primal PCG mathematically valid and gives its implicit transpose solve the same conditioning. |
| D-033 | Differentiate each production 3-D recurrence with bounded nested derivatives | Generic rectangular/layered ducts and straight pipes expose continuous controls through their retained field updates. SOLVAX uses an implicit VJP for converged electric closure and exact checkpointing for finite projection and outer iterations. Specialized ALEX B1/B2 lanes remain unavailable until their distinct coupled operators pass the same gates. |
| D-034 | Execute each PR test once and aggregate its evidence | Three duration-balanced shards run concurrently with branch coverage and save run-scoped JUnit/coverage caches. A report-only job enforces the repository threshold; release reuses the same workflow in parallel with docs and external validation. This preserves fail-closed evidence while targeting a sub-10-minute critical path. |
| D-035 | Lead with differentiable inductionless blanket-design physics, not general CFD breadth | ParaStell owns parametric stellarator CAD/neutronics, NekRS owns exascale high-order CFD, and FreeMHD owns free-surface/multi-region/full-induction development. LMX earns a distinct role through compact high-Hartmann duct/fringe equations, verified bounded-memory derivatives, accelerator design throughput, and matched external evidence. New multiphysics enters through versioned coupling data before it enters the runtime. |
| D-036 | Keep `lmx.fringing` as the only public 3-D surface and retain private fringing files only by mathematical ownership | The public module owns orchestration. Common structured-grid operations, rectangular-duct kernels, and cylindrical-pipe kernels retain separate private owners because their metrics and change reasons differ; test-only proxy solvers and duplicate recurrences are removed. |
| D-037 | Remove the label-only bent-pipe lane | Curved display coordinates never entered its straight cylindrical operators, and its validator mislabeled generic transverse velocity as Dean curvature physics. A future curved-pipe solver must introduce coherent curvilinear metrics, independent validation, and derivative gates rather than reuse the removed name. |
| D-038 | Do not nest a full transient momentum Krylov solve inside every B2 pressure-Schur action | The construction passes dense compatibility and autodiff identities, but repeats an expensive mass-dominated inverse, leaves the reduced physical trajectory effectively unchanged, and exceeds the production runtime gate. A viable block method must use a separately reusable/coarse response or factorized preconditioner with bounded primal and transpose work. |
| D-039 | Do not impose derivative-only roundoff tolerances on a primal-only specialized path | Generic traced 3-D fields retain roundoff electric closure for implicit VJP consistency. B2 remains unavailable to differentiation and instead uses a directly tested `1e-10` linear tolerance, while its independent charge, restart, and external-validation gates remain unchanged. |
| D-040 | Fold single-consumer fringing orchestration into the public module, not into another private file | `fringing.py` owns its three solve-dispatch functions; case construction belongs to `cases.py`, validation belongs to `validation.py`, and only shared, duct, and pipe numerical kernels retain private modules. The public import surface is unchanged. |
| D-041 | Reuse a compiled production evidence map across same-shape finite-difference samples | Centered differences remain mathematically independent of autodiff, but they do not require compiling an identical primal graph a second time. Production value-and-gradient executables may supply shifted primal values; physics, VJP/JVP, sensitivity, batching, memory, and tolerance assertions remain separate and unchanged. |
| D-042 | Use the frozen momentum diagonal as the B2 pressure mobility | The pressure correction must use the same local response as the implicit momentum predictor. Reusing its already assembled diagonal is allocation-light and restores the variable-coefficient discrete flux identity; a full nested momentum inverse is still rejected by D-038. |
| D-043 | Advance B2 with two relaxed SIMPLE-style pressure--momentum correctors per electric closure | Including the current pressure force and accumulating a bounded correction removes the diagonal-only plateau. Two correctors materially improve physical defect per second; a third has diminishing returns, and fixed relaxation avoids adding restart fields. |
| D-044 | Retain depth-two Anderson for B2 until a safeguard improves both early and late evidence | The corrected raw and fixed-two maps are stable and smoother late, but neither improves physical defect over Anderson early or late. A one-step growth fallback amplifies alternating spikes. Replace acceleration only with a restart-exact method that wins defect, update, runtime, and memory together. |
| D-045 | Stop the primal-only B2 pressure correction at `1e-10` while retaining volume-scaled balance checks | On the production coarse state this cuts pressure PCG work by about one quarter while leaving the state update and momentum-defect trajectory unchanged at reported precision. The resulting `2e-5`--`7e-5` divergence remains below five percent of the independent `1e-3` balance gate. Generic traced paths retain roundoff-level primal solves for implicit-derivative consistency. |
| D-046 | Add the local electromagnetic reaction to the B2 predictor as a fixed-point-neutral pseudo-mass | The term $R_B=\sigma|\boldsymbol B|^2$ is added to the implicit momentum diagonal and the identical $R_B\boldsymbol u^n$ term to the right-hand side, so it cancels at a fixed point while improving the predictor and pressure mobility. The retained operation order is deliberate: reassociating the terms changes the finite-precision nonlinear trajectory materially. Dense operator, autodiff, restart, and production-continuation gates protect the contract. |
| D-047 | Shard the generic differentiable 3-D production fields and replicate only global coarse solves | Axial `NamedSharding` constraints must remain inside the traced program rather than staging design-dependent fields through NumPy. Rectangular, layered, and straight-pipe primal fields compose with reverse mode over the same recurrence; global axial/transverse coarse solves are explicitly replicated and their corrections repartitioned. Specialized ALEX B1 stays single-device until its cylindrical production operators pass an independent sharding gate. |
| D-048 | Accelerate only the B2 mechanical state and close electricity once on the accepted velocity | Anderson/Aitken owns velocity and compact conservative flux; the one post-acceptance electric solve makes current, Lorentz force, charge, defects, checkpoints, and returned fields describe one state without retaining potential history or performing a second electric solve. |
| D-049 | Qualify evidence at the boundary it informs | Exact tests serve edits, a conservative change gate serves local candidates, and the complete covered suite runs once for a source candidate. Documentation, packaging, external validation, accelerator, and release gates run only for their owning surfaces; required workflows remain visible and skip jobs with successful job-level conditions. |
| D-050 | Spectrally filter ill-conditioned depth-two B2 Anderson histories | SOLVAX's condition filter bounds the retained residual Gram condition at 25. Well-conditioned Anderson steps are unchanged; near-dependent histories become bounded residual-space combinations without another field, map evaluation, or user parameter. Production evidence must still beat unfiltered Anderson and the raw map in defect, update, runtime, balance, and restart equivalence. |

## Work log

Append concise completed facts only. Each entry records the tranche, changed
surface, measurements, validation, decision, and next action.

### 2026-08-21 — simplification plan created

- Replaced the development diary with this product-focused plan and logbook.
- Measured the authenticated repository: normal clone 43,372 KiB, `.git`
  38,672 KiB, tracked checkout 4,428,404 bytes across 182 files; depth-one
  clone 6,832 KiB.
- Measured package/test/tool scale: 30,852 package lines, 19,669 test lines,
  8,297 script lines, 32 package modules, 28 test files, and 13 scripts.
- Identified the largest physics, validation, campaign, and external-adapter
  surfaces for function-level ownership and evidence review.
- Reviewed SOLVAX `bef55f3`, MHX `b0fa2e6`, and VMEX `0362f70` live repository
  structures for ownership, documentation, README, API, and tooling patterns.
  Reference clones are outside the LMX repository.
- Chose fully developed and 3-D fringing-field duct MHD as the product core and
  established the LMX/SOLVAX ownership rule.
- Protected the 3-D fringing, straight-pipe, and minimal FreeMHD parity paths;
  changed broad module deletion into a capability-and-evidence audit.
- Added a three-layer FreeMHD strategy: PR contract fixtures, a local pinned
  Docker smoke run, and scheduled/release production parity.
- Next action: Phase 0—create the working branch, restore required CI, build the
  capability matrix, freeze 2-D/3-D baselines, and reproduce FreeMHD in Docker.

### 2026-08-21 — Phase 0 safety tranche started

- Created `codex/lmx-simplification` from `main` at `4cd9239` and kept all
  implementation work off the default branch.
- Added the capability matrix and protected the 2-D, 3-D fringing,
  straight-pipe, and FreeMHD parity paths from deletion-by-size.
- Verified existing provenance and architecture gates. The clean baseline had
  32 package modules, 30,852 package lines, 30 root exports, and 12 curated
  examples; median import and tracked-tree limits passed.
- Ran the complete baseline gate: 818 passed, 5 optional FreeMHD-data tests
  skipped, 95.26% coverage, 145.0 s. The first safety implementation gate ran
  821 passed and 5 skipped at 95.26% in 187.6 s; the 90 s target remains open.
- Built documentation with Sphinx warnings as errors; built and checked the
  wheel and sdist. The wheel is 290,640 bytes with 45 members and the sdist is
  274,985 bytes with 51 members.
- Confirmed hosted CI is externally unavailable: GitHub refused every job
  before its first step with an account payment/spending-limit annotation.
  Actions itself is enabled. No workflow failure was attributed to LMX code.
- Built `freemhd-install:latest` from FreeMHD commit `14b54a3` and OpenFOAM
  v2206. Image ID `sha256:44e9c9c1cc9d69aa4e50ca773aef7971010905a529462273bf913e6bc5b2c238`
  is linux/amd64 and ran through Docker emulation on the audit host.
- Reproduced the B2 square-duct matched smoke from a clean detached LMX
  worktree using `scripts/run_freemhd_parity_suite.py --matched-b2-smoke`.
  Both codes executed 2 steps at `dt=1.8518518518518519e-6` with the pinned
  iteration/tolerance controls. LMX took 5.14 s and FreeMHD 6.94 s.
  Execution, artifacts, contract, observation, and comparison passed with
  pressure Linf `0.0109172453` and RMS `0.0045179771`. Production acceptance
  remains false by contract because this record is a harness smoke.
- Stored the untracked B2 record under
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-phase0-b2-clean.2ke9nY/b2`.
- Added steady-result `converged`, `status`, and `steps` metadata for the 2-D
  and 3-D result models, surfaced it in CLI JSON, returned exit code 2 for a
  recorded unconverged steady solve, and replaced the 2-D potential
  `nan_to_num` repair with typed `NumericalFailure` behavior.
- Removed every `nan_to_num` repair from LMX numerical source and added shared
  typed finite-state checks at the 2-D, 3-D Poisson, and public extruded-result
  boundaries. The physical fields and conserved observables fail closed;
  solver diagnostic matrices may still use explicit not-available sentinels.
- Reran the complete 829-test gate after the 3-D change: 824 passed and the
  same 5 optional FreeMHD-data tests skipped. Fatal Ruff checks, provenance,
  architecture, Sphinx warnings-as-errors, package build, and Twine checks
  also passed. Source is 30,938 lines; wheel/sdist sizes are 290,945 and
  275,382 bytes.
- Recorded the canonical Hartmann `Ha=20`, `48 x 48` CPU baseline under
  `/Users/rogeriojorge/local/tests/lmx-audit/phase0-baselines/`: 12.80 s cold,
  8.69 s warm, 9.57 s mean over five repeats, and 1,574,371,328-byte maximum
  host RSS. The remaining performance-matrix rows stay open.
- Repeated the pinned B2 LMX/FreeMHD smoke on committed source `82538d2`.
  Execution, artifacts, contract, observation, and comparison passed with no
  failed checks and unchanged pressure Linf `0.0109172453` and RMS
  `0.0045179771`. The harness role remains intentionally non-accepting. Its
  complete record is outside the checkout at
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-phase0-b2-82538d2/b2`.
- Created and independently cloned/fsck-verified the complete 37 MiB bundle at
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-pre-simplification-20260821.bundle`.
  SHA-256: `9a56decc7de1e8946ec1a1de3e67e2ca06fabf7a9c9e7071b4491080e736a389`.
- Next action: complete the runtime/memory matrix, decide the audit
  capabilities, and begin the reusable-algebra ownership map.

### 2026-08-21 — SOLVAX ownership tranche started

- Completed the Phase 0 research-lane decisions before deletion. Mapped
  bent-pipe and magnetic-obstacle cases onto the retained common 3-D fringing
  solver; scheduled the isolated Q2D, blanket, branded WHAM proxy, and shadow
  autodiff families for removal only after the verified archive.
- Classified the current linear, Poisson, Krylov, fixed-point, modal,
  preconditioner, sparse-direct, and small constraint solves by LMX/SOLVAX
  ownership. LMX retains MHD assembly, gauges, physical residuals, and
  acceptance; SOLVAX owns reusable algebra.
- Replaced both direct SciPy `spsolve` calls in the rectangular and pipe 3-D
  Poisson paths with released `solvax.splu_solve`. MHD-specific sparse matrix
  assembly remains in LMX. Added a regression test that observes the SOLVAX
  call rather than merely comparing the resulting field.
- Ran the complete 830-test gate: 825 passed and the same 5 optional
  FreeMHD-data tests skipped. Targeted rectangular/pipe sparse and full 3-D
  solves, fatal Ruff checks, provenance, architecture, Sphinx
  warnings-as-errors, package build, and Twine checks passed. Package source
  is 30,941 lines across 32 modules.
- Repeated the pinned B2 Docker smoke on committed source `48e30dd` after the
  sparse-direct handoff. Execution, artifacts, contract, observation, and
  comparison passed with no failed checks and unchanged pressure Linf
  `0.0109172453` and RMS `0.0045179771`. The complete untracked record is at
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-phase1-b2-48e30dd/b2`.
- Deleted `solve_five_point_solvax_pcg_state` and its redundant SOLVAX unit
  tests. The fully developed velocity path now composes released
  `solvax.pcg_linear_solve` directly with LMX coefficient action, physical
  max-norm certification, and one private JIT boundary. Source/test changes
  are 82 net lines smaller; package source is 30,926 lines.
- Rejected an uncompiled version after measurement: Hartmann `Ha=20`,
  `48 x 48` warm time regressed from 8.69 s to 36.42 s and maximum RSS from
  1.57 GB to 5.06 GB. The accepted JIT composition measured 12.62 s cold,
  8.51 s warm, 10.86 s mean, 1.59 GB maximum RSS, and 1.06 GB peak footprint.
  Its record is `phase0-baselines/hartmann-ha20-48x48-direct-solvax-pcg-jit.json`
  in the external audit directory.
- Added and locally committed generic `fixed_point_iteration` on SOLVAX branch
  `codex/stationary-fixed-point` at `573fd4e`. It supports custom physical
  residuals, relaxed tolerance stopping, and static fixed-step reverse-mode
  differentiation. SOLVAX passed 700 tests with 6 optional-backend skips,
  Ruff, Sphinx warnings-as-errors, package build, and Twine checks. LMX does
  not consume the unreleased API.
- Reran the accepted LMX gate: 822 passed and the same 5 optional FreeMHD-data
  tests skipped. Fatal Ruff checks, provenance, architecture, Sphinx
  warnings-as-errors, package build, and Twine checks passed.
- Repeated the pinned B2 Docker smoke on committed source `fc1e187` after the
  direct PCG composition. All executable gates passed with no failed checks
  and unchanged pressure Linf `0.0109172453` and RMS `0.0045179771`. The
  complete untracked record is at
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-phase1-b2-fc1e187/b2`.
- Opened draft LMX PR #1 and SOLVAX PR #78. Hosted LMX docs passed, while the
  Python 3.10 setup exposed a stale `solvax==0.8.5` pin that was never
  published. Raised the declared LMX minimum and compatibility lane to the
  released 0.13.0 required by `splu_solve`; local dependency, provenance,
  architecture, Ruff, and Sphinx checks pass on the corrected configuration.
- The corrected hosted LMX lanes completed setup, metadata, and architecture
  gates, then reached 69% without a test failure before runner termination. A
  1,200-second rehearsal proved that enlarging the script and job budgets did
  not prevent GitHub from sending a shutdown signal at 11 minutes. Replaced
  the accumulating monolithic process with exact benchmark (58), core (241),
  examples (22), physics (232), and validation (274) shards. The
  minimum-dependency lane uses isolated runners; the coverage lane uses fresh
  bounded pytest subprocesses and combines their data in place. Benchmarks and
  examples run on one worker so their existing timeouts measure the operation,
  rather than contention with a simultaneous JAX compile. The partition
  contains all 827 tests exactly once per dependency lane before enforcing the
  unchanged 95% threshold.
- Rehearsed the pre-isolation partition locally in the repository environment:
  core plus benchmarks passed 296 with 3 optional-data skips in 53 seconds,
  examples passed all 22
  on one worker in 55 seconds, physics passed 230 with 2 optional-data skips in
  63 seconds, and validation passed all 274 in 6 seconds. Coverage-enabled
  physics and validation took 74 and 11 seconds; the pre-isolation
  core/examples coverage database took 66 seconds. The raw databases combined
  successfully and passed the 95% line/branch gate. This bounded transition
  does not relax the planned below-90-second default-suite target.
- Kept the CI success path independent of Actions artifact storage after the
  hosted account reported its artifact quota full. Coverage is combined and
  enforced on the runner; normal logs retain the test and coverage evidence.
- The first bounded hosted run passed Python 3.10 validation in 51 seconds and
  exposed two example-specific minimum-endpoint assumptions. JAX 0.6.2 gives
  direct-versus-restarted mean-velocity and charge-balance differences of
  `6.7e-16` and `1.1e-12`, while its iterative `p` and `phi` fields differ by
  `0.0090` and `0.0034` on direct scales of `104` and `8.6`. Replaced the
  nonportable bitwise claim with explicit per-field tolerances while retaining
  the near-machine conserved-observable requirements.
- Reduced only the Li/AlN demonstration discretization from 2,048 to 512
  cross-section cells across its two wall models and bounded it to six updates
  with 40 potential iterations. On the clean minimum numerical endpoint its
  runtime fell from 21.7 to 11.4 seconds. The full 22-test examples shard now
  passes in 45.7 seconds there and 47.5 seconds on the current endpoint.
- The same hosted run passed the minimum-endpoint physics and validation shards
  in 7m01s and 51 seconds. Its only core failure was a B1 production-path
  compile exceeding the 120-second per-test limit while sharing a two-core
  runner; every other core item completed. Isolated all 58 benchmark tests on
  one worker, where the clean minimum endpoint passes them in 61.8 seconds.
- SOLVAX hosted lint, types, docs, and build passed. Current JAX reports its
  unsupported bfloat16 LU dtype as `TypeError`; the pre-existing portability
  test now accepts that documented backend error alongside the older runtime
  error classes. The targeted local test passes and the corrected matrix is
  pending on commit `8cfe56c`.
- Next action: release the SOLVAX stationary primitive before replacing LMX's
  remaining Jacobi loops; independently complete the canonical performance
  matrix.

### 2026-08-21 — canonical baselines and fixed-point handoff

- Completed the canonical CPU performance matrix at commit `c41bdd5` with five
  warm repeats and fresh-process RSS measurements. The external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/phase0-baselines/lmx-canonical-matrix-20260821.json`.
  It covers Hartmann `Ha=20`, Shercliff and Hunt `Ha=5/20/100`, fixed flow,
  reduced rectangular and pipe 3-D cases, and a retained Hartmann gradient.
  Every declared converged case converged; the deliberately bounded pipe case
  reported `step_limit` while satisfying its numerical conservation checks.
- Merged SOLVAX PR #78 with the generic JAX-native
  `fixed_point_iteration`, user guide, API reference, decision table, examples,
  and focused tests. Its minimum/current/AD Linux and current macOS matrices,
  lint, types, docs, distribution build, benchmark verification, and
  reproduction smoke passed with 98.16% coverage.
- Merged SOLVAX release PR #79, tagged `v0.14.0`, and verified the trusted PyPI
  publication by downloading the released 139 KiB wheel. LMX now requires
  `solvax>=0.14,<1`, and the compatibility workflow pins the published 0.14.0
  floor.
- Replaced the local two-dimensional Jacobi loop and all three rectangular,
  variable-coefficient, and cylindrical 3-D Jacobi loop/state implementations
  with `solvax.fixed_point_iteration`. LMX retains only its boundary-aware
  stencil maps, gauges, physical residuals, and fail-closed result validation.
  The tranche removes 29 net source/test lines and avoids an unnecessary first
  sweep when the initial state already satisfies the physical tolerance.
- On a deterministic `20^3`, 50-sweep CPU case, the replacement matches the
  prior field and residual to roundoff and reduces the seven-repeat warm median
  from 99.3 ms to 67.3 ms, a 32% improvement. Focused solver tests and the full
  230-test physics shard pass on both JAX 0.6.2 and JAX 0.10.2 using the
  released SOLVAX wheel.
- The released-wheel five-shard gate ran all 827 collected tests: 822 passed,
  the same 5 optional external-data tests skipped, and combined line/branch
  coverage is 95.25%. Provenance, architecture, Sphinx warnings-as-errors,
  wheel/sdist build, and Twine checks pass; the wheel is 284 KiB.
- Repeated the pinned B2 Docker smoke from committed source `1305790` with
  FreeMHD `14b54a3` and image
  `sha256:44e9c9c1cc9d69aa4e50ca773aef7971010905a529462273bf913e6bc5b2c238`.
  Execution, artifacts, contract, observation, and comparison passed with no
  failed checks and unchanged pressure Linf `0.0109172453` and RMS
  `0.0045179771`. Its harness role remains deliberately non-accepting. The
  complete external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-phase1-fixedpoint-1305790/b2`.
- Next action: continue the Phase 1 wrapper/structured-algebra audit.

### 2026-08-21 — linear-module ownership collapse

- Deleted `lmx/linear.py`. Its four boundary-aware five-point/Poisson actions
  and physical residual norms now live in `operators.py`; its two thin Jacobi
  and PCG adapters now compose the released SOLVAX APIs directly inside
  `solvers.py`. No general iteration or matrix solve remains in the moved LMX
  code.
- Folded the four direct operator/solve/implicit-gradient tests into the
  physical solver suite and deleted `tests/test_linear.py`. Removed the
  four-value injected-backend compatibility branch so the internal potential
  solve has one result shape. Samper provenance fingerprints now cover the
  operator and solver sources that implement the current path.
- The structural tranche removes one source module, one test file, 20 package
  lines, 38 test lines, and 61 tracked lines overall. The package contains 31
  Python modules and 30,891 lines; tests contain 27 files and 19,728 lines.
- All 827 tests ran against the released SOLVAX 0.14.0 wheel: 822 passed, the
  same 5 optional external-data tests skipped, and exact combined line/branch
  coverage is 95.248%. The expanded 234-test physics shard passes on both JAX
  0.6.2 and 0.10.2. Provenance, architecture, repository-integrity, Samper,
  and Sphinx warnings-as-errors gates pass.
- The rebuilt 283 KiB wheel and 268 KiB source distribution pass Twine and no
  longer contain `lmx/linear.py`. The pinned B2 Docker smoke on committed
  source `81e9122` again passed execution, artifacts, contract, observation,
  and comparison with no failed checks and unchanged pressure Linf
  `0.0109172453` and RMS `0.0045179771`. Its external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-phase1-no-linear-81e9122/b2`.
- Next action: classify the remaining line, modal, deflation, projection, and
  fixed-flow algebra by reusable SOLVAX contract before the next deletion.

### 2026-08-22 — capability trim and user-facing rebuild

- Removed the separate Q2D, blanket, centerline-field, Dean report, scaling,
  broad external-reporting, branded mirror-pipe, and shadow-autodiff surfaces
  after the verified archive. Retained bent-pipe geometry, the low-De baseline,
  magnetic-obstacle construction, generic fields, and the straight-pipe solver
  as applications of the common 3-D path.
- Deleted generated benchmark-result snapshots, unreferenced documentation
  media, case testbeds, report/release/campaign scripts, and their proxy tests.
  The live tree now contains 94 files, 24 package modules, 14 test files, four
  maintenance scripts, eight curated examples, and 17 documentation pages.
- Removed the private straight-pipe archive smoke because it did not match the
  B1 production contract. The pinned, independently observed B2 Docker path is
  the sole executed FreeMHD comparison; B1 retains its internal/manufactured
  numerical gates and an explicit external-parity promotion requirement.
- Replaced internal restart-format compatibility branches with four current
  schemas: state, compact flux, Aitken, and Anderson. Writes reject partial
  accelerator/diagnostic state and reads reject partial or unknown schemas.
- Rebuilt README and documentation around the supported product, equations,
  implementation map, first runs, tutorials, task guides, API/CLI reference,
  validation, FreeMHD reproducibility, development architecture, and primary
  literature. Sphinx passes with warnings treated as errors.
- Standardized all maintained Python with Ruff at 110 columns. Ruff formatting
  and fatal/import lint pass. Package source is 18,157 lines, tests are 12,635
  lines, scripts are 3,232 lines, documentation media are 197,588 bytes, and
  the live checkout is 1,739,118 bytes. The 15,000-line, 16-module, and
  1,800-line-largest-module completion targets remain open; `fringing.py` is
  currently 7,212 lines.
- Ran all five coverage shards: 551 passed and five optional external-data
  tests skipped. Exact combined line/branch coverage is 95.148307%. The
  architecture/import, Sphinx, package build, Twine, and distribution-content
  gates pass on the live sources. The wheel is 175,592 bytes with 37 members;
  the source distribution is 163,607 bytes with 43 members.
- Repeated the pinned B2 Docker smoke on committed source `5ed4c4b` with
  FreeMHD `14b54a3` and image
  `sha256:44e9c9c1cc9d69aa4e50ca773aef7971010905a529462273bf913e6bc5b2c238`.
  Execution, artifacts, contract, observation, and comparison passed with no
  failed checks; pressure Linf is `0.0109172453` and RMS is `0.0045179771`.
  The external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-capability-trim-5ed4c4b/b2`.
- Removed the `linear_solver` configuration parameter and its three aliases;
  the fully developed velocity system now has one direct SOLVAX PCG path.
  Collapsed the identical `block_jacobi` name into `jacobi`, retaining the
  deliberate `none`/`jacobi` preconditioner choice. All 550 tests passed with
  five optional external-data skips and exact line/branch coverage of
  95.144560%. Package source is 18,142 lines and tests are 12,589 lines.
- Added Ruff to the declared development extra after the first hosted quality
  job correctly reported that the executable was absent; local lint, format,
  architecture, and test gates pass with the declared configuration.
- Repeated the pinned B2 Docker smoke on committed source `9ce2069` after the
  solver-control collapse. Execution, artifacts, contract, observation, and
  comparison passed with no failed checks and unchanged pressure errors. The
  external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-solver-api-9ce2069/b2`.
- Consolidated units and wall-layer models in `physics.py`, and the state,
  diagnostics, solution, and numerical-failure schemas in `specs.py`. This
  removes four narrow module boundaries while preserving the lazy root API.
  The live package now contains 21 modules and 18,098 lines; tests contain
  12,588 lines. All 550 tests passed with five optional external-data skips;
  exact combined line/branch coverage is 95.117621%. Sphinx, Ruff, architecture,
  build, Twine, and distribution-content gates pass. The wheel is 173,997 bytes
  with 34 members and the source distribution is 162,797 bytes with 40 members.
  Architecture regression ceilings are ratcheted to 21 modules, 18,200 package
  lines, and 12,600 test lines.
- Repeated the pinned B2 Docker smoke on committed source `7b12a43`.
  Execution, artifacts, contract, observation, and comparison passed with no
  failed checks; pressure Linf is `0.0109172453` and RMS is `0.0045179771`.
  The report is non-accepting only because the branch has the candidate role.
  Its external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-model-fusion-7b12a43/b2`.
- Consolidated the 3-D result schemas into `specs.py`; spatial operators and
  imposed fields into `mesh.py`; runtime reporting into `config.py`; lazy
  plotting into `io.py`; and analytic references and benchmark contracts into
  `validation.py`. Seven package files were removed without changing the
  30-name root API. The live package now contains 14 modules and 18,037 lines,
  below the 16-module completion ceiling; tests contain 12,586 lines.
- All 550 tests passed with five optional external-data skips and exact combined
  line/branch coverage of 95.102633%. Sphinx, Ruff, architecture, build, Twine,
  and distribution-content gates pass. The wheel is 169,895 bytes with 27
  members and the source distribution is 161,499 bytes with 33 members.
  Architecture regression ceilings are ratcheted to 14 modules and 18,100
  package lines.
- Repeated the pinned B2 Docker smoke on committed source `0a7ea11`.
  Execution, artifacts, contract, observation, and comparison passed with no
  failed checks and unchanged pressure errors. The report is non-accepting only
  because the branch has the candidate role. Its external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-ownership-consolidation-0a7ea11/b2`.
- Split fully developed ownership between the reusable physical-system assembly
  in `solvers.py` and case factories, orchestration, and differentiable
  objectives in `cases.py`. The dedicated autodiff file was removed; both
  retained modules are below 1,500 lines. The package now contains 13 modules
  and 18,071 lines; the wheel contains 26 members.
- All 550 tests passed with five optional external-data skips and exact combined
  line/branch coverage of 95.102633%. Sphinx, Ruff, architecture, build, Twine,
  and distribution-content gates pass. The module regression ceiling is
  ratcheted to 13, leaving room for an owned decomposition of the 3-D solver
  while retaining the final 16-module ceiling.
- Repeated the pinned B2 Docker smoke on committed source `99eac41`.
  Execution, artifacts, contract, observation, and comparison passed with no
  failed checks and unchanged pressure errors. The report is non-accepting only
  because the branch has the candidate role. Its external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-solver-ownership-99eac41/b2`.
- Decomposed the 7,212-line fringing implementation behind the unchanged
  `lmx.fringing` user surface into generic, rectangular-duct, mapped-pipe, and
  orchestration owners. Consolidated run configuration and logging schemas
  into `specs.py`, removing `config.py`. The package now meets the final
  16-module ceiling; its 18,242 lines and 2,466-line orchestration module remain
  above the final source-size targets.
- All 550 tests ran after the ownership split: 545 passed and the same five
  optional external-data tests skipped. Exact combined line/branch coverage is
  95.122222%. Ruff, architecture/import, Sphinx warnings-as-errors, package
  build, Twine, and distribution-content gates pass. The wheel is 173,468
  bytes with 29 members and the source distribution is 163,983 bytes with 35
  members.
- Repeated the pinned B2 Docker smoke on committed source `1ca3735`.
  Execution, artifacts, contract, observation, and comparison passed with no
  failed checks; pressure Linf is `0.0109172453` and RMS is `0.0045179771`.
  The report is non-accepting only because the branch has the candidate role.
  Its external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-fringing-ownership-1ca3735/b2`.
- Extracted shared 3-D runtime, restart, timing, sharding, and history helpers
  into one internal owner and moved the rectangular-duct momentum/coupling
  kernels beside the duct discretization. Every installed source module is now
  below the final 1,800-line ceiling; the orchestration module is 1,789 lines.
- Moved Docker/FreeMHD execution and evidence machinery to the repository-only
  `validation/` surface so it is available to maintainers without becoming
  installed library API. Kept the stable benchmark-data loaders in
  `lmx.validation`, including in clean wheels. The package now contains 15
  modules and 16,669 lines; maintained core source is 6,333 lines, tests are
  12,599 lines in 14 files, and the sole external validation module is 1,744
  lines.
- All 550 tests ran after the final ownership and packaging changes: 545 passed
  and the same five optional external-data tests skipped. Exact combined
  line/branch coverage is 95.098775%. Ruff check and format, architecture and
  import checks, Sphinx warnings-as-errors, clean package build, Twine, and
  distribution-content audits pass.
- The clean wheel is 155,803 bytes with 28 members and the source distribution
  is 146,959 bytes with 34 members. A fresh isolated environment installed the
  wheel and passed the CLI and every packaged benchmark-data loader without
  access to the source checkout.
- Repeated the pinned B2 Docker smoke on committed source `1df69b9`.
  Execution, artifacts, contract, observation, and comparison passed with no
  failed checks; pressure Linf is `0.0109172453` and RMS is `0.0045179771`.
  The report is non-accepting only because the branch has the candidate role.
  Its external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-owned-runtime-1df69b9/b2`.
- Next action: reduce package source below 15,000 lines and tests below 12,000
  lines while profiling the retained 3-D paths and preserving all numerical,
  packaging, documentation, and independent FreeMHD gates.

### 2026-08-22 — final source-budget and solver-ownership tranche

- Removed the unbundled ClosedChannel/processed-slice adapter, example, tests,
  and media. Packaged Hartmann and ALEX evidence plus the pinned B2 Docker
  workflow remain; B1 continues to require a future matched executable case
  before any external-parity claim.
- Removed the rectangular, variable-coefficient, and cylindrical 3-D Jacobi
  and SciPy sparse-direct solvers. Duct and pipe elliptic systems now compose
  SOLVAX matrix-free PCG. The validated collocated projection keeps its LMX
  stencil, gauge, and physical residual while SOLVAX owns fixed-point state
  and stopping.
- Removed the direct SciPy dependency, using JAX's regular-grid interpolator
  for tabulated fields. Removed unused solver controls, alternate current and
  limiter lanes, duplicate diagnostic histories, report-only field metrics,
  and packaged differentiability testbeds. The retained differentiation API
  follows the canonical Hartmann solve.
- Replaced verbose emulation logging with a compact LMX-native stream and
  reduced the logging schema to enabled/banner/footer/flush/stride controls.
  Benchmark-B specifications use a canonical reviewed fingerprint rather than
  a second hand-maintained schema validator while still checking identity,
  physics, matched roles, semantics, stopping, smoke execution, and reference
  bytes independently.
- The live package is 14,946 lines across 15 modules; tests are 11,494 lines
  across 13 files; the tracked checkout is 1,529,679 bytes. The root API has
  26 names and the catalog contains seven runnable examples. Every package and
  external-validation source file remains at or below 1,800 lines. The
  architecture ceiling is ratcheted to 15,000 package lines.
- All five release shards passed: 503 tests with no skips. Exact combined
  line/branch coverage is 95.007845%; the two excluded functions are explicitly
  multi-device JAX sharding wrappers covered only on accelerator hardware.
  Ruff check/format, byte compilation, architecture/import, current-state prose,
  and Sphinx warnings-as-errors gates pass.
- The clean wheel is 141,522 bytes with 28 members and the source distribution
  is 132,566 bytes with 34 members. Both pass Twine and the distribution audit.
  A fresh isolated environment installed the wheel and passed both the CLI and
  a converged Hartmann solve without importing the checkout.
- Committed the source checkpoint as `596ce26` and repeated the pinned B2
  Docker workflow with FreeMHD `14b54a3` and the pinned image. Execution,
  artifacts, contract, native-output observation, and comparison all pass with
  no failed checks; pressure Linf is `0.0109172453` and RMS is `0.0045179771`.
  Its untracked record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-final-source-596ce26/b2`.
- Repeated same-host, same-environment performance comparisons against the
  canonical baseline commit `c41bdd5`. Warm medians changed by +2.0% for
  Hartmann 48x48, +1.2% for reduced B2, and +1.5% for reduced B1; peak RSS
  changed by -2.1%, +0.5%, and -0.9%. All remain inside the 5% regression
  guard with identical step counts, residuals, flow, and charge closure. The
  external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/performance/lmx-slim-checkpoint-596ce26.json`.
- Next action: push the evidenced checkpoint, then finish the package/API
  completeness and portable-test-runtime items before requesting approval for
  the separately gated history rewrite.

### 2026-08-22 — typed source layout and portable-runtime tranche

- Moved the installable package to the PyPA `src/lmx` layout and updated build,
  coverage, CI, release, architecture, provenance, validation, documentation,
  and fixture paths together. The wheel contains `lmx/py.typed`, and a package
  test verifies complete annotations on the 26-name root API. Repository-only
  runners bootstrap both the checkout and `src` roots, with direct CLI tests.
- Consolidated repeated full 3-D solves into shared physical integration cases.
  The retained suite still executes rectangular and layered ducts, straight
  and conducting-wall pipes, bent-pipe/low-De comparisons, analytic and
  tabulated variable fields, magnetic obstacles, SOLVAX composition, restart,
  differentiation, benchmark B1/B2, and FreeMHD contracts. Smaller example and
  test meshes preserve their stated acceptance checks while leaving production
  defaults unchanged.
- The canonical six-worker work-stealing gate passes 500 tests with no skips in
  82.2 seconds, below the 90-second audit-host target. All five release shards
  pass; exact combined line/branch coverage is 95.017844% across 5,765
  statements and 1,240 branches. Three explicit bent-pipe input-contract checks
  supply real branch evidence; no coverage exclusion was added.
- The live package contains 15 modules and 14,932 lines; tests contain 11,469
  lines across 13 files; every package and external-validation source remains
  below 1,800 lines. The checkout is 1,534,604 bytes, root import median is
  0.0156 seconds without eager JAX import, and the current-state prose scan is
  clean.
- Ruff check/format, byte compilation, architecture/import, and Sphinx
  warnings-as-errors gates pass. The source-layout wheel is 141,629 bytes with
  29 members and the source distribution is 132,735 bytes with 35 members;
  Twine and distribution-content audits pass, including the typing marker. A
  clean environment explicitly installed released SOLVAX 0.14.0 and the LMX
  wheel, then passed the CLI and a converged Hartmann solve from `/tmp`.
- Committed the source checkpoint as `cd29005` and repeated the pinned B2
  Docker workflow with FreeMHD `14b54a3`. Execution, artifacts, contract,
  native-output observation, and comparison pass with no failed checks;
  pressure Linf is `0.0109172453` and RMS is `0.0045179771`. The report is
  non-accepting only because this branch has the candidate role. Its untracked
  record is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-typed-src-cd29005/b2`.
- Next action: push the evidenced checkpoint, complete the common
  case/options/result API and opt-in history work, then finish tutorials,
  scheduled/release validation, and the separately approved history gate.

### 2026-08-22 — common solve and result contract

- Added `lmx.solve(model)` as the 2-D/3-D first-run API. It follows the mode in
  `CaseSpec` and accepts `ExtrudedInductionlessProblem`; specialized restart,
  custom-mesh, progress, and timing hooks remain in their owning modules.
- Both result types expose `converged`, `status`, `steps`, `residual`, `fields`,
  and `diagnostics`. README, tutorials, and runnable examples use the common
  call. The lazy root surface shrank from 26 to 24 names by moving specialized
  solves and power balance to their documented modules.
- The portable gate passes 501 tests with no skips in 78.9 seconds. Exact
  combined line/branch coverage is 95.036268% across 5,787 statements and
  1,244 branches. Ruff, architecture/import, current-state prose, and Sphinx
  warnings-as-errors gates pass.
- Package source is 14,977 lines across 15 modules. The wheel is 142,005 bytes
  with 29 members and the source distribution is 133,054 bytes with 35
  members; Twine and content audits pass. An isolated wheel environment used
  the common API for a converged fully developed solve and a real 3-D solve.
- Next action: make full diagnostic histories opt-in while preserving terminal
  diagnostics, exact benchmark/restart evidence, and the common result
  contract.

### 2026-08-22 — bounded diagnostic retention

- Added one `OutputSpec.history_stride` policy for 2-D and 3-D solves. The
  default retains only the terminal sample, positive values retain the first,
  periodic, and terminal samples, and `1` requests every iteration. Positive
  restart segments preserve already retained samples; terminal-only resumes
  keep only the new terminal. Benchmark builders explicitly request complete
  histories for their evidence contracts.
- Kept checkpoint state independent from returned diagnostic retention.
  Compact 3-D restart bundles retain the actual completed-step count and
  validate all stored history widths without reconstructing discarded data.
  `write_stride`, `history_stride`, and `checkpoint_interval` now have separate
  documented purposes.
- Collapsed repeated 2-D diagnostic assembly while adding the shared policy.
  Package source is 14,934 lines across 15 modules, below the 15,000-line gate;
  tests are 11,590 lines across 13 files and the root API remains 24 names.
- The complete six-worker gate passes 503 tests with no skips in 83.7 seconds.
  Exact combined line/branch coverage is 95.089601% across 5,823 statements
  and 1,264 branches. Ruff check/format, byte compilation,
  architecture/import, current-state prose, and Sphinx warnings-as-errors
  gates pass.
- The wheel is 142,349 bytes with 29 members and the source distribution is
  133,368 bytes with 35 members; Twine and distribution-content audits pass.
  A clean environment installed released SOLVAX 0.14.0 and the LMX wheel,
  then passed the CLI, a converged 2-D common-API solve, a real 3-D common-API
  solve, and terminal-history checks from `/tmp`.
- Committed the bounded-history checkpoint as `64d0292` and repeated the
  pinned B2 Docker workflow with FreeMHD `14b54a3`. Execution, artifacts,
  contract, native-output observation, and comparison pass with no failed
  checks; pressure Linf is `0.0109172453` and RMS is `0.0045179771`. The
  record is non-accepting only because the harness declares its candidate
  role. Its untracked evidence is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-history-64d0292/b2`.
- Next action: push the common-API and bounded-history commits with this
  evidence, then finish tutorial and scheduled/release validation work.

### 2026-08-22 — tutorials and continuous external validation

- Completed four indexed tutorials by promoting the existing walls-and-fields
  material into the tutorial sequence. Every tutorial maps to a curated
  workflow, and CI executes all seven shipped examples. The stable Hartmann,
  Hunt, and TOML workflows use bounded default output; plots remain an explicit
  editable option.
- Corrected the TOML first run so its magnetic field and material properties
  reproduce $Ha=20$. It converges in 17 steps with residual
  `9.55e-9` and exits successfully. The README and first-run guide use a small
  converged common-API solve, and all 24 root names are enforced in the API
  reference.
- Published current quantitative Hartmann and pinned B2 integration evidence,
  including the external comparison limits and the explicit distinction
  between the two-update harness and production-mesh acceptance. Corrected the
  FreeMHD repository link and documented the pinned installer, FreeMHD, and
  OpenFOAM inputs.
- Added one reusable PR/weekly/release external-validation workflow. It rebuilds
  the pinned FreeMHD image, verifies both source commits, runs the independently
  observed B2 Docker comparison, uploads its full evidence tree, and checks all
  external documentation links. Release wheel smoke uses the common API and
  release CI executes the separate curated-example lane.
- The portable gate passes 504 tests in 85.1 seconds; the four-test curated
  lane executes all seven examples in 69.0 seconds. Exact combined line/branch
  coverage remains 95.089601% across 5,823 statements and 1,264 branches.
  Ruff, formatting, byte compilation, architecture/import, current-state
  prose, Sphinx warnings-as-errors, external links, and Actionlint pass.
- Package source remains 14,934 lines across 15 modules; tests are 11,640 lines
  across 13 files and the tracked checkout is 1,553,913 bytes. The wheel is
  142,375 bytes with 29 members and the source distribution is 133,462 bytes
  with 35 members; Twine and distribution-content audits pass.
- The hosted documentation job and all Python 3.10 test shards pass. The prior
  combined-coverage lane exposed a coverage-instrumented physics timeout on
  the two-core runner. The coverage workflow now omits the subprocess-only
  example shard, keeps all four source-covering shards, and gives instrumented
  numerical tests an explicit 300-second per-test/900-second shard budget.
- Next action: push this tranche and require the corrected hosted coverage,
  docs, and external-workflow checks to pass before closing the documentation
  phase. Production-mesh B1/B2 promotion remains a separate acceptance gate.

### 2026-08-22 — invariant-system reuse and hosted validation repair

- Reviewed the retained source definition inventory against the ownership map
  and live call graph. Every private definition has a source call site; generic
  Krylov, fixed-point, direct, and preconditioner algorithms remain owned by
  released SOLVAX. The retained small response systems are physical fixed-flow
  closures, and the 2-D/3-D elliptic paths remain matrix-free.
- Bounded 2-D history retention during execution instead of collecting every
  sample and slicing after the solve. Per-step diagnostics now cross the
  device/host boundary as one compact vector. Full histories remain available
  with `history_stride=1`, and restart/stride semantics are unchanged.
- Separated invariant fully developed coefficient assembly from repeated
  solves. Potential coefficients, face conductances, volume scaling,
  preconditioners, velocity coefficients, and constant-field material terms
  are built once and reused. This removed 17 equivalent static-preconditioner
  recompilations in the standard Hartmann solve while retaining direct SOLVAX
  calls and exact terminal numerics.
- A fresh-process, same-environment comparison used one cold and five warm
  repetitions per matrix at `b4c6431` and `46b5bf6`. Hartmann 48x48 improved
  from 11.447 s to 6.399 s cold, 7.936 s to 2.956 s warm median, and 1.719 GB
  to 0.765 GB peak RSS. Its 17 steps and residual `9.515209016541792e-9`
  are exact. Reduced B2 warm/RSS changed by +4.18%/-0.13%; reduced B1 changed
  by +3.78%/-3.47%. Their statuses, steps, residuals, flow, and charge closure
  are identical. The external record is
  `/Users/rogeriojorge/local/tests/lmx-audit/performance/lmx-system-reuse-46b5bf6.json`.
- The portable gate passes 505 tests in 83.1 seconds. Exact combined
  line/branch coverage is 95.019049% across 5,809 statements and 1,278
  branches; the curated lane executes all seven examples in 51.0 seconds.
  Package source is 14,991 measured lines across 15 modules, tests are 11,664
  lines across 13 files, the tracked checkout is 1,556,867 bytes, and the root
  API remains 24 names.
- Ruff, formatting, Actionlint, architecture/import, Sphinx `-W`, build,
  Twine, distribution inspection, and isolated wheel smoke pass. The wheel is
  142,773 bytes with 29 members and the source distribution is 133,808 bytes
  with 35 members. The benchmark command now reports a warm median and
  coefficient of variation instead of selecting the fastest warm sample.
- Repeated the pinned B2 Docker workflow on `46b5bf6` with FreeMHD `14b54a3`.
  The oversubscription-safe two-rank invocation passes execution, artifact,
  contract, independent-observation, and comparison gates with pressure Linf
  `0.0109172453` and RMS `0.0045179771`. Its evidence is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-optimized-46b5bf6/b2`.
- Hosted coverage and examples now use explicit numerical budgets: the
  optimized examples and compatibility benchmarks pass the Python 3.10 runner
  with the proven 300-second per-test ceiling, and exact combined coverage
  passes in 8 minutes 8 seconds. OpenMPI explicitly permits the declared
  two-rank smoke inside constrained containers. PR evidence upload is
  best-effort when the repository artifact quota is exhausted; scheduled and
  release uploads remain mandatory.
- Every latest source-bearing hosted gate is green: metadata/architecture,
  core, validation, examples, benchmarks, physics, exact combined coverage,
  warnings-as-errors documentation, external links, and the pinned FreeMHD B2
  Docker comparison.
- Next action: run production B2 on the declared 201x129x129 fine matrix using
  a suitable high-memory/GPU runner and establish a genuinely matched external
  executable for B1. Private branch protection remains unavailable under the
  current GitHub plan, and no repository-history rewrite occurs without
  separate explicit approval.

### 2026-08-22 — verified archive and rewrite candidate

- Created the private read-only
  [`uwplasma/LMX-archive`](https://github.com/uwplasma/LMX-archive) repository.
  It mirrors both live branches and all seven original tags. Its archival
  release contains the release inventory and a complete 37 MiB Git bundle with
  both pull-request refs in addition to the branches and tags.
- The final bundle SHA-256 is
  `eaaa528f998860a1cfb82d9c4355f7ab504d6628003eb254f453a3ca573d5fb8`.
  Local and remote-download copies pass `git bundle verify`; independent
  ordinary and mirror clones pass strict full `git fsck`, and the mirror clone
  reproduces the complete 12-ref inventory. The archive repository was then
  set read-only through GitHub's archive state.
- Exported a seven-release inventory covering 126 existing assets and
  838,216,193 bytes. Reusable research fields and media already reside in the
  812,719,898-byte `lmx-research-assets-v1` release, outside Git history.
- Constructed an exact standalone root candidate from `47f3c0b`. Its commit is
  `ac44448b0e6845483d9a0ddfeb3dfd2260f5c5f0`; an ordinary non-local clone is
  2,304 KiB with an approximately 465 KiB pack, 1,560,830 tracked bytes, and 85
  files. Architecture/import budgets pass and all 505 portable tests pass in
  79.2 seconds against that candidate.
- Confirmed the hosted minimum endpoint as Python 3.10, JAX/JAXLIB 0.6.2,
  NumPy 2.2.6, and released SOLVAX 0.14.0. The current endpoint uses Python
  3.13, JAX/JAXLIB 0.11.1, NumPy 2.5.2, and SOLVAX 0.14.0. All bounded shards
  pass at the minimum endpoint and the exact combined coverage gate passes at
  the current endpoint.
- Removed 46 redundant Actions artifacts from the deleted Q2D validation lane,
  releasing 245,123,217 bytes of quota. These ephemeral copies are no longer
  recoverable from Actions; their source history is in the verified bundle and
  their reusable data remains in the research-assets release.
- The final hosted six-repeat Hartmann command completed with a 10.251-second
  warm median and 0.00731 warm coefficient of variation on its two-core runner.
  Manual benchmark JSON is always printed in the job log, while its
  convenience artifact copy is best-effort during GitHub's documented quota
  recalculation delay. Scheduled/release evidence remains fail-closed.
- Updated the official GitHub actions to their current Node-runtime majors:
  checkout and setup-python v7, upload-artifact v7, and download-artifact v8.
  This removes forced deprecated-runtime execution without adding workflow
  files or changing numerical gates.
- Next action: obtain production hardware and a matched B1 executable, then
  require the complete production evidence before publication. Replacing live
  history with the reviewed root, deleting refs, force-pushing, and measuring
  the final authenticated clone remain withheld pending separate explicit
  destructive-action approval.

### 2026-08-22 — archived history and slim live root

- After explicit approval, force-updated live `main` to the reviewed standalone
  root `3d28992e853ee6b883a8d10fffe66d4c6f980d36`, deleted the obsolete feature
  ref and `v1.0.1` through `v1.1.3` tags, and repointed the research-assets tag
  without changing its release assets.
- Preserved the complete former repository in private read-only
  `uwplasma/LMX-archive` and its `archive-effa657` release. The final Git bundle
  SHA-256 is
  `eaaa528f998860a1cfb82d9c4355f7ab504d6628003eb254f453a3ca573d5fb8`;
  bundle verification and independent strict `git fsck` checks pass.
- A fresh authenticated GitHub clone is 2,304 KiB total with 588 KiB of Git
  data and a 463.39 KiB pack. It contains one reachable commit, 85 tracked
  files, and 1,563,958 tracked bytes, comfortably below the 10 MB clone gate.
- Next action: develop physics and validation from this enforced slim root;
  binary evidence remains in releases or untracked audit directories.

### 2026-08-22 — compact Q2D physics and released periodic algebra

- Added one 284-line `lmx.q2d` module for periodic Sommeria--Moreau vorticity
  flow. The physical path uses a two-thirds-dealiased Fourier representation,
  integrating-factor RK4 for viscosity and Hartmann friction, optional
  vorticity forcing and sparse frames, and the common `lmx.solve` result
  semantics. It reports energy, enstrophy, energy-budget closure, spectral
  divergence, and Courant status.
- Merged SOLVAX periodic-Poisson PR #80 and release PR #81 after every hosted
  check passed. SOLVAX 0.15.0 is tagged, published through trusted PyPI, and
  documented in its GitHub release. LMX now requires `solvax>=0.15,<1` and
  calls its reusable physical and spectral periodic-Poisson APIs directly.
- Q2D gates cover exact Taylor--Green decay, forcing, dtype/history contracts,
  nonlinear three-grid refinement, CFL rejection, input errors, and the common
  API. The Q2D module has 100% line/branch coverage. The complete portable gate
  passes 506 tests at 95.14% combined line/branch coverage; it completed in
  103.0 seconds on the heavily loaded audit Mac, above the 90-second warning
  target but below the 180-second hard budget. All seven curated examples pass
  through four executable catalog tests in 56.9 seconds.
- A controlled JAX 0.6.2 float32 run used the same $256^2$, 80-step problem for
  one CPU and one NVIDIA RTX A4000. Five warm repetitions give medians of
  2.2332 and 0.08663 seconds, respectively: 25.78x GPU speedup. Final fields
  agree to relative $L_2=2.38\times10^{-6}$ and
  $L_\infty=3.67\times10^{-6}$; warm coefficients of variation are 0.21% and
  1.39%. The untracked evidence is under
  `/Users/rogeriojorge/local/tests/lmx-audit/q2d-office-20260822`.
- Replaced the standalone autodiff demo with one Q2D example that generates a
  WebP poster, JSON diagnostics, and an MP4 when FFmpeg is available. The
  shipped poster is 36,302 bytes. The prospective checkout is 1,626,380 bytes
  across 89 files, including 154,600 media bytes; package source is 15,291
  lines across 16 modules, tests are 11,770 lines, and the root API has 26
  names.
- The 1.2.0 wheel has 30 members and is about 143 KiB. Build and Twine checks
  pass, metadata requires released SOLVAX 0.15, and an isolated installation
  passes the CLI plus converged Q2D and Hartmann solves without source-tree
  imports.
- Audited FreeMHD's public `S3_Buhler_Ha616` archive and its completed 96-rank,
  10.45-million-cell OpenFOAM v2206 run. Its executed thermophysical values
  imply $Ha=573.34$, $Re=3193.09$, $N=102.95$, and $c=0.07939$; the archive's
  mesh macros imply the named $Ha=615.50$. Both differ materially from ALEX
  B1 ($Ha=6600$, $N=10700$, $c=0.027$), so the case is retained only as a
  distinct future pipe-fringe target. The untracked fail-closed contract is
  `/Users/rogeriojorge/local/tests/lmx-audit/freemhd-s3-buhler-contract-20260822.json`.
- Committed the immutable candidate as `4e4bed3` and repeated the pinned B2
  Docker comparison. Execution, artifact identity, contract, independent
  native-output observation, and comparison all pass with no failed checks;
  pressure $L_\infty=0.0109172453$ and RMS $=0.0045179771$. As required, the
  harness remains non-accepting only because the branch has the candidate role.
  Its evidence is
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-q2d-1.2.0-candidate/b2-4e4bed3`.
- Next action: open the LMX 1.2.0 pull request and require every hosted gate
  before merge. Two-GPU strong scaling waits for the unrelated office workloads
  to release both devices.

### 2026-08-22 — differentiability as a solver and physics contract

- Made end-to-end differentiability a protected product invariant. Continuous
  state, material, field, forcing, supported geometry, and objective inputs
  must reach production fields through one traced numerical core; discrete
  topology, iteration limits, checkpoint widths, convergence decisions, I/O,
  and plotting remain explicit static or host-side boundaries.
- Classified every SOLVAX numerical family by its production derivative and
  acceptance evidence. Converged linear/root systems use one implicit tangent
  or transposed solve without recording iterations; long finite recurrences
  use exact discrete derivatives with bounded checkpoint replay; specialized
  direct, mixed-precision, localized, eigenpair, and Jacobian paths retain
  their explicit memory and transform contracts.
- Merged SOLVAX PR #82 and published SOLVAX 0.16.0 after the complete hosted
  build, lint, typing, minimum/current/optional-backend, Linux, and macOS matrix
  passed. The released `checkpointed_fori_loop` retains `O(N/C+C)` recurrence
  state with a square-root default and preserves the exact finite JVP/VJP.
  The public wheel SHA-256 is
  `65ab3f93bc65bcf3317d02e92533db7d2d7f65c0ef895007172356449a980fb4`.
- Added the field-only `evolve_q2d` API and routed both it and the diagnostic
  solve through the released checkpointed recurrence. Analytical derivatives
  cover amplitude, viscosity, Hartmann friction, domain length, and timestep;
  forcing uses an independent central difference, and a JVP/VJP identity gates
  composition.
- On a 128-step, `24 x 24` float32 Q2D objective, compiled reverse temporary
  memory falls from 18,322,800 to 1,983,664 bytes (9.24x) while warm
  value-and-gradient time changes from 0.032277 to 0.032831 seconds (1.7%).
  The 96-square, 80-step primal remains 2.14% faster than the immutable
  pre-adjoint candidate.
- Audited the existing Hartmann-only gradient wrapper and rejected its current
  fixed-Jacobi/coupling tape as the production interface. Documentation now
  advertises only the accepted Q2D field core. The next 2-D tranche replaces
  that wrapper with the retained fully developed equations and a SOLVAX
  implicit derivative before restoring an end-to-end Hartmann claim.
- Reviewed Optimistix implicit/checkpoint adjoints, PETSc TSAdjoint/Revolve,
  JAX checkpointing, Diffrax, and JAX-CFD. Their shared algorithm-selection
  principle is recorded in D-030 and the bibliography; LMX keeps its own
  compact production equations rather than importing another framework.
- The updated portable LMX gate passes 506 tests at 95.15% exact combined
  line/branch coverage. Sphinx warnings-as-errors, Ruff, formatting, and the
  architecture/import audit pass at 16 modules, 15,356 package lines, and 27
  root exports. Office GPU derivative/scaling measurements remain queued
  because both devices are fully occupied by unrelated jobs.
- Built the 147,168-byte, 30-member LMX 1.2.0 wheel with an explicit
  `solvax>=0.16,<1` requirement. Twine passes, and a fresh non-editable install
  from public dependencies completes the CLI, a compiled Q2D
  value-and-gradient, and a converged Hartmann smoke with SOLVAX 0.16.0.
- Committed the immutable source candidate as `6225e3c` and repeated the pinned
  two-rank B2 Docker comparison with FreeMHD `14b54a3` and image
  `sha256:44e9c9c1cc9d69aa4e50ca773aef7971010905a529462273bf913e6bc5b2c238`.
  Execution, artifact identity, matched contract, independent observation, and
  comparison gates pass with no failed checks; pressure Linf is 0.0109172453
  and RMS is 0.0045179771. The harness remains non-accepting only because its
  two-update candidate role cannot establish production-mesh B2 acceptance.
  Evidence is under
  `/Users/rogeriojorge/local/tests/lmx-audit/lmx-q2d-adjoint-6225e3c/b2`.
- Next action: push the evidenced candidate, require every hosted PR gate, and
  merge PR #2 only when the latest source-bearing checks are green.

### 2026-08-22 — implicit fully developed field tranche

- Merged SOLVAX solver PR #84 and release PR #85, then published 0.17.0 from
  merge `d42b728` after
  every hosted minimum/current/optional JAX, Linux, macOS, build, lint, typing,
  and coverage gate passed. `affine_fixed_point_gmres` now uses one implicit
  tangent or transposed FGMRES solve and never records Krylov iterations. The
  public wheel is 145,489 bytes with SHA-256
  `9ad8d4c0e145131ec594d736f20e88fe16c435364f6d6d6e3fdd8ffa15e266cb`;
  a fresh public install compiled and ran its value-and-gradient example.
- Deleted the Hartmann-only fixed-Jacobi differentiation problem, solver,
  objective, and gradient helpers. One `solve_fully_developed_fields` API now
  composes the retained mesh/material assembly, potential equation, momentum
  equation, wall interpolation, current recovery, and Lorentz force with the
  SOLVAX implicit coupled solve. This removes a shadow discretization while
  increasing the stable root API by one name.
- Rectangular Hartmann field parity against `solve_steady` is `1.51e-6`
  relative on the acceptance mesh. Forcing sensitivity satisfies the exact
  linear response identity; the magnetic-scale reverse derivative agrees with
  a centered difference to `3.91e-7` relative on a `16 x 16`, Ha=20 case, and
  the field JVP/VJP bilinear identity passes. A separate Shercliff
  magnetic-scale finite-difference gate passes.
- On that `16 x 16` float64 objective, the compiled primal warm median is
  1.305 ms with 343,472 temporary bytes. Value-and-gradient is 2.812 ms with
  376,984 temporary bytes, a 2.15x marginal runtime and only 9.76% more
  compiler-reported temporary storage. Compile times are 2.56 and 2.42 seconds.
- Upgraded the flexible volume-potential FGMRES path to
  `solvax.linear_solve` with the same established right preconditioner applied
  to the transposed operator, so its iterations are not taped. The focused
  conducting-rectangle solve and analytical RHS-scale gradient gate pass on
  current JAX and the Python 3.10 lane's JAX 0.6.2. An attempted eager
  `linear_transpose` of the tensor preconditioner was rejected after JAX 0.6.2
  raised inside the underlying scan even during primal type checking.
- Rejected the first layered Hunt end-to-end adjoint after the transpose
  coupled solve diverged despite a converged primal. The centered field-scale
  derivative is `-9.973377e-3`; the failed adjoint returned values between
  `5.52e11` and `-1.76e21`, with a measured transposed FGMRES residual of
  `3.70e22`. The public API now fails immediately on layered geometry instead
  of returning a plausible-looking false gradient. A block formulation or
  physics-aware transpose preconditioner is required before enabling Hunt.
- LMX now requires the public `solvax>=0.17,<1` release and pins 0.17.0 in its
  minimum-dependency lane. The architecture audit passes at 16 modules, 15,272
  package lines, 6,009 maintained-core lines, and 28 root exports: the accepted
  field API and tests reduce total source by 84 lines versus the preceding
  candidate. The complete public-wheel gate passes 505 tests in 104.2 seconds
  at 95.12% exact combined line/branch coverage; all curated examples, Ruff,
  formatting, architecture/import, warning-free HTML docs, linkcheck, build,
  Twine, distribution-content, and fresh-wheel value-and-gradient checks pass.
- The LMX 1.2.0 tagged release workflow independently completed the pinned
  FreeMHD run with no failed physics checks, but the workflow failed afterward
  because GitHub artifact storage quota was exhausted; the link checker also
  received bot-only HTTP 403 responses from three valid DOI resolvers. The next
  release treats artifact upload as non-verdict infrastructure and ignores
  only those exact persistent DOI URLs while continuing to check their
  associated software/preprint links.
- Merged the production implicit-field implementation as PR #3 at `042f9ef`
  only after every hosted CI, documentation, external-link, and pinned FreeMHD
  Docker gate passed. The release candidate is 1.3.0; its fresh-wheel smoke
  compiles both the primal solver and a magnetic-field value-and-gradient.
- Removed 580 redundant non-release GitHub Actions artifacts while retaining
  26 distribution and release-validation records. This restores deliberate
  evidence retention without using transient CI products as permanent storage.

### 2026-08-22 — layered Hunt implicit adjoint

- Traced the failed Hunt transpose to momentum assembly rather than Krylov
  restart policy. Fluid boundary coefficients were correctly included in the
  diagonal but were also retained as off-diagonal couplings to solid cells
  whose rows were identity constraints. The primal stayed unchanged at zero
  solid velocity, while the full operator became nonsymmetric and invalidated
  PCG's implicit transpose solve.
- Eliminated only those inactive-neighbor off-diagonals while preserving their
  no-slip diagonal contribution. This keeps the maintained source at 6,009
  core lines and makes the volume-scaled layered momentum operator symmetric
  to `4.33e-17` relative without adding a solver or preconditioner.
- Enabled layered ducts in `solve_fully_developed_fields`. On an `8 x 8` fluid
  Hunt case with one wall cell per side, all five returned fields match the
  production steady solve within `1.67e-7` relative. The magnetic-scale VJP
  agrees with a centered difference to `8.79e-8` relative; the exact forcing
  identity agrees to `3.34e-11` relative.
- The compiled Hunt reverse pass has a 36.91 ms warm median versus 34.24 ms for
  the primal (1.08x) and 194,872 versus 129,976 temporary bytes (1.50x). It
  differentiates the converged system without retaining PCG, potential, or
  coupling iterations.
- The complete portable gate passes 506 tests in 110.3 seconds at 95.12%
  combined line/branch coverage. All curated examples, Ruff, formatting,
  architecture/import checks, warning-free HTML documentation, and external
  link checks pass at 16 modules, 15,272 package lines, 6,009 core lines, and
  28 root exports.
- Next action: merge the layered-Hunt tranche after every hosted gate passes,
  then expose a differentiated 3-D field core. GPU parity remains queued until
  an office device is unoccupied.

### 2026-08-22 — differentiated production 3-D field recurrence

- Merged the layered-Hunt formulation as PR #5 at `3748653` after every hosted
  metadata, architecture, Python 3.10, physics, benchmark, combined-coverage,
  documentation, and package-build gate passed.
- Added `lmx.fringing.evolve_extruded_fields` without adding a root export or
  source file. Generic rectangular and layered ducts now return velocity,
  pressure, potential, current, and Lorentz-force fields through the same
  finite update used by `solve_extruded_inductionless`. Specialized ALEX B2
  and pipe formulations fail closed rather than borrowing an unvalidated
  derivative contract.
- Preserved the validated finite collocated pressure projection and routed its
  reverse pass through SOLVAX checkpointing. A converged nearest-neighbor PCG
  replacement was rejected because the collocated central divergence/gradient
  pair is not that Poisson operator and it regressed mass conservation.
  Electric closure remains implicit; the outer recurrence uses the same exact
  `checkpointed_fori_loop` with a square-root storage default.
- All 11 field outputs match the ordinary production solve to numerical
  precision on the acceptance problem. Pressure-forcing and magnetic-scale
  VJPs agree with centered differences; JVP/VJP duality passes. On the
  eight-step `4 x 4 x 4` float64 gate, checkpointing reduces compiled reverse
  temporary storage from 354,456 to 228,824 bytes (35.4%) with identical value
  and gradient.
- The implementation preserves the existing architecture ceiling at 16
  modules, 28 root exports, and 15,363 package lines. It removes the duplicated
  generic production update and groups immutable step inputs once; no solver,
  proxy discretization, test file, or experimental lane was added.
- The complete local gate passes 508 tests in 158.2 seconds at 95.07% combined
  line/branch coverage. Ruff, formatting, architecture/import, all curated
  examples, warning-free HTML documentation, and external link checks pass.
- Next action: complete hosted CI, package, and FreeMHD gates, then merge the
  3-D tranche. Measure CPU/GPU parity and strong scaling as soon as an office
  GPU is available.

### 2026-08-22 — sub-10-minute evidence architecture

- Measured the hosted PR critical path rather than reducing test scope:
  combined coverage took 15m21s because it reran four already-tested shards
  serially; the unsplit physics lane took 9m23s without coverage and 11m49s
  with coverage. Core and validation took only 40s and 17s of test time.
- Split the existing physics inventory at its natural ownership boundary:
  54 fringing/3-D tests and 117 fully-developed/solver tests. With branch
  coverage enabled, the local lanes pass in 2m55s and 1m31s respectively.
  All six local evidence files combine to the unchanged 95.12% repository
  line/branch result across 5,950 statements and 1,304 branches.
- CI now runs quality and three evidence lanes immediately, runs every test once,
  records JUnit summaries in the job logs, transfers covered-lane databases by
  exact run-scoped cache keys, and performs only a report merge afterward.
  Every numerical, physics, literature-contract, regression, API, and example
  assertion remains active; no tolerance or coverage exclusion was relaxed.
  Superseded runs cancel automatically.
- Enabled lane-scoped JAX persistent compilation caching with source-keyed,
  branch-scoped GitHub caches. Release validation, documentation, external
  links, and pinned FreeMHD start concurrently and reuse the PR CI definition,
  eliminating the separate 18-minute release-only full-suite rerun.
- The cold-cache target is a PR critical path below 10 minutes, enforced by
  10-minute job and 9-minute test budgets. Warm trusted-main caches should
  reduce repeated XLA compilation further without entering numerical timing
  claims; the standalone benchmark workflow remains the performance oracle.
- A first cold hosted run with six evidence lanes passed every check, including
  the cache-transport coverage merge, but finished in about 10m16s because the
  6m30s fringing lane waited 3m29s for one of four runner slots. Core,
  validation, examples, and benchmarks are therefore one serial `support`
  lane; together they remain shorter than fringing, reduce setup/cache churn,
  and let support, fringing, physics, and metadata start within available
  concurrency. The fused covered lane passes all 336 tests locally in 2m40s.
- Next action: require the redesigned hosted matrix and combined coverage to
  pass, record its cold-cache wall time, then merge and use it to revalidate the
  pending differentiated 3-D tranche.
- The final three-lane hosted PR gate passed from creation through combined
  coverage in 8m08s, including a 1m43s GitHub runner queue. Actual lane times
  were 5m49s for support, 4m56s for fringing, 4m01s for physics, 16s for
  metadata/architecture, and 22s for the report-only coverage merge.
- The first optimized release run completed all tests, exact coverage, docs,
  links, pinned FreeMHD (7m03s), distribution inspection, and fresh-wheel
  differentiability smoke. Only the final distribution artifact upload failed
  because GitHub's previously cleared quota had not recalculated. Distribution
  handoff now uses an exact run/SHA-scoped cache, and same-commit main/release
  CI calls share a concurrency key so the release automatically replaces the
  redundant push run.
- Next action: require the quota-independent release PR to pass, publish 1.3.0,
  then revalidate and merge the pending differentiated production 3-D tranche.
- The integrated 3-D cold-cache run kept metadata (16s), physics (4m30s),
  support (5m26s), documentation (25s), links (23s), and pinned FreeMHD
  (7m06s) within budget. Its 56-test fringing lane passed in 7m57s, but
  two-worker `loadgroup` scheduling left a long-tail imbalance after a 151s
  derivative test. The runner now uses xdist work stealing; the exact covered
  two-worker lane passes locally in 2m35s. The small derivative acceptance
  problem also uses ten converged electric-closure updates instead of twenty,
  while retaining production parity, two-parameter finite differences,
  JVP/VJP duality, and the compiled reverse-memory bound.
- The integrated run's report-only coverage job never started: GitHub marked
  it failed because the organization currently reports failed payment or an
  exhausted Actions spending limit. PyPI likewise rejected the release OIDC
  exchange because this first publication has no pending trusted publisher.
  These are external account settings; no evidence gate was bypassed.
- Next action: restore GitHub Actions billing, register the exact pending PyPI
  publisher, rerun the warm-cache gate, publish 1.3.0, and merge the 3-D
  tranche only after every required check is green.

### 2026-08-28 — research position and local evidence authority

- Reviewed the live capabilities, APIs, and publications of ParaStell, NekRS,
  SAM MHD, FreeMHD/FreeMHD2, JAX-Fluids, XLB, JAX-CFD, JAX sharding and
  checkpointing, and Firedrake/JAX implicit differentiation, together with
  liquid-metal blanket and ALEX validation literature. General GPU CFD,
  differentiable CFD, stellarator CAD/neutronics, and free-surface/full-
  induction MHD are already active fields; none alone is a defensible LMX
  novelty claim.
- Fixed the product position in D-035: LMX leads through the combined evidence
  of compact high-Hartmann duct/fringe physics, exact production derivatives,
  bounded adjoint memory, accelerator design throughput, and matched external
  validation. It exchanges sampled geometry/field and engineering observables
  with ParaStell-style workflows instead of absorbing CAD or neutronics.
- Added sequential JCP methods, JPP/Physics of Plasmas physics, and Nuclear
  Fusion/Fusion Engineering and Design application programs. Each has
  falsifiable numerical, derivative, performance, physical, and reproducibility
  gates; no paper claim depends only on JAX compatibility or qualitative plots.
- GitHub Actions billing may remain unavailable until the next billing cycle,
  and PyPI/public GitHub release work is deferred to the final public phase.
  Complete recorded local evidence is therefore the temporary merge authority;
  hosted gates are not deleted or weakened.
- Next action: execute the complete local evidence gate on the optimized
  differentiated 3-D branch, merge it only if every check passes, then expose
  station-wise field coefficients and engineering objectives as the first
  optimization-grade design tranche.

### 2026-08-28 — differentiated 3-D local merge evidence

- Candidate `fb1937ec82f2005deaceee32779b65246626ab5e` was tested on macOS
  14.4.1 arm64 with Python 3.11.14, JAX/JAXLIB 0.9.2, SOLVAX 0.17.0, and the
  CPU backend. The later evidence-log commit changes only this Markdown file.
- `PYTHONPATH="$PWD/src" COVERAGE_FILE=/tmp/lmx-pr7-full-absolute.coverage
  python scripts/run_full_test_suite.py --budget-seconds 1200
  --warning-seconds 600 --test-timeout-seconds 300 --coverage-fail-under 95
  --coverage-xml /tmp/lmx-pr7-full-absolute.xml --junit-xml
  /tmp/lmx-pr7-full-absolute-junit.xml` passed 508 tests in 108.6 s with
  95.07% combined branch coverage. The coverage and JUnit SHA-256 digests are
  `5283ef2a04931c2e5020c306a25807d3a820380936339d54bcd390d00638dcf3`
  and `caee36623d53280d73f0fdf450339632ec5867786580474314e4e3486d8cb1b6`.
- Ruff check/format and `scripts/audit_architecture.py` passed: 16 modules,
  6,009 core lines, 15,363 total lines, 28 exports, and seven curated
  examples. The curated first-run test passed independently. Sphinx HTML and
  linkcheck both passed with warnings as errors when pointed at the absolute
  candidate source path.
- The isolated sdist/wheel build and Twine inspection passed. The wheel and
  sdist are 145 KiB and 137 KiB with SHA-256 digests
  `d4cf5dc1f79b3065b8c4971aabc38621bcc32c0622ff94e23df4f2479a9e9de4`
  and `67ce0c229d73ee310818211906d8a351fdd979e22e17a57beef0792bd9190706`.
  A fresh non-editable wheel environment passed the CLI smoke and a real 3-D
  value-and-gradient calculation.
- The matched-B2 preflight passed against FreeMHD
  `14b54a3e8e1a05b6ee4c98331995abaaae96e7a5` and installer
  `36f409d294ba3170d64d4073378d5ef68401072f`; contract SHA-256 is
  `e30650045508cab8fce34a421e733591ff9f7503e322b54468dfdd300e11588a`.
  A redundant local OpenFOAM execution could not start because Docker's
  62-GiB virtual filesystem had zero free bytes, causing `apt` index writes to
  surface as signature errors. No unrelated Docker data was deleted. The
  source-bearing revision had already passed the pinned hosted FreeMHD solver
  comparison, and subsequent commits do not change an MHD operator.
- The first aggregate invocation used relative `PYTHONPATH=src`; subprocess
  examples changed working directory and imported an older checkout, producing
  three import failures while 505 tests passed. The authoritative command uses
  the absolute source path above and passed completely.
- Verdict: the differentiated generic rectangular/layered 3-D field path is
  accepted for merge. Specialized B2 and pipe derivatives, genuinely matched
  B1 validation, field-profile design variables, and multi-device derivative
  evidence remain explicitly open.

### 2026-08-28 — axial field controls and engineering objectives

- Candidate `d4feb0e470a2c5ee212096b77224a05935e725d2` accepts either one
  imposed-field multiplier or one continuous coefficient per axial station in
  the production rectangular/layered 3-D recurrence. The existing acceptance
  test now differentiates forcing plus all three station coefficients, checks
  centered differences and JVP/VJP duality, compares the vector-of-ones path
  with the ordinary production solve, and retains the bounded-memory gate.
- Added `extruded_engineering_objectives` in the existing `lmx.fringing`
  module. It returns a JAX PyTree of signed pressure drop, outlet flow rate,
  pumping power, squared outlet coefficient of variation, wall-current-density
  RMS proxy, and smooth recirculation fraction. It adds no optimizer, source
  file, root export, or data container; documented units and validation limits
  remain explicit.
- The exact-commit local gate passed 509 tests in 111.6 s with 95.10% combined
  branch coverage. Coverage and JUnit SHA-256 digests are
  `6f7b60278a7e73e2629842689868552b1a19c03768092676f5e3c8ac429dc6fe`
  and `efbe21555d2a9b36a311e6348f8bee81e6f3768c4377613c7bbf0948c179e653`.
  Ruff, all seven curated examples, Sphinx HTML/linkcheck, architecture audit,
  isolated build/Twine inspection, and a fresh-wheel end-to-end profile-gradient
  smoke also passed.
- The wheel and sdist remain 146 KiB and 138 KiB, with SHA-256 digests
  `6eda66f7ae19eb4288644fa314067bf1876ddebf3443340187134ea265b1aa82`
  and `977cd375ca322256098988e048490839e74ff1113ab94556ff3a0372726d72fb`.
  The pinned matched-B2 preflight retains contract SHA-256
  `e30650045508cab8fce34a421e733591ff9f7503e322b54468dfdd300e11588a`.
- Compactness remains fail-closed: 16 modules, 28 root exports, 6,009 core
  lines, and 15,426 total package lines. The source ceiling increased by only
  60 lines and the test ceiling by five after consolidating derivative tests;
  no new file was introduced.
- Next action: add continuous wall/material conductance controls through the
  accepted layered operator, then gate batched design throughput and CPU/GPU
  derivative parity before presenting an optimization result.

### 2026-08-28 — differentiated material conductance controls

- Candidate `52889caef94c76a4dd0712c278f63e756b2ecd83` extends the accepted
  rectangular/layered production recurrence with one continuous
  `material_conductivity_scale`: a scalar scales every material, while two
  coefficients independently scale fluid and explicit solid-wall regions.
  Unit coefficients reproduce the ordinary production solve. Geometry,
  region topology, timestep, and checkpoint policy remain static so one
  optimization uses a fixed discrete recurrence.
- The layered acceptance gate differentiates the cell-centered wall-current
  objective with respect to both material coefficients. Reverse gradients
  agree with independent centered differences within `5e-4`, JVP/VJP duality
  passes, and the electric closure retains the SOLVAX implicit adjoint rather
  than tracing PCG iterations. This work also found and fixed a static-assembly
  leak in `extruded_engineering_objectives`; the README's complete layered
  value-and-gradient example now executes under `jax.jit`.
- The exact-commit local gate passed 509 tests in 116.05 seconds (117.6 seconds
  including evidence assembly) with 95.12% combined branch coverage. Coverage
  and JUnit SHA-256 digests are
  `29374a96b2d59731219b31e7943ac4d186cdc738db4f75e4b26a426aea410f35`
  and `7da6e19740aaf11db749efe61798f3e03fbb3648e134f8513f9f9e7d56a50f92`.
  Ruff, all curated examples, Sphinx HTML/linkcheck with warnings as errors,
  architecture/import budgets, isolated build, Twine, and distribution-content
  inspection passed.
- The wheel and sdist are 149,791 and 141,601 bytes with SHA-256 digests
  `f6018c81955df2242de9d4436ffea74d265572181e53dae78be05255e014ba8b`
  and `8163a034f83361755e67b4151cd16db0b27c18065ff7eae762a73e9ae7aa0361`.
  A fresh non-editable wheel environment passed the CLI and an end-to-end JIT
  layered material gradient. The pinned FreeMHD B2 preflight passed from a
  temporary detached source clone with unchanged contract SHA-256
  `e30650045508cab8fce34a421e733591ff9f7503e322b54468dfdd300e11588a`.
- No module, root export, example, or runtime dependency was added. The narrow
  capability budget is 15,440 package lines and 12,030 test lines; the
  candidate uses 15,437 and 12,026, with 6,009 core lines, 16 modules, 28
  exports, and seven curated examples.
- SOLVAX pseudo-transient continuation PR 87 merged at
  `8ce026c036a6779d88b02e790ddf771e6c8233bf` after 750 local tests at 98.33%
  coverage plus green minimum/current/advanced/macOS, docs, lint, type, and
  Codecov checks. Its hosted current/advanced jobs took more than 14 minutes;
  the next SOLVAX infrastructure audit must preserve evidence while splitting
  or parallelizing that work below the ten-minute development target.
- Next action: merge this LMX tranche, then gate batched design throughput and
  CPU/GPU material-gradient parity before the first constrained wall/field
  optimization result.

### 2026-08-28 — bounded batched design gradients

- Candidate `e6f4081c1b583981daad1d0d565dbe0a492cd1a1` extends the existing
  layered acceptance gate to one combined forcing, imposed-field, fluid-
  conductivity, and wall-conductivity vector. All four reverse derivatives
  match independent centered differences; JVP/VJP duality and direct `vmap`
  parity with separate compiled evaluations pass.
- The user workflow uses JAX composition instead of an LMX ensemble layer.
  `jax.vmap` is accepted, while the tutorial recommends bounded
  `jax.lax.map(..., batch_size=k)` chunks so design throughput does not imply
  unbounded device memory. The executable two-design tutorial passes under
  `jax.jit` and returns finite values and a `(2, 9)` gradient array.
- On the local arm64 CPU, eight full value-and-gradient evaluations took
  1.840 ms sequentially, 1.569 ms in chunks of two, 1.587 ms in chunks of
  four, and 1.535 ms under full vectorization after compilation. Compiled
  temporary storage was 59,584, 115,256, 216,632, and 408,200 bytes,
  respectively. The evidence therefore supports chunks of two as the current
  CPU throughput/memory default; it does not support an accelerator claim.
- The exact-commit local gate passed 509 tests in 114.60 seconds (115.8 seconds
  including evidence assembly) with 95.12% combined branch coverage. Coverage
  and JUnit SHA-256 digests are
  `d00407c0c0dd4016d33835a8fd10f2a0d33c5a63fa3e6f0d6f17fed0ed682832`
  and `a34a34f39728cc7d7d3d7442fc2a7173cb5cda34db24084b46b5f3aa6a64611d`.
  Ruff, formatting, architecture/import budgets, and Sphinx HTML/linkcheck
  with warnings as errors also pass. No package source, public API, file,
  dependency, or architecture budget changed in this tranche.
- A fresh authenticated clone of merged material-control commit
  `96b7f0b1cdbba19d805ec3cd477b4cfdbebb6407` is 2,736 KiB total with 872 KiB
  of Git data, 88 tracked files, a 749.92 KiB pack, and a 518,181-byte source
  archive. The live repository remains far below its 9,766 KiB limit.
- `ssh office` timed out at the configured endpoint before authentication, so
  no GPU parity, speedup, memory, or scaling claim is made. Retry the real
  A4000 gate when the host is reachable; local emulation is not a substitute.
- Next action: obtain one- and two-A4000 primal/gradient evidence, then run the
  first bounded wall/field optimization. In parallel, split or parallelize the
  SOLVAX current/advanced jobs that exceeded fourteen hosted minutes.

### 2026-08-28 — branch-covered CI and bounded wall/field design

- SOLVAX PR 88 merged at `b46affe36be04a6a15c0a8d1231f854662c203f4`.
  The exhaustive current stack is split into two timing-balanced shards that
  execute all 40 test files exactly once; focused Python 3.10/macOS and advanced
  lanes retain their distinct contracts. The final hosted run reached combined
  coverage in 9 minutes 21 seconds, with no job exceeding 8 minutes 3 seconds.
  The exact local aggregate passed 750 tests plus six optional skips in 135.0
  seconds at 97.45% branch coverage. The combined hosted report covered 3,663
  statements and 696 branches and rounded to 98%; project and patch coverage
  checks passed.
- LMX candidate `b59018b1e6122d7563ea4d8b4f06f9bcada7feac` replaces the
  former variable-field demonstration with a bounded layered-duct design over
  seven axial imposed-field coefficients and wall conductivity. It adds no
  package module, public export, optimizer dependency, or example file. Forty
  compiled application-level updates reduce normalized loss from 0.900000 to
  0.611054 while preserving the imposed-field mean exactly and constraining
  field coefficients to 0.8–1.2 and wall scale to 0.5–1.5.
- The optimized portable case reduces pumping-power magnitude by 54.97% and
  wall-current-density RMS by 45.80% while changing flow by +0.0667%.
  Nonuniformity increases by about 6%, and the published plot exposes that
  tradeoff. This is workflow and derivative evidence on a 7x6x6 mesh, not a
  resolution-independent physical optimum. The JSON and retained WebP hashes
  are `b59d3460a7ef84534bf67c47151541e7bce9296ceea262724dffea3d8cdbc02e`
  and `df201c9655e8666bc71ab98f38bfdccf1bb201481d10af2725115cfe19356f77`.
- The study exposed a numerical derivative defect rather than masking it with
  a loose example tolerance. A `1e-12` electric-potential primal tolerance left
  parameter-dependent solve error large enough that the complete implicit VJP
  disagreed with centered differences at order one. Solving the primal to a
  float64 roundoff-level tolerance and reusing the existing transverse modal
  coarse operator reduces the complete eight-control relative gradient error
  to `1.40995e-6`. The transpose solve keeps its independently sufficient
  `1e-12` tolerance, avoiding unnecessary adjoint iterations.
- On the same local 7x6x6 eight-step engineering objective, the accepted
  value-and-gradient median is 67.45 ms versus 71.04 ms on merged main, a 5.1%
  improvement. Compiled temporary storage is 2,324,032 versus 2,274,048 bytes,
  a 2.2% increase. The comparison reports accuracy and memory together; it does
  not trade the corrected derivative for a misleading speed claim.
- The exact candidate content passed 509 tests in 146.3 seconds on macOS
  14.4.1 arm64, Python 3.11.14, JAX/JAXLIB 0.9.2, SOLVAX 0.17.0, and the CPU
  backend, with 95.12% combined branch coverage. Coverage and JUnit SHA-256
  digests are `770c8c372c81652d176bda63176c1d8a4dac9daf33634d4c1eb7d572c2c6840e`
  and `3f4c42b8fc024b2a2e8870f98c5daffdb6abd6e2efcc134b2578a4e5a7c5f586`.
  Independent complete compatibility runs passed all 509 tests on Python
  3.10.21/JAX 0.6.2 in 127.9 seconds and Python 3.13.9/JAX 0.11.1 in 119.3
  seconds; their JUnit hashes are
  `c1faebf9790462d5f478a5a4980ec229775cacaff3aab0bd64cbbae731b8445f`
  and `3c4a27e43f9a0d902f56e77cee251d9cee6ea02eade456e6222a4a0032dceb7a`.
- Ruff check/format, the architecture/current-prose/import audit, every curated
  example, Sphinx HTML and external links with warnings as errors, isolated
  build, Twine, distribution inspection, CLI, and a fresh non-editable-wheel
  3-D JIT gradient smoke passed. The 150,211-byte wheel and 142,125-byte sdist
  have SHA-256 digests
  `e4495d6ac0e4ad66aac18b83a77c1cac001310f756d49bbebbad43883cd873e0`
  and `cac052427670f27cb2360be430d7a12dc65e658cd5ea89eb360fd96a61cf57c4`.
- Compactness remains fail-closed at 16 modules, 6,009 maintained-core lines,
  15,447 package lines, 13 test files, 12,034 test lines, 28 root exports,
  seven examples, and 1,759,613 checkout bytes. The two narrow ceilings moved
  only to 15,450 package and 12,035 test lines; the added lines buy a corrected
  production derivative and end-to-end physical optimization assertions.
- The pinned B2 preflight passed against FreeMHD
  `14b54a3e8e1a05b6ee4c98331995abaaae96e7a5` and installer
  `36f409d294ba3170d64d4073378d5ef68401072f`, retaining contract SHA-256
  `e30650045508cab8fce34a421e733591ff9f7503e322b54468dfdd300e11588a`.
  A newly built amd64 Docker image then executed both LMX and FreeMHD: contract,
  artifacts, execution, observation, and comparison passed with pressure
  L-infinity/RMS discrepancies 0.01092/0.004518. The report intentionally leaves
  `acceptance_pass=false` because the two-update candidate role cannot establish
  production-mesh B2 validation; its report SHA-256 is
  `839ffc7ccad567e9f99caf5368277fbaf063b9463a64c9d7331db60b69c8ae18`.
- No GPU claim is added: the office A4000 host remains unreachable. Next action:
  merge this locally evidenced tranche, then obtain one-/two-A4000 primal and
  gradient evidence, add smooth-coordinate and pipe derivatives, and promote
  B2/B1 only through production-resolution matched external comparisons.

### 2026-08-28 — fixed-topology geometry design core

- LMX PR 15, the bounded design study, merged at
  `57833cab6bc4904e580cd3df69881a9101ecc621`; geometry-control PR 16 merged at
  `069732139b3b6c071dd0360ad4a2e1d67cc93ac9`.
  Candidate commits `0dbb684be5af94709162dee09f1b886612c6383d` and
  `eb6994a` add fixed-topology axial-length, transverse-width, and
  transverse-height controls to the same production rectangular/layered 3-D
  recurrence. Coordinates, spacings, cell areas, transport, projection, and
  engineering integration weights change coherently; topology and imposed-field
  samples remain static by contract.
- The seven-control layered derivative gate is now a twelve-control gate over
  pressure forcing, station-wise magnetic field, fluid/wall conductivity, and
  three geometry scales. Independent centered differences give relative
  geometry-gradient error `6.36796e-7`; the complete design example gives
  relative gradient error `3.03215e-6` and retains JVP/VJP and batched-map
  checks. No optimizer dependency, package module, public export, or example
  file was added.
- Forty bounded updates reduce portable-example loss from `0.900000` to
  `0.554358`. The accepted scales are axial `0.909157`, width `1.040245`,
  height `0.963902`, and wall conductivity `0.539592`; the seven field
  coefficients keep mean one and stay within 0.8--1.2. Pumping-power magnitude
  falls 68.27%, wall-current-density RMS falls 46.73%, and flow changes 0.336%.
  This 7x6x6 result demonstrates the design workflow, not mesh-independent
  physical optimality. The regenerated summary and tracked WebP SHA-256 digests
  are `43a7e1fb713a13ec385aff02853025d1c2080c71b94dff99b07309ece26b1a91`
  and `f8185a85eed2f0010797d399e4bd7d8545ee24a2e443c34dfe9bc476d01796af`.
- On the same nine pre-existing controls with geometry fixed, the new source
  takes 64.33 ms median versus 64.11 ms on merged main (+0.35%) and 2,325,920
  versus 2,324,032 compiled temporary bytes (+0.08%). Requesting all three
  additional geometry derivatives takes 72.98 ms median and 3,814,952 temporary
  bytes. This reports the real marginal derivative cost rather than comparing
  unequal control vectors.
- The pinned external smoke caught an integration defect absent from the unit
  path: a traced scalar spacing made the B2 compiled-kernel cache key
  unhashable. The two-line `eb6994a` repair canonicalizes concrete scalar JAX
  arrays at the cache boundary, and the existing cache regression now exercises
  that exact key class. The rerun passed contract, artifact, execution,
  observation, and comparison checks with zero failed checks and pressure
  L-infinity/RMS discrepancies `0.0109172`/`0.00451798`. The report SHA-256 is
  `be9e6c9e52f249c6b267f9535bc3a44ae4c7f227a4a677b8ff7670bfc73edf61`;
  `acceptance_pass=false` remains correct because the two-update harness role
  cannot establish production B2 validation.
- The exact repaired source passed 509 tests in 140.4 seconds on Python
  3.11.14/JAX 0.9.2 with 95.13% branch coverage. Coverage and JUnit SHA-256
  digests are `dae36aec8b3b3c6fa69e4713e479566cf47e3fca1adfd4e6452236579a887484`
  and `6da0569277fa04f8d3f796eca0ae2be64c4630c8088c45197f8bdb658fb76742`.
  Complete Python 3.10.21/JAX 0.6.2 and Python 3.13.9/JAX 0.11.1 runs passed
  all 509 tests in 111.6 and 104.9 seconds; their JUnit digests are
  `9f90fb7c7db35efcb9160bcbdec88e46e1881017376276a7fa3d0593a2af2cda`
  and `f8eb54a6929578f0e44ea3536b9594c68c5308d8b76c0eb8ebfe455aabb0b2fb`.
  Compatibility matrices run concurrently on separate CI runners but
  sequentially on one local host: three simultaneous JAX compilation matrices
  oversubscribed this machine and pushed the otherwise 31--39 second design
  example past its 60-second per-test limit.
- Ruff check/format, architecture and prose/import gates, Sphinx HTML and
  external links with warnings as errors, isolated build, Twine, distribution
  inspection, and a fresh non-editable-wheel 12-control JIT gradient smoke pass.
  The 150,610-byte wheel and 142,675-byte sdist SHA-256 digests are
  `d84b9f7648a019c3821b7e91946bbd9383c807a0486a413aa9f510925813d06b`
  and `1092cb562bb3515a901d9cb868e27e6c3fdc64e5fe696d28e1a381e022834cba`.
  Compactness remains 16 modules, 6,009 maintained-core lines, 15,476 package
  lines, 13 test files, 12,046 test lines, 28 exports, seven examples, and
  1,772,930 tracked checkout bytes.
- Next action: open and merge the geometry-control PR, then move the existing
  straight/mapped-pipe production operators behind a fixed-work differentiable
  core with independent primal, finite-difference, JVP/VJP, runtime, and memory
  gates. GPU and production B1/B2 claims remain explicitly open.

### 2026-08-28 — differentiable straight-pipe core and fringing trim

- Candidate commits `40185cf`, `4c4da00`, and `3def842` put the generic
  straight-pipe solve and `evolve_extruded_fields` on one cylindrical
  recurrence. It accepts pressure forcing, three station-wise magnetic-field
  scales, fluid/wall conductivity, and axial/radial fixed-topology geometry.
  The public surface remains `lmx.fringing`; `_fringing_pipe.py` owns the
  cylindrical metrics and kernels rather than forming a second user API.
- Removed the test-only pipe face-flux and fixed-flow projection proxies and
  their duplicate tests. They had no production caller and repeated behavior
  already gated through the retained solve. Conducting-annulus, conservative
  current, matrix-free potential, pressure projection, fixed-flow, variable
  field, bent-pipe, B1, and production/field-core parity tests remain. The
  tranche changes 751 lines and removes 797, so the new derivative capability
  is a net source reduction.
- On a three-station, three-fluid-ring/eight-sector pipe with one conducting
  wall ring, all 11 final production fields agree with the differentiable core
  to `1.79e-18` maximum absolute error. The eight-control gradient has
  `2.74609e-7` relative L2 error against centered differences; JVP and reverse
  contraction differ by `2.53e-18`, and the wall-conductivity derivative is
  nonzero. Exact checkpointing uses 751,296 compiled temporary bytes versus
  914,624 for the full tape, a 17.86% reduction. Warm CPU value-and-gradient
  median is 2.05 ms on Python 3.11/JAX 0.9.2.
- The exact source passed 508 tests in 166.7 seconds on macOS 14.4.1 arm64,
  Python 3.11.14, JAX/JAXLIB 0.9.2, SOLVAX 0.18.0, and the CPU backend, with
  95.32% combined branch coverage. Coverage and JUnit SHA-256 digests are
  `8777530303e95ccfe697e0818cde3e5a87f79af853b3401694535a6b8678c30e`
  and `eba96eb7ecacf7e265c8331294e86f9ba2399d31b2714ff41a93a8be5ca9e63b`.
  A complete Python 3.13.9/JAX 0.11.1 run passed all 508 tests in 117.8 seconds;
  its JUnit digest is
  `31c9cfcdb8bdcdbde0b91ce499bc74caa5fef76b989a5f37b7b13a553be3ac87`.
- The minimum Python 3.10.21/JAX 0.6.2/SOLVAX 0.18.0 Docker environment passed
  all retained tests in the support, fringing, and physics ownership shards.
  The constrained local Docker VM required one worker to avoid worker OOM, so
  no misleading combined wall-time claim is made; the shipped test inventory,
  including all eight FreeMHD snapshot gates, passed.
- Ruff check/format, architecture/prose/import checks, Sphinx HTML and external
  links with warnings as errors, isolated build, Twine, distribution
  inspection, CLI, and a fresh non-editable-wheel eight-control JIT gradient
  smoke passed. The 150,564-byte wheel and 142,664-byte sdist SHA-256 digests
  are `8556c5138dc28d7976bf9dabc500d81c153828673f3815c71b61ce21983855e5`
  and `dd4152ed7415cd15bd6cfce224f14f29eca60f4dd97bd35670100830fcc9f4db`.
- Compactness is 16 modules, 6,009 maintained-core lines, 15,435 package lines,
  13 test files, 12,007 test lines, 28 root exports, seven examples, and
  1,782,519 tracked checkout bytes. The package and test ceilings tightened to
  15,440 and 12,010 lines, respectively; merged main used 15,476 and 12,046.
- A clean pinned FreeMHD Docker execution passed contract, artifact, execution,
  observation, comparison, and schema checks with pressure L-infinity/RMS
  discrepancies `0.0109172`/`0.00451798`. Report and external-record SHA-256
  digests are `c2d07ff8df95da5f32b11c5d2fb8afc404e674555049701e78bf10855dd97d4f`
  and `69450b3ad3b5b9fb145d978206d57dc76deeddc2d737ea874162e14da4f8cc50`.
  `acceptance_pass=false` remains correct because the two-update smoke cannot
  establish production-resolution B2 validation.
- Next action: merge this locally evidenced tranche, then differentiate the
  retained bent-pipe coordinate map, establish matched production B1/B2
  refinement evidence, and obtain one-/two-A4000 primal, gradient, memory, and
  strong-scaling measurements. No GPU or production B1/B2 claim is made yet.

### 2026-08-28 — remove the label-only curved-pipe proxy

- LMX PR 17 merged the differentiable straight-pipe core at
  `82da45862ea8f7b07cbe11f54ed5501be4378e41`. The follow-on candidate
  `88582c3` removes the separate bent-pipe builder, display-only mesh generator,
  pseudo-Dean validator, two schema parameters, documentation claim, and their
  duplicate tests. It retains the production straight-pipe/annulus/B1 path,
  rectangular/layered 3-D fringing, magnetic-obstacle application, and pinned
  FreeMHD boundary.
- The audit found no curvilinear physics to protect: curved Cartesian point
  coordinates were not consumed by momentum, pressure projection, electric
  closure, current, or Lorentz operators. The solve used straight cylindrical
  metrics, while the validator labeled generic MHD transverse velocity as Dean
  curvature observables. Removing the name is more accurate than exposing a
  differentiable gradient of the wrong equations. A future curved-pipe lane
  must begin with coherent curvilinear operators and an independent target.
- The change adds 29 lines and removes 359. Package source falls from 15,435 to
  15,215 lines, maintained core from 6,009 to 5,962, tests from 12,007 to
  11,898 lines, and the wheel from 150,564 to 148,562 bytes. Package and test
  ceilings tighten to 15,220 and 11,900 so the proxy cannot return unnoticed.
  The remaining inventory is 16 modules, 13 test files, 28 root exports, seven
  examples, a 1,761-line largest module, and 1,771,727 tracked checkout bytes.
- The exact source passed 505 tests in 160.65 seconds on macOS 14.4.1 arm64,
  Python 3.11.14, JAX/JAXLIB 0.9.2, SOLVAX 0.18.0, and CPU, with 95.25% combined
  branch coverage. Coverage and JUnit SHA-256 digests are
  `8406d85140d5ed7597df8f95c21e0f74bb175dacd8c7f6c431e384c641a2073e`
  and `b5e1a623e44de64a47b9ef3ad6f739894f4f62ecb5a3187ec0007ebf2ea1d83b`.
  Complete Python 3.10.21/JAX 0.6.2 and Python 3.13.9/JAX 0.11.1 runs passed all
  505 tests in 145.4 and 113.26 seconds; their JUnit digests are
  `a548d51640abf92215e0b4cac89f45c86e1f61260738abb0d663f0942b299fd4`
  and `3562d83a58c8e583092353437105550c64d5552090a20cd5703b085107a2dc52`.
- Ruff check/format, architecture/prose/import checks, Sphinx HTML and external
  links with warnings as errors, isolated build, Twine, distribution
  inspection, CLI, and a fresh non-editable-wheel straight-pipe solve passed.
  The 148,562-byte wheel and 140,496-byte sdist SHA-256 digests are
  `e4cb66e459a8de24e7e183751660c8cbe1e13e0ab8a9d66b37f2527ac31e7425`
  and `85f6e1f22a195b7f04ca8f020187730775390580d9631efb50f1e2ff44964ad3`.
- The clean pinned FreeMHD Docker rerun passed contract, artifact, execution,
  observation, comparison, and schema checks with zero failed checks and
  pressure L-infinity/RMS discrepancies `0.0109172`/`0.00451798`. Report and
  external-record SHA-256 digests are
  `c75c883a6a81d34a331c033912c206bd0dcd97275092191caa09b2f5e7a0dd65`
  and `4d0a923491fef48e7839b114d43cd9439a1fc6416e691d3e68e3f040ad39e3c8`.
  `acceptance_pass=false` remains correct for the two-update smoke role.
- The office A4000 host again timed out before connection, so no GPU or scaling
  claim is made. Next action: merge this correction, then focus the 3-D work on
  specialized ALEX derivative gates and production-resolution matched B1/B2;
  retry one-/two-device primal, gradient, memory, and scaling evidence when the
  registered GPU host is reachable.

### 2026-08-28 — reopen private-fringing trim and differentiate the B1 projection

- Commit `62db12c7a82762351acdaa10fc1b477da29daf48` keeps
  `lmx.fringing` as the only public 3-D surface. The underscore modules are not
  compatibility copies: they own shared structured-grid, rectangular-grid,
  cylindrical-grid, and orchestration mathematics. Their decomposition did
  not finish the compactness work, so Phase 4's function-level audit is open
  again with a <1,500-line immediate private-module gate, a 1,200-line target,
  and an explicit ban on increasing the private-file count to satisfy it.
- The physical fixed-flow pipe Schur projection now composes SOLVAX
  `linear_solve` with its existing GMRES primal and a GMRES transpose solve.
  The adjoint uses the actual linear transpose of the retained-modal
  preconditioner; using the primal preconditioner produced a measurable
  finite-tolerance derivative error and was rejected. Krylov `while_loop`
  iterations are no longer differentiated or stored.
- The manufactured pipe gate closes divergence and flow below `1e-9`, checks
  the implicit VJP against centered finite differences within `1e-3`, and
  checks JVP/VJP contraction within `1e-7`. Its compiled scalar
  value-and-gradient uses 66,968 temporary bytes on CPU. The grid was reduced
  from `5 x 3 x 8` to the smallest meaningful `3 x 2 x 4` system; under branch
  coverage the gate takes 74 seconds while overlapping other shards. The
  complete covered suite passes all 505 tests in 184.1 seconds at 95.26%
  combined branch coverage. Coverage and JUnit SHA-256 digests are
  `0a7da854194aab8e011277d5cc32813f05aa0adee90fe9cf87dadac99e3ec60c`
  and `fca92d258f6e977c52028d12fd54aff89b01693933466818593ad794b53d2341`.
- Complete concurrent compatibility matrices pass all 505 tests on Python
  3.10.21/JAX 0.6.2/SOLVAX 0.18.0 in 281.2 seconds and Python
  3.13.9/JAX 0.11.1/SOLVAX 0.18.0 in 251.8 seconds. Their JUnit SHA-256
  digests are
  `e82b0ce60ca1802f4d25ef00a85d42b79a8920afa814d03aa2c822437383ac6a`
  and `6fd001c507d7bdf401cc2cdf5a1021658c4786ff634700a1110b95ac2aa2ecad`.
- Ruff, formatting, architecture, all curated workflows, Sphinx HTML and
  external links with warnings as errors, isolated build, Twine, distribution
  inspection, and a fresh non-editable-wheel TOML solve pass. The wheel and
  sdist are 148,619 and 140,570 bytes with SHA-256 digests
  `e820f021b4c3de30085d570eb1f496659cfae1e5e2e5a02f4120ee4c0934fb73`
  and `66aef271fc27f4849a10f2a0d7300393799d25db2f573b1addea95febf97562f`.
  Package source is 15,210 lines across 16 modules; all five fringing files
  total 7,063 lines, tests remain exactly at the 11,900-line ceiling, and the
  root API remains 28 names.
- The clean pinned FreeMHD Docker smoke passes contract, artifact, execution,
  observation, comparison, and schema gates with zero failed checks and
  pressure L-infinity/RMS discrepancies `0.0109172`/`0.00451798`. Report and
  record SHA-256 digests are
  `147d89ef93458e3090a2da23e7315ae0419c902f063f077ba9fe05831f253612`
  and `52bd15916390473b2ffdc12e0edecef59199cd46ff0ed84fa5d2bd6fbe5f7c3c`.
  The two-update smoke remains non-accepting by role. The office A4000 host
  timed out again, so GPU and strong-scaling claims remain open.
- Next action: split the 1,662-line orchestration function by physical phase
  without adding files, remove remaining duplicate pipe/duct recurrences, and
  use the accepted implicit projection inside a traced production B1
  recurrence. Production B1/B2 derivative, refinement, GPU, and scaling claims
  remain open until their independent gates pass.

### 2026-08-29 — split fringing orchestration by physical geometry

- Commit `81f592b87dd221a0559428235939ecbbb3963425` replaces the
  1,662-line mixed-geometry solve with a 50-line validated dispatcher, a
  738-line mapped-pipe owner, and an 821-line rectangular/layered-duct owner.
  It adds no module and changes no public API or numerical expression.
- Pipe and duct paths now share one initial-state constructor and one
  field/material scaling contract. The initial-state helper validates restart
  shape once and reuses one immutable zero array for transverse velocity,
  pressure, and potential instead of allocating four equal arrays. Both pipe
  loops also share one restart-checkpoint constructor rather than maintaining
  two 27-line state serializers.
- `_fringing_solver.py` falls from 1,734 to 1,714 lines and total fringing
  source from 7,063 to 7,043 lines. Package source falls from 15,210 to 15,190
  lines across the same 16 modules; tests remain 11,900 lines, the root API
  remains 28 names, and the wheel falls from 148,619 to 148,197 bytes. The
  <1,500-line private-module gate remains open for `_fringing_duct.py`,
  `_fringing_pipe.py`, and `_fringing_solver.py`.
- The exact source passes all 505 tests in 185.3 seconds on macOS 14.4.1 arm64,
  Python 3.11.14, JAX/JAXLIB 0.9.2, SOLVAX 0.18.0, and CPU, with 95.31%
  combined branch coverage. Coverage and JUnit SHA-256 digests are
  `6acc46c3c3df876d384aae245c716a57aedc7573ff0fb73d60dacd11ac7bc0da`
  and `b86391684ba437c44aed96644b31a7bf3a86f73050d4982501037d15de463e2f`.
  Complete Python 3.10.21/JAX 0.6.2 and Python 3.13.9/JAX 0.11.1 matrices pass
  all 505 tests in 289.4 and 258.4 seconds under concurrent load; their JUnit
  digests are
  `b30493fa4ff12a7904b88c1d7b8ec919a6e29ff2022305b80b6633db6ccf831a`
  and `e4d574ba62952a8f55caa0fb7a97edb8f2591763033a815d4f8088ca93a96a5f`.
- Ruff, formatting, architecture, all curated workflows, Sphinx HTML and
  external links with warnings as errors, isolated build, Twine, distribution
  inspection, and a fresh non-editable-wheel TOML solve pass. The 148,197-byte
  wheel and 140,121-byte sdist SHA-256 digests are
  `1c3d6d847fdbd42ec213d8bf81d36e2033eb08fd846072dc1dd36f6c37cc2cb3`
  and `be2f8e6f4f4b158af085c769aaec2f0340e63836acefcd552399e3bad0441eae`.
- The clean pinned FreeMHD Docker smoke passes contract, artifact, execution,
  observation, comparison, and schema checks with zero failures and unchanged
  pressure L-infinity/RMS discrepancies `0.0109172`/`0.00451798`. Report and
  record SHA-256 digests are
  `ae3cfb49a08c2aa39f9e52705e194afcda300d98b6ee03669de4cf8deb597981`
  and `1f1e70ae83505d2b1daf03d4ae56989520f4594d7fdb513477c4d96fe758e758`.
  The two-update role remains non-accepting, and GPU/scaling evidence remains
  open while the office host is unreachable.
- Next action: reduce the three remaining >1,500-line private owners by
  consolidating duct/pipe finalization and B2 setup, without redistributing
  bulk code merely to satisfy a file metric. Then use the accepted implicit
  pipe projection in the traced production B1 recurrence and run its primal,
  derivative, memory, runtime, and refinement gates.

### 2026-08-29 — trim private B2 ownership without trimming 3-D capability

- Commit `e29ae4ec2a45ad78335b4cbf3828a5b5edca8446` keeps
  `lmx.fringing` as the single public 3-D API while making the underscore
  boundary deliberate. Generic cross-section construction and analytic,
  constant, tabulated, and volume-field sampling now belong to `lmx.mesh`;
  user-facing station-history assembly belongs to `lmx.fringing`; and
  `_fringing_duct.py` retains only rectangular/layered B2 discretization,
  linear-solve, coupling, JIT, and sharding implementation details. No 3-D,
  fringing-field, tabulated-field, restart, or validation capability was
  removed, and no new module or root export was added.
- B2 now carries the packed three-face mass flux as its single runtime state
  instead of retaining packed and component copies. Momentum and coupling JIT
  setup share one explicit data-driven compiler, using callable identities and
  paired input/output sharding contracts instead of two parallel manual
  registries. Electric-current and Lorentz reconstruction share one kernel,
  and metric arguments are passed as one physical tuple. A forced two-CPU-
  device B2 solve completed with the field sharded over both devices, unit mean
  velocity, and maximum charge residual `1.9392842887100414e-10`.
- The change adds 232 lines and removes 382. `_fringing_duct.py` falls from
  1,541 to 1,499 lines, satisfying its immediate private-module gate; all five
  fringing files fall from 7,043 to 6,798 lines. Package source falls from
  15,190 to 15,044 lines across 16 modules, tests fall to 11,896 lines, the
  root API remains 28 names, and the tracked checkout excluding build
  artifacts is 1,774,241 bytes. The 147,465-byte wheel remains far below the
  artifact budget. The packed Git history still exceeds the eventual <10 MB
  clone target, so the authorized final history rewrite remains an open
  release task rather than a claim of this tranche.
- The exact source passes the 505-test parallel CPU gate in 183.8 seconds on
  Python 3.11.14/JAX 0.9.2/SOLVAX 0.18.0 with 95.35% combined branch coverage.
  Coverage and JUnit SHA-256 digests are
  `93edf81e897ed07297406f3a3d9bf38f8735053001735a7d192c8eaf32ab09fe`
  and `e5640a6a0c0524257b1eb36c1576e7ddea9f56a9002d12718b3d6b1f1e00a446`.
  Complete Python 3.10.21/JAX 0.6.2 and Python 3.13.9/JAX 0.11.1 matrices also
  pass all 505 tests; the corrected absolute-path Python 3.10 run completes in
  560.33 seconds. The separate curated first-run workflow passes.
- Ruff check/format, architecture/prose/import budgets, Sphinx HTML and
  external-link builds with warnings as errors, isolated build, Twine,
  distribution inspection, CLI, and a fresh non-editable-wheel import and
  mesh-construction smoke pass. The pinned FreeMHD Docker boundary passes
  contract, artifact, execution, observation, comparison, and schema checks
  with zero failed checks and unchanged pressure L-infinity/RMS discrepancies
  `0.010917245284750786`/`0.004517977100484136`. Report and record SHA-256
  digests are
  `1d3e679d1236d97d3723a6169260fb90af2b2e06cecf6701ef6e583e471c1e0f`
  and `0d57cf5b73dc7d3f543aede75dd129de7643140ab2732e11566d67cd9ab2e760`.
  `acceptance_pass=false` remains correct because the two-update smoke is not
  production-resolution validation.
- Next action: trim the remaining 1,539-line pipe and 1,705-line orchestration
  owners through genuine recurrence/finalization consolidation, then trace the
  specialized production B1 recurrence through the accepted implicit
  projection and establish primal, derivative, reverse-memory, refinement,
  GPU, and strong-scaling evidence. GPU claims remain open until the office
  host is reachable.

### 2026-08-29 — trace the production B1 map and trim the pipe owner

- Commit `3d8abf1e54b4e6237a3789f38cb995b416e96b9f` exposes the
  specialized ALEX B1 finite-volume recurrence through the existing
  `evolve_extruded_fields` API. Fixed-step design execution and the ordinary
  production loop now call the same momentum, retained-modal fixed-flow
  projection, conservative electric solve, current reconstruction, and
  Lorentz map. ALEX B2 design fields still fail closed until their sharded
  recurrence has an equivalent bounded reverse-memory contract.
- Dynamic forcing, imposed field, conductivity, axial scale, and radial scale
  remain inside the traced operator. Production runs still reuse compiled
  momentum kernels and retained modal factors, while design runs deliberately
  rebuild parameter-dependent factors inside the trace so geometry and
  material gradients are not silently frozen. SOLVAX implicit linear solves
  differentiate the momentum and Schur systems without retaining PCG/GMRES
  iterations. The compiled one-step five-control value-and-gradient uses
  223,272 temporary bytes on CPU; its reverse directional derivative matches a
  centered finite difference within `3e-4`, and every field/material/geometry
  component is finite and nonzero. Fixed-flow forcing is correctly nearly
  inactive for the chosen velocity objective.
- Generic and B1 pipe production paths now share one observables, diagnostics,
  stopping, progress, and acceleration loop. Generic mapped-pipe kernels reuse
  bound gradient, divergence, boundary, Poisson, and conservative-current
  metrics; the steady projection shares face operators and its two flow
  response solves. Pipe and duct vector products share one allocation-free
  tuple cross product. Conservative current diagnostics reuse already computed
  face fluxes.
- The change adds 318 lines and removes 394 while adding the new derivative
  gate and user documentation. `_fringing_pipe.py` falls from 1,539 to 1,499
  lines, `_fringing_duct.py` falls to 1,493, and package source falls from
  15,044 to 14,910 lines across the same 16 modules. All five fringing files
  total 6,664 lines. Tests rise deliberately from 11,896 to 11,949 lines for
  the production B1 gradient/finite-difference/memory contract; no test file or
  root export is added. The wheel falls to 147,385 bytes and the tracked
  checkout excluding build artifacts is 1,776,338 bytes.
- The exact source passes all 506 tests in 182.4 seconds on Python
  3.11.14/JAX 0.9.2/SOLVAX 0.18.0 with 95.36% combined branch coverage.
  Coverage and JUnit SHA-256 digests are
  `f784a37f4107a3009726a3f606ac716dc0cd86fb469e1ba8c0cd4987da200982`
  and `7449c10085d03be0d87e7270466d2ef290f488aa75ae8bcb4a355de77838b70c`.
  The changed B1 derivative and reduced production gates also pass on Python
  3.10.21/JAX 0.6.2 and Python 3.13.9/JAX 0.11.1; the immediately preceding
  merged tranche supplies complete 505-test matrices on both endpoints.
- Ruff check/format, architecture/prose/import budgets, Sphinx HTML and
  external-link builds with warnings as errors, isolated build, Twine,
  distribution inspection, and a fresh non-editable-wheel B1 production-map
  solve pass. The pinned FreeMHD Docker boundary passes contract, artifact,
  execution, observation, comparison, and schema checks with zero failures and
  unchanged pressure L-infinity/RMS discrepancies
  `0.010917245284750786`/`0.004517977100484136`. Report and record SHA-256
  digests are
  `25494d6cfcd47f1dc75345d51644a6b6fcb4a00544e5a808eeb42063929acec9`
  and `42ecb83a59f4e87873038b6393729e30d155512f1e75f363f65efce2b41eb57d`.
  `acceptance_pass=false` remains correct for the two-update smoke role.
- Next action: reduce the remaining 1,610-line orchestration owner by sharing
  geometry-independent bundle finalization and recurrence setup without
  moving numerical bulk between private files. Then establish matched B1/B2
  mesh/tolerance refinement and production-resolution acceptance, and collect
  one-/two-GPU primal, reverse, memory, and strong-scaling evidence when the
  office host is reachable.

### 2026-08-29 — finish the private fringing ownership trim

- Commit `7338d1478fd50101cc3e689e8f5610823925903e` completes the immediate
  private-module size gate without removing a solver, field representation,
  restart state, diagnostic, or public API. `lmx.fringing` remains the sole
  user-facing 3-D surface. The private orchestrator now imports the common,
  duct, and pipe owners as modules rather than recreating a broad private
  re-export surface; geometry-independent initial-field and design-property
  setup belongs to the common owner; and `ExtrudedFieldBundle.from_groups`
  centralizes result assembly in the data model. Checkpoint calls use the same
  coordinate/field grouping, and B2 accelerator metadata is assembled once
  for both progress checkpoints and the terminal bundle.
- The change adds 222 lines and removes 287. `_fringing_solver.py` falls from
  1,610 to 1,498 lines, while `_fringing_pipe.py` remains at 1,499 and
  `_fringing_duct.py` at 1,493. All five fringing files fall from 6,664 to
  6,576 lines; package source falls from 14,910 to 14,845 lines across the same
  16 modules. Tests remain 11,949 lines in 13 files, the root API remains 28
  names, the tracked checkout excluding build artifacts is 1,778,832 bytes,
  and the wheel falls from 147,385 to 147,210 bytes. The largest package module
  is now 1,499 lines. The no-new-file constraint was preserved.
- The exact source passes all 506 tests in 183.4 seconds on Python
  3.11.14/JAX 0.9.2/SOLVAX 0.18.0 with 95.36% combined line/branch coverage.
  Coverage and JUnit SHA-256 digests are
  `5de24af55499b0709026f80c3626c030f2146035cef8dbd0adc0e100281b6a03`
  and `b9424b265342bb94829bf6ec3eeba2f4543e240efadce721927edf984869f1eb`.
  The changed B1, B2 exact-restart, strided-history, and complete IO gates also
  pass on Python 3.10.21/JAX 0.6.2 and Python 3.13.9/JAX 0.11.1.
- Ruff check/format, byte compilation, architecture/prose/import budgets,
  Sphinx HTML and external-link builds with warnings as errors, isolated
  build, Twine, distribution inspection, CLI, and a fresh non-editable-wheel
  primal/compiled-gradient smoke pass. The pinned FreeMHD Docker boundary
  passes contract, artifact, execution, observation, comparison, and schema
  checks with zero failures and unchanged pressure L-infinity/RMS
  discrepancies `0.010917245284750786`/`0.004517977100484136`. Report and
  record SHA-256 digests are
  `5e493830ead2aee1756b61a3eb01ba68dbc3464acf349b9eac1835df44873308`
  and `ab2087c2ba5c0e4186a40227371fb67ce40d655c8aee99f7de843c021211381d`.
  `acceptance_pass=false` remains correct for this two-update smoke role.
- Next action: keep the retained 3-D/fringing capability and establish matched
  B1/B2 mesh/tolerance refinement plus production-resolution acceptance.
  Collect one-/two-GPU primal, reverse, peak-memory, speedup, and strong-scaling
  evidence when the office host is reachable. Continue function-level
  simplification only where it removes duplicate work or improves a measured
  numerical/performance boundary; do not trim capabilities to chase line
  counts.

### 2026-08-29 — quantify Benchmark B refinement evidence

- Commit `38271879a04cbf940adfd55795028b8780187a8c` upgrades the Benchmark B
  campaign schema from workload labels to exact numerical evidence. Every run
  records the allocated and physical mesh shapes, cell counts, fluid volume,
  three-dimensional characteristic spacing, and SHA-256 identities of all
  face-coordinate arrays. Acceptance reconstructs the frozen mesh from the
  current case contract and rejects mismatched coordinates, including the
  device-rounded axial extent.
- The three-grid gate now reports unequal-ratio observed order, fine-grid GCI
  with the Celik et al. `1.25` safety factor, asymptotic-range ratio, and a
  minimum `1.3` refinement ratio. Numerical convergence and external FreeMHD
  validation are separate fail-closed surfaces with explicit
  `mesh_incomplete`, `numerical_rejected`, `external_validation_open`, and
  `accepted` states. Literature error, mesh independence, exact external
  evidence, and final acceptance remain distinct claims.
- B2 now persists its existing production momentum-defect history and enforces
  a conservative `1e-3` terminal balance gate. A real coarse B2 CPU run on an
  Apple M3 Max compiled its first update in about 32 seconds and advanced at
  about 6 seconds per update, but its depth-two Anderson recurrence plateaued:
  at update 48 the maximum residual was `0.006117260424545412` and the
  momentum defect was `3.026987806821503`. The run was stopped at a valid
  atomic checkpoint rather than relabeled as converged. Tolerances were not
  relaxed; production B2 acceptance remains open pending a measured
  acceleration/convergence correction.
- All 504 tests pass in 209.69 seconds on Python 3.11.14/JAX 0.9.2/SOLVAX
  0.18.0 with 95.36% combined line/branch coverage. Coverage and JUnit
  SHA-256 digests are
  `a96a185ee821f7cf5f3a8023f1853db3d68e817c2365fc2344e58cd453bc9efa`
  and `049a7544a4d524e3b0c92148bf89bf183805078e7ada585e05977216033b0847`.
  The changed campaign/spec gates also pass on Python 3.10/JAX 0.6.2 and
  Python 3.13/JAX 0.11.1. Ruff, formatting, byte compilation, architecture,
  import, Sphinx HTML/link, isolated build, Twine, wheel, and sdist gates pass;
  the wheel is 147,212 bytes.
- The pinned FreeMHD/OpenFOAM image was rebuilt from commit
  `14b54a3e8e1a05b6ee4c98331995abaaae96e7a5`. The exact two-update B2 smoke
  passes contract, artifact, execution, observation, comparison, and schema
  checks with zero failures and unchanged pressure L-infinity/RMS differences
  `0.010917245284750786`/`0.004517977100484136`. Report and record SHA-256
  digests are
  `7ba23d05a8d6f8337ef386aacf6b180f4daf7f1951b312d7392ab1888f1c8e77`
  and `1c7e5303c684fefb4fc5351cad4c8185adc47573ad055b184d063460ad1ae4a7`.
  `acceptance_pass=false` remains correct for the smoke-only role.

- Next action: audit every function in the 5,833 private fringing lines for
  production reachability, duplicated pipe/duct recurrence, redundant arrays
  and host work, and reusable SOLVAX ownership. Preserve the validated 3-D,
  B1, and B2 capabilities; delete or fuse only with physics, derivative,
  runtime, and memory evidence. Then diagnose the B2 production plateau and
  execute the frozen coarse/medium/fine campaign without weakening its gates.

### 2026-08-29 — remove private fringing testbeds and stale restart lanes

- Commit `093d5960979ac64bb23a40605cb01e505f16f49d` begins the function-level
  audit behind the public `lmx.fringing` facade. The underscore modules remain
  private implementation owners, not additional APIs, and their existence is
  not considered completion of the slimming goal. This tranche removes
  parameter switches that no production caller used: neutral duct inlets,
  injected axial neighbours/tractions/fluxes, externally supplied diffusion
  coefficients, alternate Laplacian boundary modes, pipe block-Jacobi axial
  decoupling, and a component-inverse test hook. Production boundary
  conditions, conservative duct/pipe operators, 3-D field evolution, B1/B2,
  restarts, fixed flow, differentiation, and FreeMHD comparison remain.
- Restart restoration now accepts only the current typed bundle contract. B2
  checkpoints must contain complete momentum-defect, pressure-linear, CFL,
  stopping, and accelerator state instead of fabricating placeholder histories
  for obsolete bundles. Generic duct restarts continue to omit B2-only history
  truthfully. Public result validation likewise checks the complete current
  numerical bundle rather than silently filtering missing attributes.
- The change adds 90 lines and removes 335. Package source falls by 77 lines
  from 14,845 to 14,768; the five fringing files fall from 6,576 to 6,499.
  `_fringing_common.py`, `_fringing_duct.py`, `_fringing_pipe.py`, and
  `_fringing_solver.py` are 1,314, 1,461, 1,483, and 1,498 lines respectively;
  the 743-line `fringing.py` remains the sole public surface. Test source falls
  by 168 lines to 11,781 after two duplicated ad hoc bundle mocks are replaced
  by one complete typed fixture. No module, test file, script, example, or
  public export is added. The isolated wheel is 146,402 bytes and the tracked
  checkout is 1,782,353 bytes, both well below the 10 MB target.
- All 500 tests pass in 176.82 seconds on Python 3.11.14/JAX 0.9.2/SOLVAX
  0.18.0 with 95.41% combined line/branch coverage. Coverage and JUnit SHA-256
  digests are
  `505e1005c46af8a3cdd696705ef9e98dd05a59a685f7b7197c27d4ff57c92d19`
  and `ca8d07c7ba940912c6bbaeba8a38b28426f1ec8bdaf70d427d841e35e6ce649e`.
  Changed duct, pipe, wrapper, primal, JVP, and VJP gates also pass on Python
  3.10.21/JAX 0.6.2 and Python 3.13.9/JAX 0.11.1.
- Ruff check/format, byte compilation, architecture/import budgets, Sphinx HTML
  and external-link builds with warnings as errors, isolated build, Twine,
  wheel, and sdist inspection pass. The pinned FreeMHD Docker smoke passes
  contract, artifact, execution, observation, comparison, and schema checks
  with zero failed checks and pressure L-infinity/RMS differences
  `0.010917245284750786`/`0.004517977100484136`. Report and record SHA-256
  digests are
  `49fcb194e86d189b07fc8bc825f7e6bd23c4d6ab0d370805c049fa906bc31ba1`
  and `6cad316c71ef82c4ea79793dc1f2bd645aed8a6d27be7c9601914ebf132f7ce7`.
  `acceptance_pass=false` remains correct for the smoke-only role.
- PR #25 created all seven hosted CI, documentation, and external-validation
  jobs, but every job ended before its first step with the same GitHub Actions
  account billing/spending-limit annotation. This is recorded as unavailable
  hosted execution, not numerical evidence; the equivalent local gates above
  are the merge evidence under the approved temporary policy.
- Next action: audit the remaining large recurrence/finalization functions for
  measurable duplicate work and allocations, especially the shared pipe/duct
  orchestration in `_fringing_solver.py`. Fuse private owners only when the
  result has clearer ownership or less runtime/memory; moving thousands of
  implementation lines into `fringing.py` would enlarge the public module
  without simplifying the code. In parallel, diagnose the B2 production
  plateau and run the frozen mesh/tolerance campaign without weakening its
  numerical or external-validation gates.

### 2026-08-29 — correct B2 fixed-point scaling and isolate the plateau

- Commit `a02cb8212b84885263acc59ff83cccb4a6145bfb` separates the B2
  fixed-point normalization from its velocity safety guard. The Anderson state
  now uses the prescribed mean flow velocity, $Q/A=1$ for the frozen B2 case,
  instead of the guard $2\sqrt{Ha}\approx108$. The induced-potential block
  retains its physical scale. This changes only the residual geometry used by
  acceleration; the finite-volume map, boundary conditions, tolerances,
  conservation gates, compact flux state, and restart contract are unchanged.
- Controlled runs used the exact 101x65x65 coarse B2 problem. The current
  depth-two method reaches a maximum update `0.004153443021944203` at step 32,
  already below the previous guard-scaled result `0.006117260424545412` at
  step 48. Measured depth-two coefficients remained finite and moderately
  extrapolative rather than singular. A four-map experiment with physical
  scaling reached `0.0022967856283384602` at step 32, but would multiply the
  fine-grid field/residual/flux restart memory and still left a `3.01565`
  momentum defect, so that extra production state was rejected. Dynamic
  Aitken with `[0.05, 1]` reached only `0.0114979`; fixed relaxation 2 diverged
  in transverse velocity to `2.9832` with momentum defect `307.0`.
- The remaining defect is not an Anderson-conditioning artifact or a
  wall-only diagnostic. At early coarse steps the largest axial defect occurs
  at the inlet, but an interior maximum remains `2.21818` by step 3; transverse
  maxima also occur downstream next to both wall families. Physical scaling
  improves the state update while the terminal momentum defect remains
  `3.0097461040979154`. Production B2 convergence and acceptance therefore
  stay open; no tolerance, balance limit, or validation label was relaxed.
- All 500 tests pass in 177.89 seconds on Python 3.11.14/JAX 0.9.2/SOLVAX
  0.18.0 with 95.41% combined line/branch coverage. Coverage and JUnit SHA-256
  digests are
  `ef2f84612e3af09b571f94b2654901427dde9d4a1e70ebc8c6f44bb6017e7088`
  and `98b0b871c7db8369e3b01817c43e3afd3be121301bac96d72cc0f9c987cf0b6b`.
  The B2 restart/closure and scale-contract gates also pass on Python
  3.10.21/JAX 0.6.2 and Python 3.13.9/JAX 0.11.1.
- Ruff check/format, byte compilation, architecture/import budgets, Sphinx HTML
  and external links with warnings as errors, isolated build, Twine, wheel,
  and sdist inspection pass. The wheel is 146,441 bytes. The pinned FreeMHD
  smoke passes contract, artifact, execution, observation, comparison, and
  schema checks with zero failures and pressure L-infinity/RMS differences
  `0.010917245284747775`/`0.004517977100483119`. Report and record SHA-256
  digests are
  `cb7d2f707ebd933ec3f682003e99f1b613a4b930839520e9e05367ff7e5e7582`
  and `b1252d3770cacbb59d1d7ea7f747bd3418938df7ff23831247a8fe9f3cc13528`.
  `acceptance_pass=false` remains correct for the smoke-only role.
- PR #26 created all seven hosted CI, documentation, and external-validation
  jobs; every job ended before its first step with GitHub's account
  payment/spending-limit annotation. The hosted service supplied no numerical
  evidence, so the complete recorded local matrix above remains the merge
  authority under the temporary policy.
- The current primary-source refresh adds DESC as the natural differentiable
  equilibrium/geometry provider alongside ParaStell's CAD/neutronics role.
  NekRS remains the scalable high-order primal comparison, FreeMHD the
  independently executed liquid-MHD oracle, and JAX rematerialization/sharding
  the implementation substrate rather than a novelty claim. Walker--Ni
  Anderson evidence reinforces that scaling, history, regularization, and
  safeguarding are numerical-method choices that require measured gates.
- Next action: compare the implicit momentum equation, pressure projection,
  conservative face-flux reconstruction, explicit deviatoric stress, and
  post-map defect term by term on the accepted state. Repair the first
  operator inconsistency that fails a discrete identity, then rerun the coarse
  convergence study before considering deeper history or a coupled steady
  root solve. Keep specialized B2 differentiation unavailable until the primal
  balance and convergence contracts pass.

### 2026-08-29 — localize the B2 momentum defect and reject an inconsistent preconditioner

- A term-by-term audit of the accepted 101x65x65 coarse B2 recurrence finds no
  hidden Anderson singularity or stationwise flow correction large enough to
  explain the plateau. At step 3 the largest normalized axial residual is
  `3.2047` at the inlet and an interior maximum of `2.204895` remains near the
  downstream wall. The interior contributions are approximately zero
  convection, `2.20591` negative-diffusion action, zero explicit stress, zero
  Lorentz force, and `-0.001013` pressure action. The fixed-flow cell
  correction is confined to the inlet at this precision; it is about
  `1e-14` at the penultimate station.
- Doubling the physical pseudo-step from the magnetic stability value worsens
  the step-32 update to `0.0051469` and the momentum defect to `5.9913`, so the
  magnetic bound is active. A controlled reaction-stabilized trial used the
  same B1-style fixed-point identity, a `0.01` pseudo-step, and no changed
  physics gate. It reached a step-32 update of `0.0390474103` and a momentum
  defect of `46.4474689`; every experimental source edit was then removed.
  The trial changed the momentum inverse without changing the pressure Schur
  mobility, so it is evidence against that inconsistent split, not against a
  compatible steady or pseudo-transient method.
- The remaining numerical task is a compatible coupled root solve: momentum,
  pressure constraint, conservative mass flux, potential, and Lorentz force
  must share one residual and one deliberately matched preconditioner. LMX
  will not add another general nonlinear solver. SOLVAX 0.18 already owns
  matrix-free pseudo-transient Newton--Krylov with a positive mass metric,
  consistent shifted preconditioner, globalization, finite work limits, and
  implicit-root differentiation. The next gate is a reduced B2 proof that
  composes this API, compares the converged discrete residual and memory with
  the retained recurrence, and fails closed before any production switch.
- No source, test, or runtime lane was retained from this investigation. The
  existing underscore modules remain private geometry/operator owners behind
  `lmx.fringing`; numerical globalization and Krylov policy remain SOLVAX
  responsibilities.

### 2026-08-29 — expose the exact B2 production map to SOLVAX composition

- The retained B2 predictor, mixed pressure projection, conservative compact
  flux reconstruction, electric solve, Lorentz reconstruction, and momentum
  defect now compose in one pure `lmx.b2.map` named call inside the existing
  orchestration. The Python recurrence consumes that same map, so its state,
  restart, diagnostics, sharding, and stopping contracts remain unchanged.
  This adds no public API, module, file, solver, parameter, or experimental
  runtime lane.
- A temporary, subsequently deleted proof passed the complete B2 state and
  compact mass flux to `solvax.pseudo_transient_continuation` as the exact map
  residual. On the retained 5x5x5 reduced case, six SOLVAX nonlinear steps
  reduced the algebraic residual norm from `35108.8026` to `0.0312687`, with
  five accepted steps and 38 Krylov iterations. The resulting maximum map
  update was `0.00621063` and the physical momentum defect was `0.0364371`.
  The existing six-step recurrence reached `0.0168165` and `0.0472907` on the
  same case. This is proof of numerical viability only; it is not production
  runtime, memory, convergence, or derivative evidence.
- The production-switch gate remains fail-closed. First supply the coupled
  pressure/momentum shifted preconditioner, scaled physical residual norm,
  exact restart state for the adaptive pseudo-time method, implicit JVP/VJP
  residual tests, and peak-memory measurement. Then require a material
  improvement in update and momentum balance on the 101x65x65 coarse case
  without exceeding the current runtime budget. Until those gates pass, the
  public B2 path keeps its current recurrence and specialized B2 derivatives
  remain unavailable.
- The extraction adds 72 source lines and removes 83. `_fringing_solver.py`
  falls from 1,498 to 1,490 lines, total package source falls from 14,768 to
  14,760 lines, and the repository remains at 16 modules and 28 root exports.
  The exact candidate passes all 500 tests in 179.80 seconds with 95.44%
  combined line/branch coverage. Coverage and JUnit SHA-256 digests are
  `199399ddc13978a2a4d73760618b0712fbaa97c4d1beb19be3620700a61fad1f`
  and `04977053027389e90328faeae5702f2ac5d19df8050c9fac3dfedb5ebd570173`.
  Ruff check/format, byte compilation, architecture/import budgets, Sphinx
  HTML and link checking with warnings as errors, isolated build, Twine, wheel,
  and sdist inspection also pass. The wheel is 146,565 bytes.
- LMX PR #28 merged the pure production map at
  `9497846882742712c7353649159eb184b040bfec`. All seven hosted jobs had zero
  executed steps; completed jobs reported GitHub's account payment/spending
  annotation, while the fringing entry remained queued inside an already
  concluded failed workflow. This is unavailable hosted execution, not test
  evidence; the complete local matrix above is the merge authority.
- Overall roadmap audit: Phases 0--3, 6, and 7 are complete. Phase 4 remains
  open for function-level private-fringing reduction, specialized B1/B2
  field derivatives, smooth geometry controls, production adjoint memory,
  and CPU/GPU derivative parity. Phase 5 remains open only for scheduled or
  release-grade production FreeMHD refinement on suitable hardware. Phase 8
  remains open for the final all-capability validation, coherent public
  release, and clean-install/release verification. The active numerical
  critical path is the compatible B2 coupled preconditioner and coarse-mesh
  convergence gate; GPU speedup and two-device scaling remain evidence gaps,
  not claims.

### 2026-08-29 — reject an unpreconditioned B2 map root at production scale

- The pure B2 map enabled a controlled SOLVAX pseudo-transient root audit
  without retaining a new runtime lane. Pressure was removed from the root
  state because it is a nested Lagrange-multiplier solve whose input is only a
  warm start. The tested state was exactly `(u, v, w, phi, rho_phi_plus,
  rho_phi_inlet)`. Residual orientation matters: `state - map(state)` is
  compatible with SOLVAX's positive pseudo-time mass shift; the algebraically
  equivalent reverse sign made the shifted Jacobian unstable.
- On the 5x5x5 reduced problem, six correctly oriented dimensionless steps
  reached algebraic norm `0.00549103`, maximum velocity update `0.00120980`,
  and momentum defect `0.0372833` with 33 Krylov iterations. A Newton-like
  two-step trial reduced the norm rapidly but accepted only one step and left
  momentum defect `0.267823`; converting velocity coordinates by the existing
  physical factor `N*dt=0.064` improved the two-step reduced defect to
  `0.0959909`. These are useful conditioning results, not production evidence.
- The exact 101x65x65 step-32 state was regenerated in 180.62 seconds with
  update `0.004153443021910452` and momentum defect
  `3.0097461041226583`. Its untracked 61 MiB typed restart has SHA-256
  `b1f6070150bd66db462175ee65c037cb7a557c30cb7e3fd783aff70a0bebbb8b`.
  Two loose shifted steps cost 34.94 seconds and changed the defect only to
  `3.0077351`; an unshifted Newton trial rejected every step after 16 Krylov
  directions and 57.34 seconds; an intermediate shift cost 57.63 seconds and
  left `3.0090324`; physical velocity scaling cost 57.62 seconds and left
  `3.0088956`. None is a material improvement per second.
- Decision: do not integrate unpreconditioned fixed-point rooting, do not add
  a `pseudo_transient` user option, and do not retain proof code. The next B2
  numerical implementation must expose the compatible pressure response
  `D A^-1 G` (or an equivalently verified block preconditioner) using the
  frozen conservative momentum operator, with fixed-flow constraints and
  compact flux reconstruction in the same discrete identity. Reuse the B1
  steady Schur methodology where its geometry-independent contract can be
  extracted; keep pipe modal factors geometry-local. Acceptance requires a
  dense reduced identity, coarse momentum reduction, lower total runtime than
  continued recurrence, bounded implicit transpose memory, and exact restart.
- A separate face-identity check tested distance-weighted transverse velocity
  interpolation in the pressure projection, matching the compact-flux
  initializer on the nonuniform wall mesh. Projection conservation and reduced
  restart tests passed, but the reduced six-step result worsened from update /
  momentum `0.0168165 / 0.0472907` to `0.0311214 / 0.0526960`. One continuation
  from the exact coarse restart cost 30.41 seconds, worsened the update to
  `0.00446561`, and changed momentum only to `3.00643`. The source change was
  removed. This rules out interpolation weighting alone; the accepted next
  operator remains the full compatible momentum response, not a face-flux
  cosmetic correction.

### 2026-08-29 — verify and reject a nested transient B2 Schur solve

- A disposable implementation exposed the exact homogeneous response of the
  frozen conservative momentum matrix and used it in a mixed-boundary
  compatible projection. The response matrix and right-hand side matched a
  dense oracle, its linear solve matched the dense solution, and force-scale
  JVP/VJP identities passed. With a diagonal response, the compatible
  projection reproduced the retained pressure and all three face-flux families
  within `2e-8`; divergence, flow, and linear residuals were below `1e-8`.
  These checks confirm the sign, boundary, pressure-gauge, and transpose
  contract, but do not establish an acceptable production algorithm.
- The exact response was then composed with the retained B2 frozen momentum
  operator, conservative compact flux, fixed-flow correction, and restart
  state. The reduced closure and restart gates passed. Six 5x5x5 updates took
  45.095 seconds and ended at update `0.01683414` and momentum defect
  `0.04731858`, essentially the retained trajectory
  `0.0168165 / 0.0472907` at materially greater cost. The pressure Schur itself
  converged in three iterations with residuals near `1e-15`; the cost is the
  repeated full transient momentum inverse, not pressure Krylov stagnation.
- One continuation step from the checksummed 101x65x65 step-32 restart did not
  complete after 300 seconds and was terminated at 319 seconds. The retained
  production recurrence costs about 5--6 seconds per continuation step, so the
  candidate fails the runtime gate by more than an order of magnitude before a
  terminal physics comparison is possible. No source, test, option, or private
  helper from the experiment is retained.
- Decision D-038 therefore rejects a nested full transient solve. The next
  implementation must avoid applying the mass-dominated momentum GMRES inside
  each pressure Krylov action. Derive a reusable separable, line, modal, or
  coarse approximation to `D A^-1 G`, or a verified block factorization, from
  the frozen momentum response. Gate it first on the dense identity, reduced
  physical defect, warm runtime, implicit-transpose work/storage, and exact
  restart; only then rerun the production coarse state. The B2 public path and
  all numerical tolerances remain unchanged.

### 2026-08-29 — stop oversolving primal B2 electric closure

- Repeated the private-fringing function/call-graph audit across source, tests,
  examples, validation, and scripts. The common, duct, pipe, and orchestration
  owners are each reached by protected production, restart, differentiation,
  B1/B2, or validation paths; no entire underscore file is dead. Their private
  names remain implementation ownership, not user API. Moving them into
  `fringing.py` would enlarge the public module without reducing code or work.
- A disposable shifted line-block preconditioner used SOLVAX additive
  tridiagonal solves for the three frozen momentum directions and no nested
  momentum GMRES. After six strict 5x5x5 updates, three accepted root steps
  took the total to 41.94 seconds and reached update `0.00188368` and momentum
  defect `0.0397514`. Nine retained updates took 33.90 seconds and reached the
  lower defect `0.0376508`. The algebraic update reduction therefore did not
  improve physical balance per second; all proof code was removed. A viable
  B2 block response must include the pressure/electric coupling or otherwise
  beat continued recurrence on the physical defect, not only the map norm.
- Profiling the unchanged reduced map attributed 13.72 of 32.99 seconds to
  electric closure. The specialized B2 path inherited a roundoff tolerance
  required by the generic implicit VJP even though B2 differentiation remains
  deliberately unavailable. Commit `5db42c449418c35ca31964622e0d61b24c569640`
  applies Decision D-039: generic traced fields retain the roundoff solve, while
  primal-only B2 uses a directly certified `1e-10` electric tolerance. No API,
  option, file, iteration limit, physical tolerance, or validation claim changes.
- On the same six-update reduced case, electric iterations fell from
  `56/46/46` to `41/31/31`. Two-process warm runtime fell from 18.619 to
  12.352 seconds, a 33.7% improvement. Terminal update and momentum defect
  changed by less than `7e-13`; maximum charge residual changed from
  `1.15e-10` to `4.31e-9`, still more than five orders of magnitude below the
  `1e-3` B2 balance gate. A looser `1e-8` trial gave no additional speed and
  raised reduced charge residual to `9.22e-7`, so it was rejected.
- Five warm continuations from the checksummed 101x65x65 step-32 restart had
  medians 4.682 seconds for the candidate and 4.740 seconds for the retained
  roundoff solve. Both produced update `0.004166759820736776`, momentum defect
  `3.0079227388830585`, and charge residual `0.0004964470863342285` to the
  reported precision. The production gain is a modest 1.24%; the accepted
  change is primarily a deliberate primal/derivative contract and a substantial
  reduced-development/strict-tolerance speedup, not a claim that the B2
  convergence plateau is solved.
- All 500 portable tests pass in 178.68 seconds with 95.44% combined
  line/branch coverage on Python 3.11.14, JAX/JAXLIB 0.9.2, and SOLVAX 0.18.0.
  Coverage and JUnit SHA-256 digests are
  `88f81cc014485d99183237a6e959cdc29d8dc2b94243cbf16ceddbd5df3d6afc`
  and `22773647dba1e941b078907a5bc8f30296d34d5183f61ec6cd104b37ac6b3f6c`.
  Ruff check/format, architecture/import budgets, curated examples, Sphinx HTML
  and external links with warnings as errors, build, Twine, distribution
  inspection, and isolated-wheel primal/gradient smoke pass. The package has
  16 modules, 6,084 maintained-core lines, 14,764 package lines, and 28 root
  exports; the wheel is 146,636 bytes and the sdist is 138,493 bytes.
- Rebuilt the pinned FreeMHD image from commit `14b54a3` as
  `sha256:535e995d557d2a73f5ab997380cb47ee3b044af8d2871bdadd570cff4cf175a8`.
  The committed LMX smoke passes contract, artifacts, execution, observation,
  comparison, and schema with zero failed checks. Pressure L-infinity/RMS
  differences are `0.010917245284640538`/`0.004517977100439702`; report and
  record SHA-256 digests are
  `7a8ea024591a0afc0e361ea9d7b356f4a4ae0fe996f4b357173bfa6250b73bc5`
  and `f9f7c160f364a57f91dc6f77ce6ce9e1def829f5511b6ecf8950d5aea144d526`.
  `acceptance_pass=false` remains correct for the smoke-only role.
- Next action: continue the compatible steady B2 block derivation with coupled
  pressure/electric response and continue function-level trimming only where
  reachability, runtime, memory, and physics evidence justify a deletion or
  fusion. The production B2 defect, specialized adjoint, and GPU scaling gates
  remain open.

### 2026-08-29 — consolidate fringing orchestration and ownership

- Removed `_fringing_solver.py`, whose three solve-dispatch functions had one
  consumer and no independent API or test contract. `lmx.fringing` now owns
  orchestration directly. Fringing problem builders moved to `lmx.cases` and
  validation functions moved to `lmx.validation`; `lmx.fringing` deliberately
  re-exports every prior user-facing name, so application code does not change.
  The architecture guide describes concepts instead of private filenames.
- The package falls from 16 to 15 Python modules. `fringing.py` remains below
  the enforced 1,800-line ceiling, and the only private fringing modules are
  the 1,314-line shared mapped owner, 1,461-line rectangular-duct owner, and
  1,483-line cylindrical-pipe owner. They remain because each contains live
  production, derivative, restart, B1/B2, or validation kernels; combining
  them would create an undifferentiated numerical mega-module rather than
  remove algorithms. Package source is 14,773 lines, and the architecture
  gate reports 15 modules, 28 root exports, and seven curated examples.
- All 500 portable tests pass in 173.04 seconds with 95.44% combined
  line/branch coverage on Python 3.11.14, JAX/JAXLIB 0.9.2, and SOLVAX 0.18.0.
  Coverage and JUnit SHA-256 digests are
  `ddb11e2225dfca515dc434a32db74689d20453fc60d1bee746c5a609770977d1`
  and `a0b182d01e12beba50241af3bf6d810e93b834d9f31724b5165b560d2707d135`.
  Focused source and built-wheel API smokes confirm all 13 named
  `lmx.fringing` callables remain importable and the removed module is absent
  from the distribution.
- Ruff, formatting, byte compilation, architecture/import budgets, curated
  workflows, Sphinx HTML and external links with warnings as errors, isolated
  build, Twine, and wheel/sdist content audits pass. The final build produced a
  146,089-byte wheel and a 138,760-byte sdist; their SHA-256 digests are
  `17fdf4447c0e512b790ecfd6c918997f2f1effc1d370635ebe33712e00060ff4`
  and `7f16d01169e39164e4d8a07eb766abfe575e92182bb3f52c8d3cea9e7b714662`.
- PR #34 created the metadata, documentation, link, compatibility, and pinned
  FreeMHD checks, but GitHub rejected every sampled job before its first step
  with the account payment/spending-limit annotation. This is unavailable
  hosted execution, not numerical evidence; the complete local gates above are
  the merge authority under the approved temporary policy.
- Next action: continue function-level common/duct/pipe trimming only where it
  removes actual work or duplication with measured physics, derivative,
  runtime, and memory equivalence. Resume the coupled B2 pressure/electric
  response and the queued B1 gradient-gate runtime reduction; do not remove
  the protected 3-D, B1, B2, restart, or autodiff capabilities to satisfy a
  filename metric.

### 2026-08-29 — remove redundant derivative-evidence compilations

- Decision D-041 removes four duplicate JAX primal compilations from the
  rectangular, layered, straight-pipe, and ALEX B1 production-gradient gates.
  Each already-compiled value-and-gradient executable is shape stable and
  returns the exact production primal value, so it now supplies the shifted
  samples for the same centered finite differences. The coordinate or
  directional difference formula, epsilon, tolerance, nonzero-sensitivity,
  JVP/VJP, batching, production-parity, and bounded-memory assertions are
  unchanged. This is evidence scheduling only; no LMX or SOLVAX source,
  algorithm, tolerance, public API, or physics claim changes.
- The steady-pipe projection gate now compiles its full primal projection once
  and reuses it for the conservation result and derivative samples. The
  unjitted callable remains the fail-closed oracle for unsupported retained
  modal options. Its isolated call time falls from 48.80 to 31.88 seconds, a
  34.7% reduction. The ALEX B1 gate falls from about 86.5 to 47.78 seconds in
  isolation and from 140.02 to 61.27 seconds under the six-worker full gate,
  a 56.2% reduction in the contended configuration.
- The complete six-worker portable gate falls from 174.4 to 156.3 seconds
  end-to-end (10.4%) and passes all 500 tests in 154.98 pytest seconds with
  95.43% combined line/branch coverage. Coverage and JUnit SHA-256 digests are
  `e319fa9c645f293d46c9c20fbb0c132dd284ef7fe740e95a4bec7488926558c6`
  and `9fff827a957deea19a3cdb2c755aa2d7d1a27bf895314b51ff2dad1916ef2e96`.
  The exact 51-test fringing shard completes in 87.3 seconds with coverage
  instrumentation, far below its 540-second CI budget.
- Ruff, formatting, byte compilation, architecture/import budgets, and the
  complete curated workflow pass locally. Package source and distribution
  contents are unchanged because the tranche modifies only an existing test
  file and this plan.
- PR #35 created all compatibility, coverage, metadata, and documentation
  jobs, but GitHub rejected each runnable job before its first step with the
  account payment/spending-limit annotation. The combined report correctly
  skipped without shard evidence. This is unavailable hosted execution; the
  complete local gates above remain the approved merge authority.
- Next action: return to the compatible coupled B2 pressure/electric response,
  then measure the production B1/B2 primal and gradient throughput on the
  office CPU/A4000 hardware. Keep compile-evidence reuse separate from warm
  production-performance claims.

### 2026-08-29 — match the B2 pressure and momentum diagonals

- Decision D-042 replaces the B2 pressure mobility $\Delta t/\rho$ with the
  diagonal inverse already assembled for its frozen implicit momentum
  predictor. The local fluid-only response is returned with the momentum
  result, so no second operator assembly, Krylov solve, padded wall field,
  public API, option, or solver implementation is added. Pressure assembly and
  face correction now share the same distance-weighted harmonic coefficient
  on nonuniform transverse cells.
- A variable-coefficient, nonuniform-grid regression closes the reconstructed
  compact face flux to `1e-10`; the reduced B2 conservation and exact-restart
  contract also passes unchanged. On the checksummed 101x65x65 step-32 state,
  one history-free update reduces the momentum defect from about `3.01` to
  `1.5208315033251738`, with divergence `2.974512041120647e-7`. Five raw or
  freshly restarted depth-two updates remain stable but plateau near
  `1.4926`, so this is a compatibility repair and material defect reduction,
  not a production-convergence claim. A separate reaction-only trial failed
  to transfer its reduced-case gain to the production mesh and was removed.
- The candidate warm production step is `4.406` seconds versus the retained
  `4.682`-second median. All 500 portable tests pass in 158.55 seconds (159.9
  seconds end to end) with 95.41% combined line/branch coverage. Coverage and
  JUnit SHA-256 digests are
  `703c64ab850fcc1d3712c921d6ad3395c0a588af945a19899431a74b8b14a903`
  and `441a4ca06ce909b11e617fa53a039fe93764ab103ef401194951d8dbec80506f`.
- Ruff, formatting, byte compilation, architecture/import budgets, all seven
  curated workflows, Sphinx HTML and external links with warnings as errors,
  isolated build, Twine, distribution inspection, and clean-wheel primal and
  gradient smoke pass. The wheel is 146,214 bytes and the sdist is 138,873
  bytes; their SHA-256 digests are
  `f13ab36cfe080099c958835e29829eb80326d5e3e34c1cbd5fc20f6c103db3fb`
  and `5e34c8f69b6aceee468d9ef046ba04432369088a54c0fb94faa9e07f10f47cde`.
- Committed the immutable source candidate as `dc76ee5` and repeated the
  pinned FreeMHD B2 Docker comparison against source `14b54a3` and image
  `sha256:535e995d557d2a73f5ab997380cb47ee3b044af8d2871bdadd570cff4cf175a8`.
  Contract, artifacts, execution, observation, comparison, and schema all pass
  with zero failed checks. Pressure L-infinity/RMS differences are
  `0.010917245364311495`/`0.004517977131740069`; report and record SHA-256
  digests are
  `91b0bee0c2c84e63edfdb47e9cf7227e5b7157d9ef3c3decd6381f5e9b856bf3`
  and `ec0e02af83265aad369bcd644fabf45cc0f1fae37d5bb83c123b1e9032b75ab6`.
  `acceptance_pass=false` remains correct for the smoke-only role.
- Next action: merge the compatible diagonal tranche and continue from the
  remaining coupled pressure/electric plateau. Specialized B2 differentiation
  remains unavailable until primal convergence and derivative gates both pass;
  then measure the production B1/B2 primal and gradient throughput on the
  office CPU/A4000 hardware.

### 2026-08-29 — couple the B2 pressure state to momentum

- Decision D-043 repairs the remaining segregated pressure inconsistency. Each
  B2 momentum predictor now includes the conservative force $-\nabla p^n$;
  the matched diagonal pressure equation solves $p'$, velocity receives the
  full conservative correction, and the stored pressure advances as
  $p^{n+1}=p^n+0.4p'$. Two pressure--momentum correctors execute before one
  electric/Lorentz closure. They reuse the existing jitted momentum and
  projection kernels and add no public API, solver, restart field, or file.
  The documentation gives the equations and anchors them to the original
  SIMPLE and SIMPLEC literature.
- The relaxation/corrector campaign used the checksummed 101x65x65 step-32
  restart. Unit pressure relaxation diverged after two improving updates; 0.5
  turned upward at update eight, while 0.4 remained monotone through 24
  one-corrector updates. At 0.4, eight two-corrector updates reach momentum
  defect `0.596436346649753` versus `1.4926603110597991` on merged main. Cold
  process time is `82.11` versus `62.81` seconds: 60.0% lower defect for 30.7%
  more wall time. A third corrector reaches only `0.5677557320997869` in
  `88.65` seconds, so its marginal work is rejected.
- A fresh production depth-two continuation is stable: at update 12 it reaches
  defect `0.4977992378596463`, update `0.0038591445280506953`, and divergence
  `9.862796835900562e-8` in `106.56` seconds. Compiled peak footprint rises
  from `2,313,384,384` to `2,733,733,056` bytes (18.2%); the macOS maximum-RSS
  counter rises 5.1%. Persisted and accelerator state are unchanged. The
  tranche therefore wins physical defect per second but does not yet establish
  terminal production convergence or specialized B2 differentiation.
- Manufactured pressure-force sign/boundary checks, variable-coefficient
  projection, reduced B2 conservation/convergence, and bit-exact restart pass.
  All 500 portable tests pass in 163.90 seconds (165.4 seconds end to end) with
  95.41% combined line/branch coverage. Coverage and JUnit SHA-256 digests are
  `2eb2a047d63cef5926f3abfcd9a1e8968ae9ade84ceda61c2ce1d6d1b28d49d1`
  and `8f25ba0ed50367bff8e4b2b42be4949464e540f3057a7532e3561b0644fa2813`.
- Ruff, formatting, byte compilation, architecture/import budgets, all seven
  curated workflows, Sphinx HTML and external links with warnings as errors,
  isolated build, Twine, distribution inspection, and clean-wheel primal and
  gradient smoke pass. Package source is 14,825 lines; the wheel is 146,493
  bytes and the sdist is 139,150 bytes, with SHA-256 digests
  `9daba7673c02335830b9c53e94985a312ec1d1d0cf1af40de83682d7e5e3a5e6`
  and `681440adadd4b94b731dcf59506390d6a15f9b246de76478a5ed3ce9a56cc1da`.
- Committed the immutable source candidate as `ae02c61` and repeated the pinned
  FreeMHD B2 Docker comparison against source `14b54a3` and image
  `sha256:535e995d557d2a73f5ab997380cb47ee3b044af8d2871bdadd570cff4cf175a8`.
  Contract, artifacts, execution, observation, comparison, and schema all pass
  with zero failed checks. Pressure L-infinity/RMS differences improve from
  `0.010917245364311495`/`0.004517977131740069` to
  `0.006837811881424563`/`0.003064699119643086`. Report and record SHA-256
  digests are
  `8dcd03f39046a427c237c11bec911117e76e17af4aa223d40be27d925de9de6f`
  and `27861f048e0028e696c651f20b5e7fda7a919b256d0390db7468b0dc74101908`.
  `acceptance_pass=false` remains correct for the smoke-only role.
- Next action: merge the pressure-coupled tranche, then continue the remaining
  long-horizon convergence and specialized derivative gates before office
  GPU/scaling measurements.

### 2026-08-29 — establish long-horizon B2 descent and isolate acceleration noise

- A fresh depth-two production continuation from the checksummed step-32 field
  completed 64 pressure-coupled updates in `459.899` seconds. Momentum defect
  falls monotonically from `0.5946010` at update 8 through `0.3073790` at 32
  to `0.231654340909` at 64. Divergence remains at order `1e-7` and charge
  below `3e-4`. The typed 101x65x65 restart is
  `/tmp/lmx-b2-pressure-step64.npz`, SHA-256
  `f0b3b5850d05bce0a0bacd37e83d22b6a56924d4016907f5ebc3c49993a00edc`.
  This establishes sustained physical descent, not terminal convergence.
- The raw state update bottoms at `0.00173388` near update 48 and rises to
  `0.00229761` at 64 while the momentum defect keeps falling. Eight exact
  continuations isolate the cause: unaccelerated relaxation 1 ends at update /
  defect `0.00176628 / 0.22039979`; stored Anderson ends at
  `0.00442285 / 0.22026971`. Anderson therefore provides negligible late
  defect gain while introducing update oscillation.
- A disposable “fall back when the raw update grows” safeguard worsened the
  alternating spikes to `0.0161143` and was removed. Fixed relaxation 2 is now
  stable under the corrected map: from the original state, update 12 reaches
  `0.00416786 / 0.50100227` versus Anderson
  `0.00385914 / 0.49779924`; from the step-64 state it ends at
  `0.00157275 / 0.22062741`. It is smoother but does not improve physical
  descent, so Decision D-044 retains the declared depth-two method.
- No source, API, option, test lane, or state was retained from this audit.
  Production B2 balance and specialized derivatives remain open. Next action:
  measure the merged primal on the office CPU and A4000 devices, including
  one-/two-GPU sharding, while deriving a restart-exact coupled acceleration
  that controls the physical defect rather than only the fixed-point update.

### 2026-08-29 — budget B2 pressure work and reject local acceleration heuristics

- Tested SOLVAX residual-history condition filtering, deterministic periodic
  restart, and a norm trust region on the exact step-32 and step-64 production
  states. A condition limit of 10 reduces the late maximum update from
  `0.00442285` to `0.00205132` but worsens the early 12-step momentum defect
  from `0.49779924` to `0.50222302`. Period-four restart ends late at
  `0.00163566 / 0.22043781`, no better in physical defect than the raw map.
  Trust radii two and four also lose early physical descent. All three
  restart-exact trials were removed; Decision D-044 remains in force.
- An exact frozen-convection contribution to the local momentum diagonal
  changes the late defect only from `0.22026971` to `0.22025648` and worsens
  the early 12-step result to `0.50055652`. Enabling the existing axial line
  factor raises pressure PCG work from about 157 to 185 iterations; enabling
  it for electric closure raises work from about 53 to 64 iterations. These
  source trials were also removed rather than retained as nominal safeguards.
- Decision D-045 changes only the primal-only B2 pressure linear tolerance
  from `1e-12` to `1e-10`. Eight exact late updates reduce mean pressure PCG
  work from about 157 to 118 iterations while reproducing the state updates
  and terminal momentum defect `0.22026971` at reported precision. Maximum
  projected divergence is `6.7852e-5`, below five percent of the independent
  `1e-3` balance gate; finer cell volumes still tighten the effective absolute
  target automatically. The frozen production and matched FreeMHD contracts,
  independent observers, documentation, and a direct regression now declare
  the same control.
- The complete six-worker local gate passes all 500 tests in 171.42 seconds
  (173.1 seconds end to end) with 95.41% combined line/branch coverage.
  Coverage and JUnit SHA-256 digests are
  `c5785bb4afcdbaa0b49ee47feeca070f7712a56829cbc309e6e0aa3f842528c4`
  and `68dda1635ece7dfd05aa2077f074325f587a6fef3453fba99aaa5cb1f2564174`.
  Ruff, formatting, byte compilation, architecture/import budgets, all seven
  curated workflows, Sphinx HTML and external links with warnings as errors,
  isolated build, Twine, and distribution inspection pass. The wheel is
  146,492 bytes and the sdist is 139,152 bytes.
- Committed the immutable source candidate as `d367a8b` and repeated the
  pinned Docker comparison against FreeMHD `14b54a3` and image
  `sha256:535e995d557d2a73f5ab997380cb47ee3b044af8d2871bdadd570cff4cf175a8`.
  Contract, artifacts, execution, independent observation, comparison, and
  schema pass with zero failed checks. Pressure L-infinity/RMS differences are
  `0.006837808945711196`/`0.003064697969803467`, indistinguishable at the
  validation scale from the prior `0.006837811881424563`/`0.003064699119643086`.
  Report and record SHA-256 digests are
  `585b66d071fc7b7ac2880961dac3256736464429dd30940258258b68a7877236`
  and `01f6e391220d0ae2414db49f7ca527883f49e738224bc89ea2cc40992b0e6724`.
  `acceptance_pass=false` remains correct for the smoke-only role.
- Next action: merge this pressure-work tranche, then continue the compatible
  coupled B2 block response rather than adding another fixed-point safeguard.
  Specialized B2 differentiation remains unavailable until terminal primal
  convergence is established.

### 2026-08-29 — audit private fringing ownership and retain a fixed-point-neutral B2 response

- Audited all 95 top-level functions across `fringing.py` and its three private
  implementation owners. Static source-and-test reachability found no
  unreferenced function. The 40-function common owner contains six
  cross-geometry operations, 17 orchestration/runtime consumers, eight
  duct-only consumers, four pipe-only consumers, and five internal building
  blocks. This proves reachability, not necessity: Phase 4 remains open while
  generic and ALEX recurrences are converged and reusable algebra is upstreamed.
- Retained the four-file boundary for this tranche. Flattening 5,997 lines into
  the public module would create a mega-module without reducing source, while
  duplicating the six shared kernels into both geometries would add code and
  inconsistent derivative surfaces. Private files remain implementation
  details, not additional APIs, and must still earn their size through the
  Phase 4 physics, derivative, and performance gates.
- Removed a duplicate 26-field result schema, current-schema fallback branches,
  and a single-use sharding constructor. The source changes in this tranche are
  40 net lines smaller before documentation and tests; package source is 14,785
  lines, below the committed 14,825-line baseline.
- Decision D-046 adds $R_B=\sigma|\boldsymbol B|^2$ as a fixed-point-neutral
  pseudo-mass to the B2 momentum predictor and pressure mobility. Dense matrix,
  right-hand-side, solution, fixed-point, and automatic-differentiation checks
  pass. An exact eight-update continuation reaches momentum defects
  `0.20083865, 0.19979831, 0.19896500, 0.19803814, 0.19711240,
  0.19598774, 0.19445617, 0.19313568` in `131.49` seconds, with maximum
  divergence `2.1699e-5` and charge residual `2.0855e-4`. Raw fixed-point
  updates remain oscillatory, so terminal convergence and specialized B2
  differentiation remain open.
- A separate SOLVAX audit found that a finite-zero-cotangent norm optimization
  could map NaN GMRES residual norms to zero. The fail-closed correction and
  jitted array/PyTree regression tests pass the complete local SOLVAX gate:
  753 tests, 97.50% combined coverage, types, docs, build, and distribution
  checks. PR 93 is merged; no package publication was performed.
- The complete six-worker LMX gate passes all 500 tests in 155.25 seconds
  (156.5 seconds end to end) with 95.44% combined line/branch coverage.
  Coverage and JUnit SHA-256 digests are
  `c7c47c879928b071a63a99862a8fb2d5d5aef25d864b9f97fe93ce78a7922b0b`
  and `56c922f91dc6a2fd2dbac6ea013c85c72d21f6f73b4e36b5095abf32e2ecccf8`.
  Ruff, formatting, byte compilation, architecture/import budgets, all seven
  curated workflows, Sphinx HTML and external links with warnings as errors,
  isolated build, Twine, distribution inspection, and clean-wheel primal and
  gradient smoke pass. The wheel is 146,323 bytes and the sdist is 139,033
  bytes, with SHA-256 digests
  `d747d9673e48a4b2604c46c8746394829ea38b379ec84f693b483109bcbfab00`
  and `0d5392d1f2eaff171d8ebf55a85acb84faed7776c3aa70717d1426232d1e8af7`.
- Committed the immutable source candidate as `a2d4b1e` and repeated the pinned
  Docker comparison against FreeMHD `14b54a3` and image
  `sha256:535e995d557d2a73f5ab997380cb47ee3b044af8d2871bdadd570cff4cf175a8`.
  Contract, artifacts, execution, independent observation, comparison, and
  schema pass with zero failed checks. Pressure L-infinity/RMS differences are
  `0.006838504934991994`/`0.0030649109539924494`, within the smoke contract and
  effectively unchanged at the validation scale. Report and record SHA-256
  digests are
  `4e7402e4531782130d19ffbb6961ae0078cdd4d788a06ae95e07432e78bc1aae`
  and `ffd1073e6e1963d5ce61de8b1bcd5b33d597aa29a365932f260531f85a469073`.
  `acceptance_pass=false` remains correct for the smoke-only role.
- Next action: merge this qualified tranche, then resume operator-level
  convergence of the generic and ALEX B1/B2 paths rather than deleting private
  files by name. Terminal B2 convergence and specialized derivatives remain
  open evidence gates.

### 2026-08-29 — shard the generic differentiable 3-D field core

- Rejected refreshing electric closure and Lorentz force inside both B2
  pressure--momentum correctors. From the checksummed step-64 state, eight
  updates took `162.18` seconds and ended at momentum defect `0.19317161`,
  versus `131.49` seconds and `0.19313568` for the retained pseudo-mass map.
  The trial was 23.3% slower and slightly worse in physical defect, so no
  source, option, or test lane remains from it.
- Decision D-047 removes the host NumPy staging boundary from generic 3-D
  sharding and exposes `num_devices` on `evolve_extruded_fields`. Rectangular,
  layered, and straight-pipe fields now retain axial `NamedSharding` through
  the same finite recurrence used by reverse mode. The small axial and
  transverse coarse preconditioners are replicated explicitly and their
  corrections repartitioned; full 3-D state is not replicated. ALEX B1 remains
  fail-closed for multiple devices until its specialized cylindrical map has
  independent evidence.
- A forced two-CPU-device process verifies all three generic geometry families
  against one-device fields and confirms two local axial shards. The largest
  field discrepancy is `3.79e-11` for the layered coarse solve; rectangular
  and pipe differences are at roundoff. A jitted value-and-gradient comparison
  agrees within `2.65e-23`, and the public non-differentiable solve also retains
  two shards. The gate costs 23.8 seconds while running inside the ordinary
  parallel suite, not in a slower optional lane.
- The complete six-worker local gate passes all 501 tests in 163.49 seconds
  (165.2 seconds end to end) with 95.41% combined line/branch coverage.
  Coverage and JUnit SHA-256 digests are
  `a033ac0d21fc01c2047fdac0729836b466bd03835d0997961c902c7f7d1a7b70`
  and `f1f4858944092240a868c6999866fc4a4c6eb52ee0a700bedf248aa03782cc7d`.
  Ruff, formatting, byte compilation, architecture/import budgets, Sphinx HTML
  and external links with warnings as errors, isolated build, Twine,
  distribution inspection, and clean-wheel primal/gradient smoke pass.
- The retained fringing implementation is 6,005 lines across the public owner
  and three private mathematical owners, below the prior 6,019-line planning
  baseline. Total package source is 14,807 audit lines. The wheel is 146,566
  bytes and the sdist is 139,353 bytes, with SHA-256 digests
  `e1eb75bc59c4f397ffbcc758b2a34ac60117d60978c9e9240e88bfccc2dc7c15`
  and `0ca7ff9a99571aca6e90c7cc45bfb2a26f933935d6fe8bb02d6ea51f0a8c3482`.
- Committed the immutable source candidate as `ca686a9` and repeated the pinned
  Docker comparison against FreeMHD `14b54a3` and image
  `sha256:535e995d557d2a73f5ab997380cb47ee3b044af8d2871bdadd570cff4cf175a8`.
  Contract, artifacts, execution, independent observation, comparison, and
  schema pass with zero failed checks. Pressure L-infinity/RMS differences are
  `0.006838504934991994`/`0.0030649109539924494`, bit-for-bit unchanged from
  the prior source candidate because this tranche does not alter the B2 map.
  Report and record SHA-256 digests are
  `032e7c6f970f3bf72a24bc6f6988841c8a18bba6bb3a75b16e1f7fd9ccab54db`
  and `59b1f25fa269a0ccac82c0befaee02dcccd96a545a81b26635a6909d52be1437`.
  `acceptance_pass=false` remains correct for the smoke-only role. Next action:
  merge this generic sharding tranche, then resume terminal ALEX convergence,
  specialized derivative gates, and real one-/two-GPU timing when the office
  host is reachable.

### 2026-08-29 — reject generic pressure unification and expose B2 accepted-state inconsistency

- Trialled replacing the generic duct's compact checkpointed Jacobi projection
  with its SOLVAX-PCG finite-volume electric operator, then removed the trial.
  Cold rectangular/layered production solves became 22--24% slower and serial
  derivative gates 15--35% slower for only `0.014%`/`2.7%` lower divergence.
  For the eight-step design gradient, compile time rose from `2.173` to `3.038`
  seconds, median warm execution from `1.097` to `1.289` milliseconds, and
  temporary memory from `121,200` to `136,224` bytes while the value and
  gradient were unchanged. The existing physical stencil already delegates
  bounded recurrence storage to SOLVAX and remains the deliberate faster path.
- Continued the exact pseudo-mass B2 state from step 64 to step 112. Stored
  depth-two Anderson reaches defect `0.19313568` at step 80 and `0.17726693`
  at step 96, but its update grows from order `2e-3` to `2.57e-2`. From step
  96 it briefly plateaus, then jumps to defect `0.15131795` at step 112 with
  update `0.04415093` and accepted-state charge `0.00129454`, outside the
  independent `1e-3` gate. The step-112 restart is 63 MiB and has SHA-256
  `52495d2dd664175c882f8b47895c9fe8b4640280f5065df33b87b4e497612efd`.
- Replayed step 80 with fixed relaxation two and with the raw map. Fixed two is
  smooth and reaches defect `0.17872821` at step 96, but extrapolating the
  finite-tolerance electric state yields accepted charge `0.00174633`. One
  post-acceleration electric reclosure restores charge to `9.39e-5` but raises
  16-update runtime from `124.9` to `148.1` seconds. The raw map needs no
  reclosure: it reaches defect `0.17898851` at step 96 and decreases
  monotonically to `0.16771035` at step 112 with update `0.00180815`, charge
  `2.56e-4`, and 128.4-second continuation time. Its stateless restart is 34
  MiB, SHA-256
  `4395e92560c6776a79e750195ec169ce4ee915bffb91bdbd2fb02bbbd3a4eb1d`.
- Canonical 12-step comparisons bound the early tradeoff. Anderson takes 107.9
  seconds and reaches defect/update `0.34404766 / 0.01773752`; raw takes 106.1
  seconds and reaches `0.34818429 / 0.03069781`. Raw therefore wins late
  stability, accepted charge, runtime, and restart memory but not defect at
  every fixed work point. Fixed two also fails accepted charge without the
  extra solve. Decision D-044 remains in force rather than promoting a partial
  win or adding a switching heuristic.
- The discrepancy between mapped diagnostics and the stored accelerated state
  identifies the next operator-level task: acceleration must act on mechanical
  velocity/compact flux state, after which the one production electric solve
  must close the accepted velocity. Charge, Lorentz force, momentum defect,
  update, and restart diagnostics must then be computed from that same stored
  state. This avoids a second electric solve while removing potential from the
  accelerator state and is the next candidate for both convergence and code/
  memory trimming. The office GPU endpoint timed out again, so no GPU claim is
  added.

### 2026-08-29 — close electric current on the accepted B2 state

- Decision D-048 makes the B2 fixed-point accelerator purely mechanical. The
  pressure--momentum map returns corrected velocity, pressure, and compact
  conservative face fluxes; depth-two Anderson or Aitken then accepts velocity
  and flux; one electric solve closes that accepted velocity. Current, Lorentz
  force, charge balance, momentum defect, update history, checkpoints, and the
  returned fields now describe exactly the same state. No second electric
  solve or electric-potential accelerator history is retained.
- Restart accelerator fields fall from four 3-D components to three. Exact
  reduced-case restart equivalence passes, and the production 101x65x65
  step-16 and step-32 restart files are 55 MiB rather than the prior 63 MiB
  Anderson artifact. The step-32 artifact has SHA-256
  `7bd19f6812f414b438589fb93b0d0f2991f4e055ac45d9bc92d86116c4990f22`.
- On the canonical 12-step B2 case, the accepted-state method takes 107.75
  seconds and reaches momentum defect `0.32724041`, accepted update
  `0.06554252`, divergence `7.35e-5`, and charge residual `2.33e-4`. The
  previous mixed-potential state reached defect `0.34404766` in 107.9 seconds,
  so the consistent closure improves the early physical defect by about 4.9%
  at unchanged runtime. A fresh continuation reaches defect `0.26976903` at
  step 16 and then decreases monotonically from `0.25759419` to `0.22754770`
  over steps 17--32 while charge remains `2.45e-4`. The accepted update still
  oscillates, so terminal B2 convergence and specialized B2 derivatives remain
  open; no production claim is promoted.
- The user-facing surface remains only `lmx.fringing`. Its three underscore
  modules are live private mathematical owners for shared mapped operators,
  rectangular/B2 operators, and cylindrical operators—not additional APIs or
  legacy copies. They remain subject to line and ownership budgets: generic
  solver algebra moves to SOLVAX when it can be geometry-independent, and LMX
  retains geometry metrics, boundary equations, MHD coupling, and physical
  acceptance. Deleting the filenames by concatenating their 4,288 lines into
  the 1,716-line public module would not trim code and would create a 6,004-line
  mega-module. The next trim therefore removes repeated recurrences and direct
  private-test coupling first, then fuses a private owner only if its remaining
  mathematical boundary no longer earns a file.
- The complete six-worker local gate passes all 501 tests in 169.47 seconds
  (170.8 seconds end to end) with 95.27% combined line/branch coverage on
  Python 3.11.14, JAX/JAXLIB 0.9.2, and SOLVAX 0.18.0. Coverage and JUnit
  SHA-256 digests are
  `d1aef698fc7029ba16073083277eebb914bd478e6942018668c9d2922f977edb`
  and `5cbe5a7d25302da3853124b320430d2edd5a5ce55bfb8a917735832bef4748cb`.
  Ruff, formatting, byte compilation, architecture/import budgets, all seven
  curated workflows, Sphinx HTML and external links with warnings as errors,
  isolated build, Twine, distribution inspection, and clean-wheel primal and
  gradient smoke pass. The wheel is 146,369 bytes and the sdist is 139,151
  bytes, with SHA-256 digests
  `db04d3d8d659781dc1fc188657dad7ff699ba5234a6a29dadfa351ba46f60a29`
  and `a0ceb9af7c9113235d814d6dfe334a2c32b7bab4ae8da72a87fa21ce2bcdac50`.
- The first pinned external-smoke attempt exposed two stale explicit JIT
  sharding arities after the four-to-three-field reduction. Both input and
  output contracts are corrected, and the ordinary reduced B2 physics/restart
  test now executes with `num_devices=1`, so the sharded B2 path is covered on
  every local gate rather than only by the Docker integration.
- Package source is 14,796 audit lines across 15 modules; fringing source is
  6,004 lines and this tranche removes four net fringing lines while correcting
  the accepted-state algorithm. The immutable source candidate is
  `c4866ac5af50ed5217e5be676b04577adb266f7e`.
- The pinned two-rank Docker comparison passes against FreeMHD
  `14b54a3e8e1a05b6ee4c98331995abaaae96e7a5` and image
  `sha256:535e995d557d2a73f5ab997380cb47ee3b044af8d2871bdadd570cff4cf175a8`.
  Contract, artifacts, execution, independent observation, comparison, and
  schema pass with zero failed checks. Pressure L-infinity/RMS differences are
  `0.006838504934992036`/`0.0030649109539923614`, unchanged at the smoke scale;
  report and record SHA-256 digests are
  `a99597286452f6f8fc6b5ca539f50303d4fee41c5fa95ed6c882abdfffe38f8b`
  and `02667bf4ff9ca1d8ed065a8e81571c1ab28da3f7d6f58ca87d19df286a83da7a`.
  `acceptance_pass=false` remains correct because this two-update harness is an
  integration smoke, not production-mesh acceptance. Next action: merge this
  candidate locally, then start the
  recurrence/private-test-coupling trim without removing 3-D, B1/B2, pipe,
  layered-wall, restart, sharding, or differentiation capability.

### 2026-08-29 — remove private fringing solver scaffolding

- Removed the unused injectable `_system_solve` lane from cylindrical
  diffusion and fused its single-use 48-line proxy into the only owning pipe
  solve. The resulting function still assembles the mapped cylindrical volume,
  wall/reaction sink, axial/radial additive line preconditioner, and calls
  SOLVAX `pcg_linear_solve` with the same primal and transpose tolerances. No
  geometry, boundary condition, numerical option, public API, or derivative
  path changes.
- Fused the single-use cross-duct pressure-tap kernel into its validated owner.
  The same array-only wall reduction remains sharding-compatible, while the
  redundant call boundary and misleading second “kernel” owner are gone.
- This tranche adds 29 source lines and removes 76, a net reduction of 47.
  Private/public fringing source falls from 6,004 to 5,957 lines;
  `_fringing_common.py` falls from 1,293 to 1,277 and `_fringing_pipe.py` from
  1,483 to 1,452. Package source falls from 14,796 to 14,749 audit lines across
  the same 15 modules. The wheel falls from 146,369 to 146,152 bytes and the
  sdist from 139,151 to 138,931 bytes.
- The complete six-worker local gate passes all 501 tests in 178.23 seconds
  (179.7 seconds end to end) with 95.29% combined line/branch coverage on
  Python 3.11.14, JAX/JAXLIB 0.9.2, and SOLVAX 0.18.0. Coverage and JUnit
  SHA-256 digests are
  `1775b1812079f1623ec1d7c1e450c147f80696b36d6999df75395e936dc36df1`
  and `353cdc5a10a256d3f63e9489ef3761e232fde442a9fb1a41a9e6c2b1425e04ae`.
  Ruff, formatting, byte compilation, architecture/import budgets, curated
  workflows, Sphinx HTML and external links with warnings as errors, isolated
  build, Twine, distribution inspection, and clean-wheel primal/gradient smoke
  pass. Distribution SHA-256 digests are
  `fe5031fde275e67e484e6bf827f591c36c881ce9b119a2d891954996dda5663a`
  and `714b1f4fafa6bb446d666bc6a690153086beaef96afddd37e7cb791ee43d7b36`.
- The three underscore modules remain because shared mapped, rectangular, and
  cylindrical mathematics are still independently live. Their names do not
  excuse their size: the next pass audits single-consumer B2 runtime assembly,
  repeated generic duct/pipe recurrences, and private-only test imports. A file
  will be deleted when that work leaves no coherent mathematical owner, not by
  concatenating unchanged code into `fringing.py`.
- The immutable source candidate is
  `f4b36250cda64a9809bb335df9d835b51c131144`. The pinned two-rank Docker
  comparison passes against FreeMHD `14b54a3` and image
  `sha256:535e995d557d2a73f5ab997380cb47ee3b044af8d2871bdadd570cff4cf175a8`
  with zero failed checks. Pressure L-infinity/RMS differences remain
  `0.006838504934992036`/`0.0030649109539923614`; report and record SHA-256
  digests are
  `3a78bb769955f0c9956dba691a8bf977efe82b7321136b8128588f12304d836d`
  and `cbb04c6974c771e9f95eac2f6652d0e0ceccb848d11525554e2d4d3b00f63887`.
  `acceptance_pass=false` remains correct for the smoke-only role.

### 2026-08-29 — reset execution around the scientific blockers

- A final review of the JAX implicit-solve, rematerialization, sharding,
  profiling, compilation-cache, and GPU guidance, together with current
  differentiable-CFD and FreeMHD literature, confirms the retained numerical
  direction. Converged systems keep implicit linear adjoints, finite recurrences
  use selective checkpointing, sharding stays explicit, and performance
  evidence separates cold compilation from synchronized warm execution. The
  critical path is terminal B2 primal/adjoint evidence, real A4000 execution,
  matched B1 external validation, and one fusion-design optimization—not more
  standalone repository trimming or documentation rearrangement.
- Decision D-049 adds a conservative `--changed-from` gate with fail-closed
  ownership mapping and an external JAX compilation cache. Pull requests run
  the complete three-shard 95% coverage gate once when source/tests change,
  targeted tests for script-only changes, and no Python suite for prose-only
  changes. Documentation and FreeMHD workflows always report a status but skip
  expensive jobs with job-level conditions; this avoids both redundant work
  and required checks left pending by workflow-level path filters.
- The affected local candidate selected 302 tests and passed in 70.47 seconds
  (71.1 seconds end to end). The one complete qualification passed 502 tests in
  168.15 seconds (169.9 seconds end to end) at 95.29% combined line/branch
  coverage. Ruff, formatting, YAML parsing, diff hygiene, the 14,750-line/
  15-module architecture/import audit, and warnings-as-errors Sphinx HTML pass.
- `benchmark_solver` now synchronizes final velocity and potential arrays
  before stopping each timer. The external selector was exercised against this
  hunk and correctly skipped FreeMHD because no Benchmark-B contract or B2
  numerical surface changed. Package, external-link, Docker, and GPU gates were
  intentionally not rerun. The candidate remains 88 tracked files and
  1,872,579 tracked bytes. Next action: close the terminal B2 primal and
  transpose-solve acceptance gate.

### 2026-08-29 — condition the terminal B2 Anderson history

- Continued the current accepted-state step-32 restart on the exact 101x65x65
  fluid mesh. The physical momentum defect continued descending through step
  88 to `0.16867260`, but depth-two Anderson updates became oscillatory and
  reached `0.21000656`; divergence and charge remained bounded. The campaign
  driver exposed its configured 128-update horizon in the progress record, so
  the run was stopped at the valid atomic step-88 checkpoint instead of
  spending another ten minutes on a known unstable accelerator.
- At step 89 the two residual norms are `0.30102388` and `0.45851147`. Their
  Gram matrix has condition number `58.4998` and unfiltered SOLVAX weights
  `[2.16785015, -1.16785015]`, producing a `0.41279231` accepted velocity jump.
  Decision D-050 passes `condition_limit=5` to the existing SOLVAX spectral
  filter, bounding the effective Gram condition at 25. The filtered weights are
  `[0.39237697, 0.60762303]`; no solve, field, restart value, or public option is
  added.
- On the identical step-88 state, the guarded step reduces the accepted update
  to `0.07565518` and improves defect from the unguarded `0.16809644` to
  `0.16669591`. Over steps 89--96, defect decreases monotonically to
  `0.16247107` and update contracts to `0.00166472`; maximum divergence and
  charge are `3.97e-5` and `2.42e-4`. This is also better than the previously
  measured raw-map step-96 defect `0.17898851`. The reduced B2 boundary,
  convergence, sharding, and exact-restart test passes. The office SSH endpoint
  timed out before connection, so terminal GPU continuation remains pending.
- The conservative owning-surface gate passes 213 fringing, Benchmark-B,
  FreeMHD-contract, and example tests in 103.21 seconds (103.9 seconds end to
  end). The complete covered run passes all 500 numerical/runtime assertions at
  95.29% combined line/branch coverage and rejects only a two-line test-source
  budget excess. Compressing that regression without changing its assertion
  restores the 11,950-line test budget; the three rejected architecture tests
  and the new conditioning regression then pass exactly. No runtime source
  changed after the covered run, so all 503 candidate tests have passing
  evidence without repeating 500 unchanged tests. Ruff, formatting, the
  14,754-line/15-module architecture/import audit, diff hygiene, and
  warnings-as-errors Sphinx HTML pass.
- The README now leads with a problem-to-example chooser, exact commands and
  outputs, a four-stage research adaptation workflow, differentiable 3-D/Q2D
  examples, evidence status, and explicit non-goals. It removes the nonexistent
  PyPI installation claim. The ungenerated 46,678-byte two-resolution fringing
  line plot is deleted rather than presented as a convergence curve; retained
  figures identify their full profile, 41 optimization iterates/seven design
  stations, and 41 Q2D frames. The repository falls to 87 tracked files and
  1,832,829 tracked bytes.
- Clean source commit `37f0e729c431977a7bbce99c13d464a1673c7f67` passes the
  cached pinned two-rank Docker comparison against FreeMHD `14b54a3` and image
  `sha256:535e995d557d2a73f5ab997380cb47ee3b044af8d2871bdadd570cff4cf175a8`.
  Contract, artifacts, execution, observation, comparison, and schema pass with
  zero failed checks; pressure L-infinity/RMS remain
  `0.006838504934992036`/`0.0030649109539923614`. Report and record SHA-256 are
  `5e1ebd7d7c00a47a124590be5998d413cce2168c46eb3e848f2417dc54104843`
  and `b1dc7438ce96807f6793b45b52ec132baa9b2f3b0c6eddddb0d913dcd59421fd`.
  `acceptance_pass=false` remains correct for the two-update smoke role. Next
  action: resume the conditioned step-88 state in bounded, checkpointed CPU
  segments to establish terminal primal convergence before constructing the
  specialized implicit transpose solve.

### 2026-08-29 — defer GPU work behind CPU scientific acceptance

- Decision D-051 moves accelerator performance and scaling evidence behind the
  terminal B2 primal/adjoint, matched B1 validation, and fusion-design
  demonstration. This is an ordering decision, not a reduced capability goal:
  GPU parity, speed, memory, and strong-scaling claims remain release gates.
- The current tranche therefore uses bounded CPU continuations with durable
  restarts and evaluates physical defect, accepted-state contraction, mass and
  charge closure at each boundary. It does not rerun unrelated unit, package,
  documentation, Docker, or GPU gates for evidence-only campaign checkpoints.

### 2026-08-29 — bound CPU campaigns and reject brute-force B2 continuation

- Decision D-052 adds one campaign-only `--additional-steps` control to bound
  newly executed outer updates. It propagates through isolated workers, leaves
  frozen variant controls and acceptance thresholds unchanged, and records the
  effective bound in each run. This closes the failure mode in which resuming a
  valid checkpoint silently launches the full 1,000-update production horizon.
- The checksummed guarded state advanced from step 88 to 120 in 87.2 seconds.
  At the step-96/104/112/120 boundaries the momentum defect was
  `0.16247107`/`0.15814175`/`0.15476510`/`0.14901404`; the terminal accepted
  update was `0.00158914`, maximum divergence `2.88e-5`, and maximum charge
  residual `2.42e-4`. Restart SHA-256 is
  `462946eb7d1fb3a3d6f85d80de4742aae6b72774ff3a0a8e24bcdaf362032208`.
- The official bounded runner then advanced exactly 32 more updates. At steps
  128/136/144/152 the defect was
  `0.14717852`/`0.14367309`/`0.14068780`/`0.13780740`; the update contracted
  from `0.00155001` to `0.00144850`. Mass, current, boundary-current, pressure
  diagnostics, and pressure convergence gates pass; steady residual,
  sustained stopping, and the declared `1e-3` momentum gate remain open. The
  complete recorded run, including four 59-MB-class atomic checkpoints, took
  221.1 seconds. Final restart SHA-256 is
  `3aa1c4626889b056ef601867320d80c6b5a2eab5d3fe1ae1c3d4198a43dc23f5`.
- Sustained contraction establishes that the condition guard is sound, but
  linear extrapolation would spend many more production segments before
  physical acceptance. Further unmodified recurrence is therefore rejected as
  the active strategy. The next numerical change returns to the previously
  localized blocker: expose and verify the coupled pressure response
  `D A^-1 G` for a consistent SOLVAX block preconditioner, first with a dense
  reduced identity and then on this exact step-152 state. The accepted
  recurrence remains the fallback and specialized B2 differentiation remains
  unavailable until the primal gate passes.
- The change-aware gate selected only the owning campaign tests and passed all
  21 on the final deduplicated candidate in 5.71 seconds (6.1 seconds end to
  end). Ruff check/format and the
  architecture audit also pass: 15 modules, 14,754 package lines, 11,950 test
  lines, and 28 root exports. No unrelated physics, documentation, package,
  Docker, FreeMHD, or GPU gate was rerun.
- PR #47's docs, external-validation, and CI entry jobs each ended before their
  first step in 2--3 seconds. Every check-run annotation reports the account
  payment/spending-limit condition; all downstream jobs were consequently
  skipped. This is hosted-service unavailability, not contrary test evidence,
  so the recorded local gate remains merge authority under the active
  temporary policy.

### 2026-08-29 — establish the matrix-free B2 Schur contract

- Materialized the frozen momentum block and both pressure responses on the
  maintained 4x2x2 nonuniform dense fixture. The full mixed-boundary
  `D A^-1 G` response is nonsingular with condition number `11.1027`; its
  nonsymmetry ratio is `0.09644`, consistent with the frozen convective
  momentum block. This fixes the sign, inlet-flow, outlet-pressure, and
  transverse-wall conventions before production integration.
- The retained diagonal-mobility response `D diag(A)^-1 G` has condition number
  `35.5690`, differs from the full response by `2.5673` in relative Frobenius
  norm, and leaves `S_diag^-1 S` at condition number `27.973`. Decision D-053
  therefore retains it only as a cheap inner approximation; it is not accepted
  as evidence for the compatible coupled preconditioner.
- SOLVAX owns the missing reusable operation: a matrix-free Schur block
  factorization on matching JAX pytrees. LMX will supply its frozen momentum
  inverse, conservative pressure-force action, face-flux divergence, and
  pressure-response inverse. This composition stores no Krylov trajectory and
  is compatible with shifted pseudo-transient solves and implicit transpose
  differentiation. Production promotion still requires a reduced dense
  identity test plus lower runtime and defect on the exact step-152 state.
- SOLVAX PR #96 merged the public `schur_complement_precond` composition at
  `1bba449ffa413ee884ee9113bea1929b467db2bc`. The local exhaustive gate passes
  755 tests with six optional-backend skips and 97.53% combined line/branch
  coverage; mypy, warnings-as-errors Sphinx HTML, isolated distributions, and
  Twine pass. Every hosted build, lint, type, minimum/current/advanced stack,
  macOS, combined-coverage, and Codecov check is green; the longest exhaustive
  shard is 7m59s.
- LMX now owns one conservative mixed-boundary face divergence and one
  cell-velocity-to-divergence action; the production pressure projection and
  reduced Schur contract use the same operators. On the dense fixture, the
  matrix-free factorization matches a direct solve of `[[A,G],[D,0]]` to
  `2e-12`. Projection conservation and the reduced Benchmark-B restart gate
  remain green. Two standalone result/proxy tests are consolidated into the
  maintained Benchmark-B and dense-operator fixtures, leaving 11,949 test
  lines rather than raising the budget. Next action: compose the shifted
  production preconditioner and compare bounded work on the exact step-152
  state before changing the public B2 recurrence.
- The conservative owning-surface gate selected 211 fringing, Benchmark-B,
  FreeMHD-contract, and example tests and passed in 83.85 seconds (84.2 seconds
  end to end). Ruff, formatting, the 14,779-line/15-module architecture audit,
  and diff hygiene pass. The one complete local source-PR qualification passes
  all 501 tests in 134.43 seconds (135.8 seconds end to end) with 95.24%
  combined line/branch coverage. Coverage and JUnit SHA-256 digests are
  `65001b3d23c4935f5df88fb8d2eb5892602aca3f6ba0076525671cdf330ed414`
  and `6f4bd1e3c178a4d931bc9396a6eb485cbd994a6edc51fa755229e06a4311d5cc`.
- PR #48's three entry jobs queued for 2m37s--3m51s, then ended before their
  first step with the account payment/spending-limit annotation; every
  downstream job was skipped. The complete local covered gate is therefore
  merge authority under the active temporary policy. No unrelated package,
  documentation, Docker, FreeMHD, or GPU gate is repeated.

### 2026-08-29 — expose the frozen B2 momentum response

- PR #48 merged the reduced Schur contract at
  `35910e67bf77d578950d6fce5e90da8c6dbaa8b5`. The production momentum solve
  now also accepts an already scaled algebraic right-hand side, applying the
  identical frozen nonsymmetric operator, diagonal preconditioner, bounded
  GMRES policy, and implicit linear derivative. This is private assembly for
  the coupled block method; it adds no solver, public option, export, or file.
- The response path deliberately skips affine inlet and boundary-source work.
  On the maintained nonuniform 4x2x2 fixture its primal result matches a
  direct dense solve of the production matrix to `2e-7`; the existing
  implicit JVP/VJP gate remains green, while the stricter `D A^-1 G` block
  identity remains at `2e-12`.
- This primitive is not a production-method promotion. The next bounded
  experiment will combine a small fixed number of these response applications
  with the retained diagonal pressure solve as a Schur defect correction, then
  compare runtime, peak memory, accepted update, and momentum defect on the
  exact checksummed step-152 restart. A full momentum Krylov solve inside each
  pressure Krylov iteration remains prohibited by decision D-038; a failed
  experiment leaves the public recurrence unchanged.
- The conservative change gate selected the 211 owning fringing, Benchmark-B,
  FreeMHD-contract, and example tests and passed in 83.65 seconds (84.0 seconds
  end to end). Ruff and the architecture audit pass at the enforced ceiling:
  15 modules, 14,793 package lines, 11,950 test lines, 28 root exports, and
  seven curated examples. No unrelated full-suite, documentation, package,
  Docker, or GPU job was repeated.
- PR #49's three entry jobs ended after 21--25 seconds with zero executed
  steps. Each check-run annotation reports the same account payment/spending-
  limit condition, and all downstream jobs were skipped. The targeted local
  evidence is therefore merge authority under the active temporary policy.

### 2026-08-29 — lock the finalization queue

- PR #49 merged the private frozen-momentum response at
  `d4dcc9b63ab630ffd857fa99b595be692f324964`. The live architecture inventory
  is 87 tracked files, 1,847,518 bytes, 15 modules, 14,793 package lines,
  11,950 test lines, 28 root exports, and seven curated examples.
- Converted the broad roadmap into six ordered finish gates: terminal CPU B2
  primal, its implicit derivative, matched B1 external validation, one fusion-
  design demonstration, deferred real-GPU evidence, and one release
  qualification. Completed repository, API, documentation, Q2D, packaging,
  and ownership work is no longer eligible for standalone follow-up tranches.
- Closed the function-level ownership audit. All four retained fringing owners
  have protected production responsibilities, while the rejected façade,
  testbeds, histories, proxies, and general numerical algebra have already
  been removed or moved to SOLVAX. Future simplification is opportunistic and
  must accompany a finish-gate change.
- Inspected the pinned local FreeMHD source before scheduling B1 validation.
  It contains no ready ALEX B1 circular-pipe case, so that gate requires an
  independently specified matched case; the current internal B1 benchmark is
  not promoted by assertion. The sole active code experiment remains the
  fixed-work B2 Schur defect correction, with a reduced performance/physics
  gate before one production restart run.

### 2026-08-29 — reject fixed-work predictor defect correction

- Tested the sole predictor-level candidate authorized by the finalization
  ledger. It replaced the second physical B2 momentum predictor with one
  homogeneous frozen response to the first pressure correction, retaining two
  pressure solves and the same number of momentum Krylov solves. It added no
  nested inverse, solver family, public parameter, or changed stopping gate.
- On the identical maintained 5x5x5 six-step case, current main reaches update
  `0.01569680` and momentum defect `0.16896604` in 7.223 seconds. The candidate
  reaches the smaller algebraic update `0.00903749` but the worse physical
  defect `0.19460791` in 7.732 seconds: 15.2% worse momentum balance and 7.0%
  more wall time. Charge closure remains valid, so the failure is the coupled
  physical trajectory rather than an electric-solve breakdown.
- The candidate fails before production promotion. Every experimental source
  edit was deleted, no step-152 run was spent, and no numerical test or user
  option was retained. This closes fixed-work predictor defect correction and
  reinforces that algebraic map contraction is not an acceptance surrogate.
- Remaining B2 work is narrowed to an operator-form coupled momentum/pressure
  solve with a cheap non-nested approximate block inverse. It must apply the
  frozen momentum and conservative divergence/gradient operators directly;
  another SIMPLE/predictor correction or full momentum inverse inside a
  pressure iteration is outside the finish queue.
