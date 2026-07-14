# LMX authoritative development plan

This is the single project plan. It records priorities and acceptance gates,
not campaign history. Raw fields, logs, rejected probes, and large media belong
in external workspaces or checksummed releases.

## Product target

LMX will be a lightweight JAX code for accurate, end-to-end differentiable
inductionless liquid-metal MHD on CPUs and GPUs. A stable claim requires the
appropriate combination of analytical or manufactured verification,
conservation and convergence, literature or independent-code evidence, and
reproducible performance measurements.

The public repository has four working surfaces:

- `lmx/`: source and the `lmx` command;
- `tests/`: bounded unit, numerical, physics, regression, and workflow checks;
- `examples/`: small runnable Python and TOML workflows;
- `docs/`: theory, inputs, validation, performance, and development guidance.

Large meshes, complete external-code cases, raw transient fields, restarts,
full-resolution figures, and movies do not belong in Git.

## Operating contract

### Small first

Every experiment declares one hypothesis, frozen metrics, a wall-time ceiling,
a stop rule, and a go/no-go threshold before launch. Escalation is:

1. static, analytical, plotting-only, or tiny-grid check;
2. bounded one-device smoke run;
3. medium confirmation only after the smoke gate passes;
4. fine, external-code, or multi-hour campaign only after every earlier gate,
   with durable restarts and interim stop/go checks.

Reuse existing checksummed data when it answers the question. A failed bounded
probe stops; it does not trigger an open-ended parameter search.

### Test only what changed

- Development: lint/static checks and direct node IDs, normally under two
  minutes.
- Subsystem boundary: affected test modules and repository gates, normally
  under five minutes.
- Once per coherent green tranche and in CI/release: the complete portable
  branch-coverage gate, with a 300-second target and 600-second hard limit.
- FreeMHD, accelerator, and mesh-refinement campaigns: explicit manual or
  scheduled lanes with bounded portable representatives.

Branch coverage of `lmx/` stays at least 95%. Coverage is not physics
validation. Every public workflow maps to one portable test and, where needed,
a numerical, physics, external, gradient, or performance gate. The `unit`,
`numerical`, `regression`, `physics`, `validation`, and `external` markers
remain enforced while tests are consolidated.

### Parallel work and evidence

Use subagents for disjoint literature, ownership, media, and validation audits;
one integrator owns shared files. Run independent non-timing checks in
parallel, but run performance measurements alone on the measured host. Commit
and push every coherent green tranche, keep `main` authoritative, and remove
superseded worktrees and branches.

## Current status and corrected claim hierarchy

| Area | Status | Next acceptance gate |
|---|---|---|
| README/docs | 567-word feature-led README, conservatively sourced comparison table, feature-specific docs, and a 7-second Hunt loop; tracked media is 387,683 bytes | refresh canonical B2/scaling panels only from accepted records |
| Developed ducts | Hartmann, Shercliff, Hunt, and all eight high-Ha rows accepted | preserve regression gates |
| FreeMHD closed channels | bounded Shercliff/Hunt parity accepted | do not generalize to full FreeMHD parity |
| Benchmark-B contracts | schema 2 composes shared physics with production execution roles and recomputes real artifact hashes; acceptance is observer-blocked | finish canonical B2, then independent input observers and the smoke role |
| B1 ALEX pipe | retained-modal numerical evidence exists | implement/prove the canonical formulation, then exact parity |
| B2 ALEX square duct | the mixed projection and exact frozen limited-linear operator pass tiny gates; one stacked nonsymmetric momentum solve now replaces three component diffusion solves | carry corrected mass flux through time, add the lagged deviatoric-stress correction, and integrate the exact step |
| SOLVAX | released 0.8.3 already provides the GMRES, implicit differentiation, PCG, additive, and tridiagonal APIs needed now; prepared 0.8.4 adds auxiliary diagnostics and a distinct transpose solver | do not block B2 on 0.8.4; publish it only through the independent SOLVAX release process |
| Portable quality | accepted at `b763b84`: 784 pass, 8 expected skips, 95.28% branch coverage, 149.9 s | run the next full gate after corrected-flux B2 integration; stay below the 300 s target and 600 s limit |

Current structural audit at `09806e9`: 35 modules, 34,936 package lines, 8,027
maintained-core lines, 32 test files / 21,300 lines, and 18 maintenance scripts.
In-progress metrics are not status evidence; every tranche must return below
the enforced live caps before commit.

The final audit freezes these interpretations:

- The canonical B2 field is transverse `B=(0,B_y(x),0)`. A
  Maxwell-consistent field is a labelled sensitivity study, not an acceptance
  prerequisite unless the contract is deliberately reopened.
- The previous B2 convergence and 1.66x two-GPU measurements do not validate or
  scale the new canonical path.
- FreeMHD v2206 uses implicit conservative `fvm::div(rhoPhi,U)` inertia and
  Euler time integration. `Gauss limitedLinear 1.0` on vector `U` is one
  `magSqr(U)`-derived scalar limiter applied to every velocity component; its
  limiter gradient is `cellLimited leastSquares 1.0`. A componentwise MUSCL
  approximation or explicit convection source is not exact parity.
- The same pressure-corrected, oriented, area-integrated mass flux must feed
  projection and momentum convection. The inlet constrains one integral flow,
  the outlet fixes pressure and extrapolates velocity, walls have zero normal
  mass flux, and the matched electric reduction has zero normal current at
  both axial ends.
- FreeMHD's laminar operator also contains the lagged explicit term
  `div(mu*dev2(T(grad(U))))`. It moves to the LMX momentum right-hand side
  with a positive sign. It vanishes for the fully developed verification lane,
  but not for axially varying canonical B2 flow.
- Corrected face flux cannot be reconstructed exactly from cell velocity. Store
  one positive-direction face per cell plus the inlet plane, initialize it from
  `linearInterpolate(rho*U)&Sf`, and treat velocity, pressure, and flux as one
  coupled state for relaxation and restart.
- No medium or production FreeMHD run is authorized until a tiny exact smoke
  proves those semantics in both codes.

Machine-readable evidence is in `benchmarks/results/`; interpretation belongs
in `docs/validation_report.md`, `docs/external_benchmarks.md`, and
`docs/performance.md`.

## Immediate execution order

1. **Complete:** schema 2, real artifact verification, provenance, and the
   784-pass portable gate at 95.28% branch coverage in 149.9 seconds.
2. **Complete:** re-audit FreeMHD v2206 limiter, implicit momentum assembly,
   mixed axial boundaries, SOLVAX v0.8.3 ownership, and sharding constraints.
3. **Complete:** freeze the missing `cellLimited leastSquares 1.0` gradient
   contract and pinned FreeMHD/OpenFOAM source evidence.
4. **Complete:** implement and prove the B2 mixed projection on a `4x2x2` manufactured
   problem: inlet pressure Neumann, outlet pressure Dirichlet at the half-cell
   face, one inlet flow constraint, no compatibility or gauge projection. The
   reduced `7x7x7` caller and restart gate also pass.
5. **Complete at `cb115ab`:** implement the exact v2206 limited-linear face
   weights and conservative matrix action on a `7x7x3` nonuniform manufactured
   problem. Include explicit patch-face values in the orthogonal least-squares
   gradient and cell limiter; preserve the exact guarded NVD sign/upwind rules;
   keep corrected face fluxes fused/internal; and prove conservation, scaling,
   JIT, and JVP.
6. **Release preparation complete at SOLVAX `255d280`:** SOLVAX 0.8.4 exposes
   `has_aux` and a distinct transpose solver through `linear_solve`; its full
   gate is 263 passes in 68 seconds at 98.88% coverage, and hosted tests/docs
   are green. LMX does not need those additions for its current three-value
   momentum diagnostics: released 0.8.3 supports one differentiable GMRES solve
   and LMX recomputes the physical residual. Keep `solvax>=0.8.3,<1`; tag,
   publish, and install-verify 0.8.4 separately before using its new API.
7. **Complete at `f7b66c9`:** apply the cheap user-surface/tooling correction
   from existing records only: narrow the README differentiation and multi-GPU
   claims, keep the comparison table restricted to stock documented capability,
   make a 6–8-second web loop from the existing Hunt movie, embed existing
   release-hosted visuals in their feature docs, synchronize architecture
   metrics, and set the full-test warning target to 300 seconds. Run only direct
   repository/docs/media gates, then commit and push; do not rerun a solver.
8. **Complete at `ee07544` and `09806e9`:** fold the one-caller mixed projection
   into the general face-flux projection, then replace the B2 component
   diffusion closures with one stacked nonsymmetric momentum operator. The
   `4x2x2` independent dense matrix, residual, JIT, placement, JVP/VJP, and
   zero-convection legacy-limit gates pass using released SOLVAX 0.8.3.
9. Complete canonical B2 in four bounded changes, each with direct tests:
   (a) add the compact positive-face mass-flux codec, exact fresh initializer,
   and restart schema; (b) add the lagged component-limited least-squares
   deviatoric-stress correction and its independent `5x4x3` balance gate;
   (c) carry corrected flux from projection into the next momentum solve and
   relax it with the same Aitken affine update as velocity; and (d) remove or
   explicitly reject velocity clipping that would invalidate flux consistency.
   Then prove zero-normal-current closure, two-step restart identity with
   acceleration disabled, persist Aitken residual/relaxation/streak state before
   claiming accelerated restart identity, add CFL/stopping diagnostics, and
   prove one-/two-device equivalence on an `8x4x3` forced-CPU mesh.
10. Materialize both tiny inputs, derive their contracts independently, then
   freeze smoke mesh, mapped field, time step, iterations, and stopping rules.
11. Run affected modules, then one portable gate. Commit and push before the
   external smoke.
12. Extend the existing parity command to run one exact tiny B2 smoke. Compare
   mass/current closure, stopping, hashes, and the same pressure observable. A
   failure returns to the first failed tiny gate.
13. Replace the dense axial coarse inverse with the audited O(nx) SOLVAX
   tridiagonal/gauge decomposition and prove dense, gradient, and placement
   parity on tiny grids.
14. Prove one-/multi-device equivalence on that accepted tiny path, then measure
   fixed-size Mac 1/2/4-device CPU and office one-/two-GPU warm timings alone.
15. Advance B2 one level at a time: exact coarse comparison first, then medium
   and fine only if literature, FreeMHD, conservation, restart, wall, and
   tolerance gates pass for the correct reason.
16. Refresh only the canonical B2 and scaling visuals/claims from compact
   accepted records without rerunning solvers.
17. Complete the release gate. Exact B1 parity is the first post-release physics
   tranche; existing retained-modal B1 evidence remains explicitly research-stage.

## Gate 1: authoritative matched harness

The production validator now requires both code contracts to equal the frozen
specification; equality between two submitted dictionaries is insufficient.
It rejects the legacy `exact_case_match` flag and prevents a smoke role from
self-promoting.

Completed foundation:

- Equations, groups, geometry family, wall model, field mapping, boundary
  semantics, observable, and normalization live in immutable shared sections;
  roles supply only mesh coordinates and stopping rules.
- Schema 2 resolves file/tree artifacts beneath an explicit root, rejects
  escapes, aliases, links, overlaps, missing/empty inputs, and type mismatches,
  and recomputes deterministic hashes from their contents.

Remaining work:

- Freeze the B2 smoke mesh and stopping contract only after the canonical
  solver and both materializers exist. Time step, iteration caps, correctors,
  and fixed step count are candidates—not evidence—until then.
- Report `contract_pass=True` and `acceptance_pass=False` for a valid smoke.
- Materialize LMX and FreeMHD inputs independently. Never populate observed
  contracts by copying the expected dictionary twice.
- Derive each observed contract from its real config, coordinates, field
  samples, boundary files, and source snapshot; neither observer receives the
  expected dictionary.
- Separate schema, contract, artifact, and comparison failures so an unrelated
  comparison error cannot make a valid physics contract appear invalid.
- Keep generated cases, logs, VTK, restarts, and raw histories outside Git.
  Only compact specifications, evaluators, and accepted summaries are tracked.

Exit: a record can pass its role-specific contract only when both codes
demonstrably solved the same problem; a smoke can never unlock production
acceptance.

## Gate 2: canonical B2 physics and numerics

Implement the frozen formulation without changing legacy generic behavior:

- implicit conservative finite-volume `fvm::div(rhoPhi,U)` using the exact
  pressure-corrected face mass flux and frozen v2206 `limitedLinear 1.0`
  vector semantics: one `magSqr(U)` limiter, guarded NVD ratio, owner/upwind
  blend, and `cellLimited leastSquares 1.0` gradients;
- Euler update and the matched laminar viscous-stress convention;
- inlet-only integral flow constraint, outlet zero-gradient velocity, inlet
  zero-gradient pressure, and fixed outlet pressure gauge;
- zero normal electric current at both axial ends;
- invariant single liquid phase and constant temperature/properties in the
  FreeMHD materialization.

Tiny gates precede any solve campaign:

1. a `4x2x2` cross-section-constant pressure field that exactly exercises the
   inlet-Neumann/outlet-half-cell-Dirichlet pressure coefficients;
2. inlet/outlet flux closure without stationwise forcing, including restart
   independence of the prescribed flow rate;
3. a `7x7x3` divergence-free manufactured velocity with exact conservative
   vector result, limiter weights, telescoping balance, quadratic scaling,
   JIT, and JVP away from limiter switches;
4. a tiny dense reference for the combined nonsymmetric implicit momentum
   matrix, SOLVAX GMRES residual, transpose/gradient, JIT, and placement;
5. a `5x4x3` independent explicit-stress reference: fully developed zero,
   quadratic axial value and sign, telescoping traction, JIT, and JVP;
6. compact face-flux pack/unpack, nonperiodic shard-cut, exact startup from
   interpolated velocity, and one-/two-device projection/momentum equivalence;
7. builder mutations, conservation, restart identity, CFL, and bounded
   differentiability checks.

PCG is valid for the mixed pressure SPD operator. It is not valid for the
convection-diffusion momentum operator. Use SOLVAX flexible GMRES wrapped by
`linear_solve` only after the dense and adjoint gates pass. Freeze face flux
and limiter weights for each outer momentum solve, matching the segregated
FreeMHD algebra rather than differentiating through an invented explicit term.

The mixed path now owns B2; `_fixed_flow_face_flux_projection_duct`, its
pressure-response plumbing, and its long legacy test are deleted. Retain
generic and pipe fixed-flow helpers only where still used.

Exit: tiny manufactured, gradient, boundary, conservation, and restart gates
all pass; the affected files and portable suite remain below their budgets.

## Gate 3: exact tiny FreeMHD smoke

Use the audited Docker source pin and existing `run_freemhd_parity_suite.py`;
do not add another runner. The smallest honest case retains nonzero inertia,
Lorentz force, diffusion, conducting-shell topology, and the canonical axial
conditions. It may reduce mesh and stopping requirements, not equations.

Run LMX and FreeMHD once each, then compare independently derived contracts,
mass/current closure, convergence, and the same normalized pressure observable.
Do not authorize medium work on visual similarity or a specially chosen field
that nulls advection.

Exit: the bounded smoke is contract-valid, numerically consistent, fully
checksummed, and explicitly ineligible for production acceptance.

## Gate 4: sharding and performance

Make one-solve sharding, placement, and collectives explicit. Independent-case
multiprocessing is throughput evidence, not strong scaling.

First require one-/multi-device numerical equivalence for physical observables.
Then separate compilation from repeated warm timings and report uncertainty,
memory, placement, speedup, and efficiency. Measure fixed-size Mac 1/2/4-device
CPU and office one-/two-GPU scaling; add four GPUs only on suitable hardware.
Optimize demonstrated bottlenecks rather than the superseded B2 path.

Production cells shard axially as `(nx, ny, nz)`. Persist corrected mass flux as
`rho_phi_plus` with shape `(3,nx,ny,nz)` and sharding
`P(None,"x",None,None)`, plus a replicated `(ny,nz)` inlet plane. Never
checkpoint duplicated `nx+1` face arrays. Expand full faces only inside fused
JIT kernels with nonperiodic concatenation and global slicing; do not use
periodic `roll`. Profile compiled temporary memory before attempting a more
complex codec, and do not claim `shard_map` scaling without explicit halo
exchange.

Exit: the physics-valid path has equivalent observables and useful measured
speedup on its target host.

## Architecture and SOLVAX ownership

LMX owns MHD equations, geometry, materials, boundary/interface conditions,
finite-volume interpolation and assembly, gauges, observables, sharding
policy, checkpoints, and physical acceptance. SOLVAX owns generic linear
algebra after primal, residual, gradient/transpose, JIT, placement, memory,
and repeated interleaved timing gates pass. Delete the LMX duplicate in the
same tranche.

Keep `solvax>=0.8.3,<1`: it is sufficient for the canonical B2 implementation.
Raise the lower compatibility bound only when production LMX actually consumes
a newer API, in the same tranche that updates the minimum CI lane. Test both
the minimum and newest compatible releases and record the resolved environment.
Do not tie LMX to one patch release or commit a resolver lock.

The next ownership audit is evidence-driven:

- Keep the exact limited-linear weights, face fluxes, mixed pressure stencil,
  and open-boundary semantics in LMX. SOLVAX has no finite-volume limiter or
  open-boundary assembly API, and moving them would obscure ownership without
  reducing domain code.
- Use SOLVAX PCG for the mixed pressure SPD system and SOLVAX GMRES plus
  `linear_solve` for the frozen-weight nonsymmetric momentum system. Do not use
  `FourierHelmholtz` for the open axial topology.
- The completed deletion tranche replaced `_solvax_implicit_diffusion_duct`
  with one stacked momentum solve and folded the one-caller mixed projection
  wrapper into `_face_flux_pressure_projection_duct`. Retain the shared line
  preconditioner, general face-flux projection, and generic/pipe paths.
- Replace the dense `nx`-by-`nx` axial coarse inverse with an anchored
  SOLVAX tridiagonal solve plus an analytical constant gauge mode. This is the
  smallest high-impact sharding fix; prove mixed modes, variable coefficients,
  JIT/gradient, and placement first.
- Then delete the complete legacy pipe Jacobi/SciPy Poisson stack by routing it
  through the accepted SOLVAX-PCG pipe wrapper; follow with duct electric and
  generic projection only after gauge, sign, flux, residual, and centerline
  parity. This is worth roughly 450 net core lines—far more than migrating
  isolated sparse calls.
- Use generated SOLVAX block-Thomas factors for B1 modal transpose solves only
  with dense-reference, plain-transpose, gauge, residual, gradient, cache,
  memory, and timing gates.
- Additive directional line composition already delegates to SOLVAX. The one
  remaining line-plus-axial sum may migrate after B2 parity, but its tiny
  deletion value does not justify interrupting the exact physics path.
- After the canonical B2 release, schedule the pipe-Poisson deletion, B1
  block-Thomas transpose migration, and remaining additive-combiner ownership
  as separate numbered tranches in the next plan revision. They do not block
  the present B2 release and must not interrupt its exact-parity path.
- Defer two isolated `splu_solve` substitutions and new nonlinear APIs until
  after canonical B2 parity; they do not currently buy enough deletion.
- Do not delegate MHD coupling, finite-volume coefficients, physical
  fixed-point loops, or acceptance logic.

Ratchet only through real ownership deletion:

| Surface | Current | Next target |
|---|---:|---:|
| package modules | 35 | 34 only after a complete owner disappears |
| package lines | 34,936 | below 34,900 after exact B2 integration |
| maintained-core lines | 8,027 | below 7,900 |
| test files / lines | 32 / 21,300 | 31 / below 20,900 after generic evidence replaces the dedicated SOLVAX freezer tests |
| maintenance scripts | 18 | 17 after `freeze_solvax_pcg_acceptance.py` is retired by the generic evidence gate |

Do not meet a budget through unreadable formatting, arbitrary test merging, or
relabelling code as research-stage.

## B2 and B1 production acceptance

For B2, use the frozen transverse field for primary acceptance. Run exact
coarse LMX/FreeMHD/literature comparison with restart, wall-thickness,
tolerance, steady-state, conservation, and pressure-observable gates. Medium
and fine run only after the preceding level passes. Maxwell-consistent fields
remain a separate sensitivity study.

Apply the same protocol to B1 only after the harness and shared formulation are
proved on B2: tiny parity, medium mesh independence, and one large confirmation
only if required. Existing retained-modal results remain valuable numerical
evidence but are not exact-formulation evidence.

Exit: each accepted case has source-identical equations, a three-mesh ladder,
literature and exact-FreeMHD evidence, frozen tolerances, and a reproducible
checksummed record.

## User surface and media

Keep the README below roughly 800 words: pitch, quickstart, sourced
LMX/FreeMHD/NekRS capability table, and concise visual sections for verified
ducts, fields/geometries, differentiation, research workflows, and scaling.
Each major feature gets a relevant plot or movie—not a generic media gallery.

First correct claim scope and place the existing visuals in their feature docs;
do not wait for new campaigns. Generate later presentation assets from compact
accepted records without rerunning solvers. Aim for readable 6–8-second loops,
compact posters, stills below 100 KiB, and tracked movies below 150 KiB where
practical. Keep tracked media
below 1 MiB total; host source frames, full-quality movies, and large outputs in
checksummed releases. Put provenance and acceptance status beside each asset.

Exit: a new user can understand scope, run a first case, see representative
results, and distinguish verified from research-stage functionality without
reading campaign history.

## Release gate

Consolidate tests and scripts while preserving one bounded check per public
capability. Build docs with warnings as errors, verify provenance and media,
build and inspect wheel/sdist, install the wheel in a clean environment, and
run the portable gate plus selected accepted external, gradient, and scaling
lanes. Publish exact commits, environments, references, tolerances,
limitations, and release assets. Public APIs retain concise docstrings and type
hints; non-obvious numerical choices retain comments explaining why.

Hosted CI is a release blocker, not a reporting footnote. Resolve the current
GitHub Actions billing/spending outage and require green hosted Python 3.10 with
minimum SOLVAX, Python 3.13 with newest compatible SOLVAX, strict docs, wheel,
and release jobs. The full-test driver warns at 300 seconds, fails at 600, and
each accepted portable record reports total time and slowest nodes against the
last green record.

Exit: the release is small, installable, reproducible, honestly scoped, and
suitable for external research use.
