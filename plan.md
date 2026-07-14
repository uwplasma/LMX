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
| Benchmark-B contracts | schema 2 composes shared physics with production execution roles and recomputes real artifact hashes; acceptance is observer-blocked | finish the O(nx) coarse solve and diagnostics, then independent input observers and the smoke role |
| B1 ALEX pipe | retained-modal numerical evidence exists | implement/prove the canonical formulation, then exact parity |
| B2 ALEX square duct | exact frozen limited-linear momentum, lagged stress, corrected compact flux, mixed projection, periodic Aitken restart identity, and forced one-/two-CPU equivalence pass bounded gates | replace the dense axial coarse inverse, then materialize and run the exact tiny FreeMHD smoke |
| SOLVAX | released 0.8.3 owns the GMRES, implicit differentiation, PCG, additive, Aitken, and tridiagonal algebra needed now; prepared 0.8.4 commit `4808695` adds reusable Anderson weights | use 0.8.3 for the O(nx) axial solve; publish 0.8.4 separately before consuming shared Anderson weights |
| Portable quality | accepted at `cf76c5a`: 769 pass, 8 expected skips, 95.24% combined line/branch coverage, 167.0 s | preserve the 300 s target and 600 s hard limit while consolidating rather than adding test files |

Current structural audit at `cf76c5a`: 35 modules, exactly 35,000 package
lines, 8,049 maintained-core lines, 32 test files / exactly 21,300 lines, and
18 maintenance scripts. These are ceilings, not targets: every added branch or
test must consolidate or delete at least as much code in the same tranche.
The four audited probe worktrees contained only promoted or rejected work;
their findings are recorded here, and local/remote navigation is now `main` only.

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
  `linearInterpolate(rho*U)&Sf`, checkpoint velocity, pressure, and flux
  atomically, and apply identical affine relaxation coefficients to velocity
  and flux.
- No medium or production FreeMHD run is authorized until a tiny exact smoke
  proves those semantics in both codes.
- Expanded `(nx+1)` axial face arrays are not shard-safe on an evenly divided
  cell mesh. Projection now carries `nx` positive faces plus one replicated
  inlet plane and exchanges the one-cell halo explicitly with `ppermute`.
- Periodic B2 checkpoints are exact Aitken continuation points: they persist
  the previous scaled residual, relaxation, convergence streak, compact flux,
  and all five histories. A deliberately final step still skips acceleration,
  so a completed run is not reinterpreted as a mid-run checkpoint.

Machine-readable evidence is in `benchmarks/results/`; interpretation belongs
in `docs/validation_report.md`, `docs/external_benchmarks.md`, and
`docs/performance.md`.

## Immediate execution order

1. **Complete:** schema 2, real artifact verification, provenance, and the
   current 769-pass portable gate at 95.24% coverage in 167.0 seconds.
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
   keep only expanded full-face arrays fused/internal, persist the compact
   corrected carry, and prove conservation, scaling, JIT, and JVP.
6. **Release preparation complete at SOLVAX `4808695`:** SOLVAX 0.8.4 exposes
   auxiliary/transpose solves and reusable `anderson_weights`; 264 tests pass
   in 22.46 seconds at 98.88% coverage, and build, Twine, and isolated-wheel
   checks are green. LMX does not need these additions for Aitken or its current
   momentum diagnostics. Keep `solvax>=0.8.3,<1`; tag, publish, and
   install-verify 0.8.4 only through the separate SOLVAX release process.
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
9. **Complete at `ed76e59`–`1373850`:** compact positive-face flux, exact fresh
   initialization and schema, lagged deviatoric stress, corrected-flux carry,
   fail-loud clipping removal, affine Aitken flux relaxation, periodic exact
   Aitken restart, zero-current/flux closure, and the `8x4x3` forced one-/two-CPU
   production gate. `fa4e2e7` exposed and fixed a real axial shard-boundary bug;
   the complete portable gate is green at `cf76c5a`. Anderson remains rejected
   until released shared weights and bounded field/flux histories exist.
10. Replace the dense axial coarse inverse with released SOLVAX 0.8.3
   `tridiagonal_solve`: anchor the mixed solve, apply the analytical volume
   gauge shift where required, and prove dense parity, variable coefficients,
   JIT/JVP/VJP, residual, and one-/two-device placement on tiny grids. Delete
   the dense assembly/inverse in the same tranche.
11. Add explicit CFL and stopping diagnostics to the existing compact B2 record
   and gate their restart continuity; do not add a new report or test file.
12. Materialize both tiny inputs independently, derive their observed contracts,
   then freeze smoke mesh, mapped field, time step, iterations, and stopping
   rules. Commit and push before invoking FreeMHD.
13. Extend the existing parity command to run one exact tiny B2 smoke. Compare
   mass/current closure, stopping, hashes, and the same pressure observable. A
   failure returns to the first failed tiny gate.
14. Prove one-/multi-device equivalence on that accepted tiny path, then measure
   fixed-size Mac 1/2/4-device CPU and office one-/two-GPU warm timings alone.
15. Advance B2 one level at a time: exact coarse comparison first, then medium
   and fine only if literature, FreeMHD, conservation, restart, wall, and
   tolerance gates pass for the correct reason.
16. Refresh only the canonical B2 and scaling visuals/claims from compact
   accepted records without rerunning solvers.
17. Complete the release gate. Exact B1 parity is the first post-release physics
   tranche. Do not revive composite-map FGMRES: it passed the tiny mesh but
   stagnated at `3.17e-4` after 64 Krylov iterations on the reduced production
   case, and its local-rate preconditioner regressed. A future structural steady
   coupled block solve must first hoist three host `float(...)` validation checks
   that currently block B1 JIT; never adopt the rejected silent-tracer bypass.

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
JIT kernels that have an explicit shard-cut gate; projection itself stays in
the compact cell-shaped representation and exchanges axial halos explicitly.
Do not use periodic `roll`. Profile compiled temporary memory before attempting
a more complex codec, and do not claim scaling from placement alone.

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
- Keep `_additive_line_preconditioner_3d`: it only maps LMX coefficient axes
  and periodicity into SOLVAX's existing additive/tridiagonal APIs. Moving it
  has zero net deletion until SOLVAX has a reusable axis-line builder with a
  second consumer.
- Keep the tiny dense flow-response solves and SciPy reference/fallback paths;
  they encode validation, anchor, gauge, or finite-volume semantics and are not
  duplicate generic solver ownership.
- After SOLVAX 0.8.4 is independently published and installed, use one
  `anderson_weights` vector for scaled fields and both compact-flux histories.
  That is a correctness tranche with bounded restart histories, not a slimming
  shortcut, and must update the minimum-version CI lane in the same commit.
- Do not delegate MHD coupling, finite-volume coefficients, physical
  fixed-point loops, or acceptance logic.

Ratchet only through real ownership deletion:

| Surface | Current | Next target |
|---|---:|---:|
| package modules | 35 | stay at 35 until a complete owner disappears |
| package lines | 35,000 | below 35,000 every tranche; below 34,800 after smoke cleanup |
| maintained-core lines | 8,049 | below 8,000 after the exact smoke |
| test files / lines | 32 / 21,300 | 32 / below 21,000 by consolidating post-smoke evidence |
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
