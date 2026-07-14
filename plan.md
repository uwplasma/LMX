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
| README/docs | 709-word feature-led README, sourced comparison table, feature-specific compressed media | refresh only from accepted records |
| Developed ducts | Hartmann, Shercliff, Hunt, and all eight high-Ha rows accepted | preserve regression gates |
| FreeMHD closed channels | bounded Shercliff/Hunt parity accepted | do not generalize to full FreeMHD parity |
| Benchmark-B contracts | schema 2 composes shared physics with production execution roles and recomputes real artifact hashes; acceptance is observer-blocked | independent input observers, then canonical smoke role |
| B1 ALEX pipe | retained-modal numerical evidence exists | implement/prove the canonical formulation, then exact parity |
| B2 ALEX square duct | old fine-grid and two-GPU results are diagnostic for the superseded no-inertia, stationwise-flow formulation | implement canonical inertia and axial boundaries |
| SOLVAX | v0.8.3 owns PCG, cyclic lines, anchored Poisson PCG, and additive composition | pursue only gated ownership deletions |
| Portable quality | 782 pass, 8 expected skips, 95.30% branch coverage, 162.8 s | stay below the 300 s target and 600 s limit |

Current structural audit: 35 modules, 34,951 package lines, 7,995
maintained-core lines, 32 test files / 21,179 lines, and 18 maintenance
scripts.

The final audit freezes these interpretations:

- The canonical B2 field is transverse `B=(0,B_y(x),0)`. A
  Maxwell-consistent field is a labelled sensitivity study, not an acceptance
  prerequisite unless the contract is deliberately reopened.
- The previous B2 convergence and 1.66x two-GPU measurements do not validate or
  scale the new canonical path.
- FreeMHD uses conservative `div(rhoPhi,U)` inertia, Euler time integration,
  `Gauss limitedLinear 1.0`, inlet integral flow, and an outlet pressure gauge;
  the matched reduction also requires zero normal current at both axial ends.
- No medium or production FreeMHD run is authorized until a tiny exact smoke
  proves those semantics in both codes.

Machine-readable evidence is in `benchmarks/results/`; interpretation belongs
in `docs/validation_report.md`, `docs/external_benchmarks.md`, and
`docs/performance.md`.

## Immediate execution order

1. **Complete:** regenerate provenance for the canonical-contract tranche and
   pass the complete portable gate (superseded by the 782-pass schema-2 gate).
2. **Complete:** split canonical contracts into immutable shared physics and
   role-specific mesh/stopping sections; schema 2 recomputes deterministic
   file/tree artifacts and blocks acceptance until independent observers exist.
3. Implement the complete canonical B2 formulation with tiny tests first:
   projection-consistent conservative advection, frozen discretization,
   inlet-flow/outlet-pressure boundaries, and axial electric closure. Delete
   the obsolete B2 stationwise projection path in the same tranche.
4. Materialize both tiny inputs, derive their contracts independently, then
   freeze the smoke mesh, mapped field, time step, iterations, and stopping
   rules only after their exact mapping and bounded convergence are audited.
5. Run affected modules, then one portable gate. Commit and push before the
   external smoke.
6. Extend the existing parity command to run one exact tiny B2 smoke. Compare
   mass/current closure, stopping, hashes, and the same pressure observable. A
   failure returns to step 3.
7. Replace the dense axial coarse inverse with the audited O(nx) SOLVAX
   tridiagonal/gauge decomposition and prove dense, gradient, and placement
   parity on tiny grids.
8. Prove one-/multi-device equivalence on that accepted tiny path, then measure
   fixed-size Mac 1/2/4-device CPU and office one-/two-GPU warm timings alone.
9. Advance B2 one level at a time: exact coarse comparison first, then medium
   and fine only if literature, FreeMHD, conservation, restart, wall, and
   tolerance gates pass for the correct reason.
10. Refresh the README/docs from compact accepted records without rerunning
   solvers; then apply the proven harness/formulation path to B1.
11. Complete the release gate.

## Gate 1: authoritative matched harness

The production validator now requires both code contracts to equal the frozen
specification; equality between two submitted dictionaries is insufficient.
It rejects the legacy `exact_case_match` flag and prevents a smoke role from
self-promoting.

Remaining work:

- Store equations, groups, geometry family, wall model, field mapping, boundary
  semantics, observable, and normalization once as immutable shared sections;
  roles may supply only mesh coordinates and stopping rules.
- Bump matched records to schema 2. Resolve file/tree artifacts beneath an
  explicit root, reject escapes, symlinks, duplicates, missing/empty inputs,
  and type mismatches, and recompute deterministic hashes from their contents.
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

- conservative finite-volume `div(rhoPhi,U)` using projection-consistent face
  mass flux and the frozen `limitedLinear 1.0` interpolation;
- Euler update and the matched laminar viscous-stress convention;
- inlet-only integral flow constraint, outlet zero-gradient velocity, inlet
  zero-gradient pressure, and fixed outlet pressure gauge;
- zero normal electric current at both axial ends;
- invariant single liquid phase and constant temperature/properties in the
  FreeMHD materialization.

Tiny gates precede any solve campaign:

1. divergence-free manufactured convection with exact vector result,
   quadratic velocity scaling, conservation, JIT, and JVP;
2. mixed Neumann-inlet/Dirichlet-outlet pressure reconstruction;
3. inlet/outlet flux closure without stationwise flow forcing;
4. builder mutations against every formulation switch;
5. conservation, restart identity, and bounded differentiability checks.

Delete `_fixed_flow_face_flux_projection_duct` and its long legacy test when the
new path owns B2. Retain generic fixed-flow helpers only where still used.

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

Exit: the physics-valid path has equivalent observables and useful measured
speedup on its target host.

## Architecture and SOLVAX ownership

LMX owns MHD equations, geometry, materials, boundary/interface conditions,
finite-volume assembly, gauges, observables, sharding policy, checkpoints, and
physical acceptance. SOLVAX owns generic linear algebra after primal,
residual, gradient/transpose, JIT, placement, memory, and repeated interleaved
timing gates pass. Delete the LMX duplicate in the same tranche.

Keep `solvax>=0.8.3,<1`, test both the minimum and newest compatible release,
and record the resolved environment. Do not tie LMX to one patch release or
commit a resolver lock. The ignored local `uv.lock` still resolves 0.8.2 and
must be regenerated before any future `uv sync`; the active test environment
already uses 0.8.3.

The next ownership audit is evidence-driven:

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
- Finish additive ownership: the line closures delegate today, but the duct
  outer sum still needs the SOLVAX additive combiner with unit weights.
- Defer two isolated `splu_solve` substitutions and new nonlinear APIs until
  after canonical B2 parity; they do not currently buy enough deletion.
- Do not delegate MHD coupling, finite-volume coefficients, physical
  fixed-point loops, or acceptance logic.

Ratchet only through real ownership deletion:

| Surface | Current | Next target |
|---|---:|---:|
| package modules | 35 | 34 only after a complete owner disappears |
| package lines | 34,951 | below 34,900 |
| maintained-core lines | 7,995 | below 7,900 |
| test files / lines | 32 / 21,179 | at most 32 / below 21,100 |
| maintenance scripts | 18 | at most 18 |

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

Generate presentation assets from compact accepted records without rerunning
solvers. Aim for readable 6–8-second loops, compact posters, stills below
100 KiB, and tracked movies below 150 KiB where practical. Keep tracked media
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

Exit: the release is small, installable, reproducible, honestly scoped, and
suitable for external research use.
