# LMX authoritative development plan

This is the single project plan. It records priorities and gates, not campaign
history. Detailed metrics, rejected probes, checksums, and raw outputs belong in
`benchmarks/results/`, the validation docs, or versioned release assets.

## Product target

LMX will be a lightweight JAX code for accurate, end-to-end differentiable
inductionless liquid-metal MHD on CPUs and GPUs. Stable claims require
analytical or manufactured verification, conservation and convergence,
independent-code or experimental evidence where available, and reproducible
performance measurements.

The public repository has four working surfaces:

- `lmx/`: source and the `lmx` command;
- `tests/`: bounded unit, numerical, physics, regression, and workflow checks;
- `examples/`: small runnable Python and TOML workflows;
- `docs/`: theory, inputs, validation, performance, and development guidance.

Large fields, raw external-code runs, meshes, full-resolution figures, and
movies belong in GitHub or Zenodo releases, not Git.

## Operating contract

### Small first

Every experiment declares one hypothesis, frozen metrics, a wall-time ceiling,
a stop rule, and a go/no-go threshold before launch. Escalation is:

1. static, analytical, plotting-only, or tiny-grid check;
2. bounded one-device smoke run;
3. medium confirmation only after the smoke gate passes;
4. fine, external-code, or multi-hour campaign only after all earlier gates,
   with a durable restart, bounded stages, and an interim stop/go check.

Existing checksummed data are reused when they answer the question. A failed or
unpromising bounded probe stops; parameter searches do not expand by inertia.

### Test only what changed

- During development: lint/static checks plus the directly affected tests,
  normally under two minutes.
- At a coherent subsystem boundary: the affected test modules and repository
  gates, normally under five minutes.
- Once per green tranche and in CI/release: the complete covered portable gate,
  with a 300-second engineering target and 600-second hard limit.
- External FreeMHD, accelerator, and mesh-refinement campaigns remain explicit
  manual or scheduled lanes with portable representative tests.

Branch coverage of `lmx/` remains at least 95%. Coverage is not physics
validation; promoted claims also need the relevant conservation, convergence,
gradient, literature, or independent-code gate.

Maintain a capability-to-evidence map linking every public workflow to one
bounded portable test and, where required, a numerical/physics/external gate.
Enforce the `unit`, `numerical`, `regression`, `physics`, `validation`, and
`external` markers so future test consolidation cannot silently drop a layer.

### Parallel work without contaminated evidence

Use subagents for disjoint literature, code-ownership, media, and validation
audits. One integrator owns shared files. Run independent non-timing checks in
parallel; run performance measurements alone on the measured host. Commit and
push every coherent green tranche, keep `main` authoritative, and remove
superseded worktrees and branches.

## Current status

| Area | Status | Next acceptance gate |
|---|---|---|
| README and docs | concise feature-led README, sourced comparison, compressed release media | keep claims and media checks current |
| Fully developed ducts | Hartmann, Shercliff, Hunt, and all eight high-Ha rows accepted | preserve regression gates |
| FreeMHD closed channels | bounded Shercliff/Hunt parity accepted | do not generalize to full FreeMHD parity |
| B1 ALEX pipe | retained-modal numerics, conservation, and restart accepted | exact matched FreeMHD plus literature/mesh acceptance |
| B2 ALEX square duct | fine numerical baseline converged; current LMX and FreeMHD formulations are not yet equation-identical | reconcile inertia and axial drive before any matched run |
| SOLVAX | velocity, cyclic-line, and symmetric anchored-Poisson PCG paths use 0.8.2 | contribute additive-line ownership; repeat interleaved timing |
| Parallel execution | two-GPU B2 numerical checkpoint shows 1.66x speedup and parity | accepted-case CPU/GPU scaling and four-device evidence |
| Portable quality | 761 pass, 8 expected external-data skips, 95.30% branch coverage, 182.3 s | stay below 300 s target and 600 s hard limit |

Machine-readable evidence is in `benchmarks/results/`; current interpretation is
in `docs/validation_report.md` and `docs/performance.md`.

## Priority 1: user surface and claim audit

- Keep the README below roughly 800 words: pitch, quickstart, sourced capability
  table, and short visual sections for verified ducts, geometries/fringe fields,
  differentiation, research workflows, and scaling.
- Verify LMX/FreeMHD/NekRS comparisons against primary papers or official docs.
- Give every major feature a relevant plot or movie poster; never use a generic
  “selected media” gallery.
- Put the richer feature-specific plots and movies in the relevant docs pages,
  with provenance and acceptance status beside each compressed asset.
- Keep tracked documentation media below 1 MiB, stills below 100 KiB, and
  movies below 150 KiB where practical. Store full-resolution sources and large
  outputs in checksummed releases.
- **Complete:** the plotting-only B2/ALEX field and pressure panel uses existing
  compact records, is labelled “acceptance open,” and runs no solver.

Exit: a new user can understand scope, run a first case, see representative
results, and distinguish verified from research-stage functionality without
reading internal campaign history.

## Priority 2: SOLVAX ownership and measurable slimming

LMX owns MHD equations, geometry, materials, boundary/interface conditions,
finite-volume assembly, gauges, observables, sharding policy, checkpoints, and
physical acceptance. SOLVAX owns generic linear algebra once parity is proved.

Apply this migration sequence:

1. **Complete:** obsolete optional-SOLVAX guards, empty accelerator extras, and
   stale version errors are gone; SOLVAX is a required compatible dependency.
2. **Complete:** five-point SPD velocity solves use `pcg_linear_solve` with LMX
   maximum-residual recertification. Native CG, Lineax, mock-only tests, and the
   obsolete live promotion script are gone; `cg` is only a compatibility alias.
3. **Complete:** the periodic-theta line uses SOLVAX
   `cyclic_tridiagonal_solve`; pipe primal, variable-coefficient dense-reference,
   and implicit-gradient gates replace the circulant-only FFT shortcut.
4. **Complete:** the anchored Poisson action projects both row and column
   symmetrically; a dense nonuniform audit found symmetry to `8.88e-16` and a
   positive minimum eigenvalue. Preconditioners are lifted as
   `P M(P r) + e_a e_a^T r`, explicit nonuniform `cg` routes through the
   volume-scaled formulation, and SOLVAX PCG replaces the retained recurrence.
   Unscaled stopping uses `rtol = physical_tol / sqrt(N)`; volume-scaled
   stopping uses `atol = physical_tol * min(residual_scale)`. Uniform,
   anisotropic, physical-residual, RHS/coefficient-gradient, JIT, transpose, and
   tiny high-Ha gates pass. LMX retains physical maximum-norm recertification.
5. Add and release a symmetry-preserving additive-line preconditioner in SOLVAX,
   then delete the two LMX averaging closures. SOLVAX's current multiplicative
   `line_smoother` is not a PCG-safe substitute; geometry and axis adaptation
   remain in LMX. Use a clean SOLVAX worktree from `origin/main`, because the
   existing local checkout contains unrelated work.
6. Treat migration of two direct SciPy sparse solves to `solvax.splu_solve` as
   low-value cleanup after the PCG and additive-line deletions; sparse FV
   assembly remains LMX-owned.

Do not yet delegate finite-iteration autodiff Jacobi, modal B1 geometry solves,
variable-coefficient FV assembly, physical fixed-point loops, or nonlinear
coupling to affine/Newton–Krylov APIs. SOLVAX single-reduction PCG and
sharding-preserving GMRES need real 1/2/4-device evidence before distributed
claims. Keep the accepted high-Ha fast-diagonalization/FGMRES route unchanged
during the Poisson migration; operator symmetry alone does not establish parity
for that specialized path.

Every delegation must pass primal, residual, gradient/transpose, JIT, placement,
memory, and repeated interleaved warm-timing gates. Delete the LMX duplicate in
the same tranche when SOLVAX passes.

Keep a compatible runtime range rather than an exact SOLVAX pin or repository
lockfile. CI tests both the minimum supported release and the newest release
satisfying that range; release evidence records the resolved environment. The
installed release and current SOLVAX `origin/main` are both 0.8.2, so
`solvax>=0.8.2,<1` currently expresses that policy without a stale exact pin.

Near-term ratchets after this workstream:

| Surface | Current | Next target |
|---|---:|---:|
| package modules | 35 | hold; reach 34 only through real ownership deletion |
| package lines | 34,895 | below 34,875 |
| maintained-core lines | 8,024 | below 8,000 |
| test files / lines | 32 / 21,295 | hold 32 / below 21,250 |
| maintenance scripts | 19 | at most 18; reusable logic moves into `lmx/` or an existing command |

Count semantic maintenance cost across the whole package; do not satisfy a
budget through formatting or by relabelling a large module “research-stage.”
Ratchet each enforced cap downward after an accepted migration. Mesh and
discrete-operator tests now share one independently runnable file; further file
removal must follow real ownership consolidation, not arbitrary merging.

### Immediate execution order

1. Harden the FreeMHD record validator and consolidate the standalone case
   materializer into the existing parity command; run no solver in this step.
2. Reconcile B2 momentum advection and the axial boundary/drive contract.
3. Prove one-/multi-device equivalence and bounded strong scaling for that path.
4. Run the tiny matched B2-family smoke case; only a passing contract and smoke
   can authorize medium or production B2 work.

Steps 1--3 use focused tests and tiny manufactured cases. The full portable
gate runs once after the combined tranche, not after each edit.

## Priority 3: one exact matched FreeMHD harness

- Add a machine-enforced record validator before another external run. It must
  compare equations, nondimensional groups, geometry, field data and mapping,
  wall properties, boundaries and drive, mesh coordinates, stopping rules,
  observables, normalization, and source/input/evaluator/output hashes.
- Replace the permissive boolean `exact_case_match` gate with computed contract
  checks and an `acceptance_role`. A tiny `harness-smoke` record can test the
  machinery but can never unlock `b2-production` acceptance.
- Reconcile the known B2 mismatch first: LMX currently omits convective inertia
  used by FreeMHD's finite-inertia momentum equation, and LMX stationwise
  fixed-flow/Neumann axial treatment is not yet shown equivalent to FreeMHD's
  inlet-flow/outlet-pressure treatment. Until both checks pass, an exact B2
  record is impossible and no production FreeMHD run is authorized.
- Establish one frozen, tiny B2-family case only after formulation parity, then
  use the same harness for production B2 and B1.
- Keep FreeMHD cases and large outputs outside the LMX repository; retain only
  compact specifications, evaluators, and accepted summaries.
- Move reusable case materialization and hashing into `lmx.freemhd`, expose it
  through the existing parity command, and delete the standalone materializer.

Exit: an exact-case record cannot pass unless both codes demonstrably solved the
same problem and evaluated the same observable.

## Priority 4: parallel correctness before expensive campaigns

- Make one-solve sharding, device placement, and collectives explicit for each
  promoted solver. A process pool of independent cases is throughput, not
  strong scaling.
- Use tiny numerical-equivalence checks first, then repeated warm timings run
  alone. Separate compilation, report uncertainty and memory, and require
  identical physical observables.
- Measure fixed-size Mac 1/2/4-device CPU scaling and office one-/two-GPU
  scaling before launching the next medium B1/B2 campaign. Keep the same global
  problem size; weak scaling is reported separately.
- Treat host launch scripts and large timing records as release assets or
  external infrastructure, not package surface.

Exit: the solver path selected for the next external campaign has verified
one-/multi-device equivalence and useful measured speedup on its target host.

## Priority 5: B2 field correctness, then acceptance

- Compare transverse-only and Maxwell-consistent fringe fields in tiny/coarse
  LMX and FreeMHD cases. Check field curl/divergence, conservation, steady state,
  restart identity, and the ALEX pressure observable.
- Decide the field model from source-backed matched evidence, not peak agreement
  alone. The current Maxwell-consistent pilot is diagnostic, not validation.
- Advance to medium and fine only when coarse FreeMHD/literature error improves
  for the correct physical reason.
- Resume tight-tolerance or outer-acceleration work only after formulation parity.

Exit: B2 has source-identical three-mesh, wall/tolerance, literature, and exact
FreeMHD evidence with frozen tolerances and a reproducible accepted record.

## Priority 6: B1 external acceptance

Apply the shared exact-case protocol to the promoted retained-modal solver:
small parity, medium mesh independence, then one final large confirmation only
if required. Publish compact pressure/conservation tables and compressed plots;
keep full fields in release assets.

Exit: B1 has frozen literature, mesh, conservation, and exact-FreeMHD acceptance.

## Priority 7: accepted-case CPU/GPU scaling

- Confirm CPU/GPU and one-/multi-device numerical equivalence before timing.
- Separate compilation from repeated warm solves and report uncertainty, memory,
  placement, speedup, and efficiency.
- Measure fixed-size Mac 1/2/4-device CPU scaling and office one-/two-GPU
  sharding; add a four-GPU point only on suitable hardware.
- Independent-case multiprocessing is throughput evidence, not one-solve scaling.
- Add CI timing thresholds only for stable, low-variance kernels.

Exit: at least one physics-accepted production case demonstrates useful strong
scaling with identical observables.

## Priority 8: release

Consolidate tests and scripts while preserving one bounded check per public
capability. Build docs with warnings as errors, verify provenance and media,
build and inspect wheel/sdist, install the wheel in a clean environment, run the
portable gate and selected accepted external/gradient/scaling lanes, then publish
exact commits, environments, references, tolerances, limitations, and assets.
Public APIs retain concise docstrings and type hints; non-obvious discretizations
and numerical choices retain comments explaining the reason, not the syntax.

Exit: the release is small, installable, reproducible, honestly scoped, and
suitable for external research use.
