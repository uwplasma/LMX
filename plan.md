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
| B2 ALEX square duct | fine numerical baseline converged; experimental acceptance open | resolve transverse-only versus Maxwell-consistent field with matched FreeMHD |
| SOLVAX | 0.8.2 primal, gradient, transpose, CPU/GPU, and existing integration gates pass | remove remaining safe LMX duplicates; repeat interleaved timing |
| Parallel execution | two-GPU B2 numerical checkpoint shows 1.66x speedup and parity | accepted-case CPU/GPU scaling and four-device evidence |
| Portable quality | 760 pass, 8 expected external-data skips, 95.31% branch coverage, 193.4 s | stay below 300 s target and 600 s hard limit |

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
- Add a plotting-only B2/ALEX field and pressure panel from existing compact
  records, explicitly labelled “acceptance open”; do not rerun a solver for it.

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
3. Replace the custom periodic-theta line solve with SOLVAX
   `cyclic_tridiagonal_solve` after duct/pipe primal and gradient parity.
4. Reformulate anchored Poisson operators symmetrically before moving them to
   SOLVAX PCG. Do not apply PCG to the current nonsymmetric row-replacement form.
5. Add and release a symmetry-preserving additive-line preconditioner in SOLVAX,
   then delete the LMX duplicate. Keep geometry adaptation in LMX.

Do not yet delegate finite-iteration autodiff Jacobi, modal B1 geometry solves,
variable-coefficient FV assembly, physical fixed-point loops, or nonlinear
coupling to affine/Newton–Krylov APIs. SOLVAX single-reduction PCG and sharding-
preserving GMRES need real 1/2/4-device evidence before distributed claims.

Every delegation must pass primal, residual, gradient/transpose, JIT, placement,
memory, and repeated interleaved warm-timing gates. Delete the LMX duplicate in
the same tranche when SOLVAX passes.

Keep a compatible runtime range rather than an exact SOLVAX pin or repository
lockfile. CI tests both the minimum supported release and the newest release
satisfying that range; release evidence records the resolved environment.

Near-term ratchets after this workstream:

| Surface | Current | Next target |
|---|---:|---:|
| package modules | 35 | at most 34 |
| package lines | 34,939 | below 34,750 |
| maintained-core lines | 8,067 | below 7,900 |
| test files / lines | 33 / 21,273 | at most 32 / below 21,100 |
| maintenance scripts | 19 | at most 18; reusable logic moves into `lmx/` or an existing command |

Count semantic maintenance cost across the whole package; do not satisfy a
budget through formatting or by relabelling a large module “research-stage.”
Ratchet each enforced cap downward after an accepted migration. The first
file-count gate is complete: one package module, one test file, and one
maintenance script have been removed without dropping a supported workflow.

## Priority 3: one exact matched FreeMHD harness

- Freeze equations, nondimensionalization, geometry, magnetic field, wall
  model, boundaries, mesh mapping, drive, stopping rules, and observables.
- Record source, input, evaluator, dependency, and output hashes for both codes.
- Establish tiny exact cases first, then use the same harness for B2 and B1.
- Keep FreeMHD cases and large outputs outside the LMX repository; retain only
  compact specifications, evaluators, and accepted summaries.

Exit: an exact-case record cannot pass unless both codes demonstrably solved the
same problem and evaluated the same observable.

## Priority 4: B2 field correctness, then acceptance

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

## Priority 5: B1 external acceptance

Apply the shared exact-case protocol to the promoted retained-modal solver:
small parity, medium mesh independence, then one final large confirmation only
if required. Publish compact pressure/conservation tables and compressed plots;
keep full fields in release assets.

Exit: B1 has frozen literature, mesh, conservation, and exact-FreeMHD acceptance.

## Priority 6: accepted-case CPU/GPU scaling

- Confirm CPU/GPU and one-/multi-device numerical equivalence before timing.
- Separate compilation from repeated warm solves and report uncertainty, memory,
  placement, speedup, and efficiency.
- Measure fixed-size Mac 1/2/4-device CPU scaling and office one-/two-GPU
  sharding; add a four-GPU point only on suitable hardware.
- Independent-case multiprocessing is throughput evidence, not one-solve scaling.
- Add CI timing thresholds only for stable, low-variance kernels.

Exit: at least one physics-accepted production case demonstrates useful strong
scaling with identical observables.

## Priority 7: release

Consolidate tests and scripts while preserving one bounded check per public
capability. Build docs with warnings as errors, verify provenance and media,
build and inspect wheel/sdist, install the wheel in a clean environment, run the
portable gate and selected accepted external/gradient/scaling lanes, then publish
exact commits, environments, references, tolerances, limitations, and assets.
Public APIs retain concise docstrings and type hints; non-obvious discretizations
and numerical choices retain comments explaining the reason, not the syntax.

Exit: the release is small, installable, reproducible, honestly scoped, and
suitable for external research use.
