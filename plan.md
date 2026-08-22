# LMX simplification plan

**Status:** active

**Last updated:** 2026-08-22

**Purpose:** product goal, executable roadmap, decision register, and work log

This file is the authoritative plan for simplifying LMX. It records decisions
and completed work so that user-facing code and documentation can describe only
the product that exists. Git history and GitHub releases preserve release
history; README, documentation, examples, names, and code comments describe
only the supported current state.

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
- selected, explicitly validated differentiable objectives;
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

Q2D turbulence, Dean flow, bent-pipe, magnetic-obstacle, blanket, WHAM, and
other research surfaces enter a Phase 0 capability audit. A lane stays only if
it has an owner, a clear physical contract, an executable numerical test, and a
documented path to validation. The plan will record that decision before any
end-to-end deletion. Compatibility shims, alternative solvers already supplied
by SOLVAX, dashboards, paper pipelines, media generators, and duplicated
campaign infrastructure are not protected.

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
| Bent-pipe flow | Mapped pipe plus low-Dean hydrodynamic limit | mapped finite-state and low-De baseline tests through the common 3-D solver | shared 3-D core | **retained as a 3-D geometry extension**; require metric/manufactured tests, low-De mesh convergence, and a named external target before a quantitative validation claim |
| Magnetic obstacle | Localized imposed field in a 3-D rectangular channel with velocity, pressure, current, and wake observables | internal field-response and baseline tests through the common 3-D solver | shared 3-D core | **retained as a 3-D field application**; add executable external data before a quantitative validation claim |
| Q2D MHD | Separate vorticity solver family | archived source and tests | none in live tree | **removed**; it had no named production user or quantitative parity and did not exercise the retained inductionless duct/fringing code |
| Branded mirror-pipe adapter | Product-specific proxy around generic tabulated fields and pipe fringing | archived source and tests | none in live tree | **removed**; generic tabulated/vector-field and straight-pipe capabilities remain available for a future matched case |
| Blanket reduced flow | Separate 1-D pressure-budget and filling model | archived source, tests, and media | none in live tree | **removed**; the standalone reduced model had no external validation owner and did not share the retained 3-D solver |
| Differentiable objectives | Gradients through selected physical solves and design observables | finite-difference/JVP/VJP gates for canonical Hartmann objectives | `autodiff.py` plus shared core | **retain canonical Hartmann mean/profile objectives**; add 3-D gradients only through the retained production solver |
| Reusable solver algebra | Krylov, fixed point, structured direct solves, preconditioners, projected/nullspace algebra | LMX unit/manufactured tests and overlapping SOLVAX APIs | portions of `solvers.py`/`fringing.py` | **move to SOLVAX when general**; retain LMX coefficient assembly and physics gates |

Current costs overlap where capabilities share modules; they are navigation
estimates, not additive budgets. Each audit decision must name the accountable
owner, retained public workflow, numerical gate, external or analytical
reference, and maintenance cost.

The accountable owner for retained 3-D geometry/application rows is the LMX
3-D/fringing maintainer. Their common workflow is one `FringingCase` and one
`solve` result; they do not retain separate solver families. Bent-pipe gates
are mapped-metric/manufactured operators, low-De mesh convergence, and a future
external curved-pipe target. Magnetic-obstacle gates are imposed-field
Maxwell checks, current/flow conservation, mesh convergence, and an executable
Votyakov/Andreev comparison before any quantitative claim. Until those final
external gates exist, the capabilities may remain documented as development
applications but not as validated benchmarks.

## Reusable-algebra ownership map

LMX owns physical coefficients, boundary/interface equations, dimensional
scaling, convergence gates, and MHD observables. SOLVAX owns reusable solver
algorithms and algebraic diagnostics. This table is the Phase 1 deletion map.

| Current LMX surface | Owner | Action | Replacement/gate |
|---|---|---|---|
| SciPy `spsolve` calls in rectangular and pipe 3-D Poisson paths | SOLVAX | replace now | released `solvax.splu_solve`; preserve assembled matrices and 3-D/FreeMHD gates |
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
| package source | 30,852 lines / 32 modules | **<= 15,000 lines / <= 16 implementation modules** |
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
- **SOLVAX owns algebra:** reusable operators, direct and iterative linear
  solves, Krylov methods, fixed-point algorithms, preconditioners, structured
  factorization, implicit differentiation, and solver termination metadata.

SOLVAX must not become a storage location for discarded LMX experiments. Code
moves upstream only when all of these are true:

1. the API is independent of LMX types, MHD terminology, geometry, and units;
2. it represents a reusable numerical method rather than coefficient assembly;
3. it has a second plausible consumer or a clear general structured-algebra
   role;
4. SOLVAX owns focused correctness, transpose/gradient, convergence, and
   performance tests;
5. the public SOLVAX documentation explains when to use it and its failure
   modes;
6. LMX can delete its implementation and depend on a released SOLVAX version.

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
| Q2D, Dean, bent-pipe, blanket, obstacle, and related helpers | audit before deciding | retain only a coherent, tested, owned capability; otherwise archive outside the live package |

### Upstream workflow

For each upstream candidate:

1. write a small SOLVAX issue/RFC with the mathematical contract, shapes,
   symmetry, differentiation semantics, and at least two use cases;
2. build it in SOLVAX with unit, dense-reference, transpose/gradient, and
   benchmark coverage;
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
- one selected gradient;
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
- [ ] Restore required GitHub CI and branch protection. Workflow definitions
  are present and hosted jobs now execute on the canonical `uwplasma/LMX`
  repository. The stale unpublished SOLVAX compatibility pin was corrected to
  0.13.0. The monolithic hosted lanes reached 69% without a test failure before
  runner termination; the exact bounded-shard replacement and combined-coverage
  gate are pending. Branch protection still requires a plan that supports it
  for this private repo.
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
- [ ] Replace wrappers with existing released SOLVAX APIs first.
- [ ] Propose only the projected/nullspace and separable structured APIs that
  pass the upstream criteria.
- [ ] Implement, benchmark, document, and release accepted SOLVAX additions.
- [ ] Bump LMX's SOLVAX lower bound and remove copied implementations/tests.
- [x] Delete `lmx/linear.py` when no unique owner remains. Boundary-aware
  stencil actions and physical residuals live with LMX operators; the remaining
  adapters compose released SOLVAX APIs inside the physical solver module.

Exit: no LMX module implements a general matrix solver, Krylov iteration,
fixed-point algebra, direct structured solve, or generic preconditioner.

### Phase 2 — refactor retained capabilities and trim audited lanes

- [ ] Decompose the 3-D/fringing monolith by mathematical ownership while
  preserving one end-to-end path and the frozen gates at every tranche.
- [ ] Consolidate FreeMHD code into a minimal case contract, Docker runner,
  native-output observer, and comparison layer outside the runtime wheel.
- [x] Remove duplicated reports, plots, fingerprints, frozen-output trees, and
  campaign adapters that do not contribute to a numerical gate.
- [x] For each research lane rejected in Phase 0, remove its implementation,
  exports, configuration, dependencies, tests, examples, scripts, docs, data,
  and claims together; preserve selected material in the verified archive.
- [ ] For each retained research lane, give it the same compact API/result
  semantics and an explicit validation roadmap.
- [x] Verify installed-package discovery and wheel contents against the
  capability matrix.

Exit: 3-D fringing and FreeMHD parity still pass, every remaining lane has a
user and validation contract, and package source is below 18,000 lines before
consolidation.

### Phase 3 — redesign the API and package

- [ ] Move to `src/lmx` layout.
- [ ] Introduce the single case/options/result/convergence model.
- [ ] Collapse duplicate config, cases, core, logging, units, output, and
  validation representations.
- [ ] Reduce `__all__` to the documented API.
- [ ] Remove aliases, pass-through wrappers, redundant parameters, and boolean
  feature matrices.
- [ ] Merge files according to the target ownership map without creating a
  mega-module.
- [ ] Add `py.typed` and public type-completeness verification.

Exit: the 2-D first-run, 3-D fringing, and advanced API examples work; package
is <= 16 implementation modules and every public symbol has one documented
purpose.

### Phase 4 — simplify and optimize the supported solver

- [ ] Review every supported source function for ownership and necessity.
- [ ] Reuse coefficients, factors, preconditioners, and initial guesses.
- [ ] Make full histories opt-in and remove plot-only work from solves.
- [ ] Consolidate JIT boundaries and eliminate hot-path host transfers.
- [ ] Remove dense/intermediate allocations where matrix-free structure exists.
- [ ] Benchmark 2-D and 3-D cold, warm, memory, iterations, and physical errors
  after each change.
- [ ] Delete every rejected alternative immediately.

Exit: performance gates pass, no canonical case regresses >5%, and the planned
runtime/memory reduction is either achieved or its remaining bottleneck is
demonstrated with a profile.

### Phase 5 — consolidate tests, examples, and tools

- [x] Build the target <= 14-file behavior-oriented test tree.
- [ ] Parameterize duplicated case setup and retain physical assertions.
- [x] Replace maintenance scripts with `lmx validate`, tests, or deletion.
- [ ] Implement the three-layer FreeMHD contract, local Docker smoke, and
  scheduled/release production validation workflow.
- [x] Keep no more than seven Python examples plus one TOML case, including one
  3-D fringing example and one external-validation example.
- [x] Make all examples fast, self-contained, editable, and CI-executed.
- [ ] Enforce source/file/size/media/API budgets in package tests.

Exit: default suite is <= 90 seconds, coverage >=95%, Docker smoke passes when
requested, scheduled/release parity is reproducible, all examples run, and
source/tests/scripts meet their budgets.

### Phase 6 — rebuild docs and README

- [x] Delete the existing user-facing documentation tree and recreate the
  target information architecture from the stable API.
- [x] Write equations and equation-to-code mapping from primary sources.
- [x] Generate complete API reference from docstrings.
- [ ] Write and execute four tutorials, including 3-D fringing, and focused
  how-to guides, including local FreeMHD Docker validation.
- [ ] Publish validation with regenerated current-source evidence.
- [x] Create the product-first README and one compact validated hero visual.
- [ ] Run the current-state prose scan, Sphinx `-W`, API coverage, and links.

Exit: a new user can install, solve, interpret, validate, and extend LMX from
the docs without reading source or this plan.

### Phase 7 — make the repository itself small

Deleting live files will not shrink the current 38-MiB history pack, which is
dominated by generated images, movies, repeated lockfiles, and deleted research
artifacts. Meeting the normal-clone target requires a deliberate history
operation.

- [ ] Create a private read-only archival mirror containing every branch, tag,
  release reference, and a checksummed `git bundle`.
- [ ] Clone the archive independently and run `git fsck` before rewriting.
- [ ] Export release metadata and move large reusable artifacts to releases.
- [ ] Create a reviewed squashed root for the slim product.
- [ ] Delete old refs from the live repository and force-push only after
  separate explicit approval for this destructive operation.
- [ ] Wait for/coordinate GitHub garbage collection or replace the live
  repository after archiving if retained unreachable objects prevent the
  target.
- [ ] Measure a new ordinary authenticated clone, not a local/shared clone.
- [ ] Record clone disk size, pack transfer size, checkout bytes, and file
  count in this plan.

Exit: `du -sk` of a fresh normal clone is below 9,766 KiB and the archive has
been independently verified. No history rewrite occurs without explicit
approval at execution time.

### Phase 8 — release the standalone product

- [ ] Run minimum/current Python and dependency matrices.
- [ ] Run complete 2-D duct, 3-D fringing, FreeMHD parity, conservation,
  convergence, gradient, packaging, documentation, and full validation gates.
- [ ] Build and inspect wheel/sdist contents and sizes.
- [ ] Publish one coherent major release from the same reviewed commit.
- [ ] Verify clean install, first-run Python example, CLI, docs, citation, and
  fresh clone.

Exit: release artifacts, documentation, examples, numerical evidence, and
source all describe the same standalone API and commit.

## CI gates

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
| D-011 | Remove the separate Q2D solver and blanket reduced-model families after verified archive | Neither has a named production user or quantitative external validation, and both expand dependencies, APIs, tests, and media without exercising the retained duct/fringing solver. |
| D-012 | Retain bent-pipe geometry and magnetic-obstacle cases only as applications of the common 3-D fringing solver | They broaden the retained geometry/field model without justifying separate algorithms; independent validation remains a promotion gate. |
| D-013 | Remove branded WHAM proxy builders while retaining generic tabulated fields and straight-pipe fringing | The general capability can express a future WHAM case once matched evidence exists, without carrying product-specific unvalidated code now. |
| D-014 | Retain only objectives differentiated through the canonical Hartmann path in this refactor | Shadow 3-D, nonrectangular surrogate, WHAM, and blanket objectives can report gradients of a different model than the production solver. |
| D-015 | Use released SOLVAX native sparse solves instead of calling SciPy solvers from LMX | LMX owns MHD matrix assembly; reusable host factorization and solve behavior belongs to SOLVAX. |
| D-016 | Use one explicit current restart-schema family for state, compact flux, Aitken, and Anderson checkpoints | A single fail-closed contract is easier to reason about, test, and document than internal format-version branches. |
| D-017 | Keep the executed matched B2 Docker path and remove the private straight-pipe archive smoke | The private lane explicitly could not establish B1 equation/observable parity and duplicated the accepted B2 execution boundary; B1 remains protected by internal/manufactured gates until a genuinely matched external case exists. |
| D-018 | Use Ruff with a 110-column limit for all maintained Python | The numerical expressions remain readable, every file has one formatter, and the format reduces line count without hand-compressed layouts. |
| D-019 | Expose one SOLVAX PCG velocity path instead of naming identical `auto`, `cg`, and `solvax_pcg` choices | LMX assembles and certifies the physical system; a user-facing switch between aliases of the same released algorithm adds no flexibility. |
| D-020 | Consolidate by stable concept: units and wall models into physics, and state/result schemas into specs | These types are small parts of the physical and public data contracts; separate modules added navigation and import boundaries without independent ownership. |
| D-021 | Consolidate spatial construction, run configuration, output, and validation by user-facing ownership | Meshes own spatial operators and imposed fields; configuration owns run logging; IO owns lazy plotting; validation owns references and benchmark contracts. These boundaries minimize navigation while remaining acyclic and independently testable. |

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
- Next action: decompose and consolidate the retained 3-D and FreeMHD paths
  toward the final source/module ceilings.
